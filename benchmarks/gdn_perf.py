# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Per-kernel timing harness for GatedDeltaNet.

`measure` profiles one checked-out tree and writes device time per kernel group
as JSON. `gate` builds a base worktree, profiles both arms with separate Triton
caches, and fails on a regression.

Both arms must be different trees. Comparing a tree with itself is what a broken
A/B looks like, so `gate` refuses to do it unless `--self-check` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# First substring hit wins, so the specific names come before the generic ones.
KERNEL_GROUPS = (
    ('local_cumsum', ('local_cumsum',)),
    ('fwd_kkt', ('scaled_dot_kkt',)),
    ('fwd_solve_tril', ('solve_tril', 'merge_16x16')),
    ('recompute_w_u', ('recompute_w_u',)),
    ('fwd_h', ('fwd_kernel_h',)),
    ('fwd_o', ('fwd_kernel_o',)),
    ('bwd_dv_local', ('bwd_kernel_dv_local',)),
    ('bwd_dv', ('bwd_kernel_dv',)),
    ('bwd_dhu', ('bwd_kernel_dhu',)),
    ('bwd_dqkwg', ('bwd_kernel_dqkwg',)),
    ('bwd_dw', ('bwd_kernel_dw',)),
    ('prepare_wy_bwd', ('prepare_wy_repr_bwd',)),
    ('l2norm', ('l2norm',)),
)
OTHER = 'other'
MODES = ('fwd', 'fwd_bwd')


def group_of(kernel_name: str) -> str:
    for group, keys in KERNEL_GROUPS:
        if any(k in kernel_name for k in keys):
            return group
    return OTHER


def git(repo, *args) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_runner(heads: int, seq_len: int, dim: int, dtype: str, device: str, backward: bool):
    """Build one timed closure. Inputs are generated once and reused."""
    import torch
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch_dtype = getattr(torch, dtype)
    gen = torch.Generator().manual_seed(1234)
    batch, K, V = 1, dim, dim

    def rand(*shape, dt=torch.float32):
        return torch.rand(*shape, generator=gen, dtype=torch.float32).to(device).to(dt)

    def randn(*shape, dt=torch.float32):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device).to(dt)

    q = F.normalize(rand(batch, seq_len, heads, K), p=2, dim=-1).to(torch_dtype)
    k = F.normalize(rand(batch, seq_len, heads, K), p=2, dim=-1).to(torch_dtype)
    v = rand(batch, seq_len, heads, V, dt=torch_dtype)
    beta = rand(batch, seq_len, heads).sigmoid()
    g = F.logsigmoid(rand(batch, seq_len, heads))
    h0 = torch.zeros(batch, heads, K, V, dtype=torch.float32, device=device)
    do = randn(batch, seq_len, heads, V, dt=torch_dtype)
    dht = randn(batch, heads, K, V)
    leaves = (q, k, v, beta, g, h0)
    if backward:
        for t in leaves:
            t.requires_grad_(True)

    def run():
        o, ht = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=K ** -0.5,
            initial_state=h0, output_final_state=True,
        )
        if not backward:
            return
        ((o * do).sum() + (ht * dht).sum()).backward()
        for t in leaves:
            t.grad = None

    return run


def profile_us(run, warmup: int, iters: int) -> dict[str, float]:
    """Device microseconds per call, grouped by kernel."""
    import torch

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()

    totals: dict[str, float] = defaultdict(float)
    for evt in prof.key_averages():
        us = getattr(evt, 'self_device_time_total', None)
        if not us:
            us = getattr(evt, 'self_cuda_time_total', 0.0)
        if us:
            totals[group_of(evt.key)] += us / iters
    return {k: round(v, 3) for k, v in totals.items()}


def measure(args) -> dict:
    repo = Path(args.repo).resolve()
    # The arm's tree must win over any installed copy of fla.
    sys.path.insert(0, str(repo))
    import fla  # noqa: I001
    import torch
    import triton

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

    heads = [int(x) for x in args.heads.split(',') if x]
    seq_lens = [int(x) for x in args.seq_lens.split(',') if x]
    points: dict[str, dict[str, float]] = {}

    for H in heads:
        for T in seq_lens:
            for mode in MODES:
                key = f'H{H}_T{T}/{mode}'
                try:
                    run = make_runner(H, T, args.dim, args.dtype, args.device, mode == 'fwd_bwd')
                    reps = [profile_us(run, args.warmup, args.iters) for _ in range(args.reps)]
                except RuntimeError as exc:
                    # A shape that does not fit is reported, never silently dropped.
                    print(f'[measure] {key}: skipped ({str(exc).splitlines()[0]})', flush=True)
                    points[key] = {'skipped': 1.0}
                    continue
                groups = sorted({g for rep in reps for g in rep})
                if not groups:
                    # The profiler recorded no device time, which must not look like a 1.00x.
                    print(f'[measure] {key}: skipped (no device time recorded)', flush=True)
                    points[key] = {'skipped': 1.0}
                    continue
                points[key] = {
                    g: round(statistics.median([rep.get(g, 0.0) for rep in reps]), 3)
                    for g in groups
                }
                print(f'[measure] {key}: {points[key]}', flush=True)

    meta = {
        'fla_file': str(Path(fla.__file__).resolve()),
        'git_sha': git(repo, 'rev-parse', 'HEAD'),
        'git_subject': git(repo, 'log', '-1', '--format=%s'),
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'triton': triton.__version__,
        'dtype': args.dtype,
        'dim': args.dim,
        'iters': args.iters,
        'reps': args.reps,
        'warmup': args.warmup,
        'triton_cache_dir': os.environ.get('TRITON_CACHE_DIR', ''),
    }
    return {'meta': meta, 'points': points}


def totals(point: dict[str, float]) -> float:
    return round(sum(point.values()), 3)


def run_measure(arm_dir: Path, out: Path, args) -> dict:
    """Profile one arm in a fresh process with a fresh Triton cache."""
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop('FLA_CI_ENV', None)
    arm_path = str(arm_dir)
    if env.get('PYTHONPATH'):
        arm_path = f"{arm_path}{os.pathsep}{env['PYTHONPATH']}"
    env['PYTHONPATH'] = arm_path
    env['TRITON_CACHE_DIR'] = tempfile.mkdtemp(prefix=f'triton-{arm_dir.name}-', dir=str(out.parent))
    cmd = [
        sys.executable, str(Path(__file__).resolve()), 'measure',
        '--repo', str(arm_dir), '--out', str(out),
        '--heads', args.heads, '--seq-lens', args.seq_lens,
        '--dim', str(args.dim), '--dtype', args.dtype,
        '--iters', str(args.iters), '--reps', '1', '--warmup', str(args.warmup),
    ]
    print(f"[gate] {arm_dir.name}: TRITON_CACHE_DIR={env['TRITON_CACHE_DIR']}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    return json.loads(out.read_text())


def median_points(runs: list[dict]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for point in runs[0]['points']:
        groups = {g for run in runs for g in run['points'].get(point, {})}
        merged[point] = {
            # Only rounds that measured the group count: a zero would read as a
            # cheaper arm rather than as a missing sample.
            g: round(statistics.median(
                [run['points'][point][g] for run in runs if g in run['points'].get(point, {})]), 3)
            for g in groups
        }
    return merged


def group_gaps(runs: list[dict]) -> list[str]:
    """Kernels that only some rounds measured.

    A partial sample is neither a ratio nor a new kernel, so the caller must abort.
    """
    gaps = []
    for point in runs[0]['points']:
        groups = {g for run in runs for g in run['points'].get(point, {})}
        for g in sorted(groups):
            if not all(g in run['points'].get(point, {}) for run in runs):
                gaps.append(f'{point}|{g}')
    return gaps


def compare(base: dict, head: dict, args) -> tuple[list[dict], list[dict]]:
    detail, summary = [], []
    for point in sorted(base):
        b_point, h_point = base[point], head.get(point, {})
        for group in sorted(set(b_point) | set(h_point)):
            b, h = b_point.get(group, 0.0), h_point.get(group, 0.0)
            if b <= 0 and h <= 0:
                continue
            if b <= 0:
                # A kernel that does not exist in the base arm is new work, not slower work.
                detail.append({'point': point, 'group': group, 'base_us': 0.0, 'head_us': h,
                               'delta_us': round(h, 3), 'ratio': None, 'regress': False})
                continue
            ratio = h / b if b > 0 else float('inf')
            delta = h - b
            regress = ratio > 1 + args.per_kernel_tol and delta > args.min_delta_us
            detail.append({'point': point, 'group': group, 'base_us': b, 'head_us': h,
                           'delta_us': round(delta, 3), 'ratio': round(ratio, 4), 'regress': regress})
        b_total, h_total = totals(b_point), totals(h_point)
        ratio = h_total / b_total if b_total > 0 else float('inf')
        summary.append({'point': point, 'base_us': b_total, 'head_us': h_total,
                        'ratio': round(ratio, 4), 'regress': ratio > 1 + args.tol})
    return detail, summary


def write_markdown(path: Path, base_meta: dict, head_meta: dict, detail: list[dict],
                   summary: list[dict], regressions: list[dict], self_check: bool,
                   rounds: int) -> None:
    lines = [
        '# PPU GDN performance gate',
        '',
        f"- base: `{base_meta['git_sha'][:9]}` {base_meta['git_subject']}",
        f"- head: `{head_meta['git_sha'][:9]}` {head_meta['git_subject']}",
        f"- device: {head_meta['device']}, torch {head_meta['torch']}, triton {head_meta['triton']}, "
        f"{head_meta['dtype']}, K=V={head_meta['dim']}",
        f"- timing: median over {rounds} rounds of {head_meta['reps']} rep x {head_meta['iters']} "
        f"iterations per point, first {head_meta['warmup']} iterations discarded, "
        f"fresh Triton cache per arm and round",
        '',
        f"## Regressions ({len(regressions)})",
        '',
    ]
    if not regressions:
        lines.append('None.')
    else:
        lines += ['| point | kernel | base us | head us | delta us | ratio |', '|---|---|---:|---:|---:|---:|']
        for row in regressions:
            lines.append(
                f"| {row['point']} | {row['group']} | {row['base_us']:.1f} | {row['head_us']:.1f} | "
                f"{row['delta_us']:+.1f} | {row['ratio']:.3f}x |"
            )
    lines += ['', '## Totals per shape', '',
              '| point | base us | head us | ratio |', '|---|---:|---:|---:|']
    for row in summary:
        mark = ' **REGRESSED**' if row['regress'] and not self_check else ''
        lines.append(f"| {row['point']} | {row['base_us']:.1f} | {row['head_us']:.1f} | {row['ratio']:.3f}x{mark} |")
    lines += ['', '<details><summary>All kernels</summary>', '',
              '| point | kernel | base us | head us | ratio |', '|---|---|---:|---:|---:|']
    for row in detail:
        ratio = 'new' if row['ratio'] is None else f"{row['ratio']:.3f}x"
        lines.append(f"| {row['point']} | {row['group']} | {row['base_us']:.1f} | "
                     f"{row['head_us']:.1f} | {ratio} |")
    lines += ['', '</details>', '']
    path.write_text('\n'.join(lines))


def gate(args) -> int:
    head_dir = Path(args.head_dir).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    head_sha = git(head_dir, 'rev-parse', 'HEAD')
    base_sha = git(head_dir, 'merge-base', 'HEAD', args.base_ref)
    if head_sha == base_sha and not args.self_check:
        print(f'[gate] base and head are the same commit ({head_sha[:9]}); '
              'pass a real base ref or use --self-check', file=sys.stderr)
        return 3

    base_dir = workdir / f'base-{base_sha[:8]}'
    if base_dir.exists():
        if git(base_dir, 'rev-parse', 'HEAD') != base_sha:
            print(f'[gate] {base_dir} is not at {base_sha[:9]}', file=sys.stderr)
            return 3
    else:
        git(head_dir, 'worktree', 'add', '--detach', str(base_dir), base_sha)
    print(f'[gate] base={base_sha[:9]} head={head_sha[:9]} self_check={args.self_check}', flush=True)

    arms = {'base': base_dir, 'head': head_dir}
    runs: dict[str, list[dict]] = {arm: [] for arm in arms}
    for rnd in range(args.rounds):
        # Alternate the order so a drifting clock cannot favour one arm.
        order = ['base', 'head'] if rnd % 2 == 0 else ['head', 'base']
        for arm in order:
            out = workdir / f'{arm}-round{rnd}.json'
            runs[arm].append(run_measure(arms[arm], out, args))

    base_file = runs['base'][0]['meta']['fla_file']
    head_file = runs['head'][0]['meta']['fla_file']
    print(f'[gate] fla: base={base_file} head={head_file}')
    if base_file == head_file:
        print('[gate] both arms imported the same fla, so the A/B is meaningless', file=sys.stderr)
        return 3

    base_points = median_points(runs['base'])
    head_points = median_points(runs['head'])
    skipped = sorted({p for p in base_points if 'skipped' in base_points[p]} |
                     {p for p in head_points if 'skipped' in head_points[p]})
    if skipped:
        # A point that failed to measure must never be reported as a ratio, and a run that
        # measured nothing must not look like a clean 1.00x.
        print(f'[gate] {len(skipped)} point(s) did not measure: {", ".join(skipped)}', file=sys.stderr)
        return 4
    gaps = group_gaps(runs['base']) + group_gaps(runs['head'])
    if gaps:
        # A kernel only some rounds measured is neither a ratio nor new work.
        print(f'[gate] {len(gaps)} kernel(s) were not measured in every round: {", ".join(gaps)}',
              file=sys.stderr)
        return 4

    detail, summary = compare(base_points, head_points, args)
    regressions = [row for row in detail if row['regress']]
    write_markdown(workdir / 'gate.md', runs['base'][0]['meta'], runs['head'][0]['meta'],
                   detail, summary, regressions, args.self_check, args.rounds)
    (workdir / 'gate.json').write_text(json.dumps(
        {'self_check': args.self_check, 'base': base_points, 'head': head_points,
         'detail': detail, 'summary': summary, 'regressions': regressions}, indent=2))

    worst = max((abs(row['ratio'] - 1) for row in summary if row['ratio'] != float('inf')), default=0.0)
    print(f'[gate] {len(regressions)} kernel regression(s), worst total deviation {worst:.1%}')
    if args.self_check:
        if worst > args.self_check_tol:
            print(f'[gate] the harness is too noisy: {worst:.1%} > {args.self_check_tol:.0%}', file=sys.stderr)
            return 2
        return 0

    failed = [row for row in summary if row['regress']]
    if regressions or failed:
        print('[gate] failing on regression', file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add_shape_flags(p):
        p.add_argument('--heads', default='8,16,32,48,64,128')
        p.add_argument('--seq-lens', default='4096,8192,16384')
        p.add_argument('--dim', type=int, default=128)
        p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16'])
        p.add_argument('--iters', type=int, default=5)
        p.add_argument('--reps', type=int, default=3)
        p.add_argument('--warmup', type=int, default=3)
        p.add_argument('--device', default='cuda:0')

    m = sub.add_parser('measure', help='profile one tree')
    m.add_argument('--repo', default='.')
    m.add_argument('--out', required=True)
    add_shape_flags(m)

    g = sub.add_parser('gate', help='compare a base tree against the current tree')
    g.add_argument('--base-ref', default='origin/v0.4.2')
    g.add_argument('--head-dir', default='.')
    g.add_argument('--workdir', default='out/gate')
    g.add_argument('--rounds', type=int, default=2)
    g.add_argument('--tol', type=float, default=0.05, help='total time tolerance')
    # Two arms running identical code still disagree per kernel, so a tighter tolerance would
    # flag every arm pair as regressed.
    g.add_argument('--per-kernel-tol', type=float, default=0.05)
    g.add_argument('--min-delta-us', type=float, default=8.0)
    g.add_argument('--self-check', action='store_true')
    g.add_argument('--self-check-tol', type=float, default=0.10)
    add_shape_flags(g)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == 'measure':
        result = measure(args)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f'[measure] wrote {out} (fla={result["meta"]["fla_file"]})', flush=True)
        return 0
    return gate(args)


if __name__ == '__main__':
    sys.exit(main())
