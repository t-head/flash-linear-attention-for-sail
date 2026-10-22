#!/usr/bin/env python3
"""GDN end-to-end + per-kernel benchmark campaign (opt vs OSS fla 0.5.2).

Single-file merge of three former helpers:
  * bench_kernel_grid.py -- measure one (arm, H, T) grid point with torch.profiler
  * run_all.sh           -- paired-by-H campaign driver with cold caches
  * aggregate.py         -- markdown tables (T1 E2E, T2/T3 per-kernel, optional T4 CP)

Modes
-----
all        Drive the campaign: for each H run opt then base back-to-back, each
           arm with its own fresh Triton cache, then aggregate tables.
           Run inside the GPU container, or on the host with
           ``--docker-container <name>`` (the container must see /cpfs at the
           same absolute paths as the host, otherwise the worker cwd fails).
bench      Worker: measure ONE arm over the H x T grid and write
           ``grid_<arm>_H<H>.json``. Requires torch + the PPU runtime.
aggregate  Rebuild ``tables.md`` from the JSONs already in --out-dir.
           Pure stdlib; no GPU/torch needed.

Ratio convention: speedup = base_ms / opt_ms  (> 1 means the opt arm is faster).

Cold-cache rule: each arm gets a brand-new TRITON_CACHE_DIR per campaign.
PYTHONPATH selects the fla tree; every JSON records the resolved fla file and
the repo HEAD commit in ``meta`` (verify them before trusting a table).

Examples
--------
Host, full campaign via docker:
  python3 benchmark_gdn_all.py --docker-container relax-ppu

Inside the container:
  python benchmark_gdn_all.py --mode all --out-dir out

Smoke (2 min, one grid point, numbers are NOT conclusions):
  python benchmark_gdn_all.py --mode all --heads 8 --seq-lens 4096 \
      --warmup 2 --repeats 3 --profile-iters 2 --out-dir out/_smoke

Tables only (e.g. after moving JSONs around):
  python benchmark_gdn_all.py --mode aggregate --out-dir out --include-cp

Single-kernel mode (default ``--kernels all`` = whole module):
  python benchmark_gdn_all.py --list-kernels
  python benchmark_gdn_all.py --docker-container relax-ppu \
      --kernels fwd_h,bwd_dhu --heads 8,16,32 --seq-lens 16384
Each unit is timed on its own (CUDA events + profiler) on a context built once
per (H, T), and the campaign writes a K table instead of T1/T2/T3.

Baseline provisioning (``--base-mode``; default ``auto``)
-------------------------------------------------------
CI usually has only a clean checkout of the PR and no prepared base tree.
``auto`` picks the first available: ``repo`` (``--base-repo`` has fla/) ->
``archive`` (``--base-archive`` = wheel/zip/tar.gz, fully offline) ->
``git`` (``--base-ref`` = branch/tag/sha, cut as a ``git worktree`` from
``--opt-repo``) -> ``pip`` (``--base-version``, needs an index).

For PR gating the base arm must be the *target branch*, not fla-0.5.2, or every
PR looks ~1.6x faster and a regression cannot be detected.

CI gate
-------
  # 1) recommended: base = target branch (merge-base / origin/main), 3 sweeps
  python3 benchmark_gdn_all.py --docker-container <ct> --device 0 \
      --opt-repo "$PWD" --base-mode git --base-ref origin/main \
      --sweeps 3 --gate --gate-tol 0.03 --out-dir out/ci_$BUILD_ID

  # 2) offline / no git history: ship a pinned wheel or sdist with the runner
  python3 benchmark_gdn_all.py ... --base-mode archive \
      --base-archive /opt/baseline/flash_linear_attention-0.5.2-py3-none-any.whl

  # 3) preflight only: resolve the base arm and print its identity
  python3 benchmark_gdn_all.py --base-mode archive --base-archive <pkg> \
      --provision-only --out-dir out/ci_preflight

``--sweeps N`` runs the full H x T grid N times per arm and ``--drop-sweeps``
drops the first (compile/clock-ramp) sweeps; the gate judges the median of the
remaining paired speedups. Exit codes: 0 pass, 2 regression, 3 infra/config.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

B = 1
HEAD_DIM = 128
CHUNK = 64

DEFAULT_OPT_REPO = "/cpfs/share/murong/rebase_fla_to_0.5.2"
DEFAULT_BASE_REPO = "/cpfs/share/murong/fla_v052_base"

FLA_MARKERS = ("chunk_gated", "chunk_fwd", "chunk_bwd", "chunk_local_cumsum",
               "recompute_w_u", "prepare_wy", "gdn_gate", "merge_fwd_bwd",
               "pre_process", "solve_tril", "chunk_scaled_dot", "causal_conv")


# ---------------------------------------------------------------------------
# measurement (worker side; torch is imported lazily so host modes stay light)
# ---------------------------------------------------------------------------
def setup_fla():
    import fla
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as gdr
    return gdr, fla.__file__


def _l2norm(x, eps=1e-6):
    x_f32 = x.float()
    rstd = (x_f32 * x_f32).sum(-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rstd).to(x.dtype)


def prepare_inputs(T, H):
    import torch
    import torch.nn.functional as F
    device = "cuda"
    K = V = HEAD_DIM
    scale = K ** (-0.5)
    torch.manual_seed(42)
    q = _l2norm(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16))
    k = _l2norm(torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16))
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32)) / 16
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    do = torch.randn_like(v)
    dht = torch.randn(B, H, K, V, device=device, dtype=torch.float32) / 8
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    return dict(q=q, k=k, v=v, g=g, beta=beta, scale=scale, h0=h0, do=do,
                dht=dht, cu_seqlens=cu_seqlens)


def cleanup_cuda():
    import torch
    try:
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def bench_event(fn, warmup=10, repeats=50):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def _make_fns(d, gdr_fn):
    import torch
    q, k, v, g, beta = d["q"], d["k"], d["v"], d["g"], d["beta"]
    do, dht = d["do"], d["dht"]
    kwargs = dict(scale=d["scale"], initial_state=d["h0"],
                  output_final_state=True, cu_seqlens=d["cu_seqlens"])

    def fwd():
        with torch.no_grad():
            gdr_fn(q=q, k=k, v=v, g=g, beta=beta, **kwargs)

    q_g = q.detach().requires_grad_(True)
    k_g = k.detach().requires_grad_(True)
    v_g = v.detach().requires_grad_(True)
    g_g = g.detach().requires_grad_(True)
    beta_g = beta.detach().requires_grad_(True)
    h0_g = d["h0"].detach().requires_grad_(True)
    params = [q_g, k_g, v_g, g_g, beta_g, h0_g]
    kwargs_g = dict(kwargs, initial_state=h0_g)
    o, fs = gdr_fn(q=q_g, k=k_g, v=v_g, g=g_g, beta=beta_g, **kwargs_g)
    loss = (o * do).sum()
    if fs is not None:
        loss = loss + (fs * dht).sum()

    def bwd():
        torch.autograd.grad(loss, params, retain_graph=True)

    return fwd, bwd


def profile_phase(fn, warmup=3, iters=5):
    import torch
    from torch.profiler import profile, ProfilerActivity
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = {}
    for e in prof.key_averages():
        us = e.self_device_time_total
        if us <= 0:
            continue
        rows[e.key] = {"ms": us / 1000.0 / iters, "count": round(e.count / iters, 3)}
    return rows


# ---------------------------------------------------------------------------
# single-kernel mode (--kernels): one launch unit at a time
# ---------------------------------------------------------------------------
# Unit names follow the per-kernel table rows.  Every unit is one Python wrapper
# call that issues the kernel(s) of that row; `fwd_kkt_solve` is the only unit
# launched from the raw Triton entry point, because its wrapper
# (chunk_gated_delta_rule_fwd_intra) also runs recompute_w_u.  Both arms expose
# identical wrapper signatures, so one harness covers both (opt only adds
# USE_AIU and splits the dw kernel out inside chunk_bwd_dqkwg).
KERNEL_UNITS = (
    "local_cumsum",
    "fwd_kkt_solve",
    "recompute_w_u",
    "fwd_h",
    "fwd_o",
    "bwd_dv_local",
    "bwd_dhu",
    "bwd_dqkwg",
    "prepare_wy_bwd",
)


def parse_kernels(spec):
    """'all' -> ['all']; otherwise validate and return the requested units."""
    items = [x.strip() for x in (spec or "all").split(",") if x.strip()]
    if not items or items == ["all"]:
        return ["all"]
    bad = [x for x in items if x not in KERNEL_UNITS]
    if bad:
        raise SystemExit(f"unknown kernel unit(s): {bad}\n"
                         f"choose from: {', '.join(KERNEL_UNITS)} (or 'all')")
    return items


class GridCtx:
    """Lazily built fwd/bwd intermediates for one (H, T) grid point."""

    def __init__(self, T, H):
        self.d = prepare_inputs(T, H)
        self._fwd_done = False
        self._bwd_done = False
        self.g = self.w = self.u = self.A = None
        self.h = self.v_new = self.o = None
        self.dv_local = None
        self.dh = self.dh0 = self.dv = None

    def fwd(self):
        if self._fwd_done:
            return
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla.ops.common.chunk_o import chunk_fwd_o
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from fla.ops.utils import chunk_local_cumsum
        from fla.ops.utils.constant import RCP_LN2
        d = self.d
        self.g = chunk_local_cumsum(d["g"], chunk_size=CHUNK, scale=RCP_LN2,
                                    cu_seqlens=d["cu_seqlens"])
        self.w, self.u, self.A = chunk_gated_delta_rule_fwd_intra(
            k=d["k"], v=d["v"], g=self.g, beta=d["beta"],
            cu_seqlens=d["cu_seqlens"], chunk_size=CHUNK)
        self.h, self.v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=d["k"], w=self.w, u=self.u, g=self.g, initial_state=d["h0"],
            output_final_state=True, cu_seqlens=d["cu_seqlens"], chunk_size=CHUNK)
        self.o = chunk_fwd_o(q=d["q"], k=d["k"], v=self.v_new, h=self.h,
                             g=self.g, scale=d["scale"],
                             cu_seqlens=d["cu_seqlens"], chunk_size=CHUNK)
        self._fwd_done = True

    def dv_local_tensor(self):
        if self.dv_local is None:
            from fla.ops.common.chunk_o import chunk_bwd_dv_local
            self.fwd()
            d = self.d
            self.dv_local = chunk_bwd_dv_local(
                q=d["q"], k=d["k"], g=self.g, do=d["do"], scale=d["scale"],
                cu_seqlens=d["cu_seqlens"], chunk_size=CHUNK)
        return self.dv_local

    def bwd(self):
        if self._bwd_done:
            return
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
        d = self.d
        dv = self.dv_local_tensor()
        self.dh, self.dh0, self.dv = chunk_gated_delta_rule_bwd_dhu(
            q=d["q"], k=d["k"], w=self.w, g=self.g, h0=d["h0"], dht=d["dht"],
            do=d["do"], dv=dv, scale=d["scale"],
            cu_seqlens=d["cu_seqlens"], chunk_size=CHUNK)
        self._bwd_done = True


def _kkt_solve_fn(ctx):
    """Launch the fused kkt+solve_tril kernel alone (bypasses fwd_intra)."""
    import torch
    from fla.ops.gated_delta_rule.chunk_fwd import \
        chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.utils import prepare_chunk_indices
    d = ctx.d
    k, beta, cu = d["k"], d["beta"], d["cu_seqlens"]
    Bs, T, H, K = k.shape
    HV = beta.shape[2]
    BC = 16
    chunk_indices = prepare_chunk_indices(cu, CHUNK)
    NT = len(chunk_indices)
    A = torch.zeros(Bs, T, HV, CHUNK, device=k.device, dtype=k.dtype)
    grid = (NT, Bs * HV)
    return lambda: chunk_gated_delta_rule_fwd_kkt_solve_kernel[grid](
        k=k, g=ctx.g, beta=beta, A=A, cu_seqlens=cu,
        chunk_indices=chunk_indices, T=T, H=H, HV=HV, K=K,
        BT=CHUNK, BC=BC)


def unit_fn(ctx, unit):
    """Return a zero-arg callable issuing exactly one kernel launch unit.

    All prerequisites are computed before returning, so nothing but the unit
    itself runs inside the timed region.
    """
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    d = ctx.d
    cu, scale = d["cu_seqlens"], d["scale"]

    if unit == "local_cumsum":
        return lambda: chunk_local_cumsum(d["g"], chunk_size=CHUNK,
                                          scale=RCP_LN2, cu_seqlens=cu)
    if unit == "fwd_kkt_solve":
        ctx.fwd()  # only ctx.g is needed, but the setup is reused anyway
        return _kkt_solve_fn(ctx)
    if unit == "recompute_w_u":
        ctx.fwd()
        return lambda: recompute_w_u_fwd(k=d["k"], v=d["v"], beta=d["beta"],
                                         A=ctx.A, g=ctx.g, cu_seqlens=cu)
    if unit == "fwd_h":
        ctx.fwd()
        return lambda: chunk_gated_delta_rule_fwd_h(
            k=d["k"], w=ctx.w, u=ctx.u, g=ctx.g, initial_state=d["h0"],
            output_final_state=True, cu_seqlens=cu, chunk_size=CHUNK)
    if unit == "fwd_o":
        ctx.fwd()
        return lambda: chunk_fwd_o(q=d["q"], k=d["k"], v=ctx.v_new, h=ctx.h,
                                   g=ctx.g, scale=scale, cu_seqlens=cu,
                                   chunk_size=CHUNK)
    if unit == "bwd_dv_local":
        ctx.fwd()
        return lambda: chunk_bwd_dv_local(q=d["q"], k=d["k"], g=ctx.g,
                                          do=d["do"], scale=scale,
                                          cu_seqlens=cu, chunk_size=CHUNK)
    if unit == "bwd_dhu":
        dv = ctx.dv_local_tensor()
        return lambda: chunk_gated_delta_rule_bwd_dhu(
            q=d["q"], k=d["k"], w=ctx.w, g=ctx.g, h0=d["h0"], dht=d["dht"],
            do=d["do"], dv=dv, scale=scale, cu_seqlens=cu, chunk_size=CHUNK)
    if unit == "bwd_dqkwg":
        ctx.bwd()
        return lambda: chunk_bwd_dqkwg(
            q=d["q"], k=d["k"], v=ctx.v_new, w=ctx.w, g=ctx.g, h=ctx.h,
            dv=ctx.dv, do=d["do"], dh=ctx.dh, scale=scale,
            cu_seqlens=cu, chunk_size=CHUNK)
    if unit == "prepare_wy_bwd":
        ctx.bwd()
        # dw/du are produced by dqkwg + dhu; freeze them outside the timed region
        dq, dk, dw, _ = chunk_bwd_dqkwg(
            q=d["q"], k=d["k"], v=ctx.v_new, w=ctx.w, g=ctx.g, h=ctx.h,
            dv=ctx.dv, do=d["do"], dh=ctx.dh, scale=scale,
            cu_seqlens=cu, chunk_size=CHUNK)
        return lambda: prepare_wy_repr_bwd(
            k=d["k"], v=d["v"], beta=d["beta"], g=ctx.g, A=ctx.A,
            dw=dw, du=ctx.dv, cu_seqlens=cu)
    raise ValueError(f"unknown kernel unit: {unit}")


def measure_units(args, units, heads, seqlens):
    """Time each unit alone (CUDA events) and record its device ms (profiler)."""
    results = []
    for H in heads:
        for T in seqlens:
            cleanup_cuda()
            rec = dict(H=H, T=T, mode="kernel", kernels={})
            try:
                ctx = GridCtx(T, H)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[grid] H={H:3d} T={T:6d}  ERROR ctx: {rec['error']}", flush=True)
                results.append(rec)
                continue
            for unit in units:
                try:
                    fn = unit_fn(ctx, unit)
                    wall = bench_event(fn, args.warmup, args.repeats)
                    prof = profile_phase(fn, 3, args.profile_iters)
                    # only fla/triton rows count, matching the T2/T3 denominator;
                    # torch-native fills/reductions are kept aside as other_ms
                    parts = {k: v["ms"] for k, v in prof.items() if is_fla_kernel(k)}
                    other = sum(v["ms"] for k, v in prof.items() if not is_fla_kernel(k))
                    rec["kernels"][unit] = dict(wall=wall, ms=sum(parts.values()),
                                                other_ms=other, parts=parts)
                    print(f"[unit] H={H:3d} T={T:6d} {unit:14s} wall={wall:8.3f} "
                          f"dev={rec['kernels'][unit]['ms']:8.3f} other={other:6.3f} "
                          f"parts={{{', '.join(parts)}}}", flush=True)
                except Exception as e:  # noqa: BLE001 - keep going
                    msg = f"{type(e).__name__}: {e}"
                    rec["kernels"][unit] = dict(error=msg)
                    print(f"[unit] H={H:3d} T={T:6d} {unit:14s} ERROR {msg}", flush=True)
            results.append(rec)
            del ctx
            cleanup_cuda()
    return results


def repo_commit(repo):
    """Git HEAD of the fla tree; falls back to the package version, because a
    tree copied out of a release tarball (e.g. fla_v052_base) has no .git."""
    if not repo:
        return "unknown"
    try:
        r = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    try:
        with open(os.path.join(repo, "COMMIT")) as fh:
            first = fh.readline().strip()
        if first:
            return first
    except Exception:
        pass
    try:
        with open(os.path.join(repo, "fla", "__init__.py")) as fh:
            for line in fh:
                if line.startswith("__version__"):
                    return "v" + line.split("=", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return "unknown"


def repo_dirty(repo):
    """Tracked files with uncommitted changes (untracked junk is ignored)."""
    if not repo:
        return []
    try:
        r = subprocess.run(["git", "-C", repo, "status", "--porcelain",
                            "--untracked-files=no"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return [ln[3:].strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    return []


def cmd_bench(args):
    """Worker: measure one arm over the grid and write a JSON (needs torch)."""
    import torch
    os.environ.pop("FLA_CI_ENV", None)
    # the worker may be started with cwd=<repo>; never trust a relative --out
    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    heads = [int(x) for x in args.heads.split(",")]
    seqlens = [int(x) for x in args.seq_lens.split(",")]
    units = parse_kernels(args.kernels)
    kernel_mode = units != ["all"]
    gdr_fn, fla_file = setup_fla()
    if args.repo and not os.path.abspath(fla_file).startswith(os.path.abspath(args.repo) + os.sep):
        print(f"[warn] fla resolved outside --repo: {fla_file}", flush=True)
    meta = dict(tag=args.tag or args.arm, arm=args.arm, repo=args.repo,
                commit=repo_commit(args.repo), fla_file=fla_file,
                dirty=repo_dirty(args.repo),
                mode="kernel" if kernel_mode else "module", kernels=units,
                torch=torch.__version__, device=torch.cuda.get_device_name(0),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                triton_cache=os.environ.get("TRITON_CACHE_DIR", "<default>"),
                warmup=args.warmup, repeats=args.repeats,
                profile_iters=args.profile_iters,
                heads=heads, seqlens=seqlens, time=time.strftime("%F %T"))
    print(f"[meta] arm={args.arm} mode={meta['mode']} commit={meta['commit']} "
          f"dirty={len(meta['dirty'])} files fla={fla_file} device={meta['device']} "
          f"cache={meta['triton_cache']}",
          flush=True)
    if kernel_mode:
        results = measure_units(args, units, heads, seqlens)
    else:
        results = []
        for H in heads:
            for T in seqlens:
                cleanup_cuda()
                rec = dict(H=H, T=T, mode="module")
                try:
                    d = prepare_inputs(T, H)
                    fwd_fn, bwd_fn = _make_fns(d, gdr_fn)
                    rec["e2e"] = dict(fwd=bench_event(fwd_fn, args.warmup, args.repeats),
                                      bwd=bench_event(bwd_fn, args.warmup, args.repeats))
                    rec["e2e"]["total"] = rec["e2e"]["fwd"] + rec["e2e"]["bwd"]
                    rec["profile"] = {
                        "fwd": profile_phase(fwd_fn, 3, args.profile_iters),
                        "bwd": profile_phase(bwd_fn, 3, args.profile_iters),
                    }
                    del d, fwd_fn, bwd_fn
                    print(f"[grid] H={H:3d} T={T:6d}  e2e_fwd={rec['e2e']['fwd']:.3f} "
                          f"e2e_bwd={rec['e2e']['bwd']:.3f} total={rec['e2e']['total']:.3f} "
                          f"nk_fwd={len(rec['profile']['fwd'])} nk_bwd={len(rec['profile']['bwd'])}",
                          flush=True)
                except Exception as e:  # noqa: BLE001 - long campaign: keep going
                    rec["error"] = f"{type(e).__name__}: {e}"
                    print(f"[grid] H={H:3d} T={T:6d}  ERROR {rec['error']}", flush=True)
                results.append(rec)
                cleanup_cuda()
    with open(args.out, "w") as fh:
        json.dump(dict(meta=meta, results=results), fh, indent=1)
    print("[done] wrote", args.out, flush=True)
    return 0


# ---------------------------------------------------------------------------
# campaign driver (replaces run_all.sh)
# ---------------------------------------------------------------------------
def make_cache(args, prefix):
    """Fresh Triton cache dir, created inside the container when docker is used."""
    if args.docker_container:
        r = subprocess.run(["docker", "exec", args.docker_container,
                            "mktemp", "-d", f"/tmp/{prefix}XXXXXX"],
                           capture_output=True, text=True)
        path = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
        if path:
            return path
        print(f"[warn] mktemp inside container failed (rc={r.returncode}); "
              f"falling back to a random path", flush=True)
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_cache(args, cache):
    if not cache.startswith("/tmp/tc_"):
        return  # never delete anything we did not create
    if args.docker_container:
        subprocess.run(["docker", "exec", args.docker_container, "rm", "-rf", cache],
                       check=False)
    else:
        shutil.rmtree(cache, ignore_errors=True)


# ---------------------------------------------------------------------------
# baseline provisioning (clean PR checkout with no prepared base tree)
# ---------------------------------------------------------------------------
def archive_stale_grids(out_dir, log):
    """Move grid JSONs from an earlier run out of the way.

    The gate pairs sweeps by index, so a leftover grid_opt_s2.json from a
    previous run would silently become an extra sweep and skew the median.
    This also catches a mix of legacy (_H<H>) and sweep (_s<N>) files.
    """
    stale = sorted(glob.glob(os.path.join(out_dir, "grid_*.json")))
    if not stale:
        return
    dest = os.path.join(out_dir, "_stale", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for fp in stale:
        shutil.move(fp, os.path.join(dest, os.path.basename(fp)))
    log(f"[stale] moved {len(stale)} grid JSON(s) from an earlier run -> {dest}")


def looks_like_fla_tree(path):
    return bool(path) and os.path.isfile(os.path.join(path, "fla", "__init__.py"))


def find_fla_root(path, max_depth=3):
    """Directory holding fla/__init__.py (sdists usually nest one level)."""
    if looks_like_fla_tree(path):
        return path
    base_depth = os.path.abspath(path).count(os.sep)
    for root, dirs, _files in os.walk(path):
        if os.path.abspath(root).count(os.sep) - base_depth >= max_depth:
            dirs[:] = []
            continue
        if looks_like_fla_tree(root):
            return root
    return None


def run_checked(cmd, cwd=None):
    print("[cmd] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[base] failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def unpack_fla_archive(src, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    shutil.unpack_archive(src, dest)
    root = find_fla_root(dest)
    if not root:
        raise SystemExit(f"[base] {src} does not contain fla/__init__.py")
    return root


def provision_base(args, log):
    """Resolve the base arm tree; returns (repo, kind, cleanup_or_None).

    Nothing is downloaded when --base-mode=repo/archive; pip mode is the only
    one that needs an index, and it caches the artifact under <out>/_baseline.
    """
    work = os.path.abspath(os.path.join(args.out_dir, "_baseline"))
    os.makedirs(work, exist_ok=True)
    mode = args.base_mode
    if mode == "auto":
        if looks_like_fla_tree(args.base_repo):
            mode = "repo"
        elif args.base_archive:
            mode = "archive"
        elif args.base_ref:
            mode = "git"
        else:
            mode = "pip"
    log(f"[base] mode={mode}")

    if mode == "repo":
        if not looks_like_fla_tree(args.base_repo):
            raise SystemExit(f"[base] --base-repo {args.base_repo} has no fla/__init__.py")
        if args.gate and os.path.abspath(args.base_repo) == os.path.abspath(DEFAULT_BASE_REPO):
            log("[base] WARNING: --gate with the built-in 0.5.2 tree as base. A PR gate "
                "must compare against the target branch (--base-mode git) or a pinned "
                "archive, otherwise a regression cannot be detected")
        log(f"[base] repo {args.base_repo} @ {repo_commit(args.base_repo)}")
        return args.base_repo, mode, None

    if mode == "archive":
        if not args.base_archive or not os.path.exists(args.base_archive):
            raise SystemExit("[base] --base-archive=<whl|zip|tar.gz> is required "
                             "(or use --base-mode repo/git/pip)")
        root = unpack_fla_archive(args.base_archive, os.path.join(work, "src"))
        log(f"[base] archive {args.base_archive} -> {root} @ {repo_commit(root)}")
        return root, mode, None

    if mode == "git":
        if not args.base_ref:
            raise SystemExit("[base] --base-ref=<branch|tag|sha> is required for git mode")
        if not os.path.exists(os.path.join(args.opt_repo, ".git")):
            raise SystemExit(f"[base] {args.opt_repo} is not a git checkout; "
                             "use --base-mode archive (offline) instead")
        dest = os.path.join(work, "worktree")
        if os.path.exists(dest):
            run_checked(["git", "-C", args.opt_repo, "worktree", "remove", "--force", dest])
        run_checked(["git", "-C", args.opt_repo, "worktree", "add", "--detach",
                     dest, args.base_ref])
        sha = run_checked(["git", "-C", dest, "rev-parse", "--short", "HEAD"])
        log(f"[base] git worktree {args.base_ref} -> {dest} ({sha})")
        # the worktree has no COMMIT file; pin the sha so meta shows the ref
        with open(os.path.join(dest, "COMMIT"), "w") as fh:
            fh.write(sha + "\n")

        def _cleanup():
            if args.keep_baseline:
                log(f"[base] keeping worktree {dest}")
                return
            subprocess.run(["git", "-C", args.opt_repo, "worktree", "remove",
                            "--force", dest], check=False)
        return dest, mode, _cleanup

    # pip: download once into <out>/_baseline/dist, then reuse the artifact
    dist = os.path.join(work, "dist")
    os.makedirs(dist, exist_ok=True)
    exts = (".whl", ".zip", ".tar.gz", ".tgz")
    pkgs = [f for f in sorted(os.listdir(dist)) if f.endswith(exts)]
    if not pkgs:
        req = f"flash-linear-attention=={args.base_version}"
        cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "-d", dist, req]
        if args.pip_index_url:
            cmd += ["-i", args.pip_index_url]
        log(f"[base] {req} (needs an index; use --base-archive when offline)")
        run_checked(cmd)
        pkgs = [f for f in sorted(os.listdir(dist)) if f.endswith(exts)]
    if not pkgs:
        raise SystemExit(f"[base] pip download produced no package in {dist}")
    root = unpack_fla_archive(os.path.join(dist, pkgs[0]), os.path.join(work, "src"))
    log(f"[base] {pkgs[0]} -> {root} @ {repo_commit(root)}")
    return root, mode, None


def preflight(args, pairs, log):
    """Fail fast when a repo is missing or invisible inside the container."""
    for name, repo in pairs:
        if not looks_like_fla_tree(repo):
            raise SystemExit(f"[preflight] {name}: no fla/__init__.py under {repo}")
        if args.docker_container:
            inner = os.path.join(repo, "fla", "__init__.py")
            r = subprocess.run(["docker", "exec", args.docker_container,
                                "test", "-f", inner], check=False)
            if r.returncode != 0:
                raise SystemExit(f"[preflight] {name}: {repo} is not visible inside "
                                 f"{args.docker_container} (stale mount?)")
        log(f"[preflight] {name}: {repo} ok")
    if args.docker_container:
        # the worker writes its JSON into --out-dir from inside the container; if
        # that path is not the same mount, the file lands in the container's own
        # filesystem and the driver aborts only after a full cold sweep
        if subprocess.run(["docker", "exec", args.docker_container, "test", "-d", args.out_dir],
                          check=False).returncode != 0:
            raise SystemExit(f"[preflight] --out-dir {args.out_dir} is not visible inside "
                             f"{args.docker_container}; put it on a mount shared "
                             "with the container")
        if subprocess.run(["docker", "exec", args.docker_container, "test", "-w", args.out_dir],
                          check=False).returncode != 0:
            raise SystemExit(f"[preflight] --out-dir {args.out_dir} is not writable inside "
                             f"{args.docker_container}")
        log(f"[preflight] out-dir: {args.out_dir} ok (visible + writable in container)")


def run_worker(args, arm, repo, tag, cache, out_json, heads_s, seqlens_s):
    worker = [args.python, "-u", os.path.abspath(__file__), "--mode", "bench",
              "--arm", arm, "--repo", repo, "--tag", tag, "--out", out_json,
              "--heads", heads_s, "--seq-lens", seqlens_s,
              "--kernels", args.kernels,
              "--warmup", str(args.warmup), "--repeats", str(args.repeats),
              "--profile-iters", str(args.profile_iters)]
    if args.docker_container:
        inner = ["env", f"TRITON_CACHE_DIR={cache}", f"PYTHONPATH={repo}"] + worker
        cmd = ["docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={args.device}",
               "-w", repo, args.docker_container,
               "bash", "-c", 'unset FLA_CI_ENV; exec "$@"', "--"] + inner
        return subprocess.call(cmd)
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = cache
    env["PYTHONPATH"] = repo
    env["CUDA_VISIBLE_DEVICES"] = args.device
    env.pop("FLA_CI_ENV", None)
    return subprocess.call(worker, cwd=repo, env=env)


def cmd_all(args):
    args.out_dir = os.path.abspath(args.out_dir)
    args.opt_repo = os.path.abspath(args.opt_repo)
    if args.base_repo:
        args.base_repo = os.path.abspath(args.base_repo)
    if args.base_archive:
        args.base_archive = os.path.abspath(args.base_archive)
    os.makedirs(args.out_dir, exist_ok=True)
    heads_s = args.heads
    heads = [int(x) for x in heads_s.split(",")]
    progress = os.path.join(args.out_dir, "campaign.log")

    def log(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(progress, "a") as fh:
            fh.write(line + "\n")

    archive_stale_grids(args.out_dir, log)

    arms = []
    cleanup = None
    base_repo = args.base_repo
    if not args.skip_base:
        base_repo, _kind, cleanup = provision_base(args, log)
    if not args.skip_opt:
        arms.append(("opt", args.opt_repo, args.tag_opt))
    if not args.skip_base:
        arms.append(("base", base_repo, args.tag_base))
    if not arms:
        log("ABORT: both arms skipped")
        return 1

    preflight(args, [(name, repo) for name, repo, _ in arms], log)

    caches = {arm: make_cache(args, f"tc_{arm}_grid_") for arm, _, _ in arms}
    log(f"campaign start arms={[a[0] for a in arms]} heads={heads} "
        f"seqlens={args.seq_lens} kernels={args.kernels} sweeps={args.sweeps} "
        f"out={args.out_dir}")
    log(f"caches={caches}")
    t0 = time.time()
    jobs = []
    if args.sweeps and args.sweeps > 0:
        # CI: N full-grid sweeps per arm; the gate medians the paired speedups
        for s in range(args.sweeps):
            for arm, repo, tag in arms:
                jobs.append((f"grid_{arm}_s{s}", arm, repo, tag))
        log(f"sweep mode: {args.sweeps} full grid(s) per arm")
    else:
        # legacy: one worker per H (each worker still measures the whole grid)
        for H in heads:
            for arm, repo, tag in arms:
                jobs.append((f"grid_{arm}_H{H}", arm, repo, tag))
    for name, arm, repo, tag in jobs:
        out_json = os.path.join(args.out_dir, name + ".json")
        ts = time.time()
        rc = run_worker(args, arm, repo, tag, caches[arm], out_json,
                        heads_s, args.seq_lens)
        log(f"[{name}] rc={rc} elapsed={time.time() - ts:.0f}s")
        if rc != 0:
            log(f"ABORT: {name} failed (rc={rc}); partial JSONs kept in {args.out_dir}")
            if cleanup:
                cleanup()
            return 3 if args.gate else rc
        if not os.path.exists(out_json):
            log(f"ABORT: {name} produced no JSON")
            if cleanup:
                cleanup()
            return 3 if args.gate else 1
    log(f"campaign done in {time.time() - t0:.0f}s")
    if not args.keep_cache:
        for cache in caches.values():
            cleanup_cache(args, cache)
    rc = run_aggregate(args)
    if cleanup:
        cleanup()
    return rc


# ---------------------------------------------------------------------------
# aggregation (replaces aggregate.py; stdlib only)
# ---------------------------------------------------------------------------
def is_fla_kernel(name):
    n = name.lower()
    return any(m in n for m in FLA_MARKERS)


def short(name):
    # compact display name: keep the triton kernel symbol
    if "chunk_gated_delta_rule_fwd_kernel_h_blockdim64" in name:
        return "fwd_h_blockdim64"
    if "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64" in name:
        return "bwd_dhu_blockdim64"
    if "chunk_gated_delta_rule_fwd_kkt_solve_kernel" in name:
        return "fwd_kkt_solve"
    if "chunk_fwd_kernel_o" in name:
        return "fwd_o"
    if "chunk_bwd_kernel_o" in name:
        return "bwd_o"
    if "recompute_w_u_fwd_kernel" in name:
        return "recompute_w_u"
    if "prepare_wy_repr_bwd_kernel" in name:
        return "prepare_wy_bwd"
    if "chunk_bwd_kernel_dqkwg" in name:
        return "bwd_dqkwg"
    if "chunk_bwd_kernel_dw" in name:
        return "bwd_dw"
    if "chunk_bwd_kernel_dv_local" in name:
        return "bwd_dv_local"
    if "chunk_local_cumsum_scalar_kernel" in name:
        return "local_cumsum"
    if "chunk_gated_delta_rule_fwd_intra" in name:
        return "fwd_intra"
    return name[:44]


def load_arm(out_dir, prefix):
    data, metas = {}, []
    # matches legacy grid_<arm>_H<H>.json and CI grid_<arm>_s<N>.json
    for fp in sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.json"))):
        j = json.load(open(fp))
        metas.append(j.get("meta", {}))
        for r in j["results"]:
            if "error" in r:
                print(f"[warn] error in {fp}: {r['error']}")
                continue
            data[(r["H"], r["T"])] = r
    return data, metas


def fmt3(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def spd(base, opt):
    if isinstance(base, (int, float)) and isinstance(opt, (int, float)) and opt > 0:
        return base / opt
    return None


def kernel_union_rows(grid_base, grid_opt, phase, T, heads):
    """Ordered kernel list: by summed opt+base ms at T, descending."""
    totals = {}
    for H in heads:
        for src in (grid_opt, grid_base):
            rec = src.get((H, T))
            if not rec:
                continue
            for k, v in rec["profile"][phase].items():
                if not is_fla_kernel(k):
                    continue
                totals[k] = totals.get(k, 0.0) + v["ms"]
    return [k for k, _ in sorted(totals.items(), key=lambda x: -x[1])]


def kernel_cell(grid_base, grid_opt, H, T, phase, k, pick):
    rb = grid_base.get((H, T))
    ro = grid_opt.get((H, T))
    if pick == "fla":
        bv = rb["profile"][phase].get(k) if rb else None
        ov = ro["profile"][phase].get(k) if ro else None
        return (bv["ms"] if bv is not None else None,
                ov["ms"] if ov is not None else None)
    # 'other' bucket = sum of all non-fla rows
    if pick == "other":
        bb = sum(v["ms"] for kk, v in rb["profile"][phase].items()
                 if not is_fla_kernel(kk)) if rb else None
        oo = sum(v["ms"] for kk, v in ro["profile"][phase].items()
                 if not is_fla_kernel(kk)) if ro else None
        return bb, oo
    raise ValueError(pick)


def _meta_line(name, metas):
    m = metas[0] if metas else {}
    if not m:
        return None
    dirty = m.get("dirty")
    if isinstance(dirty, list):
        tag = f", dirty: {', '.join(dirty[:4])}" if dirty else ""
    else:
        tag = ", dirty tree" if dirty else ""
    return (f"- {name}: {m.get('fla_file', '?')} @ {m.get('commit', '?')} "
            f"(tag {m.get('tag', '?')}{tag})")


def kernel_ms(rec, unit):
    if not rec:
        return None
    v = rec.get("kernels", {}).get(unit)
    return v.get("ms") if v else None


def _kernel_cell(gb, go, H, T, unit):
    b = kernel_ms(gb.get((H, T)), unit)
    o = kernel_ms(go.get((H, T)), unit)
    if b is None and o is None:
        return "-"
    s = spd(b, o)
    bs = f"{b:.3f}" if b is not None else "n/a"
    os_ = f"{o:.3f}" if o is not None else "n/a"
    return f"{bs}->{os_} ({s:.2f}x)" if s else f"{bs}->{os_}"


def aggregate_kernels(args, gb, go, mb, mo):
    """K tables for --kernels runs: one row per launch unit, one column per H."""
    gb = {k: v for k, v in gb.items() if v.get("mode") == "kernel"}
    go = {k: v for k, v in go.items() if v.get("mode") == "kernel"}
    if not gb and not go:
        print(f"[error] no kernel-mode records in {args.out_dir}")
        return 1
    heads = sorted({H for H, _ in list(gb) + list(go)})
    ts = sorted({T for _, T in list(gb) + list(go)})
    tag_opt = (mo[0].get("tag") if mo else None) or "opt"
    tag_base = (mb[0].get("tag") if mb else None) or "base"

    totals, parts = {}, {}
    for rec in list(gb.values()) + list(go.values()):
        for unit, v in rec.get("kernels", {}).items():
            if "ms" in v:
                totals[unit] = totals.get(unit, 0.0) + v["ms"]
            parts.setdefault(unit, set()).update(v.get("parts", {}))
    units = [u for u, _ in sorted(totals.items(), key=lambda x: -x[1])]

    lines = []
    w = lines.append
    w(f"# GDN single-kernel: {tag_opt} vs {tag_base} — aggregated tables")
    w("")
    w("speedup = base/opt (> 1 = opt faster)")
    w("ms = device time of the timed unit summed over its fla kernels only, the same "
      "denominator as the T2/T3 columns; torch-native fills/reductions are excluded "
      "into `other_ms` and `wall` adds launch overhead.")
    w(f"grid: base_pts={len(gb)} opt_pts={len(go)}  units={len(units)}")
    for name, metas in (("base", mb), ("opt", mo)):
        line = _meta_line(name, metas)
        if line:
            w(line)
    w("")

    for T in ts:
        w(f"## K_{T}. single-kernel @ T={T} — cell 'base -> opt (speedup)'")
        w("")
        w("| unit | " + " | ".join(f"H{H}" for H in heads) + " |")
        w("|" + "---|" * (len(heads) + 1))
        for unit in units:
            cells = [_kernel_cell(gb, go, H, T, unit) for H in heads]
            if all(c == "-" for c in cells):
                continue
            w(f"| {unit} | " + " | ".join(cells) + " |")
        w("")

    multi = {u: sorted(v) for u, v in parts.items() if len(v) > 1}
    if multi:
        w("> note: units that launch more than one kernel are summed:")
        for unit, names in sorted(multi.items()):
            w(f"> - `{unit}`: {', '.join(names)}")
        w("> base@0.5.2 fuses dw into `chunk_bwd_kernel_dqkwg`; the opt arm splits it out,")
        w("> so `bwd_dqkwg` is already the functionally aligned comparison.")
        w("")

    txt = "\n".join(lines)
    out_fp = os.path.join(args.out_dir, "tables.md")
    with open(out_fp, "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"[done] wrote {out_fp}", flush=True)
    return 0


def run_aggregate(args):
    gb, mb = load_arm(args.out_dir, "grid_base")
    go, mo = load_arm(args.out_dir, "grid_opt")
    cb, cbm = load_arm(args.out_dir, "cp_base") if args.include_cp else ({}, [])
    co, com = load_arm(args.out_dir, "cp_opt") if args.include_cp else ({}, [])
    if not gb and not go:
        print(f"[error] no grid_*.json found in {args.out_dir}")
        return 1

    heads = sorted({H for H, _ in list(gb) + list(go)})
    ts = sorted({T for _, T in list(gb) + list(go)})
    tag_opt = (mo[0].get("tag") if mo else None) or "opt"
    tag_base = (mb[0].get("tag") if mb else None) or "base"
    if "kernel" in {m.get("mode", "module") for m in mb + mo}:
        if len({m.get("mode", "module") for m in mb + mo}) > 1:
            print(f"[warn] mixed module/kernel records in {args.out_dir}; "
                  f"keeping kernel-mode records only")
        return aggregate_kernels(args, gb, go, mb, mo)

    lines = []
    w = lines.append
    w(f"# GDN: {tag_opt} vs {tag_base} — aggregated tables")
    w("")
    w("speedup = base/opt (> 1 = opt faster)")
    w(f"grid: base_pts={len(gb)} opt_pts={len(go)}"
      + (f"  cp: base_pts={len(cb)} opt_pts={len(co)}" if args.include_cp else ""))
    for name, metas in (("base", mb), ("opt", mo)):
        line = _meta_line(name, metas)
        if line:
            w(line)
    w("")

    # ---------------- Table 1: E2E ----------------
    w("## T1. Module-level E2E (chunk_gated_delta_rule, non-CP) — speedup = base/opt")
    w("")
    w("| H | T | base_fwd | base_bwd | base_total | opt_fwd | opt_bwd | opt_total | fwd x | bwd x | total x |")
    w("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for H in heads:
        for T in ts:
            rb, ro = gb.get((H, T)), go.get((H, T))
            if not rb or not ro:
                continue
            b, o = rb["e2e"], ro["e2e"]
            w(f"| {H} | {T} | {fmt3(b['fwd'])} | {fmt3(b['bwd'])} | {fmt3(b['total'])} | "
              f"{fmt3(o['fwd'])} | {fmt3(o['bwd'])} | {fmt3(o['total'])} | "
              f"{fmt3(spd(b['fwd'], o['fwd']))} | {fmt3(spd(b['bwd'], o['bwd']))} | "
              f"{fmt3(spd(b['total'], o['total']))} |")
    w("")

    # ---------------- Table 2/3: per-kernel at each T ----------------
    for T in ts:
        for phase, label in (("fwd", "T2"), ("bwd", "T3")):
            w(f"## {label}_{phase}_{T}. per-kernel {phase} @ T={T} — cell 'base -> opt (speedup)'")
            w("")
            w("| kernel | " + " | ".join(f"H{H}" for H in heads) + " |")
            w("|" + "---|" * (len(heads) + 1))
            union = kernel_union_rows(gb, go, phase, T, heads)
            for k in union:
                cells = []
                for H in heads:
                    b, o = kernel_cell(gb, go, H, T, phase, k, "fla")
                    if b is None and o is None:
                        cells.append("-")
                    else:
                        s = spd(b, o)
                        bs = f"{b:.3f}" if b is not None else "n/a"
                        os_ = f"{o:.3f}" if o is not None else "n/a"
                        cells.append(f"{bs}->{os_} ({s:.2f}x)" if s else f"{bs}->{os_}")
                w("| " + short(k) + " | " + " | ".join(cells) + " |")
            # other bucket
            cells = []
            for H in heads:
                b, o = kernel_cell(gb, go, H, T, phase, None, "other")
                if b is None and o is None:
                    cells.append("-")
                    continue
                s = spd(b, o)
                bs = f"{b:.3f}" if b is not None else "-"
                os_ = f"{o:.3f}" if o is not None else "-"
                cells.append(f"{bs}->{os_} ({s:.2f}x)" if s else f"{bs}->{os_}")
            w("| other(torch-native) | " + " | ".join(cells) + " |")
            # aligned combined row: base computes dw inside dqkwg; opt splits it out
            if phase == "bwd":
                dwk = [k for k in union if k == "chunk_bwd_kernel_dw"]
                qwk = [k for k in union if k == "chunk_bwd_kernel_dqkwg"]
                if dwk and qwk:
                    cells = []
                    for H in heads:
                        b1, o1 = kernel_cell(gb, go, H, T, phase, qwk[0], "fla")
                        b2, o2 = kernel_cell(gb, go, H, T, phase, dwk[0], "fla")
                        bb = None if (b1 is None and b2 is None) else (b1 or 0.0) + (b2 or 0.0)
                        oo = None if (o1 is None and o2 is None) else (o1 or 0.0) + (o2 or 0.0)
                        if bb is None and oo is None:
                            cells.append("-")
                            continue
                        s = spd(bb, oo)
                        bs = f"{bb:.3f}" if bb is not None else "n/a"
                        os_ = f"{oo:.3f}" if oo is not None else "n/a"
                        cells.append(f"{bs}->{os_} ({s:.2f}x)" if s else f"{bs}->{os_}")
                    w("| **bwd_dqkwg+bwd_dw (aligned)** | " + " | ".join(cells) + " |")
                    w("")
                    w("> note: base@0.5.2 computes the dw gradient fused inside `chunk_bwd_kernel_dqkwg`;")
                    w("> opt splits it into `chunk_bwd_kernel_dqkwg` + `chunk_bwd_kernel_dw`.")
                    w("> The aligned row sums both kernels for a like-for-like comparison.")
            w("")

    # ---------------- Table 4: CP kernels (optional input) ----------------
    if args.include_cp and (cb or co):
        cp_heads = sorted({H for H, _ in list(cb) + list(co)})
        cp_ts = sorted({T for _, T in list(cb) + list(co)})
        for T in cp_ts:
            w(f"## T4_cp_{T}. CP pre-process kernels @ T={T} — cell 'base -> opt (speedup) [path]'")
            w("")
            for phase in ("fwd", "bwd", "bwd_merged"):
                w(f"### phase={phase}")
                w("")
                w("| H | base(ms) | opt(ms) | speedup | path base->opt |")
                w("|---:|---:|---:|---:|---|")
                for H in cp_heads:
                    rb, ro = cb.get((H, T)), co.get((H, T))
                    if not rb or not ro:
                        continue
                    pb = rb.get(phase)
                    po = ro.get(phase)
                    if not pb or not po:
                        continue
                    s = spd(pb["median"], po["median"])
                    w(f"| {H} | {pb['median']:.4f} | {po['median']:.4f} | {s:.2f}x | "
                      f"{pb['path']}->{po['path']} |")
                w("")

    txt = "\n".join(lines)
    out_fp = os.path.join(args.out_dir, "tables.md")
    with open(out_fp, "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"[done] wrote {out_fp}", flush=True)
    if args.gate:
        return run_gate(args)
    return 0


# ---------------------------------------------------------------------------
# CI gate: median-of-N paired sweeps + regression verdict
# ---------------------------------------------------------------------------
def load_sweeps(out_dir, prefix):
    """All grid JSONs of one arm, ordered by measurement time."""
    files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.json")),
                   key=lambda f: json.load(open(f)).get("meta", {}).get("time", ""))
    sweeps, metas = [], []
    for fp in files:
        j = json.load(open(fp))
        metas.append(j.get("meta", {}))
        sweeps.append({(r["H"], r["T"]): r for r in j.get("results", [])
                       if "error" not in r and "e2e" in r})
    return sweeps, metas


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _meta_ident(m):
    return (f"{m.get('tag', '?')} @ {m.get('commit', '?')} "
            f"({m.get('fla_file', '?')})")


def run_gate(args):
    """Judge PR regressions. 0 = pass, 2 = regression, 3 = infra/config."""
    sb, mb = load_sweeps(args.out_dir, "grid_base")
    so, mo = load_sweeps(args.out_dir, "grid_opt")
    if not sb or not so:
        print(f"[gate] no grid_base/grid_opt JSON in {args.out_dir}", flush=True)
        return 3
    if any(m.get("mode") == "kernel" for m in mb + mo):
        print("[gate] --gate expects module mode; drop --kernels", flush=True)
        return 3

    problems = []
    n_total = min(len(sb), len(so))
    drop = min(max(args.drop_sweeps, 0), max(0, n_total - 2))
    sb, so = sb[drop:], so[drop:]
    n = min(len(sb), len(so))
    mb0, mo0 = (mb[-1] if mb else {}), (mo[-1] if mo else {})
    if n < 2:
        problems.append(f"only {n} usable sweep(s); need >=2 for a stable median")
    fb, fo = mb0.get("fla_file"), mo0.get("fla_file")
    if fb and fo and os.path.realpath(fb) == os.path.realpath(fo):
        problems.append(f"both arms resolved to the same fla tree: {fb}")
    for name, m in (("base", mb0), ("opt", mo0)):
        if m.get("dirty") and not args.allow_dirty:
            problems.append(f"{name} tree has tracked changes: {m['dirty']}")
    if mb0.get("torch") and mo0.get("torch") and mb0["torch"] != mo0["torch"]:
        problems.append(f"torch differs: base={mb0['torch']} opt={mo0['torch']}")

    phases = [p.strip() for p in args.gate_phase.split(",") if p.strip()]
    bad = [p for p in phases if p not in ("fwd", "bwd", "total")]
    if bad:
        print(f"[gate] unknown phase(s) {bad}", flush=True)
        return 3

    pts = set(sb[0]) & set(so[0])
    for s in sb[1:] + so[1:]:
        pts &= set(s)
    # a point that failed in one arm only would silently drop out of the gate
    pb = set(sb[0])
    po = set(so[0])
    for s in sb[1:]:
        pb &= set(s)
    for s in so[1:]:
        po &= set(s)
    if pb != po:
        problems.append(f"point sets differ: only-base={sorted(pb - po)[:6]}, "
                        f"only-opt={sorted(po - pb)[:6]}")
    pts = sorted(pts)
    if not pts:
        print("[gate] no common (H,T) points between arms", flush=True)
        return 3

    cells = []
    for H, T in pts:
        for ph in phases:
            sp = [sb[i][(H, T)]["e2e"][ph] / so[i][(H, T)]["e2e"][ph] for i in range(n)]
            cells.append(dict(
                H=H, T=T, phase=ph, speedup=median(sp), min=min(sp), max=max(sp),
                base_ms=median([sb[i][(H, T)]["e2e"][ph] for i in range(n)]),
                opt_ms=median([so[i][(H, T)]["e2e"][ph] for i in range(n)]),
                reason=None))

    tol = args.gate_tol
    # Two distinct failure modes: slower than base (a real regression) and
    # faster than base but below the requested bar (a shortfall).  Keeping them
    # apart stops the log from calling a 1.08x point a "regression".
    regressions, shortfalls = [], []
    for c in cells:
        if c["speedup"] < 1.0 - tol:
            c["reason"] = "regression"
            regressions.append(c)
        elif args.min_speedup > 0 and c["speedup"] < args.min_speedup:
            c["reason"] = "shortfall"
            shortfalls.append(c)
    failing = regressions + shortfalls

    pooled = {ph: median([c["speedup"] for c in cells if c["phase"] == ph])
              for ph in phases}
    verdict = "infra" if problems else ("fail" if failing else "pass")
    payload = dict(verdict=verdict,
                   thresholds=dict(gate_tol=tol, min_speedup=args.min_speedup),
                   sweeps=dict(files_base=len(mb), files_opt=len(mo), used=n, dropped=drop),
                   pooled=pooled, problems=problems, cells=cells,
                   failing=[dict(H=c["H"], T=c["T"], phase=c["phase"],
                                 reason=c["reason"], speedup=c["speedup"],
                                 base_ms=c["base_ms"], opt_ms=c["opt_ms"],
                                 sweep_min=c["min"], sweep_max=c["max"])
                            for c in sorted(failing, key=lambda c: c["speedup"])],
                   base=dict(tag=mb0.get("tag"), commit=mb0.get("commit"),
                             fla_file=mb0.get("fla_file"), dirty=mb0.get("dirty"),
                             torch=mb0.get("torch"), device=mb0.get("device")),
                   opt=dict(tag=mo0.get("tag"), commit=mo0.get("commit"),
                            fla_file=mo0.get("fla_file"), dirty=mo0.get("dirty"),
                            torch=mo0.get("torch"), device=mo0.get("device")))
    with open(os.path.join(args.out_dir, "gate.json"), "w") as fh:
        json.dump(payload, fh, indent=1)

    lines = [f"# GDN performance gate: {verdict.upper()}", ""]
    lines.append(f"- opt: {_meta_ident(mo0)}")
    lines.append(f"- base: {_meta_ident(mb0)}")
    lines.append(f"- sweeps: used={n} (dropped {drop} of {n_total} pairs)")
    lines.append(f"- thresholds: speedup >= {1.0 - tol:.3f}"
                 + (f", >= {args.min_speedup:.3f} (min-speedup)"
                    if args.min_speedup > 0 else ""))
    lines.append("- pooled: " + ", ".join(f"{ph} {v:.3f}x" for ph, v in pooled.items()))
    lines.append(f"- result: {len(regressions)} regression(s), "
                 f"{len(shortfalls)} shortfall(s), {len(problems)} infra problem(s)")
    if problems:
        lines += ["", "## Infra problems"] + [f"- {p}" for p in problems]
    if regressions:
        lines += ["", f"## Regressions ({len(regressions)}) - opt slower than base",
                  "",
                  "| H | T | phase | base ms | opt ms | speedup | regression | sweeps min..max |",
                  "|---:|---:|---|---:|---:|---:|---:|---|"]
        for c in sorted(regressions, key=lambda c: c["speedup"]):
            lines.append(f"| {c['H']} | {c['T']} | {c['phase']} | {c['base_ms']:.3f} | "
                         f"{c['opt_ms']:.3f} | {c['speedup']:.3f}x | "
                         f"-{(1 - c['speedup']) * 100:.1f}% | "
                         f"{c['min']:.3f}..{c['max']:.3f} |")
    if shortfalls:
        lines += ["", f"## Shortfalls ({len(shortfalls)}) - faster than base, "
                      f"below --min-speedup {args.min_speedup:.3f}x",
                  "",
                  "| H | T | phase | base ms | opt ms | speedup | gap to bar | sweeps min..max |",
                  "|---:|---:|---|---:|---:|---:|---:|---|"]
        for c in sorted(shortfalls, key=lambda c: c["speedup"]):
            lines.append(f"| {c['H']} | {c['T']} | {c['phase']} | {c['base_ms']:.3f} | "
                         f"{c['opt_ms']:.3f} | {c['speedup']:.3f}x | "
                         f"{(args.min_speedup - c['speedup']) * 100:.1f}% | "
                         f"{c['min']:.3f}..{c['max']:.3f} |")
    lines += ["", "## All points (median of paired sweeps)",
              "| H | T | " + " | ".join(f"{ph} x" for ph in phases) + " |",
              "|---:|---:|" + "---:|" * len(phases)]
    by_pt = {}
    for c in cells:
        by_pt.setdefault((c["H"], c["T"]), {})[c["phase"]] = c["speedup"]
    for (H, T), d in sorted(by_pt.items()):
        lines.append(f"| {H} | {T} | "
                     + " | ".join(f"{d[ph]:.3f}" for ph in phases) + " |")
    gate_md = os.path.join(args.out_dir, "gate.md")
    with open(gate_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[gate] {verdict.upper()}: {len(pts)} points x {len(phases)} phase(s), "
          f"sweeps used={n} dropped={drop}, pooled "
          + ", ".join(f"{ph}={v:.3f}x" for ph, v in pooled.items()), flush=True)
    if regressions:
        print("[gate] " + "!" * 72, flush=True)
        print(f"[gate] PERFORMANCE REGRESSION: {len(regressions)} of {len(cells)} cells "
              f"are slower than base by more than {tol:.1%}", flush=True)
        for c in sorted(regressions, key=lambda c: c["speedup"]):
            print(f"[gate]   H={c['H']:<3d} T={c['T']:<6d} {c['phase']:<5s} "
                  f"{c['base_ms']:9.3f} -> {c['opt_ms']:9.3f} ms   "
                  f"{c['speedup']:.3f}x  (-{(1 - c['speedup']) * 100:5.1f}%)", flush=True)
        print("[gate] " + "!" * 72, flush=True)
    if shortfalls:
        print(f"[gate] IMPROVEMENT SHORTFALL: {len(shortfalls)} of {len(cells)} cells are "
              f"faster than base but below --min-speedup {args.min_speedup:.3f}x", flush=True)
        for c in sorted(shortfalls, key=lambda c: c["speedup"]):
            print(f"[gate]   H={c['H']:<3d} T={c['T']:<6d} {c['phase']:<5s} "
                  f"{c['speedup']:.3f}x  (need {args.min_speedup:.3f}x)", flush=True)
    for p in problems:
        print(f"[gate]   infra: {p}", flush=True)
    if failing:
        print(f"[gate] full list: {gate_md}", flush=True)
        print(f"[gate] per-kernel detail: {os.path.join(args.out_dir, 'tables.md')}", flush=True)
    print(f"[gate] wrote {gate_md} and gate.json", flush=True)
    if problems:
        return 3
    return 2 if failing else 0


# ---------------------------------------------------------------------------
def build_parser():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="GDN E2E + per-kernel benchmark campaign (opt vs OSS fla 0.5.2)")
    ap.add_argument("--mode", choices=("all", "bench", "aggregate"), default="all")
    # campaign
    ap.add_argument("--opt-repo", default=DEFAULT_OPT_REPO, help="fla tree of the opt arm")
    ap.add_argument("--base-repo", default=DEFAULT_BASE_REPO, help="fla tree of the base arm")
    ap.add_argument("--base-mode", choices=("auto", "repo", "archive", "git", "pip"),
                    default="auto",
                    help="how to obtain the base tree: repo | archive (offline) | "
                         "git worktree of --base-ref | pip download | auto")
    ap.add_argument("--base-archive", default="",
                    help="wheel/zip/tar.gz that contains fla/ (offline baseline)")
    ap.add_argument("--base-ref", default="",
                    help="git ref for --base-mode git (e.g. origin/main, tag, sha)")
    ap.add_argument("--base-version", default="0.5.2",
                    help="pypi version for --base-mode pip")
    ap.add_argument("--pip-index-url", default="", help="index url for --base-mode pip")
    ap.add_argument("--keep-baseline", action="store_true",
                    help="keep a provisioned git worktree after the run")
    ap.add_argument("--provision-only", action="store_true",
                    help="resolve --base-mode, print the base identity, then exit")
    ap.add_argument("--heads", default="8,16,32,48,64,128")
    ap.add_argument("--seq-lens", default="4096,8192,16384")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--profile-iters", type=int, default=5)
    ap.add_argument("--tag-opt", default="opt")
    ap.add_argument("--tag-base", default="base")
    ap.add_argument("--out-dir", default=os.path.join(here, "out"))
    ap.add_argument("--docker-container", default="",
                    help="run workers via `docker exec -e CUDA_VISIBLE_DEVICES=<device>`")
    ap.add_argument("--device", default="0")
    ap.add_argument("--python", default="python", help="interpreter used inside the worker")
    ap.add_argument("--skip-opt", action="store_true")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the campaign Triton cache dirs (default: remove them)")
    ap.add_argument("--sweeps", type=int, default=0,
                    help="CI mode: run the whole grid N times per arm "
                         "(0 = legacy one worker per H)")
    ap.add_argument("--drop-sweeps", type=int, default=1,
                    help="gate: drop the first N sweeps (compile/clock ramp)")
    ap.add_argument("--gate", action="store_true",
                    help="CI gate: judge median paired speedups, write gate.md/gate.json; "
                         "exit 2 on regression, 3 on infra/config problem")
    ap.add_argument("--gate-phase", default="total",
                    help="comma list of fwd,bwd,total to gate on (default: total)")
    ap.add_argument("--gate-tol", type=float, default=0.03,
                    help="fail a point when its median speedup < 1 - tol (default 0.03)")
    ap.add_argument("--min-speedup", type=float, default=0.0,
                    help="also fail points below this speedup (improvement PRs)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="do not fail the gate on tracked changes in either tree")
    ap.add_argument("--include-cp", action="store_true",
                    help="aggregate also cp_{opt,base}_H*.json into T4 tables")
    ap.add_argument("--kernels", default="all",
                    help="'all' = whole module (T1/T2/T3); or a comma list of launch "
                         "units from --list-kernels to time one kernel at a time (K tables)")
    ap.add_argument("--list-kernels", action="store_true",
                    help="print the selectable single-kernel units and exit")
    # bench worker
    ap.add_argument("--arm", choices=("opt", "base"), default="opt")
    ap.add_argument("--repo", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    return ap


def main():
    args = build_parser().parse_args()
    if args.list_kernels:
        print("selectable units for --kernels (default: all):")
        for unit in KERNEL_UNITS:
            print(f"  {unit}")
        return 0
    if args.provision_only:
        args.out_dir = os.path.abspath(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        repo, kind, cleanup = provision_base(args, lambda m: print(m, flush=True))
        inner = os.path.join(repo, "fla", "__init__.py")
        visible = None
        if args.docker_container:
            visible = subprocess.run(["docker", "exec", args.docker_container,
                                      "test", "-f", inner], check=False).returncode == 0
        print(json.dumps(dict(kind=kind, repo=repo, fla_file=inner,
                              commit=repo_commit(repo), docker=args.docker_container,
                              visible_in_container=visible), indent=1))
        if cleanup:
            cleanup()
        return 0
    parse_kernels(args.kernels)  # validate early, before any GPU work
    if args.gate:
        if args.mode == "bench":
            raise SystemExit("[gate] --gate only applies to --mode all/aggregate")
        if args.kernels != "all":
            raise SystemExit("[gate] --gate compares module-level e2e; drop --kernels")
        if args.skip_base or args.skip_opt:
            raise SystemExit("[gate] --gate needs both arms")
        if args.mode == "all":
            usable = args.sweeps - max(args.drop_sweeps, 0) if args.sweeps > 0 else 0
            if usable < 2:
                raise SystemExit(f"[gate] need >=2 usable sweeps, got sweeps={args.sweeps} "
                                 f"- drop={args.drop_sweeps}; try --sweeps 3 --drop-sweeps 1")
    if args.mode == "bench":
        if not args.out:
            raise SystemExit("--out is required in bench mode")
        return cmd_bench(args)
    if args.mode == "aggregate":
        args.out_dir = os.path.abspath(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        return run_aggregate(args)
    return cmd_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
