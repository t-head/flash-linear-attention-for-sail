# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Micro-benchmark for the GDN Context-Parallel (CP) pre-process kernels.

Target kernels (fla/ops/cp/chunk_delta_h.py):
  - pre_process_fwd_kernel_merged   (computes h[K,V] and m[K,K])
  - pre_process_bwd_kernel_merged   (computes dh[K,V] and dm[K,K])

Unlike benchmarks/modules/benchmark_chunk_gated_delta_rule.py (which does NOT
pass a cp_context and therefore never launches these kernels), this script
exercises the CP pre-process path directly.

Two modes:
  * kernel  : construct the exact tensors / strides / BLOCK_SIZE / BK1 / grid
              used by the Python wrappers and launch a single merged kernel.
              This isolates the micro-kernel from all-gather and the other GDN
              kernels, and is what the ACU harness profiles.
  * cp-e2e  : run the real chunk_gated_delta_rule under a torch.distributed CP
              group (default CP4). Middle ranks exercise both the fwd
              pre-process and the bwd pre-process/merge, validating that
              micro-kernel gains reach the real pipeline.

Two variants per phase (kernel mode, --variants):
  * merged : one kernel covering both the h[K,V] and the m[K,K] part of the
              state. BLOCK_SIZE is a hand-set constant on the pre-split trees
              and an autotuned constexpr once the merged kernel gained
              {BLOCK_SIZE} configs, so the launcher detects which applies.
  * split  : stage1 (h or dh) + stage2 (m or dm) on disjoint write regions,
              which is what the CP wrapper runs for its enable-listed heads.
              Skipped with a notice on trees that predate the split kernels.

Frozen contract: this is a NEW benchmark only. It does not touch the kernels,
the reference implementation, tolerances, or fp32 accumulation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
from datetime import datetime

import torch
import triton

# RCP_LN2 = 1 / ln(2); GDN passes g already scaled by this into chunk cumsum.
RCP_LN2 = 1.4426950408889634

# Frozen GDN / Qwen3.5-35B-A3B shape family: K == V == HEAD_DIM, BT == CHUNK.
HEAD_DIM = 128
CHUNK = 64
SEED = 42


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def _percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q / 100.0 * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _bootstrap_ci(samples, n_boot=2000, ci=0.95, seed=SEED):
    """Bootstrap CI of the median. Returns (lo, hi)."""
    if len(samples) < 2:
        v = samples[0] if samples else float("nan")
        return v, v
    rng = random.Random(seed)
    n = len(samples)
    meds = []
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        resample.sort()
        meds.append(_percentile(resample, 50))
    meds.sort()
    lo = _percentile(meds, (1 - ci) / 2 * 100)
    hi = _percentile(meds, (1 + ci) / 2 * 100)
    return lo, hi


def summarize(samples):
    """samples: list of per-iteration latencies in milliseconds."""
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / (n - 1) if n > 1 else 0.0
    lo, hi = _bootstrap_ci(s)
    return {
        "n": n,
        "median": _percentile(s, 50),
        "mean": mean,
        "std": math.sqrt(var),
        "min": s[0],
        "p10": _percentile(s, 10),
        "p90": _percentile(s, 90),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "raw": samples,
    }


# ---------------------------------------------------------------------------
# input construction (kernel mode)
# ---------------------------------------------------------------------------
def _chunked_cumsum_gate(T, HV, device):
    """Mimic chunk_local_cumsum(g, scale=RCP_LN2): per-chunk cumulative decay.

    Values are non-positive within a chunk (monotonically decreasing), which
    keeps every exp2(...) argument in the kernel bounded, matching real inputs.
    The kernel always loads g as fp32, so we keep it fp32.
    """
    g = torch.nn.functional.logsigmoid(torch.randn(T, HV, device=device, dtype=torch.float32))
    g = g * RCP_LN2
    nt = triton.cdiv(T, CHUNK)
    pad = nt * CHUNK - T
    if pad:
        g = torch.cat([g, torch.zeros(pad, HV, device=device, dtype=torch.float32)], dim=0)
    g = g.view(nt, CHUNK, HV).cumsum(dim=1).view(nt * CHUNK, HV)[:T].contiguous()
    return g


def build_kernel_inputs(T, H, dtype, device):
    """Construct GDN pre-process kernel tensors (H == HV, USE_G=True, no gk/bg)."""
    K = V = HEAD_DIM
    HV = H
    scale = K ** -0.5
    k = torch.randn(T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(T, HV, V, device=device, dtype=dtype) * 0.1
    w = torch.randn(T, HV, K, device=device, dtype=dtype) * 0.1
    g = _chunked_cumsum_gate(T, HV, device)
    # backward-only tensors
    q = torch.randn(T, H, K, device=device, dtype=dtype) * 0.1
    do = torch.randn(T, HV, V, device=device, dtype=dtype) * 0.1
    dv = torch.randn(T, HV, V, device=device, dtype=dtype) * 0.1
    return dict(k=k, v=v, w=w, g=g, q=q, do=do, dv=dv, K=K, V=V, HV=HV, scale=scale)


def _autotuner_attr(kernel, name):
    """Read attribute ``name`` from the innermost Autotuner wrapping ``kernel``."""
    obj = kernel
    for _ in range(4):
        if hasattr(obj, name):
            return getattr(obj, name)
        obj = getattr(obj, "fn", None)
        if obj is None:
            break
    return None


def _block_size_is_autotuned(kernel):
    """True when BLOCK_SIZE is one of the kernel's autotune config keys."""
    configs = _autotuner_attr(kernel, "configs")
    return bool(configs) and any("BLOCK_SIZE" in c.kwargs for c in configs)


def _split_stage_kernels(phase):
    """Stage kernels of the split path, or None on trees that predate it."""
    from fla.ops.cp import chunk_delta_h as m
    names = {
        "fwd": ("pre_process_fwd_stage1_h", "pre_process_fwd_stage2_m"),
        "bwd": ("pre_process_bwd_stage1_dh", "pre_process_bwd_stage2_dm_v2"),
    }[phase]
    if not all(hasattr(m, n) for n in names):
        return None
    return tuple(getattr(m, n) for n in names)


def make_fwd_launch(inp, T, H, variant="merged"):
    from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged
    K, V, HV = inp["K"], inp["V"], inp["HV"]
    BT = CHUNK
    BK = triton.next_power_of_2(K)
    hm = inp["k"].new_zeros(HV, K, V + K, dtype=torch.float32)

    if variant == "split":
        stages = _split_stage_kernels("fwd")
        if stages is None:
            return None, None, None
        stage1, stage2 = stages

        def stage1_grid(meta): return (triton.cdiv(V, meta["BLOCK_SIZE"]), HV)
        def stage2_grid(meta): return (triton.cdiv(K, meta["BLOCK_SIZE"]), HV)

        def launch():
            stage1[stage1_grid](
                k=inp["k"], v=inp["v"], w=inp["w"], g=inp["g"], gk=None, bg=None,
                u=inp["v"], hm=hm, cu_seqlens=None, T=T, H=H, HV=HV, K=K, V=V,
                BT=BT, USE_BG=False,
            )
            stage2[stage2_grid](
                k=inp["k"], w=inp["w"], g=inp["g"], gk=None, hm=hm, cu_seqlens=None,
                T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK1=BK, USE_BG=False,
            )
        return launch, (stage1, stage2), hm

    if _block_size_is_autotuned(pre_process_fwd_kernel_merged):

        def grid(meta): return (triton.cdiv(V, meta["BLOCK_SIZE"]) + triton.cdiv(K, meta["BLOCK_SIZE"]), HV)

        def launch():
            pre_process_fwd_kernel_merged[grid](
                k=inp["k"], v=inp["v"], w=inp["w"], g=inp["g"], gk=None, bg=None,
                u=inp["v"], hm=hm, cu_seqlens=None, T=T, H=H, HV=HV, K=K, V=V,
                BT=BT, BK1=BK, MULTI_SEQS=False,
            )
        return launch, pre_process_fwd_kernel_merged, hm

    BLOCK_SIZE = 32 if K <= 64 else 64
    static_grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)

    def launch():
        pre_process_fwd_kernel_merged[static_grid](
            k=inp["k"], v=inp["v"], w=inp["w"], g=inp["g"], gk=None, bg=None,
            u=inp["v"], hm=hm, cu_seqlens=None, T=T, H=H, HV=HV, K=K, V=V,
            BT=BT, BLOCK_SIZE=BLOCK_SIZE, BK1=BK, MULTI_SEQS=False,
        )
    return launch, pre_process_fwd_kernel_merged, hm


def make_bwd_launch(inp, T, H, variant="merged"):
    from fla.ops.cp.chunk_delta_h import pre_process_bwd_kernel_merged
    K, V, HV = inp["K"], inp["V"], inp["HV"]
    BT = CHUNK
    BK = triton.next_power_of_2(K)
    dhm = inp["q"].new_zeros(HV, K, V + K, dtype=torch.float32)

    if variant == "split":
        stages = _split_stage_kernels("bwd")
        if stages is None:
            return None, None, None
        stage1, stage2 = stages

        def stage1_grid(meta): return (triton.cdiv(V, meta["BLOCK_SIZE"]), HV)
        def stage2_grid(meta): return (triton.cdiv(K, meta["BLOCK_SIZE"]), HV)

        def launch():
            stage1[stage1_grid](
                q=inp["q"], k=inp["k"], w=inp["w"], g=inp["g"], gk=None,
                do=inp["do"], dhm=dhm, dv=inp["dv"], cu_seqlens=None,
                scale=inp["scale"], T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_BG=False,
            )
            stage2[stage2_grid](
                k=inp["k"], w=inp["w"], g=inp["g"], gk=None, dhm=dhm, cu_seqlens=None,
                T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK1=BK, USE_BG=False,
            )
        return launch, (stage1, stage2), dhm

    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)

    def launch():
        pre_process_bwd_kernel_merged[grid](
            q=inp["q"], k=inp["k"], w=inp["w"], g=inp["g"], gk=None,
            do=inp["do"], dhm=dhm, dv=inp["dv"], cu_seqlens=None,
            scale=inp["scale"], T=T, H=H, HV=HV, K=K, V=V,
            BT=BT, BLOCK_SIZE=BLOCK_SIZE, BK1=BK, USE_BG=False,
        )
    return launch, pre_process_bwd_kernel_merged, dhm


def _as_kernels(kernel):
    return kernel if isinstance(kernel, tuple) else (kernel,)


def pin_config(kernel, num_warps, num_stages):
    """Pin one config for controlled schedule A/B measurements."""
    for k in _as_kernels(kernel):
        obj = k
        for _ in range(4):
            if hasattr(obj, "configs"):
                obj.configs = [triton.Config({}, num_warps=num_warps, num_stages=num_stages)]
                break
            obj = getattr(obj, "fn", None)
            if obj is None:
                raise RuntimeError("Triton Autotuner not found in wrapper chain")


def autotune_winner(kernel):
    """Extract the cached best triton.Config(s) after warmup.

    The kernel object is the heuristics wrapper; the Autotuner that owns the
    winner cache sits one or more levels down under ``.fn``.
    """
    out = {}
    for idx, k in enumerate(_as_kernels(kernel)):
        obj = k
        for _ in range(4):
            cache = getattr(obj, "cache", None)
            if isinstance(cache, dict) and cache:
                for key, cfg in cache.items():
                    prefix = "" if len(_as_kernels(kernel)) == 1 else ("stage1" if idx == 0 else "stage2")
                    out[f"{prefix}{key}".lstrip()] = str(cfg)
                break
            obj = getattr(obj, "fn", None)
            if obj is None:
                break
    return out


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_launch(launch, warmup, repeats, single_launch):
    torch.cuda.synchronize()
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    if single_launch:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end)]

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


# ---------------------------------------------------------------------------
# kernel mode
# ---------------------------------------------------------------------------
def run_kernel_mode(args, device, dtype):
    results = []
    phases = []
    if args.phase in ("fwd", "both"):
        phases.append("fwd")
    if args.phase in ("bwd", "both"):
        phases.append("bwd")
    for H in args.heads:
        for T in args.seq_lens:
            inp = build_kernel_inputs(T, H, dtype, device)
            for phase in phases:
                for variant in args.variants:
                    if phase == "fwd":
                        launch, kernel, _ = make_fwd_launch(inp, T, H, variant)
                    else:
                        launch, kernel, _ = make_bwd_launch(inp, T, H, variant)
                    if launch is None:
                        print(f"[kernel] phase={phase:3s} H={H:3d} T={T:6d}  "
                              f"variant={variant}: not available in this tree, skipped")
                        continue
                    if args.pin_num_warps is not None:
                        pin_config(kernel, args.pin_num_warps, args.pin_num_stages)
                    samples = time_launch(launch, args.warmup, args.repeats, args.single_launch)
                    stats = summarize(samples)
                    winner = autotune_winner(kernel)
                    rec = {
                        "mode": "kernel", "phase": phase, "variant": variant,
                        "H": H, "HV": H,
                        "K": HEAD_DIM, "V": HEAD_DIM, "T": T, "BT": CHUNK,
                        "dtype": str(dtype),
                        "autotune": winner,
                        "stats": {k: v for k, v in stats.items() if k != "raw"},
                        "raw_ms": stats["raw"],
                    }
                    results.append(rec)
                    print(f"[kernel] phase={phase:3s} variant={variant:6s} H={H:3d} T={T:6d}  "
                          f"median={stats['median']:.4f}ms  ci95=[{stats['ci95_lo']:.4f},{stats['ci95_hi']:.4f}]  "
                          f"min={stats['min']:.4f}  std={stats['std']:.4f}")
    return results


# ---------------------------------------------------------------------------
# cp-e2e mode (torch.distributed, default CP4)
# ---------------------------------------------------------------------------
def _cp_worker(rank, world_size, T_local, H, dtype_str, warmup, repeats, port, ret_path):
    import torch.distributed as dist
    import torch.nn.functional as F
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    dtype = getattr(torch, dtype_str)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    B = 1
    D = HEAD_DIM
    T_global = T_local * world_size
    torch.manual_seed(SEED)
    q_g = F.normalize(torch.randn(B, T_global, H, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k_g = F.normalize(torch.randn(B, T_global, H, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v_g = torch.randn(B, T_global, H, D, device=device, dtype=dtype)
    g_g = F.logsigmoid(torch.randn(B, T_global, H, device=device, dtype=dtype))
    beta_g = torch.randn(B, T_global, H, device=device, dtype=torch.float32).sigmoid()
    do_g = torch.randn(B, T_global, H, D, device=device, dtype=dtype)
    for t in (q_g, k_g, v_g, g_g, beta_g, do_g):
        dist.broadcast(t, src=0)

    cu = torch.tensor([0, T_global], device=device, dtype=torch.long)
    context = build_cp_context(cu, group=dist.group.WORLD)
    s, e = rank * T_local, (rank + 1) * T_local

    def slice_local():
        return (
            q_g[:, s:e].clone().detach().requires_grad_(True),
            k_g[:, s:e].clone().detach().requires_grad_(True),
            v_g[:, s:e].clone().detach().requires_grad_(True),
            g_g[:, s:e].clone().detach().requires_grad_(True),
            beta_g[:, s:e].clone().detach().requires_grad_(True),
        )

    do_local = do_g[:, s:e].clone()

    def one_iter(measure):
        ql, kl, vl, gl, bl = slice_local()
        fwd_ms = bwd_ms = 0.0
        if measure:
            st = torch.cuda.Event(enable_timing=True)
            mid = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
        o, _ = chunk_gated_delta_rule(q=ql, k=kl, v=vl, g=gl, beta=bl, cp_context=context)
        if measure:
            mid.record()
        o.backward(do_local)
        if measure:
            en.record()
            torch.cuda.synchronize()
            fwd_ms = st.elapsed_time(mid)
            bwd_ms = mid.elapsed_time(en)
        return fwd_ms, bwd_ms

    for _ in range(warmup):
        one_iter(False)
    dist.barrier()
    torch.cuda.synchronize()

    fwd_samples, bwd_samples = [], []
    for _ in range(repeats):
        f, b = one_iter(True)
        fwd_samples.append(f)
        bwd_samples.append(b)
    dist.barrier()

    fwd_med = _percentile(sorted(fwd_samples), 50)
    bwd_med = _percentile(sorted(bwd_samples), 50)
    # gather medians to rank 0 (max latency across ranks is the real cost)
    gather_f = [torch.zeros(1, device=device) for _ in range(world_size)]
    gather_b = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gather_f, torch.tensor([fwd_med], device=device))
    dist.all_gather(gather_b, torch.tensor([bwd_med], device=device))
    if rank == 0:
        per_rank = []
        for r in range(world_size):
            per_rank.append({"rank": r, "fwd_ms": gather_f[r].item(), "bwd_ms": gather_b[r].item()})
        payload = {
            "mode": "cp-e2e", "world_size": world_size, "T_local": T_local,
            "T_global": T_global, "H": H, "D": D, "dtype": dtype_str,
            "per_rank": per_rank,
            "max_fwd_ms": max(p["fwd_ms"] for p in per_rank),
            "max_bwd_ms": max(p["bwd_ms"] for p in per_rank),
            "middle_rank": world_size // 2,
        }
        with open(ret_path, "w") as fh:
            json.dump(payload, fh)
    dist.barrier()
    dist.destroy_process_group()


def run_cp_e2e_mode(args, dtype):
    import tempfile
    import torch.multiprocessing as mp
    results = []
    world_size = args.cp
    if torch.cuda.device_count() < world_size:
        print(f"[cp-e2e] SKIP: need {world_size} GPUs, have {torch.cuda.device_count()}")
        return results
    for H in args.heads:
        for T in args.seq_lens:
            ret_path = os.path.join(tempfile.gettempdir(), f"cp_e2e_{H}_{T}_{os.getpid()}.json")
            mp.start_processes(
                _cp_worker,
                args=(world_size, T, H, str(dtype).replace("torch.", ""),
                      args.warmup, args.repeats, args.master_port, ret_path),
                nprocs=world_size, join=True, start_method="spawn",
            )
            if os.path.exists(ret_path):
                with open(ret_path) as fh:
                    rec = json.load(fh)
                os.remove(ret_path)
                results.append(rec)
                mr = rec["middle_rank"]
                pr = rec["per_rank"][mr]
                print(f"[cp-e2e] CP{world_size} H={H:3d} T_local={T:6d}  "
                      f"mid-rank{mr} fwd={pr['fwd_ms']:.3f}ms bwd={pr['bwd_ms']:.3f}ms  "
                      f"max_fwd={rec['max_fwd_ms']:.3f} max_bwd={rec['max_bwd_ms']:.3f}")
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_str_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def print_variant_ratios(results):
    """Print split/merged latency ratio per shape (>1 means merged is faster)."""
    by_shape = {}
    for rec in results:
        by_shape.setdefault((rec["phase"], rec["H"], rec["T"]), {})[rec["variant"]] = rec["stats"]
    for (phase, H, T), variants in sorted(by_shape.items()):
        merged, split = variants.get("merged"), variants.get("split")
        if merged and split:
            print(f"[ratio ] phase={phase:3s} H={H:3d} T={T:6d}  "
                  f"split/merged={split['median'] / merged['median']:.3f}x")


def main():
    p = argparse.ArgumentParser(description="GDN CP pre-process micro-benchmark")
    p.add_argument("--heads", type=parse_int_list, default=[8, 16, 32, 48, 64, 128])
    p.add_argument("--seq-lens", type=parse_int_list, default=[4096, 8192, 16384],
                   help="LOCAL sequence length T per rank")
    p.add_argument("--phase", choices=["fwd", "bwd", "both"], default="both")
    p.add_argument("--variants", type=parse_str_list, default=["merged"],
                   help="Comma list of merged,split (kernel mode only).")
    p.add_argument("--mode", choices=["kernel", "cp-e2e"], default="kernel")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--single-launch", action="store_true",
                   help="Do exactly one measured launch (for ACU harness).")
    p.add_argument("--cp", type=int, default=4, help="world_size for cp-e2e mode")
    p.add_argument("--master-port", type=int, default=29561)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--pin-num-warps", type=int, default=None)
    p.add_argument("--pin-num-stages", type=int, default=2)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--tag", type=str, default="", help="free-form label recorded in output")
    args = p.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    dtype = getattr(torch, args.dtype)

    if not torch.cuda.is_available():
        print("CUDA/PPU device not available", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda:0")

    import fla
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "mode": args.mode, "phase": args.phase, "variants": args.variants,
        "heads": args.heads, "seq_lens": args.seq_lens,
        "warmup": args.warmup, "repeats": args.repeats,
        "single_launch": args.single_launch, "dtype": args.dtype,
        "pin_num_warps": args.pin_num_warps, "pin_num_stages": args.pin_num_stages,
        "seed": SEED,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__, "triton": triton.__version__,
        "fla_file": fla.__file__,
        "python": platform.python_version(),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "<default>"),
        "fla_ppu_disable_aiu": os.environ.get("FLA_PPU_DISABLE_AIU", "<unset>"),
    }
    print("=" * 78)
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("=" * 78)

    if args.mode == "kernel":
        results = run_kernel_mode(args, device, dtype)
    else:
        results = run_cp_e2e_mode(args, dtype)

    if args.mode == "kernel":
        print_variant_ratios(results)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
