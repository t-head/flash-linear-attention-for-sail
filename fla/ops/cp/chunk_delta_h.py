# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.utils.op import exp2
from fla.utils import IS_PPU_810E, IS_PPU_M890, autotune_cache_kwargs, check_shared_mem

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


def _prune_fwd_merged_configs(configs, named_args, **kwargs):
    """Drop BLOCK_SIZE=128 candidates when K or V < 128.

    On PPU the masked-but-out-of-bounds pointer arithmetic in the stage-2
    m-part (col = arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE with
    BLOCK_SIZE > K) triggers hggcErrorLaunchFailure.  Pruning here avoids
    the crash during autotune warmup while keeping BS=128 for K=V=128.
    """
    K = kwargs.get('K', 0)
    V = kwargs.get('V', 0)
    if K < 128 or V < 128:
        return [c for c in configs if c.kwargs.get('BLOCK_SIZE', 32) <= 64]
    return configs


# Split+Decompose head lists are per-board: the merged/split crossover flips between the
# two PPU parts, so one list cannot be optimal for both.
#
# ZW-M890P, where the narrow list was originally tuned: merged stays more than 4x ahead
# of split on the heads that lose out, leaving only H=32/128 with a >5% gain (H=8/48
# regress, H=16/64 merged faster after the stage2 decomposition).
# ZW810E: the crossover is a head count, not a work count. Measured with
# benchmarks/cp/benchmark_gdn_preprocess.py, split wins the bwd pre-process by 1%-16% from
# H=16 up, while H=4 loses 15%-30% and H=8 loses 1%-15%: below 16 heads the stage grid is
# too narrow to amortize the second launch, however long the boundary segment is (512 to
# 16384 tokens, dense and varlen).
#
# Neither list transfers to another part: the empty set keeps every non-PPU device on the
# merged kernel it was tuned for, since nothing here measures split on those boards.
_810E_SPLIT_HEADS = frozenset({16, 32, 64, 128})
_M890_SPLIT_HEADS = frozenset({32, 128})
_SPLIT_HEADS = (_810E_SPLIT_HEADS if IS_PPU_810E else
                _M890_SPLIT_HEADS if IS_PPU_M890 else
                frozenset())

# Split pays for a second launch with lower register pressure, so it also needs enough work
# to amortize that launch: the floor is `HV * boundary_seg` head-tokens. Measured end to end
# in CP2 at H=16/32, split loses up to 8% at 8k-16k head-tokens, breaks even at 32k and wins
# 3%-5% above it, while the kernel alone crosses over as low as 8k: the launch is what eats
# the gain. The end-to-end number is what users see, so it sets the floor.
_SPLIT_MIN_HEAD_TOKENS = 32768

# FWD split path: empty on both boards, the merged kernel is faster for every
# benchmarked head, so all FWD go merged.
_FWD_SPLIT_HEADS = frozenset()


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'EVEN_T': lambda args: (
        args['cu_seqlens'] is None
        and args['T'] % args['BT'] == 0
        and args['H'] in (8, 16, 32, 48, 64, 128)
    ),
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=num_warps, num_stages=num_stages)
        # num_warps=8 added (candidate 1): baseline ACU shows this kernel is
        # latency-bound on global-load memory_dependency with only ~5-10%
        # warps_active; extra warps per CTA hide that latency. Configs that do
        # not fit registers/shared memory are pruned by the autotuner, and the
        # original [2, 4] configs remain, so the selected winner cannot regress.
        for bs in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ] + [
        # D5: BLOCK_SIZE=128 halves grid columns (V/128+K/128 vs V/64+K/64),
        # reducing tail-wave overhead. H=128 cold-cache A/B: 1.074x CI95=[1.067,1.082].
        # nw=8 is the only competitive warp count; s={2,3} bracket the winner.
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=ns)
        for ns in [2, 3]
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    prune_configs_by={'early_config_prune': _prune_fwd_merged_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_merged(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
    EVEN_T: tl.constexpr = False,
):
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    if MULTI_SEQS:
        i_n = tl.program_id(2)
        # Offset hm for this subseq: hm[i_n, h, k, v+k]
        hm += i_n * HV * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles h (V part) or m (K part)
    # i_col is in range [0, cdiv(V + K, BLOCK_SIZE))
    # Columns [0, V) are for h, columns [V, V+K) are for m
    is_h_part = i_col * BLOCK_SIZE < V
    # For DPLR (USE_BG), w and bg share the same head dim H as k/ag.
    # For GDN/KDA, w has head dim HV (same as v).
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_h_part:
        # ====== Stage 1: Compute h (K x V) ======
        v += ((bos * HV + i_h) * V).to(tl.int64)
        if USE_BG:
            # DPLR keeps u and v as separate tensors; both need the per-head offset.
            # For GDN/KDA, u is aliased to v at the Python wrapper level.
            u += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        # Initialize h accumulators
        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        # Main recurrence for h
        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = tl.full([BT], 1, dtype=tl.int1) if EVEN_T else o_t < T
            # Compute decayed v
            p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_v = v + o_t[:, None] * stride_v + o_vb[None, :]
            if USE_BG:
                # DPLR mode: v2 = w @ h + u, h += kg^T @ v + bg^T @ v2
                b_v_orig = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0)
                p_u = u + o_t[:, None] * stride_v + o_vb[None, :]
                b_v = b_v_decay + tl.load(p_u, mask=m_t[:, None] & m_vb[None, :], other=0.0)
            else:
                # GDN/KDA mode: v_new = v - w @ h
                b_v = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v_decay

            last_idx = (i_t + 1) * BT - 1 if EVEN_T else min((i_t + 1) * BT, T) - 1

            # Apply g decay
            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            # Apply gk decay
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                b_h1 *= exp2(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    b_h2 *= exp2(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    b_h3 *= exp2(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    b_h4 *= exp2(b_gk_last4)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            # Update h
            p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            if USE_BG:
                # DPLR mode: h += kg^T @ v + bg^T @ v2
                p_bg = bg + o_k1[:, None] + o_t[None, :] * stride_k
                b_bg = tl.load(p_bg, mask=m_k1[:, None] & m_t[None, :], other=0.0)
                b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k2[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k3[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    p_bg = bg + o_k4[:, None] + o_t[None, :] * stride_k
                    b_bg = tl.load(p_bg, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            else:
                # GDN/KDA mode: h += k^T @ v_new
                b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                    b_h2 += tl.dot(b_k, b_v)
                if K > 128:
                    p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                    b_h3 += tl.dot(b_k, b_v)
                if K > 192:
                    p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                    b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                    b_h4 += tl.dot(b_k, b_v)

        # Store h results
        stride_hm_kv = K + V
        p_h1 = hm + o_k1[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_h2 = hm + o_k2[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_h3 = hm + o_k3[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_h4 = hm + o_k4[:, None] * stride_hm_kv + o_vb[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        # ====== Stage 2: Compute m (K x K) ======
        # i_col is for m part, map to K dimension
        # m starts at column V, so offset = i_col * BLOCK_SIZE - V
        # Use tl.cdiv to correctly compute the number of blocks for V dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design:
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size (like BK2=32 in stage2)
        # Each block computes a (BK1, BLOCK_SIZE) sub-matrix of m
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = tl.full([BT], 1, dtype=tl.int1) if EVEN_T else o_t < T
            # Load k and w with full BK1 rows
            if USE_BG:
                # DPLR mode: use bg for transition matrix
                # bg was already offset at the beginning of the kernel
                p_k = bg + o_t[:, None] * stride_k + row[None, :]
            else:
                p_k = k + o_t[:, None] * stride_k + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = (i_t + 1) * BT - 1 if EVEN_T else min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)

            if USE_G:
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)
            b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
            if USE_BG:
                b_m_i = b_diag + b_kw
            else:
                b_m_i = b_diag - b_kw
            # Keep the m chain in fp32: casting m to bf16 every chunk compounds along the chain.
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        # Store m result
        stride_hm_kv = K + V
        p_m = hm + V + row[:, None] * stride_hm_kv + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def merge_fwd_bwd_kernel(
    h,                   # [HV, K, V] or [num_non_first, HV, K, V] for intracard (or [V, K] when transposed)
    ag_hm,               # [HV, K, K+V] or [S_split, HV, K, K+V] for intracard (always [K, V+K])
    pre_or_post_num_ranks,  # num_ranks for CP, NUM_SPLIT_SEQS for intracard
    rank,                # rank for CP, not used for intracard
    seq_offsets,         # None for CP, [num_split_seqs+1] for intracard
    init_offsets,        # None for CP, [num_split_seqs+1] for intracard
    h0_seq_ids,          # None for CP, [num_split_seqs] for intracard
    h0,                  # None or [N_orig, HV, K, V] for intracard (or [V, K] when transposed)
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,                # True for FWD, False for BWD
    INTRACARD_MODE: tl.constexpr,          # True: intracard mode, False: CP mode
    NUM_SEQ_ENTRIES,         # num_split_seqs for intracard
    HAS_H0: tl.constexpr,                  # Heuristic: whether h0 is provided
    STATE_V_FIRST: tl.constexpr = False,  # When True, h0/h use [V, K] layout; ag_hm always [K, V+K]
):
    """
    Unified merge kernel for both CP and Intra-card modes.

    CP mode (INTRACARD_MODE=False):
        Grid: (V/BV, HV)
        Merges across ranks for context parallel.

    Intra-card mode (INTRACARD_MODE=True):
        Grid: (V/BV, NUM_SEQ_ENTRIES, HV)
        Merges across subseqs within card for intra-card context parallel.

    When STATE_V_FIRST=True, h0 and output h use [V, K] layout.
    ag_hm always uses [K, V+K] layout (from pre_scan).
    The recurrence h' = M @ h + he becomes h_T' = h_T @ M^T + he^T.
    """
    i_v = tl.program_id(0)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    if INTRACARD_MODE:
        i_seq = tl.program_id(1)
        i_h = tl.program_id(2)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        # Load offsets for this sequence
        ss_start = tl.load(seq_offsets + i_seq).to(tl.int32)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int32)
        init_base = tl.load(init_offsets + i_seq).to(tl.int32)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        # Initialize from h0 if provided
        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int32)
            if STATE_V_FIRST:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            else:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                b_h = tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        else:
            if STATE_V_FIRST:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        # Merge loop over subseqs
        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            # he and m are always in [K, V+K] layout from pre_scan
            p_he = ag_hm + base + o_k[:, None] * (V + K) + o_v[None, :]
            b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            p_m = ag_hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                # h_T' = h_T @ M^T + he^T
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m)) + tl.trans(b_he)
            else:
                b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)

            # Store for non-first subseqs
            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = HV * K * V
                if STATE_V_FIRST:
                    p_out = h + init_idx * stride_init + i_h * V * K + o_v[:, None] * K + o_k[None, :]
                    m_out = m_v[:, None] & m_k[None, :]
                else:
                    p_out = h + init_idx * stride_init + i_h * K * V + o_k[:, None] * V + o_v[None, :]
                    m_out = m_k[:, None] & m_v[None, :]
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=m_out)
    else:
        # CP mode
        i_h = tl.program_id(1)
        num_ranks = pre_or_post_num_ranks.to(tl.int32)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = HV * K * (K + V)
        if STATE_V_FIRST:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = ag_hm + cur_rank * stride + o_k[:, None] * (K + V) + o_v[None, :]
            b_ag_h = tl.load(p_ag_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
            p_ag_m = ag_hm + cur_rank * stride + V + o_k[:, None] * (K + V) + o_k[None, :]
            b_ag_m = tl.load(p_ag_m, mask=m_k[:, None] & m_k[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32)) + tl.trans(b_ag_h).to(tl.float32)
            else:
                b_h = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32)) + b_ag_h.to(tl.float32)
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    # Removing the tail mask helps both stages for these target head counts.
    'EVEN_T': lambda args: (
        args['cu_seqlens'] is None
        and args['T'] % args['BT'] == 0
        and args['H'] in (8, 16, 64)
    ),
    # H=32 and H=128 benefit only in stage 2 (candidates 19 and 18).
    # Stage 1 keeps the generic path for both heads.
    'STAGE2_EVEN_T': lambda args: (
        args['cu_seqlens'] is None
        and args['T'] % args['BT'] == 0
        and args['H'] in (32, 128)
    ),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # num_warps=8 added (candidate 1): same latency-bound / low-occupancy
        # rationale as the fwd kernel. Original [2, 4] configs are retained and
        # non-fitting configs are pruned, so the winner cannot regress.
        for num_warps in [2, 4, 8]
        for num_stages in ([4, 3, 2] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_merged(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EVEN_T: tl.constexpr,
    STAGE2_EVEN_T: tl.constexpr,
    G_HEAD_MAJOR: tl.constexpr = False,
):
    """
    Merged backward kernel that computes both dh (K x V) and dm (K x K) in a single kernel.

    Similar to pre_process_fwd_kernel_merged, this kernel uses a unified grid where:
    - Columns [0, V) are for computing dh (stage 1)
    - Columns [V, V+K) are for computing dm (stage 2)

    When G_HEAD_MAJOR=True (POC), g is expected in [HV, T] layout (contiguous in T)
    instead of the default [total_T, HV] layout.  Token stride becomes 1 vs HV.
    """
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles dh (V part) or dm (K part)
    is_dh_part = i_col * BLOCK_SIZE < V

    # Calculate offsets
    # For DPLR (USE_BG), w shares the same head dim H as q/k/ag.
    # For GDN/KDA, w has head dim HV (same as do/dv).
    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        if G_HEAD_MAJOR:
            # POC: g is [HV, T_local], contiguous in T.  Offset to head i_h.
            g += (i_h * T).to(tl.int64)
        else:
            g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_dh_part:
        # ====== Stage 1: Compute dh (K x V) ======
        do += ((bos * HV + i_h) * V).to(tl.int64)
        dv += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        # Initialize dh accumulators
        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        # Main recurrence for dh (reverse order)
        for i_t in range(NT - 1, -1, -1):
            last_idx = (i_t + 1) * BT - 1 if EVEN_T else min((i_t + 1) * BT, T) - 1
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = tl.full([BT], 1, dtype=tl.int1) if EVEN_T else o_t < T

            if USE_G:
                if G_HEAD_MAJOR:
                    bg_last = tl.load(g + last_idx).to(tl.float32)
                    p_g = g + o_t
                else:
                    bg_last = tl.load(g + last_idx * HV).to(tl.float32)
                    p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)

            p_dv = dv + o_t[:, None] * stride_v + o_vb[None, :]
            p_do = do + o_t[:, None] * stride_v + o_vb[None, :]
            b_do = tl.load(p_do, mask=m_t[:, None] & m_vb[None, :], other=0.0)

            # Update dv
            p_k = k + o_t[:, None] * stride_qk + o_k1[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = k + o_t[:, None] * stride_qk + o_k2[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                if USE_GK:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = k + o_t[:, None] * stride_qk + o_k3[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                if USE_GK:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = k + o_t[:, None] * stride_qk + o_k4[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                if USE_GK:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, mask=m_t[:, None] & m_vb[None, :], other=0.0)

            # Update dh
            p_w = w + o_k1[:, None] + o_t[None, :] * stride_w
            p_q = q + o_k1[:, None] + o_t[None, :] * stride_qk
            b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh1 *= exp2(b_gk_last1[:, None])
            if USE_BG:
                # DPLR mode: dh += q^T @ do + w^T @ dv2 (no scale)
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
            else:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = q + o_k2[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k2[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                if USE_BG:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = q + o_k3[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k3[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                if USE_BG:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = q + o_k4[:, None] + o_t[None, :] * stride_qk
                p_w = w + o_k4[:, None] + o_t[None, :] * stride_w
                b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                if USE_BG:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        # Store dh results
        p_dh1 = dhm + o_k1[:, None] * (V + K) + o_vb[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
        if K > 64:
            p_dh2 = dhm + o_k2[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
        if K > 128:
            p_dh3 = dhm + o_k3[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
        if K > 192:
            p_dh4 = dhm + o_k4[:, None] * (V + K) + o_vb[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])
    else:
        # ====== Stage 2: Compute dm (K x K) ======
        # i_col is for dm part, map to K dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design for backward (FORWARD=False)
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            # Reverse order for backward
            i_t = NT - 1 - _i_t
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = tl.full([BT], 1, dtype=tl.int1) if EVEN_T or STAGE2_EVEN_T else o_t < T

            # Load k and w with full BK1 rows
            p_k = k + o_t[:, None] * stride_qk + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_w + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = (i_t + 1) * BT - 1 if EVEN_T or STAGE2_EVEN_T else min((i_t + 1) * BT, T) - 1

            if USE_G:
                if G_HEAD_MAJOR:
                    b_g_last = tl.load(g + last_idx).to(tl.float32)
                    p_g = g + o_t
                else:
                    b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                    p_g = g + o_t * HV
                b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)

            if USE_G:
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)
            # dM = (diag ∓ w^T @ k) @ dM
            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            if USE_BG:
                b_m_i = b_diag + b_kw
            else:
                b_m_i = b_diag - b_kw
            # Keep the m chain in fp32: casting m to bf16 every chunk compounds along the chain.
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        # Store dm result
        p_m = dhm + V + row[:, None] * (V + K) + col[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


# Split BWD Stage 1: dh computation (K x V portion of dhm).
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'EVEN_T': lambda args: (
        args['cu_seqlens'] is None
        and args['T'] % args['BT'] == 0
        and args['H'] in (8, 16, 64)
    ),
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=nw, num_stages=ns)
        for bs in [32, 64]
        for nw in [4, 8]
        for ns in ([4, 3, 2] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_stage1_dh(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EVEN_T: tl.constexpr,
    G_HEAD_MAJOR: tl.constexpr = False,
):
    """Stage 1 only: computes dh [K, V] portion of dhm."""
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        if G_HEAD_MAJOR:
            g += (i_h * T).to(tl.int64)
        else:
            g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    do += ((bos * HV + i_h) * V).to(tl.int64)
    dv += ((bos * HV + i_h) * V).to(tl.int64)
    stride_v = HV * V

    # D4-A: capture the pointer element type so that any `.to(in_dtype)` cast
    # on the dot inputs is stable. Without this, `b_q = b_q * b_g_exp` (fp32)
    # mutates `b_q.dtype` to fp32 and the downstream `b_q.to(b_q.dtype)` becomes
    # a no-op, forcing the compiler to emit TF32 MMA (m16n16k8) instead of
    # bf16 MMA (m16n16k16) at half the FLOP/instr. Mirrors the known FWD
    # stage2 pitfall for the same `.to(b_k.dtype)` after g-scaling pattern.
    in_dtype = w.dtype.element_ty

    # Initialize dh accumulators
    b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

    o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_vb = o_vb < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K

    for i_t in range(NT - 1, -1, -1):
        last_idx = (i_t + 1) * BT - 1 if EVEN_T else min((i_t + 1) * BT, T) - 1
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = tl.full([BT], 1, dtype=tl.int1) if EVEN_T else o_t < T

        if USE_G:
            if G_HEAD_MAJOR:
                bg_last = tl.load(g + last_idx).to(tl.float32)
                p_g = g + o_t
            else:
                bg_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)

        p_dv = dv + o_t[:, None] * stride_v + o_vb[None, :]
        p_do = do + o_t[:, None] * stride_v + o_vb[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_vb[None, :], other=0.0)

        # k loads for dv: [BT, 64] row-major blocks.
        p_k = k + o_t[:, None] * stride_qk + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if USE_GK:
            p_gk_last = gk + last_idx * HV * K
            b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = k + o_t[:, None] * stride_qk + o_k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if USE_GK:
                b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = k + o_t[:, None] * stride_qk + o_k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if USE_GK:
                b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = k + o_t[:, None] * stride_qk + o_k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if USE_GK:
                b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, mask=m_t[:, None] & m_vb[None, :], other=0.0)

        # Update dh: w and q are in [64, BT] column-major layout.
        p_w = w + o_k1[:, None] + o_t[None, :] * stride_w
        p_q = q + o_k1[:, None] + o_t[None, :] * stride_qk
        b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            b_dh1 *= exp2(b_gk_last1[:, None])
        if USE_BG:
            b_dh1 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) + tl.dot(b_w, b_dv.to(in_dtype))
        else:
            b_dh1 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) * scale - tl.dot(b_w, b_dv.to(in_dtype))

        if K > 64:
            p_q = q + o_k2[:, None] + o_t[None, :] * stride_qk
            p_w = w + o_k2[:, None] + o_t[None, :] * stride_w
            b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh2 *= exp2(b_gk_last2[:, None])
            if USE_BG:
                b_dh2 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) + tl.dot(b_w, b_dv.to(in_dtype))
            else:
                b_dh2 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) * scale - tl.dot(b_w, b_dv.to(in_dtype))

        if K > 128:
            p_q = q + o_k3[:, None] + o_t[None, :] * stride_qk
            p_w = w + o_k3[:, None] + o_t[None, :] * stride_w
            b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh3 *= exp2(b_gk_last3[:, None])
            if USE_BG:
                b_dh3 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) + tl.dot(b_w, b_dv.to(in_dtype))
            else:
                b_dh3 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) * scale - tl.dot(b_w, b_dv.to(in_dtype))

        if K > 192:
            p_q = q + o_k4[:, None] + o_t[None, :] * stride_qk
            p_w = w + o_k4[:, None] + o_t[None, :] * stride_w
            b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh4 *= exp2(b_gk_last4[:, None])
            if USE_BG:
                b_dh4 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) + tl.dot(b_w, b_dv.to(in_dtype))
            else:
                b_dh4 += tl.dot(b_q.to(in_dtype), b_do.to(in_dtype)) * scale - tl.dot(b_w, b_dv.to(in_dtype))

    # Store dh results
    p_dh1 = dhm + o_k1[:, None] * (V + K) + o_vb[None, :]
    tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
    if K > 64:
        p_dh2 = dhm + o_k2[:, None] * (V + K) + o_vb[None, :]
        tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
    if K > 128:
        p_dh3 = dhm + o_k3[:, None] * (V + K) + o_vb[None, :]
        tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
    if K > 192:
        p_dh4 = dhm + o_k4[:, None] * (V + K) + o_vb[None, :]
        tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])


# Split BWD Stage 2: dm computation (K x K portion of dhm) with matrix decomposition.
# Decomposes M = decay*M ∓ w^T @ (k @ M), halving FLOPs vs the merged kernel.
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'EVEN_T': lambda args: (
        args['cu_seqlens'] is None
        and args['T'] % args['BT'] == 0
        and args['H'] in (8, 16, 32, 64, 128)
    ),
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=nw, num_stages=ns)
        for bs in [32, 64]
        for nw in [4, 8]
        for ns in ([3, 2, 1] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_stage2_dm_v2(
    k,
    w,
    g,
    gk,
    dhm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    EVEN_T: tl.constexpr,
    G_HEAD_MAJOR: tl.constexpr = False,
):
    """Stage 2 only: computes dm [K, K] portion of dhm via the fp32 m chain."""
    i_k_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        if G_HEAD_MAJOR:
            g += (i_h * T).to(tl.int64)
        else:
            g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    row = tl.arange(0, BK1)
    col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

    # Initialize as identity: M_0 = I
    b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

    for _i_t in range(NT):
        # Reverse order for backward
        i_t = NT - 1 - _i_t
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = tl.full([BT], 1, dtype=tl.int1) if EVEN_T else o_t < T

        # Load k and w with full BK1 rows.
        p_k = k + o_t[:, None] * stride_qk + row[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
        p_w = w + o_t[:, None] * stride_w + row[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

        last_idx = (i_t + 1) * BT - 1 if EVEN_T else min((i_t + 1) * BT, T) - 1

        if USE_G:
            if G_HEAD_MAJOR:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                p_g = g + o_t
            else:
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = g + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
        elif USE_GK:
            b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
            b_gk_last = exp2(b_gk_last)

        if USE_G:
            b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
        elif USE_GK:
            b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
        else:
            b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)
        # dM = (diag ∓ w^T @ k) @ dM
        b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
        if USE_BG:
            b_m_i = b_diag + b_kw
        else:
            b_m_i = b_diag - b_kw
        # Keep the m chain in fp32: casting m to bf16 every chunk compounds along the chain.
        b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

    # Store dm result
    p_m = dhm + V + row[:, None] * (V + K) + col[None, :]
    tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


# ---------------------------------------------------------------------------
# Split FWD Stage 1: h computation (K x V portion of hm).
# Standalone extraction of the is_h_part branch from merged FWD kernel.
# ---------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=nw, num_stages=ns)
        for bs in [32, 64]
        for nw in [4, 8]
        for ns in ([4, 3, 2] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_stage1_h(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Stage 1 only: computes h [K, V] portion of hm (forward recurrence)."""
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    hm += i_h * K * (K + V)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    v += ((bos * HV + i_h) * V).to(tl.int64)
    if USE_BG:
        u += ((bos * HV + i_h) * V).to(tl.int64)
    stride_v = HV * V

    # Initialize h accumulators
    b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

    o_vb = i_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m_vb = o_vb < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # Compute decayed v
        p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_v = v + o_t[:, None] * stride_v + o_vb[None, :]
        if USE_BG:
            b_v_orig = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0)
            p_u = u + o_t[:, None] * stride_v + o_vb[None, :]
            b_v = b_v_decay + tl.load(p_u, mask=m_t[:, None] & m_vb[None, :], other=0.0)
        else:
            b_v = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v_decay

        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_G:
            b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
            p_g = g + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            p_gk_last = gk + last_idx * HV * K
            b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if USE_BG:
            p_bg = bg + o_k1[:, None] + o_t[None, :] * stride_k
            b_bg = tl.load(p_bg, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            if K > 64:
                p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                p_bg = bg + o_k2[:, None] + o_t[None, :] * stride_k
                b_bg = tl.load(p_bg, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_h2 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            if K > 128:
                p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                p_bg = bg + o_k3[:, None] + o_t[None, :] * stride_k
                b_bg = tl.load(p_bg, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_h3 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            if K > 192:
                p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                p_bg = bg + o_k4[:, None] + o_t[None, :] * stride_k
                b_bg = tl.load(p_bg, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_h4 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
        else:
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_h4 += tl.dot(b_k, b_v)

    # Store h results
    stride_hm_kv = K + V
    p_h1 = hm + o_k1[:, None] * stride_hm_kv + o_vb[None, :]
    tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_k1[:, None] & m_vb[None, :])
    if K > 64:
        p_h2 = hm + o_k2[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :])
    if K > 128:
        p_h3 = hm + o_k3[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :])
    if K > 192:
        p_h4 = hm + o_k4[:, None] * stride_hm_kv + o_vb[None, :]
        tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :])


# ---------------------------------------------------------------------------
# Split FWD Stage 2: m computation (K x K portion of hm) with matrix decomposition.
# Decomposes M = decay*M ∓ k^T @ (w @ M), halving FLOPs vs the merged kernel.
# Uses transposed dot (Variant A) for bank-conflict-free shared memory access.
# ---------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=nw, num_stages=ns)
        for bs in [32, 64]
        for nw in [4, 8]
        for ns in ([4, 3, 2] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_stage2_m(
    k,
    w,
    g,
    gk,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Stage 2 only: computes m [K, K] portion of hm via the fp32 m chain (forward)."""
    i_k_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # k_param is bg when USE_BG (set by wrapper), always uses H-stride
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    hm += i_h * K * (V + K)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    row = tl.arange(0, BK1)
    col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

    # Initialize as identity: M_0 = I
    b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T

        # Load k (or bg) and w
        p_k = k + o_t[:, None] * stride_k + row[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
        p_w = w + o_t[:, None] * stride_w + row[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_G:
            b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
            p_g = g + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
        elif USE_GK:
            b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
            b_gk_last = exp2(b_gk_last)
            b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
        else:
            b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

        b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
        if USE_BG:
            b_m_i = b_diag + b_kw
        else:
            b_m_i = b_diag - b_kw
        # Keep the m chain in fp32: casting m to bf16 every chunk compounds along the chain.
        b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

    # Store m result
    stride_hm_kv = K + V
    p_m = hm + V + row[:, None] * stride_hm_kv + col[None, :]
    tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=(row < K)[:, None] & (col < K)[None, :])


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    rank = dist.get_rank(group=context.group)

    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    hm = k.new_zeros(HV, K, (V + K), dtype=torch.float32)
    if state_v_first:
        initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)
    if not context.is_last_rank:
        # Split+Decompose: stage1(h) + stage2(m) where it beats the merged kernel.
        # g=None: fall back to merged (fp32 dot) to avoid bf16 rounding in m-chain.
        # The stages take cu_seqlens[-2:] only, mirroring the merged kernel's slicing: the
        # state carried to the next rank is the one at this rank's end, which is where the
        # last sequence ends. Both stages index sequence 0, so the slice must stay single.
        if H in _FWD_SPLIT_HEADS and g is not None:
            v_param = u if v is None else v
            # Stage 1: compute h (K x V portion of hm)
            def grid_s1(meta): return (triton.cdiv(V, meta['BLOCK_SIZE']), HV)
            pre_process_fwd_stage1_h[grid_s1](
                k=k,
                v=v_param,
                w=w,
                g=g,
                gk=gk,
                bg=bg,
                u=u,
                hm=hm,
                cu_seqlens=cu_seqlens[-2:] if cu_seqlens is not None else None,
                T=T,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                USE_BG=bg is not None,
            )
            # Stage 2: compute m (K x K portion of hm) with decomposition
            k_param = k if bg is None else bg
            def grid_s2(meta): return (triton.cdiv(K, meta['BLOCK_SIZE']), HV)
            pre_process_fwd_stage2_m[grid_s2](
                k=k_param,
                w=w,
                g=g,
                gk=gk,
                hm=hm,
                cu_seqlens=cu_seqlens[-2:] if cu_seqlens is not None else None,
                T=T,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK1=BK,
                USE_BG=bg is not None,
            )
        else:
            def grid(meta): return (triton.cdiv(V, meta['BLOCK_SIZE']) + triton.cdiv(K, meta['BLOCK_SIZE']), HV)
            # For DPLR, v provides the original v for computing h contributions,
            # while u remains the WY-processed values (A_ab @ A_ak @ v) for v_new = w @ h + u.
            pre_process_fwd_kernel_merged[grid](
                k=k,
                v=u if v is None else v,
                w=w,
                g=g,
                gk=gk,
                bg=bg,
                u=u,
                hm=hm,
                cu_seqlens=cu_seqlens[-2:] if cu_seqlens is not None else None,
                T=T,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK1=BK,
                MULTI_SEQS=False,
            )
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    if not context.is_first_rank:
        def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel[grid](
            h=initial_state[0],
            ag_hm=ag_hm,
            pre_or_post_num_ranks=context.pre_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            FORWARD=True,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
        )
    return initial_state


def _bwd_split_worth_it(HV: int, context: FLACPContext) -> bool:
    """Whether the split backward pre-process is the faster choice for this rank.

    The stages parallelize over value heads, so `HV` times the length of this rank's first
    local sequence (all `dht` covers) is the work available to amortize the second launch.
    The context's CPU copy of `cu_seqlens` is read so the gate stays off the device sync path.
    """
    if _SPLIT_HEADS is not None and HV not in _SPLIT_HEADS:
        return False
    cu_seqlens_cpu = context.cu_seqlens_cpu
    if cu_seqlens_cpu is None:
        return True
    return HV * int(cu_seqlens_cpu[1] - cu_seqlens_cpu[0]) >= _SPLIT_MIN_HEAD_TOKENS


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    chunk_size: int = 64,
    _poc_g_head_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    rank = dist.get_rank(context.group)

    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    # POC: head-major g layout —— transpose g from [total_T, HV] to [HV, T_local]
    g_kernel = g
    g_head_major = False
    if _poc_g_head_major and g is not None and not context.is_first_rank:
        if cu_seqlens is not None:
            bos_py = cu_seqlens[0].item()
            eos_py = cu_seqlens[1].item()
        else:
            bos_py, eos_py = 0, T
        g_flat = g.reshape(-1, HV) if g.dim() > 2 else g
        g_seq = g_flat[bos_py:eos_py]        # [T_local, HV]
        g_kernel = g_seq.t().contiguous()     # [HV, T_local], contiguous in T
        g_head_major = True

    if not context.is_first_rank:
        # Split+Decompose: stage1(dh) + stage2(dm) where it beats the merged kernel.
        # g=None: fall back to merged (fp32 dot) to avoid bf16 rounding in m-chain.
        # The stages take cu_seqlens[:2] only, mirroring the merged kernel's slicing: the
        # dht carried back to the previous rank is the one at this rank's start, which is
        # where the first sequence starts. Both stages index sequence 0, so the slice must
        # stay single.
        if _bwd_split_worth_it(HV, context) and g is not None:
            k_param = k if bg is None else bg
            # Stage 1: compute dh (K x V portion of dhm)
            def grid_s1(meta): return (triton.cdiv(V, meta['BLOCK_SIZE']), HV)
            pre_process_bwd_stage1_dh[grid_s1](
                q=q,
                k=k_param,
                w=w,
                g=g_kernel,
                gk=gk,
                do=do,
                dhm=dhm,
                dv=dv,
                cu_seqlens=cu_seqlens[:2] if cu_seqlens is not None else None,
                scale=scale,
                T=T,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                USE_BG=bg is not None,
                G_HEAD_MAJOR=g_head_major,
            )
            # Stage 2: compute dm (K x K portion of dhm) with decomposition
            def grid_s2(meta): return (triton.cdiv(K, meta['BLOCK_SIZE']), HV)
            pre_process_bwd_stage2_dm_v2[grid_s2](
                k=k_param,
                w=w,
                g=g_kernel,
                gk=gk,
                dhm=dhm,
                cu_seqlens=cu_seqlens[:2] if cu_seqlens is not None else None,
                T=T,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK1=BK,
                USE_BG=bg is not None,
                G_HEAD_MAJOR=g_head_major,
            )
        else:
            BLOCK_SIZE = 32 if K <= 64 else 64
            grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
            pre_process_bwd_kernel_merged[grid](
                q=q,
                k=k if bg is None else bg,
                w=w,
                g=g_kernel,
                gk=gk,
                do=do,
                dhm=dhm,
                dv=dv,
                cu_seqlens=cu_seqlens[:2] if cu_seqlens is not None else None,
                scale=scale,
                T=T,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BT=BT,
                BK1=BK,
                BLOCK_SIZE=BLOCK_SIZE,
                USE_BG=bg is not None,
                G_HEAD_MAJOR=g_head_major,
            )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    if not context.is_last_rank:
        def grid(meta): return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel[grid](
            h=dht[-1],
            ag_hm=ag_dhm,
            pre_or_post_num_ranks=context.post_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
        )

    # initial_state is None in the CP mode
    # We only need to compute dht of current rank and pass it to the backward kernel
    return dht, None


def compress_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    # Here must use clone op or the full tensor will be saved for backward
    return h0[:1].clone()


def expand_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
    B = len(context.cu_seqlens) - 1
    expand_h0 = h0.new_zeros(B, *h0.shape[1:])
    expand_h0[:1] = h0
    return expand_h0
