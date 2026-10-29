# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_PPU,
    autotune_cache_kwargs,
    check_shared_mem,
)

# Curated PPU set: the full 24-config space costs ~6x first-call tuning for no measured
# win, and these five cover every winner seen (K in {64, 128, 256}, V in {64, 128, 256}).
_HO_PPU_CONFIGS = [
    ({'BV': 32}, 4, 2),
    ({'BV': 32}, 8, 2),
    ({'BV': 64}, 8, 2),
    ({'BV': 64}, 16, 2),
    ({'BV': 128}, 16, 2),
]
_HO_GENERIC_CONFIGS = [
    ({'BV': BV}, num_warps, num_stages)
    for num_warps in [2, 4]
    for num_stages in ([1, 2] if check_shared_mem('ampere') else [2, 1])
    for BV in ([32, 64] if check_shared_mem('ada') else [32])
]

# Same AIU staging policy as the standalone fwd_h kernel: the repeatedly-read q/k/w
# blocks go through the PPU async bulk copy when it is available.
_IS_AIU_ENABLED = IS_PPU and os.environ.get('FLA_PPU_DISABLE_AIU', '0') != '1'
USE_AIU = tl.constexpr(_IS_AIU_ENABLED)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config(kwargs, num_warps=num_warps, num_stages=num_stages)
        for kwargs, num_warps, num_stages in (_HO_PPU_CONFIGS if IS_PPU else _HO_GENERIC_CONFIGS)
    ],
    # T is in the key: the kernel walks chunks serially, so the best config depends on the
    # per-sequence chunk count.
    key=['H', 'HV', 'K', 'V', 'BT', 'T'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_o(
    q,
    k,
    v,
    w,
    g,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_AIU: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)

    # State is kept register-resident across the whole sequence: h and v_new never reach
    # HBM, which removes both the store/load of h and the store/load of v_new.
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v += (bos * HV + i_h).to(tl.int64) * V
    o += (bos * HV + i_h).to(tl.int64) * V

    DT_Q = q.dtype.element_ty
    DT_K = k.dtype.element_ty
    DT_W = w.dtype.element_ty

    if USE_INITIAL_STATE:
        h0 += i_nh * K*V
    if STORE_FINAL_STATE:
        ht += i_nh * K*V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K

    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_k1[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_k2[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_k3[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_k4[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_ov = m_t[:, None] & m_v[None, :]

        if USE_G:
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)

        # Interleaved tile walk: for every 64-wide K tile, load k/q/w once and issue the
        # three dots it feeds (o, intra-chunk A, and the w@h part of v_new). The k tiles
        # stay live until the state update at the end of the chunk.
        if USE_AIU and K % 64 == 0:
            b_k1 = tl.aiu_load(k, (0, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
            b_q = tl.aiu_load(q, (i_t * BT, 0), (BT, 64), (T, H*K), DT_Q, (1, 0))
            b_w = tl.aiu_load(w, (i_t * BT, 0), (BT, 64), (T, HV*K), DT_W, (1, 0))
        else:
            b_k1 = tl.load(k + o_k1[:, None] + o_t[None, :] * (H*K), mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_q = tl.load(q + o_t[:, None] * (H*K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k1[None, :], mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_o = tl.dot(b_q, b_h1.to(DT_Q))
        b_A = tl.dot(b_q, b_k1)
        b_v = tl.dot(b_w, b_h1.to(DT_W))
        if K > 64:
            if USE_AIU and K % 64 == 0:
                b_k2 = tl.aiu_load(k, (64, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
                b_q = tl.aiu_load(q, (i_t * BT, 64), (BT, 64), (T, H*K), DT_Q, (1, 0))
                b_w = tl.aiu_load(w, (i_t * BT, 64), (BT, 64), (T, HV*K), DT_W, (1, 0))
            else:
                b_k2 = tl.load(k + o_k2[:, None] + o_t[None, :] * (H*K), mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_q = tl.load(q + o_t[:, None] * (H*K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k2[None, :], mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_o += tl.dot(b_q, b_h2.to(DT_Q))
            b_A += tl.dot(b_q, b_k2)
            b_v += tl.dot(b_w, b_h2.to(DT_W))
        if K > 128:
            if USE_AIU and K % 64 == 0:
                b_k3 = tl.aiu_load(k, (128, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
                b_q = tl.aiu_load(q, (i_t * BT, 128), (BT, 64), (T, H*K), DT_Q, (1, 0))
                b_w = tl.aiu_load(w, (i_t * BT, 128), (BT, 64), (T, HV*K), DT_W, (1, 0))
            else:
                b_k3 = tl.load(k + o_k3[:, None] + o_t[None, :] * (H*K), mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_q = tl.load(q + o_t[:, None] * (H*K) + o_k3[None, :], mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k3[None, :], mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_o += tl.dot(b_q, b_h3.to(DT_Q))
            b_A += tl.dot(b_q, b_k3)
            b_v += tl.dot(b_w, b_h3.to(DT_W))
        if K > 192:
            if USE_AIU and K % 64 == 0:
                b_k4 = tl.aiu_load(k, (192, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
                b_q = tl.aiu_load(q, (i_t * BT, 192), (BT, 64), (T, H*K), DT_Q, (1, 0))
                b_w = tl.aiu_load(w, (i_t * BT, 192), (BT, 64), (T, HV*K), DT_W, (1, 0))
            else:
                b_k4 = tl.load(k + o_k4[:, None] + o_t[None, :] * (H*K), mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_q = tl.load(q + o_t[:, None] * (H*K) + o_k4[None, :], mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_w = tl.load(w + o_t[:, None] * (HV*K) + o_k4[None, :], mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_o += tl.dot(b_q, b_h4.to(DT_Q))
            b_A += tl.dot(b_q, b_k4)
            b_v += tl.dot(b_w, b_h4.to(DT_W))

        b_v = tl.load(v + o_t[:, None] * (HV*V) + o_v[None, :], mask=m_ov, other=0.0) - b_v

        if USE_G:
            b_o = b_o * exp2(b_g)[:, None]
            b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(o + o_t[:, None] * (HV*V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_ov)

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        b_h1 += tl.dot(b_k1, b_v.to(DT_K))
        if K > 64:
            b_h2 += tl.dot(b_k2, b_v.to(DT_K))
        if K > 128:
            b_h3 += tl.dot(b_k3, b_v.to(DT_K))
        if K > 192:
            b_h4 += tl.dot(b_k4, b_v.to(DT_K))

    if STORE_FINAL_STATE:
        p_ht = ht + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_k1[:, None] & m_v[None, :])
        if K > 64:
            p_ht = ht + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_k2[:, None] & m_v[None, :])
        if K > 128:
            p_ht = ht + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_k3[:, None] & m_v[None, :])
        if K > 192:
            p_ht = ht + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_k4[:, None] & m_v[None, :])


def chunk_gated_delta_rule_fwd_h_o(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *q.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    assert K <= 256, "fused fwd_h+chunk_o kernel does not support head dimension larger than 256."
    if scale is None:
        scale = K ** -0.5
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    o = torch.empty_like(u)
    final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_fwd_kernel_h_o[grid](
        q=q,
        k=k,
        v=u,
        w=w,
        g=g,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        USE_AIU=USE_AIU,
    )
    return o, final_state
