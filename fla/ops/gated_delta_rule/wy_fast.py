# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        # --- PPU 810E best configs ---
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=4),   # H=8,16,48,64
        triton.Config({'BK': 32, 'BV': 32}, num_warps=4, num_stages=4),   # H=32
        triton.Config({'BK': 64, 'BV': 128}, num_warps=8, num_stages=1),  # H=128
        # --- PPU 890P best configs ---
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),   # H=8,16,32,48
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=5), # H=64,128
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_b = beta + bos*HV + i_h + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    p_A = A + (bos*HV + i_h) * BT + o_t[:, None] * (HV*BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_u = u + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    if USE_G:
        p_g = g + (bos*HV + i_h) + o_t * HV
        b_g = exp2(tl.load(p_g, mask=m_t, other=0.0))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_w = w + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


def _prune_wy_bwd_configs(configs, named_args, **kwargs):
    """For large K (>192) use larger BK tiles to reduce the number of K splits
    and the bf16 dot accumulation error in dk."""
    K = kwargs.get('K', 0)
    if K > 192:
        return [c for c in configs if c.kwargs.get('BK', 32) >= 64]
    return configs


def _wy_bwd_autotune_kwargs():
    if IS_NVIDIA_BLACKWELL:
        # Blackwell guard (#913): use only the upstream B200-validated 2-warps / 4-stages configuration.
        return {
            'configs': [triton.Config({}, num_warps=2, num_stages=4)],
            'key': ['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
        }
    return {
        'configs': [
            triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=2),
            triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=4),
            triton.Config({'BK': 16, 'BV': 32}, num_warps=4, num_stages=3),
            triton.Config({'BK': 16, 'BV': 32}, num_warps=4, num_stages=4),
            # Large-K hardening tiles; only active when K > 192 via early_config_prune.
            triton.Config({'BK': 64, 'BV': 32}, num_warps=4, num_stages=2),
            triton.Config({'BK': 64, 'BV': 32}, num_warps=4, num_stages=3),
            triton.Config({'BK': 128, 'BV': 32}, num_warps=8, num_stages=2),
            triton.Config({'BK': 128, 'BV': 32}, num_warps=8, num_stages=3),
        ],
        'key': ['H', 'HV', 'K', 'V', 'BT', 'IS_VARLEN'],
        'prune_configs_by': {'early_config_prune': _prune_wy_bwd_configs},
    }


def _wy_bwd_tile_kwargs(K, V):
    if not IS_NVIDIA_BLACKWELL:
        return {}
    limit = 64 if check_shared_mem() else 32
    return {
        'BK': min(max(triton.next_power_of_2(K), 16), limit),
        'BV': min(max(triton.next_power_of_2(V), 16), limit),
    }


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    **_wy_bwd_autotune_kwargs(),
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_b = beta + (bos*HV + i_h) + o_t * HV
    p_db = db + (bos*HV + i_h) + o_t * HV
    p_A = A + (bos*HV + i_h) * BT + o_A[:, None] + o_t[None, :] * (HV*BT)

    b_b = tl.load(p_b, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        p_g = g + (bos*HV + i_h) + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_g_exp = exp2(b_g)
        b_bg = b_b * b_g_exp
        b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dw = dw + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        # [BT, BK]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_dw = tl.load(p_dw, mask=m_k, other=0.0)
        if USE_G:
            b_kbg = (b_k * b_bg[:, None]).to(b_dw.dtype)
        else:
            b_kbg = (b_k * b_b[:, None]).to(b_dw.dtype)

        # only accumulate dA here; dkbg=A@dw is deferred to loop3
        # to eliminate the dk1 global store/load round-trip
        b_dA = tl.dot(b_dw, tl.trans(b_kbg), acc=b_dA)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        p_du = du + (bos*HV + i_h) * V + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, mask=m_v, other=0.0)
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.dot(tl.where(m_A, b_dA, 0).to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    if USE_G:
        b_dA *= exp2(b_g[:, None] - b_g[None, :])

    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)

    # hoist (dA*beta)^T out of the i_k loop: reused by all K blocks.
    # folding beta into dA lets both loop3 dots share the same B operand (b_k)
    # (dA^T @ (k*beta) == (dA*beta)^T @ k)
    b_dAbt = tl.trans((b_dA * b_b[:, None]).to(b_dA.dtype))
    tl.debug_barrier()
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos*H + i_h // (HV // H)) * K + o_t[:, None] * (H*K) + o_k[None, :]
        p_dk = dk + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        p_dw = dw + (bos*HV + i_h) * K + o_t[:, None] * (HV*K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_dw = tl.load(p_dw, mask=m_k, other=0.0, eviction_policy="evict_first")

        # dkb/dAkb share b_k as B-operand; dkbg uses b_dw (evict_first)
        b_dkb = tl.dot(b_dA, b_k)
        b_dAkb = tl.dot(b_dAbt, b_k)
        b_dkbg = tl.dot(b_A, b_dw)

        if USE_G:
            b_dk = b_dkb * b_b[:, None] + b_dAkb + b_dkbg * b_bg[:, None]
            # db/dg from dkbg
            b_t = b_g_exp * tl.sum(b_dkbg * b_k, 1)
            b_db += b_t
            b_dg += b_t * b_b
            # db/dg from dkb
            b_t = tl.sum(b_dkb * b_k, 1)
            b_db += b_t
            b_dg += b_t * b_b
            # dg from dAkb
            b_dg -= tl.sum(b_k * b_dAkb, 1)
        else:
            b_dk = b_dkb * b_b[:, None] + b_dAkb + b_dkbg * b_b[:, None]
            b_db += tl.sum(b_dkb * b_k, 1)
            b_db += tl.sum(b_dkbg * b_k, 1)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)

    if USE_G:
        p_dg = dg + (bos*HV + i_h) + o_t * HV
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)


@dispatch('gated_delta_rule')
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
    )
    return w, u


@dispatch('gated_delta_rule')
def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dk = k.new_empty(B, T, HV, K)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if g is not None else None
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        **_wy_bwd_tile_kwargs(K, V),
    )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


fwd_recompute_w_u = recompute_w_u_fwd
bwd_prepare_wy_repr = prepare_wy_repr_bwd
