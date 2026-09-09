# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs, check_shared_mem

if IS_NVIDIA_BLACKWELL:
    """
    Compute tl.dot with SM100 workaround.

    On SM100 (Blackwell) GPUs, wraps the result in inline assembly to prevent
    the TritonGPUHoistTMEMAlloc pass from incorrectly fusing add and dot operations.
    See: https://github.com/fla-org/flash-linear-attention/issues/638

    TODO: Remove this workaround once the Triton compiler bug is fixed.
    Track upstream issue at: https://github.com/triton-lang/triton/issues/8695
    """
    @triton.jit
    def safe_dot(a, b):
        return tl.inline_asm_elementwise(
            asm="mov.f32 $0, $1;",
            constraints="=r,r",
            args=[tl.dot(a, b)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
else:
    @triton.jit
    def safe_dot(a, b):
        return tl.dot(a, b)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        # --- 810E best configs ---
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=4),   # H=8,16,48,64
        triton.Config({'BK': 32, 'BV': 32}, num_warps=4, num_stages=4),   # H=32
        triton.Config({'BK': 64, 'BV': 128}, num_warps=8, num_stages=1),  # H=128
        # --- 890P best configs ---
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),   # H=8,16,32,48
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=5), # H=64,128
    ],
    key=['H', 'K', 'V', 'BT', 'IS_VARLEN'],
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
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = exp(tl.load(p_g, boundary_check=(0,)))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _prune_wy_bwd_configs(configs, named_args, **kwargs):
    """For large K (>192) use larger BK tiles to reduce the number of K splits
    and the bf16 dot accumulation error in dk."""
    K = kwargs.get('K', 0)
    if K > 192:
        return [c for c in configs if c.kwargs.get('BK', 32) >= 64]
    return configs


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
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
    key=['H', 'K', 'V', 'BT', 'IS_VARLEN'],
    prune_configs_by={'early_config_prune': _prune_wy_bwd_configs},
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
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_exp = tl.exp(b_g)
        b_bg = b_b * b_g_exp
        b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        if USE_G:
            b_kbg = (b_k * b_bg[:, None]).to(b_dw.dtype)
        else:
            b_kbg = b_k * b_b[:, None]

        # only accumulate dA here; dkbg=A@dw is deferred to loop3
        # to eliminate the dk1 global store/load round-trip
        b_dA = tl.dot(b_dw, tl.trans(b_kbg), acc=b_dA)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA = tl.dot(b_du, tl.trans(b_vb), acc=b_dA)
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.dot(tl.where(m_A, b_dA, 0).to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    if USE_G:
        b_dA *= exp(b_g[:, None] - b_g[None, :])

    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)

    # hoist (dA*beta)^T out of the i_k loop: reused by all K blocks.
    # folding beta into dA lets both loop3 dots share the same B operand (b_k)
    # (dA^T @ (k*beta) == (dA*beta)^T @ k)
    b_dAbt = tl.trans((b_dA * b_b[:, None]).to(b_dA.dtype))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dw = tl.load(p_dw, boundary_check=(0, 1), eviction_policy="evict_first")

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

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    if USE_G:
        p_dg = tl.make_block_ptr(dg + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B*H)](
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
        K=K,
        V=V,
        BT=BT,
    )
    return w, u


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g) if g is not None else None
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(NT, B * H)](
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
        K=K,
        V=V,
        BT=BT,
    )
    return dk, dv, db, dg


fwd_recompute_w_u = recompute_w_u_fwd
bwd_prepare_wy_repr = prepare_wy_repr_bwd
