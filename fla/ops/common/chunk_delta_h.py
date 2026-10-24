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

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA_BLACKWELL,
    IS_NVIDIA_HOPPER,
    IS_PPU,
    autotune_cache_kwargs,
    check_shared_mem,
)

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]

# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# and we re-validate the wider config space.
if IS_NVIDIA_BLACKWELL:
    GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2]
elif IS_PPU:
    # PPU-verified expanded num_warps space (seq=8K, heads=16..128).
    GATED_DELTA_RULE_FWD_H_NUM_WARPS = [4, 8, 16]
else:
    GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2, 4]

# Stage the repeatedly-read k/w blocks through the PPU AIU. Gated on the device, since
# tl.aiu_load only exists in the PPU triton fork; FLA_PPU_DISABLE_AIU=1 forces the plain
# tl.load path for A/B timing. The env var is read once at import time and frozen, since
# the PPU triton fork rejects kernel-level references to plain globals.
_IS_AIU_ENABLED = IS_PPU and os.environ.get('FLA_PPU_DISABLE_AIU', '0') != '1'
USE_AIU = tl.constexpr(_IS_AIU_ENABLED)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
        for num_stages in ([2, 3, 4] if IS_PPU or check_shared_mem('ampere') else [2, 1])
        for BV in ([32, 64, 128] if IS_PPU else ([32, 64] if check_shared_mem('ada') else [32]))
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_AIU: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    # Resolve the element type once per kernel rather than at each aiu_load / cast site,
    # so the accessor is a single point of edit if it ever has to change. Well defined
    # here because w/k are kernel pointer args advanced only by integer offsets.
    DT_W = w.dtype.element_ty
    DT_K = k.dtype.element_ty

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
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
        if STATE_V_FIRST:
            p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
            m_h0_1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
            m_h0_1 = m_k1[:, None] & m_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                m_h0_2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                m_h0_2 = m_k2[:, None] & m_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                m_h0_3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                m_h0_3 = m_k3[:, None] & m_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                m_h0_4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                m_h0_4 = m_k4[:, None] & m_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV*K*V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV*K*V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV*K*V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV*K*V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV*K*V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)

        # aiu_load clamps rows only; its 64-wide column block walks the physical row
        # stride (HV*K here), so it is only safe when K tiles into exact 64-wide blocks.
        if USE_AIU and K % 64 == 0:
            b_w = tl.aiu_load(w, (i_t * BT, 0), (BT, 64), (T, HV*K), DT_W, (1, 0))
        else:
            p_w = w + o_t[:, None] * (HV*K) + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(DT_W))
        else:
            b_v = tl.dot(b_w, b_h1.to(DT_W))
        if K > 64:
            if USE_AIU and K % 64 == 0:
                b_w = tl.aiu_load(w, (i_t * BT, 64), (BT, 64), (T, HV*K), DT_W, (1, 0))
            else:
                p_w = w + o_t[:, None] * (HV*K) + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(DT_W))
            else:
                b_v += tl.dot(b_w, b_h2.to(DT_W))
        if K > 128:
            if USE_AIU and K % 64 == 0:
                b_w = tl.aiu_load(w, (i_t * BT, 128), (BT, 64), (T, HV*K), DT_W, (1, 0))
            else:
                p_w = w + o_t[:, None] * (HV*K) + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(DT_W))
            else:
                b_v += tl.dot(b_w, b_h3.to(DT_W))
        if K > 192:
            if USE_AIU and K % 64 == 0:
                b_w = tl.aiu_load(w, (i_t * BT, 192), (BT, 64), (T, HV*K), DT_W, (1, 0))
            else:
                p_w = w + o_t[:, None] * (HV*K) + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(DT_W))
            else:
                b_v += tl.dot(b_w, b_h4.to(DT_W))
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        if USE_G:
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
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        # Delay the b_v cast into each dot: the aiu path feeds DT_K-typed blocks straight
        # in, and per-dot casts avoid a full-tile conversion of an already-loaded value.
        if USE_AIU and K % 64 == 0:
            b_k = tl.aiu_load(k, (0, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
        else:
            p_k = k + o_k1[:, None] + o_t[None, :] * (H*K)
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v.to(DT_K)))
        else:
            b_h1 += tl.dot(b_k, b_v.to(DT_K))
        if K > 64:
            if USE_AIU and K % 64 == 0:
                b_k = tl.aiu_load(k, (64, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
            else:
                p_k = k + o_k2[:, None] + o_t[None, :] * (H*K)
                b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v.to(DT_K)))
            else:
                b_h2 += tl.dot(b_k, b_v.to(DT_K))
        if K > 128:
            if USE_AIU and K % 64 == 0:
                b_k = tl.aiu_load(k, (128, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
            else:
                p_k = k + o_k3[:, None] + o_t[None, :] * (H*K)
                b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v.to(DT_K)))
            else:
                b_h3 += tl.dot(b_k, b_v.to(DT_K))
        if K > 192:
            if USE_AIU and K % 64 == 0:
                b_k = tl.aiu_load(k, (192, i_t * BT), (64, BT), (H*K, T), DT_K, (0, 1))
            else:
                p_k = k + o_k4[:, None] + o_t[None, :] * (H*K)
                b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v.to(DT_K)))
            else:
                b_h4 += tl.dot(b_k, b_v.to(DT_K))

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + o_v[:, None] * K + o_k1[None, :]
            m_ht = m_v[:, None] & m_k1[None, :]
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
            m_ht = m_k1[:, None] & m_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                m_ht = m_v[:, None] & m_k2[None, :]
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                m_ht = m_k2[:, None] & m_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                m_ht = m_v[:, None] & m_k3[None, :]
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                m_ht = m_k3[:, None] & m_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                m_ht = m_v[:, None] & m_k4[None, :]
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                m_ht = m_k4[:, None] & m_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


def _prune_dhu_configs(configs, named_args, **kwargs):
    """The kernel splits K into at most four BK-wide tiles, so a config has to cover the
    whole head; K > 192 keeps BK=128 to avoid the extra bf16 dot error of narrower tiles.
    Cached configs written before BK existed carry only BV, cannot be checked for
    coverage, and are dropped so the cache entry is retired and re-tuned."""
    K = {**named_args, **kwargs}['K']
    configs = [c for c in configs if 'BK' in c.kwargs and 4 * c.kwargs['BK'] >= K]
    if K > 192:
        return [c for c in configs if c.kwargs['BK'] >= 128]
    return configs


_DHU_AUTOTUNE_KWARGS = {
    'prune_configs_by': {'early_config_prune': _prune_dhu_configs},
    'validate_cached_config': True,
} if IS_PPU else {}


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=(
        # PPU-verified best BK/BV configs (seq=8K, heads=16..128): BK is autotuned
        # jointly with BV, replacing the stock kernel's fixed 64-wide K split.
        [
            triton.Config({'BV': 128, 'BK': 128}, num_warps=16, num_stages=2),
            triton.Config({'BV': 128, 'BK': 32}, num_warps=8, num_stages=2),
            # lower register pressure (nw=4, ns=1)
            triton.Config({'BV': 128, 'BK': 32}, num_warps=4, num_stages=2),
            triton.Config({'BV': 128, 'BK': 32}, num_warps=4, num_stages=1),
            triton.Config({'BV': 128, 'BK': 32}, num_warps=8, num_stages=1),
            # smaller BV=64 to halve shared mem & dh accumulator regs
            triton.Config({'BV': 64, 'BK': 32}, num_warps=4, num_stages=2),
            triton.Config({'BV': 64, 'BK': 32}, num_warps=4, num_stages=1),
            triton.Config({'BV': 64, 'BK': 32}, num_warps=8, num_stages=2),
            # BK=64 (fewer K-loop iterations)
            triton.Config({'BV': 128, 'BK': 64}, num_warps=8, num_stages=2),
            triton.Config({'BV': 128, 'BK': 64}, num_warps=4, num_stages=2),
        ]
        if IS_PPU
        else [
            triton.Config({'BV': BV, 'BK': 64}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4]
            for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1])
            for BV in ([32, 64] if check_shared_mem('ada') else [32])
        ]
    ),
    # PPU keeps BV/BK out of the key since its shapes are fixed, so re-tuning per V
    # width is pure overhead; stock configs keep BV in the key to preserve behaviour.
    key=(['H', 'HV', 'K', 'V', 'BT', 'USE_G', 'STATE_V_FIRST'] if IS_PPU
         else ['H', 'HV', 'K', 'V', 'BT', 'BV', 'USE_G', 'STATE_V_FIRST']),
    **_DHU_AUTOTUNE_KWARGS,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_AIU: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV] or [BV, BK] per STATE_V_FIRST; BK is autotuned on PPU (32..128) and
    # pinned to 64 elsewhere, matching the stock kernel's fixed K split.
    if STATE_V_FIRST:
        b_dh1 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > BK:
            b_dh2 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > 2*BK:
            b_dh3 = tl.zeros([BV, BK], dtype=tl.float32)
        if K > 3*BK:
            b_dh4 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > BK:
            b_dh2 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > 2*BK:
            b_dh3 = tl.zeros([BK, BV], dtype=tl.float32)
        if K > 3*BK:
            b_dh4 = tl.zeros([BK, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    # Same single-point-of-edit rationale as the fwd kernel above.
    DT_K = k.dtype.element_ty

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, BK)
    m_k1 = o_k1 < K
    o_k2 = BK + o_k1
    m_k2 = o_k2 < K
    o_k3 = 2*BK + o_k1
    m_k3 = o_k3 < K
    o_k4 = 3*BK + o_k1
    m_k4 = o_k4 < K
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht1 = dht + o_v[:, None] * K + o_k1[None, :]
            m_dht1 = m_v[:, None] & m_k1[None, :]
        else:
            p_dht1 = dht + o_k1[:, None] * V + o_v[None, :]
            m_dht1 = m_k1[:, None] & m_v[None, :]
        b_dh1 += tl.load(p_dht1, mask=m_dht1, other=0.0)
        if K > BK:
            if STATE_V_FIRST:
                p_dht2 = dht + o_v[:, None] * K + o_k2[None, :]
                m_dht2 = m_v[:, None] & m_k2[None, :]
            else:
                p_dht2 = dht + o_k2[:, None] * V + o_v[None, :]
                m_dht2 = m_k2[:, None] & m_v[None, :]
            b_dh2 += tl.load(p_dht2, mask=m_dht2, other=0.0)
        if K > 2*BK:
            if STATE_V_FIRST:
                p_dht3 = dht + o_v[:, None] * K + o_k3[None, :]
                m_dht3 = m_v[:, None] & m_k3[None, :]
            else:
                p_dht3 = dht + o_k3[:, None] * V + o_v[None, :]
                m_dht3 = m_k3[:, None] & m_v[None, :]
            b_dh3 += tl.load(p_dht3, mask=m_dht3, other=0.0)
        if K > 3*BK:
            if STATE_V_FIRST:
                p_dht4 = dht + o_v[:, None] * K + o_k4[None, :]
                m_dht4 = m_v[:, None] & m_k4[None, :]
            else:
                p_dht4 = dht + o_k4[:, None] * V + o_v[None, :]
                m_dht4 = m_k4[:, None] & m_v[None, :]
            b_dh4 += tl.load(p_dht4, mask=m_dht4, other=0.0)

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_dh1 = dh + i_t_int64*HV*K*V + o_v[:, None] * K + o_k1[None, :]
            m_dh1 = m_v[:, None] & m_k1[None, :]
        else:
            p_dh1 = dh + i_t_int64*HV*K*V + o_k1[:, None] * V + o_v[None, :]
            m_dh1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_dh1)
        if K > BK:
            if STATE_V_FIRST:
                p_dh2 = dh + i_t_int64*HV*K*V + o_v[:, None] * K + o_k2[None, :]
                m_dh2 = m_v[:, None] & m_k2[None, :]
            else:
                p_dh2 = dh + i_t_int64*HV*K*V + o_k2[:, None] * V + o_v[None, :]
                m_dh2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_dh2)
        if K > 2*BK:
            if STATE_V_FIRST:
                p_dh3 = dh + i_t_int64*HV*K*V + o_v[:, None] * K + o_k3[None, :]
                m_dh3 = m_v[:, None] & m_k3[None, :]
            else:
                p_dh3 = dh + i_t_int64*HV*K*V + o_k3[:, None] * V + o_v[None, :]
                m_dh3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_dh3)
        if K > 3*BK:
            if STATE_V_FIRST:
                p_dh4 = dh + i_t_int64*HV*K*V + o_v[:, None] * K + o_k4[None, :]
                m_dh4 = m_v[:, None] & m_k4[None, :]
            else:
                p_dh4 = dh + i_t_int64*HV*K*V + o_k4[:, None] * V + o_v[None, :]
                m_dh4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_dh4)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
            p_g = g + bos * HV + i_h + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * (HV*V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]

        # Update dv
        # aiu path only when K tiles exactly into BK-wide blocks -- otherwise the tail
        # block reaches into the next head (same reason as the fwd kernel above).
        if USE_AIU and K % BK == 0:
            b_k = tl.aiu_load(k, (i_t * BT, 0), (BT, BK), (T, H*K), DT_K, (1, 0))
        else:
            p_k = k + o_t[:, None] * (H*K) + o_k1[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if USE_GK:
            o_k1 = tl.arange(0, BK)
            b_gk_last1 = tl.load(gk + last_idx * HV*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        if STATE_V_FIRST:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > BK:
            if USE_AIU and K % BK == 0:
                b_k = tl.aiu_load(k, (i_t * BT, BK), (BT, BK), (T, H*K), DT_K, (1, 0))
            else:
                p_k = k + o_t[:, None] * (H*K) + o_k2[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if USE_GK:
                b_gk_last2 = tl.load(gk + last_idx * HV*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv = tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype), acc=b_dv)
            else:
                b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), acc=b_dv)

        if K > 2*BK:
            if USE_AIU and K % BK == 0:
                b_k = tl.aiu_load(k, (i_t * BT, 2*BK), (BT, BK), (T, H*K), DT_K, (1, 0))
            else:
                p_k = k + o_t[:, None] * (H*K) + o_k3[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if USE_GK:
                b_gk_last3 = tl.load(gk + last_idx * HV*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv = tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype), acc=b_dv)
            else:
                b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), acc=b_dv)

        if K > 3*BK:
            if USE_AIU and K % BK == 0:
                b_k = tl.aiu_load(k, (i_t * BT, 3*BK), (BT, BK), (T, H*K), DT_K, (1, 0))
            else:
                p_k = k + o_t[:, None] * (H*K) + o_k4[None, :]
                b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if USE_GK:
                b_gk_last4 = tl.load(gk + last_idx * HV*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv = tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype), acc=b_dv)
            else:
                b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), acc=b_dv)

        if USE_G:
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.0, eviction_policy='evict_first')

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        # Update dh
        # b_do is only consumed here, so load it right before the dh updates instead of
        # next to the dv loads (shortens register lifetime). Only g_exp is folded into
        # the operand: folding `scale` as well rounds do * scale to the operand dtype,
        # which overflows fp16 for a large custom scale, so it stays on the fp32 result.
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0, eviction_policy='evict_first')
        if USE_G:
            b_do *= b_g_exp[:, None]
        p_w = w + o_k1[:, None] + o_t[None, :] * (HV*K)
        p_q = q + o_k1[:, None] + o_t[None, :] * (H*K)
        b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if USE_G:
            b_dh1 *= bg_last_exp
        if USE_GK:
            if STATE_V_FIRST:
                b_dh1 *= exp2(b_gk_last1)[None, :]
            else:
                b_dh1 *= exp2(b_gk_last1[:, None])
        if STATE_V_FIRST:
            # keep the stock transposed subtraction on this path: the two dots already
            # produce the [K, V] tile and only the final add needs the transpose
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            # keep `scale` on the q @ do result (see the b_do load above) and chain only
            # the w @ dv term onto the fp32 accumulator, with the w sign in the operand
            b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype)) * scale
            b_dh1 = tl.dot((-b_w), b_dv.to(b_w.dtype), acc=b_dh1)
        if K > BK:
            p_q = q + o_k2[:, None] + o_t[None, :] * (H*K)
            p_w = w + o_k2[:, None] + o_t[None, :] * (HV*K)
            b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh2 *= bg_last_exp
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_dh2 *= exp2(b_gk_last2[:, None])
            if STATE_V_FIRST:
                b_dh2 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype)) * scale
                b_dh2 = tl.dot((-b_w), b_dv.to(b_w.dtype), acc=b_dh2)
        if K > 2*BK:
            p_q = q + o_k3[:, None] + o_t[None, :] * (H*K)
            p_w = w + o_k3[:, None] + o_t[None, :] * (HV*K)
            b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh3 *= bg_last_exp
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_dh3 *= exp2(b_gk_last3[:, None])
            if STATE_V_FIRST:
                b_dh3 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype)) * scale
                b_dh3 = tl.dot((-b_w), b_dv.to(b_w.dtype), acc=b_dh3)
        if K > 3*BK:
            p_q = q + o_k4[:, None] + o_t[None, :] * (H*K)
            p_w = w + o_k4[:, None] + o_t[None, :] * (HV*K)
            b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh4 *= bg_last_exp
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_dh4 *= exp2(b_gk_last4[:, None])
            if STATE_V_FIRST:
                b_dh4 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype)) * scale
                b_dh4 = tl.dot((-b_w), b_dv.to(b_w.dtype), acc=b_dh4)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_dh0 = dh0 + o_v[:, None] * K + o_k1[None, :]
            m_dh0 = m_v[:, None] & m_k1[None, :]
        else:
            p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
            m_dh0 = m_k1[:, None] & m_v[None, :]
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_dh0)
        if K > BK:
            if STATE_V_FIRST:
                p_dh1 = dh0 + o_v[:, None] * K + o_k2[None, :]
                m_dh1 = m_v[:, None] & m_k2[None, :]
            else:
                p_dh1 = dh0 + o_k2[:, None] * V + o_v[None, :]
                m_dh1 = m_k2[:, None] & m_v[None, :]
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), mask=m_dh1)
        if K > 2*BK:
            if STATE_V_FIRST:
                p_dh2 = dh0 + o_v[:, None] * K + o_k3[None, :]
                m_dh2 = m_v[:, None] & m_k3[None, :]
            else:
                p_dh2 = dh0 + o_k3[:, None] * V + o_v[None, :]
                m_dh2 = m_k3[:, None] & m_v[None, :]
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), mask=m_dh2)
        if K > 3*BK:
            if STATE_V_FIRST:
                p_dh3 = dh0 + o_v[:, None] * K + o_k4[None, :]
                m_dh3 = m_v[:, None] & m_k4[None, :]
            else:
                p_dh3 = dh0 + o_k4[:, None] * V + o_v[None, :]
                m_dh3 = m_k4[:, None] & m_v[None, :]
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), mask=m_dh3)


@dispatch('common')
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        USE_AIU=USE_AIU,
    )
    return h, v_new, final_state


@dispatch('common')
def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        USE_AIU=USE_AIU,
    )
    return dh, dh0, dv2
