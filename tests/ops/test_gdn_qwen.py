# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""Coverage extension of the ``chunk_gated_delta_rule`` path for Qwen-series GDN configs.

Why a separate file: ``tests/ops/test_gdn.py`` stays byte-identical to upstream so
that "does upstream pass?" stays answerable. The gaps it leaves, which this file closes:

1. dtype -- upstream chunk cases are fp16/fp32 only. Production (and the PPU-810E ``aiu_load``
   path, which is bf16-only) runs bf16. The optimized kernels were therefore never validated
   in the dtype they are specialized for.
2. head count -- upstream H <= 8. Production H is ``linear_num_value_heads / TP`` in
   {4,6,8,12,16,24,32,48,64}. autotune configs are pruned per head-count bucket
   (8/16/32/64/128), so the configs actually selected in production are untested.
3. sequence length -- upstream T <= 2048; the target regime is 8K/16K, where gate cumsum
   underflow and int64 offset promotion start to matter.
4. varlen -- upstream varlen never uses D=128 nor ``use_qk_l2norm_in_kernel``, and its
   segment boundaries never stress the 1-token / off-by-one-past-chunk cases. Note also that
   "long varlen" must mean a long *segment*, not a long buffer, and a segment that fills the
   whole buffer is N=1 -- not varlen at all. The packed layouts alone top out at a 6385-token
   sequence, so Tier 4 carries explicit ``[0, 1, 8192]`` / ``[0, 1, 16385]`` layouts.
5. head dim -- upstream max D=128, so the ``K > 128`` / ``K > 192`` branches in
   ``fla/ops/common/chunk_delta_h.py`` (4th K-tile) are dead code under test, while
   ``GatedDeltaNet``'s default ``head_dim`` is 256.
6. input distribution -- upstream ``test_chunk`` builds q/k/v with ``torch.rand`` (all
   positive), which hides sign-dependent cancellation in the delta-rule accumulation.

Oracle strategy: the O(T) python ``naive_recurrent_gated_delta_rule`` is the strictest
reference but costs minutes at T=16K. ``naive_chunk_gated_delta_rule`` (exported from
``fla/ops/gated_delta_rule/naive.py``) iterates over T/64 chunks instead, i.e. ~64x cheaper.
``test_ref_consistency`` below pins the chunk oracle against the recurrent one in fp32
first, so the cheap oracle is earned rather than assumed; the matrix tests then use the
chunk oracle.

Run notes:
  * ``unset FLA_CI_ENV`` first -- ``fla.utils.assert_close`` downgrades assertions to warnings
    when it is set, which makes a green run meaningless.
  * CI scope: ``FLA_CI_ENV=1`` also shrinks the matrix to one case per compile path (15 cases
    instead of 79). ``FLA_TEST_GDN_QWEN_SCOPE=full`` opts back into the cross product; see the
    "CI scope" block below for which axis each retained case stands in for.
  * long-sequence tier is opt-in: ``FLA_TEST_LONG_SEQ=1 pytest tests/ops/test_gdn_qwen.py``
    (env-var gating follows the existing ``SKIP_TEST_CHUNK_VARLEN`` idiom in this repo).
"""

import os

import pytest
import torch
import torch.nn.functional as F

# Give this process its own triton cache dir. `chunk_delta_h.py`'s dhu kernel writes a
# `*.source` sidecar into the cache during compilation; two pytest runs sharing the default
# /root/.triton/cache race on it and one dies with FileNotFoundError mid-compile. Isolating
# per-pid makes concurrent runs (e.g. base vs ours on two cards) safe. Must be set before the
# first fla import triggers any compile.
os.environ.setdefault('TRITON_CACHE_DIR', f'/tmp/triton_cache_{os.getpid()}')

from fla.ops.gated_delta_rule import (  # noqa: E402
    chunk_gated_delta_rule,
    naive_chunk_gated_delta_rule,
    naive_recurrent_gated_delta_rule,
)
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device  # noqa: E402

# ---------------------------------------------------------------------------------------------
# CI scope
# ---------------------------------------------------------------------------------------------
# The full matrix is 79 cases and ~53 min cold on one PPU, but the cost is not spread evenly
# over cases: what hurts is the first touch of each Triton *compile key*, because that first
# touch re-compiles and re-autotunes the kernel's whole config space from an empty per-pid
# cache (``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` alone sweeps 27 configs).
#
# Reading the ``key=`` / ``@triton.heuristics`` off this path (l2norm -> cumsum -> kkt_solve ->
# recompute_w_u -> fwd_h -> fwd_o, then dv_local -> dhu -> dqkwg -> wy_bwd -> reverse cumsum)
# splits the axes in two:
#
#   compile axes  H, K (= head dim, which also selects the K > 64/128/192 tile guards),
#                 IS_VARLEN, USE_INITIAL_STATE / STORE_FINAL_STATE / USE_FINAL_STATE_GRADIENT
#                 (passing h0, asking for ht, backpropping dht), in-kernel vs external l2norm,
#                 and dtype (which recompiles without re-sweeping).
#   value axes    T, gate_logit_normalizer, mask_p. These change numbers, not paths.
#
# An untested compile-axis value is an untested kernel binary, so CI keeps one case per value
# of every compile axis and only the corners of the value axes: T=1000 already exercises the
# partial-chunk masked loads that T=2048 skips, and a nominal gate without masking is what the
# dense cases already run. CI buys less than the full matrix even where it does run, because
# ``FLA_CI_ENV=1`` also makes ``assert_close`` downgrade anything below a 1% error rate to a
# warning -- the gate there is crashes, NaN and gross divergence.
#
# ``FLA_TEST_GDN_QWEN_SCOPE=full`` restores the cross product under CI, ``=ci`` shrinks a local
# run; anything else is a typo worth failing on. Cutting below 15 cases starts dropping kernel
# binaries rather than just numbers.
_SCOPE = (os.getenv('FLA_TEST_GDN_QWEN_SCOPE') or '').strip().lower() or (
    'ci' if os.getenv('FLA_CI_ENV') == '1' else 'full')
if _SCOPE not in ('ci', 'full'):
    raise ValueError(f'FLA_TEST_GDN_QWEN_SCOPE must be "ci" or "full", got {_SCOPE!r}')

# ---------------------------------------------------------------------------------------------
# Qwen GDN config matrix
# ---------------------------------------------------------------------------------------------
# Only linear_num_value_heads varies across the family; head dims are 128 (K) / 128 (V).
# The kernel API takes no GQA/GVA (fla/layers/gated_deltanet.py repeats q/k up to num_v_heads
# before calling in), so the only kernel-visible knob is num_value_heads / TP.
QWEN_GDN_NUM_V_HEADS = {
    'Qwen3.5-35B-A3B': 32,
    'Qwen3.5-122B-A10B': 64,
    'Qwen3.6-35B-A3B': 32,
    'Qwen3.6-27B': 48,
    # The next family generation doubles it again. HV=128 matters for two reasons:
    # it is the top autotune pruning bucket, and at TP=1 it makes grid = B*H = 128 blocks,
    # which is the regime where "larger grid is slower" was measured on the serial kernels
    # (BV=128/32blk 1.90 ms < BV=64/64blk 2.29 < BV=32/128blk 2.78).
    'Qwen-Next-1': 128,
    'Qwen-Next-2': 128,
}
TP_SIZES = (1, 2, 4, 8)
HEAD_DIM = 128


def _local_head_matrix() -> dict[int, list[str]]:
    """H seen by the kernel on one rank -> which (model, TP) combos produce it."""
    matrix: dict[int, list[str]] = {}
    for name, hv in QWEN_GDN_NUM_V_HEADS.items():
        for tp in TP_SIZES:
            assert hv % tp == 0, f'{name}: HV={hv} not divisible by TP={tp}'
            matrix.setdefault(hv // tp, []).append(f'{name}/TP{tp}')
    return dict(sorted(matrix.items()))


LOCAL_HEADS = _local_head_matrix()  # {4,6,8,12,16,24,32,48,64,128}


def _tol(dtype: torch.dtype, T: int) -> dict[str, float]:
    """Relative-error budget. ``assert_close`` compares ||ref-tri||/||ref||, not elementwise.

    fp16 lives in the regime upstream ``test_chunk`` covers (same dtype, T <= 2048), so its
    budget is upstream's numbers verbatim -- one standard for the shared fp16 domain. bf16
    has no upstream counterpart (upstream has no bf16 case at all) and keeps the scaled
    budget: 8 mantissa bits vs fp16's 11 means ~8x the rounding noise at equal shape. Either
    way, gradients accumulate over T/64 chunks and db/dg go through an extra reduction, so
    they get progressively more slack; T > 4096 (upstream tops out at 2048) extends the
    budget 1.5x for gate-cumsum drift. Note the inputs here are randn (signed), a harsher
    distribution than upstream's rand -- the fp16 numbers still hold empirically.
    """
    if dtype == torch.float16:
        # verbatim from tests/ops/test_gdn.py::test_chunk
        base = {'o': 0.005, 'ht': 0.005, 'dq': 0.008, 'dk': 0.008,
                'dv': 0.008, 'dh0': 0.008, 'db': 0.02, 'dg': 0.02}
    else:
        base = {'o': 0.02, 'ht': 0.02, 'dq': 0.03, 'dk': 0.03,
                'dv': 0.03, 'dh0': 0.03, 'db': 0.06, 'dg': 0.06}
    long_seq = 1.5 if T > 4096 else 1.0
    return {name: v * long_seq for name, v in base.items()}


def _max_seg(cu_seqlens: list[int]) -> int:
    """Longest single sequence in a packed buffer.

    This -- not ``cu_seqlens[-1]`` -- is what should drive the error budget on the varlen path:
    every segment restarts the state and its own ``exp(cumsum(g))`` accumulation, so rounding
    does not compound across a packing boundary. Keying ``_tol`` on the buffer total instead
    would hand "16K buffer of 4K segments" the same 1.5x slack as a genuine 16K sequence.
    """
    return max(b - a for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]))


def _make_inputs(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    gate_logit_normalizer: float = 1.0,
    mask_p: float = 0.0,
    use_h0: bool = True,
    n_states: int | None = None,
    seed: int = 42,
):
    """randn (not rand) for q/k/v: signed inputs are what production sees and what exposes
    cancellation in the ``v - w @ h`` update."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    if mask_p > 0:
        g = g * (torch.rand_like(g) > mask_p)
    tensors = [q, k, v, beta, g]
    if use_h0:
        tensors.append(torch.randn(n_states or B, H, D, D, dtype=torch.float32))
    out = [x.to(device).requires_grad_(True) for x in tensors]
    return out if use_h0 else out + [None]


def _grads_of(tensors):
    return [t.grad if t is not None else None for t in tensors]


def _clear_grads(tensors):
    for t in tensors:
        if t is not None:
            t.grad = None


def _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, tol):
    assert_close('o', ref, tri, tol['o'])
    assert_close('ht', ref_ht, tri_ht, tol['ht'])
    for name, r, t in zip(('dq', 'dk', 'dv', 'db', 'dg', 'dh0'), ref_grads, tri_grads):
        if r is None or t is None:
            continue
        assert_close(name, r, t, tol[name])


_DTYPE_IDS = ((torch.bfloat16, 'bf16'), (torch.float16, 'fp16'))


# ---------------------------------------------------------------------------------------------
# Tier 0: earn the cheap oracle
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('T', [1000, 1024])
def test_ref_consistency(T: int):
    """naive_chunk must agree with naive_recurrent in fp32 -- forward AND backward -- before
    the cheap oracle is used at 8K/16K. Without this, a chunk-oracle bug would silently become
    the accepted answer for exactly the regime we cannot afford to check the expensive way.

    T=1000 also covers the partial trailing chunk, where naive_chunk zero-pads to a multiple
    of 64 -- the situation the kernel handles via boundary_check / aiu tensor_shape.
    """
    B, H, D, scale = 1, 2, HEAD_DIM, HEAD_DIM ** -0.5
    q, k, v, beta, g, h0 = _make_inputs(B, T, H, D, torch.float32)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    o_chk, ht_chk = naive_chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        g=g, beta=beta, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((o_chk * do).sum() + (ht_chk * dht).sum()).backward()
    chk_grads = _grads_of(inputs)
    _clear_grads(inputs)

    o_rec, ht_rec = naive_recurrent_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        beta=beta, g=g, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((o_rec * do).sum() + (ht_rec * dht).sum()).backward()
    rec_grads = _grads_of(inputs)

    # both oracles are fp32, so the budget is tight -- only chunking order differs
    assert_close('o', o_rec, o_chk, 0.002)
    assert_close('ht', ht_rec, ht_chk, 0.002)
    for name, r, c in zip(('dq', 'dk', 'dv', 'db', 'dg', 'dh0'), rec_grads, chk_grads):
        assert_close(name, r, c, 0.005)


# ---------------------------------------------------------------------------------------------
# Tier 1: Qwen x TP head matrix (the main gap)
# ---------------------------------------------------------------------------------------------
def _matrix_cases() -> list:
    """``(H, T, dtype)`` cases: the full Qwen x TP cross product, or CI's one-per-path subset."""
    if _SCOPE == 'full':
        return [
            pytest.param(H, T, dtype, id=f'H{H}-{"|".join(LOCAL_HEADS[H])}-T{T}-{tag}')
            for H in LOCAL_HEADS
            for T in (2048, 1000)
            for dtype, tag in _DTYPE_IDS
        ]
    return [
        # off-bucket head count: below the smallest prune bucket (8), and not a power of two
        pytest.param(6, 1000, torch.bfloat16, id='H6-T1000-bf16-off-bucket'),
        # mid bucket at the production shape, on a partial trailing chunk (1000 = 15*64 + 40);
        # also the only case that passes h0 *and* asks for ht *and* backprops dht, i.e. the
        # USE_INITIAL_STATE / STORE_FINAL_STATE / USE_FINAL_STATE_GRADIENT = True specialization
        pytest.param(32, 1000, torch.bfloat16, id='H32-T1000-bf16-partial-chunk'),
        # fp16 anchor -- the only dtype here with an upstream tolerance to regress against
        pytest.param(32, 1000, torch.float16, id='H32-T1000-fp16'),
        # top bucket; at TP=1 this is the 128-block grid, where bigger grids measured slower
        pytest.param(128, 1000, torch.bfloat16, id='H128-T1000-bf16-top-bucket'),
    ]


@pytest.mark.parametrize(('H', 'T', 'dtype'), _matrix_cases())
def test_chunk_qwen_tp_matrix(H: int, T: int, dtype: torch.dtype):
    """H = linear_num_value_heads / TP for every Qwen GDN config x TP in {1,2,4,8}.

    Rationale for varying H at all: grid is B*H, and autotune picks BK/BV/num_warps/num_stages
    per grid size, so H selects the code path. Upstream never goes past H=8. The CI subset keeps
    only the head counts that bracket the autotune prune buckets -- see "CI scope" above.
    """
    B, D, scale = 1, HEAD_DIM, HEAD_DIM ** -0.5
    q, k, v, beta, g, h0 = _make_inputs(B, T, H, D, dtype)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    ref, ref_ht = naive_chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        g=g, beta=beta, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = _grads_of(inputs)

    _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, _tol(dtype, T))


# The recurrent oracle is an O(T) python-side loop -- the most expensive reference in this
# file. One head is enough to prove the chunk kernel against it, and H=6's odd-head path is
# still covered by ``test_chunk_qwen_tp_matrix``.
_RECURRENT_HEADS = [6, 32] if _SCOPE == 'full' else [32]


@pytest.mark.parametrize('H', _RECURRENT_HEADS)
def test_chunk_qwen_vs_recurrent(H: int):
    """Strictest per-token check against the O(T) fp32 recurrence, on a partial-chunk length.

    H=6 is Qwen3.6-27B at TP=8 (non-power-of-two, outside every autotune pruning bucket);
    H=32 is Qwen3.5-35B-A3B at TP=1.
    """
    B, T, D, dtype, scale = 1, 1000, HEAD_DIM, torch.bfloat16, HEAD_DIM ** -0.5
    q, k, v, beta, g, h0 = _make_inputs(B, T, H, D, dtype)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    ref, ref_ht = naive_recurrent_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        beta=beta, g=g, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = _grads_of(inputs)

    _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, _tol(dtype, T))


# ---------------------------------------------------------------------------------------------
# Tier 2: head-dim branches and gate extremes
# ---------------------------------------------------------------------------------------------
def _head_dim_cases() -> list:
    """``(H, D)``: one case per K tile boundary."""
    cases = [
        pytest.param(8, 60, id='H8-D60-unaligned'),    # K not a multiple of 64 nor 32B in bf16
        pytest.param(8, 64, id='H8-D64-single-block'),
        pytest.param(8, 100, id='H8-D100-unaligned'),
        pytest.param(8, 128, id='H8-D128-two-blocks'),
        pytest.param(4, 256, id='H4-D256-four-blocks'),  # lights up the K>128 / K>192 branches
    ]
    if _SCOPE == 'full':
        return cases
    # D=128 is what every matrix case already runs, and D=64 takes the same K<=64 tile as D=60
    # minus the misalignment, so CI keeps only the boundaries: K<64, K in (64, 128], K>192.
    return [c for c in cases if c.id not in ('H8-D64-single-block', 'H8-D128-two-blocks')]


@pytest.mark.parametrize(('H', 'D'), _head_dim_cases())
def test_chunk_head_dim_branches(H: int, D: int):
    """chunk_delta_h.py tiles K in 64-wide blocks behind ``K > 64/128/192`` guards.

    D=60/100 therefore load a full 64-column block out of a narrower tensor, and D=256 is the
    only way to reach the 4th K-tile. Both are untested upstream in bf16.
    """
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    B, T, dtype, scale = 1, 1024, torch.bfloat16, D ** -0.5
    q, k, v, beta, g, h0 = _make_inputs(B, T, H, D, dtype)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    ref, ref_ht = naive_chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        g=g, beta=beta, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = _grads_of(inputs)

    _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, _tol(dtype, T))


def _gate_cases() -> list:
    """``(gate_logit_normalizer, mask_p, use_h0)`` cases."""
    if _SCOPE == 'full':
        return [
            pytest.param(normalizer, mask_p, use_h0, id=f'{use_h0}-{tag}-{mask_p}')
            for use_h0 in (True, False)
            for normalizer, tag in (
                (0.05, 'gate-sharp-decay'),   # g scaled x20 -> exp() underflow pressure
                (1.0, 'gate-nominal'),
                (20.0, 'gate-near-identity'),  # g ~ 0 -> no forgetting, state grows
            )
            for mask_p in (0.0, 0.5)
        ]
    # Only ``use_h0`` picks a compiled flag (USE_INITIAL_STATE, plus STORE_FINAL_STATE and
    # USE_FINAL_STATE_GRADIENT, since ``use_h0=False`` also drops ht and dht); gate scale and
    # masking only change values. So CI takes the False side once per gate regime -- the True
    # side is the dense matrix cases -- and leaves the sharp/near-identity gate at nominal.
    return [
        pytest.param(0.05, 0.0, False, id='False-gate-sharp-decay-0.0'),
        pytest.param(20.0, 0.5, False, id='False-gate-near-identity-0.5'),
    ]


@pytest.mark.parametrize(
    ('gate_logit_normalizer', 'mask_p', 'use_h0'), _gate_cases(),
)
def test_chunk_gate_extremes(gate_logit_normalizer: float, mask_p: float, use_h0: bool):
    """Gate dynamics decide whether the chunkwise ``exp(cumsum(g))`` rescale over/underflows.

    ``use_h0=False`` matters independently: upstream ``test_chunk`` always passes a zero h0,
    so the ``USE_INITIAL_STATE=False`` specialization of the kernel is never taken there.
    """
    B, T, H, D, dtype, scale = 1, 2048, 32, HEAD_DIM, torch.bfloat16, HEAD_DIM ** -0.5
    q, k, v, beta, g, h0 = _make_inputs(
        B, T, H, D, dtype, gate_logit_normalizer=gate_logit_normalizer,
        mask_p=mask_p, use_h0=use_h0,
    )
    do = torch.randn_like(v)
    inputs = (q, k, v, beta, g, h0)

    tri, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    (tri * do).sum().backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    ref, _ = naive_chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        g=g, beta=beta, scale=scale, initial_state=h0, output_final_state=False,
    )
    (ref * do).sum().backward()
    ref_grads = _grads_of(inputs)

    tol = _tol(dtype, T)
    assert_close('o', ref, tri, tol['o'])
    for name, r, t in zip(('dq', 'dk', 'dv', 'db', 'dg', 'dh0'), ref_grads, tri_grads):
        if r is None or t is None:
            continue
        assert_close(name, r, t, tol[name])


# ---------------------------------------------------------------------------------------------
# Tier 3: varlen
# ---------------------------------------------------------------------------------------------
_CU_LAYOUTS = (
    # 1-token segment, one-past-chunk (65), exact multiple (128), partial tail.
    ([0, 1, 66, 194, 1000], 'cu-adversarial'),
    # every boundary just short of a chunk multiple
    ([0, 63, 127, 255, 4095], 'cu-off-by-one-short'),
    # realistic long-context packing
    ([0, 700, 1500, 4096, 8000], 'cu-realistic-8k'),
)


def _varlen_cases() -> list:
    """``(cu_seqlens, H, use_qk_l2norm_in_kernel)`` cases."""
    if _SCOPE == 'full':
        return [
            pytest.param(cu_seqlens, H, l2norm, id=f'{name}-{H}-{l2norm}')
            for cu_seqlens, name in _CU_LAYOUTS
            for H in (6, 8, 32)
            for l2norm in (True, False)
        ]
    return [
        # production path: in-kernel l2norm at H=32 (the varlen *and* normalize flags at once)
        pytest.param([0, 1, 66, 194, 1000], 32, True, id='cu-adversarial-32-True'),
        # odd head count on the out-of-kernel normalize path -- the other two axis values at once,
        # and a second IS_VARLEN compile key
        pytest.param([0, 1, 66, 194, 1000], 6, False, id='cu-adversarial-6-False'),
        # the longest buffer the non-opt-in tiers reach
        pytest.param([0, 700, 1500, 4096, 8000], 32, True, id='cu-realistic-8k-32-True'),
    ]


@pytest.mark.parametrize(('cu_seqlens', 'H', 'use_qk_l2norm_in_kernel'), _varlen_cases())
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping varlen tests because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen_qwen(cu_seqlens: list[int], H: int, use_qk_l2norm_in_kernel: bool):
    """varlen at production shape (D=128, bf16).

    This is the highest-risk configuration for the aiu_load migration: ``tl.aiu_load`` has no
    ``boundary_check`` argument, it only receives ``tensor_shape``. Every segment ends on its
    own partial chunk, so varlen multiplies the number of out-of-range block loads by the
    segment count. Upstream varlen cases (H=4, D=60/64/100, fp16) cannot detect this.
    """
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    D, dtype, scale = HEAD_DIM, torch.bfloat16, HEAD_DIM ** -0.5
    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    offsets = torch.LongTensor(cu_seqlens).to(device)

    q, k, v, beta, g, h0 = _make_inputs(1, T, H, D, dtype, n_states=N)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q if use_qk_l2norm_in_kernel else F.normalize(q, p=2, dim=-1),
        k=k if use_qk_l2norm_in_kernel else F.normalize(k, p=2, dim=-1),
        v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=offsets,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    qn, kn = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    ref, ref_ht = [], []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        ref_i, ref_ht_i = naive_chunk_gated_delta_rule(
            q=qn[:, s:e], k=kn[:, s:e], v=v[:, s:e], g=g[:, s:e], beta=beta[:, s:e],
            scale=scale, initial_state=h0[i:i + 1], output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref, ref_ht = torch.cat(ref, 1), torch.cat(ref_ht, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = _grads_of(inputs)

    _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, _tol(dtype, _max_seg(cu_seqlens)))


# ---------------------------------------------------------------------------------------------
# Tier 4: long sequence (opt-in -- the fp32 oracle dominates runtime here)
# ---------------------------------------------------------------------------------------------
requires_long_seq = pytest.mark.skipif(
    os.getenv('FLA_TEST_LONG_SEQ') != '1',
    reason='set FLA_TEST_LONG_SEQ=1 to run the 8K/16K tier',
)


@requires_long_seq
@pytest.mark.parametrize('T', [pytest.param(8192, id='T8192'), pytest.param(16384, id='T16384')])
@pytest.mark.parametrize('H', [8, 32])
def test_chunk_long_seq(T: int, H: int):
    """The actual target regime. Failure modes that only appear here: exp(cumsum(g)) underflow
    accumulated over T/64 chunks, and int64 promotion of the ``bos * H * K`` offsets
    (cf. upstream commit 854c4ce, "fix potential ood for long inputs").
    """
    B, D, dtype, scale = 1, HEAD_DIM, torch.bfloat16, HEAD_DIM ** -0.5
    q, k, v, beta, g, h0 = _make_inputs(B, T, H, D, dtype)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    ref, ref_ht = naive_chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1), v=v,
        g=g, beta=beta, scale=scale, initial_state=h0, output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = _grads_of(inputs)

    _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, _tol(dtype, T))


@requires_long_seq
@pytest.mark.parametrize(
    'cu_seqlens',
    [
        # --- genuinely variable-length buffers carrying a long segment ------------------------
        # A true N-token varlen *sequence* needs a buffer strictly longer than N, otherwise the
        # single segment eats the whole buffer and the case degenerates to N=1 (no length
        # variation, no packing boundary) -- which is emphatically not varlen coverage.
        # Hence 8192/16385 rather than 8191/16384 as the totals.
        pytest.param([0, 1, 8192], id='cu-varlen-8k'),      # segments 1 / 8191 -> partial tail
        pytest.param([0, 1, 16385], id='cu-varlen-16k'),    # segments 1 / 16384 -> aligned tail
        # The 1-token neighbour above is also the worst case for the per-segment ``tensor_shape``
        # handed to ``tl.aiu_load`` (no ``boundary_check``): it jumps 1 -> 8191/16384 between
        # consecutive program ids.
        # Two long segments, unequal and both ending mid-chunk (9000 = 140*64+40, 7384 = 115*64+24):
        # the second segment's `bos * H * K` offsets all sit past the 8K mark.
        pytest.param([0, 9000, 16384], id='cu-16k-two-long-uneven'),
        # Equal 8K segments: boundary lands exactly on a chunk multiple, unlike the case above.
        pytest.param([0, 8192, 16384], id='cu-8k-x2'),
        # --- packed layouts (buffer 16K, longest segment 4096 / 6385) -------------------------
        pytest.param([0, 4096, 8192, 12288, 16384], id='cu-16k-uniform'),
        pytest.param([0, 1, 1000, 4321, 9999, 16384], id='cu-16k-skewed'),
        # --- N=1 control, NOT varlen coverage -------------------------------------------------
        # One sequence filling the buffer: still goes through the cu_seqlens indexing path
        # (prepare_chunk_indices / chunk_offsets / bos-eos) but crosses no boundary. Kept purely
        # for differential diagnosis: if this passes while cu-varlen-16k fails, the fault is in
        # boundary handling, not in length. Do not cite it as "varlen 16K".
        pytest.param([0, 16384], id='cu-1seg-16k-ctrl'),
    ],
)
def test_chunk_varlen_long_seq(cu_seqlens: list[int]):
    """Variable-length buffers whose *longest segment* reaches the 8K/16K production regime.

    The packed layouts alone top out at a 6385-token segment, and gate-cumsum underflow /
    int64 ``bos * H * K`` promotion are per-segment effects -- so before the first four entries
    above, "16K varlen" only ever meant a 16K buffer of short sequences.

    Also asks for the final state and backprops ``dht``, which the previous revision skipped
    (``output_final_state=False``): that left ``ht`` and the ``dh0`` path unchecked at exactly
    the lengths where the state recurrence has had the most chunks to drift.
    """
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    H, D, dtype, scale = 32, HEAD_DIM, torch.bfloat16, HEAD_DIM ** -0.5
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1]
    offsets = torch.LongTensor(cu_seqlens).to(device)

    q, k, v, beta, g, h0 = _make_inputs(1, T, H, D, dtype, n_states=N)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    inputs = (q, k, v, beta, g, h0)

    tri, tri_ht = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=offsets,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = _grads_of(inputs)
    _clear_grads(inputs)

    qn, kn = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    ref, ref_ht = [], []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        ref_i, ref_ht_i = naive_chunk_gated_delta_rule(
            q=qn[:, s:e], k=kn[:, s:e], v=v[:, s:e], g=g[:, s:e], beta=beta[:, s:e],
            scale=scale, initial_state=h0[i:i + 1], output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref, ref_ht = torch.cat(ref, 1), torch.cat(ref_ht, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = _grads_of(inputs)

    _compare_fwd_bwd(ref, ref_ht, tri, tri_ht, ref_grads, tri_grads, _tol(dtype, _max_seg(cu_seqlens)))
