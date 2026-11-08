# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""End-to-end parity of the PLA-routed GDN paths against the Triton path.

Runs only on PPU boards with the `pla` package installed; the tolerance tier
follows the bf16 cross-path criterion (rel_max <= 1e-2 for prefill, whose two
arms chunk at 64 and 32 respectively, 5e-3 for decode and gradients).
"""

import math

import pytest
import torch

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.backends import gdr_registry
from fla.ops.gated_delta_rule.backends.pla import PLAGDNBackend

# gate on the backend's own availability (real import probe), not just find_spec
HAS_PLA = PLAGDNBackend.is_available()
pytestmark = pytest.mark.skipif(not HAS_PLA, reason='requires a PPU board with an importable pla')

DEVICE = 'cuda'
LOG_KEY = 'gated_delta_rule:chunk_gated_delta_rule:pla'


def triton_entry(fn):
    """Unwrap dispatch and compiler.disable layers down to the Triton original."""
    while hasattr(fn, '__wrapped__'):
        fn = fn.__wrapped__
    return fn


def rel_max(a, b):
    a, b = a.float(), b.float()
    diff = (a - b).abs().max().item()
    denom = b.abs().max().item()
    assert math.isfinite(diff) and math.isfinite(denom), 'non-finite parity operands'
    # clamp the denominator so an all-zero reference cannot mask a wrong result
    return diff / max(denom, 1e-12)


def prefill_inputs(T, H, Hg, seed=0, cu=None, grad=False):
    torch.manual_seed(seed)
    q = torch.randn(1, T, Hg, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=grad)
    k = torch.randn(1, T, Hg, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=grad)
    v = torch.randn(1, T, H, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=grad)
    g = (-torch.rand(1, T, H, device=DEVICE, dtype=torch.float32) * 0.05).requires_grad_(grad)
    beta = torch.randn(1, T, H, device=DEVICE, dtype=torch.bfloat16).sigmoid().requires_grad_(grad)
    h0 = torch.randn(len(cu) - 1 if cu is not None else 1, H, 128, 128,
                     device=DEVICE, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, h0


def raw_gate_inputs(T, H, Hg, seed):
    """Raw-gate primals: un-activated g/beta plus trainable state and gate params."""
    torch.manual_seed(seed)
    q = torch.randn(1, T, Hg, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, T, Hg, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, T, H, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(1, T, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    beta = torch.randn(1, T, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    h0 = (torch.randn(1, H, 128, 128, device=DEVICE, dtype=torch.float32) * 0.1).requires_grad_(True)
    A_log = torch.randn(H, device=DEVICE, dtype=torch.float32, requires_grad=True)
    dt_bias = torch.randn(H, device=DEVICE, dtype=torch.float32, requires_grad=True)
    return q, k, v, g, beta, h0, A_log, dt_bias


@pytest.mark.parametrize('case', ['dense', 'varlen_raw_gate'])
def test_prefill_parity(case, monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN_TRAIN', '0')
    if case == 'dense':
        q, k, v, g, beta, h0 = prefill_inputs(2048, 16, 4)
        kwargs = dict(use_qk_l2norm_in_kernel=True)
    else:
        q, k, v, g, beta, h0 = prefill_inputs(3048, 16, 4, seed=1,
                                              cu=[0, 1000, 3048])
        g = torch.randn(1, 3048, 16, device=DEVICE, dtype=torch.bfloat16)
        beta = torch.randn(1, 3048, 16, device=DEVICE, dtype=torch.bfloat16)
        A_log = torch.randn(16, device=DEVICE, dtype=torch.float32)
        dt_bias = torch.randn(16, device=DEVICE, dtype=torch.float32)
        kwargs = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                      use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                      cu_seqlens=torch.tensor([0, 1000, 3048], device=DEVICE, dtype=torch.long))
    gdr_registry._logged.discard(LOG_KEY)
    with torch.no_grad():
        o_p, ht_p = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0,
                                           output_final_state=True, **kwargs)
        o_t, ht_t = triton_entry(chunk_gated_delta_rule)(q, k, v, g, beta, initial_state=h0,
                                                         output_final_state=True, **kwargs)
    assert LOG_KEY in gdr_registry._logged, 'the call did not route to PLA'
    assert rel_max(o_p, o_t) <= 1e-2, rel_max(o_p, o_t)
    assert rel_max(ht_p, ht_t) <= 1e-2, rel_max(ht_p, ht_t)


def test_decode_parity(monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN_DECODE', '1')
    torch.manual_seed(7)
    N, H, HV = 16, 4, 8
    cu = torch.arange(0, N + 1, device=DEVICE, dtype=torch.long)
    a = torch.randn(1, N, HV, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(1, N, HV, device=DEVICE, dtype=torch.bfloat16)
    q = torch.randn(1, N, H, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, N, H, 128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, N, HV, 128, device=DEVICE, dtype=torch.bfloat16)
    h0 = torch.randn(N, HV, 128, 128, device=DEVICE, dtype=torch.float32) * 0.1
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEVICE, dtype=torch.float32)
    kwargs = dict(initial_state=h0, output_final_state=True,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                  cu_seqlens=cu)
    decode_key = 'gated_delta_rule:fused_recurrent_gated_delta_rule:pla'
    gdr_registry._logged.discard(decode_key)
    with torch.no_grad():
        o_p, ht_p = fused_recurrent_gated_delta_rule(q, k, v, g=a, beta=b, **kwargs)
        o_t, ht_t = triton_entry(fused_recurrent_gated_delta_rule)(q, k, v, g=a, beta=b, **kwargs)
    assert decode_key in gdr_registry._logged
    assert rel_max(o_p, o_t) <= 5e-3, rel_max(o_p, o_t)
    assert rel_max(ht_p, ht_t) <= 5e-3, rel_max(ht_p, ht_t)
    # the functional contract: the caller's state must not be mutated
    assert ht_p is not h0 and not torch.equal(ht_p, h0)


def test_hybrid_training_parity(monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN_TRAIN', '1')
    q, k, v, g, beta, h0 = prefill_inputs(1024, 16, 4, seed=3, grad=True)
    gdr_registry._logged.discard(LOG_KEY)
    o_p, ht_p = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0,
                                       output_final_state=True,
                                       use_qk_l2norm_in_kernel=True)
    assert LOG_KEY in gdr_registry._logged, 'the training call did not route to the PLA hybrid'
    (o_p.sum() + ht_p.sum()).backward()
    grads_p = [t.grad.clone() for t in (q, k, v, g, beta)]

    q2, k2, v2, g2, beta2, h02 = prefill_inputs(1024, 16, 4, seed=3, grad=True)
    o_t, ht_t = triton_entry(chunk_gated_delta_rule)(q2, k2, v2, g2, beta2, initial_state=h02,
                                                     output_final_state=True,
                                                     use_qk_l2norm_in_kernel=True)
    (o_t.sum() + ht_t.sum()).backward()
    for gp, gt, name in zip(grads_p, [q2.grad, k2.grad, v2.grad, g2.grad, beta2.grad],
                            ('q', 'k', 'v', 'g', 'beta')):
        assert rel_max(gp, gt) <= 5e-3, (name, rel_max(gp, gt))


def test_hybrid_training_raw_gate_parity(monkeypatch):
    """Raw-gate hybrid: covers gate/beta-sigmoid backward and state/gate-param grads."""
    monkeypatch.setenv('FLA_PLA_GDN_TRAIN', '1')
    kwargs = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True)
    names = ('q', 'k', 'v', 'g', 'beta', 'initial_state', 'A_log', 'dt_bias')

    primals = raw_gate_inputs(1024, 16, 4, seed=11)
    q, k, v, g, beta, h0, A_log, dt_bias = primals
    gdr_registry._logged.discard(LOG_KEY)
    o_p, ht_p = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0,
                                       output_final_state=True, A_log=A_log, dt_bias=dt_bias,
                                       **kwargs)
    assert LOG_KEY in gdr_registry._logged, 'the raw-gate call did not route to the PLA hybrid'
    (o_p.sum() + ht_p.sum()).backward()
    grads_p = [t.grad.clone() for t in primals]

    ref = raw_gate_inputs(1024, 16, 4, seed=11)
    q2, k2, v2, g2, beta2, h02, A_log2, dt_bias2 = ref
    o_t, ht_t = triton_entry(chunk_gated_delta_rule)(q2, k2, v2, g2, beta2, initial_state=h02,
                                                     output_final_state=True, A_log=A_log2,
                                                     dt_bias=dt_bias2, **kwargs)
    (o_t.sum() + ht_t.sum()).backward()
    assert rel_max(o_p, o_t) <= 1e-2, rel_max(o_p, o_t)
    assert rel_max(ht_p, ht_t) <= 1e-2, rel_max(ht_p, ht_t)
    for gp, gt, name in zip(grads_p, [t.grad for t in ref], names):
        assert gp is not None and gt is not None, (name, 'a gradient is missing')
        assert rel_max(gp, gt) <= 5e-3, (name, rel_max(gp, gt))
