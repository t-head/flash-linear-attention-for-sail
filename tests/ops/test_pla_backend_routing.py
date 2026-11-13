# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

import fla.ops.backends as be_mod
import fla.ops.gated_delta_rule.backends.pla as pla_mod
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.backends import gdr_registry
from fla.ops.gated_delta_rule.backends.pla import PLAGDNBackend
from fla.utils import device

DEVICE = device


def triton_entry(fn):
    """Unwrap dispatch / torch.compiler.disable layers to the raw Triton fn."""
    while hasattr(fn, '__wrapped__'):
        fn = fn.__wrapped__
    return fn


@pytest.fixture(scope='module')
def pla_backend():
    return gdr_registry._backends['pla']


@pytest.fixture
def routed(pla_backend, monkeypatch, request):
    """Force the PLA backend available and record calls into its impls."""
    # with dispatch off the public functions are undecorated, so forcing
    # is_available cannot produce a PLA route and the routing-hit tests are inert.
    if be_mod._DISPATCH_DISABLED:
        pytest.skip('routing is inert under FLA_DISABLE_BACKEND_DISPATCH=1')
    monkeypatch.setattr(PLAGDNBackend, 'is_available', classmethod(lambda cls: True))
    # keep FlashQLA off so a PLA rejection falls through to the Triton reference
    monkeypatch.setenv('FLA_FLASH_QLA', '0')
    # clear the inherited PLA routing flags so each test sets its own condition
    monkeypatch.delenv('FLA_PLA_GDN', raising=False)
    monkeypatch.delenv('FLA_PLA_GDN_TRAIN', raising=False)
    calls = []

    def make(name, out_shapes):
        def fake(*args, **kwargs):
            calls.append(name)
            q, v = args[0], args[2]
            ht = torch.zeros(q.shape[0], v.shape[2], q.shape[-1], v.shape[-1],
                             device=q.device, dtype=torch.float32) if out_shapes else None
            return torch.zeros(q.shape[0], q.shape[1], v.shape[2], v.shape[-1],
                               device=q.device, dtype=q.dtype), ht
        return fake

    monkeypatch.setattr(pla_backend, 'chunk_gated_delta_rule',
                        make('prefill', request.node.get_closest_marker('stateful') is not None))
    monkeypatch.setattr(pla_backend, 'fused_recurrent_gated_delta_rule', make('decode', False))
    return calls


def gdn_inputs(T=256, H=16, Hg=4, K=128, V=128, seed=0, grad=False, cu=None):
    torch.manual_seed(seed)
    B = 1
    q = torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=grad)
    k = torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=grad)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=torch.bfloat16, requires_grad=grad)
    g = (-torch.rand(B, T, H, device=DEVICE, dtype=torch.float32) * 0.05).requires_grad_(grad)
    beta = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16).sigmoid().requires_grad_(grad)
    return q, k, v, g, beta


def test_prefill_routes_to_pla(routed):
    q, k, v, g, beta = gdn_inputs()
    with torch.no_grad():
        o, ht = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert routed == ['prefill']
    assert o.shape == (1, 256, 16, 128)


def test_prefill_v_first_routes_to_pla(routed):
    # GatedDeltaNet passes state_v_first=True; its inference prefill must still route
    q, k, v, g, beta = gdn_inputs()
    with torch.no_grad():
        o, ht = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True,
                                       state_v_first=True)
    assert routed == ['prefill']
    assert o.shape == (1, 256, 16, 128)


def test_prefill_v_first_training_falls_back(routed):
    # a V-first call that needs grad stays on Triton (the hybrid is K-first only)
    q, k, v, g, beta = gdn_inputs(grad=True)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True,
                                  state_v_first=True)
    assert routed == []
    o.sum().backward()
    assert q.grad is not None


def test_prefill_falls_back_on_head_dim(routed):
    q, k, v, g, beta = gdn_inputs(K=64, V=64)
    with torch.no_grad():
        o, ht = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert routed == []
    ref_o, ref_ht = triton_entry(chunk_gated_delta_rule)(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert torch.equal(o, ref_o)


def test_prefill_falls_back_on_head_config(routed):
    q, k, v, g, beta = gdn_inputs(H=16, Hg=16)
    with torch.no_grad():
        chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert routed == []


def test_prefill_falls_back_for_training(routed):
    q, k, v, g, beta = gdn_inputs(grad=True)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert routed == []
    o.sum().backward()
    assert q.grad is not None and g.grad is not None


def test_prefill_training_optin_routes(routed, monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN_TRAIN', '1')
    q, k, v, g, beta = gdn_inputs(grad=True)
    chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert routed == ['prefill']


def test_env_disable_bypasses_pla(routed, monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN', '0')
    q, k, v, g, beta = gdn_inputs()
    with torch.no_grad():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert routed == []
    ref_o, _ = triton_entry(chunk_gated_delta_rule)(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert torch.equal(o, ref_o)


def decode_inputs(N=16, H=4, HV=8, seed=7):
    torch.manual_seed(seed)
    K = V = 128
    cu = torch.arange(0, N + 1, device=DEVICE, dtype=torch.long)
    a = torch.randn(1, N, HV, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(1, N, HV, device=DEVICE, dtype=torch.bfloat16)
    q = torch.randn(1, N, H, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, N, H, K, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, N, HV, V, device=DEVICE, dtype=torch.bfloat16)
    h0 = torch.randn(N, HV, K, V, device=DEVICE, dtype=torch.float32) * 0.1
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEVICE, dtype=torch.float32)
    return q, k, v, a, b, h0, cu, A_log, dt_bias


def test_decode_routes_to_pla(routed, monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN_DECODE', '1')
    q, k, v, a, b, h0, cu, A_log, dt_bias = decode_inputs()
    with torch.no_grad():
        fused_recurrent_gated_delta_rule(
            q, k, v, g=a, beta=b, initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
            cu_seqlens=cu)
    assert routed == ['decode']


def test_decode_disabled_by_default(routed, monkeypatch):
    # decode is opt-in: without the flag a supported shape must not route to PLA
    monkeypatch.delenv('FLA_PLA_GDN_DECODE', raising=False)
    q, k, v, a, b, h0, cu, A_log, dt_bias = decode_inputs()
    with torch.no_grad():
        fused_recurrent_gated_delta_rule(
            q, k, v, g=a, beta=b, initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
            cu_seqlens=cu)
    assert routed == []


def test_decode_falls_back_without_raw_gating(routed, monkeypatch):
    monkeypatch.setenv('FLA_PLA_GDN_DECODE', '1')
    q, k, v, a, b, h0, cu, A_log, dt_bias = decode_inputs()
    g = -torch.rand(1, 16, 8, device=DEVICE, dtype=torch.float32) * 0.05
    with torch.no_grad():
        o, ht = fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=b.sigmoid(), initial_state=h0,
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    assert routed == []
    ref_o, ref_ht = triton_entry(fused_recurrent_gated_delta_rule)(
        q, k, v, g=g, beta=b.sigmoid(), initial_state=h0,
        output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    assert torch.equal(o, ref_o) and torch.equal(ht, ref_ht)


# --- availability probe: absent pla / broken install must report unavailable ---


def test_is_available_false_when_not_ppu(monkeypatch):
    monkeypatch.setattr(pla_mod, 'IS_PPU', False)
    monkeypatch.setattr(pla_mod, '_PLA_IMPORT_OK', None)
    assert PLAGDNBackend.is_available() is False


def test_is_available_false_when_pla_not_found(monkeypatch):
    # PPU board but pla is not installed at all: find_spec('pla') -> None.
    monkeypatch.setattr(pla_mod, 'IS_PPU', True)
    monkeypatch.setattr(be_mod, 'find_spec_cached', lambda name: None)
    monkeypatch.setattr(pla_mod, '_PLA_IMPORT_OK', None)
    assert PLAGDNBackend.is_available() is False


def test_is_available_false_on_broken_install(monkeypatch):
    # find_spec succeeds (package dir present) but the compiled _C extensions
    # fail to import -> treated as unavailable so dispatch falls back to Triton
    # instead of raising inside impl().
    monkeypatch.setattr(pla_mod, 'IS_PPU', True)
    monkeypatch.setattr(be_mod, 'find_spec_cached', lambda name: object())
    monkeypatch.setattr(pla_mod, '_probe_pla_import', lambda: False)
    monkeypatch.setattr(pla_mod, '_PLA_IMPORT_OK', None)
    assert PLAGDNBackend.is_available() is False


def test_is_available_true_when_importable(monkeypatch):
    monkeypatch.setattr(pla_mod, 'IS_PPU', True)
    monkeypatch.setattr(be_mod, 'find_spec_cached', lambda name: object())
    monkeypatch.setattr(pla_mod, '_probe_pla_import', lambda: True)
    monkeypatch.setattr(pla_mod, '_PLA_IMPORT_OK', None)
    assert PLAGDNBackend.is_available() is True


# --- dispatch fallback: unavailable backend routes to the pure Triton path ---


@pytest.fixture
def pla_unavailable(monkeypatch):
    """Simulate pla missing or a broken install: the backend reports unavailable
    and reaching any PLA impl is a hard failure (proves Triton fallback is taken)."""
    monkeypatch.setattr(PLAGDNBackend, 'is_available', classmethod(lambda cls: False))
    # keep FlashQLA off so the fallback under test is the Triton path
    monkeypatch.setenv('FLA_FLASH_QLA', '0')
    be = gdr_registry._backends['pla']

    def boom(*args, **kwargs):
        raise AssertionError('PLA impl must not run when is_available() is False')

    monkeypatch.setattr(be, 'chunk_gated_delta_rule', boom)
    monkeypatch.setattr(be, 'fused_recurrent_gated_delta_rule', boom)


def test_prefill_falls_back_when_pla_unavailable(pla_unavailable):
    # K=V=128, H=16, Hg=4 is PLA-supported, so it would route to pla if the
    # backend were available; unavailable, it MUST run Triton and match it.
    q, k, v, g, beta = gdn_inputs()
    with torch.no_grad():
        o, ht = chunk_gated_delta_rule(q, k, v, g, beta,
                                       use_qk_l2norm_in_kernel=True, output_final_state=True)
    ref_o, ref_ht = triton_entry(chunk_gated_delta_rule)(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True, output_final_state=True)
    assert torch.equal(o, ref_o) and torch.equal(ht, ref_ht)


def test_decode_falls_back_when_pla_unavailable(pla_unavailable, monkeypatch):
    # decode is off by default; enable it so the only thing between this supported
    # call and the patched PLA impl is the availability guard under test.
    monkeypatch.setenv('FLA_PLA_GDN_DECODE', '1')
    q, k, v, a, b, h0, cu, A_log, dt_bias = decode_inputs()
    with torch.no_grad():
        o, ht = fused_recurrent_gated_delta_rule(
            q, k, v, g=a, beta=b, initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
            cu_seqlens=cu)
    ref_o, ref_ht = triton_entry(fused_recurrent_gated_delta_rule)(
        q, k, v, g=a, beta=b, initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
        cu_seqlens=cu)
    assert torch.equal(o, ref_o) and torch.equal(ht, ref_ht)


# --- verifier guards: unsupported combinations must be rejected (fall to Triton) ---


def test_prefill_verifier_rejects_unsupported():
    be = gdr_registry._backends['pla']
    q, k, v, g, beta = gdn_inputs()
    base = dict(use_qk_l2norm_in_kernel=True)
    accepted = be.chunk_gated_delta_rule_verifier(q, k, v, g, beta, **base)[0]
    assert accepted
    # V-first inference (the GatedDeltaNet path) is accepted; V-first training is not
    assert be.chunk_gated_delta_rule_verifier(
        q, k, v, g, beta, state_v_first=True, **base)[0]
    q.requires_grad_(True)
    assert not be.chunk_gated_delta_rule_verifier(
        q, k, v, g, beta, state_v_first=True, **base)[0]
    q.requires_grad_(False)
    # allow_neg_eigval scales beta to [0, 2), which the PLA forward does not apply
    assert not be.chunk_gated_delta_rule_verifier(
        q, k, v, g, beta, allow_neg_eigval=True, use_beta_sigmoid_in_kernel=True, **base)[0]
    # deprecated layout alias carries a V-first intent the impl would misread
    assert not be.chunk_gated_delta_rule_verifier(
        q, k, v, g, beta, transpose_state_layout=True, **base)[0]
    # raw gate without the beta sigmoid that PLA fuses internally
    assert not be.chunk_gated_delta_rule_verifier(
        q, k, v, g, beta, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        A_log=None, dt_bias=None, **base)[0]
    # packed sequences with an undersized initial-state pool
    cu = torch.tensor([0, 128, 256], device=DEVICE, dtype=torch.long)
    h0 = torch.zeros(1, v.shape[2], 128, 128, device=DEVICE, dtype=torch.float32)
    assert not be.chunk_gated_delta_rule_verifier(
        q, k, v, g, beta, initial_state=h0, cu_seqlens=cu, **base)[0]


def test_decode_verifier_rejects_unsupported(monkeypatch):
    be = gdr_registry._backends['pla']
    q, k, v, a, b, h0, cu, A_log, dt_bias = decode_inputs()

    def verdict(**over):
        kw = dict(g=a, beta=b, initial_state=h0, cu_seqlens=cu,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias)
        kw.update(over)
        return be.fused_recurrent_gated_delta_rule_verifier(q, k, v, **kw)[0]

    # opt-in gate: decode is rejected unless explicitly enabled
    monkeypatch.delenv('FLA_PLA_GDN_DECODE', raising=False)
    assert not verdict()
    monkeypatch.setenv('FLA_PLA_GDN_DECODE', '1')
    assert verdict()
    assert not verdict(dt_bias=None)                    # raw-gate kernel needs dt_bias
    assert not verdict(A_log=None)                      # raw-gate kernel needs A_log
    assert not verdict(state_v_first=True)              # V-first layout is unsupported
    assert not verdict(transpose_state_layout=True)     # deprecated V-first alias
    assert not verdict(initial_state=h0.transpose(-1, -2))  # non-contiguous pool
    q.requires_grad_(True)
    assert not verdict()                                # no PLA backward for grad inputs
    q.requires_grad_(False)
    # head config outside the PLA dispatch table
    assert not be.fused_recurrent_gated_delta_rule_verifier(
        torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16),
        torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16),
        torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16),
        g=torch.randn(1, 16, 16, device=DEVICE, dtype=torch.bfloat16),
        beta=torch.randn(1, 16, 16, device=DEVICE, dtype=torch.bfloat16),
        initial_state=torch.zeros(16, 16, 128, 128, device=DEVICE, dtype=torch.float32),
        cu_seqlens=cu, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        A_log=torch.randn(16, device=DEVICE, dtype=torch.float32),
        dt_bias=torch.randn(16, device=DEVICE, dtype=torch.float32))[0]
