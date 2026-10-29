# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Route selection between the fused fwd_h+chunk_o kernel and the fwd_h + chunk_o pair.

`fused_ho_pays_off` encodes a timing frontier measured on ZW_M890P over the 35 shapes
pinned in SWEEP (H 4..256, B 1..8, T 2k..32k, D 64/128, MHA and GQA), best config
against best config, arm-alternated medians, +-0.5% noise, BV=64. Nothing routed to
fused measured slower than +0.5%; outside the frontier the pair wins by up to 18%.
Keeping the sweep in the test lets the frontier be re-reviewed without re-running it.
"""

import pytest
import torch
import torch.nn.functional as F

import fla.ops.gated_delta_rule.chunk as chunk_mod
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd, fused_ho_pays_off
from fla.utils import assert_close, device

# (label, B, T, HV, D, fused-vs-pair delta in %, route)
SWEEP = [
    ('s_t2k', 1, 2048, 96, 128, 2.17, 'pair'),
    ('35b_t4k', 1, 4096, 32, 128, -0.02, 'pair'),
    ('ho_gqa_4_32_t4k', 1, 4096, 32, 128, -0.50, 'pair'),
    ('122b_t4k', 1, 4096, 96, 128, 4.73, 'pair'),
    ('ho_b3_h40', 3, 4096, 40, 128, 8.29, 'pair'),
    ('ho_b2_h96_t4k', 2, 4096, 96, 128, 2.17, 'pair'),
    ('h64_b4_t4k', 4, 4096, 64, 128, -3.19, 'fused'),
    ('ho_h80_t6k', 1, 6144, 80, 128, 7.48, 'pair'),
    ('p_h8', 1, 8192, 8, 128, 17.41, 'pair'),
    ('d64_t8k', 1, 8192, 32, 64, 1.66, 'pair'),
    ('p_h16', 1, 8192, 16, 128, 8.90, 'pair'),
    ('gqa_4_16', 1, 8192, 16, 128, 9.14, 'pair'),
    ('35b_t8k', 1, 8192, 32, 128, -0.30, 'pair'),
    ('p_h32', 1, 8192, 32, 128, -1.00, 'pair'),
    ('b_b2', 2, 8192, 16, 128, -1.80, 'pair'),
    ('gqa_8_32', 1, 8192, 32, 128, -0.15, 'pair'),
    ('p_h48', 1, 8192, 48, 128, 17.80, 'pair'),
    ('p_h64', 1, 8192, 64, 128, 7.43, 'pair'),
    ('b_b4', 4, 8192, 16, 128, 9.66, 'pair'),
    ('gqa_16_64', 1, 8192, 64, 128, 5.16, 'pair'),
    ('ho_gqa_8_64', 1, 8192, 64, 128, 7.00, 'pair'),
    ('ho_d64_h64_b2', 2, 8192, 64, 64, -2.13, 'pair'),
    ('122b_t8k', 1, 8192, 96, 128, 1.59, 'pair'),
    ('p_h128', 1, 8192, 128, 128, 1.32, 'pair'),
    ('b_b8', 8, 8192, 16, 128, 5.72, 'pair'),
    ('ho_b6_h24', 6, 8192, 24, 128, 5.48, 'pair'),
    ('p_h192', 1, 8192, 192, 128, -3.08, 'fused'),
    ('p_h256', 1, 8192, 256, 128, -3.00, 'fused'),
    ('ho_h112_t10k', 1, 10240, 112, 128, -5.26, 'fused'),
    ('ho_h48_t12k', 1, 12288, 48, 128, 16.02, 'pair'),
    ('35b_t16k', 1, 16384, 32, 128, -3.33, 'fused'),
    ('h16_b2_t16k', 2, 16384, 16, 128, -0.55, 'fused'),
    ('s_t16k', 1, 16384, 96, 128, -5.20, 'fused'),
    ('ho_h96_t24k', 1, 24576, 96, 128, -6.95, 'fused'),
    ('s_t32k', 1, 32768, 96, 128, -7.60, 'fused'),
]


def _route(B, T, HV, D, chunk_size=64, cu_seqlens=None):
    # the rule only reads shapes, so meta tensors are enough
    k = torch.empty(B, T, HV, D, device='meta')
    v = torch.empty(B, T, HV, D, device='meta')
    return fused_ho_pays_off(k, v, cu_seqlens, chunk_size=chunk_size)


@pytest.mark.parametrize(('label', 'B', 'T', 'HV', 'D', 'delta', 'route'),
                         [pytest.param(*row, id=row[0]) for row in SWEEP])
def test_route_reproduces_sweep(label, B, T, HV, D, delta, route):
    assert _route(B, T, HV, D) == (route == 'fused'), f'{label} measured {delta:+.2f}%'


@pytest.mark.parametrize(('B', 'T', 'HV', 'D', 'chunk_size', 'route'), [
    (1, 16384, 1, 64, 64, True),  # NT 256 pays off on its own
    (1, 16320, 1, 64, 64, False),  # NT 255, grid 1
    (1, 10240, 192, 64, 64, True),  # NT 160 needs grid >= 192
    (1, 10240, 191, 64, 64, False),
    (1, 10176, 192, 64, 64, False),  # NT 159
    (1, 8192, 384, 64, 64, True),  # NT 128 needs grid >= 384
    (1, 8192, 383, 64, 64, False),
    (1, 8128, 384, 64, 64, False),  # NT 127
    (1, 4096, 512, 64, 64, True),  # NT 64 needs grid >= 512
    (1, 4096, 511, 64, 64, False),
    (1, 4032, 512, 64, 64, False),  # NT 63
    (1, 8192, 96, 128, 64, False),  # mainline 122b: NT 128, grid 192
    (1, 8192, 192, 128, 64, True),  # same NT, V=128 doubles the grid
    (1, 8192, 192, 128, 128, False),  # chunk 128 halves NT to 64
])
def test_route_boundaries(B, T, HV, D, chunk_size, route):
    assert _route(B, T, HV, D, chunk_size=chunk_size) == route


def test_packed_sequences_keep_fused():
    cu_seqlens = torch.tensor([0, 1024, 2048])
    assert not _route(1, 2048, 4, 64)
    assert _route(1, 2048, 4, 64, cu_seqlens=cu_seqlens)


def _inputs(B, T, HV, D):
    torch.manual_seed(0)
    return dict(
        q=F.normalize(torch.randn(B, T, HV, D, device=device), p=2, dim=-1),
        k=F.normalize(torch.randn(B, T, HV, D, device=device), p=2, dim=-1),
        v=torch.randn(B, T, HV, D, device=device),
        beta=torch.rand(B, T, HV, device=device).sigmoid(),
        g=F.logsigmoid(torch.rand(B, T, HV, device=device)),
        scale=D ** -0.5,
        initial_state=None,
        output_final_state=True,
    )


def _spy_fused(monkeypatch):
    calls = []
    fused = chunk_mod.chunk_gated_delta_rule_fwd_h_o
    monkeypatch.setattr(chunk_mod, 'chunk_gated_delta_rule_fwd_h_o',
                        lambda **kw: calls.append(1) or fused(**kw))
    return calls


def test_fused_route_matches_pair(monkeypatch):
    inputs = _inputs(1, 16384, 2, 64)  # NT 256
    assert _route(1, 16384, 2, 64)
    calls = _spy_fused(monkeypatch)
    _, o_fused, _, ht_fused, _, _ = chunk_gated_delta_rule_fwd(**inputs)
    assert len(calls) == 1

    monkeypatch.setenv('FLA_PPU_DISABLE_FUSED_HO', '1')
    _, o_pair, _, ht_pair, _, _ = chunk_gated_delta_rule_fwd(**inputs)
    assert len(calls) == 1

    assert_close('o', o_pair, o_fused, 0.002)
    assert_close('ht', ht_pair, ht_fused, 0.002)


def test_disable_env_forces_pair(monkeypatch):
    monkeypatch.setenv('FLA_PPU_DISABLE_FUSED_HO', '1')
    calls = _spy_fused(monkeypatch)
    chunk_gated_delta_rule_fwd(**_inputs(1, 16384, 2, 64))
    assert not calls


def test_wide_head_dim_stays_pair(monkeypatch):
    assert _route(1, 16384, 1, 192)
    calls = _spy_fused(monkeypatch)
    chunk_gated_delta_rule_fwd(**_inputs(1, 16384, 1, 192))
    assert not calls


def test_state_v_first_stays_pair(monkeypatch):
    calls = _spy_fused(monkeypatch)
    chunk_gated_delta_rule_fwd(**_inputs(1, 16384, 2, 64), state_v_first=True)
    assert not calls
