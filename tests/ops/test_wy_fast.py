# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Kernel-level coverage for ``recompute_w_u_fwd``.

The op-level GDN tests only exercise the tile autotune happens to pick for the one shape
they run. The PPU grids carry a 128-wide tile that is masked for D=64, so every candidate
tile is run here against a torch baseline instead of trusting the timing to land safely.
"""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import wy_fast
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.utils import IS_PPU, assert_close, device


def _make_wy_inverse(B: int, T: int, H: int, BT: int, dtype: torch.dtype) -> torch.Tensor:
    """Per-chunk lower-triangular matrix with a unit diagonal, shape ``[B, T, H, BT]``."""
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1
    mask = torch.tril(torch.ones(BT, BT, device=device)).view(1, BT, 1, BT)
    eye = torch.eye(BT, device=device).view(1, BT, 1, BT)
    for it in range(0, T, BT):
        A[:, it:it + BT] = A[:, it:it + BT] * mask + eye
    return A


def _make_gate(B: int, T: int, H: int, BT: int) -> torch.Tensor:
    """Chunk-local cumulative sum of log-sigmoid gates, as the kernel expects."""
    gl = F.logsigmoid(torch.randn(B, T, H, device=device)).float()
    g = gl.clone()
    for it in range(0, T, BT):
        g[:, it:it + BT] = gl[:, it:it + BT].cumsum(1)
    return g


def _recompute_w_u_fwd_ref(k, v, beta, A, g=None):
    B, T, H, K = k.shape
    BT = A.shape[-1]
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    beta_f = beta.float()
    g_f = None if g is None else g.float().exp()
    for it in range(0, T, BT):
        s = slice(it, it + BT)
        A_c = A[:, s].permute(0, 2, 1, 3).float()                                # [B, H, BT, BT]
        v_c = (v[:, s].float() * beta_f[:, s, :, None]).permute(0, 2, 1, 3)      # [B, H, BT, V]
        u[:, s] = (A_c @ v_c).permute(0, 2, 1, 3).to(v.dtype)
        k_c = k[:, s].float() * beta_f[:, s, :, None]
        if g_f is not None:
            k_c = k_c * g_f[:, s, :, None]
        w[:, s] = (A_c @ k_c.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).to(k.dtype)
    return w, u


@pytest.mark.parametrize(
    ('H', 'D', 'varlen', 'use_g'),
    [
        pytest.param(2, 64, False, True, id='H2-D64-dense-gated'),
        pytest.param(1, 128, False, True, id='H1-D128-dense-gated'),
        pytest.param(2, 128, False, False, id='H2-D128-dense-ungated'),
        pytest.param(2, 64, True, True, id='H2-D64-varlen-gated'),
    ],
)
def test_recompute_w_u_fwd_each_config(H: int, D: int, varlen: bool, use_g: bool, monkeypatch):
    torch.manual_seed(42)
    BT, dtype = 64, torch.bfloat16
    if varlen:
        cu_seqlens = torch.LongTensor([0, 128, 320]).to(device)
        B, T = 1, int(cu_seqlens[-1])
    else:
        cu_seqlens, B, T = None, 2, 256
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    A = _make_wy_inverse(B, T, H, BT, dtype)
    g = _make_gate(B, T, H, BT) if use_g else None
    w_ref, u_ref = _recompute_w_u_fwd_ref(k, v, beta, A, g)

    tuner = wy_fast.recompute_w_u_fwd_kernel.fn
    # A non-PPU device keeps one tile, so only the PPU grids have variants worth pinning.
    for config in tuner.configs if IS_PPU else [None]:
        with monkeypatch.context() as patch:
            if config is not None:
                patch.setattr(tuner, 'configs', [config])
                patch.setattr(tuner, 'cache', {})
            w_tri, u_tri = recompute_w_u_fwd(k, v, beta, A, g, cu_seqlens=cu_seqlens)
            if config is not None:
                assert tuner.best_config is config
        assert_close('u', u_ref, u_tri, 0.005)
        assert_close('w', w_ref, w_tri, 0.005)
