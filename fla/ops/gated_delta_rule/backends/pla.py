# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""PLA backend for GDN prefill and decode on T-Head PPU.

Routes supported shapes to the CUDA kernels of https://github.com/t-head/pla
(v1.0.0); every rejected call falls back to the default Triton path through
the generic dispatch in :mod:`fla.ops.backends`.

- prefill: ``pla.prefill.flashqla.chunk_gated_delta_rule_fwd`` (forward only).
  Training therefore runs a hybrid autograd function (PLA forward, Triton
  backward) and stays opt-in via ``FLA_PLA_GDN_TRAIN=1``.
- decode: ``pla.decode`` k-last kernels, whose state pool is ``[N, HV, V, K]``
  while FLA states are ``[N, HV, K, V]``; the adapter transposes around it.
  Those transpose copies make decode slower than Triton, so it is opt-in via
  ``FLA_PLA_GDN_DECODE=1`` and stays off by default.

The prefill verifier also accepts the V-first state layout that
``fla.layers.GatedDeltaNet`` passes, transposing at the boundary (a one-off
copy amortized over the whole prefill), so the layer's inference prefill routes
to PLA transparently; V-first training stays on Triton and decode is K-first
only.

Disable with ``FLA_PLA_GDN=0``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_PPU, autocast_custom_bwd, autocast_custom_fwd, input_guard

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext

# Mirrors pla.prefill.flashqla.ops.SUPPORTED_HEAD_CONFIGS of PLA v1.0.0,
# as (num_v_heads, num_qk_heads) pairs.
PLA_HEAD_CONFIGS = frozenset({
    (128, 16), (64, 16), (64, 8), (48, 16), (32, 16), (32, 8), (32, 4),
    (24, 8), (16, 8), (16, 4), (16, 2), (12, 4), (8, 4), (8, 2), (6, 2), (4, 2),
})

_STATE_DTYPES = (torch.float32, torch.bfloat16)


def _needs_grad(tensors) -> bool:
    return torch.is_grad_enabled() and any(
        t is not None and torch.is_tensor(t) and t.requires_grad for t in tensors
    )


class _PLAGDNHybridFunction(torch.autograd.Function):
    """PLA forward with the Triton backward of FLA (PLA v1.0.0 ships no backward).

    The backward recomputes the chunk-cumsum gate and the WY factor in FLA
    conventions before entering ``chunk_gated_delta_rule_bwd``, which itself
    recomputes w/u/h from the primals.
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        use_beta_sigmoid_in_kernel: bool,
        use_gate_in_kernel: bool,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        chunk_size: int,
    ):
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
        # PLA derives ht dtype from initial_state; feed it fp32 so ht matches native's fp32.
        h0 = initial_state.float() if initial_state is not None else None
        if use_gate_in_kernel:
            _, _, o, _, ht = chunk_gated_delta_rule_fwd(
                q, k, v, None, None, scale=scale, initial_state=h0,
                cu_seqlens=cu_seqlens, output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
                raw_gate_a=g[0], raw_beta_b=beta[0],
                A_log=A_log.float(), dt_bias=dt_bias.float(),
            )
        else:
            # native activates beta in fp32; widen first so PLA sees the same value
            beta_act = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta
            _, _, o, _, ht = chunk_gated_delta_rule_fwd(
                q, k, v, g.float(), beta_act.float(), scale=scale,
                initial_state=h0, cu_seqlens=cu_seqlens,
                output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
        ctx.save_for_backward(q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), ht  # PLA already returns q.dtype; cast mirrors native's contract

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
        from fla.ops.common.gate import fused_beta_sigmoid_bwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
        from fla.ops.utils import chunk_local_cumsum
        from fla.ops.utils.constant import RCP_LN2

        q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = ctx.saved_tensors
        # native activates beta in fp32; widen first to match the Triton backward
        beta_act = torch.sigmoid(beta.float()) if ctx.use_beta_sigmoid_in_kernel else beta
        if ctx.use_gate_in_kernel:
            g_cum = gdn_gate_chunk_cumsum(
                g=g, A_log=A_log, chunk_size=ctx.chunk_size, scale=RCP_LN2,
                dt_bias=dt_bias, cu_seqlens=cu_seqlens,
            )
        else:
            g_cum = chunk_local_cumsum(
                g, chunk_size=ctx.chunk_size, scale=RCP_LN2, cu_seqlens=cu_seqlens,
            )
        q_in, k_in = q, k
        if ctx.use_qk_l2norm_in_kernel:
            q_norm, q_rstd = l2norm_fwd(q)
            k_norm, k_rstd = l2norm_fwd(k)
        else:
            q_norm, k_norm = q, k
        _, _, A = chunk_gated_delta_rule_fwd_intra(
            k=k_norm, v=v, g=g_cum, beta=beta_act, cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        dq, dk, dv, db, dg, dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
            q=q_norm, k=k_norm, v=v, g=g_cum, beta=beta_act, A=A, scale=ctx.scale,
            initial_state=initial_state, do=do, dht=dht, cu_seqlens=cu_seqlens,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            g_input=g if ctx.use_gate_in_kernel else None,
            A_log=A_log, dt_bias=dt_bias, chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            # l2norm_bwd expects the normalized output (y), matching FLA's own bwd.
            dq = l2norm_bwd(q_norm, q_rstd, dq)
            dk = l2norm_bwd(k_norm, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta, db, scale=1.0)
        return (
            dq.to(q_in), dk.to(k_in), dv.to(v), dg.to(g), db.to(beta),
            None, dh0, None, None, None, None, dA_log, ddt_bias, None, None,
        )


# Cache for the real-import probe below (None = not yet probed). Tests reset it
# via monkeypatch to re-run the probe under simulated conditions.
_PLA_IMPORT_OK: bool | None = None


def _probe_pla_import() -> bool:
    """Return True only if pla's routed entrypoints actually import.

    ``BaseBackend.is_available()`` merely runs ``find_spec('pla')``, which also
    succeeds for a *broken* install (package dir present but the compiled ``_C``
    extensions fail to load). Such an install must fall back to Triton too, so
    probe a real import here; a failed import or extension load means the
    backend is unavailable and dispatch stays on the Triton path.
    """
    try:
        # import the exact routed entrypoints, not just their packages, so a build
        # missing either export reports unavailable instead of raising inside impl.
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last  # noqa: F401
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd  # noqa: F401
        return True
    except (ImportError, OSError, RuntimeError, AttributeError):
        # broken build, missing export, or failed extension load -> unavailable
        return False


class PLAGDNBackend(BaseBackend):
    """CUDA kernels of T-Head PLA for ``chunk_gated_delta_rule`` and decode.

    The verifier encodes the PLA v1.0.0 support matrix; anything outside it
    (head dims, head configs, inter-card CP, state layouts, ...) is left to
    the Triton path by the dispatch loop.
    """

    backend_type = 'pla'
    package_name = 'pla'
    env_var = 'FLA_PLA_GDN'
    default_enable = True
    priority = 2

    @classmethod
    def is_available(cls) -> bool:
        if not IS_PPU:
            return False
        # find_spec('pla') is None -> pla not installed at all -> Triton fallback.
        if not super().is_available():
            return False
        # spec found; ALSO require the compiled pla entrypoints to import, so a
        # broken install falls back to Triton instead of crashing inside impl().
        global _PLA_IMPORT_OK
        if _PLA_IMPORT_OK is None:
            _PLA_IMPORT_OK = _probe_pla_import()
        return _PLA_IMPORT_OK

    def chunk_gated_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False, f'PLA requires fp16/bf16 inputs, got {q.dtype}'
        if not (q.dtype == k.dtype == v.dtype):
            return False, f'PLA requires q/k/v of one dtype, got {q.dtype}/{k.dtype}/{v.dtype}'
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            return False, f'PLA requires K=V=128, got K={q.shape[-1]}, V={v.shape[-1]}'
        if (v.shape[2], k.shape[2]) not in PLA_HEAD_CONFIGS:
            return False, f'head config (H={v.shape[2]}, Hg={k.shape[2]}) is outside the PLA dispatch table'
        if q.shape[2] != k.shape[2]:
            return False, f'PLA requires matching q/k head counts, got {q.shape[2]}/{k.shape[2]}'
        if q.shape[0] != 1:
            return False, 'PLA prefill takes packed [1, T, ...] inputs only'
        if cp_context is not None:
            return False, 'inter-card context parallel is not supported by PLA'
        # dispatch runs before the native entry point folds the deprecated
        # transpose_state_layout alias into state_v_first, so reject it here rather
        # than silently reading a V-first state as K-first.
        if kwargs.get('transpose_state_layout'):
            return False, 'PLA does not accept the deprecated transpose_state_layout alias'
        # dispatch also bypasses the native head_first/chunk_size validation, so reject
        # them here and let the native entry point raise its explicit errors.
        if 'head_first' in kwargs:
            return False, 'PLA does not accept the removed head_first option'
        if kwargs.get('chunk_size', 64) not in (16, 32, 64):
            return False, 'PLA requires chunk_size in (16, 32, 64)'
        if allow_neg_eigval:
            return False, 'PLA prefill does not scale beta for allow_neg_eigval'
        needs_grad = _needs_grad(
            (q, k, v, g, beta, initial_state, kwargs.get('A_log'), kwargs.get('dt_bias')))
        # the hybrid training function threads a K-first state only, so a V-first
        # call that needs grad stays on Triton (also the faster training path).
        if state_v_first and needs_grad:
            return False, 'PLA V-first prefill is inference-only; training stays on Triton'
        if initial_state is not None and (
            initial_state.dtype not in _STATE_DTYPES or not initial_state.is_contiguous()
        ):
            return False, 'PLA needs a contiguous fp32/bf16 initial state'
        # dispatch bypasses the native packed-sequence checks, so enforce them here.
        if (
            cu_seqlens is not None and initial_state is not None
            and initial_state.shape[0] != len(cu_seqlens) - 1
        ):
            return False, 'PLA needs one initial state per packed sequence'
        # PLA fuses the raw gate and the beta sigmoid and needs both gate parameters;
        # unsupported combinations fall back to Triton instead of crashing in impl.
        if kwargs.get('use_gate_in_kernel') and not (
            use_beta_sigmoid_in_kernel
            and kwargs.get('A_log') is not None
            and kwargs.get('dt_bias') is not None
        ):
            return False, 'PLA raw-gate prefill needs beta sigmoid plus explicit A_log and dt_bias'
        if needs_grad and os.environ.get('FLA_PLA_GDN_TRAIN', '0') != '1':
            return False, 'training needs FLA_PLA_GDN_TRAIN=1 (hybrid PLA fwd + Triton bwd)'
        return True, None

    @input_guard
    def chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        use_gate_in_kernel = kwargs.get('use_gate_in_kernel', False)
        A_log = kwargs.get('A_log')
        dt_bias = kwargs.get('dt_bias')
        chunk_size = kwargs.get('chunk_size', 64)
        if scale is None:
            scale = k.shape[-1] ** -0.5
        if _needs_grad((q, k, v, g, beta, initial_state, A_log, dt_bias)):
            return _PLAGDNHybridFunction.apply(
                q, k, v, g, beta, scale, initial_state, output_final_state,
                use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel,
                use_gate_in_kernel, A_log, dt_bias, cu_seqlens, chunk_size,
            )
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
        # PLA derives ht dtype from initial_state; feed fp32 (V-first transposed) to match native.
        h0 = initial_state
        if h0 is not None:
            if state_v_first:
                h0 = h0.transpose(-1, -2)
            h0 = h0.float().contiguous()
        if use_gate_in_kernel:
            _, _, o, _, ht = chunk_gated_delta_rule_fwd(
                q, k, v, None, None, scale=scale, initial_state=h0,
                cu_seqlens=cu_seqlens, output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
                raw_gate_a=g[0], raw_beta_b=beta[0],
                A_log=A_log.float(), dt_bias=dt_bias.float(),
            )
        else:
            # native activates beta in fp32; widen first so PLA sees the same value
            beta_act = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta
            _, _, o, _, ht = chunk_gated_delta_rule_fwd(
                q, k, v, g.float(), beta_act.float(), scale=scale,
                initial_state=h0, cu_seqlens=cu_seqlens,
                output_final_state=output_final_state,
                normalize_qk=use_qk_l2norm_in_kernel,
            )
        if state_v_first and ht is not None:
            ht = ht.transpose(-1, -2).contiguous()
        return o.to(q.dtype), ht  # PLA already returns q.dtype; cast mirrors native's contract

    def fused_recurrent_gated_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        # decode is opt-in: the adapter's state transpose copies make it slower
        # than Triton, so keep it off unless the user explicitly enables it.
        if os.environ.get('FLA_PLA_GDN_DECODE', '0') != '1':
            return False, 'PLA decode needs FLA_PLA_GDN_DECODE=1 (opt-in)'
        if gk is not None or gv is not None:
            return False, 'PLA decode supports a scalar gate only'
        if g is None or beta is None:
            return False, 'PLA decode requires g and beta'
        if beta.dim() != 3:
            return False, 'PLA decode takes one beta scalar per head, i.e. [B, T, HV]'
        if not (use_gate_in_kernel and use_beta_sigmoid_in_kernel):
            return False, 'PLA decode fuses the raw gating and beta sigmoid paths only'
        # the raw-gate kernel needs both gate parameters; a bias-free or A_log-free
        # call would crash in impl, so leave it to the native bias-free path.
        if A_log is None or dt_bias is None:
            return False, 'PLA decode requires A_log and dt_bias'
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False, f'PLA requires fp16/bf16 inputs, got {q.dtype}'
        if not (q.dtype == k.dtype == v.dtype == g.dtype == beta.dtype):
            return False, 'PLA decode requires q/k/v/g/beta of one activation dtype'
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            return False, f'PLA requires K=V=128, got K={q.shape[-1]}, V={v.shape[-1]}'
        if (v.shape[2], k.shape[2]) not in PLA_HEAD_CONFIGS:
            return False, f'head config (H={v.shape[2]}, Hg={k.shape[2]}) is outside the PLA dispatch table'
        # dispatch runs before the native entry point folds the deprecated
        # transpose_state_layout alias into state_v_first, so reject it here.
        if kwargs.get('transpose_state_layout'):
            return False, 'PLA does not accept the deprecated transpose_state_layout alias'
        if state_v_first or allow_neg_eigval:
            return False, 'PLA supports neither state_v_first nor allow_neg_eigval'
        if initial_state is None:
            return False, 'PLA decode updates an existing state pool only'
        # PLA updates the pool in place, so require a contiguous state; the impl then
        # copies it into an independent fp32 pool that never aliases the caller's.
        if initial_state.dtype not in _STATE_DTYPES or not initial_state.is_contiguous():
            return False, f'PLA needs a contiguous fp32/bf16 state pool, got {initial_state.dtype}'
        if cu_seqlens is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            return False, 'PLA needs one initial state per packed sequence'
        # the PLA decode kernel has no backward, so keep gradient-requiring inputs on
        # the native path, which raises its explicit unsupported-backward error.
        if _needs_grad((q, k, v, g, beta, initial_state, A_log, dt_bias)):
            return False, 'PLA decode has no backward; run it under no_grad'
        if q.shape[0] != 1 and not (cu_seqlens is None and q.shape[1] == 1):
            return False, 'PLA decode takes packed [1, N, ...] or [N, 1, ...] inputs'
        return True, None

    @input_guard
    def fused_recurrent_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ):
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
        B, T = q.shape[:2]
        if B != 1:
            q, k, v, g, beta = (t.reshape(1, B * T, *t.shape[2:]) for t in (q, k, v, g, beta))
            # each of the packed sequences spans T steps
            cu = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * T
        elif cu_seqlens is not None:
            cu = cu_seqlens.to(torch.int32).contiguous()
        else:
            # device-side offsets: a Python-list tensor would force a host sync and
            # block CUDA graph capture on every decode call.
            cu = torch.arange(0, 2, dtype=torch.int32, device=q.device) * T
        if scale is None:
            scale = q.shape[-1] ** -0.5
        N = cu.numel() - 1
        # The k-last pool is [N, HV, V, K] and PLA updates it in place, so build an
        # independent fp32 copy: the caller's state stays untouched and the returned
        # state is fp32, matching the native entry point.
        pool = initial_state.transpose(-1, -2).float().contiguous()
        indices = torch.arange(N, dtype=torch.int32, device=q.device)
        # the k-last kernel takes dt_bias in the activation dtype, so the native fp32
        # bias is rounded here; this is an inherent PLA decode parity limitation.
        o = fused_sigmoid_gating_delta_rule_forward_k_last(
            A_log, g, dt_bias.to(q.dtype), 1.0, 20.0, q, k, v, beta,
            pool, indices, float(scale), use_qk_l2norm_in_kernel,
            cu, False, False, None, None, None, None,
        )
        ht = pool.transpose(-1, -2).contiguous() if output_final_state else None
        return o.reshape(B, T, v.shape[2], v.shape[3]), ht
