# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl


@triton.jit
def group_heads_sum_kernel(
    x,
    out,
    n_rows,
    G: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    i_r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m_r = i_r < n_rows
    o_k = tl.arange(0, K)
    b_acc = tl.zeros([BLOCK_R, K], dtype=tl.float32)
    for i_g in tl.static_range(G):
        p_x = x + (i_r * G + i_g)[:, None] * K + o_k[None, :]
        b_acc += tl.load(p_x, mask=m_r[:, None], other=0.0)
    tl.store(
        out + i_r[:, None] * K + o_k[None, :],
        b_acc.to(out.dtype.element_ty),
        mask=m_r[:, None],
    )


def sum_group_heads(x: torch.Tensor, group: int) -> torch.Tensor:
    """(B, T, H * group, K) -> (B, T, H, K) by summing each `group` consecutive head slots.

    The torch equivalent `x.view(B, T, H, group, K).sum(3)` reduces along a
    non-innermost axis of a 5-D view, which on the bwd shapes moves 302 MB at
    0.59 TB/s on the ZW-M890P and 1.3 TB/s on H200. This kernel reads the
    `group` slices of one output row back to back and accumulates in fp32:
    2.4 TB/s / 4.2 TB/s on the same shapes, bit-identical to torch.
    """
    B, T, HV, K = x.shape
    if group == 1:
        return x
    H = HV // group
    n_rows = B * T * H
    if n_rows == 0:
        return x.new_empty(B, T, H, K)
    if not x.is_contiguous() or K & (K - 1):
        return x.reshape(B, T, H, group, K).sum(3)
    out = torch.empty(B, T, H, K, dtype=x.dtype, device=x.device)
    block_r = max(1, 1024 // K)
    group_heads_sum_kernel[(triton.cdiv(n_rows, block_r),)](
        x, out, n_rows, G=group, K=K, BLOCK_R=block_r, num_warps=4,
    )
    return out
