# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h import _prune_dhu_configs
from fla.ops.utils import cache as cache_module
from fla.ops.utils.cache import AutotuneKey, FlaCacheMode, fla_cache_autotune
from fla.utils import device


# Multiple configs are required to actually exercise the autotune benchmark path, which is where Triton's default pre_hook
# clones each `restore_value` argument and previously crashed when the argument was None.
@fla_cache_autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
    restore_value=['x'],
)
@triton.jit
def _optional_restore_kernel(x, y, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    if x is not None:
        tl.store(y + idx, tl.load(x + idx, mask=mask), mask=mask)
    else:
        tl.store(y + idx, tl.zeros_like(idx), mask=mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fla_cache_autotune_handles_none_restore_value():
    """A None argument listed in restore_value must not crash autotuning.

    Triton's default pre_hook clones every ``restore_value`` argument during benchmarking;
    if the argument is None this raises ``AttributeError: 'NoneType' object has no attribute 'clone'``.
    CachedAutotuner installs None-safe pre/post hooks to avoid this;
    the test exercises the multi-config (benchmark) path so the pre_hook is actually invoked.
    """
    N = 1024
    y = torch.zeros(N, dtype=torch.int32, device=device)

    # Baseline: restore-value arg is a real tensor — should produce ones.
    x = torch.ones(N, dtype=torch.int32, device=device)
    _optional_restore_kernel[(triton.cdiv(N, 128),)](x, y, N)
    assert torch.equal(y, torch.ones(N, dtype=torch.int32, device=device))

    # Use a different key so the benchmark path runs again, this time with the restore_value arg set to None.
    # Pre-fix this raised AttributeError: 'NoneType' object has no attribute 'clone'.
    M = 2048
    y2 = torch.full((M,), 7, dtype=torch.int32, device=device)
    _optional_restore_kernel[(triton.cdiv(M, 128),)](None, y2, M)
    assert torch.equal(y2, torch.zeros(M, dtype=torch.int32, device=device))


@fla_cache_autotune(
    configs=[triton.Config({'BK': BK}, num_warps=4, num_stages=1) for BK in [32, 64, 128]],
    key=['K'],
    prune_configs_by={'early_config_prune': _prune_dhu_configs},
    validate_cached_config=True,
)
@triton.jit
def _cached_coverage_kernel(y, K: tl.constexpr, BK: tl.constexpr):
    tl.store(y, BK)


@pytest.mark.parametrize('cache_source', ['strict', 'always', 'memory'])
@pytest.mark.parametrize(('K', 'cached_bk'), [(128, 32), (160, 32), (192, 32), (192, 64), (256, 64)])
def test_fla_cache_autotune_validates_cached_config(K, cached_bk, cache_source, monkeypatch):
    """不安全的磁盘或内存配置必须被裁剪，合法缓存配置应保留。"""
    tuner = _cached_coverage_kernel
    cached = triton.Config({'BK': cached_bk}, num_warps=4, num_stages=1)
    config_data = {
        'kwargs': cached.kwargs,
        'num_warps': cached.num_warps,
        'num_stages': cached.num_stages,
        'num_ctas': cached.num_ctas,
    }
    y = torch.zeros(1, dtype=torch.int32, device=device)
    key = AutotuneKey.build(tuner.arg_names, tuner.keys, (y,), {'K': K})
    mode = FlaCacheMode.DISABLED if cache_source == 'memory' else FlaCacheMode(cache_source)
    monkeypatch.setattr(cache_module, 'FLA_CACHE_MODE', mode)
    monkeypatch.setattr(cache_module, 'load_cached_config', lambda *args: config_data)
    monkeypatch.setattr(tuner, 'cache', {key.autotune_key: cached} if cache_source == 'memory' else {})

    # 内存缓存用位置参数调用，同时验证裁剪能取到 K。
    if cache_source == 'memory':
        tuner[(1,)](y, K)
    else:
        tuner[(1,)](y, K=K)
    allowed = _prune_dhu_configs(tuner.configs, {}, K=K)
    assert tuner.best_config in allowed
    assert y.item() == tuner.best_config.kwargs['BK']
    assert tuner.cache[key.autotune_key] == tuner.best_config
    if cached in allowed:
        assert tuner.best_config == cached
    else:
        assert tuner.best_config != cached
