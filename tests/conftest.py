#!/usr/bin/python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest


def is_nvhm_plugin_available():
    """Check if the nvhm plugin is available by attempting to load it."""
    try:
        import torch
        from pynve.torch import nve_ps
        # Try to create a minimal NVEParameterServer - this loads the nvhm plugin
        ps = nve_ps.NVEParameterServer(0, 1, torch.float32)
        del ps
        return True
    except Exception:
        return False


# Cache the result to avoid repeated checks
_nvhm_available = None


def nvhm_available():
    global _nvhm_available
    if _nvhm_available is None:
        _nvhm_available = is_nvhm_plugin_available()
    return _nvhm_available


# Pytest marker for tests requiring NVHM
requires_nvhm = pytest.mark.skipif(
    not nvhm_available(),
    reason="NVHM plugin is not available"
)


def make_layer(num_embeddings, embedding_size, dtype, layer_type, device, *,
               storage=None, weight_init=None, gpu_cache_size=0, host_cache_size=0,
               bag_mode=None, config=None, optimize_for_training=False):
    """Layer of any LayerType, for inference unless optimize_for_training=True.

    bag_mode selects an NVEmbeddingBag with that pooling mode (forward(keys, offsets));
    None gives an NVEmbedding (forward(keys)). Backward needs optimize_for_training=True.
    """
    import pynve.torch.nve_layers as nve_layers

    if bag_mode is None:
        layer_cls, extra = nve_layers.NVEmbedding, {}
    else:
        layer_cls, extra = nve_layers.NVEmbeddingBag, {"mode": bag_mode}
    return layer_cls(
        num_embeddings, embedding_size, dtype,
        layer_type=layer_type,
        storage=storage,
        weight_init=weight_init,
        gpu_cache_size=gpu_cache_size,
        host_cache_size=host_cache_size,
        optimize_for_training=optimize_for_training,
        device=device,
        config=config,
        **extra,
    )


def make_nvhm_ps(weight):
    """NVHashMap parameter server holding row i of `weight` under key i.

    Built with num_embeddings=0 so eviction is disabled and every row stays resident.
    """
    from pynve.torch import nve_ps

    num_rows, row_size = weight.shape
    init = nve_ps.SimpleInitializer(num_rows, row_size, weight.dtype, weight)
    return nve_ps.NVEParameterServer(0, row_size, weight.dtype, init)


def make_host_layer_model(num_embeddings, embedding_size, weight, device, *,
                          bag_mode=None, config=None):
    """nn.Module wrapping a single inference HostLayer seeded from `weight`.

    bag_mode selects an NVEmbeddingBag (forward(keys, offsets)); None gives an
    NVEmbedding (forward(keys)). Shared by the CPU and CUDA export round trips.
    """
    import torch
    import pynve.torch.nve_layers as nve_layers

    class HostLayerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = make_layer(num_embeddings, embedding_size, weight.dtype,
                                  nve_layers.LayerType.HostLayer, device,
                                  weight_init=weight, bag_mode=bag_mode, config=config)

        def forward(self, keys, offsets=None):
            if bag_mode is None:
                return self.emb(keys)
            return self.emb(keys, offsets)

    return HostLayerModel()
