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

# Smoke tests for HostLayer with device=cpu — exercises the CPU-only inference
# path end-to-end. On a GPU-enabled host these still pass; the meaningful win is
# that they also work on a driverless system.

import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_export as nve_export
import pynve.nve as nve
import pytest
import tempfile
import torch


def _make_layer(num_embeddings, embed_size, weight, *, storage_kind):
    if storage_kind == "memblock":
        memblock = nve.UserMemBlock(weight.data_ptr())
        layer = nve_layers.NVEmbedding(
            num_embeddings, embed_size, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            storage=memblock,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )
    elif storage_kind == "host_memblock":
        # Owning malloc-backed block; weight_init is copied into it by __init__.
        memblock = nve.HostMemBlock(embed_size, num_embeddings, nve.DataType_t.Float32)
        layer = nve_layers.NVEmbedding(
            num_embeddings, embed_size, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            storage=memblock,
            weight_init=weight,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )
        layer._host_memblock = memblock  # keep alive
    elif storage_kind == "auto":
        layer = nve_layers.NVEmbedding(
            num_embeddings, embed_size, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            weight_init=weight,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )
    else:
        raise ValueError(storage_kind)
    layer._host_weight = weight  # keep alive
    return layer


def test_host_layer_cpu_gather():
    num_embeddings = 1024
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    keys = torch.tensor([0, 5, 17, 256, 1023], dtype=torch.int64)
    out = layer(keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_layer_cpu_auto_storage():
    num_embeddings = 256
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="auto")
    keys = torch.tensor([0, 13, 200, 255], dtype=torch.int64)
    out = layer(keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_layer_cpu_update():
    num_embeddings = 256
    embed_size = 4
    host_weight = torch.zeros(num_embeddings, embed_size, dtype=torch.float32).contiguous()
    layer = _make_layer(num_embeddings, embed_size, host_weight, storage_kind="memblock")
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    updates = torch.arange(1.0, 1.0 + 5 * embed_size, dtype=torch.float32).reshape(5, embed_size)
    layer.update(keys, updates)
    out = layer(keys)
    assert torch.equal(out, updates)


def test_host_layer_cpu_host_memblock_gather():
    # HostMemBlock (owning, malloc-backed) used directly as HostLayer storage.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="host_memblock")
    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    out = layer(keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_memblock_type_tag():
    mb = nve.HostMemBlock(4, 16, nve.DataType_t.Float32)
    assert mb.get_type() == nve.MemBlockType.Host


def test_host_layer_cpu_export_load_roundtrip():
    # Exercises load_nve_layers' CPU path, which allocates a HostMemBlock internally.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings, embed_size, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                weight_init=weight, optimize_for_training=False)

    save_dir = tempfile.mkdtemp()
    nve_export.save_nve(M(), save_dir)
    layers = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))
    assert len(layers) == 1
    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    out = layers[0](keys)
    assert out.device.type == "cpu"
    assert torch.equal(out, weight[keys])


def test_host_layer_cpu_aot_export_load_roundtrip():
    # export_aot + load_aot on a CPU HostLayer: exercises the Python AOT path on
    # a CPU device (device_index=-1, CPU marker constant). Verifies load_aot
    # honors device='cpu' rather than forcing CUDA.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings, embed_size, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                weight_init=weight, optimize_for_training=False)

        def forward(self, keys):
            return self.emb(keys)

    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.export_aot(M(), (keys,), save_dir)
        loader, layers = nve_export.load_aot(save_dir, device=torch.device("cpu"))
        out = loader.run([keys])[0]
        assert out.device.type == "cpu"
        assert torch.equal(out, weight[keys])


def test_host_layer_cpu_backprop_raises():
    # Backprop is GPU/training-only; on a host (device_id < 0) layer the binding
    # must raise a clear error rather than crashing in the CUDA runtime.
    num_embeddings = 256
    embed_size = 4
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    keys = torch.tensor([1, 2, 3], dtype=torch.int64)
    grads = weight[:3].contiguous()
    with pytest.raises(Exception, match="host-only layer"):
        layer.emb_layer.concat_backprop(
            3, keys.data_ptr(), grads.data_ptr(), 0, 0, 0)


def test_host_layer_cpu_rejects_for_non_host_layer_type():
    with pytest.raises(ValueError, match="HostLayer"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.GPULayer,
            device=torch.device("cpu"),
            optimize_for_training=False,
        )


def test_embedding_bag_rejects_host_layer():
    # NVEmbeddingBag has no pooled-lookup path for HostLayer; must fail fast at
    # construction rather than crash at the first forward().
    with pytest.raises(ValueError, match="does not support LayerType.HostLayer"):
        nve_layers.NVEmbeddingBag(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            mode="sum",
            device=torch.device("cpu"),
            optimize_for_training=False,
        )


def test_host_layer_rejects_optimize_for_training():
    # HostLayer is inference-only; constructing with the default
    # optimize_for_training=True must fail fast at __init__, not at backward time.
    with pytest.raises(ValueError, match="inference-only"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            device=torch.device("cpu"),
            # optimize_for_training defaults to True
        )
