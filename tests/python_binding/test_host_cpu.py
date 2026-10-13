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
import subprocess
import sys
import tempfile
import threading
import torch
from conftest import make_host_layer_model, make_layer


def _make_layer(num_embeddings, embed_size, weight, *, storage_kind, bag_mode=None, config=None):
    # CPU HostLayer over `weight`; storage_kind picks how the rows are backed.
    weight_init = None
    if storage_kind == "memblock":
        storage = nve.UserMemBlock(weight.data_ptr(), weight.nbytes)
    elif storage_kind == "host_memblock":
        # Owning malloc-backed block; weight_init is copied into it by __init__.
        storage = nve.HostMemBlock(embed_size, num_embeddings,
                                   nve_layers.torch_type_to_nve_type(weight.dtype))
        weight_init = weight
    elif storage_kind == "auto":
        storage, weight_init = None, weight
    else:
        raise ValueError(storage_kind)
    layer = make_layer(num_embeddings, embed_size, weight.dtype,
                       nve_layers.LayerType.HostLayer, torch.device("cpu"),
                       storage=storage, weight_init=weight_init, bag_mode=bag_mode, config=config)
    layer._host_weight = weight  # keep alive
    layer._host_memblock = storage  # keep alive
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


def test_embedding_bag_rejects_invalid_mode():
    # 'max' (documented unsupported) and typos fail at construction, before any
    # storage is allocated, instead of at the first forward().
    for bad_mode in ("max", "mena"):
        with pytest.raises(ValueError, match="Unsupported pooling mode"):
            nve_layers.NVEmbeddingBag(
                16, 4, torch.float32,
                layer_type=nve_layers.LayerType.HostLayer,
                mode=bad_mode,
                device=torch.device("cpu"),
                optimize_for_training=False,
            )


def test_host_layer_cpu_default_embedding_for_misses():
    # Keys outside the table return the configured default_embedding.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    default = torch.full((embed_size,), -7.0, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock",
                        config={"default_embedding": default})
    out = layer(torch.tensor([7, num_embeddings + 5], dtype=torch.int64))
    assert torch.equal(out[0], weight[7])
    assert torch.equal(out[1], default)


def test_embedding_bag_host_layer_default_embedding_pooled():
    # A missed key contributes the default embedding to its bag (zeros: drops out).
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    bag = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock", bag_mode="sum",
                      config={"default_embedding": torch.zeros(embed_size, dtype=torch.float32)})
    keys = torch.tensor([0, num_embeddings + 100, 5, num_embeddings],
                        dtype=torch.int64)
    out = bag(keys, torch.tensor([0, 2, 4], dtype=torch.int64))
    assert torch.equal(out, torch.stack([weight[0], weight[5]]))


def test_host_layer_cpu_misses_return_zeros_by_default():
    # With no config at all, out-of-range keys return zeros.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    out = layer(torch.tensor([7, num_embeddings + 5], dtype=torch.int64))
    assert torch.equal(out[0], weight[7])
    assert torch.equal(out[1], torch.zeros(embed_size))


def test_host_layer_cpu_default_embedding_opt_out():
    # An explicit empty default disables the miss fill (misses undefined).
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock",
                        config={"default_embedding": []})
    assert len(layer.config.default_embedding) == 0
    # Valid keys still work; missed rows are undefined so only hits are checked.
    out = layer(torch.tensor([7, num_embeddings + 5], dtype=torch.int64))
    assert torch.equal(out[0], weight[7])


def test_default_embedding_rejected_for_unsupported_layer_types():
    # Only HostLayer/Hierarchical consume default_embedding; a non-empty value
    # on other layer types would be a silent no-op, so it fails fast instead.
    with pytest.raises(ValueError, match="only supported for HostLayer"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.GPULayer,
            device=torch.device("cpu"),
            optimize_for_training=False,
            config={"default_embedding": torch.zeros(4, dtype=torch.float32)},
        )


def test_host_layer_cpu_default_embedding_wrong_size_rejected():
    with pytest.raises(ValueError, match="default_embedding"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            device=torch.device("cpu"),
            optimize_for_training=False,
            config={"default_embedding": torch.zeros(5, dtype=torch.float32)},
        )
    # A wrong dtype is rejected even when its byte length happens to match
    # (2 x float64 == 4 x float32 == 16 bytes).
    with pytest.raises(ValueError, match="does not match layer data_type"):
        nve_layers.NVEmbedding(
            16, 4, torch.float32,
            layer_type=nve_layers.LayerType.HostLayer,
            device=torch.device("cpu"),
            optimize_for_training=False,
            config={"default_embedding": torch.zeros(2, dtype=torch.float64)},
        )


def test_host_layer_cpu_default_embedding_export_load():
    # default_embedding round-trips through save_nve/load_nve_layers config json.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    default = torch.full((embed_size,), 3.5, dtype=torch.float32)
    model = make_host_layer_model(num_embeddings, embed_size, weight, torch.device("cpu"),
                                  config={"default_embedding": default})

    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.save_nve(model, save_dir)
        layers = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))
        out = layers[0](torch.tensor([3, num_embeddings + 5], dtype=torch.int64))
        assert torch.equal(out[0], weight[3])
        assert torch.equal(out[1], default)


def test_host_layer_cpu_disable_threadpool_lookup():
    # Inline execution returns the same rows as the threaded default.
    num_embeddings = 256
    embed_size = 8
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    keys = torch.randint(0, num_embeddings, (1000,), dtype=torch.int64)
    threaded = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    inline = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock",
                         config={"disable_threadpool": True})
    assert threaded.config.disable_threadpool is False
    assert inline.config.disable_threadpool is True
    assert torch.equal(inline(keys), threaded(keys))
    assert torch.equal(inline(keys), weight[keys])


def test_disable_threadpool_rejected_for_unsupported_layer_types():
    if not torch.cuda.is_available():
        pytest.skip("GPULayer requires CUDA")
    weight = torch.randn(16, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="disable_threadpool"):
        make_layer(16, 4, torch.float32, nve_layers.LayerType.GPULayer, torch.device("cuda"),
                   storage=nve.UserMemBlock(weight.data_ptr(), weight.nbytes),
                   config={"disable_threadpool": True})


def test_host_layer_cpu_disable_threadpool_aot_config_override():
    # Override the host layer's thread pool setting without changing its export.
    import json
    import os
    num_embeddings = 64
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    model = make_host_layer_model(num_embeddings, embed_size, weight, torch.device("cpu"))

    keys = torch.tensor([3, 5, 63], dtype=torch.int64)
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.export_aot(model, (keys,), save_dir)
        meta_path = os.path.join(save_dir, "metadata.json")
        with open(meta_path) as f:
            doc = json.load(f)
        assert doc["layers"][0]["config"]["disable_threadpool"] is False
        overrides = nve_export.LayerConfigOverrides(
            per_layer={"emb": {"config": {"disable_threadpool": True}}})
        loader, layers = nve_export.load_aot(save_dir, device=torch.device("cpu"),
                                            overrides=overrides)
        assert layers[0].config.disable_threadpool is True
        out = loader.run([keys])[0]
        assert torch.equal(out, weight[keys])
        with open(meta_path) as f:
            assert json.load(f) == doc


def test_host_layer_cpu_export_load_roundtrip():
    # Exercises load_nve_layers' CPU path, which allocates a HostMemBlock internally.
    num_embeddings = 512
    embed_size = 8
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    model = make_host_layer_model(num_embeddings, embed_size, weight, torch.device("cpu"))

    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.save_nve(model, save_dir)
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
    model = make_host_layer_model(num_embeddings, embed_size, weight, torch.device("cpu"))

    keys = torch.tensor([0, 5, 17, 256, 511], dtype=torch.int64)
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.export_aot(model, (keys,), save_dir)
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


@pytest.mark.parametrize("mode,weighted", [
    ("sum", False), ("sum", True), ("mean", False), ("mean", True)])
def test_embedding_bag_host_layer_pooled_lookup(mode, weighted):
    # HostEmbeddingLayer::lookup supports pool_params, so NVEmbeddingBag works
    # end-to-end on a CPU HostLayer for every pooling mode the bag can emit.
    num_embeddings = 64
    embed_size = 4
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight,
                        storage_kind="memblock", bag_mode=mode)

    keys = torch.tensor([0, 1, 2, 5, 10, 20], dtype=torch.int64)
    # Bag boundaries, with a trailing sentinel equal to len(keys)
    # (torch's include_last_offset=True convention).
    offsets = torch.tensor([0, 3, 4, 6], dtype=torch.int64)
    psw = torch.rand(keys.numel(), dtype=torch.float32) if weighted else None
    out = layer(keys, offsets, per_sample_weights=psw)

    if weighted and mode == "mean":
        # torch has no weighted mean; NVE's WeightedMean is sum(w*v)/sum(w) per bag.
        expected = torch.stack(
            [(weight[keys[s:e]] * psw[s:e, None]).sum(dim=0) / psw[s:e].sum()
             for s, e in zip(offsets[:-1], offsets[1:])])
    else:
        expected = torch.nn.functional.embedding_bag(
            keys, weight, offsets, mode=mode, include_last_offset=True,
            per_sample_weights=psw)
    assert out.device.type == "cpu"
    assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_embedding_bag_host_layer_pooled_lookup_fp16():
    num_embeddings = 64
    embed_size = 4
    # Row values are small integers, exactly representable in fp16.
    weight = (torch.arange(num_embeddings, dtype=torch.float16)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight,
                        storage_kind="memblock", bag_mode="sum")
    keys = torch.tensor([0, 1, 2, 5, 10, 20], dtype=torch.int64)
    offsets = torch.tensor([0, 3, 4, 6], dtype=torch.int64)
    out = layer(keys, offsets)
    expected = torch.nn.functional.embedding_bag(
        keys, weight.float(), offsets, mode="sum",
        include_last_offset=True).to(torch.float16)
    assert out.dtype == torch.float16
    assert torch.equal(out, expected)


def test_embedding_bag_host_layer_empty_batch():
    # An all-empty-bags batch returns a zero row per bag, matching
    # torch.nn.EmbeddingBag, instead of raising.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight,
                        storage_kind="memblock", bag_mode="sum")
    out = layer(torch.empty(0, dtype=torch.int64),
                torch.tensor([0, 0, 0], dtype=torch.int64))
    assert torch.equal(out, torch.zeros(2, embed_size))


def test_host_layer_cpu_empty_lookup():
    # Empty key tensor on the non-pooled path returns an empty result.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")
    out = layer(torch.empty(0, dtype=torch.int64))
    assert out.shape == (0, embed_size)


def test_embedding_bag_host_layer_input_validation():
    # forward() rejects inputs the binding would silently misread: the raw
    # data_ptr() handoff is int64-keyed and assumes contiguous fp32/fp16 weights.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight,
                        storage_kind="memblock", bag_mode="sum")
    keys = torch.tensor([0, 1, 2, 5], dtype=torch.int64)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)

    with pytest.raises(TypeError, match="keys must be an int64"):
        layer(keys.to(torch.int32), offsets)
    with pytest.raises(TypeError, match="offsets must be an int64"):
        layer(keys, offsets.to(torch.int32))
    with pytest.raises(ValueError, match="keys must be contiguous"):
        layer(torch.arange(8, dtype=torch.int64)[::2], offsets)
    with pytest.raises(ValueError, match="offsets must be contiguous"):
        layer(keys, torch.tensor([0, 0, 2, 2, 4, 4], dtype=torch.int64)[::2])
    with pytest.raises(TypeError, match="float32 or float16"):
        layer(keys, offsets,
              per_sample_weights=torch.ones(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="one element per key"):
        layer(keys, offsets,
              per_sample_weights=torch.ones(3, dtype=torch.float32))


def test_embedding_bag_host_layer_offsets_validation():
    # offsets must follow the include_last_offset=True convention; malformed
    # boundaries fail loudly instead of dropping bags or reading out of bounds.
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight,
                        storage_kind="memblock", bag_mode="sum")
    keys = torch.tensor([0, 1, 2, 5], dtype=torch.int64)

    with pytest.raises(ValueError, match="at least two entries"):
        layer(keys, torch.tensor([0], dtype=torch.int64))
    with pytest.raises(ValueError, match="at least two entries"):
        layer(keys, torch.empty(0, dtype=torch.int64))
    # torch-default offsets (no trailing sentinel) are rejected, not silently
    # truncated to one bag fewer.
    with pytest.raises(RuntimeError, match="trailing sentinel"):
        layer(keys, torch.tensor([0, 2], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="start at 0"):
        layer(keys, torch.tensor([1, 2, 4], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="non-decreasing"):
        layer(keys, torch.tensor([0, 3, 2, 4], dtype=torch.int64))


def test_torch_ops_input_validation():
    # The C++ ops validate independently of the Python wrappers, guarding
    # runtimes that bypass forward() (torch.export / AOTI).
    num_embeddings = 16
    embed_size = 4
    weight = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")

    with pytest.raises(RuntimeError, match="keys must be an int64"):
        torch.ops.nve_ops.embedding_lookup(
            layer.marker_tensor, torch.tensor([0, 1], dtype=torch.int32),
            layer.embedding_size, layer.dtype_tag)
    with pytest.raises(RuntimeError, match="keys must be contiguous"):
        torch.ops.nve_ops.embedding_lookup(
            layer.marker_tensor, torch.arange(8, dtype=torch.int64)[::2],
            layer.embedding_size, layer.dtype_tag)

    # Same for the bag op's guards (check_index_tensor / check_bag_offsets /
    # check_pooling_weights in the pooling kernel dispatch).
    bag = _make_layer(num_embeddings, embed_size, weight,
                      storage_kind="memblock", bag_mode="sum")
    keys = torch.tensor([0, 1, 2, 5], dtype=torch.int64)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    pool_sum = int(nve.PoolingType_t.Sum)

    def pooled(k, o, w=None):
        return torch.ops.nve_ops.embedding_lookup_with_pooling(
            bag.marker_tensor, k, o, w, pool_sum,
            bag.embedding_size, bag.dtype_tag)

    with pytest.raises(RuntimeError, match="keys must be an int64"):
        pooled(keys.to(torch.int32), offsets)
    with pytest.raises(RuntimeError, match="offsets must be an int64"):
        pooled(keys, offsets.to(torch.int32))
    with pytest.raises(RuntimeError, match="offsets must be contiguous"):
        pooled(keys, torch.tensor([0, 0, 2, 2, 4, 4], dtype=torch.int64)[::2])
    with pytest.raises(RuntimeError, match="at least two entries"):
        pooled(keys, torch.tensor([0], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="float32 or float16"):
        pooled(keys, offsets, torch.ones(4, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="one element per key"):
        pooled(keys, offsets, torch.ones(3, dtype=torch.float32))


def test_host_layer_cpu_concurrent_gather():
    # Regression: the CPU lookup op used to pass stream=0 for every thread, so all
    # worker threads keyed into the *same* execution context and raced on its
    # scratch buffers. get_cpu_stream() now hands each thread a distinct sentinel
    # (address of a thread_local), giving every thread its own context. Many
    # threads gathering disjoint, easily-verifiable rows must all see correct
    # output with no cross-thread corruption.
    num_embeddings = 4096
    embed_size = 16
    # Row i is filled with the value i, so a gathered row immediately reveals
    # whether another thread's lookup clobbered the shared scratch.
    weight = (torch.arange(num_embeddings, dtype=torch.float32)
              .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
    layer = _make_layer(num_embeddings, embed_size, weight, storage_kind="memblock")

    num_threads = 8
    iters_per_thread = 200
    errors = []
    barrier = threading.Barrier(num_threads)

    def worker(tid):
        # Each thread repeatedly gathers its own disjoint slice of rows.
        keys = torch.arange(tid, num_embeddings, num_threads, dtype=torch.int64)
        expected = weight[keys]
        try:
            barrier.wait()  # maximize overlap to provoke any shared-context race
            for _ in range(iters_per_thread):
                out = layer(keys)
                if not torch.equal(out, expected):
                    raise AssertionError(
                        f"thread {tid}: gathered rows did not match expected")
        except Exception as exc:  # noqa: BLE001 - surface in main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent gather failures: {errors}"


def test_host_layer_cpu_concurrent_distinct_layers():
    # Same race surface, but each thread drives its own layer instance to confirm
    # the per-thread context keying is correct even when bindings differ. Values
    # are offset per layer so a leaked context between layers is also caught.
    num_embeddings = 1024
    embed_size = 8
    num_threads = 6

    layers = []
    weights = []
    for tid in range(num_threads):
        w = ((torch.arange(num_embeddings, dtype=torch.float32) + tid * num_embeddings)
             .unsqueeze(1).expand(num_embeddings, embed_size).contiguous())
        weights.append(w)
        layers.append(_make_layer(num_embeddings, embed_size, w, storage_kind="memblock"))

    errors = []
    barrier = threading.Barrier(num_threads)
    keys = torch.tensor([0, 5, 17, 256, 1023], dtype=torch.int64)

    def worker(tid):
        expected = weights[tid][keys]
        try:
            barrier.wait()
            for _ in range(200):
                out = layers[tid](keys)
                if not torch.equal(out, expected):
                    raise AssertionError(
                        f"thread {tid}: layer output corrupted across threads")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent multi-layer failures: {errors}"


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


def test_host_layer_clean_interpreter_exit():
    # A layer alive at interpreter shutdown must not abort the process: on a
    # driver-present machine the CUDA runtime may unload before the layer's
    # destructor runs, and teardown must treat cudaErrorCudartUnloading as
    # benign instead of throwing through the destructor chain (-> terminate).
    script = (
        "import torch\n"
        "import pynve.torch.nve_layers as nve_layers\n"
        "import pynve.nve as nve\n"
        "weight = torch.zeros(16, 4, dtype=torch.float32)\n"
        "memblock = nve.UserMemBlock(weight.data_ptr(), weight.nbytes)\n"
        "layer = nve_layers.NVEmbedding(16, 4, torch.float32,\n"
        "                               layer_type=nve_layers.LayerType.HostLayer,\n"
        "                               storage=memblock,\n"
        "                               device=torch.device('cpu'),\n"
        "                               optimize_for_training=False)\n"
        "layer._w = weight\n"
        "out = layer(torch.tensor([0, 1], dtype=torch.int64))\n"
        "assert out.shape == (2, 4)\n"
    )
    res = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, f"exit code {res.returncode}\n{res.stderr}"
