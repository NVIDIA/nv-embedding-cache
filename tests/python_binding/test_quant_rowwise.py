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

# Rowwise-quantized storage through the Python layers. Expected values always come
# from decoding the stored bytes (dequantize_rowwise), never from the float source.

import os
import tempfile
import numpy as np
import pytest
import torch
import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from pynve.torch.nve_dtypes import quantize_rowwise, dequantize_rowwise
from conftest import make_layer, nvhm_available

LT = nve_layers.LayerType
QTYPES = [nve.DataType_t.QInt8RowwiseF32, nve.DataType_t.QInt8RowwiseF16,
          nve.DataType_t.QUint8RowwiseF32, nve.DataType_t.QUint8RowwiseF16]
QIDS = ["qint8_f32", "qint8_f16", "quint8_f32", "quint8_f16"]
NUM_EMB = 256
DIM = 16
GPU_CACHE = 1 << 20

has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not has_cuda, reason="CUDA is not available")
requires_nvhm = pytest.mark.skipif(not nvhm_available(), reason="NVHM plugin is not available")

# (id, layer_type, device, host_cache_size); every config except the CPU HostLayer needs CUDA.
LAYER_CONFIGS = [
    pytest.param(LT.GPULayer, "cuda", 0, id="gpu", marks=requires_cuda),
    pytest.param(LT.LinearUVM, "cuda", 0, id="uvm", marks=requires_cuda),
    pytest.param(LT.Hierarchical, "cuda", 0, id="hier", marks=[requires_cuda, requires_nvhm]),
    pytest.param(LT.Hierarchical, "cuda", GPU_CACHE, id="hier_host_cache",
                 marks=[requires_cuda, requires_nvhm]),
    pytest.param(LT.HostLayer, "cuda", 0, id="host_cuda", marks=requires_cuda),
    pytest.param(LT.HostLayer, "cpu", 0, id="host_cpu"),
]


def _tol(qtype):
    out = nve.quant_rowwise_output_dtype(qtype)
    return dict(rtol=1e-3, atol=1e-2) if out == nve.DataType_t.Float16 else dict(rtol=1e-5, atol=1e-4)


def _make_raw(qtype, num_rows=NUM_EMB, dim=DIM, seed=0):
    """Raw rows with distinct per-row scales, negative values and nonzero offsets."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_rows, dim, generator=g) * torch.linspace(0.1, 4.0, num_rows)[:, None]
    x = x + torch.linspace(-3.0, 3.0, num_rows)[:, None]
    return quantize_rowwise(x, qtype)


def _make_ps(qtype, raw):
    init = nve_ps.SimpleInitializer(raw.shape[0], DIM, qtype, raw)
    return nve_ps.NVEParameterServer(0, DIM, qtype, init)


def _build(layer_type, device, host_cache, qtype, raw, *, bag_mode=None, config=None):
    kwargs = dict(bag_mode=bag_mode, config=config)
    if layer_type == LT.Hierarchical:
        layer = make_layer(NUM_EMB, DIM, qtype, layer_type, torch.device(device),
                           storage=_make_ps(qtype, raw), gpu_cache_size=GPU_CACHE,
                           host_cache_size=host_cache, **kwargs)
    else:
        gpu_cache = GPU_CACHE if layer_type == LT.LinearUVM else 0
        layer = make_layer(NUM_EMB, DIM, qtype, layer_type, torch.device(device),
                           weight_init=raw, gpu_cache_size=gpu_cache, **kwargs)
    return layer


# --------------------------------------------------------------------------- helpers

@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
def test_rowwise_quantize_roundtrip(qtype):
    x = torch.randn(64, DIM) * 3 + 1
    x[0] = 0.0          # all-zero row
    x[1] = 2.5          # constant row
    raw = quantize_rowwise(x, qtype)
    assert raw.dtype == torch.uint8
    assert raw.shape == (64, nve.row_size_in_bytes(qtype, DIM))
    out = dequantize_rowwise(raw, qtype)
    assert out.dtype == nve_layers.nve_type_to_torch_type(nve.quant_rowwise_output_dtype(qtype))
    meta = raw[:, DIM:].contiguous().view(out.dtype).float()
    half_step = meta[:, 0:1] / 2
    err = (out.float() - x).abs()
    assert torch.all(err <= half_step * 1.01 + x.abs() * 2e-3 + 1e-6)
    assert torch.all(out[0] == 0)


@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("dim", [1, 6, 7])
@pytest.mark.parametrize("row_index", [0, 1])
def test_rowwise_dequantize_single_row(qtype, dim, row_index):
    raw = quantize_rowwise(torch.arange(2 * dim, dtype=torch.float32).reshape(2, dim), qtype)
    expected = dequantize_rowwise(raw, qtype)[row_index:row_index + 1]
    actual = dequantize_rowwise(raw[row_index:row_index + 1], qtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("qtype", [nve.DataType_t.QInt8RowwiseF16, nve.DataType_t.QUint8RowwiseF16],
                         ids=["qint8_f16", "quint8_f16"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_rowwise_quantize_fp16_scale_underflow(qtype, device):
    x = torch.tensor([[0.0, 0.0], [1e-6, -1e-6], [1e-7, -1e-7]], device=device)
    raw = quantize_rowwise(x, qtype)
    actual = dequantize_rowwise(raw, qtype)
    # The smallest fp16 subnormal is enough to preserve these rows within one output ULP.
    torch.testing.assert_close(actual, x.half(), rtol=0, atol=2**-24)
    assert torch.all(actual[0] == 0)
    assert torch.all(actual[1:, 0] > 0)
    assert torch.all(actual[1:, 1] < 0)


def _pooling_reference(rows, offsets, mode, weights, dtype):
    bags = []
    for b in range(offsets.numel() - 1):
        lo, hi = int(offsets[b]), int(offsets[b + 1])
        w = weights[lo:hi, None] if weights is not None else torch.ones(hi - lo, 1)
        acc = (rows[lo:hi] * w).sum(0)
        if mode == "mean":
            denom = float(w.sum())
            acc = acc / denom if denom != 0 else torch.zeros_like(acc)
        bags.append(acc)
    return torch.stack(bags).to(dtype)


# --------------------------------------------------------------------------- lookups

@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("layer_type,device,host_cache", LAYER_CONFIGS)
def test_rowwise_lookup(layer_type, device, host_cache, qtype):
    raw = _make_raw(qtype)
    layer = _build(layer_type, device, host_cache, qtype, raw)
    assert layer.storage_data_type == qtype
    assert layer.data_type == nve_layers.nve_type_to_torch_type(nve.quant_rowwise_output_dtype(qtype))
    keys = torch.tensor([0, 3, 3, 17, 255, 128, 64], dtype=torch.int64, device=device)
    expected = dequantize_rowwise(raw[keys.cpu()], qtype)
    for _ in range(2):  # cold misses, then warm cache hits
        out = layer(keys)
        assert out.dtype == layer.data_type
        assert out.shape == (keys.numel(), DIM)
        torch.testing.assert_close(out.cpu(), expected, **_tol(qtype))


@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("mode", ["sum", "mean", "concat"])
@pytest.mark.parametrize("layer_type,device,host_cache", LAYER_CONFIGS)
def test_rowwise_bag_pooling(layer_type, device, host_cache, qtype, mode):
    raw = _make_raw(qtype)
    layer = _build(layer_type, device, host_cache, qtype, raw, bag_mode=mode)
    # Repeated keys and an empty bag (offsets 3..3).
    keys = torch.tensor([1, 2, 2, 9, 40, 40, 40, 200], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 3, 5, 8], dtype=torch.int64, device=device)
    rows = dequantize_rowwise(raw[keys.cpu()], qtype).float()
    if mode == "concat":
        out = layer(keys)
        torch.testing.assert_close(out.cpu(), rows.to(out.dtype), **_tol(qtype))
        return

    out = layer(keys, offsets)
    assert out.shape == (offsets.numel() - 1, DIM)
    torch.testing.assert_close(out.cpu(), _pooling_reference(rows, offsets, mode, None, out.dtype), **_tol(qtype))
    for wdtype in (torch.float32, torch.float16):
        # The last bag's weights sum to zero, which yields a zero row for weighted mean.
        w = torch.tensor([0.5, 1.0, 2.0, 0.25, 1.0, -1.0, 0.5, -0.5], dtype=wdtype)
        out_w = layer(keys, offsets, w.to(device))
        torch.testing.assert_close(out_w.cpu(), _pooling_reference(rows, offsets, mode, w.float(), out_w.dtype), **_tol(qtype))


@pytest.mark.skipif(not nve_layers.HAS_TORCH_OPS, reason="torch custom ops not built")
@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("default_mode", ["sum", "mean", "concat"])
@pytest.mark.parametrize("layer_type,device,host_cache", LAYER_CONFIGS)
def test_rowwise_bag_mode_override(layer_type, device, host_cache, qtype, default_mode):
    raw = _make_raw(qtype)
    layer = _build(layer_type, device, host_cache, qtype, raw, bag_mode=default_mode)
    keys = torch.tensor([1, 2, 2, 9, 40, 40, 40, 200], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 3, 5, 8], dtype=torch.int64, device=device)
    rows = dequantize_rowwise(raw[keys.cpu()], qtype).float()
    cached_weight = layer.weight
    for _ in range(2):  # cold lookups followed by cache hits
        for override in ["concat", "mean", "sum", None]:
            mode = default_mode if override is None else override
            if mode == "concat":
                out = layer(keys, mode_override=override)
                torch.testing.assert_close(out.cpu(), rows.to(out.dtype), **_tol(qtype))
            else:
                for wdtype in [None, torch.float32, torch.float16]:
                    weights = None if wdtype is None else torch.tensor(
                        [0.5, 1.0, 2.0, 0.25, 1.0, -1.0, 0.5, -0.5], dtype=wdtype)
                    out = layer(keys, offsets, None if weights is None else weights.to(device),
                                mode_override=override)
                    expected = _pooling_reference(rows, offsets, mode, weights, out.dtype)
                    torch.testing.assert_close(out.cpu(), expected, **_tol(qtype))
            assert layer.mode == default_mode
            assert layer.weight is cached_weight
            assert not out.requires_grad  # quantized storage supports inference only


# --------------------------------------------------------------------------- writes and misses

@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("layer_type,device,host_cache", [
    p for p in LAYER_CONFIGS if p.id in ("gpu", "uvm", "hier", "hier_host_cache", "host_cpu")])
def test_rowwise_update_and_insert_raw_rows(layer_type, device, host_cache, qtype):
    raw = _make_raw(qtype)
    layer = _build(layer_type, device, host_cache, qtype, raw)
    keys = torch.tensor([5, 6, 7], dtype=torch.int64, device=device)
    layer(keys)  # pull the rows into the caches first
    new_rows = _make_raw(qtype, num_rows=3, seed=1)
    layer.update(keys, new_rows.to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(layer(keys).cpu(), dequantize_rowwise(new_rows, qtype), **_tol(qtype))
    if layer_type == LT.Hierarchical:
        # Insert into the GPU cache tier only; subsequent lookups hit it.
        ins_keys = torch.tensor([20, 21], dtype=torch.int64, device=device)
        ins_rows = _make_raw(qtype, num_rows=2, seed=2)
        layer.insert(ins_keys, ins_rows.to(device), 0)
        torch.cuda.synchronize()
        torch.testing.assert_close(layer(ins_keys).cpu(), dequantize_rowwise(ins_rows, qtype),
                                   **_tol(qtype))


@requires_cuda
@requires_nvhm
@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
def test_rowwise_hierarchical_miss_returns_zeros(qtype):
    raw = _make_raw(qtype, num_rows=16)
    ps = _make_ps(qtype, raw)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.Hierarchical, torch.device("cuda"),
                       storage=ps, gpu_cache_size=GPU_CACHE)
    out = layer(torch.tensor([3, 100], dtype=torch.int64, device="cuda"))
    torch.testing.assert_close(out[0].cpu(), dequantize_rowwise(raw[3:4], qtype)[0], **_tol(qtype))
    assert torch.all(out[1] == 0)


@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("code_dtype", [torch.uint8, torch.int8])
def test_rowwise_custom_default_embedding(qtype, code_dtype):
    raw = _make_raw(qtype)
    default_row = _make_raw(qtype, num_rows=1, seed=3)[0]
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw,
                       config={"default_embedding": default_row.view(code_dtype)})
    out = layer(torch.tensor([1, NUM_EMB + 5], dtype=torch.int64))
    torch.testing.assert_close(out[1], dequantize_rowwise(default_row[None], qtype)[0], **_tol(qtype))


@requires_cuda
@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("layer_type", [LT.GPULayer, LT.LinearUVM], ids=["gpu", "uvm"])
def test_rowwise_default_row_index(layer_type, qtype):
    raw = _make_raw(qtype)
    layer = make_layer(NUM_EMB, DIM, qtype, layer_type, torch.device("cuda"), weight_init=raw,
                       gpu_cache_size=GPU_CACHE if layer_type == LT.LinearUVM else 0,
                       config={"default_row_index": 3})
    out = layer(torch.tensor([7, NUM_EMB + 10], dtype=torch.int64, device="cuda"))
    torch.testing.assert_close(out[1].cpu(), dequantize_rowwise(raw[3:4], qtype)[0], **_tol(qtype))


# --------------------------------------------------------------------------- storage

@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
def test_rowwise_weight_is_raw_bytes(qtype):
    raw = _make_raw(qtype)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw)
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (NUM_EMB, layer.row_bytes)
    assert not layer.weight.requires_grad
    assert torch.equal(layer.weight.detach(), raw)


@requires_cuda
@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
def test_rowwise_uvm_byte_view_and_managed_memblock(qtype):
    raw = _make_raw(qtype)
    memblock = nve.ManagedMemBlock(DIM, NUM_EMB, qtype, [torch.cuda.current_device()])
    assert memblock.get_size_in_bytes() == NUM_EMB * nve.row_size_in_bytes(qtype, DIM)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.LinearUVM, torch.device("cuda"), storage=memblock,
                       weight_init=raw, gpu_cache_size=GPU_CACHE)
    assert layer.weight.dtype == torch.uint8
    assert torch.equal(layer.weight.detach().cpu(), raw)
    keys = torch.arange(NUM_EMB, dtype=torch.int64, device="cuda")
    torch.testing.assert_close(layer(keys).cpu(), dequantize_rowwise(raw, qtype), **_tol(qtype))


@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
def test_rowwise_host_memblock_storage(qtype):
    raw = _make_raw(qtype)
    memblock = nve.HostMemBlock(DIM, NUM_EMB, qtype)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), storage=memblock,
                       weight_init=raw)
    keys = torch.tensor([0, 9, 255], dtype=torch.int64)
    torch.testing.assert_close(layer(keys), dequantize_rowwise(raw[keys], qtype), **_tol(qtype))


@requires_cuda
@requires_nvhm
@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
def test_rowwise_ps_load_from_file(qtype):
    raw = _make_raw(qtype)
    with tempfile.TemporaryDirectory() as tmp:
        keys_path, values_path = os.path.join(tmp, "keys.npy"), os.path.join(tmp, "values.npy")
        np.save(keys_path, np.arange(NUM_EMB, dtype=np.int64))
        np.save(values_path, raw.numpy())
        ps = nve_ps.NVEParameterServer(0, DIM, qtype)
        ps.load_from_file(keys_path, values_path)
    assert ps.num_keys() == NUM_EMB
    layer = make_layer(NUM_EMB, DIM, qtype, LT.Hierarchical, torch.device("cuda"), storage=ps,
                       gpu_cache_size=GPU_CACHE)
    keys = torch.tensor([0, 77, 255], dtype=torch.int64, device="cuda")
    torch.testing.assert_close(layer(keys).cpu(), dequantize_rowwise(raw[keys.cpu()], qtype),
                               **_tol(qtype))


@requires_cuda
@requires_nvhm
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_rowwise_multi_device_shares_ps():
    qtype = nve.DataType_t.QUint8RowwiseF16
    raw = _make_raw(qtype)
    ps = _make_ps(qtype, raw)
    keys = torch.tensor([4, 8, 15], dtype=torch.int64)
    expected = dequantize_rowwise(raw[keys], qtype)
    for dev in (0, 1):
        device = torch.device("cuda", dev)
        layer = make_layer(NUM_EMB, DIM, qtype, LT.Hierarchical, device, storage=ps,
                           gpu_cache_size=GPU_CACHE)
        torch.testing.assert_close(layer(keys.to(device)).cpu(), expected, **_tol(qtype))


# --------------------------------------------------------------------------- rejections

def test_rowwise_rejects_training():
    with pytest.raises(ValueError, match="inference-only"):
        nve_layers.NVEmbedding(NUM_EMB, DIM, nve.DataType_t.QInt8RowwiseF32, LT.HostLayer,
                               optimize_for_training=True, device=torch.device("cpu"))


@pytest.mark.parametrize("layer_type", [
    pytest.param(LT.GPULayer, id="gpu", marks=requires_cuda),
    pytest.param(LT.LinearUVM, id="uvm", marks=requires_cuda),
    pytest.param(LT.Hierarchical, id="hier", marks=[requires_cuda, requires_nvhm]),
])
def test_rowwise_rejects_odd_width_on_gpu_layers(layer_type):
    qtype = nve.DataType_t.QInt8RowwiseF16
    storage = nve_ps.NVEParameterServer(0, 7, qtype) if layer_type == LT.Hierarchical else None
    # Rejected by the C++ GPU table/layer row-width checks.
    with pytest.raises(RuntimeError, match="must be even|must divide by 2"):
        make_layer(NUM_EMB, 7, qtype, layer_type, torch.device("cuda"), storage=storage,
                   gpu_cache_size=GPU_CACHE)


def test_rowwise_host_layer_accepts_odd_width():
    qtype = nve.DataType_t.QInt8RowwiseF16
    raw = quantize_rowwise(torch.randn(NUM_EMB, 7), qtype)
    layer = make_layer(NUM_EMB, 7, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw)
    keys = torch.tensor([2, 3], dtype=torch.int64)
    torch.testing.assert_close(layer(keys), dequantize_rowwise(raw[keys], qtype), **_tol(qtype))


@pytest.mark.parametrize("bad", ["float", "width", "non_contiguous"])
def test_rowwise_rejects_bad_rows(bad):
    qtype = nve.DataType_t.QUint8RowwiseF32
    raw = _make_raw(qtype)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw)
    keys = torch.tensor([1, 2], dtype=torch.int64)
    rows = {
        "float": torch.zeros(2, DIM),
        "width": torch.zeros(2, DIM, dtype=torch.uint8),
        "non_contiguous": torch.zeros(layer.row_bytes, 2, dtype=torch.uint8).t(),
    }[bad]
    with pytest.raises((TypeError, ValueError)):
        layer.update(keys, rows)
    with pytest.raises((TypeError, ValueError)):
        layer.insert(keys, rows, 0)
    with pytest.raises((TypeError, ValueError)):
        make_layer(2, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=rows)


def test_rowwise_update_accepts_extra_rows():
    # Same rule as float layers: values may hold more rows than keys.
    qtype = nve.DataType_t.QInt8RowwiseF32
    raw = _make_raw(qtype)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw)
    keys = torch.tensor([1, 2], dtype=torch.int64)
    new_rows = _make_raw(qtype, num_rows=4, seed=1)
    layer.update(keys, new_rows)
    torch.testing.assert_close(layer(keys), dequantize_rowwise(new_rows[:2], qtype), **_tol(qtype))
    with pytest.raises(ValueError, match="at least 2 rows"):
        layer.update(keys, new_rows[:1])


@pytest.mark.parametrize("qtype", [nve.DataType_t.QInt8RowwiseF16, nve.DataType_t.QUint8RowwiseF16],
                         ids=["qint8_f16", "quint8_f16"])
def test_quantize_rowwise_rejects_meta_overflow(qtype):
    x = torch.tensor([[1.0, -2.0], [1e9, -1e9]])
    with pytest.raises(ValueError, match=r"Rows \[1\]"):
        quantize_rowwise(x, qtype)
    # Scale fits in fp16 but decoded values don't.
    x = torch.tensor([[1.0, -2.0], [1e5, -1e4]])
    with pytest.raises(ValueError, match=r"Rows \[1\].*decoded values overflow"):
        quantize_rowwise(x, qtype)


def test_quantize_rowwise_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        quantize_rowwise(torch.tensor([[1.0, float("inf")]]), nve.DataType_t.QInt8RowwiseF32)


def test_rowwise_rejects_small_memblock():
    qtype = nve.DataType_t.QInt8RowwiseF32
    backing = torch.zeros(NUM_EMB, DIM, dtype=torch.uint8)  # values only, no room for scales
    memblock = nve.UserMemBlock(backing.data_ptr(), backing.nbytes)
    with pytest.raises(RuntimeError, match="too small"):
        make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), storage=memblock)


@requires_nvhm
def test_rowwise_rejects_ps_mismatch():
    qtype = nve.DataType_t.QUint8RowwiseF16
    float_ps = nve_ps.NVEParameterServer(0, DIM, torch.float16)
    with pytest.raises(ValueError, match="doesn't match"):
        make_layer(NUM_EMB, DIM, qtype, LT.Hierarchical, torch.device("cuda"), storage=float_ps,
                   gpu_cache_size=GPU_CACHE)
    # Raw tables are checked by the binding, including ones whose rows have the same byte width:
    # QUint8RowwiseF32 with 8 values (8 + 8 bytes) vs float32 with 4 values (16 bytes).
    same_bytes_table = nve.ParameterServerTable(num_rows=0, row_elements=4,
                                                data_type=nve.DataType_t.Float32)
    with pytest.raises(RuntimeError, match="doesn't match"):
        make_layer(NUM_EMB, 8, nve.DataType_t.QUint8RowwiseF32, LT.Hierarchical,
                   torch.device("cuda"), storage=same_bytes_table, gpu_cache_size=GPU_CACHE)
    narrow_table = nve.ParameterServerTable(num_rows=0, row_elements=DIM - 2, data_type=qtype)
    with pytest.raises(RuntimeError, match="same row size"):
        make_layer(NUM_EMB, DIM, qtype, LT.Hierarchical, torch.device("cuda"), storage=narrow_table,
                   gpu_cache_size=GPU_CACHE)
    # A plugin table without 'value_dtype' is untyped, which quantized layers reject.
    untyped_ps = nve_ps.NVEParameterServer(
        NUM_EMB, DIM, qtype, plugin_name="libnve-plugin-nvhm.so", factory_config={},
        table_config={"key_size": 8, "max_value_size": nve.row_size_in_bytes(qtype, DIM)})
    with pytest.raises(RuntimeError, match="declare its value type"):
        make_layer(NUM_EMB, DIM, qtype, LT.Hierarchical, torch.device("cuda"), storage=untyped_ps,
                   gpu_cache_size=GPU_CACHE)


@requires_nvhm
def test_ps_load_from_file_checks_element_type():
    # Each file matches the table's row bytes but not its element type.
    cases = [
        (torch.float32, np.zeros((4, 2 * DIM), dtype=np.float16)),
        (torch.float16, np.zeros((4, 2 * DIM), dtype=np.uint8)),
        (nve.DataType_t.QInt8RowwiseF32, np.zeros((4, (DIM + 4) // 4), dtype=np.float32)),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        keys_path, values_path = os.path.join(tmp, "keys.npy"), os.path.join(tmp, "values.npy")
        np.save(keys_path, np.arange(4, dtype=np.int64))
        for dtype, values in cases:
            np.save(values_path, values)
            ps = nve_ps.NVEParameterServer(0, DIM, dtype)
            with pytest.raises(RuntimeError, match="element type"):
                ps.load_from_file(keys_path, values_path)


@requires_nvhm
def test_rowwise_ps_insert_checks_dtype():
    keys = torch.arange(4, dtype=torch.int64)
    quant_ps = nve_ps.NVEParameterServer(0, DIM, nve.DataType_t.QInt8RowwiseF32)
    with pytest.raises(TypeError):
        quant_ps.insert(keys, torch.zeros(4, DIM))
    float_ps = nve_ps.NVEParameterServer(0, DIM, torch.float32)
    with pytest.raises(TypeError):
        float_ps.insert(keys, torch.zeros(4, DIM, dtype=torch.float16))


@requires_nvhm
def test_layer_data_type_deprecated_alias():
    """layer_data_type is kept as a deprecated alias of storage_data_type."""
    qtype = nve.DataType_t.QInt8RowwiseF32
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"))
    ps = nve_ps.NVEParameterServer(0, DIM, qtype)
    for obj in (layer, ps):
        with pytest.warns(DeprecationWarning, match="storage_data_type"):
            assert obj.layer_data_type == qtype


def test_rowwise_simple_initializer_requires_rows():
    with pytest.raises(ValueError, match="init_tensor"):
        nve_ps.SimpleInitializer(4, DIM, nve.DataType_t.QUint8RowwiseF32)


def test_rowwise_binding_rejects_unsupported_dtype():
    memblock = nve.HostMemBlock(DIM, 4, nve.DataType_t.Float64)
    with pytest.raises(RuntimeError, match="Unsupported layer data type"):
        nve.HostEmbedding(DIM, 4, nve.DataType_t.Float64, memblock, -1, nve.EmbedLayerConfig())


# --------------------------------------------------------------------------- dispatch

@pytest.mark.parametrize("qtype", QTYPES, ids=QIDS)
@pytest.mark.parametrize("default_mode", ["sum", "mean", "concat"])
def test_rowwise_bag_mode_override_fallback(monkeypatch, qtype, default_mode):
    raw = _make_raw(qtype)
    bag = _build(LT.HostLayer, "cpu", 0, qtype, raw, bag_mode=default_mode)
    monkeypatch.setattr(nve_layers, "HAS_TORCH_OPS", False)
    keys = torch.tensor([1, 2, 2, 9, 40], dtype=torch.int64)
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int64)
    rows = dequantize_rowwise(raw[keys], qtype).float()
    cached_weight = bag.weight

    for override in ["concat", "sum", "mean", None]:
        mode = default_mode if override is None else override
        kwargs = {} if override is None else {"mode_override": override}
        if mode == "concat":
            out = bag(keys, **kwargs)
            torch.testing.assert_close(out, rows.to(out.dtype), **_tol(qtype))
        else:
            for weights in [None, torch.tensor([0.5, 1.0, 2.0, 0.25, 1.0])]:
                out = bag(keys, offsets, per_sample_weights=weights, **kwargs)
                expected = _pooling_reference(rows, offsets, mode, weights, out.dtype)
                torch.testing.assert_close(out, expected, **_tol(qtype))
        assert bag.mode == default_mode
        assert bag.weight is cached_weight
        assert not out.requires_grad


@pytest.mark.parametrize("qtype", [nve.DataType_t.QInt8RowwiseF32, nve.DataType_t.QUint8RowwiseF16],
                         ids=["qint8_f32", "quint8_f16"])
def test_rowwise_legacy_ops(monkeypatch, qtype):
    """HAS_TORCH_OPS=False routes through the pybind fallback ops."""
    raw = _make_raw(qtype)
    emb = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw)
    bag = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw,
                     bag_mode="sum")
    monkeypatch.setattr(nve_layers, "HAS_TORCH_OPS", False)
    keys = torch.tensor([3, 4, 5], dtype=torch.int64)
    offsets = torch.tensor([0, 1, 3], dtype=torch.int64)
    rows = dequantize_rowwise(raw[keys], qtype)
    out = emb(keys)
    assert out.dtype == emb.data_type and out.shape == (3, DIM)
    torch.testing.assert_close(out, rows, **_tol(qtype))
    out_bag = bag(keys, offsets)
    expected_bag = torch.stack([rows[:1].float().sum(0), rows[1:].float().sum(0)]).to(out.dtype)
    torch.testing.assert_close(out_bag, expected_bag, **_tol(qtype))
    assert not out.requires_grad  # packed weights carry no gradient


@pytest.mark.skipif(not nve_layers.HAS_TORCH_OPS, reason="torch custom ops not built")
@pytest.mark.parametrize("qtype", [nve.DataType_t.QInt8RowwiseF16, nve.DataType_t.QUint8RowwiseF32],
                         ids=["qint8_f16", "quint8_f32"])
def test_rowwise_fake_matches_eager(qtype):
    raw = _make_raw(qtype)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.HostLayer, torch.device("cpu"), weight_init=raw)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = layer

        def forward(self, keys):
            return self.emb(keys)

    keys = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    batch = torch.export.Dim("batch", min=2, max=1024)
    ep = torch.export.export(M(), (keys,), dynamic_shapes={"keys": {0: batch}})
    fake_out = ep.graph_module.graph.find_nodes(op="output")[0].args[0][0].meta["val"]
    eager = M()(keys)
    assert fake_out.dtype == eager.dtype
    assert fake_out.shape[1] == eager.shape[1] == DIM


@requires_cuda
@pytest.mark.parametrize("qtype", [nve.DataType_t.QUint8RowwiseF32, nve.DataType_t.QUint8RowwiseF16],
                         ids=["quint8_f32", "quint8_f16"])
def test_rowwise_fbgemm_layout(qtype):
    """FBGEMM fused 8-bit rowwise tables feed straight into weight_init."""
    pytest.importorskip("fbgemm_gpu")
    x = torch.randn(NUM_EMB, DIM, device="cuda")
    if qtype == nve.DataType_t.QUint8RowwiseF32:
        raw = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(x)
        ref = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(raw)
    else:
        raw = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(x, 8)
        ref = torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(raw, 8)
    assert raw.shape[1] == nve.row_size_in_bytes(qtype, DIM)
    layer = make_layer(NUM_EMB, DIM, qtype, LT.GPULayer, torch.device("cuda"), weight_init=raw)
    keys = torch.tensor([0, 31, 255], dtype=torch.int64, device="cuda")
    torch.testing.assert_close(layer(keys).float(), ref[keys].float(), **_tol(qtype))
