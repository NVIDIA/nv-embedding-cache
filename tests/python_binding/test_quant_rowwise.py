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

# Rowwise-quantized storage through the torch layers: Hierarchical layers over a parameter
# server holding raw quantized rows ([codes][scale][offset]) return dequantized rows.

import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers
import pytest
import torch
from conftest import make_layer, requires_nvhm
from pynve.torch import nve_dtypes, nve_ps

NUM_EMBEDDINGS = 256
EMBED_SIZE = 16
CACHE_SIZE = 4 * 1024 * 1024
MISS_KEY = 10 * NUM_EMBEDDINGS

QUANT_TYPES = [
    nve.DataType_t.QUint8RowwiseF16,
    nve.DataType_t.QUint8RowwiseF32,
    nve.DataType_t.QInt8RowwiseF16,
    nve.DataType_t.QInt8RowwiseF32,
]
QUANT_IDS = ["quint8_f16", "quint8_f32", "qint8_f16", "qint8_f32"]


def _is_affine(qtype):
    return qtype in (nve.DataType_t.QUint8RowwiseF16, nve.DataType_t.QUint8RowwiseF32)


def _quantize(weight, qtype):
    """Quantize float rows to stored rows; returns (rows uint8 [n, row_bytes], dequantized fp32)."""
    param_type = nve_dtypes.output_torch_type(qtype)
    if _is_affine(qtype):
        offset = weight.min(dim=1, keepdim=True).values.to(param_type)
        span = weight.max(dim=1, keepdim=True).values - offset.float()
        scale = torch.where(span > 0, span / 255.0, torch.ones_like(span)).to(param_type)
        codes = ((weight - offset.float()) / scale.float()).round().clamp(0, 255).to(torch.uint8)
        params = [scale, offset]
        dequant = codes.float() * scale.float() + offset.float()
    else:
        amax = weight.abs().max(dim=1, keepdim=True).values
        scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).to(param_type)
        codes = (weight / scale.float()).round().clamp(-127, 127).to(torch.int8)
        params = [scale]
        dequant = codes.float() * scale.float()
    rows = torch.cat([codes.view(torch.uint8)] + [p.contiguous().view(torch.uint8) for p in params], dim=1)
    assert rows.shape[1] == nve_dtypes.stored_row_bytes(EMBED_SIZE, qtype)
    return rows.contiguous(), dequant


def _make(qtype, *, bag_mode=None, config=None, device=torch.device("cuda")):
    weight = torch.randn(NUM_EMBEDDINGS, EMBED_SIZE)
    rows, dequant = _quantize(weight, qtype)
    ps = nve_ps.NVEParameterServer(0, EMBED_SIZE, qtype)
    ps.insert(torch.arange(NUM_EMBEDDINGS, dtype=torch.int64), rows)
    layer = make_layer(NUM_EMBEDDINGS, EMBED_SIZE, qtype, nve_layers.LayerType.Hierarchical,
                       device, storage=ps, gpu_cache_size=CACHE_SIZE, bag_mode=bag_mode,
                       config=config)
    return layer, rows, dequant


def _tol(qtype):
    if nve_dtypes.output_torch_type(qtype) == torch.float16:
        return dict(rtol=1e-3, atol=1e-2)
    return dict(rtol=1e-5, atol=1e-5)


@requires_nvhm
@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=QUANT_IDS)
def test_quant_lookup_dequantizes(qtype):
    layer, _, dequant = _make(qtype)
    assert layer.data_type == nve_dtypes.output_torch_type(qtype)
    keys = torch.tensor([3, 0, 255, 3, 17], dtype=torch.int64, device="cuda")
    # Two rounds: the first resolves from the parameter server, the second from the GPU cache.
    for _ in range(2):
        out = layer(keys)
        assert out.dtype == layer.data_type
        assert out.shape == (keys.numel(), EMBED_SIZE)
        torch.testing.assert_close(out.float().cpu(), dequant[keys.cpu()], **_tol(qtype))


@requires_nvhm
@pytest.mark.parametrize("mode", ["sum", "mean"])
@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=QUANT_IDS)
def test_quant_pooling(qtype, mode):
    bag, _, dequant = _make(qtype, bag_mode=mode)
    keys = torch.tensor([1, 2, 3, 9, MISS_KEY, 40], dtype=torch.int64, device="cuda")
    offsets = torch.tensor([0, 3, 3, 6], dtype=torch.int64, device="cuda")
    zeros = torch.zeros(EMBED_SIZE)
    rows = [dequant[1], dequant[2], dequant[3], dequant[9], zeros, dequant[40]]
    bags = [torch.stack(rows[0:3]), torch.zeros(0, EMBED_SIZE), torch.stack(rows[3:6])]
    reduce = (lambda b: b.sum(0)) if mode == "sum" else (lambda b: b.mean(0) if len(b) else b.sum(0))
    expected = torch.stack([reduce(b) for b in bags])
    for _ in range(2):
        out = bag(keys, offsets)
        assert out.dtype == bag.data_type
        torch.testing.assert_close(out.float().cpu(), expected, **_tol(qtype))


@requires_nvhm
@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=QUANT_IDS)
def test_quant_misses_return_zeros_by_default(qtype):
    layer, _, dequant = _make(qtype)
    assert len(layer.config.default_embedding) == nve_dtypes.stored_row_bytes(EMBED_SIZE, qtype)
    keys = torch.tensor([5, MISS_KEY], dtype=torch.int64, device="cuda")
    for _ in range(2):
        out = layer(keys).float().cpu()
        torch.testing.assert_close(out[0], dequant[5], **_tol(qtype))
        assert torch.equal(out[1], torch.zeros(EMBED_SIZE))


@requires_nvhm
@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=QUANT_IDS)
def test_quant_custom_default_embedding(qtype):
    # The default row is given as raw stored bytes and dequantized like any other row.
    default_rows, default_dequant = _quantize(torch.randn(1, EMBED_SIZE), qtype)
    layer, _, dequant = _make(qtype, config={"default_embedding": default_rows[0]})
    keys = torch.tensor([MISS_KEY, 7], dtype=torch.int64, device="cuda")
    expected = torch.stack([default_dequant[0], dequant[7]])
    for _ in range(2):
        out = layer(keys)
        torch.testing.assert_close(out.float().cpu(), expected, **_tol(qtype))


@requires_nvhm
def test_quant_multi_device_shares_parameter_server():
    # Layers on different GPUs read the same parameter server, each through its own cache.
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    qtype = nve.DataType_t.QUint8RowwiseF16
    rows, dequant = _quantize(torch.randn(NUM_EMBEDDINGS, EMBED_SIZE), qtype)
    ps = nve_ps.NVEParameterServer(0, EMBED_SIZE, qtype)
    ps.insert(torch.arange(NUM_EMBEDDINGS, dtype=torch.int64), rows)
    keys = torch.tensor([3, 100, MISS_KEY], dtype=torch.int64)
    expected = torch.stack([dequant[3], dequant[100], torch.zeros(EMBED_SIZE)])
    for device_index in range(2):
        device = torch.device("cuda", device_index)
        layer = make_layer(NUM_EMBEDDINGS, EMBED_SIZE, qtype, nve_layers.LayerType.Hierarchical,
                           device, storage=ps, gpu_cache_size=CACHE_SIZE)
        for _ in range(2):
            out = layer(keys.to(device))
            assert out.device == device
            torch.testing.assert_close(out.float().cpu(), expected, **_tol(qtype))


@requires_nvhm
def test_quant_validation():
    qtype = nve.DataType_t.QUint8RowwiseF16
    ps = nve_ps.NVEParameterServer(0, EMBED_SIZE, qtype)
    with pytest.raises(ValueError, match="only supported for Hierarchical"):
        make_layer(NUM_EMBEDDINGS, EMBED_SIZE, qtype, nve_layers.LayerType.LinearUVM,
                   torch.device("cuda"), gpu_cache_size=CACHE_SIZE)
    with pytest.raises(ValueError, match="inference-only"):
        nve_layers.NVEmbedding(NUM_EMBEDDINGS, EMBED_SIZE, qtype, nve_layers.LayerType.Hierarchical,
                               gpu_cache_size=CACHE_SIZE, storage=ps, optimize_for_training=True)
    with pytest.raises(ValueError, match="does not match layer data_type"):
        make_layer(NUM_EMBEDDINGS, EMBED_SIZE, qtype, nve_layers.LayerType.Hierarchical,
                   torch.device("cuda"), storage=ps, gpu_cache_size=CACHE_SIZE,
                   config={"default_embedding": torch.zeros(EMBED_SIZE)})
    with pytest.raises(ValueError, match="doesn't match layer"):
        make_layer(NUM_EMBEDDINGS, EMBED_SIZE, nve.DataType_t.QInt8RowwiseF16,
                   nve_layers.LayerType.Hierarchical, torch.device("cuda"), storage=ps,
                   gpu_cache_size=CACHE_SIZE)
    with pytest.raises(ValueError, match="too few bytes"):
        ps.insert(torch.arange(4, dtype=torch.int64), torch.zeros(4, EMBED_SIZE, dtype=torch.uint8))
