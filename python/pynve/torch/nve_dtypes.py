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

"""Storage data types shared by NVEmbedding layers and parameter servers.

A layer's storage type is a ``torch.dtype`` (float32/float16) or one of the rowwise-quantized
``nve.DataType_t`` values. Quantized rows hold one int8/uint8 code per element followed by the
row's scale (QInt8) or scale and offset (QUint8) in the type's F32/F16 precision; lookups
dequantize them to that precision.
"""

from typing import Union

import pynve.nve as nve
import torch

StorageType = Union[torch.dtype, "nve.DataType_t"]

# quantized storage type -> (dequantized output type, number of trailing scale/offset values)
_QUANT_ROWWISE = {
    nve.DataType_t.QInt8RowwiseF32: (torch.float32, 1),
    nve.DataType_t.QInt8RowwiseF16: (torch.float16, 1),
    nve.DataType_t.QUint8RowwiseF32: (torch.float32, 2),
    nve.DataType_t.QUint8RowwiseF16: (torch.float16, 2),
}


def is_quant_rowwise(data_type: StorageType) -> bool:
    """Return True for the rowwise-quantized storage types."""
    return isinstance(data_type, nve.DataType_t) and data_type in _QUANT_ROWWISE


def output_torch_type(data_type: StorageType) -> torch.dtype:
    """Return the torch dtype lookups produce for a storage type."""
    if is_quant_rowwise(data_type):
        return _QUANT_ROWWISE[data_type][0]
    if data_type in (torch.float32, torch.float16):
        return data_type
    raise ValueError(f"Invalid data type: {data_type}")


def storage_nve_type(data_type: StorageType) -> "nve.DataType_t":
    """Return the nve.DataType_t a storage type is stored as."""
    if is_quant_rowwise(data_type):
        return data_type
    if data_type == torch.float32:
        return nve.DataType_t.Float32
    if data_type == torch.float16:
        return nve.DataType_t.Float16
    raise ValueError(f"Invalid data type: {data_type}")


def stored_row_bytes(embedding_size: int, data_type: StorageType) -> int:
    """Return the bytes of one stored row, including any quantization metadata."""
    if is_quant_rowwise(data_type):
        output_type, meta_count = _QUANT_ROWWISE[data_type]
        return embedding_size + meta_count * torch.empty(0, dtype=output_type).element_size()
    return embedding_size * torch.empty(0, dtype=data_type).element_size()
