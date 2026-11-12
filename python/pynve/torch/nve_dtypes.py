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

"""Storage dtype helpers shared by the layers, the parameter server and export.

A layer's storage type (``storage_data_type``) is an ``nve.DataType_t``: Float32,
Float16 or one of the rowwise-quantized types. A quantized row stores
``embedding_size`` 1-byte codes followed by a scale (QInt8, symmetric) or a scale
and an offset (QUint8, affine) in the F32/F16 precision named by the type, and
decodes to ``q * scale [+ offset]`` in that precision.

This module imports only torch and pynve.nve so every pynve.torch module can use it.
"""

import torch
import pynve.nve as nve

StorageType = torch.dtype | nve.DataType_t

_FLOAT_TYPES = {
    torch.float32: nve.DataType_t.Float32,
    torch.float16: nve.DataType_t.Float16,
}
_TORCH_FROM_NVE = {v: k for k, v in _FLOAT_TYPES.items()}
_RAW_ROW_DTYPES = (torch.uint8, torch.int8)


def nve_type_to_torch_type(nve_type: nve.DataType_t) -> torch.dtype:
    if nve_type in _TORCH_FROM_NVE:
        return _TORCH_FROM_NVE[nve_type]
    raise ValueError(f"Invalid data type: {nve_type}")


def torch_type_to_nve_type(torch_type: torch.dtype) -> nve.DataType_t:
    if torch_type in _FLOAT_TYPES:
        return _FLOAT_TYPES[torch_type]
    raise ValueError(f"Invalid data type: {torch_type}")


def resolve_data_type(data_type: StorageType) -> nve.DataType_t:
    """Return the storage nve.DataType_t for a torch float dtype or an nve.DataType_t."""
    if isinstance(data_type, torch.dtype):
        return torch_type_to_nve_type(data_type)
    if isinstance(data_type, nve.DataType_t) and (
            data_type in _TORCH_FROM_NVE or nve.is_quant_rowwise(data_type)):
        return data_type
    quant_names = ', '.join(t.name for t in nve.DataType_t.__members__.values() if nve.is_quant_rowwise(t))
    raise ValueError(
        f"Invalid data type: {data_type}; expected torch.float32, torch.float16 or a "
        f"rowwise-quantized nve.DataType_t ({quant_names})")


def output_nve_type(storage_data_type: nve.DataType_t) -> nve.DataType_t:
    """Dtype returned by lookups: the decoded float type for quantized storage."""
    if nve.is_quant_rowwise(storage_data_type):
        return nve.quant_rowwise_output_dtype(storage_data_type)
    return storage_data_type


def output_torch_type(storage_data_type: nve.DataType_t) -> torch.dtype:
    return nve_type_to_torch_type(output_nve_type(storage_data_type))


def dtype_str(storage_data_type: nve.DataType_t) -> str:
    """Canonical NVE name, e.g. 'float32' or 'qint8_rowwise_f32'."""
    return nve.dtype_to_string(storage_data_type)


def parse_dtype(name: str) -> nve.DataType_t:
    """Inverse of dtype_str; also accepts the legacy 'torch.float32' spelling."""
    try:
        return resolve_data_type(nve.dtype_from_string(name.removeprefix("torch.")))
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Unsupported dtype string {name!r}") from e


def check_num_rows(t: torch.Tensor, row_width: int, what: str, num_rows: int | None, exact: bool):
    """Require at least num_rows rows of row_width elements (exactly num_rows when exact)."""
    if num_rows is None:
        return
    expected = num_rows * row_width
    if t.numel() < expected or (exact and t.numel() != expected):
        qualifier = "exactly" if exact else "at least"
        raise ValueError(f"{what} must hold {qualifier} {num_rows} rows of {row_width} elements, "
                         f"got shape {tuple(t.shape)}")


def check_raw_rows(t: torch.Tensor, row_bytes: int, what: str, num_rows: int | None = None,
                   exact: bool = False):
    """Validate a tensor of raw stored rows before its data_ptr() is handed to C++."""
    if t.dtype not in _RAW_ROW_DTYPES:
        raise TypeError(
            f"{what} for a rowwise-quantized layer must be a uint8 or int8 tensor of raw "
            f"stored rows (see nve_dtypes.quantize_rowwise), got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{what} must be contiguous")
    if t.dim() == 0 or t.shape[-1] != row_bytes:
        raise ValueError(
            f"{what} must have last dimension {row_bytes} (stored row bytes), got shape "
            f"{tuple(t.shape)}")
    check_num_rows(t, row_bytes, what, num_rows, exact)


def check_rows(values: torch.Tensor, storage_data_type: nve.DataType_t, embedding_size: int,
               num_rows: int, what: str, exact: bool = False):
    """Validate rows written to a layer or parameter server: raw stored rows for quantized
    types, contiguous values of the storage dtype otherwise."""
    if nve.is_quant_rowwise(storage_data_type):
        check_raw_rows(values, nve.row_size_in_bytes(storage_data_type, embedding_size), what,
                       num_rows, exact)
        return
    data_type = nve_type_to_torch_type(storage_data_type)
    if values.dtype != data_type:
        raise TypeError(f"{what} must be {data_type}, got {values.dtype}")
    if not values.is_contiguous():
        raise ValueError(f"{what} must be contiguous")
    check_num_rows(values, embedding_size, what, num_rows, exact)


def _has_offset(dtype: nve.DataType_t) -> bool:
    return nve.quant_rowwise_meta_count(dtype) == 2


def quantize_rowwise(x: torch.Tensor, dtype: nve.DataType_t) -> torch.Tensor:
    """Quantize float rows [n, D] to raw stored rows uint8 [n, row_bytes].

    QInt8 (symmetric): scale = max|x| / 127, q = round(x / scale).
    QUint8 (affine): scale = (max - min) / 255, offset = min, q = round((x - offset) / scale).
    Scale and offset are rounded to the type's precision before quantizing; rows with
    zero range get scale 1. Nonzero scales are bounded by the smallest subnormal
    magnitude in that precision. Rows with non-finite metadata or decoded values are
    rejected. Any producer writing the same row layout is accepted.
    """
    if not nve.is_quant_rowwise(dtype):
        raise ValueError(f"{dtype} is not a rowwise-quantized type")
    if x.dim() != 2:
        raise ValueError(f"quantize_rowwise expects a 2D [rows, embedding_size] tensor, got {tuple(x.shape)}")
    meta_type = output_torch_type(dtype)
    xf = x.detach().float()

    def _rounded(v):
        return v.to(meta_type).float()

    def _rounded_scale(v):
        info = torch.finfo(meta_type)
        min_scale = info.tiny * info.eps
        scale = _rounded(v.abs().clamp_min(min_scale)).copysign(v)
        return torch.where(v == 0, torch.ones_like(scale), scale)

    if _has_offset(dtype):
        amin, amax = torch.aminmax(xf, dim=1)
        offset = _rounded(amin)
        scale = _rounded_scale((amax - offset) / 255.0)
        meta = [scale, offset]
    else:
        scale = _rounded_scale(xf.abs().amax(dim=1) / 127.0)
        meta = [scale]
    # The row is non-finite or its scale/offset overflow meta_type.
    bad = ~torch.stack(meta).isfinite().all(dim=0)
    if bad.any():
        rows = bad.nonzero().flatten()[:8].tolist()
        raise ValueError(f"Rows {rows} can't be quantized to {dtype.name}: scale/offset overflow "
                         f"{meta_type} or the row has non-finite values")
    if _has_offset(dtype):
        q = torch.round((xf - offset[:, None]) / scale[:, None]).clamp(0, 255).to(torch.uint8)
    else:
        q = torch.round(xf / scale[:, None]).clamp(-127, 127).to(torch.int8).view(torch.uint8)
    parts = [q] + [m.to(meta_type)[:, None].contiguous().view(torch.uint8) for m in meta]
    raw = torch.cat(parts, dim=1).contiguous()
    # Finite metadata can still overflow after decoding with a rounded scale.
    bad = ~dequantize_rowwise(raw, dtype).isfinite().all(dim=1)
    if bad.any():
        rows = bad.nonzero().flatten()[:8].tolist()
        raise ValueError(f"Rows {rows} can't be quantized to {dtype.name}: decoded values overflow "
                         f"{meta_type}")
    return raw


def dequantize_rowwise(raw: torch.Tensor, dtype: nve.DataType_t) -> torch.Tensor:
    """Decode raw stored rows [n, row_bytes] to [n, D] in the type's float precision."""
    if not nve.is_quant_rowwise(dtype):
        raise ValueError(f"{dtype} is not a rowwise-quantized type")
    meta_bytes = nve.row_size_in_bytes(dtype, 0)
    raw = raw.view(torch.uint8).reshape(-1, raw.shape[-1])
    dim = raw.shape[-1] - meta_bytes
    if dim <= 0:
        raise ValueError(f"Row of {raw.shape[-1]} bytes is too short for {dtype}")
    meta_type = output_torch_type(dtype)
    code_type = torch.uint8 if _has_offset(dtype) else torch.int8
    q = raw[:, :dim].view(code_type).float()
    # A single-row slice may already be contiguous but have an unaligned storage offset.
    meta = raw[:, dim:].clone(memory_format=torch.contiguous_format).view(meta_type).float()
    out = q * meta[:, 0:1]
    if _has_offset(dtype):
        out = out + meta[:, 1:2]
    return out.to(meta_type)
