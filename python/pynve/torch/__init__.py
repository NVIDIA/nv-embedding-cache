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

import torch

HAS_TORCH_OPS = False
try:
    import pynve.nve  # triggers loading libnve-torch-ops.so if linked
    _ = torch.ops.nve_ops.embedding_lookup
    HAS_TORCH_OPS = True
except (ImportError, AttributeError):
    pass


if HAS_TORCH_OPS:
    # Meta (shape-inference) kernels for torch.export / AOTInductor.
    # The C++ stable-ABI Meta impls were removed because aoti_torch_get_numel
    # can't resolve SymInts; Python fakes handle dynamic shapes natively.
    _NVE_TO_TORCH_DTYPE = {
        pynve.nve.DataType_t.Float32: torch.float32,
        pynve.nve.DataType_t.Float16: torch.float16,
    }

    def _torch_dtype_for(nve_dtype):
        try:
            return _NVE_TO_TORCH_DTYPE[nve_dtype]
        except KeyError:
            raise RuntimeError(
                f"nve_ops fake kernel: unsupported NVE dtype {nve_dtype}")

    @torch.library.register_fake("nve_ops::embedding_lookup")
    def _embedding_lookup_fake(keys, layer_id):
        emb_dim, nve_dtype = pynve.nve.get_torch_binding_info(layer_id)
        return keys.new_empty(
            (keys.size(0), emb_dim),
            dtype=_torch_dtype_for(nve_dtype))

    @torch.library.register_fake("nve_ops::embedding_lookup_with_pooling")
    def _embedding_lookup_with_pooling_fake(
            keys, offsets, weights, pooling_type, layer_id):
        emb_dim, nve_dtype = pynve.nve.get_torch_binding_info(layer_id)
        return offsets.new_empty(
            (offsets.size(0) - 1, emb_dim),
            dtype=_torch_dtype_for(nve_dtype))
