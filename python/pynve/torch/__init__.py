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
    #
    # The fakes are PURE: output shape comes from the baked `embedding_size`
    # arg and dtype from the baked `dtype` tag (= int(nve.DataType_t)). No
    # registry lookup — the real layer is located at runtime via the marker
    # tensor's data_ptr (forward path only), which the meta path can't read.
    def _fake_out_dtype(dtype_tag):
        # lazy import avoids a package-init cycle
        # (pynve.torch.__init__ <-> nve_layers). Fakes only run at trace time.
        from pynve.torch.nve_layers import nve_type_to_torch_type
        return nve_type_to_torch_type(pynve.nve.DataType_t(dtype_tag))

    @torch.library.register_fake("nve_ops::embedding_lookup")
    def _embedding_lookup_fake(marker, keys, embedding_size, dtype):
        return keys.new_empty(
            (keys.size(0), embedding_size), dtype=_fake_out_dtype(dtype))

    @torch.library.register_fake("nve_ops::embedding_lookup_with_pooling")
    def _embedding_lookup_with_pooling_fake(
            marker, keys, offsets, weights, pooling_type, embedding_size, dtype):
        return offsets.new_empty(
            (offsets.size(0) - 1, embedding_size), dtype=_fake_out_dtype(dtype))
