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
import pynve.nve as nve
from pynve.torch.nve_layers import NVEmbedding, NVEmbeddingBag, CacheType

def save(obj, other_file, emb_file):
    stream = open(emb_file, "wb") if isinstance(emb_file, str) else emb_file
    nve.TensorFileFormat().write_table_file_header(stream)

    if isinstance(obj, torch.nn.Module):
        for module in obj.modules():
            if isinstance(module, NVEmbedding) or isinstance(module, NVEmbeddingBag):
                if module.cache_type == CacheType.LinearUVM:
                    module.set_save_stream(stream)
    torch.save(obj, other_file, pickle_protocol=5)

def load(other_file, emb_file, ps_dict = {}):
    stream = open(emb_file, "rb") if isinstance(emb_file, str) else emb_file
    obj = torch.load(other_file, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        for module in obj.modules():
            if isinstance(module, NVEmbedding) or isinstance(module, NVEmbeddingBag):
                if module.cache_type == CacheType.LinearUVM:
                    module.load_from_stream(stream)
                elif module.cache_type == CacheType.Hierarchical:
                    if module.id in ps_dict:
                        module.emb_layer.set_ps_table(ps_dict[module.id])
                    else:
                        raise ValueError(f"Embedding layer with id {module.id} not found in ps_dict")
    return obj