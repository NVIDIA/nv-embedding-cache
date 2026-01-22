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
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
import numpy as np

def main():
    cache_size = 1024*1024
    embed_size = 2
    num_embeddings = 100000
    data_type = torch.float32
    device = torch.device("cuda")

    mw_ps = nve_ps.NVLocalParameterServer(
            0, # Setting num_embeddings as 0 to disable eviction policy
            embed_size,
            data_type,
            None
        )

    keys_path = "/tmp/keys.npy"
    values_path = "/tmp/values.npy"

    keys_array = np.array([10, 5, 700, 1050], dtype=np.int64)
    values_array = np.array([[10.0, 10.5], [5.0, 5.5], [700.0, 700.5], [1050.0, 1050.5]], dtype=np.float32)

    with open(keys_path, "wb") as f:
        np.save(f, keys_array)

    with open(values_path, "wb") as f:
        np.save(f, values_array)

    mw_ps.load_from_file(keys_path, values_path)

    nv_mw_ps_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, remote_interface=mw_ps, device=device)

    out = nv_mw_ps_emb_layer(torch.tensor([10, 5, 700, 1050], dtype=torch.int64, device=device))
    assert torch.equal(out, torch.tensor([[10.0, 10.5], [5.0, 5.5], [700.0, 700.5], [1050.0, 1050.5]], dtype=torch.float32, device=device))
