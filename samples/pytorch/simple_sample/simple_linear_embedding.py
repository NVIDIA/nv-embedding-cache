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

def main():
    # create a simple embedding layer with million embeddings of dimension 128
    # have it use 1GB of GPU memory
    num_embeddings = 1000000
    embedding_dim = 16
    weight_init = torch.randn(num_embeddings, embedding_dim, dtype=torch.float32, device="cuda")
    layers = nve_layers.NVEmbedding(num_embeddings, embedding_dim, torch.float32, nve_layers.CacheType.LinearUVM, gpu_cache_size=1024**3, weight_init=weight_init)

    # create keys
    keys = torch.tensor([7], dtype=torch.int64, device="cuda")

    # lookup the keys
    output = layers(keys)
    print(output)
    print(weight_init[keys])

if __name__ == "__main__":
    main()

