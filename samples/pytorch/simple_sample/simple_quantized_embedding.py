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
import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_dtypes import quantize_rowwise, dequantize_rowwise

def main():
    # Store 100K embeddings of dimension 16 as uint8 codes with an fp16 scale/offset per row.
    num_embeddings = 100000
    embedding_dim = 16
    qtype = nve.DataType_t.QUint8RowwiseF16
    weights = torch.randn(num_embeddings, embedding_dim)
    rows = quantize_rowwise(weights, qtype)  # uint8 [num_embeddings, embedding_dim + 4]

    # Quantized layers are inference-only.
    embedding = nve_layers.NVEmbedding(num_embeddings, embedding_dim, qtype, nve_layers.LayerType.LinearUVM,
                                       gpu_cache_size=1024**2, weight_init=rows, optimize_for_training=False)
    bag = nve_layers.NVEmbeddingBag(num_embeddings, embedding_dim, qtype, nve_layers.LayerType.LinearUVM, "sum",
                                    gpu_cache_size=1024**2, weight_init=rows, optimize_for_training=False)

    # Lookups return decoded fp16 rows.
    keys = torch.tensor([7, 42, 99999], dtype=torch.int64, device="cuda")
    print(embedding(keys))
    print(dequantize_rowwise(rows[keys.cpu()], qtype))

    offsets = torch.tensor([0, 1, 3], dtype=torch.int64, device="cuda")
    print(bag(keys, offsets))

    # Replace a row with new raw stored bytes.
    new_row = quantize_rowwise(torch.full((1, embedding_dim), 0.5), qtype)
    embedding.update(keys[:1], new_row.cuda())
    torch.cuda.synchronize()
    print(embedding(keys[:1]))

if __name__ == "__main__":
    main()
