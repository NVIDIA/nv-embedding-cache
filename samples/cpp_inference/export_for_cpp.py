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

"""Export an NVEmbedding model for C++ inference via AOTInductor.

Produces:
  output_dir/
  ├── model.pt2          # AOT-compiled model package for AOTIModelPackageLoader
  ├── metadata.json      # NVE layer metadata (id, num_embeddings, emb_size, etc.)
  └── embeddings.nve     # NVE weight data (LinearUVM)
"""

import os
import torch
import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_export import export_aot

DEVICE = torch.device("cuda")
NUM_EMB = 1024
EMB_SIZE = 8
GPU_CACHE = 4 * 1024 * 1024  # 4 MiB


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings=NUM_EMB,
            embedding_size=EMB_SIZE,
            data_type=torch.float32,
            cache_type=nve_layers.CacheType.LinearUVM,
            gpu_cache_size=GPU_CACHE,
            optimize_for_training=False,
            device=DEVICE,
        )

    def forward(self, keys):
        return self.emb(keys)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    model = SimpleModel()

    # Initialize weights: row[i] = all i's (so we can verify in C++)
    weight_data = torch.arange(NUM_EMB, dtype=torch.float32, device=DEVICE).unsqueeze(1).expand(NUM_EMB, EMB_SIZE)
    model.emb.weight.data.copy_(weight_data)
    print(f"Weight[0]: {model.emb.weight[0]}")
    print(f"Weight[5]: {model.emb.weight[5]}")

    keys = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)

    # Verify forward works
    with torch.no_grad():
        out = model(keys)
    print(f"Forward output for keys [0,1,5,10]:\n{out}")

    # Export with AOTInductor (saves metadata.json + embeddings.nve + model.pt2)
    print("Exporting with AOTInductor...")
    export_aot(model, (keys,), output_dir)
    print(f"Saved to {output_dir}")

    # Quick validation: load AOT model in Python and verify
    print("Validating in Python...")
    loader = torch._inductor.aoti_load_package(os.path.join(output_dir, "model.pt2"))
    with torch.no_grad():
        out2 = loader(keys)
    print(f"AOT output matches: {torch.allclose(out, out2, atol=1e-5)}")
    print("Done!")


if __name__ == "__main__":
    main()
