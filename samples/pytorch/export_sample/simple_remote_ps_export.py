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

"""
Sample: Export and load a Hierarchical NVEmbedding with a remote parameter server.

Demonstrates:
1. Creating a Hierarchical embedding backed by an NVHashMap parameter server
2. Populating the PS with known data from .npy files
3. Exporting the model (metadata + PS config serialization via export_config())
4. Loading the model — PS is recreated by directly invoking the plugin-based
   ParameterServerTable ctor with the JSON saved at export time.
5. Verifying that lookups produce the same results after load
"""

import os
import tempfile
import numpy as np
import torch
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from pynve.torch.nve_export import save_nve, load_nve_layers


class HierarchicalModel(torch.nn.Module):
    """A minimal model with one Hierarchical NVEmbedding layer."""
    def __init__(self, num_embeddings, embedding_size, data_type, gpu_cache_size, remote_ps):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings, embedding_size, data_type,
            nve_layers.CacheType.Hierarchical,
            gpu_cache_size=gpu_cache_size,
            remote_interface=remote_ps,
            optimize_for_training=False,
        )

    def forward(self, keys):
        return self.emb(keys)


def main():
    num_embeddings = 1000000
    embedding_size = 4
    gpu_cache_size = 1024 * 1024  # 1 MiB
    data_type = torch.float32
    device = torch.device("cuda")

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Prepare data files ---
        keys_array = np.arange(num_embeddings, dtype=np.int64)
        # row[i] = [i, i+0.1, i+0.2, i+0.3]
        values_array = np.array(
            [[float(i) + j * 0.1 for j in range(embedding_size)]
             for i in range(num_embeddings)],
            dtype=np.float32,
        )

        keys_path = os.path.join(tmpdir, "keys.npy")
        values_path = os.path.join(tmpdir, "values.npy")
        np.save(keys_path, keys_array)
        np.save(values_path, values_array)

        # --- Create PS and load data ---
        ps = nve_ps.NVEParameterServer(
            num_embeddings=0,  # 0 = no eviction
            embedding_size=embedding_size,
            data_type=data_type,
        )
        ps.load_from_file(keys_path, values_path)

        # --- Create model ---
        model = HierarchicalModel(num_embeddings, embedding_size, data_type, gpu_cache_size, ps)

        # --- Run forward pass ---
        test_keys = torch.tensor([0, 10, 50, 999], dtype=torch.int64, device=device)
        expected = model(test_keys)
        print(f"Original output:\n{expected}")

        # --- Export (save_nve only, no torch.export needed for this demo) ---
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir, ps_data_paths={"emb": (keys_path, values_path)})

        # Verify metadata was saved with remote_ps_config
        import json
        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert len(metadata) == 1
        assert metadata[0]["cache_type"] == "Hierarchical"
        assert "remote_ps_config" in metadata[0]
        assert metadata[0]["remote_ps_config"]["remote_ps_type"] == "plugin"
        assert metadata[0]["remote_ps_config"]["plugin_name"] == "libnve-plugin-nvhm.so"
        print(f"Metadata saved with remote_ps_config: {metadata[0]['remote_ps_config']}")

        # --- Load (recreate PS directly via the plugin ctor, reload data from files) ---
        del model, ps  # Simulate a fresh process

        layers = load_nve_layers(save_dir)
        loaded_layer = layers[0]

        # --- Verify ---
        actual = loaded_layer(test_keys)
        print(f"Loaded output:\n{actual}")

        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Mismatch! max diff = {(expected - actual).abs().max():.2e}"
        print("PASS: Export/load round-trip outputs match")


if __name__ == "__main__":
    main()
