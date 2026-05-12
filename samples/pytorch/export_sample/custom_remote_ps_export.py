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
Sample: Export and load a Hierarchical NVEmbedding with a custom-plugin remote PS.

This sample demonstrates how a user-defined C++ host-table plugin
participates in the NVE export/load flow. The plugin used here is
``samples/common/custom_remote_plugin`` (built as
``libnve-plugin-custom_remote.so``), but the same pattern works for any
plugin DLL that implements the host-table contract.

The flow:
1. Construct a ``ParameterServerTable`` with
   ``plugin_name="libnve-plugin-custom_remote.so"``
   plus factory/table JSON configs.
2. ``ParameterServerTable.export_config_json()`` returns the same JSON shape
   regardless of which plugin produced it; ``save_nve`` records it under
   ``metadata.json:remote_ps_config``.
3. ``load_nve_layers`` reads the JSON and reconstructs an identical
   ``ParameterServerTable`` directly via the plugin ctor — no Python
   registry is involved.

Prerequisites:
    Build the custom_remote plugin (built by default with samples enabled)::

        cd build && cmake .. && make nve-plugin-custom_remote
"""

import os
import tempfile
import json
import torch
import numpy as np
import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from pynve.torch.nve_export import save_nve, load_nve_layers


class HierarchicalModel(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_size, gpu_cache_size, remote_interface):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings, embedding_size, torch.float32,
            nve_layers.CacheType.Hierarchical,
            gpu_cache_size=gpu_cache_size,
            remote_interface=remote_interface,
            optimize_for_training=False,
        )

    def forward(self, keys):
        return self.emb(keys)


def main():
    num_embeddings = 1024
    embedding_size = 4
    gpu_cache_size = 1024 * 1024
    device = torch.device("cuda")

    # --- Create a custom-plugin parameter server ---
    ps = nve_ps.NVEParameterServer(
        num_embeddings=num_embeddings,
        embedding_size=embedding_size,
        data_type=torch.float32,
        plugin_name="libnve-plugin-custom_remote.so",
        factory_config={"implementation": "custom_remote"},
        table_config={"key_size": 8, "max_value_size": embedding_size * 4},
    )

    # Populate: row[i] = [i, i+0.1, i+0.2, i+0.3]
    keys = torch.arange(num_embeddings, dtype=torch.int64)
    values = torch.tensor(
        [[float(i) + j * 0.1 for j in range(embedding_size)]
         for i in range(num_embeddings)],
        dtype=torch.float32,
    )
    ps.insert(keys, values)

    config = ps.export_config()
    print(f"export_config() returned: {config}")
    assert config["remote_ps_type"] == "plugin"
    assert config["plugin_name"] == "libnve-plugin-custom_remote.so"

    # --- Build and run the model ---
    model = HierarchicalModel(num_embeddings, embedding_size, gpu_cache_size, ps)

    test_keys = torch.tensor([0, 10, 100, 500], dtype=torch.int64, device=device)
    expected = model(test_keys)
    print(f"Original output:\n{expected}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Persist PS data so the reload picks it up — custom_remote is
        # in-memory and would otherwise be empty after load.
        keys_path = os.path.join(tmpdir, "keys.npy")
        values_path = os.path.join(tmpdir, "values.npy")
        np.save(keys_path, keys.numpy())
        np.save(values_path, values.numpy())

        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir, ps_data_paths={"emb": (keys_path, values_path)})

        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata[0]["remote_ps_config"]["plugin_name"] == "libnve-plugin-custom_remote.so"
        print(f"Metadata remote_ps_config: {metadata[0]['remote_ps_config']}")

        # --- Load — PS is reconstructed directly through the plugin ctor ---
        del model, ps

        layers = load_nve_layers(save_dir)
        loaded_layer = layers[0]
        actual = loaded_layer(test_keys)
        print(f"Loaded output:\n{actual}")

        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Mismatch! max diff = {(expected - actual).abs().max():.2e}"
        print("PASS: custom_remote plugin export/load round-trip")


if __name__ == "__main__":
    main()
