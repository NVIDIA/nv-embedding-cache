#!/usr/bin/python
#
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

"""Tests for NVEmbedding export / load via nve_export."""

import os
import json
import subprocess
import tempfile
import torch
import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_export import export as nve_export, export_aot, load as nve_load

DEVICE = torch.device("cuda")
NUM_EMB = 1024
EMB_SIZE = 8
GPU_CACHE = 4 * 1024 * 1024  # 4 MiB


def _make_embedding(optimize_for_training=False):
    """Create a simple NVEmbedding with LinearUVM cache."""
    return nve_layers.NVEmbedding(
        num_embeddings=NUM_EMB,
        embedding_size=EMB_SIZE,
        data_type=torch.float32,
        cache_type=nve_layers.CacheType.LinearUVM,
        gpu_cache_size=GPU_CACHE,
        optimize_for_training=optimize_for_training,
        device=DEVICE,
    )


class SimpleModel(torch.nn.Module):
    """A minimal model with one NVEmbedding layer."""
    def __init__(self, optimize_for_training=False):
        super().__init__()
        self.emb = _make_embedding(optimize_for_training=optimize_for_training)

    def forward(self, keys):
        return self.emb(keys)


def test_export_and_load():
    """Export a model with NVEmbedding, load it, verify outputs match."""
    model = SimpleModel(optimize_for_training=False)
    keys = torch.randint(0, NUM_EMB, (32,), device=DEVICE, dtype=torch.int64)

    # Get expected output before export
    expected = model(keys)

    with tempfile.TemporaryDirectory() as save_dir:
        # Export
        nve_export(model, (keys,), save_dir)

        # Verify files were created
        assert os.path.exists(os.path.join(save_dir, "model.pt2"))
        assert os.path.exists(os.path.join(save_dir, "metadata.json"))
        assert os.path.exists(os.path.join(save_dir, "weights"))

        # Verify metadata
        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert len(metadata) == 1
        assert metadata[0]["num_embeddings"] == NUM_EMB
        assert metadata[0]["embedding_size"] == EMB_SIZE
        assert metadata[0]["cache_type"] == "LinearUVM"

        # Load
        loaded, layers = nve_load(save_dir)

        # Run forward on loaded model
        actual = loaded.module()(keys)

        # Verify outputs match
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Output mismatch: max diff = {(expected - actual).abs().max()}"
        print(f"PASS: export/load outputs match (max diff = {(expected - actual).abs().max():.2e})")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NVE_INFERENCE_BIN = os.path.join(REPO_ROOT, "./build/bin/nve_inference")


def test_cpp_inference():
    """Export model with known weights via AOTInductor, run C++ binary, verify output."""
    if not os.path.exists(NVE_INFERENCE_BIN):
        print(f"SKIP: C++ binary not found: {NVE_INFERENCE_BIN}")
        return

    model = SimpleModel(optimize_for_training=False)
    # row[i] = all i's so output is verifiable
    weight_data = torch.arange(NUM_EMB, dtype=torch.float32, device=DEVICE) \
        .unsqueeze(1).expand(NUM_EMB, EMB_SIZE)
    model.emb.weight.data.copy_(weight_data)

    keys = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)

    with tempfile.TemporaryDirectory() as save_dir:
        export_aot(model, (keys,), save_dir)

        env = os.environ.copy()
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(
            [NVE_INFERENCE_BIN, save_dir],
            capture_output=True, text=True, timeout=30,
            cwd=REPO_ROOT,
            env=env,
        )
        assert result.returncode == 0, f"C++ binary failed:\n{result.stderr}"

        stdout = result.stdout
        assert "Output shape: [4, 8]" in stdout
        assert "key=0 -> [0, 0, 0, 0" in stdout
        assert "key=1 -> [1, 1, 1, 1" in stdout
        assert "key=5 -> [5, 5, 5, 5" in stdout
        assert "key=10 -> [10, 10, 10, 10" in stdout
        print("PASS: C++ inference output matches expected values")


if __name__ == "__main__":
    test_export_and_load()
    test_cpp_inference()
