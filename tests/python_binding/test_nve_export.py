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
import numpy as np
import pytest
import torch
import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from pynve.torch.nve_export import (
    export as nve_export, export_aot, load as nve_load,
    save_nve, load_nve_layers,
)
from conftest import requires_nvhm

DEVICE = torch.device("cuda")
NUM_EMB = 1024
EMB_SIZE = 8
GPU_CACHE = 4 * 1024 * 1024  # 4 MiB


def _make_embedding(cache_type=nve_layers.CacheType.LinearUVM, optimize_for_training=False):
    """Create a simple NVEmbedding for the given cache type."""
    kwargs = dict(
        num_embeddings=NUM_EMB,
        embedding_size=EMB_SIZE,
        data_type=torch.float32,
        cache_type=cache_type,
        optimize_for_training=optimize_for_training,
        device=DEVICE,
    )
    if cache_type == nve_layers.CacheType.LinearUVM:
        kwargs["gpu_cache_size"] = GPU_CACHE
    return nve_layers.NVEmbedding(**kwargs)


class SimpleModel(torch.nn.Module):
    """A minimal model with one NVEmbedding layer."""
    def __init__(self, cache_type=nve_layers.CacheType.LinearUVM, optimize_for_training=False):
        super().__init__()
        self.emb = _make_embedding(cache_type=cache_type,
                                   optimize_for_training=optimize_for_training)

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


def test_export_dynamic_shapes():
    """AOT-compile NVEmbedding with Dim.AUTO batch dim and verify two batch sizes.

    Regression test for the Python-side register_fake kernels: the C++
    stable-ABI Meta impl used to fail with `aoti_torch_get_numel` on SymInts.
    """
    model = SimpleModel(optimize_for_training=False)
    weight = torch.arange(NUM_EMB, dtype=torch.float32, device=DEVICE) \
        .unsqueeze(1).expand(NUM_EMB, EMB_SIZE)
    model.emb.weight.data.copy_(weight)

    keys_example = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)
    sc = torch.export.ShapesCollection()
    sc[keys_example] = {0: torch.export.Dim.AUTO(min=1, max=1000)}
    ds = sc.dynamic_shapes(model, (keys_example,))

    with tempfile.TemporaryDirectory() as save_dir:
        export_aot(model, (keys_example,), save_dir, dynamic_shapes=ds)
        loader = torch._inductor.aoti_load_package(
            os.path.join(save_dir, "model.pt2"))

        # Batch size matches example input.
        out_a = loader(keys_example)
        assert out_a.shape == (4, EMB_SIZE)
        assert torch.all(out_a[0] == 0)
        assert torch.all(out_a[2] == 5)

        # Different batch size — only works if the dynamic dim is honored.
        keys_b = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
        out_b = loader(keys_b)
        assert out_b.shape == (2, EMB_SIZE)
        assert torch.all(out_b[0] == 3)
        assert torch.all(out_b[1] == 7)


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


# ---------------------------------------------------------------------------
# NoCache (GPU-only) export/load tests
# ---------------------------------------------------------------------------


def test_export_and_load_gpu():
    """Export a NoCache (GPU-only) model, load it back, verify outputs match."""
    model = SimpleModel(cache_type=nve_layers.CacheType.NoCache,
                        optimize_for_training=False)
    keys = torch.randint(0, NUM_EMB, (32,), device=DEVICE, dtype=torch.int64)

    expected = model(keys)

    with tempfile.TemporaryDirectory() as save_dir:
        nve_export(model, (keys,), save_dir)

        assert os.path.exists(os.path.join(save_dir, "model.pt2"))
        assert os.path.exists(os.path.join(save_dir, "metadata.json"))
        assert os.path.exists(os.path.join(save_dir, "weights", "emb.nve"))

        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert len(metadata) == 1
        assert metadata[0]["num_embeddings"] == NUM_EMB
        assert metadata[0]["embedding_size"] == EMB_SIZE
        assert metadata[0]["cache_type"] == "NoCache"
        assert metadata[0]["memblock_type"] == str(nve.MemBlockType.User)

        loaded, layers = nve_load(save_dir)

        actual = loaded.module()(keys)
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Output mismatch: max diff = {(expected - actual).abs().max()}"
        print(f"PASS: NoCache export/load outputs match (max diff = {(expected - actual).abs().max():.2e})")


def test_cpp_inference_gpu():
    """Export NoCache model with known weights via AOTInductor, run C++ binary, verify output."""
    if not os.path.exists(NVE_INFERENCE_BIN):
        print(f"SKIP: C++ binary not found: {NVE_INFERENCE_BIN}")
        return

    model = SimpleModel(cache_type=nve_layers.CacheType.NoCache,
                        optimize_for_training=False)
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
        print("PASS: NoCache C++ inference output matches expected values")


# ---------------------------------------------------------------------------
# Hierarchical export/load tests
# ---------------------------------------------------------------------------

HIER_NUM_EMB = 10000
HIER_EMB_SIZE = 4
HIER_GPU_CACHE = 1024 * 1024  # 1 MiB


class HierarchicalModel(torch.nn.Module):
    """Model with a single Hierarchical NVEmbedding."""
    def __init__(self, remote_ps):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            HIER_NUM_EMB, HIER_EMB_SIZE, torch.float32,
            nve_layers.CacheType.Hierarchical,
            gpu_cache_size=HIER_GPU_CACHE,
            remote_interface=remote_ps,
            optimize_for_training=False,
            device=DEVICE,
        )

    def forward(self, keys):
        return self.emb(keys)


def _make_ps_with_data(tmpdir):
    """Create an NVHashMap PS with known data and save data files.

    Returns (NVEParameterServer, keys_path, values_path).
    """
    keys_array = np.arange(HIER_NUM_EMB, dtype=np.int64)
    values_array = np.array(
        [[float(i) + j * 0.1 for j in range(HIER_EMB_SIZE)]
         for i in range(HIER_NUM_EMB)],
        dtype=np.float32,
    )
    keys_path = os.path.join(tmpdir, "keys.npy")
    values_path = os.path.join(tmpdir, "values.npy")
    np.save(keys_path, keys_array)
    np.save(values_path, values_array)

    ps = nve_ps.NVEParameterServer(
        num_embeddings=0,
        embedding_size=HIER_EMB_SIZE,
        data_type=torch.float32,
    )
    ps.load_from_file(keys_path, values_path)
    return ps, keys_path, values_path


@requires_nvhm
def test_save_load_hierarchical():
    """save_nve + load_nve_layers round-trip for Hierarchical with data reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ps, keys_path, values_path = _make_ps_with_data(tmpdir)
        model = HierarchicalModel(ps)

        test_keys = torch.tensor([0, 10, 50, 999], dtype=torch.int64, device=DEVICE)
        expected = model(test_keys)

        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir, ps_data_paths={"emb": (keys_path, values_path)})

        # Verify metadata
        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert len(metadata) == 1
        assert metadata[0]["cache_type"] == "Hierarchical"
        assert "remote_ps_config" in metadata[0]
        assert metadata[0]["remote_ps_config"]["remote_ps_type"] == "plugin"
        assert metadata[0]["remote_ps_config"]["plugin_name"] == "libnve-plugin-nvhm.so"
        assert "remote_ps_data" in metadata[0]

        # Load
        del model, ps
        layers = load_nve_layers(save_dir)
        assert len(layers) == 1

        actual = layers[0](test_keys)
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Mismatch: max diff = {(expected - actual).abs().max():.2e}"
        print("PASS: Hierarchical save/load round-trip")


@requires_nvhm
def test_save_load_hierarchical_no_data_paths():
    """Hierarchical export without ps_data_paths — config only, no data reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ps, _, _ = _make_ps_with_data(tmpdir)
        model = HierarchicalModel(ps)

        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)  # no ps_data_paths

        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert "remote_ps_config" in metadata[0]
        assert "remote_ps_data" not in metadata[0]

        # Load succeeds (PS is empty since no data was reloaded)
        del model, ps
        layers = load_nve_layers(save_dir)
        assert len(layers) == 1
        print("PASS: Hierarchical config-only export/load")


def test_export_config_nve_parameter_server():
    """NVEParameterServer.export_config() returns the new plugin-shape dict."""
    ps = nve_ps.NVEParameterServer(
        num_embeddings=1000,
        embedding_size=64,
        data_type=torch.float32,
        ps_type=nve.NVHashMap,
        extra_params={"table": {"max_num_keys_per_task": 8192}},
    )
    config = ps.export_config()
    assert config["remote_ps_type"] == "plugin"
    assert config["plugin_name"] == "libnve-plugin-nvhm.so"
    assert config["row_elements"] == 64
    assert config["num_rows"] == 1000
    assert config["data_type"].lower().startswith("float32")
    assert config["factory_config"]["implementation"] == "nvhm_map"
    # extra_params["table"]["max_num_keys_per_task"] gets merged into table_config
    assert config["table_config"]["max_num_keys_per_task"] == 8192
    print("PASS: export_config returns plugin-shape dict")


# ---------------------------------------------------------------------------
# Custom plugin (custom_remote) export/load test
# ---------------------------------------------------------------------------

def _find_custom_remote_plugin():
    """Return a loadable libnve-plugin-custom_remote.so spec, or None.

    Checks the system search path first, then the pynve package directory
    (where pip install drops the plugin alongside nve.so).
    """
    import ctypes
    try:
        ctypes.CDLL("libnve-plugin-custom_remote.so")
        return "libnve-plugin-custom_remote.so"
    except OSError:
        pass
    try:
        import pynve
        plugin_path = os.path.join(
            os.path.dirname(pynve.__file__), "libnve-plugin-custom_remote.so")
        if os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path)
            return plugin_path
    except OSError:
        pass
    return None


CUSTOM_REMOTE_PLUGIN = _find_custom_remote_plugin()

requires_custom_remote = pytest.mark.skipif(
    CUSTOM_REMOTE_PLUGIN is None,
    reason="libnve-plugin-custom_remote.so not on LD_LIBRARY_PATH "
           "(build samples/common/custom_remote_plugin)"
)


def _make_custom_remote_ps(num_embeddings, embedding_size, data_type=torch.float32):
    """Create a ParameterServerTable backed by the custom_remote plugin."""
    return nve_ps.NVEParameterServer(
        num_embeddings=num_embeddings,
        embedding_size=embedding_size,
        data_type=data_type,
        plugin_name=CUSTOM_REMOTE_PLUGIN,
        factory_config={"implementation": "custom_remote"},
        table_config={
            "key_size": 8,
            "max_value_size": embedding_size * (4 if data_type == torch.float32 else 2),
        },
    )


@requires_custom_remote
def test_plugin_ps_export_config():
    """Plugin-based PS export_config returns the expected shape."""
    ps = _make_custom_remote_ps(num_embeddings=500, embedding_size=8)
    config = ps.export_config()
    assert config["remote_ps_type"] == "plugin"
    assert config["plugin_name"] == CUSTOM_REMOTE_PLUGIN
    assert config["factory_config"]["implementation"] == "custom_remote"
    assert config["row_elements"] == 8
    assert config["num_rows"] == 500
    print("PASS: plugin-based export_config returns correct dict")


@requires_custom_remote
def test_plugin_ps_export_load_roundtrip():
    """End-to-end: export an NVEmbedding(Hierarchical, custom_remote) and reload it."""
    num_emb = 256
    emb_size = 4
    gpu_cache = 1024 * 1024

    ps = _make_custom_remote_ps(num_emb, emb_size)
    # Insert a few known keys directly into the PS so we can verify lookups.
    keys = torch.arange(num_emb, dtype=torch.int64)
    values = torch.tensor(
        [[float(i) + j * 0.1 for j in range(emb_size)] for i in range(num_emb)],
        dtype=torch.float32,
    )
    ps.insert(keys, values)

    model = torch.nn.Module()
    model.emb = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.CacheType.Hierarchical,
        gpu_cache_size=gpu_cache,
        remote_interface=ps,
        optimize_for_training=False,
        device=DEVICE,
    )

    test_keys = torch.tensor([0, 10, 100, 255], dtype=torch.int64, device=DEVICE)
    expected = model.emb(test_keys)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        # Persist the PS data so load can reload it (custom_remote is in-memory
        # — the new instance starts empty otherwise).
        keys_path = os.path.join(tmpdir, "keys.npy")
        values_path = os.path.join(tmpdir, "values.npy")
        np.save(keys_path, keys.numpy())
        np.save(values_path, values.numpy())

        save_nve(model, save_dir, ps_data_paths={"emb": (keys_path, values_path)})

        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata[0]["remote_ps_config"]["remote_ps_type"] == "plugin"
        assert metadata[0]["remote_ps_config"]["plugin_name"] == CUSTOM_REMOTE_PLUGIN

        del model, ps
        layers = load_nve_layers(save_dir)
        actual = layers[0](test_keys)
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Mismatch: max diff = {(expected - actual).abs().max():.2e}"
        print("PASS: custom_remote plugin export/load round-trip")


@requires_custom_remote
def test_cpp_inference_custom_ps():
    """Export Hierarchical(custom_remote) model via AOTInductor, run C++ binary, verify."""
    if not os.path.exists(NVE_INFERENCE_BIN):
        print(f"SKIP: C++ binary not found: {NVE_INFERENCE_BIN}")
        return

    num_emb = 1024
    emb_size = 8
    gpu_cache = 4 * 1024 * 1024

    ps = nve_ps.NVEParameterServer(
        num_embeddings=num_emb,
        embedding_size=emb_size,
        data_type=torch.float32,
        plugin_name=CUSTOM_REMOTE_PLUGIN,
        factory_config={"implementation": "custom_remote"},
        table_config={"key_size": 8, "max_value_size": emb_size * 4},
    )
    keys_full = torch.arange(num_emb, dtype=torch.int64)
    values_full = torch.zeros(num_emb, emb_size, dtype=torch.float32)
    for i in range(num_emb):
        values_full[i] = float(i)
    ps.insert(keys_full, values_full)

    class _CustomPSModel(torch.nn.Module):
        def __init__(self, remote_ps):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_emb, emb_size, torch.float32,
                nve_layers.CacheType.Hierarchical,
                gpu_cache_size=gpu_cache,
                remote_interface=remote_ps,
                optimize_for_training=False,
                device=DEVICE,
            )

        def forward(self, keys):
            return self.emb(keys)

    model = _CustomPSModel(ps)

    keys = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)

    with tempfile.TemporaryDirectory() as save_dir:
        keys_path = os.path.join(save_dir, "keys.npy")
        values_path = os.path.join(save_dir, "values.npy")
        np.save(keys_path, keys_full.numpy())
        np.save(values_path, values_full.numpy())

        export_aot(model, (keys,), save_dir,
                   ps_data_paths={"emb": (keys_path, values_path)})

        env = os.environ.copy()
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(
            [NVE_INFERENCE_BIN, save_dir],
            capture_output=True, text=True, timeout=60,
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
        print("PASS: C++ inference custom_ps output matches expected values")


if __name__ == "__main__":
    test_export_and_load()
    test_cpp_inference()
    test_export_config_nve_parameter_server()
    test_plugin_ps_export_config()
    test_plugin_ps_export_load_roundtrip()
    test_cpp_inference_custom_ps()
