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
    export as nve_export, export_aot, load as nve_load, load_aot,
    save_nve, load_nve_layers, rebind_markers,
)
from conftest import requires_nvhm

DEVICE = torch.device("cuda")
NUM_EMB = 1024
EMB_SIZE = 8
GPU_CACHE = 4 * 1024 * 1024  # 4 MiB


def _make_embedding(layer_type=nve_layers.LayerType.LinearUVM, optimize_for_training=False):
    """Create a simple NVEmbedding for the given layer type."""
    kwargs = dict(
        num_embeddings=NUM_EMB,
        embedding_size=EMB_SIZE,
        data_type=torch.float32,
        layer_type=layer_type,
        optimize_for_training=optimize_for_training,
        device=DEVICE,
    )
    if layer_type == nve_layers.LayerType.LinearUVM:
        kwargs["gpu_cache_size"] = GPU_CACHE
    return nve_layers.NVEmbedding(**kwargs)


class SimpleModel(torch.nn.Module):
    """A minimal model with one NVEmbedding layer."""
    def __init__(self, layer_type=nve_layers.LayerType.LinearUVM, optimize_for_training=False):
        super().__init__()
        self.emb = _make_embedding(layer_type=layer_type,
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
        assert metadata["version"] == 2
        assert len(metadata["layers"]) == 1
        layer0 = metadata["layers"][0]
        assert layer0["num_embeddings"] == NUM_EMB
        assert layer0["embedding_size"] == EMB_SIZE
        assert layer0["layer_type"] == "LinearUVM"

        # Load (non-AOT). load() now returns a ready-to-call module (markers
        # already PUSHed into it) rather than the raw ExportedProgram.
        loaded, layers = nve_load(save_dir)

        # Run forward on loaded model
        actual = loaded(keys)

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
        # load_aot writes the per-layer marker (user_managed) into the AOTI
        # container and registers it; keep `layers` alive for the loader's life.
        loader, layers = load_aot(save_dir)

        # Batch size matches example input.
        out_a = loader.run([keys_example])[0]
        assert out_a.shape == (4, EMB_SIZE)
        assert torch.all(out_a[0] == 0)
        assert torch.all(out_a[2] == 5)

        # Different batch size — only works if the dynamic dim is honored.
        keys_b = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
        out_b = loader.run([keys_b])[0]
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
    model = SimpleModel(layer_type=nve_layers.LayerType.GPULayer,
                        optimize_for_training=False)
    keys = torch.randint(0, NUM_EMB, (32,), device=DEVICE, dtype=torch.int64)

    expected = model(keys)

    with tempfile.TemporaryDirectory() as save_dir:
        nve_export(model, (keys,), save_dir)

        assert os.path.exists(os.path.join(save_dir, "model.pt2"))
        assert os.path.exists(os.path.join(save_dir, "metadata.json"))
        assert len(os.listdir(os.path.join(save_dir, "weights"))) == 1

        with open(os.path.join(save_dir, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata["version"] == 2
        assert len(metadata["layers"]) == 1
        layer0 = metadata["layers"][0]
        assert layer0["num_embeddings"] == NUM_EMB
        assert layer0["embedding_size"] == EMB_SIZE
        assert layer0["layer_type"] == "GPULayer"
        # memblock_type moved to resources section in v2
        storage_ref = layer0["storage_ref"]
        assert storage_ref in metadata["resources"]["memblocks"]

        loaded, layers = nve_load(save_dir)

        actual = loaded(keys)
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Output mismatch: max diff = {(expected - actual).abs().max()}"
        print(f"PASS: NoCache export/load outputs match (max diff = {(expected - actual).abs().max():.2e})")


# ---------------------------------------------------------------------------
# Per-layer marker / multi-instance tests
# ---------------------------------------------------------------------------

def _model_with_id(layer_id, base):
    """A SimpleModel whose embedding has an explicit layer id and a weight
    pattern offset by `base` (row i -> i + base). Forcing two models to share
    layer_id reproduces the registry-collision scenario the marker fixes."""
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nve_layers.NVEmbedding(
                num_embeddings=NUM_EMB, embedding_size=EMB_SIZE,
                data_type=torch.float32,
                layer_type=nve_layers.LayerType.LinearUVM,
                gpu_cache_size=GPU_CACHE, device=DEVICE, id=layer_id)

        def forward(self, keys):
            return self.emb(keys)

    m = M()
    w = (torch.arange(NUM_EMB, dtype=torch.float32, device=DEVICE) + base) \
        .unsqueeze(1).expand(NUM_EMB, EMB_SIZE).contiguous()
    m.emb.weight.data.copy_(w)
    return m


def test_marker_persistent_false_survives():
    """The non-persistent marker buffer survives torch.export save/load."""
    model = SimpleModel(optimize_for_training=False)
    keys = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export(model, (keys,), save_dir)
        ep = torch.export.load(os.path.join(save_dir, "model.pt2"))
        mod = ep.module()
        bufs = dict(mod.named_buffers())
        assert "emb.marker_tensor" in bufs, \
            f"marker_tensor missing; buffers = {list(bufs)}"
        # Value is the layer id (used for per-layer distinctness / debug).
        assert bufs["emb.marker_tensor"].item() == model.emb.id


def test_nonaot_injection_correct():
    """load() PUSHes the layer's marker_tensor into the loaded module (same
    data_ptr) and the loaded module reproduces the eager output."""
    model = SimpleModel(optimize_for_training=False)
    keys = torch.randint(0, NUM_EMB, (16,), device=DEVICE, dtype=torch.int64)
    expected = model(keys)
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export(model, (keys,), save_dir)
        loaded, layers = nve_load(save_dir)
        sub = loaded.get_submodule("emb")
        assert sub.marker_tensor.data_ptr() == layers[0].marker_tensor.data_ptr()
        actual = loaded(keys)
        assert torch.allclose(actual, expected, atol=1e-6)


def test_two_instances_one_process():
    """Two models that share layer_id=0 load + run independently in one process
    (non-AOT). Under the old id-keyed registry the second load would overwrite
    the first; per-layer markers keep them distinct."""
    keys = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)
    model_a = _model_with_id(0, 0.0)
    model_b = _model_with_id(0, 1000.0)
    expected_a = model_a(keys)
    expected_b = model_b(keys)
    assert not torch.allclose(expected_a, expected_b)

    with tempfile.TemporaryDirectory() as dir_a, \
         tempfile.TemporaryDirectory() as dir_b:
        nve_export(model_a, (keys,), dir_a)
        nve_export(model_b, (keys,), dir_b)

        loaded_a, layers_a = nve_load(dir_a)
        loaded_b, layers_b = nve_load(dir_b)

        # distinct marker ptrs despite identical layer_id
        assert layers_a[0].marker_tensor.data_ptr() \
            != layers_b[0].marker_tensor.data_ptr()

        actual_a = loaded_a(keys)
        actual_b = loaded_b(keys)
        assert torch.allclose(actual_a, expected_a, atol=1e-6)
        assert torch.allclose(actual_b, expected_b, atol=1e-6)
        assert not torch.allclose(actual_a, actual_b)


def test_two_instances_one_process_aot():
    """AOT variant of test_two_instances_one_process: two loaders, shared
    layer_id, independent correct outputs."""
    keys = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)
    model_a = _model_with_id(0, 0.0)
    model_b = _model_with_id(0, 1000.0)
    expected_a = model_a(keys)
    expected_b = model_b(keys)

    with tempfile.TemporaryDirectory() as dir_a, \
         tempfile.TemporaryDirectory() as dir_b:
        export_aot(model_a, (keys,), dir_a)
        export_aot(model_b, (keys,), dir_b)

        loader_a, layers_a = load_aot(dir_a)
        loader_b, layers_b = load_aot(dir_b)

        actual_a = loader_a.run([keys])[0]
        actual_b = loader_b.run([keys])[0]
        assert torch.allclose(actual_a, expected_a, atol=1e-6)
        assert torch.allclose(actual_b, expected_b, atol=1e-6)
        assert not torch.allclose(actual_a, actual_b)


def test_cpp_inference_gpu():
    """Export NoCache model with known weights via AOTInductor, run C++ binary, verify output."""
    if not os.path.exists(NVE_INFERENCE_BIN):
        print(f"SKIP: C++ binary not found: {NVE_INFERENCE_BIN}")
        return

    model = SimpleModel(layer_type=nve_layers.LayerType.GPULayer,
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
            nve_layers.LayerType.Hierarchical,
            gpu_cache_size=HIER_GPU_CACHE,
            storage=remote_ps,
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
        assert metadata["version"] == 2
        assert len(metadata["layers"]) == 1
        layer0 = metadata["layers"][0]
        assert layer0["layer_type"] == "Hierarchical"
        storage_ref = layer0["storage_ref"]
        ps_cfg = metadata["resources"]["remote_ps"][storage_ref]
        assert ps_cfg["remote_ps_type"] == "plugin"
        assert ps_cfg["plugin_name"] == "libnve-plugin-nvhm.so"
        assert "remote_ps_data" in ps_cfg

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
        layer0 = metadata["layers"][0]
        storage_ref = layer0["storage_ref"]
        ps_cfg = metadata["resources"]["remote_ps"][storage_ref]
        assert ps_cfg["remote_ps_type"] == "plugin"
        assert "remote_ps_data" not in ps_cfg

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
        nve_layers.LayerType.Hierarchical,
        gpu_cache_size=gpu_cache,
        storage=ps,
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
        layer0 = metadata["layers"][0]
        storage_ref = layer0["storage_ref"]
        ps_cfg = metadata["resources"]["remote_ps"][storage_ref]
        assert ps_cfg["remote_ps_type"] == "plugin"
        assert ps_cfg["plugin_name"] == CUSTOM_REMOTE_PLUGIN

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
                nve_layers.LayerType.Hierarchical,
                gpu_cache_size=gpu_cache,
                storage=remote_ps,
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


# ---------------------------------------------------------------------------
# Schema v2: shared storage tests
# ---------------------------------------------------------------------------

class TwoLayerModel(torch.nn.Module):
    """Model with two NVEmbedding layers sharing the same backing storage."""
    def __init__(self, emb_a, emb_b):
        super().__init__()
        self.emb_a = emb_a
        self.emb_b = emb_b

    def forward(self, keys):
        return self.emb_a(keys) + self.emb_b(keys)


def test_v2_metadata_schema():
    """Exported metadata.json is v2 object format with resources + layers."""
    model = SimpleModel(layer_type=nve_layers.LayerType.LinearUVM,
                        optimize_for_training=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)
        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)
        assert isinstance(meta, dict), "v2 metadata must be a JSON object, not array"
        assert meta["version"] == 2
        assert "resources" in meta
        assert "layers" in meta
        assert "memblocks" in meta["resources"]
        assert len(meta["layers"]) == 1
        layer0 = meta["layers"][0]
        assert "storage_ref" in layer0
        ref = layer0["storage_ref"]
        assert ref in meta["resources"]["memblocks"]
        mb_desc = meta["resources"]["memblocks"][ref]
        assert mb_desc["row_elements"] == EMB_SIZE
        assert mb_desc["num_rows"] == NUM_EMB
        assert "float32" in mb_desc["dtype"]
        assert "devices" not in mb_desc, "No device indices should be serialized"
    print("PASS: v2 metadata schema is correct")


def test_shared_memblock_within_model():
    """Two layers sharing one ManagedMemBlock round-trip with one .nve file."""
    num_emb, emb_size, gpu_cache = 512, 8, 1024 * 1024
    nve_dtype = nve.DataType_t.Float32
    shared_mb = nve.ManagedMemBlock(emb_size, num_emb, nve_dtype,
                                    [torch.cuda.current_device()])

    emb_a = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.LinearUVM,
        gpu_cache_size=gpu_cache, storage=shared_mb,
        optimize_for_training=False, device=DEVICE)
    emb_b = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.LinearUVM,
        gpu_cache_size=gpu_cache, storage=shared_mb,
        optimize_for_training=False, device=DEVICE)

    model = TwoLayerModel(emb_a, emb_b)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)

        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)

        # Both layers must share the same storage_ref
        refs = [l["storage_ref"] for l in meta["layers"]]
        assert refs[0] == refs[1], "Shared memblock must produce a single storage_ref"

        # Only one .nve file should exist
        nve_files = os.listdir(os.path.join(save_dir, "weights"))
        assert len(nve_files) == 1, f"Expected 1 weight file, got: {nve_files}"

        # Load and confirm both layers get the same storage object
        layers = load_nve_layers(save_dir)
        assert len(layers) == 2
        assert layers[0].storage is layers[1].storage, \
            "Loaded layers must share the same storage instance"
    print("PASS: shared memblock within model round-trips correctly")


@requires_custom_remote
def test_shared_ps_within_model():
    """Two Hierarchical layers sharing one PS round-trip with one PS instance."""
    num_emb, emb_size, gpu_cache = 256, 4, 1024 * 1024
    ps = _make_custom_remote_ps(num_emb, emb_size)
    keys = torch.arange(num_emb, dtype=torch.int64)
    values = torch.ones(num_emb, emb_size, dtype=torch.float32)
    ps.insert(keys, values)

    emb_a = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.Hierarchical,
        gpu_cache_size=gpu_cache, storage=ps,
        optimize_for_training=False, device=DEVICE)
    emb_b = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.Hierarchical,
        gpu_cache_size=gpu_cache, storage=ps,
        optimize_for_training=False, device=DEVICE)
    model = TwoLayerModel(emb_a, emb_b)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        keys_path = os.path.join(tmpdir, "keys.npy")
        values_path = os.path.join(tmpdir, "values.npy")
        np.save(keys_path, keys.numpy())
        np.save(values_path, values.numpy())
        save_nve(model, save_dir,
                 ps_data_paths={"emb_a": (keys_path, values_path)})

        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)

        refs = [l["storage_ref"] for l in meta["layers"]]
        assert refs[0] == refs[1], "Shared PS must produce a single storage_ref"
        assert len(meta["resources"]["remote_ps"]) == 1

        del model, ps, emb_a, emb_b
        layers = load_nve_layers(save_dir)
        assert len(layers) == 2
        assert layers[0].storage is layers[1].storage, \
            "Loaded layers must share the same storage instance"
    print("PASS: shared PS within model round-trips correctly")


@requires_custom_remote
def test_cross_model_registry_sharing():
    """Two models exported from the same process sharing a PS reuse one instance on load."""
    num_emb, emb_size, gpu_cache = 128, 4, 512 * 1024

    ps = _make_custom_remote_ps(num_emb, emb_size)

    model_a = torch.nn.Module()
    model_a.emb = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.Hierarchical,
        gpu_cache_size=gpu_cache, storage=ps,
        optimize_for_training=False, device=DEVICE)

    model_b = torch.nn.Module()
    model_b.emb = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.Hierarchical,
        gpu_cache_size=gpu_cache, storage=ps,
        optimize_for_training=False, device=DEVICE)

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "model_a")
        dir_b = os.path.join(tmpdir, "model_b")
        save_nve(model_a, dir_a)
        save_nve(model_b, dir_b)

        # Both exports from the same process → same id() → same storage_ref key
        with open(os.path.join(dir_a, "metadata.json")) as f:
            meta_a = json.load(f)
        with open(os.path.join(dir_b, "metadata.json")) as f:
            meta_b = json.load(f)
        ref_a = meta_a["layers"][0]["storage_ref"]
        ref_b = meta_b["layers"][0]["storage_ref"]
        assert ref_a == ref_b, "Same PS object must produce the same resource key"

        del model_a, model_b, ps

        # Load both models with a shared registry — second load must hit the registry
        registry = {}
        layers_a = load_nve_layers(dir_a, registry=registry)
        layers_b = load_nve_layers(dir_b, registry=registry)
        assert layers_a[0].storage is layers_b[0].storage, \
            "Cross-model load with shared registry must reuse the same PS instance"
    print("PASS: cross-model registry sharing works")


def test_resource_remap():
    """resource_remap redirects a file key to an existing registry entry."""
    num_emb, emb_size, gpu_cache = 256, 8, 512 * 1024

    mb = nve.ManagedMemBlock(emb_size, num_emb, nve.DataType_t.Float32,
                             [torch.cuda.current_device()])
    model = torch.nn.Module()
    model.emb = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.LinearUVM,
        gpu_cache_size=gpu_cache, storage=mb,
        optimize_for_training=False, device=DEVICE)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)

        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)
        original_key = meta["layers"][0]["storage_ref"]

        # Simulate: build the memblock via first load, then redirect a second
        # load's key to the already-registered object via remap.
        registry = {}
        layers_a = load_nve_layers(save_dir, registry=registry)
        existing_key = original_key

        # Remap a fictitious key → existing_key (as if a different export used it)
        fake_key = "mb-9999999999"
        # Patch the metadata file to use the fake key
        meta["layers"][0]["storage_ref"] = fake_key
        meta["resources"]["memblocks"][fake_key] = meta["resources"]["memblocks"].pop(original_key)
        # Rename the weight file too
        import shutil
        shutil.copy(
            os.path.join(save_dir, "weights", original_key + ".nve"),
            os.path.join(save_dir, "weights", fake_key + ".nve"))
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        layers_b = load_nve_layers(save_dir, registry=registry,
                                   resource_remap={fake_key: existing_key})
        assert layers_a[0].storage is layers_b[0].storage, \
            "resource_remap must redirect to the existing registry entry"
    print("PASS: resource_remap redirects to existing registry entry")


def test_embed_config_roundtrip():
    """Non-default EmbedLayerConfig fields survive export → load."""
    num_emb, emb_size, gpu_cache = 256, 8, 512 * 1024
    config = {"kernel_mode": 2, "logging_interval": 100,
              "kernel_mode_value_1": 7, "kernel_mode_value_2": 9, "max_modify_size": 1024}

    # Cover both NVEmbedding and the bag path (which forwards config separately).
    emb = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.LinearUVM,
        gpu_cache_size=gpu_cache,
        optimize_for_training=False, device=DEVICE, config=config)
    bag = nve_layers.NVEmbeddingBag(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.LinearUVM, "sum",
        gpu_cache_size=gpu_cache,
        optimize_for_training=False, device=DEVICE, config=config)
    model = TwoLayerModel(emb, bag)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)

        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)
        for layer in meta["layers"]:
            assert layer["config"] == config, \
                "Exported config must match the layer's config"

        layers = load_nve_layers(save_dir)
        assert len(layers) == 2
        for loaded in layers:
            for k, v in config.items():
                assert getattr(loaded.config, k) == v, \
                    f"Loaded config.{k}={getattr(loaded.config, k)} != {v}"
    print("PASS: EmbedLayerConfig round-trips through export/load")


def test_future_version_raises():
    """metadata.json with version > current raises a clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        model = SimpleModel(optimize_for_training=False)
        save_nve(model, save_dir)

        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)
        meta["version"] = meta["version"] + 100
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        with pytest.raises(RuntimeError, match="schema version"):
            load_nve_layers(save_dir)
    print("PASS: future schema version raises RuntimeError")


def test_v1_legacy_load():
    """A v1 (flat array) metadata.json still loads correctly."""
    num_emb, emb_size, gpu_cache = 256, 8, 512 * 1024
    model = torch.nn.Module()
    model.emb = nve_layers.NVEmbedding(
        num_emb, emb_size, torch.float32,
        nve_layers.LayerType.LinearUVM,
        gpu_cache_size=gpu_cache,
        optimize_for_training=False, device=DEVICE)

    keys = torch.arange(16, dtype=torch.int64, device=DEVICE)
    expected = model.emb(keys)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)

        # Read the v2 metadata and convert it to a v1 flat array manually
        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta_v2 = json.load(f)

        layer0 = meta_v2["layers"][0]
        ref = layer0["storage_ref"]
        mb_cfg = meta_v2["resources"]["memblocks"][ref]

        # Build a v1-style layer entry
        v1_entry = {k: v for k, v in layer0.items() if k != "storage_ref"}
        v1_entry["memblock_type"] = mb_cfg["type"]
        v1_metadata = [v1_entry]

        # Rename weight file to the old per-layer naming
        module_path = layer0["module_path"]
        old_name = module_path.replace(".", "_") + ".nve"
        import shutil
        nve_files = os.listdir(os.path.join(save_dir, "weights"))
        shutil.copy(
            os.path.join(save_dir, "weights", nve_files[0]),
            os.path.join(save_dir, "weights", old_name))

        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(v1_metadata, f)

        del model
        layers = load_nve_layers(save_dir)
        assert len(layers) == 1
        actual = layers[0](keys)
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"v1 load mismatch: {(expected - actual).abs().max():.2e}"
    print("PASS: v1 legacy metadata.json loads correctly")


def test_geometry_mismatch_raises():
    """A storage_ref whose geometry disagrees with the layer raises on load."""
    model = SimpleModel(optimize_for_training=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        save_nve(model, save_dir)

        with open(os.path.join(save_dir, "metadata.json")) as f:
            meta = json.load(f)

        # Corrupt the resource's num_rows
        ref = meta["layers"][0]["storage_ref"]
        meta["resources"]["memblocks"][ref]["num_rows"] = 9999
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="num_rows"):
            load_nve_layers(save_dir)
    print("PASS: geometry mismatch raises ValueError")


if __name__ == "__main__":
    test_export_and_load()
    test_cpp_inference()
    test_export_config_nve_parameter_server()
    test_plugin_ps_export_config()
    test_plugin_ps_export_load_roundtrip()
    test_cpp_inference_custom_ps()
    test_v2_metadata_schema()
    test_shared_memblock_within_model()
    test_future_version_raises()
    test_v1_legacy_load()
    test_geometry_mismatch_raises()
