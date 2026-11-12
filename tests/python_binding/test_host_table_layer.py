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

# HostLayer backed by a host hash table (NVHashMap) instead of a LinearHostTable.

import json
import os
import tempfile

import numpy as np
import pytest
import torch

import pynve.nve as nve
import pynve.torch.nve_export as nve_export
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from conftest import make_layer, requires_nvhm

pytestmark = requires_nvhm

NUM_EMB = 64
EMB_SIZE = 8
# Sparse, non-linear keys (well beyond NUM_EMB) that a LinearHostTable can't hold.
KEYS = torch.tensor([7, 1 << 20, 123456789, (1 << 40) + 3], dtype=torch.int64)

DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


def _rows(keys):
    # Row for key k holds (k % 1000) + column index.
    return ((keys % 1000).float().unsqueeze(1) + torch.arange(EMB_SIZE, dtype=torch.float32)).contiguous()


def _make_ps():
    ps = nve_ps.NVEParameterServer(0, EMB_SIZE, torch.float32, ps_type=nve.NVHashMap)
    ps.insert(KEYS, _rows(KEYS))
    return ps


def _host_table_layer(storage, device, **kwargs):
    return make_layer(NUM_EMB, EMB_SIZE, torch.float32, nve_layers.LayerType.HostLayer, device,
                      storage=storage, **kwargs)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.type)
@pytest.mark.parametrize("raw_table", [False, True], ids=["ps", "table"])
def test_lookup_hits_and_misses(device, raw_table):
    ps = _make_ps()
    default = torch.full((EMB_SIZE,), -1.0)
    layer = _host_table_layer(ps.parameter_server if raw_table else ps, device,
                              config={"default_embedding": default})
    assert layer.table_ids == {0: 'host'}
    missing = torch.tensor([8, 999], dtype=torch.int64)
    out = layer(torch.cat([KEYS, missing]).to(device)).cpu()
    assert torch.equal(out[:len(KEYS)], _rows(KEYS))
    assert torch.equal(out[len(KEYS):], default.expand(len(missing), EMB_SIZE))


@pytest.mark.parametrize("bag_mode", ["sum", "mean"])
def test_pooled_lookup(bag_mode):
    layer = make_layer(NUM_EMB, EMB_SIZE, torch.float32, nve_layers.LayerType.HostLayer,
                       torch.device("cpu"), storage=_make_ps(), bag_mode=bag_mode)
    offsets = torch.tensor([0, 2, len(KEYS)], dtype=torch.int64)
    out = layer(KEYS, offsets)
    rows = _rows(KEYS)
    pool = torch.sum if bag_mode == "sum" else torch.mean
    expected = torch.stack([pool(rows[:2], dim=0), pool(rows[2:], dim=0)])
    torch.testing.assert_close(out, expected)


def test_insert_update_erase_through_layer():
    ps = _make_ps()
    layer = _host_table_layer(ps, torch.device("cpu"))
    new_keys = torch.tensor([42, 1 << 33], dtype=torch.int64)
    layer.insert(new_keys, _rows(new_keys), 0)
    assert torch.equal(layer(new_keys), _rows(new_keys))
    assert ps.num_keys() == len(KEYS) + len(new_keys)

    updates = torch.full((len(new_keys), EMB_SIZE), 5.0)
    layer.update(new_keys, updates)
    assert torch.equal(layer(new_keys), updates)

    layer.erase(new_keys, 0)
    assert torch.equal(layer(new_keys), torch.zeros(len(new_keys), EMB_SIZE))
    assert ps.num_keys() == len(KEYS)


def test_weight_init_rejected():
    with pytest.raises(ValueError, match="weight_init is not supported for a table-backed HostLayer"):
        _host_table_layer(_make_ps(), torch.device("cpu"),
                          weight_init=torch.zeros(NUM_EMB, EMB_SIZE))


def test_data_type_mismatch_rejected():
    ps = nve_ps.NVEParameterServer(0, EMB_SIZE, torch.float16, ps_type=nve.NVHashMap)
    with pytest.raises(ValueError, match="doesn't match layer data type"):
        _host_table_layer(ps, torch.device("cpu"))
    # Raw tables are checked by the binding.
    with pytest.raises(RuntimeError, match="doesn't match layer data type"):
        _host_table_layer(ps.parameter_server, torch.device("cpu"))


def test_row_size_mismatch_rejected():
    # A larger table row is rejected too: the layer gathers whole table rows.
    ps = nve_ps.NVEParameterServer(0, EMB_SIZE * 2, torch.float32, ps_type=nve.NVHashMap)
    with pytest.raises(RuntimeError, match="doesn't match layer row size"):
        _host_table_layer(ps.parameter_server, torch.device("cpu"))


def test_key_size_mismatch_rejected():
    ps = nve_ps.NVEParameterServer(0, EMB_SIZE, torch.float32, ps_type=nve.NVHashMap,
                                   extra_params={"table": {"key_size": 4}})
    with pytest.raises(RuntimeError, match="doesn't match layer key size"):
        _host_table_layer(ps.parameter_server, torch.device("cpu"))


def test_weight_stream_unsupported():
    layer = _host_table_layer(_make_ps(), torch.device("cpu"))
    with tempfile.TemporaryFile() as f:
        with pytest.raises(RuntimeError, match="not supported for a table-backed HostLayer"):
            nve.write_tensor_to_stream(layer.emb_layer, f, layer.id)


class _Model(torch.nn.Module):
    """One HostLayer per storage; forward returns the first layer's lookup."""

    def __init__(self, *storages, device=torch.device("cpu")):
        super().__init__()
        for i, storage in enumerate(storages):
            setattr(self, f"emb{i}", _host_table_layer(storage, device))

    def forward(self, keys):
        return self.emb0(keys)


def _data_paths(tmpdir):
    keys_path = os.path.join(tmpdir, "keys.npy")
    values_path = os.path.join(tmpdir, "values.npy")
    np.save(keys_path, KEYS.numpy())
    np.save(values_path, _rows(KEYS).numpy())
    return keys_path, values_path


def _read_metadata(save_dir):
    with open(os.path.join(save_dir, "metadata.json")) as f:
        return json.load(f)


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _Model(_make_ps())
        save_dir = os.path.join(tmpdir, "export")
        nve_export.save_nve(model, save_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        # Table data is saved through the PS, not as a weight file.
        assert os.listdir(os.path.join(save_dir, "weights")) == []
        meta = _read_metadata(save_dir)
        ps_cfg = meta["resources"]["remote_ps"][meta["layers"][0]["storage_ref"]]
        assert ps_cfg["plugin_name"] == "libnve-plugin-nvhm.so"
        assert "remote_ps_data" in ps_cfg

        del model
        layers = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))
        assert len(layers) == 1
        assert isinstance(layers[0].storage, nve.Table)
        assert torch.equal(layers[0](KEYS), _rows(KEYS))


def test_save_raw_table_and_resave():
    # Raw-table storage is exported through export_config_json, before and after a reload.
    with tempfile.TemporaryDirectory() as tmpdir:
        ps = _make_ps()
        model = _Model(ps.parameter_server)
        save_dir = os.path.join(tmpdir, "export")
        nve_export.save_nve(model, save_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        ps_cfg = _read_metadata(save_dir)["resources"]["remote_ps"]
        assert [c["plugin_name"] for c in ps_cfg.values()] == ["libnve-plugin-nvhm.so"]

        del model, ps
        reloaded = _Model()
        reloaded.emb0 = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))[0]
        resave_dir = os.path.join(tmpdir, "resave")
        nve_export.save_nve(reloaded, resave_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        layers = nve_export.load_nve_layers(resave_dir, device=torch.device("cpu"))
        assert torch.equal(layers[0](KEYS), _rows(KEYS))


def test_save_load_without_data_paths():
    # Config-only export: the reloaded table is empty, so every key misses.
    default = torch.full((EMB_SIZE,), -1.0)
    ps = _make_ps()
    model = torch.nn.Module()
    model.emb = _host_table_layer(ps, torch.device("cpu"), config={"default_embedding": default})
    with tempfile.TemporaryDirectory() as save_dir:
        nve_export.save_nve(model, save_dir)
        meta = _read_metadata(save_dir)
        assert "remote_ps_data" not in meta["resources"]["remote_ps"][meta["layers"][0]["storage_ref"]]

        del model, ps
        layers = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))
        assert layers[0].storage.num_keys() == 0
        assert torch.equal(layers[0](KEYS), default.expand(len(KEYS), EMB_SIZE))


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.type)
@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason="torch.export.save can't serialize the sparse weight placeholder "
                          "(same limitation as Hierarchical)")
def test_export_load(device):
    keys = KEYS.to(device)
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _Model(_make_ps(), device=device)
        save_dir = os.path.join(tmpdir, "export")
        nve_export.export(model, (keys,), save_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        del model
        module, layers = nve_export.load(save_dir, device=device)
        out = module(keys)
        assert out.device.type == device.type
        assert torch.equal(out.cpu(), _rows(KEYS))


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.type)
def test_export_aot_load_aot(device):
    keys = KEYS.to(device)
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _Model(_make_ps(), device=device)
        save_dir = os.path.join(tmpdir, "export")
        nve_export.export_aot(model, (keys,), save_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        del model
        loader, layers = nve_export.load_aot(save_dir, device=device)
        out = loader.run([keys])[0]
        assert out.device.type == device.type
        assert torch.equal(out.cpu(), _rows(KEYS))


def test_shared_ps_within_model():
    ps = _make_ps()
    model = _Model(ps, ps)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        nve_export.save_nve(model, save_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        meta = _read_metadata(save_dir)
        refs = [l["storage_ref"] for l in meta["layers"]]
        assert refs[0] == refs[1]
        assert len(meta["resources"]["remote_ps"]) == 1

        del model, ps
        layers = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"))
        assert len(layers) == 2
        assert layers[0].storage is layers[1].storage
        for layer in layers:
            assert torch.equal(layer(KEYS), _rows(KEYS))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Hierarchical requires CUDA")
def test_ps_shared_with_hierarchical():
    # One PS backs both a Hierarchical layer and a HostLayer.
    device = torch.device("cuda")
    ps = _make_ps()
    model = torch.nn.Module()
    model.hier = make_layer(NUM_EMB, EMB_SIZE, torch.float32, nve_layers.LayerType.Hierarchical,
                            device, storage=ps, gpu_cache_size=1024 * 1024)
    model.host = _host_table_layer(ps, device)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        nve_export.save_nve(model, save_dir, ps_data_paths={"host": _data_paths(tmpdir)})
        meta = _read_metadata(save_dir)
        assert len(meta["resources"]["remote_ps"]) == 1

        del model, ps
        layers = nve_export.load_nve_layers(save_dir, device=device)
        assert [l.layer_type for l in layers] == [nve_layers.LayerType.Hierarchical,
                                                  nve_layers.LayerType.HostLayer]
        assert layers[0].storage is layers[1].storage
        for layer in layers:
            assert torch.equal(layer(KEYS.to(device)).cpu(), _rows(KEYS))


def test_cross_model_registry_sharing():
    ps = _make_ps()
    model_a, model_b = _Model(ps), _Model(ps)
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "model_a")
        dir_b = os.path.join(tmpdir, "model_b")
        nve_export.save_nve(model_a, dir_a, ps_data_paths={"emb0": _data_paths(tmpdir)})
        nve_export.save_nve(model_b, dir_b)
        # Same PS object in one process gives the same resource key.
        assert (_read_metadata(dir_a)["layers"][0]["storage_ref"] ==
                _read_metadata(dir_b)["layers"][0]["storage_ref"])

        del model_a, model_b, ps
        registry = {}
        layers_a = nve_export.load_nve_layers(dir_a, device=torch.device("cpu"), registry=registry)
        layers_b = nve_export.load_nve_layers(dir_b, device=torch.device("cpu"), registry=registry)
        assert layers_a[0].storage is layers_b[0].storage
        # dir_b has no data paths; its rows come from the shared table loaded by dir_a.
        assert torch.equal(layers_b[0](KEYS), _rows(KEYS))


def test_resource_remap():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "export")
        nve_export.save_nve(_Model(_make_ps()), save_dir, ps_data_paths={"emb0": _data_paths(tmpdir)})
        registry = {}
        layers_a = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"), registry=registry)

        # Rename the resource as if it came from another export, then remap it to the loaded PS.
        meta = _read_metadata(save_dir)
        original_key = meta["layers"][0]["storage_ref"]
        fake_key = "ps-9999999999"
        meta["layers"][0]["storage_ref"] = fake_key
        meta["resources"]["remote_ps"][fake_key] = meta["resources"]["remote_ps"].pop(original_key)
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

        layers_b = nve_export.load_nve_layers(save_dir, device=torch.device("cpu"), registry=registry,
                                              resource_remap={fake_key: original_key})
        assert layers_a[0].storage is layers_b[0].storage
        assert torch.equal(layers_b[0](KEYS), _rows(KEYS))
