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

"""Load-time config overrides shared by the C++ and Python loaders."""

import copy
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
import pynve.torch.nve_export as nve_export
import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_export import LayerConfigOverrides


@pytest.mark.parametrize("version", [1, 2])
def test_per_layer_overrides_load_and_change_missing_rows(tmp_path, version):
    model = torch.nn.Module()
    weights = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    for name in ("user", "item", "other"):
        setattr(model, name, nve_layers.NVEmbedding(
            8, 4, torch.float32, nve_layers.LayerType.HostLayer,
            weight_init=weights, device=torch.device("cpu"), optimize_for_training=False,
            config={"logging_interval": -1, "default_embedding": torch.ones(4)}))
    nve_export.save_nve(model, str(tmp_path))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if version == 1:
        for entry in metadata["layers"]:
            shutil.copyfile(tmp_path / "weights" / (entry["storage_ref"] + ".nve"),
                            tmp_path / "weights" / (entry["module_path"] + ".nve"))
            entry.pop("storage_ref")
        metadata = metadata["layers"]
        metadata_path.write_text(json.dumps(metadata))
    original_metadata = metadata_path.read_text()
    user_default = torch.full((4,), 3.5).view(torch.uint8).tolist()
    item_default = torch.full((4,), 7.0).view(torch.uint8).tolist()
    overrides = LayerConfigOverrides(
        defaults={"config": {"logging_interval": 100, "disable_threadpool": True}},
        per_layer={
            "user": {"optimize_for_training": False,
                     "config": {"logging_interval": -1, "default_embedding": user_default}},
            "item": {"config": {"default_embedding": item_default,
                                "disable_threadpool": False}},
        })
    original_overrides = copy.deepcopy(overrides)
    layers = nve_export.load_nve_layers(str(tmp_path), device=torch.device("cpu"),
                                       overrides=overrides)
    assert [layer._export_module_path for layer in layers] == ["user", "item", "other"]
    for layer, expected in zip(layers, (3.5, 7.0, 1.0)):
        output = layer(torch.tensor([2, 100], dtype=torch.int64))
        assert torch.equal(output[0], weights[2])
        assert torch.equal(output[1], torch.full((4,), expected))
    assert layers[0].config.logging_interval == -1
    assert layers[1].config.logging_interval == 100
    assert layers[2].config.logging_interval == 100
    assert layers[0].optimize_for_training is False
    assert [layer.config.disable_threadpool for layer in layers] == [True, False, True]
    assert overrides == original_overrides
    assert metadata_path.read_text() == original_metadata


@pytest.mark.parametrize("version", [1, 2])
def test_cache_overrides_reach_hierarchical_constructor(tmp_path, monkeypatch, version):
    entries = [dict(id=index, module_path=name, num_embeddings=8, embedding_size=4,
                    dtype="float32", layer_type="Hierarchical", gpu_cache_size=1024,
                    host_cache_size=2048, optimize_for_training=True,
                    config={"kernel_mode": 2})
               for index, name in enumerate(("user", "item"))]
    if version == 1:
        for entry in entries:
            entry["remote_ps_config"] = {}
        metadata = entries
    else:
        for entry in entries:
            entry["storage_ref"] = "ps-shared"
        metadata = {"version": 2, "layers": entries,
                    "resources": {"remote_ps": {"ps-shared": {}}}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(nve_export, "_build_ps_from_config", lambda config: object())
    calls = []

    def make_layer(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(nve_export, "_make_layer", make_layer)
    overrides = LayerConfigOverrides(
        defaults={"gpu_cache_size": 4096, "config": {"logging_interval": 100}},
        per_layer={"user": {"gpu_cache_size": 0, "host_cache_size": 0,
                            "optimize_for_training": False,
                            "config": {"default_embedding": []}},
                   "item": {"host_cache_size": 8192,
                            "config": {"logging_interval": -1}}})
    nve_export.load_nve_layers(str(tmp_path), device=torch.device("cuda", 0), overrides=overrides)
    assert calls[0][0][4:7] == (0, 0, False)
    assert calls[1][0][4:7] == (4096, 8192, True)
    assert calls[0][1]["config"] == {
        "kernel_mode": 2, "logging_interval": 100, "default_embedding": []}
    assert calls[1][1]["config"] == {"kernel_mode": 2, "logging_interval": -1}


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("overrides,match", [
    (LayerConfigOverrides(per_layer={"typo": {"gpu_cache_size": 1}}), "Unknown layer"),
    (LayerConfigOverrides(defaults={"gpu_cache_sze": 1}), "Unsupported layer override"),
    (LayerConfigOverrides(per_layer={"item": {"config": {"kernel_mod": 1}}}), "kernel_mod"),
    (LayerConfigOverrides(defaults={"gpu_cache_size": -1}), "nonnegative integer"),
    (LayerConfigOverrides(defaults={"gpu_cache_size": None}), "nonnegative integer"),
    (LayerConfigOverrides(defaults={"optimize_for_training": 0}), "boolean"),
    (LayerConfigOverrides(defaults={"config": None}), "must be an object"),
    (LayerConfigOverrides(defaults={"config": {"disable_threadpool": 1}}), "boolean"),
    (LayerConfigOverrides(defaults={"config": {"default_embedding": [256]}}), "integers"),
    (LayerConfigOverrides(per_layer=[]), "per_layer must be an object"),
])
def test_invalid_overrides_fail_before_loading_resources(tmp_path, monkeypatch, version,
                                                        overrides, match):
    entries = [{"module_path": "user", "layer_type": "Hierarchical"},
               {"module_path": "item", "layer_type": "Hierarchical"}]
    metadata = entries if version == 1 else {"version": 2, "layers": entries}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid overrides must fail before loading any layers or resources")

    monkeypatch.setattr(nve_export, "_load_nve_layers_v1", unexpected_load)
    monkeypatch.setattr(nve_export, "_load_nve_layers_v2", unexpected_load)
    with pytest.raises(RuntimeError, match=match):
        nve_export.load_nve_layers(str(tmp_path), device=torch.device("cpu"), overrides=overrides)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("config", [None, [], "invalid", 1, True])
@pytest.mark.parametrize("use_defaults", [True, False])
def test_non_object_metadata_config_fails_before_loading_resources(
        tmp_path, monkeypatch, version, config, use_defaults):
    entries = [{"module_path": "user", "layer_type": "HostLayer", "config": config}]
    metadata = entries if version == 1 else {"version": 2, "layers": entries}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    original_metadata = metadata_path.read_text()
    patch = {"config": {"logging_interval": 100}}
    overrides = (LayerConfigOverrides(defaults=patch) if use_defaults else
                 LayerConfigOverrides(per_layer={"user": patch}))

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid metadata must fail before loading any layers or resources")

    monkeypatch.setattr(nve_export, "_load_nve_layers_v1", unexpected_load)
    monkeypatch.setattr(nve_export, "_load_nve_layers_v2", unexpected_load)
    with pytest.raises(RuntimeError, match="Layer 'user' config must be an object"):
        nve_export.load_nve_layers(str(tmp_path), device=torch.device("cpu"), overrides=overrides)
    assert metadata_path.read_text() == original_metadata


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("use_defaults", [True, False])
@pytest.mark.parametrize("layer_type,patch", [
    ("LinearUVM", {"host_cache_size": 0}),
    ("GPULayer", {"host_cache_size": 8192}),
    ("HostLayer", {"host_cache_size": 8192}),
    ("GPULayer", {"gpu_cache_size": 0}),
    ("HostLayer", {"gpu_cache_size": 4096}),
    ("Hierarchical", {"config": {"kernel_mode": 0}}),
    ("GPULayer", {"config": {"kernel_mode_value_1": 1024}}),
    ("HostLayer", {"config": {"kernel_mode_value_2": 16}}),
    ("GPULayer", {"config": {"max_modify_size": 0}}),
    ("HostLayer", {"config": {"max_modify_size": 100}}),
    ("Hierarchical", {"config": {"default_row_index": -1}}),
    ("HostLayer", {"config": {"default_row_index": 3}}),
    ("LinearUVM", {"config": {"default_embedding": []}}),
    ("GPULayer", {"config": {"default_embedding": [1, 2]}}),
    ("LinearUVM", {"config": {"disable_threadpool": False}}),
    ("Hierarchical", {"config": {"disable_threadpool": True}}),
    ("GPULayer", {"config": {"disable_threadpool": False}}),
])
def test_unsupported_knobs_fail_before_loading_resources(
        tmp_path, monkeypatch, version, use_defaults, layer_type, patch):
    entries = [{"module_path": "user", "layer_type": layer_type}]
    metadata = entries if version == 1 else {"version": 2, "layers": entries}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    overrides = (LayerConfigOverrides(defaults=patch) if use_defaults else
                 LayerConfigOverrides(per_layer={"user": patch}))

    def unexpected_load(*args, **kwargs):
        pytest.fail("Unsupported overrides must fail before loading any layers or resources")

    monkeypatch.setattr(nve_export, "_load_nve_layers_v1", unexpected_load)
    monkeypatch.setattr(nve_export, "_load_nve_layers_v2", unexpected_load)
    field = next(iter(patch["config"] if "config" in patch else patch))
    with pytest.raises(RuntimeError, match="Unsupported layer override") as error:
        nve_export.load_nve_layers(str(tmp_path), device=torch.device("cpu"), overrides=overrides)
    message = str(error.value)
    assert "layer 'user' (" + layer_type + ")" in message
    assert field in message


def test_root_layer_without_exported_config():
    overrides = LayerConfigOverrides(per_layer={"": {"config": {"default_row_index": 3}}})
    assert nve_export._apply_layer_config_overrides(
        [{"module_path": "", "layer_type": "GPULayer"}], overrides) == [
        {"module_path": "", "layer_type": "GPULayer", "config": {"default_row_index": 3}}]


@pytest.mark.parametrize("aot", [False, True])
def test_model_loaders_forward_overrides(tmp_path, monkeypatch, aot):
    overrides = LayerConfigOverrides(defaults={"config": {"logging_interval": 100}})
    calls = []

    def load_layers(save_dir, **kwargs):
        calls.append(kwargs["overrides"])
        return []

    monkeypatch.setattr(nve_export, "load_nve_layers", load_layers)
    if aot:
        class Loader:
            def __init__(self, *args):
                pass

            def get_constant_fqns(self):
                return []

        monkeypatch.setattr(torch._C._aoti, "AOTIModelPackageLoader", Loader)
        nve_export.load_aot(str(tmp_path), device=torch.device("cpu"), overrides=overrides)
    else:
        monkeypatch.setattr(torch.export, "load", lambda path: SimpleNamespace(module=lambda: object()))
        nve_export.load(str(tmp_path), device=torch.device("cpu"), overrides=overrides)
    assert calls == [overrides]
