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
torch.export support for NVEmbedding / NVEmbeddingBag models.

Usage::

    from pynve.torch.nve_export import export, export_aot, load

    # Export (torch.export only)
    export(model, example_inputs, "save_dir/")

    # Export with AOTInductor compilation (for C++ inference)
    export_aot(model, example_inputs, "save_dir/")

    # Load on another process / after restart
    #   - non-AOT:  module, layers = load("save_dir/");      out = module(keys)
    #   - AOTI:     loader, layers = load_aot("save_dir/");  out = loader.run([keys])[0]
    # Keep `layers` alive for the lifetime of the loaded model.

Saved layout::

    save_dir/
    ├── model.pt2          # torch.export-ed or AOT-compiled graph
    ├── metadata.json      # per-layer: id, module_path, num_emb, emb_size, dtype, …
    └── embeddings.nve     # LinearUVM weights via nve.TensorFileFormat
"""

import builtins
import os
import json
import torch
import pynve.nve as _nve_mod
from pynve.torch import HAS_TORCH_OPS
from pynve.torch.nve_layers import NVEmbeddingBase, NVEmbedding, NVEmbeddingBag, LayerType, torch_type_to_nve_type

METADATA_SCHEMA_VERSION = 2


def _require_torch_ops(func_name: str):
    if not HAS_TORCH_OPS:
        raise RuntimeError(
            f"{func_name} requires PyTorch custom op bindings (libnve-torch-ops.so). "
            "Build with NVE_DISABLE_TORCH_BINDINGS=OFF (default) and PyTorch >= 2.10.")

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dtype_str(data_type: torch.dtype) -> str:
    return str(data_type).removeprefix("torch.")


def _parse_dtype(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str.removeprefix("torch."))


def _memblock_resource_descriptor(module: NVEmbeddingBase) -> dict:
    """Build the resource descriptor for a memblock-backed layer.

    Geometry is taken from the layer (the memblock has no getters for it).
    No device indices are recorded — topology is resolved at load time.
    UserMemBlock is normalised to its reconstructable form (Managed/host) since
    raw pointers are process-specific and cannot be serialised.
    """
    mb_type = str(module.memblock_type)
    if "User" in mb_type:
        # Normalise: rebuild as Managed on reload (same shape, weights from .nve)
        mb_type = str(_nve_mod.MemBlockType.Managed)
    return {
        "type": mb_type,
        "row_elements": module.embedding_size,
        "num_rows": module.num_embeddings,
        "dtype": _dtype_str(module.data_type),
    }


def _collect_resources(model: torch.nn.Module, *,
                       ps_data_paths: dict[str, tuple[str, str]] | None = None
                       ) -> tuple[dict, dict, dict]:
    """Return (resources, obj_to_key, layer_storage_refs).

    resources     – the top-level resources dict for metadata.json
    obj_to_key    – {id(storage_obj) -> resource_key}, used by _export_weights
    layer_storage_refs – {module_path -> resource_key}
    """
    resources = {"remote_ps": {}, "memblocks": {}}
    obj_to_key: dict[int, str] = {}
    layer_storage_refs: dict[str, str] = {}

    for name, module in model.named_modules():
        if not isinstance(module, NVEmbeddingBase):
            continue

        storage = getattr(module, 'storage', None)
        if storage is None:
            continue

        obj_id = builtins.id(storage)
        if isinstance(storage, _nve_mod.MemBlock):
            if obj_id not in obj_to_key:
                key = f"mb-{obj_id}"
                resources["memblocks"][key] = _memblock_resource_descriptor(module)
                obj_to_key[obj_id] = key
        else:
            if not hasattr(storage, 'export_config'):
                raise ValueError(
                    f"Layer '{name}' has a storage object with no export_config() method. "
                    "PS storage must implement export_config() returning a dict with a "
                    "'remote_ps_type' key.")
            if obj_id not in obj_to_key:
                key = f"ps-{obj_id}"
                resources["remote_ps"][key] = storage.export_config()
                obj_to_key[obj_id] = key
            if ps_data_paths and name in ps_data_paths:
                key = obj_to_key[obj_id]
                keys_path, values_path = ps_data_paths[name]
                resources["remote_ps"][key]["remote_ps_data"] = {
                    "keys": keys_path, "values": values_path}
        layer_storage_refs[name] = obj_to_key[obj_id]

    return resources, obj_to_key, layer_storage_refs


def _export_metadata(model: torch.nn.Module, save_dir: str, *,
                     ps_data_paths: dict[str, tuple[str, str]] | None = None):
    """Collect per-layer metadata and write metadata.json (schema v2)."""
    resources, obj_to_key, layer_storage_refs = _collect_resources(
        model, ps_data_paths=ps_data_paths)

    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, NVEmbeddingBase):
            continue
        entry = {
            "id": module.id,
            "module_path": name,
            "num_embeddings": module.num_embeddings,
            "embedding_size": module.embedding_size,
            "dtype": _dtype_str(module.data_type),
            "layer_type": module.layer_type.name,
            "gpu_cache_size": module.gpu_cache_size,
            "host_cache_size": module.host_cache_size,
            "optimize_for_training": module.optimize_for_training,
            "is_bag": isinstance(module, NVEmbeddingBag),
            "mode": getattr(module, "mode", None),
            "storage_ref": layer_storage_refs.get(name),
            "config": json.loads(module.config.to_json()),
        }
        layers.append(entry)

    doc = {
        "version": METADATA_SCHEMA_VERSION,
        "resources": resources,
        "layers": layers,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(doc, f, indent=2)
    return doc, obj_to_key


def _export_weights(model: torch.nn.Module, save_dir: str,
                    obj_to_key: dict[int, str] | None = None):
    """Save embedding weights, one .nve file per resource key.

    obj_to_key maps id(memblock) -> resource key. When called from save_nve /
    export, it is pre-built by _collect_resources. The first layer referencing
    each key writes the file; subsequent layers sharing the same storage are
    skipped.
    """
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    written: set[str] = set()

    for name, module in model.named_modules():
        if not isinstance(module, NVEmbeddingBase):
            continue
        mb = getattr(module, 'storage', None)
        if not isinstance(mb, _nve_mod.MemBlock):
            continue

        if obj_to_key is not None:
            key = obj_to_key.get(builtins.id(mb))
        else:
            key = None
        # Fall back to per-layer filename for callers that don't supply obj_to_key
        filename = (key or name.replace(".", "_")) + ".nve"

        if filename in written:
            continue
        written.add(filename)

        filepath = os.path.join(weights_dir, filename)
        with open(filepath, "wb") as f:
            _nve_mod.TensorFileFormat().write_table_file_header(f)
            _nve_mod.write_tensor_to_stream(module.emb_layer, f, module.id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_nve(model: torch.nn.Module, save_dir: str, *,
             ps_data_paths: dict[str, tuple[str, str]] | None = None):
    """Save NVE layer metadata and weights to ``save_dir``.

    Produces metadata.json and per-module weight files.

    Args:
        model: Model containing NVEmbedding layers.
        save_dir: Directory to save metadata and weights.
        ps_data_paths: Optional mapping of module path to (keys_file, values_file)
            for Hierarchical layers. These paths are recorded in metadata.json so
            that :func:`load_nve_layers` can reload the PS data automatically.
    """
    os.makedirs(save_dir, exist_ok=True)
    _, obj_to_key = _export_metadata(model, save_dir, ps_data_paths=ps_data_paths)
    _export_weights(model, save_dir, obj_to_key=obj_to_key)


def export(model: torch.nn.Module, example_inputs, save_dir: str, *,
           dynamic_shapes=None,
           ps_data_paths: dict[str, tuple[str, str]] | None = None):
    """Export *model* to ``save_dir`` using torch.export.

    Saves metadata.json, per-module weight files, and model.pt2 (torch.export
    format).  Requires torch custom op bindings.

    Args:
        model: Model containing NVEmbedding layers.
        example_inputs: Example inputs for torch.export.
        save_dir: Directory to save the exported model.
        dynamic_shapes: Optional dynamic shapes for torch.export.
        ps_data_paths: Optional mapping of module path to (keys_file, values_file)
            for Hierarchical layers.
    """
    _require_torch_ops("export")
    save_nve(model, save_dir, ps_data_paths=ps_data_paths)

    exported = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes)
    torch.export.save(exported, os.path.join(save_dir, "model.pt2"))


def export_aot(model: torch.nn.Module, example_inputs, save_dir: str, *,
               dynamic_shapes=None, inductor_configs=None,
               ps_data_paths: dict[str, tuple[str, str]] | None = None):
    """Export *model* to ``save_dir`` with AOTInductor compilation.

    Saves metadata.json, per-module weight files, and model.pt2 (AOT-compiled
    format loadable by C++ AOTIModelPackageLoader).
    Requires torch custom op bindings.

    Args:
        model: Model containing NVEmbedding layers.
        example_inputs: Example inputs for torch.export.
        save_dir: Directory to save the exported model.
        dynamic_shapes: Optional dynamic shapes for torch.export.
        inductor_configs: Optional AOTInductor configuration dict.
        ps_data_paths: Optional mapping of module path to (keys_file, values_file)
            for Hierarchical layers.
    """
    _require_torch_ops("export_aot")
    save_nve(model, save_dir, ps_data_paths=ps_data_paths)

    # The per-layer marker buffer must remain a runtime-updatable constant so
    # the loader can point it (user_managed) at the live layer's marker tensor.
    # Without runtime constant folding, AOTI bakes the buffer inline and it
    # never appears in get_constant_fqns(), making update_constant_buffer a
    # no-op. Force it on (caller overrides win if they pass the key explicitly).
    configs = {"aot_inductor.use_runtime_constant_folding": True}
    if inductor_configs:
        configs.update(inductor_configs)

    exported = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes)
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=os.path.join(save_dir, "model.pt2"),
        inductor_configs=configs,
    )





def _build_ps_from_config(ps_config: dict):
    """Reconstruct a ParameterServerTable from a serialised PS resource entry."""
    if ps_config.get("remote_ps_type") != "plugin":
        raise ValueError(
            f"Unsupported remote_ps_type={ps_config.get('remote_ps_type')!r}; "
            "only 'plugin' is supported.")
    remote_ps = _nve_mod.ParameterServerTable(
        row_elements=ps_config["row_elements"],
        data_type=torch_type_to_nve_type(_parse_dtype(ps_config["data_type"])),
        plugin_name=ps_config["plugin_name"],
        factory_config_json=json.dumps(ps_config.get("factory_config", {})),
        table_config_json=json.dumps(ps_config.get("table_config", {})),
        num_rows=ps_config.get("num_rows", 0),
    )
    ps_data = ps_config.get("remote_ps_data")
    if ps_data:
        remote_ps.insert_keys_from_filepath(
            ps_data["keys"], ps_data["values"], 1 << 20)
    return remote_ps


def _build_memblock_from_descriptor(mb_cfg: dict, device_index: int,
                                    device_type: str,
                                    nvl_devices_override: list[int] | None = None):
    """Reconstruct a MemBlock from a serialised memblock resource descriptor."""
    mb_type_str = mb_cfg["type"]
    row_elements = mb_cfg["row_elements"]
    num_rows = mb_cfg["num_rows"]
    nve_dtype = torch_type_to_nve_type(_parse_dtype(mb_cfg["dtype"]))

    if "NVL" in mb_type_str:
        gpu_ids = _nve_mod.resolve_memblock_devices(
            _nve_mod.MemBlockType.NVL, device_index,
            nvl_devices_override or [])
        return _nve_mod.NVLMemBlock(row_elements, num_rows, nve_dtype, gpu_ids)
    if "Host" in mb_type_str or device_type == "cpu":
        return _nve_mod.HostMemBlock(row_elements, num_rows, nve_dtype)
    # Managed (default) and UserMemBlock-normalised-to-Managed
    gpu_ids = _nve_mod.resolve_memblock_devices(
        _nve_mod.MemBlockType.Managed, device_index, [])
    return _nve_mod.ManagedMemBlock(row_elements, num_rows, nve_dtype, gpu_ids)


def load_nve_layers(save_dir: str, *, gpu_cache_size_override: int = None,
                    device: torch.device | None = None,
                    registry: dict | None = None,
                    resource_remap: dict[str, str] | None = None,
                    nvl_devices_override: list[int] | None = None):
    """Recreate NVEmbedding/NVEmbeddingBag layers from an export directory.

    Supports both v1 (legacy flat array) and v2 (versioned object with
    ``resources`` section) metadata.json formats.

    Shared storage is preserved on v2 exports: resources are built once and
    reused across layers that carry the same ``storage_ref``.

    Args:
        save_dir: Directory containing metadata.json and weights/.
        gpu_cache_size_override: If set, replaces ``gpu_cache_size`` from
            metadata for all layers.
        device: Device to place the loaded layers on. Defaults to the current
            CUDA device when available, otherwise ``cpu``.
        registry: Optional dict shared across ``load_nve_layers`` calls.
            Built resources are inserted under their resource key; on
            subsequent calls the existing object is reused without rebuilding.
            Pass the same dict to multiple calls to share storage across models
            exported from the same Python process.
        resource_remap: Optional ``{key_in_file: key_to_lookup}`` mapping
            applied before the registry check. Useful when two exports from
            different processes should share the same storage.
        nvl_devices_override: If set, passed to
            :func:`pynve.nve.resolve_memblock_devices` as the explicit gpu_ids
            for any NVL memblock resource (overrides the default span).

    Returns a list of layers (caller keeps alive).
    Requires torch custom op bindings (NVE_DISABLE_TORCH_BINDINGS=OFF (default)).
    """
    _require_torch_ops("load_nve_layers")
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    elif device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device_index = device.index if device.type == "cuda" else -1

    with open(os.path.join(save_dir, "metadata.json")) as f:
        raw = json.load(f)

    # Detect schema version: top-level list = v1 legacy; object = v2.
    if isinstance(raw, list):
        return _load_nve_layers_v1(raw, save_dir, device, device_index,
                                   gpu_cache_size_override)

    version = raw.get("version", 1)
    if version > METADATA_SCHEMA_VERSION:
        raise RuntimeError(
            f"metadata.json was exported with schema version {version}, but this "
            f"NVE runtime only supports up to version {METADATA_SCHEMA_VERSION}. "
            "Upgrade the NVE runtime.")

    return _load_nve_layers_v2(raw, save_dir, device, device_index,
                               gpu_cache_size_override, registry,
                               resource_remap, nvl_devices_override)


def _load_nve_layers_v1(metadata: list, save_dir: str,
                         device, device_index: int,
                         gpu_cache_size_override: int | None) -> list:
    """Load from legacy v1 flat-array metadata.json."""
    layers = []
    weights_dir = os.path.join(save_dir, "weights")

    for entry in metadata:
        layer_id = entry["id"]
        module_path = entry["module_path"]
        num_emb = entry["num_embeddings"]
        emb_size = entry["embedding_size"]
        data_type = _parse_dtype(entry["dtype"])
        layer_type = LayerType[entry["layer_type"]]
        is_bag = entry.get("is_bag", False)
        mode = entry.get("mode", None)
        gpu_cache_size = entry["gpu_cache_size"] if gpu_cache_size_override is None else gpu_cache_size_override
        optimize_for_training = entry["optimize_for_training"]
        config = entry.get("config")

        if device.type == "cpu" and layer_type != LayerType.HostLayer:
            raise ValueError(
                f"load_nve_layers: layer '{module_path}' is {layer_type.name}, "
                "which requires a CUDA device, but device='cpu' was requested.")

        if layer_type == LayerType.Hierarchical:
            ps_config = entry.get("remote_ps_config")
            if ps_config is None:
                raise ValueError(
                    f"Hierarchical layer '{module_path}' is missing 'remote_ps_config'.")
            storage = _build_ps_from_config(ps_config)
            host_cache_size = entry.get("host_cache_size", 0)
            layer = _make_layer(num_emb, emb_size, data_type, layer_type,
                                gpu_cache_size, host_cache_size,
                                optimize_for_training, device, layer_id,
                                mode, is_bag, storage=storage, config=config)
        elif layer_type in (LayerType.LinearUVM, LayerType.GPULayer, LayerType.HostLayer):
            nve_dtype = torch_type_to_nve_type(data_type)
            if layer_type == LayerType.HostLayer:
                if device.type == "cpu":
                    storage = _nve_mod.HostMemBlock(emb_size, num_emb, nve_dtype)
                else:
                    storage = _nve_mod.ManagedMemBlock(emb_size, num_emb, nve_dtype, [device_index])
            else:
                storage = None
            layer = _make_layer(num_emb, emb_size, data_type, layer_type,
                                gpu_cache_size, 0,
                                optimize_for_training, device, layer_id,
                                mode, is_bag, storage=storage, config=config)
            weight_file = os.path.join(
                weights_dir, module_path.replace(".", "_") + ".nve")
            with open(weight_file, "rb") as f:
                layer.load_from_stream(f)
        else:
            raise NotImplementedError(
                f"load_nve_layers: layer_type '{layer_type.name}' not yet supported")

        layer._export_module_path = module_path
        layers.append(layer)

    return layers


def _load_nve_layers_v2(doc: dict, save_dir: str,
                         device, device_index: int,
                         gpu_cache_size_override: int | None,
                         registry: dict | None,
                         resource_remap: dict[str, str] | None,
                         nvl_devices_override: list[int] | None) -> list:
    """Load from v2 versioned metadata.json with resources section."""
    if registry is None:
        registry = {}
    if resource_remap is None:
        resource_remap = {}

    resources = doc.get("resources", {})
    ps_resources = resources.get("remote_ps", {})
    mb_resources = resources.get("memblocks", {})
    weights_dir = os.path.join(save_dir, "weights")

    def _resolve_key(key: str) -> str:
        return resource_remap.get(key, key)

    def _get_or_build_ps(key: str) -> object:
        effective = _resolve_key(key)
        if effective in registry:
            return registry[effective]
        if effective not in ps_resources:
            raise ValueError(
                f"storage_ref '{effective}' not found in resources.remote_ps.")
        obj = _build_ps_from_config(ps_resources[effective])
        registry[effective] = obj
        return obj

    def _get_or_build_memblock(key: str, num_emb: int, emb_size: int,
                               data_type: torch.dtype) -> tuple:
        """Returns (memblock, is_new). is_new is False on registry hit so the
        caller skips re-loading weights into an already-populated block."""
        effective = _resolve_key(key)
        if effective in registry:
            mb = registry[effective]
            mb_cfg = mb_resources.get(effective, {})
            if mb_cfg:
                if mb_cfg["row_elements"] != emb_size:
                    raise ValueError(
                        f"Resource '{effective}' row_elements={mb_cfg['row_elements']} "
                        f"!= layer embedding_size={emb_size} — stale or hand-edited metadata?")
                if mb_cfg["num_rows"] != num_emb:
                    raise ValueError(
                        f"Resource '{effective}' num_rows={mb_cfg['num_rows']} "
                        f"!= layer num_embeddings={num_emb} — stale or hand-edited metadata?")
            return mb, False
        if effective not in mb_resources:
            raise ValueError(
                f"storage_ref '{effective}' not found in resources.memblocks.")
        mb_cfg = mb_resources[effective]
        # Assert geometry
        if mb_cfg["row_elements"] != emb_size:
            raise ValueError(
                f"Resource '{effective}' row_elements={mb_cfg['row_elements']} "
                f"!= layer embedding_size={emb_size} — stale or hand-edited metadata?")
        if mb_cfg["num_rows"] != num_emb:
            raise ValueError(
                f"Resource '{effective}' num_rows={mb_cfg['num_rows']} "
                f"!= layer num_embeddings={num_emb} — stale or hand-edited metadata?")
        obj = _build_memblock_from_descriptor(
            mb_cfg, device_index, device.type, nvl_devices_override)
        registry[effective] = obj
        return obj, True

    layers = []

    for entry in doc["layers"]:
        layer_id = entry["id"]
        module_path = entry["module_path"]
        num_emb = entry["num_embeddings"]
        emb_size = entry["embedding_size"]
        data_type = _parse_dtype(entry["dtype"])
        layer_type = LayerType[entry["layer_type"]]
        is_bag = entry.get("is_bag", False)
        mode = entry.get("mode", None)
        gpu_cache_size = entry["gpu_cache_size"] if gpu_cache_size_override is None else gpu_cache_size_override
        host_cache_size = entry.get("host_cache_size", 0)
        optimize_for_training = entry["optimize_for_training"]
        storage_ref = entry.get("storage_ref")
        config = entry.get("config")

        if device.type == "cpu" and layer_type != LayerType.HostLayer:
            raise ValueError(
                f"load_nve_layers: layer '{module_path}' is {layer_type.name}, "
                "which requires a CUDA device, but device='cpu' was requested.")

        if layer_type == LayerType.Hierarchical:
            if not storage_ref:
                raise ValueError(
                    f"Hierarchical layer '{module_path}' is missing 'storage_ref'.")
            storage = _get_or_build_ps(storage_ref)
            layer = _make_layer(num_emb, emb_size, data_type, layer_type,
                                gpu_cache_size, host_cache_size,
                                optimize_for_training, device, layer_id,
                                mode, is_bag, storage=storage, config=config)

        elif layer_type in (LayerType.LinearUVM, LayerType.GPULayer, LayerType.HostLayer):
            if storage_ref:
                storage, is_new = _get_or_build_memblock(storage_ref, num_emb, emb_size, data_type)
            else:
                raise ValueError(
                    f"v2 metadata: layer '{module_path}' has no 'storage_ref'. "
                    "Every layer in a v2 export must reference a resource.")

            layer = _make_layer(num_emb, emb_size, data_type, layer_type,
                                gpu_cache_size, 0,
                                optimize_for_training, device, layer_id,
                                mode, is_bag, storage=storage, config=config)

            # Load weights only when the memblock was freshly built. On a
            # registry hit (shared layer within a call, or a shared registry
            # across load_nve_layers calls) the block is already populated.
            if is_new:
                weight_key = storage_ref or module_path.replace(".", "_")
                weight_file = os.path.join(weights_dir, weight_key + ".nve")
                with open(weight_file, "rb") as f:
                    layer.load_from_stream(f)
        else:
            raise NotImplementedError(
                f"load_nve_layers: layer_type '{layer_type.name}' not yet supported")

        layer._export_module_path = module_path
        layers.append(layer)

    return layers


def _make_layer(num_emb, emb_size, data_type, layer_type,
                gpu_cache_size, host_cache_size,
                optimize_for_training, device, layer_id,
                mode, is_bag, *, storage=None, config=None):
    """Construct an NVEmbedding or NVEmbeddingBag with the given parameters."""
    kwargs = dict(optimize_for_training=optimize_for_training, device=device,
                  id=layer_id, config=config)
    if layer_type == LayerType.LinearUVM:
        kwargs["gpu_cache_size"] = gpu_cache_size
    if layer_type == LayerType.Hierarchical:
        kwargs["gpu_cache_size"] = gpu_cache_size
        kwargs["host_cache_size"] = host_cache_size
        kwargs["storage"] = storage
    elif storage is not None:
        kwargs["storage"] = storage
    if is_bag:
        return NVEmbeddingBag(num_emb, emb_size, data_type, layer_type, mode, **kwargs)
    return NVEmbedding(num_emb, emb_size, data_type, layer_type, **kwargs)


def _marker_fqn(module_path: str) -> str:
    """Fully-qualified name of a layer's marker buffer in the exported graph."""
    return f"{module_path}.marker_tensor" if module_path else "marker_tensor"


def load(save_dir: str, *, gpu_cache_size_override: int = None,
         device: torch.device | None = None):
    """Load a non-AOT model previously saved with :func:`export`.

    Recreates NVE layers (with weights loaded and registered in
    NVELayerRegistry by marker ptr), loads the ExportedProgram from model.pt2,
    materializes its module ONCE, and reassigns each layer's marker constant in
    that module to the live layer's ``marker_tensor`` (PUSH) — so the op
    dispatches to the right binding. Returns the ready-to-call module.

    Args:
        save_dir: Directory containing model.pt2, metadata.json and weights/.
        gpu_cache_size_override: If set, replaces ``gpu_cache_size`` from
            metadata for all layers.
        device: CUDA device to place loaded layers on. Defaults to the current
            CUDA device.

    Returns:
        (torch.nn.Module, list[NVEmbeddingBase]): a callable module
        (``module(keys)``) and the NVE layers. The caller MUST keep the layers
        alive for the lifetime of the module — if they are garbage-collected,
        the marker tensors free and the custom ops fail to find them in the
        registry.
    """
    layers = load_nve_layers(save_dir,
                             gpu_cache_size_override=gpu_cache_size_override,
                             device=device)

    ep = torch.export.load(os.path.join(save_dir, "model.pt2"))
    module = ep.module()
    # PUSH: make the running graph feed each layer's own marker_tensor, so its
    # registered data_ptr matches what the op receives. ExportedProgram.module()
    # builds a fresh GraphModule per call, so we build it once and return it.
    for layer in layers:
        mp = layer._export_module_path
        sub = module.get_submodule(mp) if mp else module
        sub.marker_tensor = layer.marker_tensor

    return module, layers


def load_aot(save_dir: str, *, gpu_cache_size_override: int = None,
             device: torch.device | None = None):
    """Load an AOTInductor model previously saved with :func:`export_aot`.

    Recreates NVE layers (registered by marker ptr), constructs an
    ``AOTIModelPackageLoader``, and points each baked marker constant at the
    live layer's ``marker_tensor`` via ``update_constant_buffer(user_managed)``
    in BOTH constant buffers (so the dispatch ptr survives a bare swap). The op
    then receives the registered ptr.

    Requires ``export_aot`` to have set ``use_runtime_constant_folding=True``
    (the default) — otherwise the marker is baked inline and not updatable.

    Returns:
        (AOTIModelPackageLoader, list[NVEmbeddingBase]): the loader (call
        ``loader.run([keys])``) and the NVE layers. Keep both alive: the loader
        holds only a non-owning handle to each marker tensor.
    """
    _require_torch_ops("load_aot")
    from torch._C._aoti import AOTIModelPackageLoader
    # Resolve the device the same way load_nve_layers does: default to the
    # current CUDA device, and fill in the index only for an index-less CUDA
    # device. A CPU device is honored as-is (HostLayer), never silently forced
    # to CUDA. AOTIModelPackageLoader takes a c10::DeviceIndex, so pass -1 for CPU.
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device_index = device.index if device.type == "cuda" else -1

    layers = load_nve_layers(save_dir,
                             gpu_cache_size_override=gpu_cache_size_override,
                             device=device)

    loader = AOTIModelPackageLoader(
        os.path.join(save_dir, "model.pt2"), "model", False, 1, device_index)
        
    _apply_markers(loader, layers)
    return loader, layers


def _apply_markers(loader, layers):
    """Point each layer's marker constant (both buffers) at its marker_tensor."""
    fqns = set(loader.get_constant_fqns())
    for layer in layers:
        fqn = _marker_fqn(layer._export_module_path)
        if fqn not in fqns:
            raise RuntimeError(
                f"load_aot: marker constant '{fqn}' is not an updatable constant "
                f"in the AOTI package (available: {sorted(fqns)}). It was likely "
                "baked inline; re-export with export_aot, which sets "
                "aot_inductor.use_runtime_constant_folding=True.")
        for use_inactive in (False, True):
            loader.load_constants(
                {fqn: layer.marker_tensor},
                use_inactive,
                False,            # check_full_update
                True)             # user_managed


def rebind_markers(loader, layers):
    """Re-point every layer's marker constant at its ``marker_tensor``.

    Call after a host-side ``swap_constant_buffer`` that was preceded by an
    inactive-buffer rebuild (e.g. a dense-weight hot-swap), which re-copies the
    marker as a fresh non-user-managed tensor and breaks dispatch. Idempotent;
    no re-registration is needed since the marker tensor (and its registered
    data_ptr) is unchanged — only the container's pointer-to-it is restored.
    """
    _apply_markers(loader, layers)
