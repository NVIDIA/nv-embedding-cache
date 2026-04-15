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
    loaded = load("save_dir/")
    out = loaded.module()(keys)

Saved layout::

    save_dir/
    ├── model.pt2          # torch.export-ed or AOT-compiled graph
    ├── metadata.json      # per-layer: id, module_path, num_emb, emb_size, dtype, …
    └── embeddings.nve     # LinearUVM weights via nve.TensorFileFormat
"""

import os
import json
import warnings
import torch
import pynve.nve as _nve_mod
from pynve.torch import HAS_TORCH_OPS
from pynve.torch.nve_layers import NVEmbeddingBase, NVEmbedding, NVEmbeddingBag, CacheType


def _require_torch_ops(func_name: str):
    if not HAS_TORCH_OPS:
        raise RuntimeError(
            f"{func_name} requires PyTorch custom op bindings (libnve-torch-ops.so). "
            "Build with NVE_DISABLE_TORCH_BINDINGS=OFF (default) and PyTorch >= 2.10.")

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _export_metadata(model: torch.nn.Module, save_dir: str):
    """Collect per-layer metadata and write to metadata.json."""
    metadata = []
    for name, module in model.named_modules():
        if not isinstance(module, NVEmbeddingBase):
            continue
        metadata.append({
            "id": module.id,
            "module_path": name,
            "num_embeddings": module.num_embeddings,
            "embedding_size": module.embedding_size,
            "dtype": str(module.data_type),
            "cache_type": module.cache_type.name,
            "gpu_cache_size": module.gpu_cache_size,
            "host_cache_size": module.host_cache_size,
            "memblock_type": str(module.memblock_type) if hasattr(module, 'memblock_type') else None,
            "optimize_for_training": module.optimize_for_training,
            "is_bag": isinstance(module, NVEmbeddingBag),
            "mode": getattr(module, "mode", None),
        })
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def _export_weights(model: torch.nn.Module, save_dir: str):
    """Save LinearUVM embedding weights, one .nve file per module.

    Each file is named after the module path (dots replaced with underscores).
    """
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    for name, module in model.named_modules():
        if not isinstance(module, NVEmbeddingBase):
            continue
        if module.cache_type != CacheType.LinearUVM:
            continue
        filename = name.replace(".", "_") + ".nve"
        filepath = os.path.join(weights_dir, filename)
        with open(filepath, "wb") as f:
            _nve_mod.TensorFileFormat().write_table_file_header(f)
            _nve_mod.write_tensor_to_stream(module.emb_layer, f, module.id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_nve(model: torch.nn.Module, save_dir: str):
    """Save NVE layer metadata and weights to ``save_dir``.

    Produces metadata.json and embeddings.nve. No torch.export or AOT
    compilation — use this when you only need to persist the NVE state.
    """
    os.makedirs(save_dir, exist_ok=True)
    _export_metadata(model, save_dir)
    _export_weights(model, save_dir)


def export(model: torch.nn.Module, example_inputs, save_dir: str, *,
           dynamic_shapes=None):
    """Export *model* to ``save_dir`` using torch.export.

    Saves metadata.json, embeddings.nve, and model.pt2 (torch.export format).
    Requires torch custom op bindings (NVE_DISABLE_TORCH_BINDINGS=OFF (default)).
    """
    _require_torch_ops("export")
    save_nve(model, save_dir)

    exported = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes)
    torch.export.save(exported, os.path.join(save_dir, "model.pt2"))


def export_aot(model: torch.nn.Module, example_inputs, save_dir: str, *,
               dynamic_shapes=None, inductor_configs=None):
    """Export *model* to ``save_dir`` with AOTInductor compilation.

    Saves metadata.json, embeddings.nve, and model.pt2 (AOT-compiled format
    loadable by C++ AOTIModelPackageLoader).
    Requires torch custom op bindings (NVE_DISABLE_TORCH_BINDINGS=OFF (default)).
    """
    _require_torch_ops("export_aot")
    save_nve(model, save_dir)

    exported = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes)
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=os.path.join(save_dir, "model.pt2"),
        inductor_configs=inductor_configs or {},
    )


def load_nve_layers(save_dir: str, *, gpu_cache_size_override: int = None):
    """Recreate NVEmbedding/NVEmbeddingBag layers from an export directory.

    Reads metadata.json, creates layers via NVEmbedding/NVEmbeddingBag __init__
    (which handles memblock allocation and register_for_torch), and loads
    weights from weights/<module_name>.nve.

    Returns a list of layers (caller keeps alive).
    Requires torch custom op bindings (NVE_DISABLE_TORCH_BINDINGS=OFF (default)).
    """
    _require_torch_ops("load_nve_layers")
    with open(os.path.join(save_dir, "metadata.json")) as f:
        metadata = json.load(f)

    layers = []
    weights_dir = os.path.join(save_dir, "weights")

    for entry in metadata:
        layer_id = entry["id"]
        module_path = entry["module_path"]
        num_emb = entry["num_embeddings"]
        emb_size = entry["embedding_size"]
        dtype_str = entry["dtype"]
        cache_type_name = entry["cache_type"]
        is_bag = entry.get("is_bag", False)
        mode = entry.get("mode", None)
        gpu_cache_size = gpu_cache_size_override or entry["gpu_cache_size"]
        optimize_for_training = entry["optimize_for_training"]

        # Resolve dtype
        if "float32" in dtype_str.lower():
            data_type = torch.float32
        elif "float16" in dtype_str.lower():
            data_type = torch.float16
        else:
            raise ValueError(f"Unsupported dtype in metadata: {dtype_str}")

        # Resolve cache type
        cache_type = CacheType[cache_type_name]
        if cache_type != CacheType.LinearUVM:
            raise NotImplementedError(
                f"load_nve_layers: cache_type '{cache_type_name}' not yet supported")

        # Recreate full NVEmbedding / NVEmbeddingBag via __init__
        # (handles memblock, register_for_torch, weight allocation)
        if is_bag:
            layer = NVEmbeddingBag(
                num_emb, emb_size, data_type, cache_type, mode,
                gpu_cache_size=gpu_cache_size,
                optimize_for_training=optimize_for_training,
                id=layer_id)
        else:
            layer = NVEmbedding(
                num_emb, emb_size, data_type, cache_type,
                gpu_cache_size=gpu_cache_size,
                optimize_for_training=optimize_for_training,
                id=layer_id)

        # Load weights
        weight_file = os.path.join(
            weights_dir, module_path.replace(".", "_") + ".nve")
        with open(weight_file, "rb") as f:
            layer.load_from_stream(f)

        layers.append(layer)

    return layers


def load(save_dir: str, *, gpu_cache_size_override: int = None):
    """Load a model previously saved with :func:`export`.

    Recreates NVE layers (with weights loaded and registered in
    NVELayerRegistry), then loads the exported program from model.pt2.

    Returns:
        (ExportedProgram, list[NVEmbeddingBase]): The loaded model and the
        NVE layers. The caller must keep the layers alive for the lifetime
        of the model — if the layers are garbage-collected, the custom ops
        will fail to find them in the registry.
    """
    layers = load_nve_layers(save_dir,
                             gpu_cache_size_override=gpu_cache_size_override)

    loaded = torch.export.load(os.path.join(save_dir, "model.pt2"))

    return loaded, layers
