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

Three modes are supported via ``--mode``:

* ``linearuvm`` (default) — a single ``NVEmbedding(LinearUVM)`` layer with
  weights stored on host memory and a GPU cache. Output layout::

      output_dir/
      ├── model.pt2
      ├── metadata.json
      └── weights/<module_path>.nve

* ``hierarchical`` — a single ``NVEmbedding(Hierarchical)`` layer whose
  remote PS is backed by the ``custom_remote`` host-table plugin
  (``samples/common/custom_remote_plugin``). Output layout::

      output_dir/
      ├── model.pt2
      ├── metadata.json     # includes remote_ps_config + remote_ps_data
      ├── keys.npy
      └── values.npy

* ``gpu`` — a single ``NVEmbedding(NoCache)`` layer whose weights live
  entirely in GPU memory (no host cache, no remote PS). Output layout
  matches the linearuvm mode (model.pt2 + metadata.json + weights/).

All modes are loadable by the same ``nve_inference`` C++ binary.
"""

import argparse
import os
import torch
import numpy as np
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from pynve.torch.nve_export import export_aot

DEVICE = torch.device("cuda")
NUM_EMB = 1024
EMB_SIZE = 8
GPU_CACHE = 4 * 1024 * 1024  # 4 MiB


class _LinearUVMModel(torch.nn.Module):
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


class _HierarchicalModel(torch.nn.Module):
    def __init__(self, remote_ps):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings=NUM_EMB,
            embedding_size=EMB_SIZE,
            data_type=torch.float32,
            cache_type=nve_layers.CacheType.Hierarchical,
            gpu_cache_size=GPU_CACHE,
            remote_interface=remote_ps,
            optimize_for_training=False,
            device=DEVICE,
        )

    def forward(self, keys):
        return self.emb(keys)


class _GPUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nve_layers.NVEmbedding(
            num_embeddings=NUM_EMB,
            embedding_size=EMB_SIZE,
            data_type=torch.float32,
            cache_type=nve_layers.CacheType.NoCache,
            optimize_for_training=False,
            device=DEVICE,
        )

    def forward(self, keys):
        return self.emb(keys)


def _build_linearuvm_model(output_dir):
    """Build, populate, and return a LinearUVM model + extra export kwargs."""
    model = _LinearUVMModel()
    weight_data = (
        torch.arange(NUM_EMB, dtype=torch.float32, device=DEVICE)
        .unsqueeze(1).expand(NUM_EMB, EMB_SIZE)
    )
    model.emb.weight.data.copy_(weight_data)
    print(f"Weight[0]: {model.emb.weight[0]}")
    print(f"Weight[5]: {model.emb.weight[5]}")
    return model, {}


def _build_gpu_model(output_dir):
    """Build, populate, and return a NoCache (GPU-only) model + extra export kwargs."""
    model = _GPUModel()
    weight_data = (
        torch.arange(NUM_EMB, dtype=torch.float32, device=DEVICE)
        .unsqueeze(1).expand(NUM_EMB, EMB_SIZE)
    )
    model.emb.weight.data.copy_(weight_data)
    print(f"Weight[0]: {model.emb.weight[0]}")
    print(f"Weight[5]: {model.emb.weight[5]}")
    return model, {}


def _build_hierarchical_model(output_dir):
    """Build, populate, and return a Hierarchical(custom_remote) model
    plus extra export kwargs (ps_data_paths so load can refill the PS)."""
    ps = nve_ps.NVEParameterServer(
        num_embeddings=NUM_EMB,
        embedding_size=EMB_SIZE,
        data_type=torch.float32,
        plugin_name="libnve-plugin-custom_remote.so",
        factory_config={"implementation": "custom_remote"},
        table_config={"key_size": 8, "max_value_size": EMB_SIZE * 4},
    )
    keys = torch.arange(NUM_EMB, dtype=torch.int64)
    values = torch.zeros(NUM_EMB, EMB_SIZE, dtype=torch.float32)
    for i in range(NUM_EMB):
        values[i] = float(i)
    ps.insert(keys, values)

    keys_path = os.path.join(output_dir, "keys.npy")
    values_path = os.path.join(output_dir, "values.npy")
    np.save(keys_path, keys.numpy())
    np.save(values_path, values.numpy())

    model = _HierarchicalModel(ps)
    return model, {"ps_data_paths": {"emb": (keys_path, values_path)}}


_MODE_BUILDERS = {
    "linearuvm":    _build_linearuvm_model,
    "hierarchical": _build_hierarchical_model,
    "gpu":          _build_gpu_model,
}

_MODE_DEFAULT_SUBDIR = {
    "linearuvm":    "output",
    "hierarchical": "output_hierarchical",
    "gpu":          "output_gpu",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(_MODE_BUILDERS.keys()),
                        default="linearuvm",
                        help="Which kind of NVEmbedding model to export (default: linearuvm)")
    parser.add_argument("--output", default=None,
                        help="Output directory. Default: ./output (linearuvm), "
                             "./output_hierarchical (hierarchical), or ./output_gpu (gpu), "
                             "relative to this script.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    if args.output is None:
        output_dir = os.path.join(script_dir, _MODE_DEFAULT_SUBDIR[args.mode])
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Mode:       {args.mode}")
    print(f"Output dir: {output_dir}")

    builder = _MODE_BUILDERS[args.mode]
    model, extra_export_kwargs = builder(output_dir)

    keys = torch.tensor([0, 1, 5, 10], device=DEVICE, dtype=torch.int64)
    with torch.no_grad():
        out = model(keys)
    print(f"Forward output for keys [0,1,5,10]:\n{out}")

    sc = torch.export.ShapesCollection()
    sc[keys] = {0: torch.export.Dim.AUTO(min=1, max=1000)}
    ds = sc.dynamic_shapes(model, (keys,))

    print("Exporting with AOTInductor...")
    export_aot(model, (keys,), output_dir,
               dynamic_shapes=ds,
               **extra_export_kwargs)
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
