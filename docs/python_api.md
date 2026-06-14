# Python API

The NV Embedding Cache SDK provides PyTorch-like wrappers for easy integration with Python code.
The wrappers NVEmbedding and NVEmbeddingBag aim to mimic the PyTorch modules [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and [torch.nn.EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) respectively.

## Installation 

```console
     pip install .
```

## Quick Start
First, import the nve_layers and nve modules

```python
    import torch
    import pynve.nve as nve
    import pynve.torch.nve_layers as nve_layers
```

Create an Embedding with 10 million vectors stored the on CPU and 1 MB of cache on GPU 
```python
    weight_init = torch.randn(10*1024*1024, 1, dtype=torch.float32)
    embedding = nve_layers.NVEmbedding(num_embeddings=10*1024*1024, 
                                        embedding_size=1, 
                                        data_type=torch.float32, 
                                        layer_type=nve_layers.LayerType.LinearUVM, gpu_cache_size=1024*1024,
                                        weight_init=weight_init)
```

Perform a lookup
```python
    keys = torch.Tensor([1,2,1000]).to(torch.int64).to(device='cuda')
    emb = embedding(keys)
```
See [../samples/pytorch/simple_sample/](../samples/pytorch/simple_sample/) for full code.

## Overview
The nve_layers module creates a [torch.nn.module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) like interface that wraps around the python bindings to the NV Embedding Cache SDK C++ API.
The two operations [NVEmbedding](../python/pynve/torch/nve_layers.py) and [NVEmbeddingBag](../python/pynve/torch/nve_layers.py) are controlled by the `LayerType` enum:

| LayerType | Storage | GPU cache | Use case |
|-----------|---------|-----------|----------|
| `GPULayer` | GPU memory | — | All data fits on GPU |
| `LinearUVM` | UVM / host memory | yes | Large tables, GPU cache required |
| `Hierarchical` | Remote parameter server (+ optional host cache) | yes | Distributed / very large tables; user supplies a PS |
| `HostLayer` | CPU host memory | — | CPU-only inference; no GPU required |

nve_ops module provides functional like interface and autograd functionality for training. It isn't generally advised to use this interface directly

nve_tensors module provides wrappers to Torch tensors whose storage is backed by a cache hierarchy.

## Layer Configuration

`NVEmbedding` and `NVEmbeddingBag` accept an optional `config` dictionary for lower-level layer and GPU-table tuning:

```python
embedding = nve_layers.NVEmbedding(
    num_embeddings,
    embedding_size,
    torch.float32,
    nve_layers.LayerType.LinearUVM,
    gpu_cache_size=gpu_cache_size,
    config={
        "logging_interval": 1000,
        "kernel_mode": 0,
        "kernel_mode_value_1": 0,
        "kernel_mode_value_2": 0,
        "max_modify_size": 1 << 20,
    },
)
```

`max_modify_size` limits the number of insert/update/accumulate entries handled in one modify operation. Leaving it unset uses the table default.

## Using a Parameter Server
To use the parameter server API, users are expected to implement the Table Interface defined in [table.hpp](../include/table.hpp) and provide a Python binding to it. See the [custom remote plugin](../samples/common/custom_remote_plugin/src/custom_remote_table.cpp) and [custom remote PS export sample](../samples/pytorch/export_sample/custom_remote_ps_export.py) for examples.

Once a python binding to an implementation of nve::Table is available, creating an "Embedding Layer" with parameter server is simple.
```python
    embedding = nve_layers.NVEmbedding(num_embeddings=10*1024*1024, 
                                        embedding_size=1, 
                                        data_type=torch.float32, 
                                        layer_type=nve_layers.LayerType.Hierarchical, gpu_cache_size=1024*1024,
                                        storage=binding_to_remote_ps)
```
## Host Layer (CPU Inference)

`LayerType.HostLayer` stores all embedding vectors in CPU host memory with no GPU cache. It is
intended for inference pipelines that cannot or do not want to use a GPU.

```python
embedding = nve_layers.NVEmbedding(
    num_embeddings=10*1024*1024,
    embedding_size=128,
    data_type=torch.float32,
    layer_type=nve_layers.LayerType.HostLayer,
    optimize_for_training=False,   # required — HostLayer is inference-only
    device=torch.device("cpu"),    # optional; defaults to the current CUDA device
)
```

When `device='cpu'` is passed, keys and outputs are CPU tensors. When a CUDA device is used
(the default), keys must be on that device and the output is returned there; the lookup itself
runs on the host and the result is copied back.

To use C++-managed host memory instead of auto-allocating a pinned PyTorch tensor, pass a
`HostMemBlock` as the `storage` argument:

```python
import pynve.nve as nve

memblock = nve.HostMemBlock(
    row_size=128,           # embedding_size
    num_embeddings=10*1024*1024,
    dtype=nve.DataType_t.Float32,
)
embedding = nve_layers.NVEmbedding(
    num_embeddings=10*1024*1024,
    embedding_size=128,
    data_type=torch.float32,
    layer_type=nve_layers.LayerType.HostLayer,
    optimize_for_training=False,
    storage=memblock,
)
```

**Constraints:**
- `optimize_for_training=False` is required; HostLayer does not support gradient computation.
- In Python, only `NVEmbedding` supports `HostLayer`; `NVEmbeddingBag` rejects `LayerType.HostLayer`.
- If providing a custom `storage` memblock, it must be host-accessible: `HostMemBlock`, `ManagedMemBlock`, a `UserMemBlock` wrapping a pinned tensor, or a host-side `LinearMemBlock` (device_id=-1). `NVL` and `MPI` memblocks are rejected.

## Multi Device
NV Embedding Cache takes PyTorch's approach to multi-device usage. Users should specify the device where the layer resides. Users should ensure all input tensors are on the same device as the layer (or accessible from the layer's device) when calling forward.
There is no need to set the device context as the underlying implementation handles this automatically.

## Inference Sample

A sample that shows usage of inferencing with NVEmbedding
usage:
``` inference_sample.py --batch-size 1024 --hotness 32 --num-iterations 1000 --cache-size 1024 --data-type float32```

this will run an inference for a 1000 iterations each of 1024 bags of 32 keys using linearUVM with a cache of size 1024 bytes.
See [inference_sample.py](../samples/pytorch/inference_sample/inference_sample.py)

## Loading Checkpoint
Loading a dcp checkpoint is straightforward. When calling dcp.load, the user is required to supply a map from a saved tensor name to NVEmbedding.weight.
Special consideration is required when loading into a multi-process environment with "Distributed Memblock"; since all processes share the same NVEmbedding.weight underlying physical memory, only one rank should load the tensor (usually rank 0).
See [load_sample.py](../samples/pytorch/load_checkpoint_sample/load_sample.py)

## Note on PyTorch Versions

PyTorch integration uses the LibTorch Stable ABI, which is only available since PyTorch 2.10.
`pip install .` uses build isolation and typically installs the latest stable PyTorch from PyPI to build against.
If the locally installed PyTorch version is older than 2.10, you may encounter an import error when importing `pynve.nve`, as it links against `libnve-torch-ops.so` by default.

To work around this, install without torch bindings:
```bash
PYNVE_DISABLE_TORCH_BINDINGS=1 pip install .
```

This uses the legacy Python API path. Embedding lookups and training work normally, but exporting via `torch.export` or AOTInductor will not be available.

## Torch Binding and AOT Export

The torch binding exposes NVEmbedding / NVEmbeddingBag as registered PyTorch custom ops
(`torch.ops.nve_ops.embedding_lookup` and `embedding_lookup_with_pooling`). This lets a model
containing NVE layers be captured with [`torch.export`](https://pytorch.org/docs/stable/export.html)
and compiled with [AOTInductor](https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
for use in a C++ (or Python) inference pipeline.

The custom ops are *stateless* — the embedding storage and cache live in the C++ binding layer,
which the ops locate through an out-of-band registry. Each NVE layer carries a small per-layer
**marker tensor** (a non-persistent buffer threaded into every op call), and the registry is keyed
by that tensor's `data_ptr()`. Because pointers are unique across the host and all CUDA devices,
multiple exported models — or several instances of the same model — can coexist in one process
without colliding.

### Export

```python
from pynve.torch.nve_export import export, export_aot

# torch.export only (Python replay / debugging)
export(model, (example_keys,), "save_dir/")

# AOTInductor-compiled package (for C++ or Python AOT inference)
export_aot(model, (example_keys,), "save_dir/")
```

Both write `model.pt2`, `metadata.json`, and one weight file per memblock-backed storage resource under `save_dir/weights/`.

### Load (Python)

```python
from pynve.torch.nve_export import load, load_aot

# Non-AOT (torch.export) — returns a ready-to-call module
module, layers = load("save_dir/")
out = module(keys)

# AOTInductor package
loader, layers = load_aot("save_dir/")
out = loader.run([keys])[0]
```

> **Keep `layers` alive for the duration of all inference**

### Inference in C++

Construct the `AOTIModelPackageLoader`, then hand it to `nve::LayerDirectory`, which recreates the
layers and wires their markers into the package. See
[samples/cpp_inference/](../samples/cpp_inference/) for the full program.

```cpp
auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
    save_dir + "/model.pt2");
auto resources = std::make_shared<nve::ResourceDirectory>();
nve::LayerDirectory dir(save_dir, *loader, device_index, resources);
auto outputs = loader->run({keys});
loader.reset();  // before dir releases its marker tensors
```

### Shared storage & resources

Several layers can share one backing store (an `NVLMemBlock`/`MemBlock` or a remote PS) —
each layer is just a cache interface onto it. To preserve this across export/load, `metadata.json`
(schema v2) normalizes storage into a top-level `resources` section; layers reference it by key
rather than inlining a copy:

```json
{
  "version": 2,
  "resources": {
    "remote_ps": { "ps-<id>": { "...export_config()..." } },
    "memblocks": { "mb-<id>": { "type": "NVL", "row_elements": 128, "num_rows": 1000000, "dtype": "float32" } }
  },
  "layers": [
    { "id": 0, "module_path": "emb_a", "layer_type": "Hierarchical", "storage_ref": "ps-<id>", "config": { ... }, ... },
    { "id": 1, "module_path": "emb_b", "layer_type": "Hierarchical", "storage_ref": "ps-<id>", "config": { ... }, ... }
  ]
}
```

Resource keys are the `id()` of the shared Python object, so two layers (or two models exported
from the same process) that share a store get the same key. No device indices are serialized —
topology is resolved at load. To share across models, pass one `registry` dict:

```python
registry = {}
layers_a = load_nve_layers("model_a/", registry=registry)
layers_b = load_nve_layers("model_b/", registry=registry)   # reuses model_a's storage on key match
# resource_remap={"mb-A": "mb-B"} aliases a file key to an existing registry entry
```

In C++, share by passing one `nve::ResourceDirectory` to multiple `LayerDirectory` instances.

### Load-time device topology (C++)

The `LayerDirectory` constructor takes an optional `nve::TopologyMap`
(`std::map<resource_id, std::vector<int>>`) mapping a resource key to its devices. For an NVL block
this is the GPU span; for any other block it's a single-element device list. Resources absent from
the map default to: **NVL** → all devices, **Host** → CPU, **everything else** → the load
`device_index`.

```cpp
nve::TopologyMap topo{{"mb-<id>", {0, 1, 2, 3}}};   // pin this NVL block to GPUs 0–3
nve::LayerDirectory dir(save_dir, *loader, device_index, resources, /*gpu_cache_override=*/0, topo);
```

### Notes

1. **Runtime constant folding is required for the AOTI path.** `export_aot` sets
   `aot_inductor.use_runtime_constant_folding=True` for you. If you compile a package by hand with
   `torch._inductor.aoti_compile_and_package`, you must set it as well — otherwise the marker
   constants are folded inline and `get_constant_fqns()` returns empty. In that case the markers
   cannot be wired in: `load_aot` raises `RuntimeError` ("AOTI package exposes no updatable
   constants"), and the `nve::LayerDirectory` constructor fails the same check (`NVE_CHECK_`).
2. **Re-apply markers after a constant-buffer swap.** If NVE layers are embedded in a larger model
   that hot-swaps weights via `swap_constant_buffer` (preceded by an inactive-buffer rebuild), the
   marker constant is overwritten; call `rebind_markers(loader, layers)` (Python) or
   `dir.rebind_markers(loader)` (C++) afterward. Standalone NVE deployment never needs this.
