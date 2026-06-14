# Advanced Topics

## Execution contexts (C++ only)
The collection of state and resources needed during execution of lookup/modify is called an [execution context](include/execution_context.hpp). These objects are created by a layer or table (depending on the selected integration level) and are used by the calling software stack to separate resources of parallel executions. I.e. each execution context represents a single reuseable parallel execution environment. For example, using 3 contexts, you can run 3 operations in parallel (each using a different context). Using the same context in multiple parallel ops can result in undefined behavior.

The application is expected to create a fixed amount of execution contexts during initialization, then use them in some repeating order (e.g. round-robin).
Creating additional contexts during runtime is allowed, but is less performant. Bear in mind that each context will rquire additional memory allocations.
Execution contexts use lazy memory allocation, so expect an operation with larger than seen before dimensions (e.g. number of indices to lookup) to potentially result in memory allocation.

During teardown, the application must destroy all execution contexts of a layer/table before destroying the layer/table.

## Cache management

### Modify Operations
All operations that change cache residency or cache storage such as insert/update are collectively call Modify operations.
Every Modify operation requires a ModifyContext which provides a parallel execution unit. The cache doesn't support multiple Modify operations inflight on the GPU side - it is the user's responsibility to ensure that, e.g using a single stream to launch all Modify operations.

### Invalidate and Commit
To be able to run Lookup and Modify operations in parallel, we employ a paradigm called invalidate and commit.
The modify operation will first launch a kernel to invalidate the relevant cache entries, then wait until all inflight lookups
have concluded, before altering the cache and re-enabling the affected cache entries.

#### Custom Flows
Invalidate and Commit relies on Lookup operation being queued on the GPU in an atomic fashion e.g a single CUDA kernel. Some users may implement their own complex gather flows. In order to maintain the required atomicity, if the flow uses more than one CUDA kernel, the user needs to call the start/end_custom_flow APIs.

## Multi device
Embeddings can span multiple devices/nodes by way of sharding. This is accomplished with a CUDA buffer tha spans multiple devices as detailed in the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#).
The user can either create this allocation on their own or use the CUDADistributedBuffer class (see: [../include/distributed.hpp](../include/distributed.hpp)). Similarily, the python MPIMemBlock class can be used to share such a buffer across MPI process group.

## Insert heuristic
The embedding layers that use a cache, can choose when it's time to update the cache residency and insert new cache lines by calling an InsertHeuristic object that determines when to initiate this update.

By default, `LinearUVMEmbeddingLayer` and `HierarchicalEmbeddingLayer` use the `DefaultInsertHeuristic` when no heuristic is provided (i.e. `insert_heuristic = nullptr` in the layer config). 

To disable auto-insertion entirely, use the `NeverInsertHeuristic` class. The application can also derive its own heuristic from the `InsertHeuristic` base class.

See [../include/insert_heuristic.hpp](../include/insert_heuristic.hpp) for more details.

## C++ Deployment with PyTorch (AOTInductor)

NVE models that use `NVEmbedding` / `NVEmbeddingBag` can be deployed for C++ inference without Python at runtime. The flow uses PyTorch's AOTInductor to compile the model graph, while NVE handles the embedding layer setup and weight loading natively in C++.

### Prerequisites

Build NVE with torch bindings enabled (default):
```bash
pip install .
```

### Step 1: Export from Python

Use `export_aot()` to save the model graph, layer metadata, and weights:

```python
from pynve.torch.nve_export import export_aot

model = MyModel()  # contains NVEmbedding layers
export_aot(model, (example_keys,), "save_dir/")
```

This produces:
- `save_dir/model.pt2` — AOTInductor-compiled graph (loadable by C++ `AOTIModelPackageLoader`)
- `save_dir/metadata.json` — schema-v2 layer configuration plus shared storage resources (`resources`, `storage_ref`, `layer_type`, `config`, etc.)
- `save_dir/weights/<resource_key>.nve` — embedding weights in NVE binary format, one file per memblock-backed storage resource

### Step 2: Load and run in C++

Load the AOT model, then use `nve::LayerDirectory` (RAII) to create embedding layers, load weights, wire marker constants, and register the layers in the `NVELayerRegistry`:

```cpp
#include <memory>
#include <string>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "python/pynve/torch_bindings/nve_loader.hpp"

const std::string save_dir = "save_dir";
const int device_index = 0;

// The loader holds non-owning handles to marker tensors owned by LayerDirectory.
auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
    save_dir + "/model.pt2");

auto resources = std::make_shared<nve::ResourceDirectory>();
nve::LayerDirectory dir(save_dir, *loader, device_index, resources);

// Run inference
auto keys = torch::tensor({0L, 1L, 5L, 10L},
    torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index));
c10::InferenceMode mode;
auto outputs = loader->run({keys});

// Destroy the AOTI container before LayerDirectory releases its marker tensors.
loader.reset();
```

No Python or pybind11 is required at C++ runtime. The C++ program links against `libnve-torch-ops.so` (custom op registration) and `libnve-common.so` (NVE core).

### Architecture

The custom op `nve_ops::embedding_lookup` is registered in the C10 dispatcher via `STABLE_TORCH_LIBRARY` (LibTorch Stable ABI) in `libnve-torch-ops.so`. When the AOT model calls this op at runtime, it uses the per-layer marker tensor pointer to find the layer in the `NVELayerRegistry` singleton and calls the appropriate NVE layer lookup.

```
AOT Model → C10 Dispatcher → nve_ops::embedding_lookup → NVELayerRegistry → LinearUVMEmbedding::lookup()
```

### Example

A complete working sample is in [../samples/cpp_inference/](../samples/cpp_inference/). See the [README](../samples/cpp_inference/README.md) for build and run instructions.

## Unit tests
Unit tests are located in [../tests/](../tests/) and are using 2 frameworks: [GoogleTest](https://github.com/google/googletest) and [PyTest](https://docs.pytest.org/en/stable/).

To run the GoogleTests use:
```bash
./tests/embedding_layer/redis_cluster.sh start
sleep 5
for test in build_dir/bin/*test*; do  ./$test; done
./tests/embedding_layer/redis_cluster.sh stop
```
* Note: the redis_cluster.sh script starts the local Redis cluster and a standalone single-node server (for the Redis string-mode tests) needed by some of the tests

To run the PyTest tests use:
```bash
pytest tests
```
* Note: make sure to install the python bindings and required packages before testing, using:
    ```bash
    pip install .
    pip install -r benchmarks/requirements.txt
    ```

## Pooling support status

Pooling support is under active development. At this time, only a limited set of pooling configurations is supported and expected to work reliably.
