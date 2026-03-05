# NVE with Multiple GPUs

## Table of Contents

- [Introduction](#introduction)
- [Embedding Configurations](#embedding-configurations)
  - [GPUEmbeddingLayer / CacheType.NoCache](#gpuembeddinglayer-c--cachetypenocache-py)
  - [LinearUVMEmbeddingLayer / CacheType.LinearUVM](#linearuvmembeddinglayer-c--cachetypelinearuvm-py)
  - [HierarchicalEmbeddingLayer / CacheType.Hierarchical](#hierarchicalembeddinglayer-c--cachetypehierarchical-py)
- [Code Examples](#code-examples)
- [Data Races and Read-After-Write Safety](#data-races-and-read-after-write-safety)
- [Glossary](#glossary)

## Introduction

This document describes how to share resources between multiple instances of NVE embedding layers that reference the same underlying embedding data across multiple GPUs.

The relevant use cases for these techniques are:
* Deploying multiple instances of the same model in the same node (typically one per GPU)
* Sharding a large embedding across multiple GPUs/nodes using NVLink

Different layer types and table configurations have different sharing options available. It's up to the user to choose the appropriate configuration.

**Notes:**
- When running multiple parallel inferences on the same GPU, you can use a single embedding layer with different execution contexts (C++ only).
- Embedding layers do not share GPU caches; sharing is limited to the larger backing resources.
- Embedding layers are unaware of any resource sharing. This means you always work with embedding layers in DDP mode and the application needs to synchronize between layers when writing (see [Data Races and Read-After-Write Safety](#data-races-and-read-after-write-safety)).

## Embedding Configurations

Every layer type handles sharing differently.

### GPUEmbeddingLayer (C++) / CacheType.NoCache (Py)

> **Source:** [include/gpu_embedding_layer.hpp](../include/gpu_embedding_layer.hpp) / [python/pynve/torch/nve_layers.py](../python/pynve/torch/nve_layers.py)

This layer is meant for small embedding tables that fit in GPU memory and need no caching. However, the layer will function as long as the table is GPU accessible - even if it's in host memory (e.g. when using `cudaMallocManaged`/`cudaMallocHost`).

The C++ class gets a pointer to the embedding table buffer, so you can use multiple instances of `GPUEmbeddingLayer` with the same pointer to share this buffer. In Python, every `NVEmbedding` with `CacheType.NoCache` has its own tensor for the table (weights) and there's no sharing.

> **Tip:** The better approach to share this buffer in C++ is to use the same `GPUEmbeddingLayer` object with different execution contexts (using multiple threads/streams).

### LinearUVMEmbeddingLayer (C++) / CacheType.LinearUVM (Py)

> **Source:** [include/linear_embedding_layer.hpp](../include/linear_embedding_layer.hpp) / [python/pynve/torch/nve_layers.py](../python/pynve/torch/nve_layers.py)

This layer uses a GPU cache backed by a [linear](#linear-table) embedding table.

**Host Memory Sharing:**

If the linear table is located in host memory, the same table can be used by multiple layers running on different GPUs:
- In C++, you can provide each layer's `gpu_table` with the same pointer for the `embedding_table`.
- In Python, you can use the same memblock for the `NVEmbedding` objects (it's recommended to use a `ManagedMemBlock` object for this case).

This sharing is naturally limited to the same node/host.

**NVLink Sharding:**

The linear table can also be sharded across multiple GPUs connected with NVLink (potentially on different nodes). This is achieved by allocating multiple GPU buffers (shards) on different GPUs, each holding a piece of the table, then mapping them to a single continuous address range in virtual memory ([UVM](#uvm)). For details, consult the [CUDA Programming Guide on Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#).

To allocate this sharded buffer:
- In C++, use `CUDADistributedBuffer`
- In Python, use `MPIMemBlock` or `DistMemBlock`

This requires multiple processes to be launched with `mpirun`/`torchrun`/etc. and [IMEX](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/imexchannels.html) channels setup for NVLink.

> **Note:** The linear layer is unaware of any of these considerations. The lookup kernel only sees a pointer in virtual memory and the CUDA runtime is responsible for directing accesses to the correct location. This means every layer handles its own lookups independently—there are no cooperative operations (e.g., [NCCL](https://developer.nvidia.com/nccl)).

For a working example of multi-GPU sharding, see `benchmarks/multi_gpu_bench.py`. More details in [benchmarks.md](benchmarks.md#multi-gpu-benchmark)

### HierarchicalEmbeddingLayer (C++) / CacheType.Hierarchical (Py)

> **Source:** [include/hierarchical_embedding_layer.hpp](../include/hierarchical_embedding_layer.hpp) / [python/pynve/torch/nve_layers.py](../python/pynve/torch/nve_layers.py)

This layer uses a combination of GPU cache, host cache, and remote parameter server (not all must be present).

In this layer, there's an inherent sharing of the parameter server since every layer only operates a small frontend locally. You can also share the host cache by using the same host table object in C++ (this isn't supported yet in Python).

## Code Examples

### C++: Sharing a Linear Table Across GPUs

```cpp
#include "linear_embedding_layer.hpp"

constexpr int64_t row_size = int64_t(1)<<10; // 1KB
constexpr int64_t cache_size = int64_t(1)<<22; // 4MB
constexpr int64_t linear_table_size = int64_t(1<<26); // 64MB
using key_type = int64_t;
using table_type = nve::GpuTable<key_type>;
using layer_type = nve::LinearUVMEmbeddingLayer<key_type>;

// Allocate a linear table in host memory
void* linear_table = nullptr;
NVE_CHECK_(cudaMallocManaged(&linear_table, linear_table_size));
NVE_CHECK_(cudaMemAdvise(linear_table, linear_table_size, cudaMemAdviseSetAccessedBy, 0));
NVE_CHECK_(cudaMemAdvise(linear_table, linear_table_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

// Create GPU tables for the layer with a 1GB cache in GPU memory
table_type::config_type tab_cfg;
tab_cfg.device_id = 0;
tab_cfg.cache_size = cache_size;
tab_cfg.row_size_in_bytes = row_size;
tab_cfg.uvm_table = linear_table;
auto gpu_tab1 = std::make_shared<table_type>(tab_cfg);
tab_cfg.device_id = 1;
auto gpu_tab2 = std::make_shared<table_type>(tab_cfg);

// Create the linear layer2
auto layer1 = std::make_shared<layer_type>(layer_type::Config(), gpu_tab1);
auto layer2 = std::make_shared<layer_type>(layer_type::Config(), gpu_tab2);

// At this point both layers have their own GPU cache on devices 0 and 1, but both use the host memory for the linear table

...

// Cleanup
layer1.reset();
layer2.reset();
gpu_tab1.reset();
gpu_tab2.reset();
NVE_CHECK_(cudaFree(ptr_));
```

### Python: Sharding a Table with MPIMemBlock

```python
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve
import torch
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()

# The 2 lines below assume the code was run with `mpirun -n N python ...` on K nodes with N/K gpus on each. Amend if different params are used
local_device_id = mpi_rank % torch.cuda.device_count()
local_device = torch.device(f"cuda:{local_device_id}")

embedding_dim = 256
num_table_rows = 2**30
data_type = nve.DataType_t.Float32
memblock = nve.MPIMemBlock(embedding_dim, num_table_rows, data_type)

cache_size = 2**20
emb_layer = nve_layers.NVEmbedding(
    num_table_rows,
    embedding_dim,
    data_type,
    nve_layers.CacheType.LinearUVM,
    gpu_cache_size=cache_size,
    optimize_for_training=False,
    memblock=memblock,
    device=local_device)

# At this point every process forked from mpirun has it's own memblock and emb_layer, but all memblocks are using the same group of GPU buffers.
# Each layer still has it's own GPU cache.

...

# Cleanup
comm.Barrier()
del emb_layer
del memblock
gc.collect()
```

> **Note:** Pay special attention to the destruction order of `MPIMemBlock`/`DistMemBlock` and layers. Every process that participated in the creation of the memblock needs to participate in its destruction, which must happen in the reverse order of creation (typically, layer then memblock and finally env).
>
> **Autograd caveat:** When using NVEmbedding with PyTorch autograd enabled, some tensors (mainly output tensors) may retain a reference to the memblock and must be destroyed before the process group is torn down. It is recommended to run inference under `torch.no_grad()` to avoid this.

## Data Races and Read-After-Write Safety

Embedding layers synchronize reads/writes for their own execution contexts. When resources are shared between different layers, they are unaware of these "external" reads/writes. Synchronization must be handled by the application in these cases.

**Read-only access:** If the embedding tables are only being read, there's no need to synchronize anything.

**Write access:** When a shared resource in GPU/Host memory is being modified by a layer or host, other layers must be blocked from reading/writing until the modification is complete.

**Cache invalidation:** If you modify a shared buffer directly, the layers using it are unaware of the modification. Any GPU cache relying on this buffer needs to be notified by either:
- Calling `update()` with the new value, or
- Calling `erase()` to evict the modified line from the cache (it will re-enter the cache later when accessed)

> **Note:** This usually does not apply to remote parameter servers as they handle synchronization on the server side. However, if you're using a custom parameter server, your mileage may vary.

## Glossary

### Linear Table

An embedding table allocated in virtual memory as a single continuous buffer where the address of embedding *i* is:

```
buffer_start + ( i * embedding_size_in_bytes )
```

### UVM

CUDA Unified Virtual Memory. A virtual address space that can be mapped to both GPU and host physical memory.

See: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) and [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/).

### MPI

Message Passing Interface. A set of APIs used to communicate between processes (potentially on different nodes).

See: [Wikipedia: Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface) and [Open MPI](https://www.open-mpi.org/).
