# PyTorch Bindings for NVE Embedding Layers

This directory contains PyTorch custom class bindings for NVE embedding layers.

## Files

- `torch_embedding_base.hpp/cpp` - Base class template for all embedding layers
- `torch_linear_uvm_embedding.hpp/cpp` - LinearUVM embedding layer binding
- `torch_bindings.cpp` - PyTorch library registration
- `CMakeLists.txt` - Build configuration

## Building

```bash
cd python/pynve/torch_bindings
mkdir build && cd build
cmake ..
make
```

## Usage Example

```python
import torch

# Load the library
torch.classes.load_library("build/libnve_torch.so")

# Create memory block for embeddings using factory method
mem_block = torch.classes.nve.MemBlock.create_linear(
    128,        # row_size (embedding_dim)
    1000000,    # num_embeddings
    0           # dtype enum (0=Float32, 1=Float16)
)

# Create configuration
config = torch.classes.nve.EmbedLayerConfig()
config.logging_interval = 100
config.kernel_mode = 0  # Default gather kernel

# Create embedding layer
embedding = torch.classes.nve.LinearUVMEmbedding(
    embedding_dim=128,
    num_embeddings=1000000,
    dtype=torch.float32,
    mem_block=mem_block,
    gpu_cache_size=1 << 30,  # 1GB GPU cache
    use_private_stream=True,
    device_id=0,
    config=config
)

# Forward pass - lookup embeddings
indices = torch.randint(0, 1000000, (1024,), dtype=torch.int64, device='cuda:0')
embeddings = embedding.lookup(indices)  # Shape: [1024, 128]

# Forward pass with pooling (e.g., for bag of embeddings)
keys = torch.randint(0, 1000000, (5000,), dtype=torch.int64, device='cuda:0')
offsets = torch.tensor([0, 100, 300, 600, 1000, 5000], dtype=torch.int64, device='cuda:0')
pooled = embedding.lookup_with_pooling(
    keys,
    offsets,
    None,  # weights (optional)
    0      # pooling_type: 0=Sum, 1=Mean, 2=Concat
)  # Shape: [5, 128] for batch_size=5

# Backward pass - compute gradients
grads = torch.randn_like(pooled)
unique_keys, unique_grads, num_unique = embedding.pooling_backprop(
    keys, grads, offsets, None, 0
)

# Update embeddings with gradients
embedding.accumulate(unique_keys, -0.01 * unique_grads)  # SGD step

# Get underlying weight tensor (for inspection or checkpointing)
weights = embedding.get_weight_tensor()  # Shape: [1000000, 128]

# Save/load state (PyTorch style)
state = embedding.state_dict()
torch.save(state, "embedding.pt")

# Later, load state
loaded_state = torch.load("embedding.pt")
embedding.load_state_dict(loaded_state)
```

## Memory Block Types

Different memory allocation strategies are supported via factory methods:

### LinearMemBlock
Standard UVM allocation (most common):
```python
mem_block = torch.classes.nve.MemBlock.create_linear(
    row_size, num_embeddings, dtype_enum
)
# Or specify total size:
mem_block = torch.classes.nve.MemBlock.create_linear_from_size(total_bytes)
```

### ManagedMemBlock
CUDA managed memory with multi-GPU support:
```python
mem_block = torch.classes.nve.MemBlock.create_managed(
    row_size, num_embeddings, dtype_enum, [0, 1]  # GPU IDs
)
```

### NVLMemBlock
NVLink-based multi-GPU memory:
```python
mem_block = torch.classes.nve.MemBlock.create_nvl(
    row_size, num_embeddings, dtype_enum, [0, 1]  # GPU IDs
)
```

### MPIMemBlock
Distributed memory across MPI ranks:
```python
mem_block = torch.classes.nve.MemBlock.create_mpi(
    row_size, num_embeddings, dtype_enum,
    [0, 1],  # ranks
    [0, 1]   # devices
)
```

### UserMemBlock
Wrap existing GPU-accessible pointer:
```python
ptr = some_existing_cuda_memory_ptr
mem_block = torch.classes.nve.MemBlock.create_user(int(ptr))
```

## Kernel Modes

Configure different gather kernel implementations via `config.kernel_mode`:

- `0`: Default gather kernel
- `1`: Sort-based gather (set `config.kernel_mode_value_1` for tile size)
- `2`: Pipeline gather (set `config.kernel_mode_value_1` for task size, `config.kernel_mode_value_2` for num streams)

```python
# Example: Use sort-based gather with tile size 1024
config.kernel_mode = 1
config.kernel_mode_value_1 = 1024
```

## Integration with PyTorch nn.Module

```python
class LinearUVMEmbeddingModule(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        mem_block = torch.classes.nve.MemBlock.create_linear(
            embedding_dim, num_embeddings, 0  # Float32
        )
        config = torch.classes.nve.EmbedLayerConfig()
        self.core = torch.classes.nve.LinearUVMEmbedding(
            embedding_dim, num_embeddings, torch.float32,
            mem_block, 1 << 30, True, 0, config
        )

    def forward(self, indices):
        return self.core.lookup(indices)
```

## Notes

- Only `int64_t` key type is currently instantiated
- Embeddings must be `float32` or `float16`
- All tensors must be on CUDA device
- The binding automatically extracts CUDA stream from tensors
