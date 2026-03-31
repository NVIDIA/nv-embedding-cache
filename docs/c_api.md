# NV Embedding Cache C API

The C API provides a pure C interface to the NV Embedding Cache SDK, enabling integration from C programs, FFI bindings (e.g. Go, Rust), or any language that can call C functions. The C API is exposed by the same shared library as the C++ API (`libnve-common.so`), so no additional linking is required.

The header is [`include/nve_c_api.h`](../include/nve_c_api.h).

## Compilation

Link with `libnve-common.so` and `libcudart.so`, the same as for the C++ API.
Any host table plugin used (e.g. `libnve-plugin-nvhm.so`) must be present in the `LD_LIBRARY_PATH` at runtime.

```c
#include <nve_c_api.h>
```

## Error handling

Every C API function returns an `nve_status_t`. On failure, call `nve_get_last_error()` to retrieve a thread-local error string describing the problem.

```c
nve_status_t st = nve_gpu_table_create(&table, NVE_KEY_INT64, &cfg, NULL);
if (st != NVE_SUCCESS) {
    const char* msg = NULL;
    nve_get_last_error(&msg);
    fprintf(stderr, "Error %d: %s\n", st, msg);
}
```

Status codes:

| Code | Meaning |
|------|---------|
| `NVE_SUCCESS` | Operation completed successfully |
| `NVE_ERROR_INVALID_ARGUMENT` | Invalid parameter passed |
| `NVE_ERROR_CUDA` | CUDA error occurred |
| `NVE_ERROR_RUNTIME` | General runtime error |
| `NVE_ERROR_NOT_IMPLEMENTED` | Feature not implemented |
| `NVE_ERROR_OUT_OF_MEMORY` | Memory allocation failed |

## Opaque handles

All objects are represented as opaque handles:

| Handle | Description |
|--------|-------------|
| `nve_table_t` | GPU or host table |
| `nve_layer_t` | Embedding layer |
| `nve_context_t` | Execution context |
| `nve_heuristic_t` | Insert heuristic |
| `nve_host_factory_t` | Host table factory |
| `nve_thread_pool_t` | Thread pool |
| `nve_allocator_t` | Memory allocator |

Each handle has a corresponding `_destroy` function.

## Configuration

Config structs are initialized using `_default()` functions that provide sensible defaults matching the C++ API:

```c
nve_gpu_table_config_t cfg = nve_gpu_table_config_default();
cfg.device_id = 0;
cfg.cache_size = 1 << 20;
cfg.row_size_in_bytes = 128;
cfg.value_dtype = NVE_DTYPE_FLOAT32;
```

Available config initializers:
- `nve_gpu_table_config_default()`
- `nve_gpu_embedding_layer_config_default()`
- `nve_linear_uvm_layer_config_default()`
- `nve_hierarchical_layer_config_default()`
- `nve_overflow_policy_config_default()`
- `nve_host_table_config_default()`

## Tables

### GPU Table

```c
nve_gpu_table_config_t cfg = nve_gpu_table_config_default();
cfg.device_id = 0;
cfg.cache_size = 4 * 1024 * 1024;  // 4 MB
cfg.row_size_in_bytes = 128;
cfg.value_dtype = NVE_DTYPE_FLOAT32;

nve_table_t gpu_table = NULL;
nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &cfg, NULL);
```

### Host Tables (via plugins)

Host tables are created through a two-step factory pattern:

1. Load a plugin and create a factory with an implementation-specific JSON config
2. Produce tables from the factory with a table-specific JSON config

```c
// Load plugin and create factory
nve_load_host_table_plugin("nvhm");
nve_host_factory_t factory = NULL;
nve_create_host_table_factory(&factory, "{\"implementation\": \"nvhm_map\"}");

// Produce a table
nve_table_t host_table = NULL;
nve_host_factory_produce(factory, 0, "{"
    "\"mask_size\": 8,"
    "\"key_size\": 8,"
    "\"max_value_size\": 128,"
    "\"value_dtype\": \"float32\","
    "\"num_partitions\": 4,"
    "\"initial_capacity\": 4096,"
    "\"value_alignment\": 32"
"}", &host_table);
```

Available plugin implementations:
- `nvhm_map` — NVIDIA nvHashMap
- `abseil_flat_map` — Google Abseil
- `phmap_flat_map` — Parallel Hashmap
- `redis_cluster` — Redis cluster (requires `"address"` in factory config)
- `rocksdb` — RocksDB

### Table operations

All table operations require an execution context:

```c
nve_context_t ctx = NULL;
nve_table_create_execution_context(table, &ctx, NULL, NULL, NULL, NULL);

// Find/Insert/Update/Erase
nve_table_find(table, ctx, n, keys, hit_mask, stride, values, NULL);
nve_table_insert(table, ctx, n, keys, stride, size, values);
nve_table_update(table, ctx, n, keys, stride, size, values);
nve_table_erase(table, ctx, n, keys);
nve_table_clear(table, ctx);

nve_context_wait(ctx);
nve_context_destroy(ctx);
nve_table_destroy(table);
```

## Embedding Layers

Three layer types are available, mirroring the C++ API:

### GPU Embedding Layer

Stores all data in GPU memory. No caching.

```c
nve_gpu_embedding_layer_config_t cfg = nve_gpu_embedding_layer_config_default();
cfg.layer_name = "my_gpu_layer";
cfg.device_id = 0;
cfg.embedding_table = d_table_ptr;  // GPU memory pointer
cfg.num_embeddings = 1024;
cfg.embedding_width_in_bytes = 128;
cfg.value_dtype = NVE_DTYPE_FLOAT32;

nve_layer_t layer = NULL;
nve_gpu_embedding_layer_create(&layer, NVE_KEY_INT64, &cfg, NULL);
```

### Linear UVM Embedding Layer

GPU cache backed by a UVM (Unified Virtual Memory) table in host memory.

```c
// GPU table must have uvm_table set
nve_gpu_table_config_t gpu_cfg = nve_gpu_table_config_default();
gpu_cfg.uvm_table = h_pinned_table;  // cudaMallocHost'd memory
// ... other config ...

nve_table_t gpu_table = NULL;
nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &gpu_cfg, NULL);

nve_linear_uvm_layer_config_t cfg = nve_linear_uvm_layer_config_default();
cfg.layer_name = "my_uvm_layer";

nve_layer_t layer = NULL;
nve_linear_uvm_layer_create(&layer, NVE_KEY_INT64, &cfg, gpu_table, NULL);
```

### Hierarchical Embedding Layer

Multi-tier caching with GPU cache, host table, and optional remote storage.

```c
nve_hierarchical_layer_config_t cfg = nve_hierarchical_layer_config_default();
cfg.layer_name = "my_hier_layer";
// cfg.insert_heuristic is NULL by default, which enables the default heuristic
// To disable auto-inserts: nve_heuristic_create_never(&cfg.insert_heuristic)

nve_table_t tables[] = {gpu_table, host_table};
nve_layer_t layer = NULL;
nve_hierarchical_layer_create(&layer, NVE_KEY_INT64, &cfg, tables, 2, NULL);
```

### Layer operations

```c
nve_context_t ctx = NULL;
nve_layer_create_execution_context(layer, &ctx, NULL, NULL, NULL, NULL);

// Lookup
float hitrates[2];
nve_layer_lookup(layer, ctx, num_keys, keys, output, row_size, NULL, hitrates);

// Insert into a specific table (table_id = 0 for first table)
nve_layer_insert(layer, ctx, num_keys, keys, stride, size, values, 0);

// Update across all tables
nve_layer_update(layer, ctx, num_keys, keys, stride, size, values);

// Accumulate gradients
nve_layer_accumulate(layer, ctx, num_keys, keys, stride, size, grads, NVE_DTYPE_FLOAT32);

// Erase/Clear
nve_layer_erase(layer, ctx, num_keys, keys, 0);
nve_layer_clear(layer, ctx);

nve_context_wait(ctx);
```

## Insert Heuristics

Insert heuristics control when the layer auto-populates caches during lookup. When the `insert_heuristic` field is `NULL` (the default), a `DefaultInsertHeuristic` is created automatically with a threshold of 0.75 for each table.

```c
// Explicitly create a default heuristic with custom thresholds
float thresholds[] = {0.5f, 0.5f};
nve_heuristic_t h = NULL;
nve_heuristic_create_default(&h, thresholds, 2);

// Disable auto-inserts entirely
nve_heuristic_t never = NULL;
nve_heuristic_create_never(&never);
```

## Execution Contexts

Contexts manage per-operation state (CUDA streams, buffers). Create one per concurrent operation stream. Pass `NULL` for streams to use the default CUDA stream.

```c
nve_context_t ctx = NULL;
nve_layer_create_execution_context(layer, &ctx,
    lookup_stream,   // cudaStream_t, or NULL
    modify_stream,   // cudaStream_t, or NULL
    thread_pool,     // nve_thread_pool_t, or NULL
    allocator);      // nve_allocator_t, or NULL

// After async operations, wait for completion
nve_context_wait(ctx);

// Clean up
nve_context_destroy(ctx);
```

## Thread Pool

```c
nve_thread_pool_t pool = NULL;
nve_thread_pool_create(&pool, "{\"num_threads\": 4}");

int64_t workers;
nve_thread_pool_num_workers(pool, &workers);

nve_thread_pool_destroy(pool);
```

## Samples

* The [c_api](../samples/c_api/) sample demonstrates creating a hierarchical layer with GPU and NVHM host tables, inserting embeddings, and performing lookups via the C API.
