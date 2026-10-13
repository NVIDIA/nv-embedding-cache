# NV Embedding Cache C API

The C API provides a pure C interface to the NV Embedding Cache SDK, enabling integration from C programs, FFI bindings (e.g. Go, Rust), or any language that can call C functions. The C API is exposed by the same shared library as the C++ API (`libnve-common.so`), so no additional linking is required.

The header is [`include/nve_c_api.h`](../include/nve_c_api.h).

## Compilation

Link with `libnve-common.so` and `libcudart.so`, the same as for the C++ API.
Pass table plugins to `nve_create_table_factory()` as shared object
names that the dynamic linker can resolve (e.g. `libnve-plugin-nvhm.so`) or as
explicit paths (e.g. `/tmp/my_plugin.so`). See [plugins.md](plugins.md) for the
full plugin system documentation.

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
| `nve_table_factory_t` | Table factory (plugin) |
| `nve_thread_pool_t` | Thread pool |
| `nve_allocator_t` | Memory allocator |

Each handle has a corresponding `_destroy` function.

## Configuration

Configs begin with an opaque `header` field. Initialize caller-owned
storage with `_config_init(&cfg, sizeof(cfg))`, check the status, assign fields,
and pass the config to its existing creation function:

```c
nve_status_t create_gpu_table(nve_table_t* out) {
    /* Create config */
    nve_gpu_table_config_t cfg;
    /* Init config to defaults */
    nve_status_t st = nve_gpu_table_config_init(&cfg, sizeof(cfg));
    if (st != NVE_SUCCESS) {
        return st;
    }
    /* Override config defaults */
    cfg.device_id = 0;
    cfg.cache_size = 1 << 20;
    cfg.row_size_in_bytes = 128;
    cfg.value_dtype = NVE_DTYPE_FLOAT32;
    /* Create the gpu table */
    return nve_gpu_table_create(out, NVE_KEY_INT64, &cfg, NULL);
}
```

| Config type | Initializer | Creation function |
| --- | --- | --- |
| `nve_gpu_table_config_t` | `nve_gpu_table_config_init` | `nve_gpu_table_create` |
| `nve_linear_host_table_config_t` | `nve_linear_host_table_config_init` | `nve_linear_host_table_create` |
| `nve_gpu_embedding_layer_config_t` | `nve_gpu_embedding_layer_config_init` | `nve_gpu_embedding_layer_create` |
| `nve_linear_uvm_layer_config_t` | `nve_linear_uvm_layer_config_init` | `nve_linear_uvm_layer_create` |
| `nve_hierarchical_layer_config_t` | `nve_hierarchical_layer_config_init` | `nve_hierarchical_layer_create` |
| `nve_host_embedding_layer_config_t` | `nve_host_embedding_layer_config_init` | `nve_host_embedding_layer_create` |

The initializer fills fields with the corresponding C++ config defaults and
writes the `header`, which records the size. The caller should not modify the
header. Pointers set by the initializer refer to library-owned defaults.

Initialization:

- Writes at most `size` bytes and leaves storage unchanged on failure.
- Returns `NVE_ERROR_INVALID_ARGUMENT` if `size` is below `sizeof(uint64_t)` or
  ends inside a field.
- Returns `NVE_ERROR_NOT_IMPLEMENTED` if `size` exceeds what the library
  supports, rather than silently dropping newer options.
- Cannot check the real allocation: `size` bytes must be readable and writable.

Creation:

- Returns `NVE_ERROR_INVALID_ARGUMENT` for configs not set up by the matching
  initializer (for example uninitialized storage or another config type).
- Reads only complete fields within the recorded size and uses defaults for the
  rest, so applications built with an older, smaller config within ABI major 2
  run on a newer library without recompiling. Required resources must still be
  supplied.
- Clears its output handle on failure.
- Leaves pointer ownership and handle lifetimes unchanged.

### ABI evolution rules

Within the same ABI major version, future config fields must be appended beyond
the previous struct's full `sizeof`, including trailing padding. Existing
fields, offsets, types, alignment, and meanings stay fixed. Register new fields
and map their C++ defaults in the corresponding `nve_config_traits`
specialization in `src/nve_c_api_config.hpp`, then map them to the C++ config
in the creation implementation. 

For brevity, the remaining examples omit status handling; application code must
check every returned status before continuing.

### Data Types

`nve_data_type_t` covers floating-point storage and rowwise quantized storage:

| Value | Description |
| --- | --- |
| `NVE_DTYPE_FLOAT32` | 32-bit floating point |
| `NVE_DTYPE_BFLOAT16` | bfloat16 |
| `NVE_DTYPE_FLOAT16` | IEEE fp16 |
| `NVE_DTYPE_E4M3` | fp8 E4M3 |
| `NVE_DTYPE_E5M2` | fp8 E5M2 |
| `NVE_DTYPE_FLOAT64` | 64-bit floating point |
| `NVE_DTYPE_QINT8_ROWWISE_F32` | int8 values with one fp32 scale per row |
| `NVE_DTYPE_QINT8_ROWWISE_F16` | int8 values with one fp16 scale per row |
| `NVE_DTYPE_QUINT8_ROWWISE_F32` | uint8 values with fp32 scale and offset per row |
| `NVE_DTYPE_QUINT8_ROWWISE_F16` | uint8 values with fp16 scale and offset per row |

For rowwise quantized formats, `row_size_in_bytes` / `max_value_size` is the full stored row size: value bytes plus trailing scale metadata, and offset metadata for the affine `QUINT8` variants.

## Tables

### GPU Table

```c
nve_gpu_table_config_t cfg;
nve_gpu_table_config_init(&cfg, sizeof(cfg));
cfg.device_id = 0;
cfg.cache_size = 4 * 1024 * 1024;  // 4 MB
cfg.row_size_in_bytes = 128;
cfg.value_dtype = NVE_DTYPE_FLOAT32;

nve_table_t gpu_table = NULL;
nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &cfg, NULL);
```

### Linear Host Table

A host table over a caller-owned dense row buffer: key `k` maps to row `k`, and
keys outside `[0, num_rows)` miss. Use it with a [host embedding layer](#host-embedding-layer).

```c
float* rows = NULL;  // num_rows * 32 floats, preferably cudaMallocHost'd
cudaMallocHost((void**)&rows, num_rows * 128);

nve_linear_host_table_config_t cfg;
nve_linear_host_table_config_init(&cfg, sizeof(cfg));
cfg.emb_table = rows;
cfg.num_rows = num_rows;
cfg.row_size_in_bytes = 128;
cfg.value_dtype = NVE_DTYPE_FLOAT32;

nve_table_t host_table = NULL;
nve_linear_host_table_create(&host_table, NVE_KEY_INT64, &cfg);
```

`max_threads` caps the CPU threads used per operation and `task_size_threshold`
limits them to `ceil(num_keys / task_size_threshold)` (C++ defaults: unlimited
and 1024).

### Host Tables (via plugins)

Host tables are created through a factory pattern:

1. Create a factory from a plugin shared object with a plugin-specific JSON config
2. Produce tables from the factory with a table-specific JSON config

`nve_create_table_factory()` loads the plugin and creates its table factory in
one call. The factory and every table it produces keep the plugin loaded for
their own lifetimes. The JSON config contains only plugin-specific options —
pass `"{}"` when the plugin needs none. See [plugins.md](plugins.md) for the
full plugin documentation, including how to write your own plugin.

```c
// Create factory from a plugin shared object
nve_table_factory_t factory = NULL;
nve_create_table_factory(&factory, "libnve-plugin-nvhm.so", "{}");

// Produce a table
nve_table_t host_table = NULL;
nve_table_factory_produce(factory, 0, "{"
    "\"key_size\": 8,"
    "\"max_value_size\": 128,"
    "\"value_dtype\": \"float32\","
    "\"num_partitions\": 4,"
    "\"initial_capacity\": 4096,"
    "\"value_alignment\": 32"
"}", &host_table);
```

`nve_table_factory_produce()` derives the handle's key type from the produced
table's key size (4 → `NVE_KEY_INT32`, 8 → `NVE_KEY_INT64`); other key sizes
are rejected. Table configs still carry `key_size` where the table needs it
for its own configuration.

Available in-tree plugins:
- `libnve-plugin-nvhm.so` — NVIDIA nvHashMap
- `libnve-plugin-abseil.so` — Google Abseil
- `libnve-plugin-phmap.so` — Parallel Hashmap
- `libnve-plugin-stl-map.so` — STL `std::unordered_map` (reference plugin)
- `libnve-plugin-redis.so` — Redis backend (requires `"address"` in factory config). Connects to a
  Redis **Cluster** by default, or to a **standalone single-node** server when
  `"single_node": true` — see [Redis backend configuration](#redis-backend-configuration) below.
- `libnve-plugin-rocksdb.so` — RocksDB

#### Redis backend configuration

The `redis_cluster` plugin supports two deployment modes, selected by the `single_node` factory
option and the table's `num_partitions`:

- **Cluster mode** (default, `single_node: false`): connects to a Redis Cluster and shards keys
  across `num_partitions` (a power of two) Redis hashes. Use a high partition count relative to the
  number of cluster nodes for parallelism.
- **Standalone string mode** (`single_node: true`, empty `hash_key`): connects to a single Redis
  server and stores each entry as a Redis **string** (`MSET`/`MGET`/`DEL`). Here `num_partitions` is
  purely a client-side parallelism knob — `0`/`1` run single-threaded, while `>1` (a power of two)
  splits the command-building/parsing work across that many threads (storage stays plain strings).
  Set `connections_per_node ≥ num_partitions` for full parallelism. Setting a `hash_key` instead
  stores everything in a single named Redis hash.

```c
// Cluster mode: shard across Redis hashes.
nve_table_factory_t cluster_factory = NULL;
nve_create_table_factory(&cluster_factory, "libnve-plugin-redis.so", "{"
    "\"address\": \"localhost:7000\""
"}");
nve_table_t cluster_table = NULL;
nve_table_factory_produce(cluster_factory, 0, "{"
    "\"key_size\": 8,"
    "\"max_value_size\": 128,"
    "\"value_dtype\": \"float32\","
    "\"num_partitions\": 16"
"}", &cluster_table);

// Standalone string mode: single-node Redis with MSET/MGET.
nve_table_factory_t standalone_factory = NULL;
nve_create_table_factory(&standalone_factory, "libnve-plugin-redis.so", "{"
    "\"address\": \"localhost:6379\","
    "\"single_node\": true"
"}");
nve_table_t standalone_table = NULL;
nve_table_factory_produce(standalone_factory, 0, "{"
    "\"key_size\": 8,"
    "\"max_value_size\": 128,"
    "\"value_dtype\": \"float32\","
    "\"num_partitions\": 0,"
    "\"string_namespace_id\": 1"
"}", &standalone_table);
```

Factory config options (`nve_create_table_factory`):

| Option | Default | Description |
| --- | --- | --- |
| `address` | `localhost:6379` | Address of a Redis node (`host:port`). Required. |
| `single_node` | `false` | If `true`, connect to a standalone server instead of a cluster (string mode, or a single hash if `hash_key` is set). |
| `user_name` | `default` | Redis username. |
| `password` | `""` | Plaintext password. |
| `keep_alive` | `true` | Keep TCP connections alive. |
| `connections_per_node` | `5` | Max parallel connections per Redis node. In standalone string mode set this `≥ num_partitions` so the parallel work isn't bottlenecked on the pool. |
| `use_tls` | `false` | Encrypt connections with TLS (with `ca_certificate`, `client_certificate`, `client_key`, `server_name_identification`). |

Table config options (`nve_table_factory_produce`), in addition to the common fields above:

| Option | Default | Description |
| --- | --- | --- |
| `num_partitions` | `1` | In **cluster** mode: number of Redis hashes (power of two) to shard keys across. In **standalone string** mode: the client-side parallelism degree — `0`/`1` = single-threaded, `>1` (power of two) splits the `MSET`/`MGET`/`DEL` work across that many threads while still storing plain Redis strings (no sharding). |
| `hash_key` | `""` | When set with `num_partitions: 0`, store all entries in this single named Redis hash instead of strings. |
| `string_namespace_id` | `-1` | String mode only. `-1` = raw key bytes; `≥0` = prefix every key with these bytes, namespacing the table so several tables can share one Redis DB and `clear()` only removes this table's keys (via `SCAN`+`DEL`). With `-1`, `clear()` issues `FLUSHDB` and wipes the entire server. |
| `overflow_policy` | `evict_random` | String mode supports `evict_random` only (no LRU/LFU eviction — bound capacity via the Redis server's `maxmemory` policy). |

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

The C API exposes constructors for GPU, Linear UVM, Hierarchical, and Host embedding layers.

### GPU Embedding Layer

Stores all data in GPU memory. No caching.

```c
nve_gpu_embedding_layer_config_t cfg;
nve_gpu_embedding_layer_config_init(&cfg, sizeof(cfg));
cfg.layer_name = "my_gpu_layer";
cfg.device_id = 0;
cfg.embedding_table = d_table_ptr;  // GPU memory pointer
cfg.num_embeddings = 1024;
cfg.embedding_width_in_bytes = 128;
cfg.value_dtype = NVE_DTYPE_FLOAT32;
cfg.default_row_index = 0;  // optional: row returned for keys outside [0, num_embeddings)

nve_layer_t layer = NULL;
nve_gpu_embedding_layer_create(&layer, NVE_KEY_INT64, &cfg, NULL);
```

### Linear UVM Embedding Layer

GPU cache backed by a UVM (Unified Virtual Memory) table in host memory.

```c
// GPU table must have uvm_table set
nve_gpu_table_config_t gpu_cfg;
nve_gpu_table_config_init(&gpu_cfg, sizeof(gpu_cfg));
gpu_cfg.uvm_table = h_pinned_table;  // cudaMallocHost'd memory
// ... other config ...

nve_table_t gpu_table = NULL;
nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &gpu_cfg, NULL);

nve_linear_uvm_layer_config_t cfg;
nve_linear_uvm_layer_config_init(&cfg, sizeof(cfg));
cfg.layer_name = "my_uvm_layer";
cfg.default_row_index = 0;  // optional: row returned for keys outside [0, uvm_num_rows)

nve_layer_t layer = NULL;
nve_linear_uvm_layer_create(&layer, NVE_KEY_INT64, &cfg, gpu_table, NULL);
```

Both layers fill the caller-owned row named by `default_row_index` into the output of any key outside the table, and update/accumulate ignore such keys. Leaving `default_row_index` negative (the default) skips the range check, and looking up an out of range key is then undefined.

### Hierarchical Embedding Layer

Multi-tier caching with GPU cache, host table, and optional remote storage.

```c
nve_hierarchical_layer_config_t cfg;
nve_hierarchical_layer_config_init(&cfg, sizeof(cfg));
cfg.layer_name = "my_hier_layer";
// cfg.insert_heuristic is NULL by default, which enables the default heuristic
// To disable auto-inserts: nve_heuristic_create_never(&cfg.insert_heuristic)

nve_table_t tables[] = {gpu_table, host_table};
nve_layer_t layer = NULL;
nve_hierarchical_layer_create(&layer, NVE_KEY_INT64, &cfg, tables, 2, NULL);
```

### Host Embedding Layer

CPU-only layer wrapping one host table, either a [linear host table](#linear-host-table) or a plugin table. The table must use the same key type as the layer and must report a host/CPU device.

```c
// Create any host table first, for example through a plugin factory.
nve_table_factory_t factory = NULL;
nve_create_table_factory(&factory, "libnve-plugin-nvhm.so", "{}");

nve_table_t host_table = NULL;
nve_table_factory_produce(factory, 0, "{"
    "\"key_size\": 8,"
    "\"max_value_size\": 128,"
    "\"value_dtype\": \"float32\","
    "\"num_partitions\": 4,"
    "\"initial_capacity\": 4096,"
    "\"value_alignment\": 32"
"}", &host_table);

float default_row[32] = {0};  // optional miss fallback
nve_host_embedding_layer_config_t cfg;
nve_host_embedding_layer_config_init(&cfg, sizeof(cfg));
cfg.layer_name = "my_host_layer";
cfg.default_embedding = default_row;
cfg.default_embedding_size = sizeof(default_row);

nve_layer_t layer = NULL;
nve_host_embedding_layer_create(&layer, NVE_KEY_INT64, &cfg, host_table, NULL);
```

Host layers support lookup, insert, update, accumulate, erase, clear, and pooled lookup through the generic `nve_layer_*` APIs. Pooled host lookup supports `NVE_SPARSE_FIXED` and `NVE_SPARSE_CSR` layouts.

The host config exposes `task_size_threshold` and `disable_threadpool`.
`task_size_threshold` must be positive and limits CPU task count to
`ceil(num_keys / task_size_threshold)` (C++ default: 1024).
Setting `disable_threadpool` to a nonzero value runs lookups on the caller
thread instead of the supplied thread pool (C++ default: false).

### Layer operations

```c
nve_context_t ctx = NULL;
nve_layer_create_execution_context(layer, &ctx, NULL, NULL, NULL, NULL);

// Query the number of tables in the layer
int64_t num_tables = 0;
nve_layer_get_num_tables(layer, &num_tables);

// Lookup
float hitrates[2];
nve_layer_lookup(layer, ctx, num_keys, keys, output, row_size, NULL, hitrates);

// Lookup with pooling: CSR offsets {0, 2, 5} describe two bags over five keys
int64_t offsets[] = {0, 2, 5};
nve_layer_lookup_pooled(
    layer, ctx, 5, keys, pooled_output, row_size, NULL,
    NVE_POOL_MEAN, NVE_SPARSE_CSR, offsets, 3, 0,
    NULL, NVE_DTYPE_UNKNOWN, NVE_DTYPE_FLOAT32, hitrates);

// Fixed hotness 2 uses the scalar argument and does not require a CSR offsets buffer.
nve_layer_lookup_pooled(
    layer, ctx, num_keys, keys, pooled_output, row_size, NULL,
    NVE_POOL_SUM, NVE_SPARSE_FIXED, NULL, 0, 2,
    NULL, NVE_DTYPE_UNKNOWN, NVE_DTYPE_FLOAT32, hitrates);

// Concatenate emits one row per key. Sparse layout and weight arguments are ignored.
nve_layer_lookup_pooled(
    layer, ctx, num_keys, keys, output, row_size, NULL,
    NVE_POOL_CONCATENATE, NVE_SPARSE_FIXED, NULL, 0, 0,
    NULL, NVE_DTYPE_UNKNOWN, NVE_DTYPE_FLOAT32, hitrates);

// Insert into a specific table (table_id = 0 for first table)
nve_layer_insert(layer, ctx, num_keys, keys, stride, size, values, 0);

// Update across all tables
nve_layer_update(layer, ctx, num_keys, keys, stride, size, values, -1);

// Accumulate gradients
nve_layer_accumulate(layer, ctx, num_keys, keys, stride, size, grads, NVE_DTYPE_FLOAT32, -1);

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
