# Changelog

Releases will be listed below, latest at the top.

Releases are named/tagged in the format of `vYY.MM[.P]` e.g. `v26.02.3` means release of February 2026 with patch 3.

## NV Embedding Cache 26.10
### New Features
- Added the Linear File plugin (`linear_file_plugin` feature, `libnve-plugin-linear-file.so`): a host table over a dense row-major file, with batched reads through io_uring, Linux AIO or pread (see: [plugins.md](docs/plugins.md#bundled-plugins))
- Model loaders accept per-layer configuration overrides (`LayerConfigOverrides` in Python and C++) to tune layer knobs at load time without re-exporting (see: [python_api.md](docs/python_api.md))
- Added the `single_layer_bench` C++ benchmark, which drives a single GPU, LinearUVM or hierarchical layer and reports throughput, hit rates and latency (see: [benchmarks.md](docs/benchmarks.md))
### Improvements
- Added `task_size_threshold` to `LinearHostTable` and `HostEmbeddingLayer` (C++ and C API): caps CPU task fan-out to `ceil(num_keys / threshold)` so small batches don't dispatch across the whole thread pool (default 1024)
- Added a `disable_threadpool` option to `HostEmbeddingLayer` (and the Python HostLayer config) to run lookups on the calling thread
### Bug Fixes
- Fixed the set-associative GPU cache with lazy decay sizing its sets without the per-set timestamp, so its allocation exceeded the configured cache size
- Fixed `nve_loader` silently accepting unsupported dtypes (e.g. `bfloat16` matched as `float16`); the C++ loader now requires exactly `float32`/`float16`, with or without the `torch.` prefix.
- Fixed QInt8Rowwise dequantization producing wrong values on platforms where plain `char` is unsigned (e.g. aarch64)
- Fixed Redis cluster mode storing values and eviction metadata in the same per-partition hash, which corrupted values with LRU and applied LFU counter updates to value entries
  - **Breaks compatibility** with existing Redis cluster data: partition hash keys were renamed, so data written by earlier releases will not be found and must be regenerated to the new schema
### API Changes
- **Breaking C API change:** bumped the C ABI version to **2.0** and added an opaque `header` to GPU table and embedding layer configs
  - Applications must rebuild against the updated headers and replace by-value `_config_default()` calls with `_config_init(&cfg, sizeof(cfg))`, checking the returned status. Config type names and creation function names are unchanged.
  - Future appended config fields within ABI major 2 do not require existing applications to rebuild: newer libraries accept smaller configs and use C++ config defaults for absent fields. Older libraries reject unsupported larger configs instead of silently ignoring options.
- Added `nve_linear_host_table_create()` and `nve_linear_host_table_config_t` to the C API, so host embedding layers can be backed by a `LinearHostTable`
- Bumped the shared plugin ABI version to **2.0** after making `ThreadPool::execute` and `execute_n` virtual
  - Plugins need to be rebuilt and report version 2.0 (no other functional change).
- **Breaking:** replaced `gpu_cache_size_override` in the model loaders with `LayerConfigOverrides`
  - The Python keyword argument was removed from `load`, `load_aot` and `load_nve_layers`, and the C++ `nve::LayerDirectory` constructor now takes a `LayerConfigOverrides` in its place
  - Move a global override to `overrides.defaults` (Python) or `LayerConfigOverrides::defaults` (C++), or to per-layer patches when not every exported layer supports it
- Removed the unused `nve_host_table_config_t` and `nve_overflow_policy_config_t` types and their `_default()` initializers from the C API

## NV Embedding Cache 26.09
### New Features
- Published the experimental SPH plugin (`sph_plugin` feature, `libnve-plugin-sph-experimental.so`): a GPU-resident sharded perfect-hash table that slots in as a GPU tier (see: [plugins.md](docs/plugins.md#bundled-plugins))
- `NVEmbeddingBag` now supports `LayerType.HostLayer`: sum/mean and weighted pooled lookups run on CPU threads, and empty batches return zero-filled bags
### Improvements
- Upgraded nvHashMap to v0.99.5 and aligned STL and NVHM eviction behavior (make sure to update git submodules in existing clones)
- Python layer config exposes `default_embedding` for HostLayer and Hierarchical layers (misses return zeros unless configured, see: [python_api.md](docs/python_api.md))
- Host table accumulate handles different stored and update data types and vector lengths consistently across plugins
- Log messages are formatted only when their level is enabled
- Improved Python embedding input validation for tensor types and layouts, per-sample weights, bag offsets, and unsupported pooling modes
### Bug Fixes
- Hardened GPU and UVM modification paths, including padded-row handling, CPU-accumulate validation, CUDA event lifetime, buffer growth failures, and error handling during CUDA runtime teardown
- Fixed concurrency and failure propagation in the default thread pool, allocator, and asynchronous auto-insert paths
- Fixed a `NaN` hit rate reported for hierarchical tiers left with no candidate keys, which poisoned the statistical insert heuristic
- Fixed table validation and scratch state across STL, NVHM, Redis, and RocksDB, including null keys, internal hit masks, eviction metadata alignment, and the default NVHM kernel size
- Fixed persistence and eviction correctness in RocksDB and Redis, including clear and read-only paths, probe-only lookups, overflow handling, and LFU counters
- Fixed plugin loading by keeping plugin code resident while its objects are in use
### API Changes
- Added `nve_abi_version` and the `NVE_C_ABI_VERSION_MAJOR`/`NVE_C_ABI_VERSION_MINOR` macros to the C API
- C API status codes: CUDA runtime and driver failures now return `NVE_ERROR_CUDA` (or `NVE_ERROR_OUT_OF_MEMORY`) and unimplemented paths return `NVE_ERROR_NOT_IMPLEMENTED` instead of `NVE_ERROR_RUNTIME`; C++ adds `nve::NotImplementedError`

## NV Embedding Cache 26.08
### New Features
- New table-plugin architecture with versioned internal C++ and external pure-C ABI modes (see: [plugins.md](docs/plugins.md))
- Added defined out-of-range key handling for dense GPU and Linear UVM layers: lookups return a configured default row, while update and accumulate ignore invalid keys (see default_row_index in layer configs)
- Python parameter-server (NVEParameterServer) construction from an arbitrary table plugin shared object in Python through `plugin_name`, `factory_config`, and `table_config`
### Improvements
- Reduced build time by pruning and repartitioning generated template instantiations across a fixed set of translation units
- Bounds-checked the Python binding buffer copies and staged writable views through Python-owned storage
### Bug Fixes
- Fixed dense GPU, Linear UVM, and linear host tables reading and writing beyond their allocated rows for out-of-range keys
- Fixed `RocksDBTable::clear()` leaving behind keys whose first byte was at least `0x7E`
- Fixed single-host distributed buffer sharing failing with a `pidfd_getfd` permission error under `torchrun`
- Fixed NumPy tensor loading to validate the declared shape and size against the file instead of trusting the header
### API Changes
- Replaced the host-specific plugin APIs with general table factories
  - C callers now use `nve_create_table_factory`, `nve_table_factory_produce` instead of  `nve_load_host_table_plugin`, `nve_create_host_table_factory`, `nve_host_factory_produce`
  - C++ callers now use `nve::Plugin` or `nve::create_table_from_plugin` instead of `nve::load_host_table_plugin`, `nve::create_host_table_factory`
  - Removed the plugin registry and the `"implementation"` JSON selector, plugins are selected by shared-object path.
  - See the updated code examples in [docs/c_api.md](docs/c_api.md) / [docs/cpp_api.md](docs/cpp_api.md)
- `GPUTableConfig::uvm_num_rows` is required whenever `uvm_table` is set, `LinearHostTableConfig::num_rows` is required, and GPU/Linear UVM layer configs add `default_row_index` for invalid keys
- Hit masks are fixed at 64 bits elements. Removed `HostTableConfig::mask_size`, the C `mask_size` field, the `ht_mask_*` build features, and the `max_bitmask_*` C++ types in favor of `bitmask64_t`
- C status and data-type scalars are fixed-width 32-bit values with unchanged symbolic constants, for plugin ABI stability

## NV Embedding Cache 26.07
### New Features
- Extended pooling and rowwise dequantization to GPU, Linear UVM, and hierarchical embedding layers, with fixed-hotness and CSR layouts; sum, mean, weighted-sum, and weighted-mean reductions; and QInt8/QUint8 rowwise storage
- Added a driverless host-inference build that removes the `libcuda` and `libtorch_cuda` dependencies for C++, Python, and AOTInductor CPU deployments (`NVE_DRIVERLESS_BUILD` / `PYNVE_DRIVERLESS_BUILD=1`)
- Added optional build without OpenMPI (`NVE_DISABLE_MPI` / `PYNVE_DISABLE_MPI=1`); MPI-dependent APIs remain available as unsupported stubs
### Improvements
- Linear UVM pooled lookups now participate in automatic cache insertion
- Set-associative cache frequency counters now decay lazily per touched set, avoiding a full-cache decay pass on every insert
- Added centralized build-flag documentation (see: [build_flags.md](docs/build_flags.md))
- The default thread pool now reserves two hardware threads for non-pool work, with a minimum of one worker
### Bug Fixes
- Fixed a race where asynchronous auto-insert work could retain an execution context after its owner released it
- Fixed Python pooled lookup bag-count handling and isolation of execution contexts between host-layer instances
- Fixed export and reload of user-provided memory blocks, including shared blocks used by different layer types; unsupported distributed memory blocks now fail explicitly instead of being silently converted
- Fixed empty CSR bags and weighted-mean bags with a zero weight sum producing NaNs on CUDA paths; they now return zero-filled rows
- Fixed CUDA pooling output-stride validation that could over-read rows or return corrupt data, while preserving support for padded stored rows
- Fixed raw concatenate routing on GPU and Linear UVM layers and rejected unsupported PyTorch pooling modes instead of treating them as mean pooling
- Invalid pooling parameters and unknown pooling or sparse enum values are now rejected instead of being silently accepted or reported as generic runtime failures
### API Changes
- Changed C++ `EmbeddingLayerBase::PoolingParams` and C `nve_layer_lookup_pooled`: CSR offsets and fixed hotness are now separate arguments, weights have dedicated fields, and callers must provide an explicit output data type
- Removed the unsupported COO sparse layout (`SparseType_t::COO` / `NVE_SPARSE_COO`); fixed-hotness and CSR layouts remain supported
- Invalid lookup and pooling arguments now throw `nve::InvalidArgumentError` in C++ and return `NVE_ERROR_INVALID_ARGUMENT` through the C API

## NV Embedding Cache 26.06
### New Features
- Host CPU inference layer for C, C++, and Python (`nve_host_embedding_layer_create`, `HostEmbeddingLayer`, `LayerType.HostLayer`)
- Redis standalone single-node string mode for the Redis backend
- Shared-storage export/load metadata schema for models with shared memblocks or parameter servers
- CPU pooling and dequantization support, including rowwise int8/uint8 quantized data types
### Improvements
- Per-instance marker tensors for torch custom-op/AOTInductor model loading
- Layer update and accumulate operations can target a specific table with `table_id`
- Added layer table-count query API (`nve_layer_get_num_tables`)
- Python layer config exposes `max_modify_size`
- Benchmark CSV output includes run metadata columns
- Thread-safety improvements for binding execution-context maps and GPU-table UVM access
- C++ Table API refactored: GPU table and host table interfaces updated; custom plugin/table implementations will need to be updated accordingly
### Bug Fixes
- Fixed serialization overflow issue found by static analysis
- Fixed AOT export/load marker handling for multiple model instances and shared resources
- Fixed performance regression in Python inference (redundant buffer copy)
- Removed stale torch-save serialization path
### API Changes
- Embedding layers update/accumulate how have an additional table_id param for fine grain control (using -1 will keep current behavior)
- Tables no longer expose raw buffer pointer APIs, use the BufferWrapper APIs instead.

## NV Embedding Cache 26.05
### New Features
- New PyTorch bindings for C++ deployments (see: [advanced.md](docs/advanced.md#c-deployment-with-pytorch-aotinductor))
### Improvements
- Benchmarks will translate pareto distribution to power-law coefficient
- Code style alignment
- Many small improvements from static code scan
- Plugins can now have arbitrary filenames (no longer require libnve-plugin- prefix)
- Default embedding values for hierarchical layer
### Bug Fixes
- Missing lock added in modify_context_destroy()
- Fixed illegal memory access in embedding cache with negative keys
- Fixed auto-inserting undefined embeddings for keys that were not resolved by any table in hierarchical layer

## NV Embedding Cache 26.04
### New Features
- C API (see: [c_api.md](docs/c_api.md))
### Improvements
- Updated base NVIDIA containers to 26.01
### Bug Fixes
- Default insert heuristic used by C++ embedding layers - was already the case for Python

## NV Embedding Cache 26.03
### New Features
- New samples:
    - Load/Save checkpoint
    - Triton Inference Server
- Multi-GPU docs
- Benchmark scripts (single/multi gpu)
### Improvements
- Kernel improvements for additional tile sizes
- Python bindings for more features (Redis parameter server, shared host buffer, erase() api)
### Bug Fixes
- Fixed excessive overflow handling in NVHM tables

## NV Embedding Cache 26.02.1
### New Features
- Added NVHM submodule

## NV Embedding Cache 26.02
### New Features
- Initial Release
- Note: NVHM is not included in this release, so some tests/samples that rely on it will fail and complain about a missing libnve-plugin-nvhm.so.

***

## Format for new entries:
```
## NV Embedding Cache YY.MM(.PP)
### New Features
- ...
### Improvements
- ...
### Bug Fixes
- ...
### API Changes
- ...
```
