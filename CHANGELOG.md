# Changelog

Releases will be listed below, latest at the top.

Releases are named/tagged in the format of `vYY.MM[.P]` e.g. `v26.02.3` means release of February 2026 with patch 3.

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
```
