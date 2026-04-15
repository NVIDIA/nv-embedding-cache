# Changelog

Releases will be listed below, latest at the top.

Releases are named/tagged in the format of `vYY.MM[.P]` e.g. `v26.02.3` means release of February 2026 with patch 3.

## EXPERIMENTAL Feature Branch
This branch introduces a new integration with Torch.
See the ["C++ Deployment with PyTorch"](docs/advanced.md#c-deployment-with-pytorch-aotinductor) section in [docs/advanced.md](docs/advanced.md).
* Only some layer types are currently supported.
* Implementation / API is not yet stable.

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
