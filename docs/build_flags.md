# Build Flags

NV Embedding Cache is configured with CMake cache variables. Pass project-specific
variables when configuring the build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVE_SM_VERSION="80;90;100" \
  -DNVE_DISABLE_MPI=ON
cmake --build build -j
```

CMake boolean values accept the usual `ON`/`OFF`, `TRUE`/`FALSE`, and `1`/`0`
forms unless noted otherwise below.

## Build contents and deployment

| Variable | Default | Description |
|---|---|---|
| `NVE_DRIVERLESS_BUILD` | `OFF` | Builds without the CUDA driver API or `libcuda`. Driver-dependent functionality is compiled as unsupported stubs. A direct CMake build must set an appropriate `NVE_SM_VERSION` separately. |
| `NVE_DISABLE_TESTS_AND_SAMPLES` | `OFF` | Set to `ON` to exclude all tests and samples from the build. |
| `NVE_ORIGIN_RPATH` | `OFF` | Sets the install RPATH to `$ORIGIN` and the build library directory so relocated shared libraries can find local dependencies. |

## Python package builds

`pip install .` invokes CMake through `setup.py`. It always uses a `Release`
build, enables `NVE_ORIGIN_RPATH`, and supports these environment variables:

| Environment variable | Default | Description |
|---|---|---|
| `PYNVE_BUILD_DIR` | `build` | Selects the CMake and Python build directory. |
| `PYNVE_BUILD_SAMPLES` | Unset | Set to exactly `1` to include tests and samples. They are disabled for all other values. |
| `PYNVE_DISABLE_AVX512` | Unset | Set to exactly `1` to set `NVE_DISABLE_AVX512=1`. |
| `PYNVE_DISABLE_TORCH_BINDINGS` | Unset | Set to exactly `1` to set `NVE_DISABLE_TORCH_BINDINGS=ON`. |
| `PYNVE_DRIVERLESS_BUILD` | Unset | Set to exactly `1` to enable `NVE_DRIVERLESS_BUILD` and set `NVE_SM_VERSION=100`. |
| `PYNVE_DISABLE_MPI` | Unset | Set to exactly `1` to set `NVE_DISABLE_MPI=ON`. |

For example:

```bash
PYNVE_BUILD_DIR=build-py \
PYNVE_DISABLE_MPI=1 \
PYNVE_DISABLE_TORCH_BINDINGS=1 \
pip install .
```

## Feature/Plugin selection

| Variable | Default | Description |
|---|---|---|
| `NVE_FEATURES` | Standard feature set | Space- or semicolon-separated list of features to compile. Feature names are case-insensitive. |
| `NVE_DISABLE_PLUGINS` | `OFF` | Selects the minimal feature set when `NVE_FEATURES` is not provided. |
| `NVE_DISABLE_NVHM_PLUGIN` | `OFF` | Removes `nvhm_plugin` from the selected feature set, including an explicitly supplied `NVE_FEATURES` list. |

The supported `NVE_FEATURES` values are:

| Category | Values |
|---|---|
| Plugins | `abseil_plugin`, `linear_file_plugin`, `nvhm_plugin`, `phmap_plugin`, `redis_plugin`, `rocksdb_plugin` |
| Host tables key sizes | `ht_key_8`, `ht_key_16`, `ht_key_32`, `ht_key_64` |
| Host tables kernel sizes | `ht_kernel_1`, `ht_kernel_2`, `ht_kernel_4`, `ht_kernel_8`, `ht_kernel_16`, `ht_kernel_32`, `ht_kernel_64`, `ht_kernel_128`, `ht_kernel_256`, `ht_kernel_512` |
| Host tables partitioners | `ht_part_fnv1a`, `ht_part_murmur3`, `ht_part_rrxmrrxmsx0`, `ht_part_std_hash` |

The standard default set contains:

```text
abseil_plugin;linear_file_plugin;nvhm_plugin;phmap_plugin;redis_plugin;rocksdb_plugin;
ht_key_32;ht_key_64;ht_kernel_8;ht_kernel_16;ht_part_fnv1a
```

The minimal set selected by `NVE_DISABLE_PLUGINS=ON` contains:

```text
ht_key_64;ht_kernel_8;ht_part_fnv1a
```

For example, a build with Redis and a wider set of hash-table specializations can
be configured with:

```bash
cmake -S . -B build \
  -DNVE_FEATURES="redis_plugin;ht_key_64;ht_kernel_8;ht_kernel_16;ht_kernel_32;ht_part_murmur3"
```

An unknown feature causes CMake configuration to fail.

## Architecture and code generation

| Variable | Default | Description |
|---|---|---|
| `NVE_SM_VERSION` | Auto-detected | CUDA architectures in CMake's numeric format, such as `80`, `100`, or `"80;90;100"`. The value is copied to `CMAKE_CUDA_ARCHITECTURES`. |
| `NVE_CACHE_LINE_SIZE` | Auto-detected | Overrides the CPU cache-line size in bytes. CMake uses `getconf LEVEL1_DCACHE_LINESIZE` on non-Windows systems and `64` on Windows. |
| `NVE_DISABLE_AVX512` | `OFF` | Prevents AVX-512 code generation. AVX2 or AVX is still selected when supported. |
| `NVE_DISABLE_CUDA_AVX_FLAGS` | `OFF` | Does not pass the selected host AVX flags directly to NVCC. Host C and C++ compilation still uses the selected AVX level. |
| `FAC_PER_FILE` | Empty (`100`) | Overrides the number of FindAndCombine explicit template instantiations emitted per generated `.cu` file. An empty value uses the generator default of 100. |

When `NVE_SM_VERSION` is not set, CMake queries the first GPU with `nvidia-smi`.
If no GPU is available it targets `80;86;89;90;100`; if `nvidia-smi` itself is
not available it targets `80;86;90;100`.

Set `NVE_SM_VERSION`, rather than `CMAKE_CUDA_ARCHITECTURES`, to override the
target architectures. The top-level build forcibly updates
`CMAKE_CUDA_ARCHITECTURES` from `NVE_SM_VERSION`.

## Optional libraries and instrumentation

| Variable | Default | Description |
|---|---|---|
| `NVE_WITH_LIBHWLOC` | `ON` when found, otherwise `OFF` | Enables hwloc-based NUMA control. |
| `NVE_WITH_LIBNUMA` | `ON` when found, otherwise `OFF` | Enables libnuma-based NUMA control. |
| `NVE_ENABLE_NVTX` | `OFF` | Enables NVTX instrumentation. |
| `NVE_DISABLE_MPI` | `OFF` | Builds without OpenMPI. MPI-backed APIs are compiled as unsupported stubs. |
| `NVE_DISABLE_TORCH_BINDINGS` | `OFF` | Skips the PyTorch custom-op library, `libnve-torch-ops.so`, and Torch-dependent samples. |

Torch bindings require an importable PyTorch 2.10 or later installation with the
supported stable ABI. If PyTorch is missing or the ABI check fails, CMake
automatically disables the bindings. When enabled, `TORCH_CUDA_ARCH_LIST` can
override the architecture list passed to PyTorch; otherwise it is derived from
`NVE_SM_VERSION`.

## Standard CMake variables

The build also accepts normal CMake configuration variables, including compiler
selection and `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS`, and `CMAKE_CUDA_FLAGS`.
Variables with project-specific behavior include:

| Variable | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Selects the build configuration for a single-configuration generator. |
| `CMAKE_PREFIX_PATH` | Environment-dependent | Adds package search prefixes. The Python build adds PyTorch's CMake prefix when PyTorch is importable. |
| `TORCH_CUDA_ARCH_LIST` | Derived | Overrides the CUDA architectures used by the PyTorch custom-op target. |

`CMAKE_EXPORT_COMPILE_COMMANDS` and position-independent code are always enabled
by the top-level build.
