# Versioning, Upgrades and Rollbacks

Use this guide to check NVE versions, plan an upgrade, and restore a previous release.

An NVE **release** has a version such as `26.11.0`, in the format `YY.MM.PATCH`. Separate
**ABI (application binary interface) versions** track binary compatibility for the C API and
plugins. Applications, plugins, exported models and stored data each have their own upgrade
requirements.

Before you start:

1. **Replace NVE libraries as a set.** `libnve-common.so`, `libnve-plugin-*.so` and
   `libnve-torch-ops.so` must come from the same build. A `pynve` wheel contains one complete set;
   install it as a unit. Do not copy `.so` files into or out of the installed package.
2. **Give each release its own directory.** Library file names do not contain a version, so two
   releases cannot share a directory.
3. **Load only one release per process.** For a process that imports `pynve`, keep other NVE
   build and install directories off `LD_LIBRARY_PATH`.

## 1. Checking versions

| Version or build detail | How to check |
|---|---|
| Release version (C/C++) | `nve_version(&major, &minor, &patch)` |
| Release version (Python) | `python -c "import pynve; print(pynve.__version__)"`. This is the version of the installed wheel. |
| C ABI version of the library | `nve_abi_version(&major, &minor)` |
| Plugin ABI version of a release | `NVE_PLUGIN_ABI_VERSION_MAJOR`, `NVE_PLUGIN_ABI_VERSION_MINOR` in `include/plugin/nve_plugin_common.h` |
| Plugin ABI version of a plugin | `nve_plugin_abi_version_major()`, `nve_plugin_abi_version_minor()`, exported by the plugin's library |

## 2. What to keep for rollback

Save these assets before upgrading. Keep them until you no longer need to roll back.

| Asset | What to keep |
|---|---|
| NVE libraries (C/C++ deployments) | The deployed release's library directory, unchanged. |
| `pynve` wheel (Python deployments) | The wheel file used for the current installation. Without it, rollback requires rebuilding the old release from source. |
| Build settings | The build command or `CMakeCache.txt`, plus the CUDA and PyTorch versions. Record `NVE_SM_VERSION`, `NVE_FEATURES` and other build flags; the binaries do not record these settings. |
| Your applications and custom plugins | The binaries built for the deployed release. |
| Exported models | The export directories, unchanged, and any parameter-server keys and values files they refer to. Save new exports in separate directories. |
| Data in Redis, RocksDB or linear files | A snapshot taken after stopping the processes and before starting the new release ([Step 5](#step-5-deploy)). |

## 3. What to rebuild or replace

### Libraries, applications and plugins

Always replace the NVE libraries or `pynve` wheel with the target release's build. Then use this
table to decide which application and custom plugin binaries to rebuild:

| Component | When to rebuild |
|---|---|
| C application using only the C API | When the C ABI major version changes. |
| C++ application, including inference applications using `nve::LayerDirectory` | For every NVE release. Also rebuild inference applications whenever the PyTorch version changes. Use the same toolchain and build settings as the NVE build. |
| Custom internal-mode plugin | For every NVE release, using the same toolchain and build settings as the NVE build. |
| Custom external-mode plugin | When the plugin ABI major version changes. |

If the CUDA major version, the GPU architecture list (`NVE_SM_VERSION`) or a build flag changes,
rebuild NVE, the wheel, custom internal-mode plugins and C++ applications, even within one release.

On rollback, restore the previous NVE libraries or wheel and any application or plugin binaries
you rebuilt for the upgrade.

### Exported models

Re-export a model if any of these conditions apply:

- `CHANGELOG.md` lists a breaking change to model export or loading.
- The PyTorch version changes and PyTorch does not guarantee compatibility for that artifact.
- The model is an AOTInductor package and the GPU architecture changes.

On rollback, use the exports made before the upgrade. The older release does not support exports
made with the newer release.

### Stored data

For **Redis, RocksDB and linear files**, check `CHANGELOG.md` for changes to the plugin's storage
layout. Reuse existing data only if the layout and table settings are unchanged. Otherwise,
migrate or regenerate the data.

On rollback, reuse the data only if neither release changed the plugin's storage layout and the
table settings match those from before the upgrade. For RocksDB, the two releases must also bundle
the same RocksDB version (`ROCKSDB_MAJOR` and `ROCKSDB_MINOR` in
`third_party/rocksdb/include/rocksdb/version.h`). Otherwise, restore the snapshot or regenerate
the data. Restoring a snapshot discards writes made after the snapshot.

Other bundled table types only store data in host/GPU memory.
Their tables are empty after a restart.

## 4. Upgrade

### Step 1: Plan the upgrade

1. Read [CHANGELOG.md](../CHANGELOG.md) for every release between the current and target releases,
   including the target. Breaking changes are marked with bold text that starts with
   **Breaking** or **Breaks**.
2. Record the current versions ([section 1](#1-checking-versions)) and build settings listed in
   [section 2](#2-what-to-keep-for-rollback). For each exported model, record the NVE release,
   PyTorch version and, for AOTInductor packages, GPU architecture used during export. Exports do
   not record these details.
3. Use [section 3](#3-what-to-rebuild-or-replace) to list what you need to rebuild, re-export,
   migrate or regenerate.

### Step 2: Save the rollback assets

Save everything listed in [section 2](#2-what-to-keep-for-rollback), except the data snapshot.
Take that after stopping the processes in Step 5.

### Step 3: Build the target release

Check out the target release:

```bash
git fetch --tags
git checkout <target_release_tag>
git submodule update --init --recursive
```

For **C/C++ deployments**, build the libraries:

```bash
cmake -S . -B build_<target_release>
cmake --build build_<target_release> -j"$(nproc)"
```

The libraries to deploy are the `.so` files in `build_<target_release>/lib`.

For **Python deployments**, build a wheel in an environment with the deployed PyTorch version
(2.10 or later) and the requirements listed under `[build-system]` in `pyproject.toml`:

```bash
PYNVE_BUILD_DIR=/tmp/nve_build_<target_release> \
pip wheel --no-build-isolation --no-deps -w dist/ .
```

Check these build settings before running either build:

- **Build options:** match the deployed build settings saved in Step 2 unless you planned a
  change. Pass `-D` options to CMake, or use `PYNVE_CMAKE_ARGS` and the `PYNVE_*` variables for
  wheels. See [build_flags.md](build_flags.md).
- **GPU architectures:** the build uses the architecture of the first GPU on the build host,
  or a default list when the host has no GPU. If you need to target a specific architecture,
  use the `NVE_SM_VERSION` build flag (see: [build_flags.md](build_flags.md)).
- **PyTorch:** `--no-build-isolation` uses the PyTorch installed in the build environment.
  Without it, pip builds against the newest `torch` on the package index.
- **Wheel labels:** set `PYNVE_LOCAL_VERSION=<label>` (for example `cu130`) to distinguish wheels
  built for different CUDA versions.
- **CPU instructions:** the build uses the highest AVX level supported by the build host's CPU.
  Build on a CPU no newer than the deployment CPUs, or set `-DNVE_DISABLE_AVX512=1` (for wheels,
  `PYNVE_DISABLE_AVX512=1`). Otherwise, the libraries can crash with `SIGILL` on older CPUs.

### Step 4: Rebuild applications and plugins

Rebuild the applications and custom plugins identified in Step 1 against the target release's
headers and libraries. Apply any source changes required by `CHANGELOG.md`.

### Step 5: Deploy

1. Stop all processes that use NVE. A running process keeps the libraries it has already loaded.
2. Take a snapshot of the data in Redis, RocksDB or linear files.
3. **C/C++ deployments:** copy the target release's libraries into a new directory for that release
   (for example `/opt/nve/<release>/lib`). Do not overwrite the deployed release's directory. Point
   the applications at the new directory by switching a symbolic link (for example
   `/opt/nve/current -> /opt/nve/<release>`) or by updating their `RUNPATH` or `LD_LIBRARY_PATH`.
4. **Python deployments:** install the new wheel with
   `pip install --force-reinstall --no-deps dist/pynve-<version>-<tags>.whl`.
5. Deploy the rebuilt applications and plugins.
6. Re-export models and migrate or regenerate stored data as planned in Step 1.
7. Start the processes.

## 5. Rollback

1. Stop all processes that use NVE.
2. Restore the previous release:
   - **C/C++ deployments:** point the symbolic link, `RUNPATH` or `LD_LIBRARY_PATH` back to the
     previous release's directory.
   - **Python deployments:** install the saved wheel with
     `pip install --force-reinstall --no-deps <previous pynve wheel file>`.
3. Restore the saved application and plugin binaries for anything rebuilt during the upgrade.
4. Point the processes at the exports made before the upgrade.
5. Reuse, restore or regenerate stored data according to [section 3](#stored-data).
6. Start the processes.
