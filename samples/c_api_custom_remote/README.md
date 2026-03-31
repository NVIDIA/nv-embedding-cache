# Custom Remote Host Table Sample (C API)

This sample demonstrates how to implement a custom "remote" host table as an
NVE plugin and use it with the C API in a three-tier hierarchical embedding
layer:

```
GPU table  (L1, GPU SRAM / HBM cache)
  → phmap host table  (L2, CPU RAM cache)
    → custom_remote   (L3, your parameter server / DB)
```

## What this sample shows

1. **Writing a custom host table plugin** — subclass `nve::HostTableLike`,
   implement the six CRUD operations (`find`, `insert`, `update`,
   `update_accumulate`, `clear`, `erase`), and expose it via the standard
   plugin entry points (`plugin_ident`, `enum_host_table_implementations`,
   `create_<name>_table_factory`).

2. **Loading the plugin at runtime** — the sample executable calls
   `nve_load_host_table_plugin("custom_remote")` to dynamically load
   `libnve-plugin-custom_remote.so`, then uses the factory to produce a table.

3. **Pure C API usage** — all layer, table, and context operations use
   `nve_c_api.h`; no C++ NVE headers are included in the sample executable.

## Directory layout

```
samples/c_api_custom_remote/
├── CMakeLists.txt                  # Builds plugin + sample executable
├── c_api_custom_remote_sample.cpp  # Sample executable (C API only)
├── README.md
└── custom_remote_plugin/           # Plugin shared object
    ├── CMakeLists.txt              # Builds libnve-plugin-custom_remote.so
    ├── include/
    │   └── custom_remote_table.hpp # CustomRemoteTable & CustomRemoteTableFactory
    └── src/
        ├── custom_remote_table.cpp # Table implementation (std::map-backed)
        └── plugin.cpp              # Plugin entry points
```

## Building

The sample is built automatically as part of the default CMake build:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The build produces:
- `libnve-plugin-custom_remote.so` — the plugin shared library
- `c_api_custom_remote_sample` — the sample executable

## Running

```bash
./build/bin/c_api_custom_remote_sample
```

Expected output shows:
- Creation of the three table tiers (GPU, phmap, custom_remote)
- Insertion of 1000 synthetic embeddings into the remote tier
- Lookup of 16 keys traversing all three tiers (100% remote hit rate)
- Verification that retrieved embeddings match expected values

## Adapting to a real remote table

To connect to a real parameter server, replace the `std::map` in
`CustomRemoteTable` with your network client. The key methods to modify are:

- `find()` — issue a batch GET to your KV store
- `insert()` — issue a batch PUT
- `update()` / `update_accumulate()` — issue batch UPDATE / INCREMENT
- `erase()` / `clear()` — issue batch DELETE

The plugin JSON configuration (passed to `nve_host_factory_produce`) can carry
connection strings, timeouts, or other parameters specific to your backend.
