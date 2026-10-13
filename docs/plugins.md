# NVE Table Plugins

NVE table backends can be packaged as plugin shared objects. A plugin exposes a
single table factory; the host loads the SO, creates the factory, and produces
`nve::Table` instances.

## Bundled plugins

All of these are internal-mode plugins built from this repository. The SO name is
what you pass to the loader; the feature is the `NVE_FEATURES` entry that builds
it (see `docs/build_flags.md`).

| Plugin | SO | Feature | What it is |
|---|---|---|---|
| stl_map | `libnve-plugin-stl-map.so` | *(always built)* | Host table over `std::unordered_map`. The reference/fallback backend and the smallest example of the plugin API. |
| abseil | `libnve-plugin-abseil.so` | `abseil_plugin` | Host table over Abseil's `flat_hash_map`, as an open-addressing alternative to `stl_map`'s node-based map. |
| linear_file | `libnve-plugin-linear-file.so` | `linear_file_plugin` | Host table over a dense row-major file. |
| phmap | `libnve-plugin-phmap.so` | `phmap_plugin` | Host table over the parallel-hashmap flat map — same shape as `abseil`, different implementation and size/speed trade-offs. |
| nvhm | `libnve-plugin-nvhm.so` | `nvhm_plugin` | Host table over NVIDIA's nvHashMap. |
| redis | `libnve-plugin-redis.so` | `redis_plugin` | Remote table backed by a Redis cluster or a single node. Needs a reachable server; capacity is the server's, not the process's. |
| rocksdb | `libnve-plugin-rocksdb.so` | `rocksdb_plugin` | Table backed by RocksDB, so the working set is bounded by disk rather than RAM. |
| sph | `libnve-plugin-sph-experimental.so` | `sph_plugin` | **Experimental.** GPU-resident perfect-hash table; the whole table lives in device memory, so it slots in as a GPU tier rather than a host tier. |

## The two ABI modes

Every plugin declares one of two ABI modes. The mode defines which object types are safe to transfer to the plugin, not where its source lives.

| | Internal mode | External mode |
|---|---|---|
| Toolchain | Same compiler/stdlib as the NVE build | Any toolchain (pure-C ABI) |
| ABI | C++ (`shared_ptr`, `nlohmann::json`, exceptions) | C structs + function pointers, `nve_status_t` returns |
| Creation symbol | `nve_plugin_create_internal_factory` | `nve_plugin_create_external_factory` |
| Plugin header | `plugin/nve_internal_plugin.hpp` | `plugin/nve_external_plugin.h` |
| Buffer residency | Anything a `Table` supports | Host or device pointers (declared once at adoption) |

Pick **internal** when the plugin is built inside this repository (or with an
identical toolchain). Pick **external** when the plugin is built and shipped
independently.

## Common contract (both modes)

Every plugin SO exports five metadata symbols (see
`include/plugin/nve_plugin_common.h`):

```c
uint32_t          nve_plugin_abi_version_major(void);
uint32_t          nve_plugin_abi_version_minor(void);
const char*       nve_plugin_name(void);
const char*       nve_plugin_author(void);
nve_plugin_mode_t nve_plugin_mode(void);
```

The plugin macros emit all of them; hand-written exports are never needed.

### Versioning

A plugin reports the ABI contract version it was **built against** (the value
of the `NVE_PLUGIN_ABI_VERSION_*` macros at its compile time). The loader
enforces:

- `plugin.major == host.major` — mismatch is a hard error.
- `plugin.minor <= host.minor` — a newer host accepts older-minor plugins; a
  plugin built against a newer minor is rejected.

The external ABI evolves by appending fields to the end of its structs and
bumping the minor. Every growable struct starts with `uint32_t struct_size`;
external consumers check `NVE_ABI_HAS_FIELD(ptr, type, field)` before reading
any field introduced by a later minor.

### Layout baseline

The external ABI targets 64-bit ELF platforms with 32-bit `int32_t`, 64-bit
pointers and default (non-packed) struct layout. `nve_external_plugin.h` contains
static assertions for every published scalar width, struct size and field
offset — a build with incompatible packing options fails at compile time.

## Loading a plugin (host side)

Plugin selection is **by SO path only**:

```cpp
#include <plugin/plugin_loader.hpp>

nve::table_factory_ptr_t factory;
{
  nve::Plugin plugin("libnve-plugin-abseil.so");   // dlopen + validate
  factory = plugin.create_table_factory({});        // plugin-specific JSON
}  // safe: the factory retains the shared-library lease
auto table = factory->produce(table_id, table_config);
```

`Plugin` may live on the stack. Every factory — and every table it produces —
holds a lease on the loaded SO, so plugin code is never unloaded while an
object that references it is alive; the last lease performs the matching
`dlclose`. Bare sonames resolve through the dynamic linker
(`RUNPATH`/`LD_LIBRARY_PATH`) with a fallback next to `libnve-common.so`.
SOs are opened `RTLD_LOCAL`, so plugins from unknown toolchains cannot clash
over symbols.

The convenience helper does all three steps at once:

```cpp
auto table = nve::create_table_from_plugin("libnve-plugin-phmap.so",
                                           factory_config, table_config, id);
```

From C:

```c
nve_table_factory_t factory = NULL;
CHECK_NVE(nve_create_table_factory(&factory, "libnve-plugin-abseil.so",
                                   "{\"num_partitions\": 8}"));
```

`nve_table_factory_produce` derives the C handle's key type from the produced
table's `get_key_size()` (4 → `NVE_KEY_INT32`, 8 → `NVE_KEY_INT64`).

## Writing an internal plugin

Keep your existing `TableFactory` subclass; the whole `plugin.cpp` is one
macro invocation:

```cpp
#include <my_table.hpp>
#include <plugin/nve_internal_plugin.hpp>

NVE_DEFINE_INTERNAL_PLUGIN("My plugin", "My Company",
                           my::MyTableFactory, my::MyTableFactoryConfig)
```

The macro emits the metadata symbols plus a creation symbol that constructs
the factory via `std::make_shared<Factory>(static_cast<Config>(json))`. For a
default-constructed factory that ignores factory JSON, use
`NVE_DEFINE_INTERNAL_PLUGIN_NO_CONFIG(NAME, AUTHOR, FACTORY_TYPE)` (see
`samples/common/custom_remote_plugin`).

CMake: `add_library(nve-plugin-<name> SHARED src/plugin.cpp ...)` linking
`nve-common` — see `plugins/stl_map/CMakeLists.txt` for the smallest example.

In-tree examples: `plugins/{stl_map,abseil,phmap,nvhm,redis,rocksdb,sph}` — see
[Bundled plugins](#bundled-plugins) for what each one is.

## Writing an external plugin

An external plugin includes the pure-C `plugin/nve_external_plugin.h` header
and must not link `nve-common`. The complete working example is
`samples/external_plugin/`.

A produced factory or table is an opaque `self` pointer plus a const,
size-prefixed vtable of C function pointers:

```c
static const nve_ext_table_ops_t kOps = { sizeof(nve_ext_table_ops_t),
  my_destroy, my_pref_loc, my_device_id, my_max_row, my_key_size,
  my_invalid_key, my_dtype, my_counter_hits, my_reset_counter, my_get_counter,
  my_find, my_insert, my_update, my_update_accum, my_erase, my_clear,
  my_last_error };

NVE_DEFINE_EXTERNAL_PLUGIN("My External Plugin", "My Company")
```

Key rules (normative; see the external plugin header for the full contract):

- **No unwinding.** No exception may cross any external function pointer. A
  C++ implementation catches everything and returns `nve_status_t`.
- **Output ownership.** Creation/production functions zero their output
  before doing work, keep it zero on failure (cleaning up any partial state),
  and transfer a nonzero handle only with `NVE_SUCCESS`.
- **Errors.** Fallible ops return `nve_status_t`; `last_error` returns the
  calling thread's message for its last failed op (thread-local storage).
- **Buffers.** The host resolves every buffer to the table's declared
  residency (`preferred_buffer_location`) before the call — the plugin only
  ever sees raw pointers in its preferred space. A `DEVICE` table must report
  a non-negative `device_id`; a `HOST` table must report `-1`.
- **Device bootstrap.** Factory creation, `produce`, factory destruction and
  the initial `device_id` query must select/restore any needed CUDA device
  themselves. After the host caches the table's device ID, every callback and
  destructor runs with that device current.
- **NVE services.** After load, the host calls the optional
  `nve_plugin_set_nve_services` with a process-lifetime, size-prefixed
  callback table: `num_workers`, `execute_n`, `execute_n_wg` (workgroup-aware,
  NUMA parity with in-tree tables), `get_lookup_stream`, `get_modify_stream`,
  `get_scratch`, `last_error`. `execute_n`/`execute_n_wg` are synchronous
  barriers; tasks must not throw and return statuses. `get_scratch` is **not
  thread-safe**: call it only from the table-op thread, never from inside an
  `execute_n` task, and growing a named buffer invalidates pointers previously
  returned for that name. The context handle passed to table ops is valid only
  for the dynamic extent of that call — never retain it.

On the host side the table is wrapped by `ExternalPluginTableFactory`/`ExternalPluginTable`
(`include/plugin/external_plugin_table.hpp`); failure statuses round-trip to C callers
with their exact category (`ExternalPluginError` in C++).
