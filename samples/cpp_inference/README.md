# C++ Inference Sample

Demonstrates end-to-end C++ inference using an NVEmbedding model exported from Python via AOTInductor.

## Flow

### 1. Export from Python

```bash
# LinearUVM (default): host-stored weights with a GPU cache
python samples/cpp_inference/export_for_cpp.py

# Hierarchical: GPU cache → custom_remote plugin-backed remote PS
python samples/cpp_inference/export_for_cpp.py --mode hierarchical
```

`--mode linearuvm` writes `samples/cpp_inference/output/`:
- `model.pt2` — AOTInductor-compiled model
- `metadata.json` — NVE layer configuration
- `weights/emb.nve` — embedding weights in NVE binary format

`--mode hierarchical` writes `samples/cpp_inference/output_hierarchical/` with
`metadata.json` carrying a `remote_ps_config` (`plugin_name`, `factory_config`,
`table_config`) plus `keys.npy` / `values.npy` referenced via `remote_ps_data`,
so the loader can refill the in-memory PS on a fresh process.

### 2. Run C++ Inference

```bash
# Same binary handles both modes — point it at the export directory:
./<build_folder>/bin/nve_inference samples/cpp_inference/output
./<build_folder>/bin/nve_inference samples/cpp_inference/output_hierarchical
```

The `nve_inference` binary doesn't care which mode produced the directory.
For Hierarchical entries, `nve::LayerDirectory` reads `remote_ps_config` and
constructs an `nve::ParameterServerTable` via the plugin-based ctor —
`libnve-plugin-custom_remote.so` (built from
`samples/common/custom_remote_plugin/`) is `dlopen`'d at runtime by the
host-table loader, so the binary has no link-time dependency on it.

The C++ program:
1. Reads `metadata.json` and creates NVE embedding layers
2. Loads weights from `.nve` files into the layers
3. Registers layers in `NVELayerRegistry`
4. Loads the AOT model via `AOTIModelPackageLoader`
5. Runs inference and prints results

## Key APIs

**Python export** (`pynve.torch.nve_export`):
```python
from pynve.torch.nve_export import export_aot
export_aot(model, (example_keys,), "output_dir/")
```

**C++ load and run** (`nve_loader.hpp`):
```cpp
auto layers = nve::load_nve_layers("output_dir/");
torch::inductor::AOTIModelPackageLoader loader("output_dir/model.pt2");
auto outputs = loader.run({keys});
nve::unload_nve_layers(layers);
```
