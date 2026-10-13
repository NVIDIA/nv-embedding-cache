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
- `weights/mb-<id>.nve` — embedding weights in NVE binary format (one file per
  storage resource, named by the layer's `storage_ref`)

`--mode hierarchical` writes `samples/cpp_inference/output_hierarchical/` with
`metadata.json` carrying a `resources.remote_ps` entry (`plugin_name`,
`factory_config`, `table_config`) plus `keys.npy` / `values.npy` referenced via
`remote_ps_data`, so the loader can refill the in-memory PS on a fresh process.

### 2. Run C++ Inference

```bash
# Same binary handles linearuvm/hierarchical/gpu — point it at the export directory:
./<build_folder>/bin/nve_inference samples/cpp_inference/output
./<build_folder>/bin/nve_inference samples/cpp_inference/output_hierarchical
```

The `nve_inference` binary doesn't care which mode produced the directory.
For Hierarchical entries, `nve::LayerDirectory` reads `resources.remote_ps` and
constructs an `nve::ParameterServerTable` via the plugin-based ctor —
`libnve-plugin-custom_remote.so` (built from
`samples/common/custom_remote_plugin/`) is `dlopen`'d at runtime by the
host-table loader, so the binary has no link-time dependency on it.

The C++ program:
1. Loads the AOT model via `AOTIModelPackageLoader`
2. Constructs `nve::LayerDirectory`, which reads `metadata.json`, creates the
   NVE embedding layers and loads weights from the `.nve` files
3. Binds each layer's marker constant in the model and registers the layer in
   `NVELayerRegistry`
4. Runs inference and prints results

## Key APIs

**Python export** (`pynve.torch.nve_export`):
```python
from pynve.torch.nve_export import export_aot
export_aot(model, (example_keys,), "output_dir/")
```

**C++ load and run** (`nve_loader.hpp`):
```cpp
auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>("output_dir/model.pt2");
auto resources = std::make_shared<nve::ResourceDirectory>();
nve::LayerDirectory dir("output_dir", *loader, device_index, resources);
auto outputs = loader->run({keys});
loader.reset();  // dir owns the marker tensors and must outlive the loader
```
