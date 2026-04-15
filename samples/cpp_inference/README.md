# C++ Inference Sample

Demonstrates end-to-end C++ inference using an NVEmbedding model exported from Python via AOTInductor.

## Flow

### 1. Export from Python

```bash
python samples/cpp_inference/export_for_cpp.py
```

This creates `samples/cpp_inference/output/`:
- `model.pt2` — AOTInductor-compiled model
- `metadata.json` — NVE layer configuration
- `weights/emb.nve` — embedding weights in NVE binary format

### 2. Run C++ Inference

```bash
./<build_folder>/bin/nve_inference samples/cpp_inference/output
```

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
