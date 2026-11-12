# Layer Sample: Embedding Layer Inference

This sample runs an inference-style lookup workload on NVE embedding layers using the C++ API and reports per-tier hit rates and timing.

The sample:
1. Creates one or more embedding layers of the selected type (see [Layer types](#layer-types))
2. Generates random lookup inputs with a power-law key distribution
3. Warms up the caches by inserting the warmup inputs into the layer
4. Runs lookups from one or more parallel workload runners, each with its own execution context and CUDA streams
5. Optionally runs a TensorRT engine on the lookup output, to simulate the dense part of a model
6. Prints timing and hit rates, and optionally appends them to a CSV file

To build the sample see the [Getting started](../../README.md#getting-started) section in the main [README.md](../../README.md).

## Layer types

Select the layer type with `--layer_type`:

|Value|Layer|Tables|
|-----|-----|------|
|0 (default)|`HierarchicalEmbeddingLayer`|GPU cache (`GpuTable`) -> host cache (NVHM plugin table) -> mock parameter server (`MockHostTable`)|
|1|`LinearUVMEmbeddingLayer`|GPU cache backed by a linear table in pinned host memory|
|2|`HierarchicalEmbeddingLayer`|A single `LinearHostTable` in host memory (CPU lookups, no GPU cache)|

Hit rates are printed as three values per layer: GPU, host and remote.

> **Note:** layer types 1 and 2 have a single table, but the warmup step inserts into two tables and fails with `Invalid table_id`. Run these types with `--warmup_iterations 0`. With layer type 1, use `--cache_heuristic_target` to fill the GPU cache during inference.

## Running

Hierarchical layer, printing hit rates every 250 iterations:

```bash
$ ./build/bin/layer_sample --verbose -li 250 -nr 1
Creating layer (0)
Cache warmup (250)
Warmup done (3922.38 ms)
Inference (1000)
Inference done (total E2E: 4381.2 ms)
Worker 0:
  Iteration 0, Layer 0: Hitrates: 0.805462 0.0673485 0.12719
  Iteration 250, Layer 0: Hitrates: 0.807148 0.0663986 0.126453
  Iteration 500, Layer 0: Hitrates: 0.805462 0.0673485 0.12719
  Iteration 750, Layer 0: Hitrates: 0.807148 0.0663986 0.126453
```

Linear layer with a 4 GB table (16M rows of 256 bytes), filling the GPU cache with the insert heuristic:

```bash
$ ./build/bin/layer_sample --verbose -lt 1 -r 16777216 -wi 0 -cht 0.9 -li 250 -nr 1
...
Worker 0:
  Iteration 0, Layer 0: Hitrates: 0 0 0
  Iteration 250, Layer 0: Hitrates: 0.790401 0 0
  Iteration 500, Layer 0: Hitrates: 0.997929 0 0
  Iteration 750, Layer 0: Hitrates: 0.998684 0 0
```

## Options

Run `./build/bin/layer_sample --help` for the full list.

|Option|Default|Description|
|------|-------|-----------|
|`-lt`, `--layer_type`|0|Layer type, see [Layer types](#layer-types)|
|`-nl`, `--num_layers`|1|Number of layers (all of the same type)|
|`-gcs`, `--gpu_cache_size`|1|GPU cache size in GB|
|`-hcs`, `--host_cache_size`|10|Host cache size in GB (layer type 0)|
|`-cht`, `--cache_heuristic_target`|0|Target hit rate for the insert heuristic, 0 disables inserts during inference (layer types 0, 1)|
|`-r`, `--rows`|67108864|Number of rows (key range); also the linear table size for layer types 1, 2|
|`-rs`, `--row_size`|256|Row size in bytes|
|`--hotness`|512|Number of keys per batch. Use the long form, `-h` prints help|
|`-bs`, `--batch_size`|512|Number of batches per lookup|
|`-a`, `--alpha`|1.05|Power-law alpha for key generation|
|`-is`, `--input_sets`|100|Number of pre-generated inference inputs|
|`-ws`, `--warmup_sets`|250|Number of pre-generated warmup inputs|
|`-wi`, `--warmup_iterations`|250|Number of warmup iterations|
|`-i`, `--iterations`|1000|Number of inference iterations per runner|
|`-nr`, `--num_runners`|2|Number of parallel workload runners|
|`-pl`, `--ps_latency`|0|Simulated mock parameter server latency per key, in nanoseconds (layer type 0)|
|`-gm`, `--gpu_modify`|off|Run cache modify operations on GPU (layer type 1)|
|`-km`, `--kernel_mode`|0|Force a GPU table kernel mode, 0 uses defaults (layer type 1)|
|`-te`, `--trt_engine`|none|TensorRT engine file to run after the lookups|
|`-dc`, `--disable_copy_output`|off|Skip copying the output back to host memory|
|`-li`, `--logging_interval`|0|Print hit rates every N iterations, 0 disables|
|`-csv`, `--csv_filename`|none|Append parameters and metrics to a CSV file|
|`--verbose`|off|Print progress and timing|

## Code structure

* [layer_sample.cu](layer_sample.cu) - command line parsing, layer creation and metrics reporting.
* [workload_runner.hpp](workload_runner.hpp) - input generation (`LayerData`) and the warmup and inference loops (`WorkloadRunner`).
