# NV Embedding Cache Benchmarks

Benchmarks are located in [../benchmarks/](../benchmarks/), and running instructions are listed below.

All benchmarks print results in the following format:
Time: X.XX, MKPS: Y.YY, Algo BW: Z.ZZ GB/s

Where:
* X.XX is the time it took to run all the inference steps in seconds
* Y.YY is the lookup throughput in million keys per second
* Z.ZZ it the algorithmic BW of the inference loop in GB/s 

## Environment Setup

To run the benchmarks, follow the instructions in the [installation](../README.md#installation) section.
After installing NVE, you also need to install the benchmarks' requirements, using:
```bash
pip install -r benchmarks/requirements.txt
```
Finally, you'll need to install torchrec and fbgemm_gpu.

The examples below were measured on GB200 using driver version: 590.48.01

## Single GPU Benchmark

This benchmark measures the performance of a single embedding table using a single GPU.

* 256GB embedding table in host memory with 25.6GB cache in GPU memory
    ```bash
    # Using NVE
    python benchmarks/single_gpu_bench.py --data_type float32 --embedding_dim 128 --batch 256 --num_table_rows 536870912 --alpha 1.05 --hotness 1024 --num_warmup_steps 10000 --num_steps 100 --mode nv_linear --load_factor 0.1
    Time: 0.06, MKPS: 432.31, Algo BW: 206.14 GB/s

    # Using TorchRec
    python benchmarks/single_gpu_bench.py --data_type float32 --embedding_dim 128 --batch 256 --num_table_rows 536870912 --alpha 1.05 --hotness 1024 --num_warmup_steps 10000 --num_steps 100 --mode torchrec --load_factor 0.1
    Time: 0.14, MKPS: 185.95, Algo BW: 88.67 GB/s
    ```
* 128GB embedding table in GPU memory (no cache)
    ```bash
    # Using NVE
    python benchmarks/single_gpu_bench.py --data_type float32 --embedding_dim 128 --batch 256 --num_table_rows 268435456 --alpha 1.05 --hotness 1024 --num_warmup_steps 100 --num_steps 100 --mode nv_gpu
    Time: 0.01, MKPS: 2910.36, Algo BW: 1387.77 GB/s

    # Using Torch
    python benchmarks/single_gpu_bench.py --data_type float32 --embedding_dim 128 --batch 256 --num_table_rows 268435456 --alpha 1.05 --hotness 1024 --num_warmup_steps 100 --num_steps 100 --mode torch
    Time: 0.01, MKPS: 1880.76, Algo BW: 896.81 GB/s
    ```

## Single Layer C++ Benchmark

`single_layer_bench` drives one embedding layer through the C++ API (no Python or PyTorch) and reports
throughput, per-tier hit rates and per-step latency.
Supported layer types are:
| Layer | Flags | Storage |
|---|---|---|
| GPU layer | `--layer gpu` | table in HBM (`LinearMemBlock`) |
| LinearUVM | `--layer linear_uvm` | GPU cache + table in a `ManagedMemBlock` (host memory), `--kernel 0/1/2` |
| Hierarchical, two tiers | `--layer hierarchical --host_table nvhm\|rocksdb\|linear_file\|redis` | GPU cache + a host tier holding the whole table |
| Hierarchical, three tiers | `--layer hierarchical --host_table nvhm --remote_table rocksdb\|linear_file\|redis` | GPU cache + host cache (`--host_load_factor`) + remote store |
| Hierarchical, no GPU cache | any of the above with `--load_factor 0` | host tiers only, e.g. `linear_file` alone or an `nvhm` cache + `linear_file`; keys and output stay in host memory |
* For the full list of flags use: `./bin/single_layer_bench --help`

```bash
# 32 GB table in host memory, 10% GPU cache
./build_release/bin/single_layer_bench --layer linear_uvm -ed 128 -b 256 --hotness 1024 -nr 67108864 -a 1.05 -lf 0.1 -nw 10000 -ns 100
# GPU cache + nvhm holding the table, 4 inference threads, CSV row per run
./build_release/bin/single_layer_bench --layer hierarchical --host_table nvhm -nr 16777216 -lf 0.01 -t 4 -csv layer.csv
# GPU cache + nvhm cache (10% of the table) + RocksDB on an NVMe drive
./build_release/bin/single_layer_bench --layer hierarchical --host_table nvhm --remote_table rocksdb -hlf 0.1 --file_dir /mnt/nvme
# GPU cache + the table file read with O_DIRECT
./build_release/bin/single_layer_bench --layer hierarchical --host_table linear_file --file_dir /mnt/nvme
# no GPU cache: nvhm cache (10% of the table) in front of the table file
./build_release/bin/single_layer_bench --layer hierarchical --host_table nvhm --remote_table linear_file -lf 0 -hlf 0.1 --file_dir /mnt/nvme
```

Output:
```
Layer: linear_uvm layer, GPU cache + table in a ManagedMemBlock (host memory); float32 x 128, 67108864 rows (32.00 GB), GPU cache 6710886 rows (3.200 GB), keys on gpu
Setup: 5.60 s (prefill 5.57 s, file create 0.00 s), host mem delta 32.21 GB
Time: 0.08, MKPS: 322.28, Algo BW: 153.68 GB/s (1 thread x 100 steps of 256x1024 keys)
Hitrates: gpu (gpu) 0.8428 [0.8413, 0.8443] over 100 samples
Latency-phase hitrates (fresh keys): gpu (gpu) 0.8428 [0.8413, 0.8451] over 100 samples
Latency (100 steps of 256x1024 keys): mean 0.817 ms, p50 0.811, p95 0.818, p99 1.033, max 1.192 ms
```
Hit rates are the fraction of all requested keys resolved by each tier (they sum to 1 when every key is found).
The CSV holds the run parameters, `mkps`/`gbps`, `<tier>_hitrate_mean/min/max/samples`, `lat_*` percentiles and
the setup, prefill and file creation times.

## Multi GPU Benchmark

This benchmark measures the performance of a single embedding table sharded on multiple GPUs. Performance metrics will be printed per GPU.  
Examples below use torchrun, this can be replaced with torchx and for NVE also with mpirun.

* 0.5TB embedding table sharded on 8 GPUs
    ```bash
    # Using NVE with torchrun (with 5GB cache per GPU)
    torchrun --nproc_per_node=8 benchmarks/multi_gpu_bench.py --data_type float32 --embedding_dim 128 --batch 2048 --num_table_rows 1073741824 --alpha 1.05 --hotness 1024 --num_warmup_steps 10000 --num_steps 100 --mode nve --load_factor 0.01
    ...
    [0:8]  Time: 0.040, MKPS: 5306.04, Algo BW: 2530.12 GB/s
    [1:8]  Time: 0.040, MKPS: 5259.12, Algo BW: 2507.74 GB/s
    [2:8]  Time: 0.040, MKPS: 5213.92, Algo BW: 2486.19 GB/s
    [3:8]  Time: 0.040, MKPS: 5216.35, Algo BW: 2487.35 GB/s
    [4:8]  Time: 0.040, MKPS: 5258.73, Algo BW: 2507.56 GB/s
    [5:8]  Time: 0.040, MKPS: 5261.06, Algo BW: 2508.67 GB/s
    [6:8]  Time: 0.041, MKPS: 5072.43, Algo BW: 2418.72 GB/s
    [7:8]  Time: 0.040, MKPS: 5262.38, Algo BW: 2509.30 GB/s
    [Average]  Time: 0.040, MKPS: 5231.25, Algo BW: 2494.46 GB/s

    # Using TorchRec (no cache)
    torchrun --nproc_per_node=8 benchmarks/multi_gpu_bench.py --data_type float32 --embedding_dim 128 --batch 2048 --num_table_rows 1073741824 --alpha 1.05 --hotness 1024 --num_warmup_steps 10000 --num_steps 100 --mode torchrec
    ...
    [0:8]  Time: 0.354, MKPS: 592.34, Algo BW: 282.45 GB/s
    [1:8]  Time: 0.354, MKPS: 592.36, Algo BW: 282.46 GB/s
    [2:8]  Time: 0.354, MKPS: 592.33, Algo BW: 282.44 GB/s
    [3:8]  Time: 0.354, MKPS: 592.31, Algo BW: 282.44 GB/s
    [4:8]  Time: 0.354, MKPS: 592.35, Algo BW: 282.45 GB/s
    [5:8]  Time: 0.354, MKPS: 592.33, Algo BW: 282.44 GB/s
    [6:8]  Time: 0.354, MKPS: 592.36, Algo BW: 282.46 GB/s
    [7:8]  Time: 0.354, MKPS: 592.34, Algo BW: 282.45 GB/s
    [Average]  Time: 0.354, MKPS: 592.34, Algo BW: 282.45 GB/s
    ```
