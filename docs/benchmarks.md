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
PIP_CONSTRAINT="" pip install -r benchmarks/requirements.txt
```
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
