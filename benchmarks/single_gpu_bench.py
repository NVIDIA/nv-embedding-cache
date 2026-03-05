#!/usr/bin/python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import tempfile
import torch.distributed as dist
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
import pynve.nve as nve
import time
import nvtx
import psutil
import socket

import warnings
warnings.filterwarnings("ignore")

from benchmark_util import write_benchmark_csv, benchmark_arg_parser
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'samples', 'pytorch'))
from pytorch_samples_common import gen_key, gen_jagged_key, parse_hitrates_from_log, gen_permute

def is_redis_running(address: str, timeout: float = 1.0) -> bool:
    """Check if a Redis server is running at the given address (host:port)."""
    host, port = address.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as sock:
            sock.sendall(b"PING\r\n")
            response = sock.recv(1024)
            return response.strip() == b"+PONG"
    except (socket.timeout, ConnectionRefusedError, OSError, ValueError):
        return False

try:
    import torchrec
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.modules.embedding_configs import DataType
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embedding import EmbeddingCollectionSharder
    from torchrec.distributed.types import CacheParams
    from torchrec.distributed.fbgemm_qcomm_codec import get_qcomm_codecs_registry, QCommsConfig, CommType
    from fbgemm_gpu.split_embedding_configs import EmbOptimType
    from fbgemm_gpu.split_embedding_configs import SparseType

except ImportError:
    print("torchrec is not installed, torchrec benchmarks will not be run")

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

dist.init_process_group(backend="nccl")

class AlgMode:
    NVLinearUVM="nv_linear"
    NVHierarchical="nv_hierarchical"
    NVGPU="nv_gpu"
    TorchRec="torchrec"
    Torch="torch"
    TorchCPU="torch_cpu"

def write_metrics_to_csv(args, infer_time, kps, gbps, log_content=None):
    """Write benchmark metrics to CSV file, parsing hitrates from log content if available.
    
    Args:
        args: Command line arguments
        infer_time: Inference time in seconds
        kps: Keys per second
        gbps: Bandwidth in GB/s
        log_content: Optional log content string to parse for hitrates
    """
    # Base metrics (always included)
    metrics = {
        'inference_time_s': f"{infer_time:.4f}",
        'mkps': f"{(kps/1e6):.2f}",
        'gbps': f"{gbps:.2f}",
    }
    
    # Parse and add hitrates from log content if available
    if log_content and (args.mode == AlgMode.NVLinearUVM or args.mode == AlgMode.NVHierarchical or args.mode == AlgMode.NVGPU):
        gpu_stats, host_stats, remote_stats = parse_hitrates_from_log(log_content)
        gpu_mean, gpu_min, gpu_max, gpu_samples = gpu_stats
        host_mean, host_min, host_max, host_samples = host_stats
        remote_mean, remote_min, remote_max, remote_samples = remote_stats
        
        # Add hitrate metrics
        metrics.update({
            'gpu_hitrate_mean': f"{gpu_mean:.6f}",
            'gpu_hitrate_min': f"{gpu_min:.6f}",
            'gpu_hitrate_max': f"{gpu_max:.6f}",
            'gpu_hitrate_samples': gpu_samples,
            'host_hitrate_mean': f"{host_mean:.6f}",
            'host_hitrate_min': f"{host_min:.6f}",
            'host_hitrate_max': f"{host_max:.6f}",
            'host_hitrate_samples': host_samples,
            'remote_hitrate_mean': f"{remote_mean:.6f}",
            'remote_hitrate_min': f"{remote_min:.6f}",
            'remote_hitrate_max': f"{remote_max:.6f}",
            'remote_hitrate_samples': remote_samples,
        })
    
    # Write to CSV - args are auto-extracted, metrics passed as kwargs
    write_benchmark_csv(args.csv_filename, args, **metrics)

def loop(layer, key_set, offsets, results, target, waitable, dictable):
    for keys in key_set:
        if offsets != None:
            output = layer(keys, offsets)
        else:
            output = layer(keys)
        if waitable:
            output = output.wait()
        if dictable:
            output = output.to_dict()

        if isinstance(output, torch.Tensor) and (output.device.type == "cpu"):
            output = output.to(torch.device("cuda"))
        if results != None:
            results.append(output)

def parse_command_line():
    parser = benchmark_arg_parser()
    parser.add_argument("--mode", default=AlgMode.NVLinearUVM, choices=[AlgMode.NVLinearUVM, AlgMode.NVHierarchical, AlgMode.NVGPU, AlgMode.TorchRec,AlgMode.Torch, AlgMode.TorchCPU ],help=f"Algorithm Mode of the script can be either [default: {AlgMode.NVLinearUVM}]")
    parser.add_argument("--clear", "-c", action='store_true',help="Clear tables before warmup (useful persistent parameter server)")
    parser.add_argument("--prefill", "-p", action='store_true',help="Prefill tables before warmup")
    parser.add_argument("--host_load_factor", "-hlf", default=0.01 ,help="Load factor of the CPU cache(default: 0.01)", type=float)
    parser.add_argument("--redis_address", "-ra", default='localhost:7000', help="Address of a Redis server in formatted as 'host:port' (default: 'localhost:7000', only for hierarchical)")
    args = parser.parse_args()
    return args

def get_sharder_params(args):
    #set optimizer args
    learning_rate = 0.1
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0
    eps = 0.001

    #Put args into a optimizer kwargs , which is same usage of TorchREC
    optimizer_kwargs = {"optimizer":EmbOptimType.EXACT_SGD,
                        "learning_rate": learning_rate,
                        "beta1":beta1,
                        "beta2":beta2,
                        "weight_decay":weight_decay,
                        "eps":eps}

    fused_params = {}
    fused_params["output_dtype"] = SparseType.FP32 if args.data_type == 'float32' else SparseType.FP16
    fused_params.update(optimizer_kwargs)

    qcomm_codecs_registry = (
            get_qcomm_codecs_registry(
                qcomms_config=QCommsConfig(
                    # pyre-ignore
                    forward_precision= CommType.FP32 if args.data_type == 'float32' else CommType.FP16,
                    # pyre-ignore
                    backward_precision= CommType.FP32 if args.data_type == 'float32' else CommType.FP16,
                )
            )
        )

    return qcomm_codecs_registry, fused_params

def get_torchrec_model(args):
    tables_cfg=[]
    tables_cfg.append(
        torchrec.EmbeddingConfig(
            name="large_table",
            embedding_dim=args.embedding_dim,
            num_embeddings=int(args.num_table_rows),
            feature_names=["my_feature"],
            data_type=DataType.FP32 if args.data_type == 'float32' else DataType.FP16,
        )
    )
    ec = torchrec.EmbeddingCollection(
        device=torch.device("meta"),
        tables=tables_cfg
    )
    qcomm_codecs_registry, fused_params = get_sharder_params(args)
    sharder = EmbeddingCollectionSharder(
        qcomm_codecs_registry=qcomm_codecs_registry,
        fused_params=fused_params,
        use_index_dedup=True,
    )

    free_bytes, _ = torch.cuda.mem_get_info()
    hbm_cap = free_bytes
    memory_info = psutil.virtual_memory()
    ddr_cap = memory_info.free

    # Setup sharding environment
    topology = Topology(
        world_size=1,
        local_world_size=1,
        compute_device="cuda",
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,
    )

    uvm_caching_constraints = {
        "large_table": ParameterConstraints(
            sharding_types=["table_wise"],
            cache_params=CacheParams(
                load_factor=float(args.load_factor),
            ),
        )
    }

    # Create sharding plan
    uvm_caching_plan = EmbeddingShardingPlanner(
            topology=topology, constraints=uvm_caching_constraints
        ).plan(
            ec, [sharder]
        )

    model = torchrec.distributed.DistributedModelParallel(
           ec,
           device=torch.device("cuda"),
           plan = uvm_caching_plan
    )

    if args.verbose: print(model.plan)
    return model

def get_nve_model(args, data_type, embedding_dim, num_embeddings):
    if data_type == torch.float16:
        bytes_per_element = 2
    elif data_type == torch.float32:
        bytes_per_element = 4
    else:
        bytes_per_element = 4

    load_factor = float(args.load_factor)
    host_load_factor = float(args.host_load_factor)
    gpu_cache_size = int(int(args.num_table_rows) * load_factor ) * bytes_per_element * embedding_dim
    host_cache_size = int(int(args.num_table_rows) * host_load_factor ) * bytes_per_element * embedding_dim
    config = {"kernel_mode": args.kernel,
              "kernel_mode_value_1": args.kernel_mode_value_1,
              "logging_interval": args.logging_interval}
    if (args.logging_interval >  0) and ('NVE_LOG_LEVEL' not in os.environ):
        os.environ['NVE_LOG_LEVEL'] = 'PERF'
    if args.mode == AlgMode.NVLinearUVM:
        model = nve_layers.NVEmbedding(num_embeddings, embedding_dim, data_type, nve_layers.CacheType.LinearUVM, gpu_cache_size=gpu_cache_size, optimize_for_training=False, config=config)
    elif args.mode == AlgMode.NVHierarchical:
        if args.host_load_factor >= 1.0:
            # Host cache is storing everything (no remote)
            # Using GPU cache + NVHM as a parameter server (using local host memory)
            nvhm_ps = nve_ps.NVEParameterServer(
                0, # Setting num_embeddings as 0 to disable eviction policy
                embedding_dim,
                data_type,
                initializer=None, # use the prefill_last_table option to initialize cache
                initial_size=int(num_embeddings * 1.1),
                ps_type=nve.NVHashMap,
            )
            model = nve_layers.NVEmbedding(
                num_embeddings, embedding_dim, data_type, nve_layers.CacheType.Hierarchical,
                gpu_cache_size=gpu_cache_size, host_cache_size=0, remote_interface=nvhm_ps, optimize_for_training=False, config=config)
        else:
            # Using 3 levels of storage: GPU, CPU and Remote (Redis)

            # Redis parameter server
            if not is_redis_running(address=args.redis_address):
                raise RuntimeError(f"Redis server is not reachable at: {args.redis_address}, use the --redis_address flag to set the address of a running Redis server")
            redis_ps = nve_ps.NVEParameterServer(
                num_embeddings,
                embedding_dim,
                data_type,
                initializer=None, # use the prefill_last_table option to initialize cache
                ps_type=nve.Redis,
                extra_params = {"plugin": {"address": args.redis_address}}
            )

            model = nve_layers.NVEmbedding(
                num_embeddings, embedding_dim, data_type, nve_layers.CacheType.Hierarchical,
                gpu_cache_size=gpu_cache_size, host_cache_size=host_cache_size, remote_interface=redis_ps,
                optimize_for_training=False, config=config)
    elif args.mode == AlgMode.NVGPU:
        model = nve_layers.NVEmbedding(num_embeddings, embedding_dim, data_type, nve_layers.CacheType.NoCache, optimize_for_training=False, config=config)

    if args.clear:
        model.clear()
    return model

def get_torch_model(args, data_type, embedding_dim, num_embeddings):
    if args.mode == AlgMode.TorchCPU:
        model = torch.nn.Embedding(num_embeddings, embedding_dim, sparse=True, device=torch.device("cpu"))
    else:
        model = torch.nn.Embedding(num_embeddings, embedding_dim, sparse=True, device=torch.device("cuda"))
    return model

def get_device(args):
    if args.mode == AlgMode.TorchCPU:
        return torch.device("cpu")
    else:
        return torch.device("cuda")

def main():
    torch.set_num_threads(os.cpu_count())
    args = parse_command_line()

    embedding_dim = args.embedding_dim
    batch = args.batch
    hotness = args.hotness
    alpha = args.alpha
    num_warmup_steps = args.num_warmup_steps
    num_steps = args.num_steps
    num_embeddings = args.num_table_rows

    if (args.data_type == 'float32'):
        data_type = torch.float32
    elif (args.data_type == 'float16'):
        data_type = torch.float16
    else:
        data_type = torch.float32
        print("ignoring data type parameter, running with float32")

    if args.mode == AlgMode.TorchRec:
        model = get_torchrec_model(args)
        waitable = True
        dictable = False
    elif args.mode == AlgMode.NVLinearUVM or args.mode == AlgMode.NVHierarchical or args.mode == AlgMode.NVGPU:
        model = get_nve_model(args, data_type, embedding_dim, num_embeddings)
        waitable = False
        dictable = False
    elif args.mode == AlgMode.Torch or args.mode == AlgMode.TorchCPU:
        model = get_torch_model(args, data_type, embedding_dim, num_embeddings)
        waitable = False
        dictable = False

    num_keys = batch*hotness
    target = torch.zeros(num_keys, embedding_dim, dtype=data_type).cuda()

    if args.verbose: print("Starting to generate keys")
    key_set = []

    if hasattr(model, 'cache_type') and model.cache_type == nve_layers.CacheType.Hierarchical:
        perm = torch.randint(low=0, high=2**60, size=(num_embeddings,), dtype=torch.int64, device=torch.device("cuda"))
    else:
        perm = gen_permute(args.num_table_rows, torch.device("cuda"))
    if args.mode == AlgMode.TorchRec:
        key_set = [gen_jagged_key(batch, hotness, alpha, num_embeddings, torch.device("cuda"), "my_feature", permute=perm) for i in range(num_steps)]
    else:
        key_set = [gen_key(batch, hotness, alpha, num_embeddings, torch.device("cuda"), permute=perm) for i in range(num_steps)]

    if args.prefill and isinstance(model, nve_layers.NVEmbedding):
        if args.verbose: print("Prefill")
        batch_size = 2**16
        start_key = 0
        values = torch.zeros(batch_size, embedding_dim, dtype=data_type, device="cuda")
        while start_key < num_embeddings:
            end_key = min(start_key + batch_size, num_embeddings)
            keys = torch.arange(start_key, end_key, device="cuda")
            if perm != None:
                keys = perm[keys]
            model.insert(keys, values, 0)
            if model.cache_type == nve_layers.CacheType.Hierarchical:
                model.insert(keys, values, 1)
                model.insert(keys, values, 2) # Calling insert on a nonexistent table is safe, will only yield a warning.
            start_key += batch_size

    # move if needed
    if get_device(args) != torch.device("cuda"):
        key_set = [key.to(get_device(args)) for key in key_set]

    if args.verbose: print("Warming up")

    if (args.verbose): print("Starting Warmup")
    with nvtx.annotate("Warmup", color="purple"):
        for i in range(max(1, int(num_warmup_steps/100))):
            if args.mode == AlgMode.TorchRec:
                warmup_key_set = [gen_jagged_key(batch, hotness, alpha, num_embeddings, torch.device("cuda"), "my_feature", permute=perm) for i in range(100)]
            else:
                warmup_key_set = [gen_key(batch, hotness, alpha, num_embeddings, torch.device("cuda"), permute=perm) for i in range(100)]
            # move if needed
            if get_device(args) != torch.device("cuda"):
                warmup_key_set = [key.to(get_device(args)) for key in warmup_key_set]
            loop(model, warmup_key_set, None, None, target, waitable, dictable)
    torch.cuda.synchronize()
    if args.verbose: print("Done warming up")

    # Redirect stdout to temp file if CSV output is requested (to capture C++ hitrate logs)
    temp_log_path = None
    saved_stdout_fd = None
    if args.csv_filename:
        # Create temp file for capturing stdout
        temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_log_path = temp_log_file.name
        temp_log_file.close()
        # Save the original stdout file descriptor and redirect to temp file
        saved_stdout_fd = os.dup(1)
        log_fd = os.open(temp_log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(log_fd, 1)
        os.close(log_fd)

    try:
        if (args.verbose): print("Starting Inference")
        with nvtx.annotate("Inference", color="blue"):
            start_time = time.perf_counter()
            loop(model, key_set, None, None, target, waitable, dictable)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
    finally:
        if saved_stdout_fd is not None:
            try:
                os.fsync(1)
            except (OSError, IOError):
                pass
            # Now restore the original stdout
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
    
    # Read and print captured output from temp file (if we redirected stdout)
    log_content = None
    if temp_log_path:
        try:
            with open(temp_log_path, 'r') as f:
                log_content = f.read()
                print(log_content, end='')
        except Exception as e:
            print(f"Warning: Could not read temp log file: {e}")
        finally:
            # Delete temp log file now that we've read it
            try:
                os.unlink(temp_log_path)
            except:
                pass

    infer_time = end_time - start_time
    kps = int(args.num_steps) * int(args.batch) * int(args.hotness) / float(infer_time)
    bytes_per_elem = 4 if args.data_type == 'float32' else 2 if args.data_type == 'float16' else 0
    gbps = kps * int(args.embedding_dim) * bytes_per_elem / 2**30
    print(f"Time: {infer_time:.2f}, MKPS: {(kps/1e6):.2f}, Algo BW: {gbps:.2f} GB/s")
    dist.destroy_process_group()
    if args.verbose: print("Finished Benchmark")

    # Delete model to trigger cleanup
    del model
    
    # Write metrics to CSV if requested
    if args.csv_filename:
        write_metrics_to_csv(args, infer_time, kps, gbps, log_content)

if __name__ == "__main__":
    with torch.no_grad():
        main()
