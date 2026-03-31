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
import torch
import torch.distributed as dist
from socket import gethostname
import pynve.torch.nve_layers as nve_layers
import pynve.nve as nve
import nvtx
import time
import sys
import torchrec
from torchrec.modules.embedding_configs import DataType
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.fbgemm_qcomm_codec import get_qcomm_codecs_registry, QCommsConfig, CommType
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.types import (
    BoundsCheckMode,
    ShardingType,
)
import psutil
import subprocess

import warnings
warnings.filterwarnings("ignore")

from benchmark_util import write_benchmark_csv, benchmark_arg_parser
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'samples', 'pytorch'))
from pytorch_samples_common import gen_key, gen_jagged_key, gen_permute, torch_data_type_to_nve_data_type

needs_dist_cleanup = False

# Initialize multi-process backend
class MultiProc:
    def __init__(self, dist_backend: str = 'gloo'):
        # If torch.distributed was setup, use it (env variables or torchx/torchrun)
        try:
            dist.init_process_group(backend=dist_backend)
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_node_local_rank()
            self.local_device_id = self.local_rank % torch.cuda.device_count()
            self.local_device = torch.device(f"cuda:{self.local_device_id}")

            self.backend = 'dist'
            self.node_name = gethostname()
            global needs_dist_cleanup
            needs_dist_cleanup = True
            print(f"Using torch.distributed [{self.rank}:{self.world_size}] node={self.node_name}, local_rank={self.local_rank}, local_device_id={self.local_device_id},{torch.cuda.get_device_name(self.local_device_id)}")
        except:
            # Failed to initialize dist, fallback to MPI
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.world_size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.node_name = gethostname()
            self.all_node_names = self.comm.allgather(self.node_name)
            self.local_rank = self.all_node_names[:self.rank].count(self.node_name)
            self.local_device_id = self.local_rank % torch.cuda.device_count()
            self.local_device = torch.device(f"cuda:{self.local_device_id}")
            print(f"Using MPI [{self.rank}:{self.world_size}] node={self.node_name}, local_rank={self.local_rank}, local_device_id={self.local_device_id},{torch.cuda.get_device_name(self.local_device_id)}")
            self.backend = 'mpi'

    def Barrier(self):
        if self.backend == 'mpi':
            self.comm.Barrier()
        else:
            dist.barrier()
    def AllGather(self, val):
        if self.backend == 'mpi':
            return self.comm.allgather(val)
        else:
            my_val = torch.tensor([val])
            if dist.get_backend() == 'nccl':
                my_val = my_val.to(device=self.local_device)
            all_values = [torch.zeros_like(my_val) for _ in range(dist.get_world_size())]
            dist.all_gather(all_values, my_val)
            return torch.cat(all_values).tolist()

class AlgMode:
    NVE="nve"
    TorchRec="torchrec"

def parse_command_line():
    parser = benchmark_arg_parser()
    parser.add_argument("--mode", default=AlgMode.NVE, choices=[AlgMode.NVE, AlgMode.TorchRec], help=f"Mode of running (default: {AlgMode.NVE})")
    parser.add_argument("--sharding", default=None, choices=[None, 'row', 'col'], help="TorchRec Sharding constraint (default: None)")
    parser.add_argument("--partial_ranks", "-pr", default=[] ,help="Subset of MPI ranks to use (default: None)", type=int, nargs='*')
    args = parser.parse_args()
    return args

def get_table_size(args):
    if args.data_type == 'float16':
        bytes_per_element = 2
    elif args.data_type == 'float32':
        bytes_per_element = 4
    else:
        bytes_per_element = 4
    return args.num_table_rows * bytes_per_element * args.embedding_dim

def get_nve_model(args, mp : MultiProc):
    device = mp.local_device
    if (args.data_type == 'float32'):
            data_type = torch.float32
    elif (args.data_type == 'float16'):
            data_type = torch.float16
    else:
            data_type = torch.float32

    table_size = get_table_size(args)
    if (args.verbose and mp.rank == 0):
        print(f"Table size: {table_size/(2**30)} GB")
    cache_size = int(table_size * args.load_factor)

    partial_devices = []
    if (args.partial_ranks):
        if (mp.backend != 'mpi'):
            raise ValueError('Partial ranks only supported for MPI')

        last_processor = None
        rank_device = None
        num_devices = torch.cuda.device_count() # assuming this is true for all nodes
        for i,p in enumerate(mp.all_node_names):
            if p != last_processor:
                # new processor
                last_processor = p
                rank_device = 0
            else:
                # next rank in same processor
                rank_device = (rank_device + 1) % num_devices
            if i in args.partial_ranks:
                partial_devices.append(rank_device)
    if (mp.backend == 'mpi'):
        NVL = nve.MPIMemBlock(args.embedding_dim, args.num_table_rows, torch_data_type_to_nve_data_type(data_type), args.partial_ranks, partial_devices)
    else: 
        if (mp.backend == 'dist'):
            from pynve.torch.nve_distributed import TorchDistEnv
            env = TorchDistEnv()
            NVL = nve.DistMemBlock(env, args.embedding_dim, args.num_table_rows, torch_data_type_to_nve_data_type(data_type))
        else:
            raise ValueError('Invalid MP backend')

    if args.partial_ranks and not mp.rank in args.partial_ranks:
        return None
    config = {"kernel_mode": args.kernel,
              "kernel_mode_value_1": args.kernel_mode_value_1,
              "logging_interval": args.logging_interval}
    if (args.logging_interval >  0) and ('NVE_LOG_LEVEL' not in os.environ):
        os.environ['NVE_LOG_LEVEL'] = 'PERF'
    emb_layer = nve_layers.NVEmbedding(
        args.num_table_rows,
        args.embedding_dim,
        data_type,
        nve_layers.CacheType.LinearUVM,
        gpu_cache_size=int(cache_size),
        optimize_for_training=False,
        memblock=NVL,
        device=device,
        config=config)
    return emb_layer

def get_nvlink_bw():
    res = subprocess.run(["nvidia-smi", "nvlink", "-s", "-i", "0"], capture_output=True).stdout
    words = str(res).split()
    nvlink_gbs = 1e-6 # Initialize to epsilon for cases where there's no NVL
    for i in range(len(words)):
      if words[i] == 'Link':
        nvlink_gbs += float(words[i+2])
    return nvlink_gbs * 1e9

def get_nic_bw(nic):
    try:
      with open(f"/sys/class/net/{nic}/speed", 'r', encoding='utf-8') as file:
        return(float(file.read()) * 1e6)
    except Exception as e:
        return 25e9 # default BW

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

def get_torchrec_model(args, mp : MultiProc):
    device = mp.local_device
    rank = mp.rank
    world_size = mp.world_size
    tables_cfg=[]
    tables_cfg.append(
        torchrec.EmbeddingConfig(
            name="t_0",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_table_rows,
            feature_names=["f_0"],
            data_type=DataType.FP32 if args.data_type == 'float32' else DataType.FP16,
        )
    )

    if args.verbose and rank == 0:
        print("tables_cfg: ", tables_cfg)

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
    intra_host_bw = get_nvlink_bw()
    inter_host_bw = get_nic_bw('eth0')

    if args.verbose and rank == 0:
        print(f"Free HBM: {hbm_cap/1e9:.2f} GB")
        print(f"Free DDR: {ddr_cap/1e9:.2f} GB")
        print(f"NVL bandwidth: {intra_host_bw/1e9:.2f} GB/s")
        print(f"NIC bandwidth: {inter_host_bw/1e9:.2f} GB/s")

    # Setup sharding environment
    topology = Topology(
        world_size=world_size,
        local_world_size=world_size,
        compute_device="cuda",
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )

    constraints={}
    for tc in tables_cfg :
        if (args.sharding == 'row') :
            constraints[tc.name] = ParameterConstraints(sharding_types=[ShardingType.ROW_WISE.value],
                                                        enforce_hbm=True,
                                                        bounds_check_mode=BoundsCheckMode.NONE)
        elif (args.sharding == 'col') :
            constraints[tc.name] = ParameterConstraints(sharding_types=[ShardingType.COLUMN_WISE.value],
                                                        enforce_hbm=True,
                                                        bounds_check_mode=BoundsCheckMode.NONE)
        else:
            constraints[tc.name] = ParameterConstraints(enforce_hbm=True,
                                                        bounds_check_mode=BoundsCheckMode.NONE)

    if args.verbose and rank == 0:
        print("Constraints: ", constraints)

    # Create sharding plan
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
    )
    plan = planner.collective_plan(ec, [sharder], dist.GroupMember.WORLD)

    if args.verbose and rank == 0:
        print("Plan: ", plan)

    model = torchrec.distributed.DistributedModelParallel(
        module=ec,
        device=device,
        sharders=[sharder],
        plan=plan,
    )

    return model

def get_model(args, mp : MultiProc):
    if args.mode == AlgMode.NVE:
        return get_nve_model(args, mp)
    else: 
        if args.mode == AlgMode.TorchRec:
            return get_torchrec_model(args, mp)
        else:
            raise ValueError("Invalid mode!")

def gen_keys(args, steps, device, permute):
    if args.mode == AlgMode.NVE:
        keys = []
        keys.extend([gen_key(args.batch, args.hotness, args.alpha, args.num_table_rows, device, permute) for x in range(steps)])
        return keys
    else:
        return [gen_jagged_key(args.batch, args.hotness, args.alpha, args.num_table_rows, device, "f_0", permute) for i in range(steps)]

def main():
    # Init multi process backend
    mp = MultiProc(dist_backend='nccl')

    args = parse_command_line()
    if args.mode == AlgMode.TorchRec and mp.backend != 'dist':
        raise ValueError('Mode \'torchrec\' requires running with torchrun/torchx')
    rank = mp.rank
    if (args.verbose and rank == 0):
        print(args)
    local_device = mp.local_device
    torch.cuda.set_device(local_device)
    stream = torch.cuda.Stream(local_device)
    torch.cuda.set_stream(stream)

    permute=gen_permute(args.num_table_rows, local_device)
    keys = gen_keys(args, args.num_steps, local_device, permute)
    torch.cuda.empty_cache() # free any cached memory held by torch before we allocate our table
    model = get_model(args, mp)

    # warmup
    if (args.verbose and rank == 0): print("Starting Warmup")
    with nvtx.annotate("Warmup", color="purple"):
        for i in range(args.num_warmup_steps):
            warmup_keys = gen_jagged_key(args.batch, args.hotness, args.alpha, args.num_table_rows, local_device, "f_0", permute)
            if args.mode == AlgMode.NVE:
                model(warmup_keys.values())
            else:
                model(warmup_keys)

    torch.cuda.synchronize(local_device)
    mp.Barrier() # barrier to wait for all warmups

    # inference
    if (args.verbose and rank == 0): print("Starting Inference")
    start_time = time.perf_counter()
    with nvtx.annotate("Inference", color="blue"):
        for key in keys:
            model(key)
    torch.cuda.synchronize(local_device)
    run_time = time.perf_counter() - start_time
    all_times = mp.AllGather(run_time)
    total_runtime = 0.0
    total_kps = 0.0
    total_gbps = 0.0
    total_ranks = 0
    if (rank==0): # print perf summary for all ranks
        for i in range(mp.world_size):
            if not args.partial_ranks or i in args.partial_ranks:
                rank_time = all_times[i]
                rank_kps = args.num_steps * args.batch * args.hotness / float(rank_time)
                bytes_per_elem = 4 if args.data_type == 'float32' else 2 if args.data_type == 'float16' else 0
                rank_gbps = rank_kps * args.embedding_dim * bytes_per_elem / 2**30
                print(f"[{i}:{mp.world_size}]  Time: {rank_time:.3f}, MKPS: {(rank_kps/1e6):.2f}, Algo BW: {rank_gbps:.2f} GB/s")
                total_runtime += rank_time
                total_kps += rank_kps
                total_gbps += rank_gbps
                total_ranks += 1
        average_runtime = total_runtime/total_ranks
        average_mkps = total_kps/total_ranks/1e6
        average_gbps = total_gbps/total_ranks
        print(f"[Average]  Time: {average_runtime:.3f}, MKPS: {average_mkps:.2f}, Algo BW: {average_gbps:.2f} GB/s")
        if args.csv_filename:
            write_benchmark_csv(
                args.csv_filename,
                args,
                time_sec=f"{average_runtime:.2f}",
                mkps=f"{average_mkps:.2f}",
                gbps=f"{average_gbps:.2f}"
            )

if __name__ == "__main__":
    try:
        with torch.no_grad():
            main()
    finally:
        if needs_dist_cleanup:
            dist.barrier()
            dist.destroy_process_group()

