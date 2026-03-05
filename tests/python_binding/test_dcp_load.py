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
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

import pynve.nve as nve
import pynve.torch.nve_ps as nve_ps
import pynve.torch.nve_layers as nve_layers


class ToyModel(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, use_nve):
        super(ToyModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if not use_nve:
            self.embed = nn.Embedding(num_embeddings, embedding_dim)
        else:
            cache_size = 1024
            cache_type = nve_layers.CacheType.LinearUVM
            remote_interface = nve_ps.NVEParameterServer(0, embedding_dim, torch.float32, None) if cache_type == nve_layers.CacheType.Hierarchical else None
            memblock = nve.LinearMemBlock(num_embeddings, embedding_dim, nve.DataType_t.Float32) if cache_type == nve_layers.CacheType.LinearUVM else None
            self.embed = nve_layers.NVEmbedding(num_embeddings, embedding_dim, torch.float32, memblock=memblock, cache_type=cache_type, remote_interface=remote_interface, gpu_cache_size=cache_size)

    def forward(self, x):
        return self.embed(x)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


num_embeddings = 10
embedding_dim = 2
num_keys = 4

def run_fsdp_checkpoint_save_example(rank, world_size, checkpoint_dir):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")

    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel(num_embeddings, embedding_dim, False).to(rank)
    model = fully_shard(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    
    keys = torch.randint(1, num_embeddings, (num_keys,), device=model.embed.weight.device, dtype=torch.int64)
    output = model(keys)
    loss = loss_fn(output, torch.randn(num_keys, embedding_dim, device=model.embed.weight.device, dtype=torch.float32))
    loss.backward()
    optimizer.step()

    full_state_dict = model.state_dict()
    state_dict = { "embedding": full_state_dict }
    
    dcp.save(state_dict, checkpoint_id=checkpoint_dir)
    cleanup()

def run_fsdp_checkpoint_load_example(rank, world_size, checkpoint_dir, compare=False):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    

    setup(rank, world_size)

    # create a reference model to test our loading (don't use NVE layer)
    model_ref = ToyModel(num_embeddings, embedding_dim, False).to(rank)
    model_ref = fully_shard(model_ref)

    state_dict = { "embedding": model_ref.state_dict() }
    dcp.load(state_dict, checkpoint_id=checkpoint_dir)

    
    # create a test model with an NVE layer for embedding
    model_test = ToyModel(num_embeddings, embedding_dim, True)
    
    
    # in case of multi nodes sharing a single memblock only one node will load the tensor
    if rank == 0:
        # to load it is required to map the required tensor to load to the the appropriate 
        # nve layer.weight tensor
        # e.g the model have a torch.nn.Embedding layer with a weight parameter -> 
        # this means the checkpoint saved dict will have a key "embed.weight" 
        # we want to load it to the corresponding nve layer.weight tensor
        # you can inspect the model.state_dict() to figure the keys and create the proper state_dict
        tensor_name = "embed.weight"
        partial_state_dict = { "embedding": { tensor_name: model_test.embed.weight } }
    else:
        partial_state_dict = { "embedding": {  } }
    
    dcp.load(partial_state_dict, checkpoint_id=checkpoint_dir)

    if compare:
        assert torch.equal(model_ref.embed.weight.to_local(), model_test.embed.weight)
    else:
        print(f"model {rank} ", model_ref.embed.weight.to_local())
        print(f"layer {rank} ", model_test.embed.weight)

    cleanup()

def test_dcp_load():
    world_size = torch.cuda.device_count()
    checkpoint_dir = "/tmp/checkpoint"
    mp.spawn(
            run_fsdp_checkpoint_save_example,
            args=(world_size, checkpoint_dir),
            nprocs=world_size,
            join=True,
        )
    load_world_size = 1
    
    mp.spawn(
            run_fsdp_checkpoint_load_example,
            args=(load_world_size, checkpoint_dir, True),
            nprocs=load_world_size,
            join=True,
        )

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    checkpoint_dir = "checkpoint"
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    if sys.argv[1] == "save":
        mp.spawn(
            run_fsdp_checkpoint_save_example,
            args=(world_size, checkpoint_dir),
            nprocs=world_size,
            join=True,
        )
    elif sys.argv[1] == "load":
        world_size = 1
        mp.spawn(
            run_fsdp_checkpoint_load_example,
            args=(world_size, checkpoint_dir, False),
            nprocs=world_size,
            join=True,
        )