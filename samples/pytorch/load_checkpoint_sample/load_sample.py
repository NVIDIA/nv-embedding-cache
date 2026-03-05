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
import argparse
from torch.distributed.fsdp import fully_shard
import gc
from enum import Enum

CHECKPOINT_DIR = "/tmp/checkpoint"

class EmbeddingType(Enum):
    NVE = 0
    TORCHREC = 1
    TORCH = 2

def get_embed_weight_name(saved_model_type: EmbeddingType):
    """ This function returns the name of the saved embedding weight tensor for the given embedding type. """
    if saved_model_type == EmbeddingType.TORCHREC:
        return "sparse.embed.sparse.embed.embeddings.embed_table.weight"
    else:
        return "sparse.embed.weight"

class ToyModelDense(nn.Module):
    def __init__(self, embedding_dim: int, device: torch.device):
        super(ToyModelDense, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim, device=device)

    def forward(self, x):
        return self.linear(x)

class ToyModelSparse(nn.Module):   
    def __init__(self, num_embeddings, embedding_dim, embedding_type: EmbeddingType, device: torch.device):
        super(ToyModelSparse, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        
        if embedding_type == EmbeddingType.NVE:
            import pynve.nve as nve
            import pynve.torch.nve_ps as nve_ps
            import pynve.torch.nve_layers as nve_layers
            from pynve.torch.nve_distributed import TorchDistEnv
            self.env = TorchDistEnv()
            self.memblock = nve.DistMemBlock(self.env, num_embeddings, embedding_dim, nve.DataType_t.Float32)
            cache_size = 1024
            self.cache_type = nve_layers.CacheType.LinearUVM
            self.remote_interface = nve_ps.NVEParameterServer(0, embedding_dim, torch.float32, None) if self.cache_type == nve_layers.CacheType.Hierarchical else None
            self.embed = nve_layers.NVEmbedding(num_embeddings, embedding_dim, torch.float32, cache_type=self.cache_type, memblock=self.memblock, gpu_cache_size=cache_size, remote_interface=self.remote_interface, optimize_for_training=False, device=device)
        elif embedding_type == EmbeddingType.TORCHREC:
            from torchrec.modules.embedding_modules import EmbeddingCollection
            from torchrec.modules.embedding_configs import EmbeddingConfig
            from torchrec.distributed.types import ShardingEnv
            from torchrec.distributed.model_parallel import DistributedModelParallel

            embedding_config = EmbeddingConfig(
                name="embed_table",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=["feature_0"],
            )

            embedding_collection = EmbeddingCollection(
                device=torch.device("meta"),
                tables=[embedding_config],
            )

            sharded_model = DistributedModelParallel(
                module=embedding_collection,
                device=device,
                env=ShardingEnv.from_process_group(dist.group.WORLD)
            )                
            self.embed = sharded_model
        elif embedding_type == EmbeddingType.TORCH:
            self.embed = nn.Embedding(num_embeddings, embedding_dim, device=device)
        else:
            raise ValueError(f"Invalid embedding type: {embedding_type}")   

    def forward(self, x):
        return self.embed(x)
    
    def cleanup(self):
        # NVE objects must be destroyed before dist.destroy_process_group(), because
        # CUDADistributedBuffer::~CUDADistributedBuffer() calls env->rank() and env->barrier().
        # Destruction order: embed first (holds shared_ptr<MemBlock>), then memblock, then env.
        #
        # Note: if autograd is enabled, output tensors may reatin refrence to NVE objects and require explcit removal before calling cleanup
        # use torch.no_grad() for NVE inference to avoid this.
        if self.embedding_type == EmbeddingType.NVE:
            del self.embed
            del self.memblock
            del self.env
        gc.collect()

class ToyModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, embedding_type: EmbeddingType, device: torch.device):
        super(ToyModel, self).__init__()
        self.sparse = ToyModelSparse(num_embeddings, embedding_dim, embedding_type, device)
        self.dense = ToyModelDense(embedding_dim, device)

    def forward(self, x):
        if self.sparse.embedding_type == EmbeddingType.TORCHREC:
            from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
            kjt = KeyedJaggedTensor(
                keys=["feature_0"],
                values=x,
                lengths=torch.tensor([x.numel()], device=x.device, dtype=torch.int64),
            )
            sparse_output = self.sparse(kjt)        
            sparse_output = sparse_output.wait()
            sparse_output = sparse_output["feature_0"].values()
        else:
            sparse_output = self.sparse(x)
        dense_output = self.dense(sparse_output)
        return dense_output

    def get_embed_weight(self):
        if self.sparse.embedding_type == EmbeddingType.NVE:
            return self.sparse.embed.weight
        else:
            state_dict = self.state_dict()
            return state_dict[get_embed_weight_name(self.sparse.embedding_type)]

    def print_weight(self):
        a = self.get_embed_weight()
        if hasattr(a, "local_shards"):
            local_data = a.local_shards()[0].tensor
            print(f"Rank {dist.get_rank()} Local Shard: {local_data}, shape: {local_data.shape}")
        else:            
            print(f"Rank {dist.get_rank()} Weight: {a}, shape: {a.shape}")

    def load_model(self, checkpoint_dir: str, saved_model_type: EmbeddingType, rank: int):
        if (self.sparse.embedding_type == EmbeddingType.NVE):
            # first we load the sparse model
            # since NVE shares the same weight tensor across all ranks, we need only to load the weight tensor for the rank 0
            if rank == 0:
                # for rank 0, we build a mapping from the name of the saved embedding weight tensor and to the nve weight
                partial_state_dict = { "embedding": { get_embed_weight_name(saved_model_type): self.get_embed_weight() } }
            else:
                # for other ranks, we don't need to load the sparse model
                partial_state_dict = { "embedding": {  } }
            dcp.load(partial_state_dict, checkpoint_id=checkpoint_dir)

            # loading the dense model is straightforward, we just need to remove the already loaded sparse model weight

            dense_state = {k: v for k, v in self.state_dict().items() if k != get_embed_weight_name(EmbeddingType.NVE)}
            dense_state_dict = { "embedding": dense_state }
            dcp.load(dense_state_dict, checkpoint_id=checkpoint_dir)
        else:
            state_ref = { "embedding": self.state_dict() }
            dcp.load(state_ref, checkpoint_id=checkpoint_dir)

            # important note regarding model cleanup, since NVE require explict clean up of it memory, we need to make sure all refrences to the NVE tensor are removed before calling dist.destroy_process_group
            # partial state dict holds a refrence to the NVE tensor, so we need to remove it before calling dist.destroy_process_group, since it will derefrenced when this scope ends we don't need to explicitly call del on it
            
    def cleanup(self):
        self.sparse.cleanup()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)  # Required by TorchRec

    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    torch.cuda.set_device(rank)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


num_embeddings = 16*1024*1024
embedding_dim = 4 # torchrec embedding dim must be divisible by 4
num_keys = 4


def save_model(rank: int, world_size: int, args):
    """
    This function performs a single backward pass and saves the model to a checkpoint, using dcp.
    """
    print(f"Running nve save example on rank {rank}.")
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # 1. create a model with the desired embedding implemenation
    if args.torchrec_checkpoint:
        model = ToyModel(num_embeddings, embedding_dim, EmbeddingType.TORCHREC, device)
    else:
        model = ToyModel(num_embeddings, embedding_dim, EmbeddingType.TORCH, device)
        model.sparse = fully_shard(model.sparse)
    model.dense = fully_shard(model.dense)

    # 2. perform a single backward pass
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    keys = torch.randint(0, num_embeddings, (num_keys,), device=device, dtype=torch.int64)
    output = model(keys)
    loss = loss_fn(output, torch.randn_like(output))
    loss.backward()
    optimizer.step()

    # 3. save the model to a checkpoint
    full_state_dict = model.state_dict()
    state_dict = { "embedding": full_state_dict }
    dcp.save(state_dict, checkpoint_id=args.checkpoint_dir)

    # 4. do cleanup
    model.cleanup()
    cleanup()

def load_model(rank: int, world_size: int, args):
    """
    This function loads the saved dcp checkpoint into an NVE model. across all ranks.
    """
    print(f"Running nve load example on rank {rank}.")
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    saved_model_type = EmbeddingType.TORCHREC if args.torchrec_checkpoint else EmbeddingType.TORCH
    with torch.no_grad():
        # 1. create a model with NVE embedding Note that all sparse model implementations share the same dense arch.
        model = ToyModel(num_embeddings, embedding_dim, EmbeddingType.NVE, device)

        # 2. load the model from the checkpoint
        model.load_model(args.checkpoint_dir, saved_model_type, rank)

        # 3. do cleanup
        dist.barrier() # first barrier to ensure all processes have loaded the model
        # before calling dist.destroy_process_group, we need to call model cleanup to ensure proper destruction of shared objects before destroying the process group
        model.cleanup()
        cleanup()
   
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["save", "load"], required=True)
    parser.add_argument("--torchrec_checkpoint", action="store_true", help="In save mode this flag indicates that the model uses torchrec embedding collection, In load mode this flag indicates that the checkpoint is saved a torchrec embedding collection model")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    world_size = args.world_size
    if world_size > torch.cuda.device_count():
        raise ValueError(f"World size {world_size} is greater than the number of available devices {torch.cuda.device_count()}")

    print(f"Running dcp load/save example on {world_size} devices.")
    if args.mode == "save":
        mp.spawn(
            save_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    elif args.mode == "load":
        mp.spawn(
            load_model,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )