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

import torch
import pynve.nve as nve
import torch.distributed as dist
from typing import override

class TorchDistEnv(nve.DistributedEnv):
    """Torch.distributed wrapper for NVE

    This class implements the DistributedEnv API using torch.distributed for the purposes of creating a distributed CUDA 
    buffer, sharded across multiple GPUs.

    Notes:
        1. The application is expected to initialize torch.distributed before instantiating this class
           (e.g. dist.init_process_group(backend="gloo"))
        2. The application is expected to remove all references to these objects and any NVE memblocks generated with them
           before calling dist.destroy_process_group()
        3. It's recommended to run Python applications using this, with torchrun/torchx
    """
    def __init__(self):
        nve.DistributedEnv.__init__(self)
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed isn't initialized!")
        if not torch.cuda.is_initialized():
            torch.cuda.init()

        if dist.get_backend() == "nccl":
            self.device = torch.device("cuda", self.local_device())
        else:
            self.device = torch.device("cpu")

        my_rank = torch.tensor([dist.get_rank()], device=self.device)
        all_ranks = [torch.zeros_like(my_rank) for _ in range(dist.get_world_size())]
        dist.all_gather(all_ranks, my_rank)
        self.single_host_ = (torch.unique(torch.cat(all_ranks)).size(0) == dist.get_world_size())

    @override
    def rank(self):
        return dist.get_rank()

    @override
    def world_size(self):
        return dist.get_world_size()

    @override
    def device_count(self):
        return torch.cuda.device_count()

    @override
    def local_device(self):
        return dist.get_node_local_rank() % self.device_count()

    @override
    def single_host(self):
        return self.single_host_

    @override
    def barrier(self):
        dist.barrier()

    @override
    def broadcast(self,
                  buffer_ptr,
                  size,
                  root: int = 0):
        data = torch.zeros(size, dtype=torch.int8)
        nve.raw_copy(data.data_ptr(), buffer_ptr, size)
        data = data.to(device=self.device)
        dist.broadcast(tensor=data, src=root)
        data = data.to(device=torch.device("cpu"))
        nve.raw_copy(buffer_ptr, data.data_ptr(), size)

    @override
    def all_gather(self,
                   send_buffer,
                   recv_buffer,
                   size):
        send_data = torch.zeros(size, dtype=torch.int8)
        nve.raw_copy(send_data.data_ptr(), send_buffer, size)
        send_data = send_data.to(device=self.device)
        recv_data = [torch.zeros_like(send_data) for _ in range(self.world_size())]
        dist.all_gather(recv_data, send_data)
        recv_data = torch.cat(recv_data)
        recv_data = recv_data.to(device=torch.device("cpu"))
        nve.raw_copy(recv_buffer, recv_data.data_ptr(), recv_data.size(0))
