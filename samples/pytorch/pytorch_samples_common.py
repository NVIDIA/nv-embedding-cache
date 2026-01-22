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
import sys
import fnmatch
import os
import subprocess
import re

def translateToPowerLaw(min, max, alpha, x):
    gamma = torch.tensor([1 - alpha], device=x.device)
    y = torch.pow(x * (torch.pow(max, gamma) - torch.pow(min, gamma)) + torch.pow(min, gamma),
          1.0 / gamma)
    b = (y >= max)
    y[b] = max-1
    return y

def gen_permute(max, device=torch.device('cuda')):
    g = torch.Generator(device=device)
    g.manual_seed(31337)
    perm = torch.randperm(max, generator=g, device=device, dtype=torch.int64)
    return perm

def PowerLaw(min, max, alpha, N, device=torch.device('cuda'), permute=None):
    x = torch.rand(N, device=device, dtype=torch.float64)
    y = translateToPowerLaw(min, max, alpha, x).to(torch.int64)

    if permute != None:
        y = permute[y]

    return y

def gen_key(batch, hotness, alpha, N, device, permute=None):
    ret = PowerLaw(1, N, alpha, hotness*batch, device, permute)
    return ret

def gen_jagged_key(batch, hotness, alpha, num_table_rows, device, feature_name, permute=None):
    import torchrec
    key = gen_key(batch, hotness, alpha, num_table_rows, device, permute)
    lengths = torch.tensor([hotness]*batch, dtype=torch.int64, device=device)
    return torchrec.KeyedJaggedTensor(
        keys=[feature_name],
        values=key,
        lengths=lengths,
    )

def print_mem_stats():
    import psutil

    # Memory currently allocated by tensors
    # Get the properties of the current CUDA device
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # Total memory in bytes
    total_memory = props.total_memory
    allocated_memory = torch.cuda.memory_allocated(device)

    # Memory currently cached by the allocator
    cached_memory = torch.cuda.memory_reserved(device)

    # Available memory
    available_memory = total_memory - (allocated_memory + cached_memory)

    print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Cached memory: {cached_memory / (1024 ** 3):.2f} GB")
    print(f"Available memory: {available_memory / (1024 ** 3):.2f} GB")

    # Get the virtual memory details
    virtual_memory = psutil.virtual_memory()

    # Total memory in bytes
    sys_total_memory = virtual_memory.total

    # Available memory in bytes
    sys_available_memory = virtual_memory.available

    print(f"Sys Total system memory: {sys_total_memory / (1024 ** 3):.2f} GB")
    print(f"Sys Available system memory: {sys_available_memory / (1024 ** 3):.2f} GB")


def data_type_to_torch_dtype(data_type: str):
    if data_type == "float32":
        return torch.float32
    elif data_type == "float16":
        return torch.float16
    else:
        raise ValueError(f"Invalid data type: {data_type}")

def data_type_to_nve_data_type(data_type: str):
    if data_type == "float32":
        return nve.DataType_t.Float32
    elif data_type == "float16":
        return nve.DataType_t.Float16
    else:
        raise ValueError(f"Invalid data type: {data_type}")

def torch_data_type_to_nve_data_type(data_type: torch.dtype):
    if data_type == torch.float32:
        return nve.DataType_t.Float32
    elif data_type == torch.float16:
        return nve.DataType_t.Float16
    else:
        raise ValueError(f"Invalid data type: {data_type}")

def FindFirstPath(file, root_dir=None):
    if root_dir is None:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        git_rev_dir = str(subprocess.run(['git', '-C', file_dir, 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout)
        root_dir = git_rev_dir.replace('b\'','').replace('\\n\'','')

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, file):
            return dirpath
    return None

def get_remote_interface(path_to_remote_table: str, num_embeddings: int, embedding_dim: int, data_type: torch.dtype):
    sys.path.append(path_to_remote_table)
    try:
        import sample_remote
        remote_interface = sample_remote.RemoteTable(num_embeddings, embedding_dim, torch_data_type_to_nve_data_type(data_type))
    except ModuleNotFoundError as e:
        print("Failed to import sample_remote - Searching for plugin")

        so_name = 'sample_remote.cpython-*.so'
        dir_path = FindFirstPath(so_name)
        if dir_path is not None:
                print("Defaulting to {}".format(dir_path))
                sys.path.append(dir_path)
                import sample_remote
                remote_interface = sample_remote.RemoteTable(num_embeddings, embedding_dim, torch_data_type_to_nve_data_type(data_type))
    return remote_interface

def calc_stats(values):
    """Calculate mean, min, max, and count statistics for a list of values."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0
    return sum(values) / len(values), min(values), max(values), len(values)

def parse_hitrates_from_log(log_content):
    """Parse hitrate statistics from log content
    
    Args:
        log_content: Log content string to parse
        
    Returns:
        Tuple of (gpu_stats, host_stats, remote_stats) where each stats is a tuple of
        (mean, min, max, count)
    """
    hitrate_pattern = re.compile(r'\[NVE\]\[P\] Hit rates: ([\d.]+), ([\d.]+), ([\d.]+)')
    matches = hitrate_pattern.findall(log_content)
    hitrates_gpu, hitrates_host, hitrates_remote = [], [], []
    for gpu, host, remote in matches:
        hitrates_gpu.append(float(gpu))
        hitrates_host.append(float(host))
        hitrates_remote.append(float(remote))
    
    gpu_stats = calc_stats(hitrates_gpu)
    host_stats = calc_stats(hitrates_host)
    remote_stats = calc_stats(hitrates_remote)
    
    return gpu_stats, host_stats, remote_stats