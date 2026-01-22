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
import gc
import subprocess
import argparse
import torch
import torch.distributed as dist
import pynve.nve as nve
from pynve.torch.nve_distributed import TorchDistEnv
import pytest
import platform

def kernel_doesnt_supports_pidfd() -> bool:
    ver = platform.release()
    dot_pos = ver.find('.')
    major = int(ver[:dot_pos])
    ver = ver[dot_pos+1:]
    dot_pos = ver.find('.')
    minor = int(ver[:dot_pos])
    return [major, minor] < [5,6]

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo", help="Torch.distributed backend", type=str)
    parser.add_argument("--data_type", default='float32' ,help="Data type (float32, float16)")
    parser.add_argument("--num_table_rows", "-nr", default=1024*1024 ,help="Number of rows in the embedding table", type=int)
    parser.add_argument("--embedding_dim", "-ed", default=128 ,help="Embedding dimension", type=int)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    return args

@pytest.mark.skipif(kernel_doesnt_supports_pidfd(), reason="Test disabled for unsupported kernel version (pidfd)")
def test_dist_nccl():
    cmd = [
        'torchrun',
        f'--nproc_per_node={torch.cuda.device_count()}',
        os.path.abspath(__file__),
        '--backend=nccl']
    subprocess.check_call(cmd)

@pytest.mark.skipif(kernel_doesnt_supports_pidfd(), reason="Test disabled for unsupported kernel version (pidfd)")
def test_dist_gloo():
    cmd = [
        'torchrun',
        '--nproc_per_node=8',
        os.path.abspath(__file__),
        '--backend=gloo']
    subprocess.check_call(cmd)

def main():
    args = parse_command_line()
    if (args.verbose):
        print(args)

    dist.init_process_group(backend=args.backend)
    env = TorchDistEnv()
    match args.data_type:
        case "float32":
            dtype = nve.DataType_t.Float32
        case "float16":
            dtype = nve.DataType_t.Float16
        case _:
            raise RuntimeError("Invalid data type")
    NVL = nve.DistMemBlock(env, args.embedding_dim, args.num_table_rows, dtype)

    # Must explicitly destroy NVL and env before the dist process group, since dist is needed during their destruction
    NVL = None
    env = None
    gc.collect()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
