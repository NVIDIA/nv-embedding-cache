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
import subprocess
from conftest import requires_nvhm
import pytest

@pytest.mark.parametrize(
    "mode, extra_params",
    [
        pytest.param('torch', [], id='torch'),
        pytest.param('torch_cpu', [], id='torch_cpu'),
        pytest.param('torchrec', [], id='torchrec'),
        pytest.param('nv_linear', [], id='nv_linear'),
        pytest.param('nv_hierarchical', ['-pc'], id='nv_hierarchical'),
        pytest.param('nv_gpu', [], id='nv_gpu'),
    ])
def test_single_gpu_bench(mode, extra_params):
    base_params = [ '--batch', '128',
                    '--hotness','128',
                    '-nr', '1000000',
                    '-ns', '100',
                    '--load_factor', '0.1',
                    '-nw', '100',
                    '-ed', '128',
                    '-a', '1.05', ]
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(root_path, 'benchmarks', 'single_gpu_bench.py')
    cmd = ['python', sample_path, '--mode', mode] + base_params + extra_params
    print(f'\nRunning: {cmd}')
    subprocess.check_call(cmd)

@pytest.mark.parametrize(
    "runner, mode",
    [
        pytest.param(['mpirun', '--allow-run-as-root', '-n', '1', 'python'] , 'nve', id='mpirun_nve'),
        pytest.param(['torchrun', '--nproc_per_node=1'] , 'nve', id='torchrun_nve'),
        pytest.param(['torchx', 'run', '-s', 'local_cwd', 'dist.ddp', '-j1x1', '--script'], 'nve', id='torchx_nve'),
        pytest.param(['torchrun', '--nproc_per_node=1'] , 'torchrec', id='torchrun_torchrec'),
        pytest.param(['torchx', 'run', '-s', 'local_cwd', 'dist.ddp', '-j1x1', '--script'] , 'torchrec', id='torchx_torchrec'),
    ])
def test_multi_gpu_bench(runner, mode):
    base_params = [ '--batch', '128',
                    '--hotness','128',
                    '-nr', '1000000',
                    '-ns', '100',
                    '--load_factor', '0.1',
                    '-nw', '100',
                    '-ed', '128',
                    '-a', '1.05', ]
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(root_path, 'benchmarks', 'multi_gpu_bench.py')
    cmd = runner + [sample_path] + (['--'] if runner[0] == 'torchx' else []) +['--mode', mode] + base_params
    print(f'\nRunning: {cmd}')
    subprocess.check_call(cmd)
