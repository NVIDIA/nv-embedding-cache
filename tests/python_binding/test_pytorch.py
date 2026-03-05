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

import sys
import os
import argparse
# Workaround to import the python sample
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{dir_path}/../../samples/pytorch")
import pytorch_samples_common as common
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_serialization as nve_serialization
import pynve.nve as nve
import pynve.torch.nve_ps as nve_ps
import nvtx
import threading
import torch
from conftest import requires_nvhm

def build_key(batch, hotness, alpha, N):
    ret = common.PowerLaw(1, N, alpha, hotness*batch)
    return ret

def loop(layer, key_set, offsets, results, target, *, inference_only: bool = False, use_multi_stream: bool = False):
    #note currently works with optimizers who have no state you can configure parameters per layer
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()
    for keys in key_set:
        if use_multi_stream:
            stream = torch.cuda.Stream()
            torch.cuda.set_stream(stream)

        if offsets != None:
            output = layer(keys, offsets)
        else:
            output = layer(keys)
        if not inference_only:
            l1 = loss(output, target)
            l1.backward()
            optimizer.step()
            optimizer.zero_grad()
        if results != None:
            results.append(output)

def functional(quiet : bool, data_type : torch.dtype, device : torch.device = torch.device("cuda")):
    num_embeddings = 10000
    embed_size = 2
    cache_size = 128*1024*1024
    # create a regular embedding layer for reference
    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, dtype=data_type, sparse=True, device=device)

    # create the NVHM PS
    nvhm_ps_init = nve_ps.SimpleInitializer(num_embeddings, embed_size, data_type, emb_layer.weight)
    nvhm_ps = nve_ps.NVEParameterServer(
            0, # Setting num_embeddings as 0 to disable eviction policy
            embed_size,
            data_type,
            nvhm_ps_init
        )

    # create nv emb layer wrapping the nvHashMap as PS
    nv_nvhm_ps_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, remote_interface=nvhm_ps, device=device)

    # create a uvm backed nv emb layer
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, remote_interface=None, weight_init=emb_layer.weight, device=device)

    # create nv emb gpu layer
    nv_gpu_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.NoCache, remote_interface=None, weight_init=emb_layer.weight, device=device)

    # from here the interface should be same for all layers as demonstrated by the loop function

    num_steps = 5
    nvhm_ps_res = []
    torch_res = []
    uvm_res = []
    gpu_res = []
    
    keys = torch.tensor([5, 4, 3, 100, 1024, 6], dtype=torch.int64, device=device)
    target = torch.randn(torch.numel(keys), embed_size, dtype=data_type, device=device)
    loop(emb_layer, [keys], None, torch_res, target)
    loop(nv_nvhm_ps_emb_layer, [keys], None, nvhm_ps_res, target)
    loop(nv_uvm_emb_layer, [keys], None, uvm_res, target)
    loop(nv_gpu_emb_layer, [keys], None, gpu_res, target)


    with nvtx.annotate("torch emb", color="green"):
        loop(emb_layer, [keys]*num_steps, None, torch_res, target)
    with nvtx.annotate("nvhm ps emb", color="green"):
        loop(nv_nvhm_ps_emb_layer, [keys]*num_steps, None, nvhm_ps_res, target)
    with nvtx.annotate("uvm emb", color="green"):
        loop(nv_uvm_emb_layer, [keys]*num_steps, None, uvm_res, target)
    with nvtx.annotate("gpu emb", color="green"):
        loop(nv_gpu_emb_layer, [keys]*num_steps, None, gpu_res, target)
    if not quiet:
        for nvhm_ps_output, uvm_output, torch_output, gpu_output in zip(nvhm_ps_res, uvm_res, torch_res, gpu_res):
            print(f"torch_output={torch_output}")
            print(f"nvhm_ps_output={nvhm_ps_output}")
            print(f"uvm_output={uvm_output}")
            print(f"gpu_output={gpu_output}")
            print("==========================================")
    return nvhm_ps_res, uvm_res, torch_res, gpu_res

def functional_bag(quiet : bool, data_type : torch.dtype, device : torch.device = torch.device("cuda")):
    num_embeddings = 10000
    embed_size = 2
    cache_size = 128*1024*1024
    # create a regular embedding layer for reference
    emb_layer = torch.nn.EmbeddingBag(num_embeddings, embed_size, sparse=True, mode='sum', dtype=data_type, include_last_offset=True, device=device)

    # TRTREC-47 pooling is not enabled yet for PS
    # See functional() for how to test with PS based on NVHM
    
    # create a uvm backed nv emb layer
    nv_uvm_emb_layer = nve_layers.NVEmbeddingBag(num_embeddings, embed_size, data_type, nve_layers.CacheType.LinearUVM, mode='sum', gpu_cache_size=cache_size, remote_interface=None, weight_init=emb_layer.weight, device=device)

    # create nv emb layer wrapping the PS
    nv_gpu_emb_layer = nve_layers.NVEmbeddingBag(num_embeddings, embed_size, data_type, nve_layers.CacheType.NoCache, mode='sum', remote_interface=None, weight_init=emb_layer.weight, device=device)

    # from here the interface should be same for all layers as demonstrated by the loop function
    num_steps = 5
    # TRTREC-47 pooling is not enabled yet for PS
    ps_res = []
    torch_res = []
    uvm_res = []
    gpu_res = []
    num_keys = 6
    target = torch.randn(2, embed_size, dtype=data_type, device=device)
    keys = torch.tensor([5, 4, 3, 100, 1024, 6], dtype=torch.int64, device=device)
    offsets = torch.tensor([0,3,6], dtype=torch.int64, device=device)
    loop(nv_uvm_emb_layer, [keys], offsets, uvm_res, target)
    loop(emb_layer, [keys], offsets, torch_res, target)
    loop(nv_gpu_emb_layer, [keys], offsets, gpu_res, target)
    # TRTREC-47 pooling is not enabled yet for PS
    #loop(nv_ps_emb_layer, [keys], offsets, ps_res, target)

    with nvtx.annotate("torch emb", color="green"):
        loop(emb_layer, [keys]*num_steps, offsets, torch_res, target)
    # TRTREC-47 pooling is not enabled yet for PS
    #with nvtx.annotate("ps emb", color="green"):    
    #    loop(nv_ps_emb_layer, [keys]*num_steps, offsets, ps_res, target)
    with nvtx.annotate("uvm emb", color="green"):
        loop(nv_uvm_emb_layer, [keys]*num_steps, offsets, uvm_res, target)
    with nvtx.annotate("gpu emb", color="green"):
        loop(nv_gpu_emb_layer, [keys]*num_steps, offsets, gpu_res, target)
    if not quiet:
        # TRTREC-47 pooling is not enabled yet for PS
        #for ps_output, uvm_output, torch_output, gpu_output in zip(ps_res, uvm_res, torch_res, gpu_res):
        for uvm_output, torch_output, gpu_output in zip(uvm_res, torch_res, gpu_res):
            print(f"torch_output={torch_output}")
            # TRTREC-47 pooling is not enabled yet for PS
            #print(f"ps_output={ps_output}")
            print(f"uvm_output={uvm_output}")
            print(f"gpu_output={gpu_output}")
            print("==========================================")
    return ps_res, uvm_res, torch_res, gpu_res

def benchmark(data_type):    
    num_embeddings = 1000000
    embed_size = 2
    cache_size = 128*1024*1024
    
    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, dtype=data_type)
    
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, remote_interface=None, weight_init=emb_layer.weight)

    num_steps = 25
    batch = 1024
    hotness = 32
    num_keys = batch*hotness
    target = torch.randn(num_keys, embed_size).cuda()
    print("starting to generate keys")
    key_set = [build_key(batch, hotness, 1.05, num_embeddings) for i in range(num_steps)]
    print("Done generating keys")
    uvm_res = []

    #warmup
    loop(nv_uvm_emb_layer, [key_set[0]], None, uvm_res, target)

    with nvtx.annotate("uvm emb", color="green"):
        loop(nv_uvm_emb_layer, key_set, None, uvm_res, target)


def main(args):    
    class RunMode:
        FunctionalBag="FunctionalBag"
        Functional="Functional"
        Benchmark="Benchmark"

    class DataType:
        float32="float32"
        float16="float16"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=RunMode.Functional, choices=[RunMode.Functional, RunMode.FunctionalBag, RunMode.Benchmark],help=f"Mode of script can be either [Default={RunMode.Functional}/{RunMode.FunctionalBag}/{RunMode.Benchmark}] ")
    parser.add_argument("--data_type", default=DataType.float32, choices=[DataType.float32, DataType.float16],help=f"Table data type can be either [Default={DataType.float32}/{DataType.float16}] ")

    # TODO: change type to the one specified from command line
    args = parser.parse_args(args)
    torch_data_type = common.data_type_to_torch_dtype(args.data_type)
    if args.mode == RunMode.Functional:
        functional(False, torch_data_type)
    elif args.mode == RunMode.Benchmark:
        benchmark(torch_data_type)
    elif args.mode == RunMode.FunctionalBag:
        functional_bag(False, torch_data_type)

############################################### TESTS ###############################################

def test_pytorch_init_linear():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 128*1024*1024
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=torch.zeros(num_embeddings, embed_size, dtype=torch.float32), optimize_for_training=False)

def test_pytorch_init_gpu():
    num_embeddings = 100000
    embed_size = 2
    gpu_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.NoCache,  weight_init=torch.zeros(num_embeddings, embed_size, dtype=torch.float32), optimize_for_training=False)

@requires_nvhm
def test_pytorch_init_hierarchical_nvhm():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 128*1024*1024

    # create the NVHM PS
    nvhm_ps = nve_ps.NVEParameterServer(
            0, # Setting num_embeddings as 0 to disable eviction policy
            embed_size,
            torch.float32,
        )
    hierarchical_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, host_cache_size=cache_size, remote_interface=nvhm_ps, optimize_for_training=False)

def test_pytorch_init_hierarchical_redis():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 128*1024*1024

    # create the MW PS
    mw_ps = nve_ps.NVEParameterServer(
            num_embeddings,
            embed_size,
            torch.float32,
            ps_type = nve.Redis,
            extra_params = {"plugin": {"address": "localhost:7001"}}
        )
    hierarchical_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, host_cache_size=cache_size, remote_interface=mw_ps, optimize_for_training=False)

def test_pytorch_init_memblock():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 128*1024*1024
    memblock = nve.NVLMemBlock(num_embeddings, embed_size, common.torch_data_type_to_nve_data_type(torch.float32), [0])
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, memblock=memblock, gpu_cache_size=cache_size, weight_init=torch.zeros(num_embeddings, embed_size, dtype=torch.float32), optimize_for_training=False)

@requires_nvhm
def test_pytorch_concat():
    nvhm_ps_list, uvm_list, torch_list, gpu_list = functional(True, torch.float16)
    for nvhm_ps_output, uvm_output, torch_output, gpu_output in zip(nvhm_ps_list, uvm_list, torch_list, gpu_list):
        assert all(torch.isclose(nvhm_ps_output, torch_output).tolist())
        assert all(torch.isclose(uvm_output, torch_output).tolist())
        assert all(torch.isclose(gpu_output, torch_output).tolist())

def test_pytorch_linear_gather_flows():
    num_embeddings = 10000000
    embed_size = 2
    data_type = torch.float32
    cache_size = 128*1024*1024
    config = {"kernel_mode": 3}
    alpha = 1.05
    num_keys = 100000

    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, dtype=data_type, device=torch.device("cuda"))
    nve_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=emb_layer.weight , optimize_for_training=False, config=config)

    for i in range(100):
        print(f"test {i}")
        keys = common.gen_key(num_keys, 1, alpha, num_embeddings, torch.device("cuda"))
        out = nve_emb_layer(keys)
        out_ref = emb_layer(keys)
        assert torch.equal(out, out_ref)

def test_pytorch_managed_memblock():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 128*1024
    memblock = nve.ManagedMemBlock(num_embeddings, embed_size, common.torch_data_type_to_nve_data_type(torch.float32), [0])
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, memblock=memblock, gpu_cache_size=cache_size, weight_init=torch.zeros(num_embeddings, embed_size, dtype=torch.float32), optimize_for_training=False)
    keys = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64, device=torch.device("cuda"))
    out = nv_uvm_emb_layer(keys)
    assert torch.equal(out, torch.zeros(10, 2, device=torch.device("cuda")))
    
@requires_nvhm
def test_pytorch_load_from_binary_file():
    import numpy as np

    num_embeddings = 10000000
    embed_size = 2
    cache_size = 1024*1024
    data_type = torch.float32
    device = torch.device("cuda")

    nvhm_ps = nve_ps.NVEParameterServer(
            0, # Setting num_embeddings as 0 to disable eviction policy
            embed_size,
            data_type,
            None
        )

    keys_path = "/tmp/keys.dyn"
    values_path = "/tmp/values.dyn"
    
    keys_array = np.array([10, 5, 700, 1050], dtype=np.int64)
    values_array = np.array([[10.0, 10.5], [5.0, 5.5], [700.0, 700.5], [1050.0, 1050.5]], dtype=np.float32)

    
    with open(keys_path, "wb") as f:
        f.write(keys_array.tobytes())
            
    with open(values_path, "wb") as f:
        f.write(values_array.tobytes())

    nvhm_ps.load_from_file(keys_path, values_path)

    nv_nvhm_ps_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, remote_interface=nvhm_ps, device=device)

    out = nv_nvhm_ps_emb_layer(torch.tensor([10, 5, 700, 1050], dtype=torch.int64, device=device))
    assert torch.equal(out, torch.tensor([[10.0, 10.5], [5.0, 5.5], [700.0, 700.5], [1050.0, 1050.5]], dtype=torch.float32, device=device))

def test_pytorch_multi_layer():
    num_embeddings = 1000
    embed_size = 4
    cache_size = 128*1024
    ref_embedding_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, dtype=torch.float32, device=torch.device("cuda"))
    ref_embedding_layer_2 = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, dtype=torch.float32, device=torch.device("cuda"))
    nve_emb_layer_1 = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=ref_embedding_layer.weight, optimize_for_training=False)
    nve_emb_layer_2 = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=ref_embedding_layer_2.weight, optimize_for_training=False)
    keys = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64, device=torch.device("cuda"))
    out_1 = nve_emb_layer_1(keys)
    out_2 = nve_emb_layer_2(keys)
    out_ref_1 = ref_embedding_layer(keys)
    out_ref_2 = ref_embedding_layer_2(keys)
    assert torch.equal(out_1, out_ref_1)
    assert torch.equal(out_2, out_ref_2)

@requires_nvhm
def test_pytorch_load_from_numpy_file():
    import numpy as np

    cache_size = 1024*1024
    embed_size = 2
    num_embeddings = 100000
    data_type = torch.float32
    device = torch.device("cuda")

    nvhm_ps = nve_ps.NVEParameterServer(
            0, # Setting num_embeddings as 0 to disable eviction policy
            embed_size,
            data_type,
            None
        )

    keys_path = "/tmp/keys.npy"
    values_path = "/tmp/values.npy"
    
    keys_array = np.array([10, 5, 700, 1050], dtype=np.int64)
    values_array = np.array([[10.0, 10.5], [5.0, 5.5], [700.0, 700.5], [1050.0, 1050.5]], dtype=np.float32)

    with open(keys_path, "wb") as f:
        np.save(f, keys_array)
        
    with open(values_path, "wb") as f:
        np.save(f, values_array)

    nvhm_ps.load_from_file(keys_path, values_path)

    nv_nvhm_ps_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, remote_interface=nvhm_ps, device=device)

    out = nv_nvhm_ps_emb_layer(torch.tensor([10, 5, 700, 1050], dtype=torch.int64, device=device))
    assert torch.equal(out, torch.tensor([[10.0, 10.5], [5.0, 5.5], [700.0, 700.5], [1050.0, 1050.5]], dtype=torch.float32, device=device))

def test_pytorch_update():
    weight_init = torch.zeros(10000, 2, dtype=torch.float32)
    layer = nve_layers.NVEmbedding(10000, 2, torch.float32, nve_layers.CacheType.LinearUVM, gpu_cache_size=1*1024, weight_init=weight_init, optimize_for_training=False)
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device='cuda')
    updates = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32, device='cuda')

    layer.update(keys, updates)
    out = layer(keys)
    assert torch.equal(out, updates.reshape(5, 2).to(device='cuda'))

@requires_nvhm
def test_pytorch_erase():
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device='cuda')
    values0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32, device='cuda')
    values1 = torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0], dtype=torch.float32, device='cuda')

    num_embeddings = 2**20
    embed_size = 2
    data_type = torch.float32
    cache_size = 2**16
    # create the NVHM PS
    nvhm_ps_init = nve_ps.SimpleInitializer(num_embeddings, embed_size, data_type, torch.zeros(num_embeddings, embed_size, dtype=torch.float32))
    nvhm_ps = nve_ps.NVEParameterServer(
            num_embeddings,
            embed_size,
            data_type,
            nvhm_ps_init
        )

    # create nv emb layer wrapping the nvHashMap as PS
    layer = nve_layers.NVEmbedding(num_embeddings, embed_size, data_type, nve_layers.CacheType.Hierarchical, gpu_cache_size=cache_size, remote_interface=nvhm_ps, device=torch.device('cuda'))

    # Inserting different values on tables
    # This is an intentional inconsistency for testing purposes.
    layer.insert(keys, values0, table_id=0)
    layer.insert(keys, values1, table_id=1)

    # After this insert we expect to get values0 (all hits in table 0)
    out = layer(keys)
    assert torch.equal(out, values0.reshape(5, 2).to(device='cuda'))

    # Now erase from table 0
    layer.erase(keys, table_id=0)

    # After erase, looking up the keys should return the values from table 1 (after cache misses in the table 0)
    out_after_erase = layer(keys)
    assert torch.equal(out_after_erase, values1.reshape(5, 2).to(device='cuda'))

def test_pytorch_pooling():
    #right now we are testing unweighted sum. If other pooling types are tested,
    #the tolerance might be too low
    tolerance = 1e-10
    
    ps_list, uvm_list, torch_list, gpu_list = functional_bag(True, torch.float32)
    # ps_list is empty until pooling is enabled for hierarchical, ignore it
    # once enabled, ps_output should be compared to the ref too
    for uvm_output, torch_output, gpu_output in zip(uvm_list, torch_list, gpu_list):
        assert all(torch.isclose(gpu_output, torch_output, rtol=tolerance).tolist())
        assert all(torch.isclose(uvm_output, torch_output, rtol=tolerance).tolist())
        assert all(torch.isclose(uvm_output, gpu_output, rtol=tolerance).tolist())

    ps_list, uvm_list, torch_list, gpu_list = functional_bag(True, torch.float16)
    # ps_list is empty until pooling is enabled for hierarchical, ignore it
    # once enabled, ps_output should be compared to the ref too
    for uvm_output, torch_output, gpu_output in zip(uvm_list, torch_list, gpu_list):
        assert all(torch.isclose(gpu_output, torch_output, rtol=tolerance).tolist())
        assert all(torch.isclose(uvm_output, torch_output, rtol=tolerance).tolist())
        assert all(torch.isclose(uvm_output, gpu_output, rtol=tolerance).tolist())

def test_pytorch_multi_stream():
    num_embeddings = 10000
    embed_size = 2
    cache_size = 128*1024*1024
    # create a regular embedding layer for reference
    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True)

    # create a uvm backed nv emb layer
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=emb_layer.weight, optimize_for_training=False)
    # from here the interface should be same for all layers as demonstrated by the loop function

    emb_layer.to(device="cuda")
    torch_res = []
    uvm_res = []
    num_keys = 6
    target = torch.randn(num_keys, embed_size).cuda()
    keys_1 = torch.tensor([5, 4, 3, 100, 1024, 6], dtype=torch.int64).cuda()
    keys_2 = torch.tensor([345, 7, 19, 45, 2837, 34], dtype=torch.int64).cuda()
    keys_3 = torch.tensor([134, 295, 20, 1, 76, 207], dtype=torch.int64).cuda()
    loop(emb_layer, [keys_1, keys_2, keys_3], None, torch_res, target, inference_only=True, use_multi_stream=True)
    loop(nv_uvm_emb_layer, [keys_1, keys_2, keys_3], None, uvm_res, target, inference_only=True, use_multi_stream=True)
    for nv_output, torch_output in zip(uvm_res, torch_res):
        assert torch.equal(nv_output, torch_output)

def test_pytorch_update_non_default_device():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device = torch.device("cuda:1")
    weight_init = torch.zeros(7, 2, dtype=torch.float32, device=device)
    layer = nve_layers.NVEmbedding(7, 2, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=1*1024, weight_init=weight_init, optimize_for_training=False, device=device)
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=device)
    updates = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32, device=device)
    layer.update(keys, updates)
    out = layer(keys)
    assert torch.equal(out, updates.reshape(5, 2))

def test_pytorch_training_loop_non_default_device():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device = torch.device("cuda:1")
    nvhm_ps_list, uvm_list, torch_list, gpu_list = functional(True, torch.float32, device)
    for nvhm_ps_output, uvm_output, torch_output, gpu_output in zip(nvhm_ps_list, uvm_list, torch_list, gpu_list):
        assert all(torch.isclose(nvhm_ps_output, torch_output).tolist())
        assert all(torch.isclose(uvm_output, torch_output).tolist())
        assert all(torch.isclose(gpu_output, torch_output).tolist())
    
def test_pytorch_training_pooling_non_default_device():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device = torch.device("cuda:1")
    ps_list, uvm_list, torch_list, gpu_list = functional_bag(True, torch.float32, device)
    for ps_output, uvm_output, torch_output, gpu_output in zip(ps_list, uvm_list, torch_list, gpu_list):
        #do not test hierarchical output, not implemented yet
        assert torch.equal(ps_output, torch_output)
        assert torch.equal(uvm_output, torch_output)
        assert torch.equal(gpu_output, torch_output)

def test_pytorch_non_default_device():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device = torch.device("cuda:1")
    layer = nve_layers.NVEmbedding(10000, 2, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=1*1024, weight_init=torch.zeros(10000, 2, dtype=torch.float32), optimize_for_training=False, device=device)
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=device)
    out = layer(keys)
    assert torch.equal(out, torch.zeros(5, 2, device=device))

def test_pytorch_pooling_non_default_device():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device = torch.device("cuda:1")
    layer = nve_layers.NVEmbeddingBag(10000, 2, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, mode='sum', gpu_cache_size=1*1024, weight_init=torch.zeros(10000, 2, dtype=torch.float32), optimize_for_training=False, device=device)
    keys = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 6], dtype=torch.int64, device=device)
    out = layer(keys, offsets)
    assert torch.equal(out, torch.zeros(2, 2, device=device))

def test_pytorch_multi_gpu():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device_1 = torch.device("cuda:0")
    device_2 = torch.device("cuda:1")
    num_embeddings = 100000
    embed_size = 2
    cache_size = 1024*1024
    # create a regular embedding layer for reference, we place it on cpu to avoid device mismatch
    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, device='cpu')
    nv_uvm_emb_layer_device_1 = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=emb_layer.weight, optimize_for_training=False, device=device_1)
    nv_uvm_emb_layer_device_2 = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=emb_layer.weight, optimize_for_training=False, device=device_2)

    def thread_func(layer, num_iterations, num_keys, results):
        torch.cuda.set_device(layer.device)
        for i in range(num_iterations):
            keys = torch.randint(0, num_embeddings, (num_keys,), device=layer.device, dtype=torch.int64)
            nve_out = layer(keys)
            emb_out = emb_layer(keys.to(device=emb_layer.weight.device)).to(device=layer.device)
            r = torch.equal(nve_out, emb_out)
            results.append(r)
        torch.cuda.current_stream().synchronize()

    num_iterations = 10
    num_keys = 100
    results_1 = []
    results_2 = []
    thread_1 = threading.Thread(target=thread_func, args=(nv_uvm_emb_layer_device_1, num_iterations, num_keys, results_1))
    thread_2 = threading.Thread(target=thread_func, args=(nv_uvm_emb_layer_device_2, num_iterations, num_keys, results_2))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    assert all(results_1)
    assert all(results_2)
    
def test_pytorch_inference_sample():
    sys.path.append(f"{dir_path}/../../samples/pytorch/inference_sample")
    import inference_sample
    inference_sample.main([])

def test_pytorch_simple_sample_linear():
    sys.path.append(f"{dir_path}/../../samples/pytorch/simple_sample")
    import simple_linear_embedding
    simple_linear_embedding.main()

def load_sample_parse_args(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchrec_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str)
    return parser.parse_args(args)

def test_pytorch_load_checkpoint_sample_torchrec():
    sys.path.append(f"{dir_path}/../../samples/pytorch/load_checkpoint_sample")
    import load_sample
    import argparse

    args = load_sample_parse_args(["--torchrec_checkpoint", "--checkpoint_dir", "/tmp/checkpoint"])
    load_sample.save_model(0, 1, args)
    load_sample.load_model(0, 1, args)

def test_pytorch_load_checkpoint_sample_torch():
    sys.path.append(f"{dir_path}/../../samples/pytorch/load_checkpoint_sample")
    import load_sample
    import argparse
    args = load_sample_parse_args(["--checkpoint_dir", "/tmp/checkpoint"])
    load_sample.save_model(0, 1, args)
    load_sample.load_model(0, 1, args)

@requires_nvhm
def test_pytorch_simple_sample_hierarchical():
    sys.path.append(f"{dir_path}/../../samples/pytorch/simple_sample")
    import simple_hierarchical_embedding
    simple_hierarchical_embedding.main()

def test_pytorch_multi_thread():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 1024*1024
    # create a regular embedding layer for reference, we place it on cpu to avoid device mismatch
    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, device="cuda")
    nv_uvm_emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=emb_layer.weight, optimize_for_training=False, device=torch.device("cuda"))
    num_iterations = 3
    num_keys = 5
    num_threads = 2

    def thread_func(layer, keys, target_, results, device = torch.device("cuda")):
        target = target_.clone().to(device=device)
        loop(layer, keys, None, results, target, inference_only=True, use_multi_stream=True)

    
    threads = []
    thread_results = []
    thread_keys_set = []
    thread_reference_results = []
    target = torch.randn(num_keys, embed_size, device=torch.device("cuda"))
    for i in range(num_threads):
        keys_set = []
        results = []
        for i in range(num_iterations):
            keys_set.append(torch.randint(0, num_embeddings, (num_keys,), device=torch.device("cuda"), dtype=torch.int64))
        threads.append(threading.Thread(target=thread_func, args=(nv_uvm_emb_layer, keys_set, target, results)))
        thread_results.append(results)
        thread_keys_set.append(keys_set)
    for i in range(num_threads):
        threads[i].start()
    for i in range(num_threads):
        threads[i].join()
    for i in range(num_threads):
        for j in range(num_iterations):
            emb_out = emb_layer(thread_keys_set[i][j].to(device=emb_layer.weight.device)).to(device=nv_uvm_emb_layer.device)
            assert torch.equal(thread_results[i][j], emb_out)

def test_pytorch_multi_gpu_memblock():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    device_1 = torch.device("cuda:0")
    device_2 = torch.device("cuda:1")
    num_embeddings = 10000000
    embed_size = 2
    cache_size = 1024*1024
    # create a regular embedding layer for reference, we place it on cpu to avoid device mismatch
    emb_layer = torch.nn.Embedding(num_embeddings, embed_size, sparse=True, device='cpu')
    memblock = nve.NVLMemBlock(num_embeddings, embed_size, common.torch_data_type_to_nve_data_type(torch.float32), [0, 1])
    nv_uvm_emb_layer_device_1 = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, memblock=memblock, gpu_cache_size=cache_size, weight_init=emb_layer.weight, optimize_for_training=False, device=device_1)
    nv_uvm_emb_layer_device_2 = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, memblock=memblock, gpu_cache_size=cache_size, optimize_for_training=False, device=device_2)
    def thread_func(layer, num_iterations, num_keys, results):
        torch.cuda.set_device(layer.device)
        for i in range(num_iterations):
            keys = torch.randint(0, num_embeddings, (num_keys,), device=layer.device, dtype=torch.int64)
            nve_out = layer(keys)
            emb_out = emb_layer(keys.to(device=emb_layer.weight.device)).to(device=layer.device)
            print(nve_out)
            r = torch.equal(nve_out, emb_out)
            results.append(r)
        torch.cuda.current_stream().synchronize()

    num_iterations = 5
    num_keys = 20
    results_1 = []
    results_2 = []
    thread_1 = threading.Thread(target=thread_func, args=(nv_uvm_emb_layer_device_1, num_iterations, num_keys, results_1))
    thread_2 = threading.Thread(target=thread_func, args=(nv_uvm_emb_layer_device_2, num_iterations, num_keys, results_2))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    assert all(results_1)
    assert all(results_2)


import tempfile

def compare_layer(emb_layer_1, emb_layer_2):
    same_cache = emb_layer_1.cache_type == emb_layer_2.cache_type
    same_gpu_cache_size = emb_layer_1.gpu_cache_size == emb_layer_2.gpu_cache_size
    same_optimize_for_training = emb_layer_1.optimize_for_training == emb_layer_2.optimize_for_training
    same_num_embeddings = emb_layer_1.num_embeddings == emb_layer_2.num_embeddings
    same_embedding_size = emb_layer_1.embedding_size == emb_layer_2.embedding_size
    same_weight = torch.equal(emb_layer_1.weight, emb_layer_2.weight)
    return same_cache and same_gpu_cache_size and same_optimize_for_training and same_num_embeddings and same_embedding_size and same_weight

def save_load_module(module):
    tf_torch = tempfile.TemporaryFile()
    tf_nve = tempfile.TemporaryFile()
    nve_serialization.save(module, tf_torch, tf_nve)
    tf_torch.seek(0)
    tf_nve.seek(0)
    return nve_serialization.load(tf_torch, tf_nve)

def test_save_load_embedding():
    import tempfile
    num_embeddings = 100000
    embed_size = 2
    cache_size = 1024*1024
    weight_init = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    
    emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=cache_size, weight_init=weight_init, optimize_for_training=False, device=torch.device("cuda"))
    emb_layer_2 = save_load_module(emb_layer)
    assert compare_layer(emb_layer, emb_layer_2)
    
    emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.NoCache, weight_init=weight_init, optimize_for_training=False, device=torch.device("cuda"))
    emb_layer_2 = save_load_module(emb_layer)
    assert compare_layer(emb_layer, emb_layer_2)

def test_save_load_embedding_bag():
    num_embeddings = 100000
    embed_size = 2
    cache_size = 1024*1024
    weight_init = torch.randn(num_embeddings, embed_size, dtype=torch.float32)
    
    emb_layer = nve_layers.NVEmbeddingBag(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, mode='sum', gpu_cache_size=cache_size, weight_init=weight_init, optimize_for_training=False, device=torch.device("cuda"))
    emb_layer_2 = save_load_module(emb_layer)
    assert compare_layer(emb_layer, emb_layer_2)
    
    emb_layer = nve_layers.NVEmbeddingBag(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.NoCache, mode='sum', weight_init=weight_init, optimize_for_training=False, device=torch.device("cuda"))
    emb_layer_2 = save_load_module(emb_layer)
    assert compare_layer(emb_layer, emb_layer_2)

class MyBlock(torch.nn.Module):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.seq = torch.nn.Sequential(
            nve_layers.NVEmbedding(1000, 2, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=1024*1024, weight_init=torch.randn(1000, 2, dtype=torch.float32), optimize_for_training=False, device=torch.device("cuda")),
            torch.nn.ReLU()
        )
        self.emb_layer = nve_layers.NVEmbeddingBag(100, 2, torch.float32, cache_type=nve_layers.CacheType.NoCache, mode='sum', weight_init=torch.randn(100, 2, dtype=torch.float32), optimize_for_training=False, device=torch.device("cuda"))

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb_layer = nve_layers.NVEmbedding(200, 2, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=1024*1024, weight_init=torch.randn(200, 2, dtype=torch.float32), optimize_for_training=False, device=torch.device("cuda"))
        self.block = MyBlock()


def test_load_save_nested_model():
    model = MyModel()
    model_2 = save_load_module(model)
    assert compare_layer(model.emb_layer, model_2.emb_layer)
    assert compare_layer(model.block.emb_layer, model_2.block.emb_layer)
    assert compare_layer(model.block.seq[0], model_2.block.seq[0])

@requires_nvhm
def test_pytorch_multi_thread_multi_device_training():
    if torch.cuda.device_count() < 2:
        print("Skipping multi-GPU test as only one GPU is available")
        return
    num_threads = torch.cuda.device_count()

    def thread_func(device):
        nvhm_ps_res, uvm_res, torch_res, gpu_res = functional(True, torch.float32, device)
        for nvhm_ps_output, uvm_output, torch_output, gpu_output in zip(nvhm_ps_res, uvm_res, torch_res, gpu_res):
            assert all(torch.isclose(nvhm_ps_output, torch_output).tolist())
            assert all(torch.isclose(uvm_output, torch_output).tolist())
            assert all(torch.isclose(gpu_output, torch_output).tolist())

    threads = []
    for i in range(num_threads):
        device = torch.device(f"cuda:{i}")
        threads.append(threading.Thread(target=thread_func, args=[device]))
    for i in range(num_threads):
        threads[i].start()
    for i in range(num_threads):
        threads[i].join()

# Simple test to check user block 
def test_pytorch_user_memblock():
    num_embeddings = 2**21
    embed_size = 2
    cache_size = 2**20
    linear_memblock = nve.LinearMemBlock(embed_size, num_embeddings, nve.DataType_t.Float32)
    user_memblock = nve.UserMemBlock(linear_memblock.get_handle())
    emb_layer = nve_layers.NVEmbedding(num_embeddings, embed_size, torch.float32, cache_type=nve_layers.CacheType.LinearUVM, memblock=user_memblock, gpu_cache_size=cache_size, weight_init=torch.ones(num_embeddings, embed_size, dtype=torch.float32), optimize_for_training=False)
    num_keys = 100
    keys = torch.randint(1, num_embeddings, (num_keys,), device="cuda", dtype=torch.int64)
    res = emb_layer(keys)
    torch.cuda.current_stream().synchronize()
    assert (res == 1.0).all(), "Output should be all ones"

if __name__ == "__main__":
    #main(sys.argv[1:])
    test_pytorch_load_from_binary_file()
