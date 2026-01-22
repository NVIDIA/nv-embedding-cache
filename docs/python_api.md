# Python API

The NV Embedding Cache SDK provides PyTorch-like wrappers for easy integration with Python code.
The wrappers NVEmbedding and NVEmbeddingBag aim to mimic the PyTorch modules [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and [torch.nn.EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) respectively.

## Installation 

```console
     pip install .
```

## Quick Start
First, import the nve_layers and nve modules

```python
    import torch
    import pynve.nve as nve
    import pynve.torch.nve_layers as nve_layers
```

Create an Embedding with 10 million vectors stored the on CPU and 1 MB of cache on GPU 
```python
    weight_init = torch.randn(10*1024*1024, 1, dtype=torch.float32)
    embedding = nve_layers.NVEmbedding(num_embeddings=10*1024*1024, 
                                        embedding_size=1, 
                                        data_type=torch.float32, 
                                        cache_type=nve_layers.CacheType.LinearUVM, gpu_cache_size=1024*1024,
                                        weight_init=weight_init)
```

Perform a lookup
```python
    keys = torch.Tensor([1,2,1000]).to(torch.int64).to(device='cuda')
    emb = embedding(keys)
```
See [../samples/pytorch/simple_sample/](../samples/pytorch/simple_sample/) for full code.

## Overview
The nve_layers module creates a [torch.nn.module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) like interface that wraps around the python bindings to the NV Embedding Cache SDK C++ API.
The two operations [NVEmbedding](../python/pynve/torch/nve_layers.py#143) and [NVEmbeddingBag](../python/pynve/torch/nve_layers.py#175) provide several caching functionality:

1. NoCache - the entire storage of the embedding will be allocated on GPU
2. LinearUVM - storage of the embedding will be allocated in linear system memory with GPU cache
3. Hierarchical - storage is located in some non linear fashion with hierarchical cache structure, the user needs to supply a "parameter server" interface 

nve_ops module provides functional like interface and autograd functionality for training. It isn't generally advised to use this interface directly

nve_tensors module provides wrappers to Torch tensors whose storage is backed by a cache hierarchy.

## Using a Parameter Server
To use the parameter server API, users are expected to implement the Table Interface defined in [table.hpp](../include/nve/table.hpp) and provide a python binding to it. See [ps_binding.cpp](../samples/pytorch/ps_binding.cpp) for an example using pybind11.

Once a python binding to an implementation of nve::Table is available, creating an "Embedding Layer" with parameter server is simple.
```python
    embedding = nve_layers.NVEmbedding(num_embeddings=10*1024*1024, 
                                        embedding_size=1, 
                                        data_type=torch.float32, 
                                        cache_type=nve_layers.CacheType.Hierarchical, gpu_cache_size=1024*1024,
                                        remote_interface=binding_to_remote_ps)
```
## Multi Device
NV Embedding Cache takes PyTorch's approach to multi-device usage. Users should specify the device where the layer resides. Users should ensure all input tensors are on the same device as the layer (or accessible from the layer's device) when calling forward.
There is no need to set the device context as the underlying implementation handles this automatically.

## Inference Sample

A sample that shows usage of inferencing with NVEmbedding
usage:
``` inference_sample.py --batch-size 1024 --hotness 32 --num-iterations 1000 --cache-size 1024 --data-type float32```

this will run an inference for a 1000 iterations each of 1024 bags of 32 keys using linearUVM with a cache of size 1024 bytes.
See [inference_sample.py](../samples/pytorch/inference_sample/inference_sample.py)





