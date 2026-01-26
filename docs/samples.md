# Embedding SDK Samples

Samples are located in [../samples/](../samples/) and are listed below with a short description.

## Basic samples

These simple samples are meant to show basic API usage, and ignore any efficiency considerations.

|Name|Location|Description|
|-|-|-|
|C++ simple sample|[../samples/simple_cpp/](../samples/simple_cpp/)|Basic API usage in C++ (single layer lookup)|
|Embedding cache simple_sample|[../samples/ecache/simple_sample/](../samples/ecache/simple_sample/)|Basic API usage for the GPU embedding cache with mock system memory cache and parameter server|
|PyTorch API|[../samples/pytorch/simple_sample/](../samples/pytorch/simple_sample/)|Simple pytorch usage|
|Import Dynamic Embedding (C++)|[../samples/import_sample](../samples/import_sample)|Import embeddings trained with the Dynamic Embedding SDK (see: [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples))|
|Import Dynamic Embedding (Py)|[../samples/pytorch/simple_sample/](../samples/pytorch/simple_sample/)|The [simple_hierarchical_embedding](../samples/pytorch/simple_sample/simple_hierarchical_embedding.py) sample demonstrates importing of embeddings trained with the Dynamic Embedding SDK (see: [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples))|
||[]()||

## Advanced Samples

These samples demonstrate more complex usages.

|Name|Location|Description|
|-|-|-|
|Layer sample|[../samples/layer_sample/](../samples/layer_sample/)|Layer inference (lookup)|
|PyTorch Inference|[../samples/pytorch/inference_sample/](../samples/pytorch/inference_sample/)|Multi threaded inference with parallel updated|
||[]()||
