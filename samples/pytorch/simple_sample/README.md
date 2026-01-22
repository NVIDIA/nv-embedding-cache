# Simple Samples
These samples aim to introduce the basics of the NV Embedding PyTorch APIs. 

## Simple Linear Embedding
This sample creates a linear embedding with a GPU cache, that is backed by a host memory buffer.
The embedding is initialized with random values, then performs a lookup.

## Import Dynamic Embedding Sample
This sample creates a hierarchical embedding with a GPU cache, that is backed by a host key-vector storage.
The embedding is initialized with values taken from npy files, then performs a lookup.
This flow would be used when importing embeddings trained with Dynamic Embeddings.

## Installation
``` pip install <NVE_ROOT_DIR> ```

## Usage Example
``` ./simple_linear_embedding.py ```

``` ./simple_hierarchical_embedding.py ```
