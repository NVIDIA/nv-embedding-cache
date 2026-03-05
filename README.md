# NV Embedding Cache

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-26.03.0-green.svg)](version.txt)

NV Embedding Cache is a domain-specific SDK for high performance recommender systems embedding lookup.
We accelerate embedding lookups with a combination of SW caches in GPU/Host memory and customized CUDA kernels.
The main focus is recommender inference with embeddings that exceed the local GPU's memory capacity.

The SDK offers several configurations to support different memory allocations:
- All embeddings are allocated in linear GPU memory: use **NVEmbedding** with cache_type **NoCache**(Py) / **GPUEmbeddingLayer** (C++)
- Some embeddings are cached in GPU memory and all embeddings are in linear memory (Host or other GPUs): use **NVEmbedding** with cache_type **LinearUVM**(Py) / **LinearUVMEmbeddingLayer** (C++)
- Some embeddings are cached in GPU memory, Some embeddings cached in host memory and all embeddings kept in a remote parameter server:
use **NVEmbedding** with cache_type **Hierarchical**(Py) / **HierarchicalEmbeddingLayer** (C++)

** Linear memory in this context, means all embeddings are consecutive in virtual memory space. More specifically, the address of embedding *i* can be computed as start_address + *i* * embedding_size

## Getting Started 

### Prerequisites
- C++17 capable compiler (we test with both GCC 13.3 and Clang 20.1)
- CUDA 12.8+ (earlier version will work with minor code changes)
- CMake 3.18+
- Python 3.10+
- Torch
- (Optional) Redis 7.0.15+ - used in some tests

The provided [Dockerfile](Dockerfile) satisfies these prerequisites. If you're using your own environment, you can skip step (2) in the installation instructions below.

### Installation

1. Clone the repo:
    ```bash
    git clone git@github.com:NVIDIA/nv-embedding-cache.git
    cd nv-embedding-cache
    git submodule update --init --recursive
    ```
2. Start the docker container:
    ```bash
    docker build -t nve --build-arg START_DIR=$(pwd) --build-arg UID=$(id -u) --build-arg UNAME=$(id -u -n) --build-arg GID=$(id -g) --build-arg GNAME=$(id -g -n) .
    docker run --cap-add=ALL --net=host --ipc=host --gpus all -it --rm -v $(pwd):$(pwd) nve
    ```
3. Build and install the Python bindings (by default in ./build)
    ```bash
    pip install .
    ```
4. Alternatively, build C++ sources with samples and tests
    ```bash
    mkdir build_dir
    cd build_dir
    cmake ..
    make all -j
    cd -
    ```

## Documentation & Samples
The [docs](docs/) dir contains our documentation. It's structured as follows:
```bash
docs
├── advanced.md   # Advanced topics
├── benchmarks.md # Benchmarking instructions
├── cpp_api.md    # C++ API documentation
├── overview.md   # SDK Overview            <-- Start Here!
├── python_api.md # Python bindings documentation
└── samples.md    # Samples listing and description
```
A good place to start is: [docs/overview.md](docs/overview.md).

Samples are listed in [docs/samples.md](docs/samples.md). The basics are covered in [simple_cpp](samples/simple_cpp/) and [pytorch/simple_sample](samples/pytorch/simple_sample/)

Benchmarking scripts are available in [benchmarks/](benchmarks/). See instructions at [docs/benchmarks.md](docs/benchmarks.md)

## License
 The NV Embedding Cache SDK is licensed under the terms of the Apache 2.0 license. See [LICENSE](LICENSE) for more information.

 ### Third-party dependencies
 This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
 
 Third party dependencies are available as git submodules and can be found at [third_party](third_party). Their respective licenses are listed below.

 |Name|License|
 |-|-|
 |abseil-cpp|https://github.com/abseil/abseil-cpp/blob/master/LICENSE|
 |argparse|https://github.com/p-ranav/argparse/blob/master/LICENSE|
 |dlpack|https://github.com/dmlc/dlpack/blob/main/LICENSE|
 |googletest|https://github.com/google/googletest/blob/main/LICENSE|
 |hiredis|https://github.com/redis/hiredis/blob/master/COPYING|
 |json|https://github.com/nlohmann/json/blob/develop/LICENSE.MIT|
 |parallel-hashmap|https://github.com/greg7mdp/parallel-hashmap/blob/master/LICENSE|
 |pybind11|https://github.com/pybind/pybind11/blob/master/LICENSE|
 |redis-plus-plus|https://github.com/sewenew/redis-plus-plus/blob/master/LICENSE|
 |rocksdb|https://github.com/facebook/rocksdb/blob/main/LICENSE.Apache| 
