# NV Embedding Cache

NV Embedding Cache is a domain-specific SDK for high performance recommender system embeddings.
We accelerate embedding lookups with a combination of SW caches in GPU/Host memory and customized CUDA kernels.
The main target usage is recommender systems inference with embeddings that exceed the local GPU's memory capacity.

## Getting Started 

1. Clone the repo:
```
git clone git@github.com:NVIDIA/nv-embedding-cache.git
cd nv-embedding-cache
git submodule update --init --recursive
```
2. Start the development docker image:
```
docker build -t nve --build-arg START_DIR=$(pwd) --build-arg UID=$(id -u) --build-arg UNAME=$(id -u -n) --build-arg GID=$(id -g) --build-arg GNAME=$(id -g -n) .
docker run --cap-add=ALL --net=host --ipc=host --gpus all -it --rm -v $(pwd):$(pwd) nve
```
* Alternatively use the .devcontainer
3. Build from sources:
```
mkdir build_dir
cd build_dir
cmake ..
make all -j
cd -
```
4. Install Python bindings (will build in ./build without samples/tests):
```
pip install .
```

## Documentation & Samples
The [docs](docs/) dir contains more documentation files. A good place to start is: [overview.md](docs/overview.md).

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
