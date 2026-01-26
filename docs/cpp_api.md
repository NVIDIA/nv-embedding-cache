# NV Embedding Cache C++ API

This section will overview the various C++ APIs offered by the NV Embedding Cache SDK.  
Similarlry to the [overview](overview.md) section, the APIs will be divided to layers, tables and low-level components.  
In the document below, table and cache are used interchangeably, as we often use tables as caches for the entire embedding.

For more specific information, please consult the headers in [include/](../include/) and the [Samples](samples.md) section.

## Compilation

To build the SDK, follow the instructions in [Getting started](../README.md#getting-started). If you installed the Python bindings, then it's already built in the [build](../build/) folder.
Most applications using the SDK should link with `build/lib/libnve-common.so`, `libcudart.so`. Any plugin used (e.g. `libnve-plugin-redis.so`) should be present in the `LD_LIBRARY_PATH` environment variable during runtime.

## Embedding Layer API

Layers are objects that have associated embedding data and provide read/write operations on embedding rows. The layer API is in [embedding_layer.hpp](../include/embedding_layer.hpp) (see EmbeddingLayerBase class). Embedding layers have one or more tables associated with them, depending on the locations in which data is stored. If a table holds only some of the embedding rows, it is used as a cache. Typically, a faster memory will have smaller capacity - leading us to use a small part of the faster memory as a cache while storing the rest of the data in slower memory. A layer may have multiple tables (caches) and a given key-row pair may exist in multiple tables at the same time.
Layers may allocate and copy memory buffers during operation (e.g. a lookup using a GPU table with input keys in system memory may allocate a buffer on the GPU and copy the keys to it).

There are several layer types, differentiated by the embedding storage:
1. [GPUEmbeddingLayer](../include/gpu_embedding_layer.hpp) holds all embedding data in GPU memory.
2. [LinearUVMEmbeddingLayer](../include/linear_embedding_layer.hpp) holds the embedding data in linear system memory and also keeps a cache in GPU memory.
3. [HierarchicalEmbeddingLayer](../include/hierarchical_embedding_layer.hpp) holds the embedding data in a remote parameter server (e.g. Redis or RocksDB) and optional caches in GPU and system memory.

The main operations supported by layers are:
1. Lookup: given a list of keys, return the embeddings rows corresponding to them. This can involve lookups in multiple tables and reading from local or remote memories, depending on the layer. This op may change the content of tables (see [insert heuristic](advanced.md#insert-heuristic)).
2. Insert: given a table, a list of keys and their respective embedding rows, the table decids which keys should reside in it. This method is for updating table residency (i.e. which key should reside in the table). Layers perform this internally (see [insert heuristic](advanced.md#insert-heuristic)).  
    Notes:
    * Inserting new rows may evict existing ones.  
    * This op is not meant for updating new row values for existing keys - that should be accomplished with "update".  
      Furthermore, "insert" may ignore new row values for keys already existing in the table.  
    * Keys/rows given to "insert" are not guaranteed to be resident in the table after the operation.  
3. Update: given a list of keys and their respective embedding row values, find all occurrences of the keys in the layer's tables and replace their existing row values with the provided ones.  
    Note: this op does not change any table residency (i.e. which key exists in each table), it will only change row values for keys that already exist in the table.
4. Accumulate: given a list of keys and their respective embedding row gradients, find all occurrences of the keys in the layer's tables and accumulate the given gradients into their existing row values.  
    Note: this op does not change any table residency (i.e. which key exists in each table), it will only change row values for keys that already exist in the table.
5. Erase: given a list of keys and a table type, remove the given keys from the table. Keys that are not resident in the table are ignored.  
    Note: typically, there's no reason to use "Erase" - rows will be evicted as needed during "Insert".
6. Create execution context: execution contexts are containers for state variables and memory buffers needed during execution of all the other operations. See: [Execution contexts](advanced.md#execution-contexts) for more details.

### Samples

* The [simple_cpp](../samples/simple_cpp/) sample is a simple example of using the layer API.
* A more complex sample using the C++ API can be found in [layer_sample](../samples/layer_sample/).

## Table API

Tables are a lower level of layers, typically with only one data storage location.
They offer similar operations to layers but will require a more careful usage. For example, while layers accept buffers on either GPU or system memory, copying them as needed, tables will require buffers to be present in specified memories.
The main table API is in: [table.hpp](../include/table.hpp) and specific table types are:
1. [GpuTable](../include/gpu_table.hpp) - a table holding a GPU cache of embedding rows, with optional direct access to system memory.
2. [NvhmMapTable](../plugins/nvhm/include/nvhm_map_table.hpp) - a table holding a cache in system memory based on the [nvHashMap](https://github.com/NVIDIA/nvhashmap) hashmap.
3. [AbseilFlatMapTable](../plugins/abseil/include/abseil_flat_map_table.hpp) - a table holding a cache in system memory based on the [Abseil](https://github.com/abseil/abseil-cpp) library.
4. [PHMapFlatMapTable](../plugins/phmap/include/phmap_flat_map_table.hpp) - a table holding a cache in system memory based on the [parallel-hashmap](https://github.com/greg7mdp/parallel-hashmap) library.
5. [RedisClusterTable](../plugins/redis/include/redis_cluster_table.hpp) - a table accessing a remote [Redis](https://redis.io/) cluster.
6. [RocksDBTable](../plugins/rocksdb/include/rocksdb_table.hpp) - a table accessing remote [RocksDB](https://rocksdb.org/) storage.

Tables (except the GPUTable) are built separately as plugins, to reduce dependencies external where possible.

### Samples
* TBD

## Low-level Componmenets API

Some classes are only exposed as sources and application wishing to utilize them need to include them in their build. We try to keep these as headers-only to make building easier.

### GPU Embedding Cache

The GPU Embedding cache is located in the [ecache](../include/ecache/) folder. This component implements a low-level software-managed cache in GPU memory. This cache is optimized for recommendation systems embeddings. I.e. we assume the keys' distribution is close to [power-law](https://en.wikipedia.org/wiki/Power_law) and stable. The main APIs are in [embed_cache.h](../include/ecache/embed_cache.h) and the SDK uses the [ec_set_associative](../include/ecache/ec_set_associative.h) variant.

#### Samples
* See the code in the [ecache/simple_sample](../samples/ecache/simple_sample/) sample for a simple example of the cache API usage.

## Utility classes
Several key functionalities have a default implementation which users can override without resorting to source code changes in the SDK.  
1. [Allocator](../include/allocator.hpp) - this class will be used to allocate buffers on system or GPU memory.
    An allocator object can be provided when constructing a layer/table/execution context.
    It is possible to use different allocators for different objects, but often a single allocator will suffice.
2. [Logger](include/logging.hpp) - while the `Logger` class isn't overrideable, the application can implement a `LoggerBackend` then call `GetGlobalLogger()->set_backend()`. You'll need to include [common.hpp](../include/common.hpp) for `GetGlobalLogger()`.
3. [Threadpool](../include/thread_pool.hpp) - this class implements a multi-threaded execution environment for tasks (`std::function`). The SDK uses the thread pool to execute parallel on the CPU. By default the `SimpleThreadPool` will be used. Note that all custom tasks that perform CUDA calls must set the CUDA current device (using cudaSetDevice or using the ScopedDevice class), as the device context is tied to the thread performing the task which can differ from the calling thread.
