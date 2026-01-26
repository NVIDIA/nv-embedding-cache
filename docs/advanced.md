# Advanced Topics

## Execution contexts
The collection of state and resources needed during execution of lookup/modify is called an [execution context](include/execution_context.hpp). These objects are created by a layer or table (depending on the selected integration level) and are used by the calling software stack to separate resources of parallel executions. I.e. each execution context represents a single reuseable parallel execution environment. For example, using 3 contexts, you can run 3 operations in parallel (each using a different context). Using the same context in multiple parallel ops can result in undefined behavior.

The application is expected to create a fixed amount of execution contexts during initialization, then use them in some repeating order (e.g. round-robin).
Creating additional contexts during runtime is allowed, but is less performant. Bear in mind that each context will rquire additional memory allocations.
Execution contexts use lazy memory allocation, so expect an operation with larger than seen before dimensions (e.g. number of indices to lookup) to potentially result in memory allocation.

During teardown, the application must destroy all execution contexts of a layer/table before destroying the layer/table.

## Cache management

### Modify Operations
All operations that change cache residency or cache storage such as insert/update are collectively call Modify operations.
Every Modify operation requires a ModifyContext which provides a parallel execution unit. The cache doesn't support multiple Modify operations inflight on the GPU side - it is the user's responsibility to ensure that, e.g using a single stream to launch all Modify operations.

### Invalidate and Commit
To be able to run Lookup and Modify operations in parallel, we employ a paradigm called invalidate and commit.
The modify operation will first launch a kernel to invalidate the relevant cache entries, then wait until all inflight lookups
have concluded, before altering the cache and re-enabling the affected cache entries.

#### Custom Flows
Invalidate and Commit relies on Lookup operation being queued on the GPU in an atomic fashion e.g a single CUDA kernel. Some users may implement their own complex gather flows. In order to maintain the required atomicity, if the flow uses more than one CUDA kernel, the user needs to call the Start/end_custom_flow APIs.

## Multi device
Embeddings can span multiple devices/nodes by way of sharding. This is accomplished with a CUDA buffer tha spans multiple devices as detailed in the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#).
The user can either create this allocation on their own or use the CUDADistributedBuffer class (see: [../include/distributed.hpp](../include/distributed.hpp)). Similarily, the python MPIMemBlock class can be used to share such a buffer across MPI process group.

## Insert heuristic
The embedding layers that use a cache, can choose when it's time to update the cache residency and insert new cache lines by calling an InsertHeuristic object that determines when to initiate this update. The application can use the DefaultInsertHeuristic or derive it's own.
See [../include/insert_heuristic.hpp](../include/insert_heuristic.hpp) for more details.

## Unit tests
Unit tests are located in [../tests/](../tests/) and are using 2 frameworks: [GoogleTest](https://github.com/google/googletest) and [PyTest](https://docs.pytest.org/en/stable/).

To run the GoogleTests use:
```bash
./tests/embedding_layer/redis_cluster.sh start
sleep 5
for test in build/bin/*test*; do  ./$test; done
./tests/embedding_layer/redis_cluster.sh stop
```
* Note: the redis_cluster.sh script is handling local redis server nodes needed for some of the tests

To run the PyTest tests use:
```bash
pytest tests
```
* Note: make sure to run ```pip install .``` before testing.
