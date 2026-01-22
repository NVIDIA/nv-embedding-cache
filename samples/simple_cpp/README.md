# Simple C++ API Usage Sample

This is a "hellow world" type sample for the C++ APIs.

The sample will initialize a GPU table, the use it to initialize a linear layer.
Next, the sample explicitly inserts a single embedding row to the layer and perform a lookup on the row's key to retrieve it from the layer.

To build the layer see the [Getting started](../../README.md#getting-started) section in the main [README.md](../../README.md).

To run the sample call:

```bash
$ ./build/bin/simple_cpp
Allocating linear table in host memory
Creating a GPU table with cache
Creating a linear Layer
Creating an execution context
Inserting a key-row to the GPU table
Looking up a key in the layer
Tearing down allocated objects
```
