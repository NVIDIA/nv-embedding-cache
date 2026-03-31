# C API Sample: Hierarchical Embedding Layer

This sample demonstrates using the NVE C API to create a hierarchical embedding layer with a GPU cache backed by an NVHM host table.

The sample:
1. Creates a GPU table (cache layer)
2. Loads the NVHM plugin and creates a host table
3. Builds a hierarchical layer (GPU cache -> NVHM host table)
4. Inserts 1000 embeddings to the host table and 250 embeddings to the GPU table
5. Performs a lookup on 16 keys and verifies the results

To build the sample see the [Getting started](../../README.md#getting-started) section in the main [README.md](../../README.md).

To run the sample call:

```bash
$ ./build/bin/c_api_sample
NVE version: 26.3.0

[1] Creating GPU table (cache)...
  GPU table created (4 MB cache, 128-byte rows)

[2] Creating NVHM host table...
  NVHM host table created

[3] Creating hierarchical embedding layer...
  Hierarchical layer created: GPU cache -> NVHM host table

[4] Inserting 1000 embeddings...
  Inserted 250 embeddings to the GPU table
  Inserted 1000 embeddings to the host table

[5] Looking up 16 keys...
  Hit rates: GPU=31.2%, Host=68.8%

  Sample results:
    key=   0  expected=0.0000  got=0.0000  OK
    key=  62  expected=0.6200  got=0.6200  OK
    key= 124  expected=1.2400  got=1.2400  OK
    key= 186  expected=1.8600  got=1.8600  OK
    key= 248  expected=2.4800  got=2.4800  OK
  All 16 lookups verified successfully!

[6] Cleaning up...
  Done.
```
