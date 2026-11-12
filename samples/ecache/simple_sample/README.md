# Embedding Cache Simple Sample

This sample demonstrates direct use of the low-level GPU embedding cache API ([embed_cache.h](../../../include/ecache/embed_cache.h)) without the table and layer abstractions. It uses the set-associative cache variant `CacheSAHostModify` from [ec_set_associative.h](../../../include/ecache/ec_set_associative.h).

The sample:
1. Configures and initializes a 4 MB cache for a single table with 512-byte rows
2. Allocates a 10,000 row embedding table in pinned host memory, used to resolve cache misses
3. Creates a miss counter (`PerformanceMetric`) and a lookup context
4. Looks up 100 keys in the empty cache and verifies they all miss
5. Ranks the keys with `DefaultHistogram` and inserts them into the cache through a modify context
6. Looks up the same keys again and verifies there are no misses
7. Calls the cache from a custom CUDA kernel (`embedding_mul`) using `AddressFunctor::get_address()` on the cache data returned by `get_cache_data()`
8. Destroys the contexts and frees the buffers

Cache population is a two-phase process: first choose which keys to insert (here with `DefaultHistogram`), then call `insert()`. Lookups must not run while `insert()` is modifying the cache; the `DefaultECEvent` passed to `insert()` lets the cache synchronize with in-flight lookups on the given streams.

To build the sample see the [Getting started](../../../README.md#getting-started) section in the main [README.md](../../../README.md).

To run the sample call:

```bash
$ ./build/bin/simple_sample
```

The sample prints nothing on success and exits with code 0. On failure it prints the failing check (for example `Should have 0 misses`) followed by `Unhandled exception caught!` and exits with code 1.
