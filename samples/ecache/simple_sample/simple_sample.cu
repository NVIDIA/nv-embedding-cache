/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <embedding_cache_combined.h>
#include <embedding_cache_combined.cuh>
#include <iostream>
#include <cuda_runtime.h>
#include <default_allocator.hpp>

using namespace nve;
using IndexT = uint32_t;
using TagT = uint32_t;
using CacheType = CacheSAHostModify<IndexT, TagT>;

#define CHECK_AND_THROW(cond, msg) if (!(cond)) {std::cout << msg << std::endl; throw 0;}
#define ECCheck(err) CHECK_AND_THROW(err == ECERROR_SUCCESS, "Cache error detected")
#define CUDACheck(err) CHECK_AND_THROW(err == cudaSuccess, "CUDA error detected")

void __global__ embedding_mul(IndexT* keys, 
                            const int8_t* embeddingTable, 
                            int8_t* outBuffer, 
                            CacheType::CacheData cache, 
                            size_t stride)
{
    IndexT key = keys[blockIdx.x]; // each block does one key
    
    // device side service function the cache provides to get pointer to where data resides
    using CacheFunctor = AddressFunctor<IndexT, CacheType::CacheData>; 
    int8_t* dataPtr = (int8_t*)CacheFunctor::GetAddress(key, embeddingTable, 0, cache); 
    outBuffer[blockIdx.x * stride + threadIdx.x] = dataPtr[threadIdx.x] * 2;
}

#define MB 1024*1024

int main()
{
    try {
        ECError err = 0;
        const size_t rowSizeInBytes = 512;
        // create instance of default allocator and logger for use in the cache
        DefaultAllocator allocator(DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD);
        Logger logger;
        CacheType::CacheConfig cConfig;
        cConfig.cacheSzInBytes = 4*MB; // the maximum size of device size memory the cache can use
        cConfig.embedWidth = rowSizeInBytes; // the row size of the embedding in bytes
        cConfig.numTables = 1; // number of tables the cache can back up

        // constructing an instance of CacheType so far no memory is allocated, 
        CacheType cache(&allocator, &logger, cConfig);

        // Calling init will calculate required memory and allocate it device side
        err = cache.Init();
        ECCheck(err);

        //allocate table
        const size_t nRows = 10000;
        int8_t* data = nullptr;
        CUDACheck(cudaMallocHost(&data, nRows * rowSizeInBytes));
        // populate data buffer


        // create some keys to query
        const size_t N = 100; // number of keys
        uint32_t keys[N];
        for (size_t i = 0; i < N; i++)
        {
            keys[i] = static_cast<uint32_t>(i);
        }
        IndexT* deviceKeys = nullptr; // device side buffer to hold the keys to query the cache
        CUDACheck(cudaMalloc(&deviceKeys, sizeof(IndexT)*N));
        CUDACheck(cudaMemcpy(deviceKeys, keys, sizeof(IndexT)*N, cudaMemcpyDefault));


        // allocate output buffers
        int8_t* deviceOutBuffer = nullptr; // buffer to hold the embedding vectors found in cache
        CUDACheck(cudaMalloc(&deviceOutBuffer, rowSizeInBytes*N));

        // Creating LookupContext 

        PerformanceMetric missCount;
        err = cache.PerformanceMetricCreate(missCount, MERTIC_COUNT_MISSES); // create a cache miss counter
        ECCheck(err);
        LookupContextHandle lookupHandle;
        err = cache.LookupContextCreate(lookupHandle, &missCount, 1); // create the context with miss counter
        ECCheck(err);

        // perform lookup
        err = cache.Lookup(
            lookupHandle, // the context to lookup in
            deviceKeys, // keys to lookup in cache
            N, // number of keys in deviceKeys buffer
            deviceOutBuffer, // output buffer
            data, // pointer to the embedding table to handle misses
            0, // table index in cache 
            rowSizeInBytes, // stride of output buffer
            0 // default stream
        );
        ECCheck(err);
        CUDACheck(cudaStreamSynchronize(0));
        // cache is unpopulated so all keys should be missed
        int64_t cacheMisses;
        err = cache.PerformanceMetricGetValue(missCount, &cacheMisses, 0);
        ECCheck(err);
        CHECK_AND_THROW(cacheMisses == N, "Should have 0 hits");
        err = cache.PerformanceMetricReset(missCount, 0);
        ECCheck(err);
        // perform some cache population
        // create a modify context
        ModifyContextHandle modifyHandle;
        err = cache.ModifyContextCreate(
            modifyHandle, // handle to the context
            N // maximal update size needed to allocate some buffers
        );

        //Create the sync event this is required to allow the internal cache implementation to synchronize with in-flight inferences
        DefaultECEvent syncEvent(std::vector<cudaStream_t>{0});

        ECCheck(err);
        // population is performd in two phases 1. Calculate which indices is best to fetch to cache this is a non interuptive call
        // 2. modify the cache, this is an interuptive call and no lookups should be called during

        // we can use the Histogram helper class to calculate the priority of each key
        DefaultHistogram hist(
            keys, // a host side buffer of representative keys
            N, // size of buffer in keys
            data, // buffer of values that matches the keys
            rowSizeInBytes, // stride of the value buffer
            true    // linear table
            );
        err = cache.Insert(
            modifyHandle, // modify context to work with
            hist.GetKeys(), // sorted keys by prio
            hist.GetPriority(), // the priority of keys - value the represent how important key is can be a frequency of key
            hist.GetData(), // an array of pointers for each key where is data
            hist.GetNumBins(), // num of keys in the array
            0, // table index
            &syncEvent,
            0
        );
        ECCheck(err);
        CUDACheck(cudaStreamSynchronize(0));

        // query again
        err = cache.Lookup(
            lookupHandle, // the context to lookup in
            deviceKeys, // keys to lookup in cache
            N, // number of keys in deviceKeys buffer
            deviceOutBuffer, // output buffer
            data, // pointer to the embedding table to handle misses
            0, // table index in cache 
            rowSizeInBytes, // stride of output buffer
            0 // default stream
        );
        ECCheck(err);
        CUDACheck(cudaStreamSynchronize(0));
        // all keys should be in cache
        err = cache.PerformanceMetricGetValue(missCount, &cacheMisses, 0);
        ECCheck(err);
        CHECK_AND_THROW(cacheMisses == 0, "Should have 0 misses");
        // use outputbuffer in a different kerenl


        // integrate to into a kernel
        dim3 block(rowSizeInBytes); // each thread in block handles one element in row
        dim3 grid(N); // each block handles one row
        embedding_mul<<<grid, block>>>(
            deviceKeys,  // key buffer
            data, // pointer to table to resolve misses
            deviceOutBuffer, // output buffer
            cache.GetCacheData(lookupHandle), // casting the lookup handle into a struct for kernel usage 
            rowSizeInBytes // outbuffer stride
        );
        CUDACheck(cudaGetLastError()); // Check kernel launch didn't generate an error
        CUDACheck(cudaDeviceSynchronize());

        // perform clean up
        err = cache.LookupContextDestroy(lookupHandle);
        ECCheck(err);
        err = cache.ModifyContextDestroy(modifyHandle);
        ECCheck(err);
        err = cache.PerformanceMetricDestroy(missCount);
        ECCheck(err);

        CUDACheck(cudaFreeHost(data));
        CUDACheck(cudaFree(deviceKeys));
        CUDACheck(cudaFree(deviceOutBuffer));
    } catch (...) {
        std::cerr << "Unhandled exception caught!" << std::endl;
        return 1;
    }
    return 0;
}