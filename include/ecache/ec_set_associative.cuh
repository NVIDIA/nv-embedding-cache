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

#pragma once
#include "embed_cache.cuh"
#include "ec_set_associative.h"
#include "ec_kernel_common.cuh"
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <type_traits>
#include <limits>

// Disables `pipeline_shared_state` initialization warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas" // ignore warning for Clang
#pragma nv_diag_suppress static_var_with_dynamic_init
#pragma GCC diagnostic pop

namespace nve {

constexpr __host__ __device__ uint32_t calc_pipe_buffer_size(uint32_t row_size)
{
  return row_size * 2 + 16;
}

template<typename IndexT, typename TagT>
static __device__ inline uint32_t EmbedCacheGetWayMask(IndexT laneIdx, uint32_t currTable, const typename EmbedCacheSA<IndexT, TagT>::CacheData data)
{
    uint64_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, TagT>::NUM_WAYS;
    uint64_t setIdx = laneIdx % data.nSets;
    const TagT* pWays = (const TagT*)(data.pTags + (cacheOffset + setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS) * sizeof(TagT));
    uint32_t out = 0;
    for (uint32_t i = 0; i < EmbedCacheSA<IndexT, TagT>::NUM_WAYS; i++)
    {
        TagT way = pWays[i];
        IndexT key = way * data.nSets + setIdx;
        uint32_t b = key == laneIdx;
        out |= (b << i); 
    }

    return out;
}

template<typename IndexT, typename TagT>
static __device__ inline uint64_t EmbedCacheGetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCacheSA<IndexT, TagT>::CacheData data)
{
    uint64_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, TagT>::NUM_WAYS;
    uint64_t setIdx = laneIdx % data.nSets;
    uint32_t out = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
    uint32_t way = __ffs(out) - 1;
    uint64_t lanePtr = (out == 0) ? ((pTable == nullptr) ? 0 : (uint64_t)pTable + (laneIdx)*(uint64_t)data.rowSizeInBytes) : 
        (uint64_t)data.pCache + (cacheOffset + setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes;

    if (out == 0 && data.bCountMisses)
    {
        atomicAdd((unsigned long long*)data.misses, 1);
    }

    return lanePtr;
}

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCacheSA<IndexT, uint16_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCacheSA<IndexT, uint16_t>::CacheData data)
    {
        uint32_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, uint16_t>::NUM_WAYS;
        uint32_t setIdx = laneIdx % data.nSets;
        uint4 ways = *(uint4*)(data.pTags + (cacheOffset + setIdx * EmbedCacheSA<IndexT, uint16_t>::NUM_WAYS) * sizeof(uint16_t));
        uint16_t* pWays = (uint16_t*)(&ways);
        uint32_t out = 0;
        for (uint32_t i = 0; i < EmbedCacheSA<IndexT, uint16_t>::NUM_WAYS; i++)
        {
            uint32_t way = pWays[i];
            uint32_t key = way * data.nSets + setIdx;
            uint32_t b = key == laneIdx;
            out |= (b << i); 
        }
        uint32_t way = __ffs(out) - 1;
        uint64_t lanePtr = (out == 0) ? 
                           ((pTable == nullptr) ? 0 : ((uint64_t)pTable + (laneIdx)*(uint64_t)data.rowSizeInBytes)) : 
            (uint64_t)data.pCache + (cacheOffset + setIdx * EmbedCacheSA<IndexT, uint16_t>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes;

        if (out == 0 && data.bCountMisses)
        {
            atomicAdd((unsigned long long*)data.misses, 1llu);
        }

        return lanePtr;
    }
};




template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCacheSA<IndexT, uint32_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCacheSA<IndexT, uint32_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, uint32_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCacheSA<IndexT, uint64_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCacheSA<IndexT, uint64_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, uint64_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCacheSA<IndexT, int64_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCacheSA<IndexT, int64_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, int64_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT>
class AddressFunctor<IndexT, typename EmbedCacheSA<IndexT, int32_t>::CacheData>
{
public:
    static __device__ inline uint64_t GetAddress(IndexT laneIdx, const int8_t* pTable, uint32_t currTable, const typename EmbedCacheSA<IndexT, int32_t>::CacheData data)
    {
        return EmbedCacheGetAddress<IndexT, int32_t>(laneIdx, pTable, currTable, data);
    }
};

template<typename IndexT, typename TagT, typename DataType>
__global__ void MemUpdateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t sz)
{
    if (blockIdx.x < pList->nEntries) {
        typename EmbedCacheSA<IndexT, TagT>::ModifyEntry e = pList->pEntries[blockIdx.x];
        MemcpyWarp<32, DataType>(e.pDst, e.pSrc, sz);
    }
}

template<typename IndexT, typename TagT, typename DataType>
__global__ void MemUpdateAccumulateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t sz)
{
    if (blockIdx.x < pList->nEntries) {
        typename EmbedCacheSA<IndexT, TagT>::ModifyEntry e = pList->pEntries[blockIdx.x];
        constexpr uint32_t ELEMENT_SIZE = sizeof(DataType);
        constexpr uint32_t SUBWARP_WIDTH = 32;
        for (uint32_t k = 0; k < sz; k += ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t offset = k + threadIdx.x * ELEMENT_SIZE;
            if (offset < sz)
            {
                DataType d = *(DataType*)((int8_t*)e.pSrc + offset);
                DataType* dst_ptr = (DataType*)((int8_t*)e.pDst + offset);
                atomicAdd(dst_ptr, d);
            }
        }
    }
}

template <typename InputDataT, typename LoadDataT>
static __device__ inline  void unpack(LoadDataT inputVec, InputDataT* outputArray);

template <>
__device__ inline  void unpack<int8_t, char4>(char4 inputVec, int8_t* outputArray)
{
    outputArray[0] = inputVec.x;
    outputArray[1] = inputVec.y;
    outputArray[2] = inputVec.z;
    outputArray[3] = inputVec.w;
}

template <>
__device__ inline  void unpack<int8_t, char2>(char2 inputVec, int8_t* outputArray)
{
    outputArray[0] = inputVec.x;
    outputArray[1] = inputVec.y;
}

template<typename IndexT, typename TagT, typename InputDataT, typename LoadDataT, typename ScaleT, typename CacheDataT>
__global__ void MemUpdateAccumulateQuantizedKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t sz)
{
    if (blockIdx.x < pList->nEntries) {
        typename EmbedCacheSA<IndexT, TagT>::ModifyEntry e = pList->pEntries[blockIdx.x];
        // assume alignment is a multiple of sizeof(ScaleT), should be checked before calling the kernel
        constexpr uint32_t ELEMENT_SIZE = sizeof(CacheDataT);
        constexpr uint32_t INPUT_ELEMENT_SIZE = sizeof(InputDataT);
        constexpr uint32_t LOAD_ELEMENT_SIZE = sizeof(LoadDataT);
        constexpr uint32_t VECTOR_WIDTH = LOAD_ELEMENT_SIZE / INPUT_ELEMENT_SIZE;
        auto num_elements = sz / sizeof(ELEMENT_SIZE);
        auto input_sz = num_elements * INPUT_ELEMENT_SIZE;
        ScaleT scale = *((ScaleT*) (e.pSrc + input_sz));

        constexpr uint32_t SUBWARP_WIDTH = 32;
        for (uint32_t k = 0; k < input_sz; k += LOAD_ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t src_offset = k + threadIdx.x * LOAD_ELEMENT_SIZE;
            uint32_t dst_offset = k + threadIdx.x * ELEMENT_SIZE * VECTOR_WIDTH;
            if (src_offset < input_sz)
            {
                LoadDataT d = *(LoadDataT*)((int8_t*)e.pSrc + src_offset);
                CacheDataT* dst_ptr = (CacheDataT*)((int8_t*)e.pDst + dst_offset);
                InputDataT d_unpacked[VECTOR_WIDTH];
                unpack<InputDataT, LoadDataT>(d, d_unpacked);
                for (uint32_t e = 0; e < VECTOR_WIDTH; e++) {
                    CacheDataT val = static_cast<CacheDataT>(d_unpacked[e]) * scale;
                    atomicAdd(dst_ptr + e, static_cast<CacheDataT>(d_unpacked[e]) * scale);
                }
            }
        }
    }
}

template<typename IndexT, typename TagT>
__global__ void InvalidateTagKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, TagT* pTags)
{
    auto tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < pList->nEntries)
    {
        typename EmbedCacheSA<IndexT, TagT>::ModifyEntry e = pList->pEntries[tid];
        TagT* pToModTag = pTags + e.set * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + e.way;
        *pToModTag = static_cast<TagT>(-1);
    }
}

template<typename IndexT, typename TagT>
__global__ void TagUpdateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, TagT* pTags)
{
    auto tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < pList->nEntries)
    {
        typename EmbedCacheSA<IndexT, TagT>::ModifyEntry e = pList->pEntries[tid];
        TagT* pToModTag = pTags + e.set * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + e.way;
        *pToModTag = e.tag;
    }
}

template<typename IndexT, typename TagT, uint32_t SUBWARP_WIDTH, typename DataType>
__global__ void Query(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data, uint32_t currTable, size_t stride)
{
    const uint32_t blockDims = blockDim.x * blockDim.y;
    uint32_t block_ptr = blockIdx.x * blockDims;
    uint32_t tid = block_ptr + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    const uint32_t subwarp_idx = threadIdx.x / SUBWARP_WIDTH;
    const uint32_t subwarp_ptr = block_ptr + subwarp_idx * SUBWARP_WIDTH;
    const uint32_t intra_subwarp_idx = threadIdx.x % SUBWARP_WIDTH;

    uint64_t laneptr;
    if (tid >= len)
    {
        laneptr = 0;
    }
    else
    {
        IndexT laneIdx = d_keys[tid];
        uint32_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, TagT>::NUM_WAYS;
        uint32_t setIdx = laneIdx % data.nSets;
        uint32_t laneout = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
        uint32_t laneway = __ffs(laneout) - 1;
        if (laneout == 0)
        {
            unsigned long long old = atomicAdd((unsigned long long*)d_missing_len, 1llu);
            if (data.bCountMisses)
            {
                atomicAdd((unsigned long long*)data.misses, 1);
            }
            d_missing_index[old] = tid;
            d_missing_keys[old] = laneIdx;
            laneptr = 0;
        }
        else
        {
            auto way = laneway;
            laneptr = (uint64_t)(data.pCache + (cacheOffset + setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes);
        }
    }

    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint32_t ELEMENT_SIZE = sizeof(DataType);
        uint64_t src_ptr = __shfl_sync(0xffffffff, laneptr, s, SUBWARP_WIDTH);
        if (src_ptr == 0)
        {
            continue;
        }
        for (uint32_t k = 0; k < data.rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t offset = k + intra_subwarp_idx * ELEMENT_SIZE;
            if (offset < data.rowSizeInBytes)
            {
                DataType d = *(DataType*)(src_ptr + offset);
                DataType* dst_ptr = (DataType*)(d_values + (subwarp_ptr + s) * stride + offset);
                *dst_ptr = d;
            }
        }
        
    }
}

template<typename IndexT, typename TagT, uint32_t SUBWARP_WIDTH, typename DataType>
__global__ void UpdateAccumulateNoSync(const IndexT* d_keys, const size_t len,
    const int8_t* d_values, typename EmbedCacheSA<IndexT, TagT>::CacheData data, uint32_t currTable, size_t stride)
{
    const uint32_t blockDims = blockDim.x * blockDim.y;
    uint32_t block_ptr = blockIdx.x * blockDims;
    uint32_t tid = block_ptr + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    const uint32_t intra_subwarp_idx = threadIdx.x % SUBWARP_WIDTH;

    uint64_t lane_src_ptr = 0;
    uint64_t lane_dst_ptr = 0;
    if (tid >= len)
    {
        lane_dst_ptr = 0;
        lane_src_ptr = 0;
    }
    else
    {
        IndexT laneIdx = d_keys[tid];
        uint32_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, TagT>::NUM_WAYS;
        uint32_t setIdx = laneIdx % data.nSets;
        uint32_t laneout = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
        uint32_t laneway = __ffs(laneout) - 1;
        if (laneout == 0)
        {
            lane_dst_ptr = 0;
            lane_src_ptr = 0;
        }
        else
        {
            auto way = laneway;
            lane_dst_ptr = (uint64_t)(data.pCache + (cacheOffset + setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes);
            lane_src_ptr = (uint64_t)(d_values + (tid * stride));
        }
    }

    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint32_t ELEMENT_SIZE = sizeof(DataType);
        uint64_t src_ptr = __shfl_sync(__activemask(), lane_src_ptr, s, SUBWARP_WIDTH);
        uint64_t dst_ptr = __shfl_sync(__activemask(), lane_dst_ptr, s, SUBWARP_WIDTH);
        if (dst_ptr == 0)
        {
            continue;
        }
        
        for (uint32_t k = 0; k < data.rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t offset = k + intra_subwarp_idx * ELEMENT_SIZE;
            if (offset < data.rowSizeInBytes)
            {
                DataType d = *(DataType*)(src_ptr + offset);
                atomicAdd((DataType*)(dst_ptr + offset), d);
            }
        }
        
    }
}


template<typename IndexT, typename TagT, uint32_t SUBWARP_WIDTH, typename DataType>
__global__ void Query(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint32_t* d_hit_mask,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data, uint32_t currTable, size_t stride)
{
    const uint32_t blockDims = blockDim.x * blockDim.y;
    const uint32_t warp_ptr = blockIdx.x * blockDims + threadIdx.y*blockDim.x;
    const uint32_t tid = warp_ptr + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    const uint32_t subwarp_idx = threadIdx.x / SUBWARP_WIDTH;
    const uint32_t subwarp_ptr = warp_ptr + subwarp_idx * SUBWARP_WIDTH;
    const uint32_t intra_subwarp_idx = threadIdx.x % SUBWARP_WIDTH;
    uint64_t laneptr;

    if (subwarp_ptr >= len) {
        return;
    }

    if (tid >= len)
    {
        laneptr = 0;
    }
    else
    {
        //check if there was a hit earlier
        const uint32_t bit = 1ul << (threadIdx.x);
        const bool hit = (d_hit_mask[subwarp_ptr / 32] & bit) == bit;
        if (hit) {
            laneptr = 0; //as data already in place
        }
        else {
            const IndexT laneIdx = d_keys[tid];
            const uint32_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, TagT>::NUM_WAYS;
            const uint32_t setIdx = laneIdx % data.nSets;
            const uint32_t laneout = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
            const uint32_t laneway = __ffs(laneout) - 1;

            if (laneout == 0)
            {
                if (data.bCountMisses)
                {
                    atomicAdd((unsigned long long*)data.misses, 1);
                }
                laneptr = 0;
            }
            else
            {
                auto way = laneway;
                laneptr = (uint64_t)(data.pCache + (cacheOffset + setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes);
            }
        }
    }
    uint32_t localHitMask = 0;
    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint32_t ELEMENT_SIZE = sizeof(DataType);
        uint64_t src_ptr = __shfl_sync(__activemask(), laneptr, s, SUBWARP_WIDTH);
        uint32_t outputIdx = subwarp_ptr + s;
        if (src_ptr == 0)
        {
            continue;
        }
        // hit 
        localHitMask |= 1 << s;
        for (uint32_t k = 0; k < data.rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH)
        {
            uint32_t offset = k + intra_subwarp_idx * ELEMENT_SIZE;
            if (offset < data.rowSizeInBytes)
            {
                DataType d = *(DataType*)(src_ptr + offset);
                DataType* dst_ptr = (DataType*)(d_values + outputIdx * stride + offset);
                *dst_ptr = d;
            }
        }
    }
    if (threadIdx.x == 0)
    {
        d_hit_mask[subwarp_ptr / 32] |= localHitMask;
    }
}

// need to have an argument explicity depend on IndexT type or the compiler gets confused
template<typename IndexT, typename TagT>
cudaError_t callTagInvalidateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEntries, TagT* pTags, cudaStream_t stream)
{
    dim3 gridSize((nEntries + 32 - 1)/32,1);
    dim3 blockSize(32, 1); 
    InvalidateTagKernel<IndexT, TagT><<<gridSize, blockSize, 0, stream>>>(pList, pTags);
    return cudaGetLastError();
}

template<typename IndexT, typename TagT>
cudaError_t callMemUpdateAccumulateQuantizedKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEntries, uint32_t rowSizeInBytes, DataTypeFormat inputFormat, DataTypeFormat outputFormat, cudaStream_t stream)
{
    dim3 gridSize(nEntries,1);
    dim3 blockSize(32, 1);

    if (inputFormat != DATATYPE_INT8_SCALED) {
        //not implemented
        return cudaErrorNotSupported;
    }
    switch (outputFormat) {
      case DATATYPE_FP16:
        {
            auto srcRowSizeInBytes = rowSizeInBytes/sizeof(__half) + 2;
            assert((srcRowSizeInBytes % sizeof(__half)) == 0);
            if ((srcRowSizeInBytes % 4) == 0) {
                MemUpdateAccumulateQuantizedKernel<IndexT, TagT, int8_t, char4, __half, __half><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
            } else {
                MemUpdateAccumulateQuantizedKernel<IndexT, TagT, int8_t, char2, __half, __half><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
            }
        }
        break;
      case DATATYPE_FP32:
        {
            [[maybe_unused]] auto srcRowSizeInBytes = rowSizeInBytes/sizeof(float);
            assert((srcRowSizeInBytes % sizeof(float)) == 0);
            MemUpdateAccumulateQuantizedKernel<IndexT, TagT, int8_t, char4, float,  float><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
        }
        break;
      default:
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

// need to have an argument explicity depend on IndexT type or the compiler gets confused
template<typename IndexT, typename TagT>
cudaError_t callMemUpdateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEntries, uint32_t rowSizeInBytes, cudaStream_t stream)
{
    dim3 gridSize(nEntries,1);
    dim3 blockSize(32, 1); 
    if (rowSizeInBytes % sizeof(uint4) == 0)
    {
        MemUpdateKernel<IndexT, TagT, uint4><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
    }
    else if (rowSizeInBytes % sizeof(uint32_t) == 0)
    {
        MemUpdateKernel<IndexT, TagT, uint32_t><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
    }
    else
    {
        MemUpdateKernel<IndexT, TagT, int8_t><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
    }
    return cudaGetLastError();
}

template<typename IndexT, typename TagT>
cudaError_t callMemUpdateAccumulateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEntries, uint32_t rowSizeInBytes, DataTypeFormat inputFormat, [[maybe_unused]] DataTypeFormat outputFormat, cudaStream_t stream)
{
    assert(inputFormat == outputFormat);
    dim3 gridSize(nEntries,1);
    dim3 blockSize(32, 1); 
    switch (inputFormat)
    {
    case DATATYPE_FP16:
        MemUpdateAccumulateKernel<IndexT, TagT, __half><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
        break;
    case DATATYPE_FP32:
        MemUpdateAccumulateKernel<IndexT, TagT, float><<<gridSize, blockSize, 0, stream>>>(pList, rowSizeInBytes);
        break;
    default:
        //not implemented
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

template<typename IndexT, typename TagT>
cudaError_t callMemUpdateAccumulateNoSyncKernel(const IndexT* d_keys, const size_t len, const int8_t* d_values, typename EmbedCacheSA<IndexT, TagT>::CacheData data, uint32_t currTable, size_t stride, DataTypeFormat inputFormat, [[maybe_unused]] DataTypeFormat outputFormat, cudaStream_t stream)
{
    assert(inputFormat == outputFormat);
    const uint32_t blockX = 32;
    const uint32_t blockY = 4;
    const uint32_t blockSize = blockX * blockY;
    const uint32_t nBlock = static_cast<uint32_t>(len / blockSize + std::min(len % blockSize, (size_t)1));
    dim3 gridDims(nBlock);
    dim3 blockDims(blockSize);
    switch (inputFormat)
    {
    case DATATYPE_FP16:
        UpdateAccumulateNoSync<IndexT, TagT, 32, __half><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, data, currTable, stride);
        break;
    case DATATYPE_FP32:
        UpdateAccumulateNoSync<IndexT, TagT, 32, float><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, data, currTable, stride);
        break;
    default:
        //not implemented
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

// need to have an argument explicity depend on IndexT type or the compiler gets confused
template<typename IndexT, typename TagT>
cudaError_t callTagUpdateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEntries, TagT* pTags, cudaStream_t stream)
{
    dim3 gridSize((nEntries + 32 - 1)/32,1);
    dim3 blockSize(32, 1); 
    TagUpdateKernel<IndexT, TagT><<<gridSize, blockSize, 0, stream>>>(pList, pTags);
    return cudaGetLastError();
}

template<typename IndexT, typename TagT>
cudaError_t callCacheQuery(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t currTable, size_t stride)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 4;
    const uint32_t blockSize = blockX * blockY;
    const uint32_t nBlock = static_cast<uint32_t>(len / blockSize + std::min(len % blockSize, (size_t)1));
    dim3 gridDims(nBlock);
    dim3 blockDims(blockSize);
    if (data.rowSizeInBytes % (sizeof(uint4)*32) == 0)
    {
        Query<IndexT, TagT, 32, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    else if (data.rowSizeInBytes % (sizeof(uint32_t)*32) == 0)
    {
        Query<IndexT, TagT, 32, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    else if (data.rowSizeInBytes % (sizeof(uint4)*4) == 0)
    {
        Query<IndexT, TagT, 4, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    else
    {
        Query<IndexT, TagT, blockX, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, currTable, stride);
    }
    return cudaGetLastError();
}

template<typename IndexT, typename TagT>
cudaError_t callCacheQueryHitMask(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint32_t* d_hit_mask,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t currTable, size_t stride)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 4;
    const uint32_t blockSize = blockX * blockY;
    const uint32_t nBlock = static_cast<uint32_t>(len / blockSize + std::min(len % blockSize, (size_t)1));
    dim3 gridDims(nBlock);
    dim3 blockDims(blockX, blockY);
    if (data.rowSizeInBytes % sizeof(uint4) == 0)
    {
        Query<IndexT, TagT, 32, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_hit_mask, data, currTable, stride);
    }
    else if (data.rowSizeInBytes % sizeof(uint32_t) == 0)
    {
        Query<IndexT, TagT, 32, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_hit_mask, data, currTable, stride);
    }
    else
    {
        Query<IndexT, TagT, 32, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_hit_mask, data, currTable, stride);
    }
    return cudaGetLastError();
}

template<uint32_t ROW_SIZE, typename IndexT, int NUM_PRODUCER, uint stages_count, typename DataType, typename TagT>
void __global__ UpdateAccumalateNoSyncFusedWithPipeline(int8_t* global0, int8_t* global1, int8_t* dst, 
                                    const IndexT* indices, 
                                    IndexT num_idx, 
                                    typename EmbedCacheSA<IndexT, TagT>::CacheData data
                                    )
 {
    extern __shared__ int8_t shmem[];
    auto group = cooperative_groups::this_thread_block();
    const uint32_t pipe_buffer_size = calc_pipe_buffer_size(ROW_SIZE);
    uint32_t shared[stages_count];
    // TODO optimize this to be in line calculation instead of using a register
    for (uint32_t i = 0; i < stages_count; i++) 
    {
        shared[i] = i * pipe_buffer_size * NUM_PRODUCER;
    }

    // Create a pipeline.
    constexpr auto scope = cuda::thread_scope_block;
    
    __shared__ cuda::pipeline_shared_state<scope, (uint8_t)stages_count> shared_state;
    const cuda::pipeline_role thread_role
        = group.thread_index().y < NUM_PRODUCER ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    
    auto pipeline = cuda::make_pipeline(group, &shared_state, thread_role);  
    
    group.sync();
    
    if (thread_role == cuda::pipeline_role::producer) 
    {
        int group_block = (group.group_index().x * NUM_PRODUCER + group.thread_index().y) * 32; 
        int tid = group_block + group.thread_index().x;
        uint64_t lane_table_ptr = 0;
        uint64_t cache_ptr = 0;
        uint64_t weight_ptr = 0;
        
        if (tid < num_idx)
        {
            IndexT laneIdx =  indices[tid];

            uint32_t setIdx = laneIdx % data.nSets;
            uint32_t laneout = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, 0, data);
            uint32_t laneway = __ffs(laneout) - 1;
            if (laneout == 0)
            {
                cache_ptr = 0;
                weight_ptr = reinterpret_cast<uint64_t>(global1 + (laneIdx) * ROW_SIZE);
            }
            else
            {
                auto way = laneway;
                cache_ptr = (uint64_t)(data.pCache + (setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes);
                weight_ptr = cache_ptr;   
            }
            lane_table_ptr = reinterpret_cast<uint64_t>(dst + laneIdx * ROW_SIZE);
        }
    
        __syncwarp();
    
        for (int s = 0; s < 32; s++)
        {
            auto j = group.thread_index().y;
            auto w_ptr = __shfl_sync(0xffffffff, weight_ptr, s, 32); 
            auto d_ptr = __shfl_sync(0xffffffff, lane_table_ptr, s, 32); 
            auto c_ptr = __shfl_sync(0xffffffff, cache_ptr, s, 32);
            pipeline.producer_acquire();
            auto curr_writer = s % stages_count;
        
            if (d_ptr)
            {
                    int8_t* src0 = global0 + (group_block + s) * ROW_SIZE + sizeof(DataType)*group.thread_index().x;
                    int8_t* src1 = (int8_t*)w_ptr + sizeof(DataType)*group.thread_index().x;
                
                    cuda::memcpy_async(shmem + shared[curr_writer] + j * calc_pipe_buffer_size(ROW_SIZE) + sizeof(DataType) * group.thread_index().x,
                                        src0 , sizeof(DataType), pipeline);
                    cuda::memcpy_async(shmem + shared[curr_writer] + ROW_SIZE + j * calc_pipe_buffer_size(ROW_SIZE) + sizeof(DataType) * group.thread_index().x,
                                        src1 , sizeof(DataType), pipeline);
                
            }
            if (group.thread_index().x == 0)
            {
                *(uint64_t*)(shmem + shared[curr_writer] + j * calc_pipe_buffer_size(ROW_SIZE) + 2 * ROW_SIZE) = d_ptr;
                *(uint64_t*)(shmem + shared[curr_writer] + j * calc_pipe_buffer_size(ROW_SIZE) + 2 * ROW_SIZE + sizeof(d_ptr)) = c_ptr;
            }
            __syncwarp();
            pipeline.producer_commit();
        }
    }
    else
    {
        int num_works = 32;
        int read_index = 0;
        for (int i = 0; i < num_works; i++)
        {
            __syncwarp();
            pipeline.consumer_wait();
            for (int j = 0; j < NUM_PRODUCER; j++)
            {
                uint64_t d_ptr = *(uint64_t*)(shmem + shared[read_index] + j * calc_pipe_buffer_size(ROW_SIZE) + 2 * ROW_SIZE );
                uint64_t c_ptr = *(uint64_t*)(shmem + shared[read_index] + j * calc_pipe_buffer_size(ROW_SIZE) + 2 * ROW_SIZE + sizeof(d_ptr) );
                if (d_ptr)
                {
                    DataType* src0 = (DataType*)(shmem + shared[read_index] + j * calc_pipe_buffer_size(ROW_SIZE));
                    DataType* src1 = (DataType*)(shmem + shared[read_index] + j * calc_pipe_buffer_size(ROW_SIZE) + ROW_SIZE);
                    DataType s0 = *(src0 + group.thread_index().x);
                    DataType s1 = *(src1 + group.thread_index().x);
                    DataType acc = nve::Add(s0, s1);
                    DataType* r_dst = (DataType*)(d_ptr);
                    DataType* r_c = (DataType*)(c_ptr); 
                    
                    r_dst[group.thread_index().x] = acc;
                    if (r_c)
                    {
                        r_c[group.thread_index().x] = acc;
                    }
                }
            }
            pipeline.consumer_release();
            read_index = (read_index + 1) % stages_count;
        }
    } 
}

template<typename IndexT, typename TagT>
cudaError_t callUpdateAccumalateNoSyncFusedWithPipeline( const int8_t* grads, 
                    int8_t* dst, 
                    const IndexT* keys, 
                    IndexT num_keys,
                    uint32_t row_size_in_bytes,
                    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
                    cudaStream_t stream)
{
    constexpr uint blocY = 16;
    constexpr int stage_count = 2;
    
    dim3 blockDims(32, blocY);
    const auto num_keys_per_block = blockDims.x * (blockDims.y - 1);
    dim3 gridDims(static_cast<uint32_t>((num_keys + num_keys_per_block - 1)/num_keys_per_block));
    size_t shared_mem_size_per_producer = (blocY -1 ) * calc_pipe_buffer_size(row_size_in_bytes);
    size_t shared_mem = shared_mem_size_per_producer * stage_count;
    switch (row_size_in_bytes)
    {
    case 128:
    {
        UpdateAccumalateNoSyncFusedWithPipeline<128, IndexT, blocY - 1, stage_count, float, TagT><<<gridDims, blockDims, shared_mem, stream>>>((int8_t*)grads, dst, dst, keys, num_keys, data);
        break;
    }
    case 512:
    {
        UpdateAccumalateNoSyncFusedWithPipeline<512, IndexT, blocY - 1, stage_count, float4, TagT><<<gridDims, blockDims, shared_mem, stream>>>((int8_t*)grads, dst, dst, keys, num_keys, data);
        break;
    }   
    default:
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

// internal struct for a return type of the find kernel, returns a pair of src dst pointer 
// for a given key.
struct FindOutput
{
    uint64_t src_ptr; // represent a pointer as a unit64_t as it is easier to shuffle around the warp.
    uint64_t dst_ptr;
};

template<typename IndexT, typename SortKeyType>
__device__ SortKeyType get_sort_key(const IndexT& idx, bool hit)
{
    if (std::is_signed<IndexT>::value)
    {
        return hit ? idx : -idx;
    }
    else
    {
        constexpr SortKeyType offset = 512*1024*1024;
        return hit ? (min(std::numeric_limits<IndexT>::max() - offset, idx) + offset) : (idx % offset);
    }
    
}

template<typename IndexT, typename TagT, uint32_t BLOC_Y, typename SortKeyType>
__global__ void find(const int8_t* uvm,
                            int8_t* dst,
                            SortKeyType* miss_buff,
                            FindOutput* out_buff,
                            const IndexT* indices,
                            size_t num_idx,
                            typename EmbedCacheSA<IndexT, TagT>::CacheData data)
{
    uint32_t tid_batch = blockIdx.x * 32 * BLOC_Y + threadIdx.y * 32;
    uint32_t tid = tid_batch + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    uint64_t laneptr;
    if (tid >= num_idx)
    {
        laneptr = 0;
    }
    else
    {
        auto currTable = 0;
        IndexT laneIdx = indices[tid];
        uint32_t cacheOffset = currTable * data.nSets * EmbedCacheSA<IndexT, TagT>::NUM_WAYS;
        uint32_t setIdx = laneIdx % data.nSets;
        uint32_t laneout = EmbedCacheGetWayMask<IndexT, TagT>(laneIdx, currTable, data);
        uint32_t laneway = __ffs(laneout) - 1;
        FindOutput out;
        if (laneout == 0)
        {
            if (data.bCountMisses)
            {
                atomicAdd((unsigned long long*)data.misses, 1);
            }
            miss_buff[tid] = get_sort_key<IndexT, SortKeyType>(laneIdx, false);
            out.src_ptr =(uint64_t)(uvm + laneIdx * (uint64_t)data.rowSizeInBytes);
        }
        else
        {
            auto way = laneway;
            out.src_ptr = (uint64_t)(data.pCache + (cacheOffset + setIdx * EmbedCacheSA<IndexT, TagT>::NUM_WAYS + way)*(uint64_t)data.rowSizeInBytes); 
            miss_buff[tid] = get_sort_key<IndexT, SortKeyType>(laneIdx, true);
        }
        out.dst_ptr = (uint64_t)(dst + tid * (uint64_t)data.rowSizeInBytes);
        out_buff[tid] = out;
    }
    
}

template<typename IndexT, typename DataType, uint32_t BLOCK_Y, uint32_t UNROLL, uint32_t SUBWARP_SIZE>
__global__ void gather(const FindOutput* out_buff, size_t num_idx, uint32_t KEYS_PER_Y, uint32_t row_size_in_bytes)
{
    
    auto blockId = blockIdx.x;
    uint32_t tid_batch = blockId * KEYS_PER_Y * BLOCK_Y + threadIdx.y * KEYS_PER_Y;
    auto base_output = out_buff + tid_batch;
    
    if (tid_batch < num_idx)
    {
        auto rem_outer = num_idx - tid_batch;
        auto num_keys_for_warp = min(rem_outer, static_cast<size_t>(KEYS_PER_Y));
        for (uint32_t s = 0; s < num_keys_for_warp; s += SUBWARP_SIZE)
        {
            if (tid_batch + s + SUBWARP_SIZE <= num_idx)
            {
                uint64_t l_src_ptr = 0;
                uint64_t l_dst_ptr = 0;
                if (threadIdx.x < SUBWARP_SIZE)
                {
                    //uint4 output_ = *(uint4*)(base_output + s + threadIdx.x);
                    //FindOutput output = *(FindOutput*)&output_;
                    FindOutput output = base_output[s + threadIdx.x];
                    l_src_ptr = output.src_ptr;
                    l_dst_ptr = output.dst_ptr;
                }
                __syncwarp();
                for (uint32_t ii = 0; ii < row_size_in_bytes; ii += SUBWARP_SIZE*sizeof(DataType)) 
                {
                    uint32_t offset = ii + threadIdx.x * sizeof(DataType);        
                    for (uint32_t i = 0; i < SUBWARP_SIZE; i+= UNROLL)
                    {
                        DataType tt[UNROLL];
                        #pragma unroll UNROLL
                        for (uint32_t j = 0; j < UNROLL; j++)
                        {
                            auto src_ptr_ = __shfl_sync(0xffffffff, l_src_ptr, i+j, SUBWARP_SIZE); 
                            if (offset < row_size_in_bytes)
                            {
                                DataType d = *(DataType*)(src_ptr_ + offset);
                                tt[j] = d;
                            }
                        }

                        #pragma unroll UNROLL
                        for (uint32_t j = 0; j < UNROLL; j++)
                        {
                            auto dst_ptr_ = __shfl_sync(0xffffffff, l_dst_ptr, i+j, SUBWARP_SIZE);    
                            if (offset < row_size_in_bytes)
                            {
                                DataType* dst_ptr = (DataType*)(dst_ptr_ + offset);
                                *dst_ptr = tt[j];
                            }
                            
                        }        
                    }
                }
            }
            else
            {
                auto rem = num_idx - (tid_batch + s);
                for (uint32_t i = 0; i < rem; i++)
                {
                    FindOutput output = base_output[s+i];
                    uint64_t src_ptr_ = output.src_ptr;
                    uint64_t dst_ptr_ = output.dst_ptr;
                    
                    for (uint32_t ii = 0; ii < row_size_in_bytes; ii += SUBWARP_SIZE*sizeof(DataType)) 
                    {
                        uint32_t offset = ii + threadIdx.x * sizeof(DataType);
                        if (offset < row_size_in_bytes)
                        {
                            DataType d = *(DataType*)(src_ptr_ + offset);
                            DataType* dst_ptr = (DataType*)(dst_ptr_ + offset);
                            *dst_ptr = d;
                        }
                    }
                }
            }
        }
    }
}

template<typename IndexT, typename TagT>
cudaError_t callSortGather( const int8_t* uvm, 
                    int8_t* dst, 
                    const IndexT* keys, 
                    int8_t* auxBuf,
                    size_t& auxBufBytes,
                    size_t num_keys,
                    size_t row_size_in_bytes,
                    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
                    cudaStream_t stream)
{
    using SortKeyType = IndexT;
    cudaError_t err = cudaSuccess;
    SortKeyType* sortKeyBuf = nullptr;
    FindOutput* findBuf = nullptr;
    SortKeyType* sortKeySortedBuf = nullptr;
    FindOutput* findSortedBuf = nullptr;
    size_t cubAuxBytes = 0;
    err = cub::DeviceRadixSort::SortPairs(nullptr, cubAuxBytes,
                    sortKeyBuf, sortKeyBuf, findBuf, findBuf, num_keys);
    if (err != cudaSuccess)
    {
        return err;
    }
    
    size_t findBytes = sizeof(FindOutput)*num_keys;
    size_t sortKeyBytes = sizeof(SortKeyType)*num_keys;
    size_t findSortedBytes = findBytes;
    size_t sortKeySortedBytes = sortKeyBytes;

    if (auxBuf == nullptr)
    {
        auxBufBytes = cubAuxBytes + findBytes + sortKeyBytes + findSortedBytes + sortKeySortedBytes; 
        return cudaSuccess;
    }

    sortKeyBuf = reinterpret_cast<SortKeyType*>(auxBuf);
    sortKeySortedBuf = reinterpret_cast<SortKeyType*>(auxBuf + sortKeyBytes);
    findBuf = reinterpret_cast<FindOutput*>(auxBuf + sortKeyBytes + sortKeySortedBytes);
    findSortedBuf = reinterpret_cast<FindOutput*>(auxBuf + sortKeyBytes + sortKeySortedBytes + findBytes);
    int8_t* cubAuxBuf = auxBuf + findBytes + sortKeyBytes + findSortedBytes + sortKeySortedBytes;

    constexpr auto num_keys_per_block = 32*2;
    dim3 blockDims(32, 2);
    dim3 gridDims(static_cast<uint32_t>((num_keys + num_keys_per_block - 1)/num_keys_per_block));

    find<IndexT, TagT, 2, SortKeyType><<<gridDims, blockDims, 0, stream>>>(uvm,
                            dst,
                            sortKeyBuf,
                            findBuf,
                            keys,
                            num_keys,
                            data);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }
    
    err =cub::DeviceRadixSort::SortPairs(cubAuxBuf, cubAuxBytes,
            sortKeyBuf, sortKeySortedBuf, findBuf, findSortedBuf, num_keys, 0, sizeof(SortKeyType)*8, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    
    constexpr auto block_y = 4;
    dim3 blockDims2(32, block_y);
    constexpr auto num_keys_per_y = 4096;
    constexpr auto num_keys_per_block_gather = num_keys_per_y*block_y;
    constexpr auto unroll = 8;
    dim3 gridDims2(static_cast<uint32_t>((num_keys + num_keys_per_block_gather - 1)/num_keys_per_block_gather));
    if (row_size_in_bytes % 16 == 0)
    {
        
        gather<IndexT, uint4, block_y, unroll, 32><<<gridDims2, blockDims2, 0, stream>>>(findSortedBuf, num_keys, num_keys_per_y, static_cast<uint32_t>(row_size_in_bytes));
        return cudaGetLastError();
    }
    else if (row_size_in_bytes % 4 == 0)
    {
        gather<IndexT, uint, block_y, unroll, 32><<<gridDims2, blockDims2, 0, stream>>>(findSortedBuf, num_keys, num_keys_per_y, static_cast<uint32_t>(row_size_in_bytes));
        return cudaGetLastError();
    }
    else
    {
        gather<IndexT, uint8_t, block_y, unroll, 32><<<gridDims2, blockDims2, 0, stream>>>(findSortedBuf, num_keys, num_keys_per_y, static_cast<uint32_t>(row_size_in_bytes));
        return cudaGetLastError();
    }
    
}

}
