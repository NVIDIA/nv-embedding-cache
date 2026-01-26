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
#include <embedding_cache_combined.cuh>
#include "cuda_ops/kernels_common.cuh"
#include "cuda_ops/cuda_common.h"

using namespace nve;

#define MASK(x) ((1u << x) - 1)
#define CASE_MASK(x) \
    case MASK(x): \
        callEmbeddingBagKernelFixedMask<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, Cache, MASK(x)>(batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, tags, rowSizeInBytes / 4, output, num_sets, misses); \
        break;

template<typename VECTOR_TYPE, typename INDEX_TYPE, uint32_t ElEMENT_PER_LANE, uint32_t SZ_ACCUM, bool COUNT_MISSES, 
    uint32_t Y_BLOCK, uint32_t PARTIAL_UNROLL, typename CacheDataT, uint32_t SUBWARP_WIDTH>
__global__ void EmbeddingBagMain(const uint32_t batchSz, const INDEX_TYPE* __restrict__ indices, int32_t num_hot, const int8_t* const* __restrict__ table, CacheDataT cache, int32_t rowSizeInElements, VECTOR_TYPE* pOutput)
{
    constexpr size_t BATCH_PER_WARP = 32 / SUBWARP_WIDTH;
    const size_t num_batch = gridDim.x * Y_BLOCK * BATCH_PER_WARP;

    const uint32_t hotnessTid = threadIdx.x % SUBWARP_WIDTH;
    uint32_t currTable = blockIdx.y;
    uint32_t currBatch = (blockIdx.x * Y_BLOCK * BATCH_PER_WARP  + threadIdx.y * BATCH_PER_WARP + threadIdx.x / SUBWARP_WIDTH);

    // this is not good as all warp threads participate in address compute
    if (currBatch >= batchSz) {
        return;
    }

    int32_t rowSizeInVecs = rowSizeInElements / ElEMENT_PER_LANE;
    
    VECTOR_TYPE acc[SZ_ACCUM];

    for (uint32_t k = 0; k < SZ_ACCUM; k++)
    {
        InitAcc(acc[k]);
    }

    const INDEX_TYPE* currIdx = indices + currBatch * num_hot + currTable * num_hot * num_batch;

    for (int i = 0; i < num_hot / SUBWARP_WIDTH; i++)
    {
        INDEX_TYPE laneIdx = __ldcv(currIdx + i * SUBWARP_WIDTH + hotnessTid );
        uint64_t lanePtr = AddressFunctor<INDEX_TYPE, CacheDataT>::get_address(laneIdx, table[currTable], currTable, cache);
        #pragma unroll 4
        for (int s = 0; s < SUBWARP_WIDTH; s++)
        {
            uint64_t ptr = __shfl_sync(0xffffffff, lanePtr, s, SUBWARP_WIDTH);
            for (uint32_t k = 0; k < SZ_ACCUM; k++)
            {
                uint32_t offset = hotnessTid + k * SUBWARP_WIDTH;
                if ( offset < rowSizeInElements)
                {
                    VECTOR_TYPE tempOut = *((VECTOR_TYPE*)ptr + offset);              
                    Accumulate(acc[k], tempOut);
                }
            }
        }
    }

    VECTOR_TYPE* currOut = pOutput + currBatch * rowSizeInVecs + currTable * num_batch * rowSizeInVecs + hotnessTid;

    for (int j = 0, k = 0; k < SZ_ACCUM; /*j < rowSizeInVecs*/ j += SUBWARP_WIDTH, k++)
    {
        if (hotnessTid + j < rowSizeInVecs)
        {
            currOut[j] = acc[k];
        }
    }
}

template<bool COUNT_MISSES, uint32_t Y_BLOCK, uint32_t PARTIAL_UNROLL, class CacheDataT, uint32_t SUBWARP_WIDTH,
         uint32_t ElEMENT_PER_LANE, typename ELEMENT_TYPE, typename INDEX_TYPE>
void callEmbeddingBagKernelFixedMaskSubwarp(uint32_t batchSz, uint32_t nTables, uint32_t rowSizeInBytes,  
    const INDEX_TYPE* indices, int32_t num_hot, const int8_t* const* table, CacheDataT cache, size_t numElements, ELEMENT_TYPE* output, cudaStream_t stream)
{
    uint32_t batchPerWarp = 32 / SUBWARP_WIDTH;
    uint32_t batchSzRounded = DivRoundUp(batchSz, Y_BLOCK * batchPerWarp) * Y_BLOCK * batchPerWarp;
    dim3 gridSize (batchSzRounded/(Y_BLOCK * batchPerWarp), nTables);
    dim3 blockSize (32, Y_BLOCK);

    uint32_t SZ_ACCUM = DivRoundUp(rowSizeInBytes , (sizeof(ELEMENT_TYPE) * SUBWARP_WIDTH));
    switch (SZ_ACCUM)
    {
    case 2:
        EmbeddingBagMain<ELEMENT_TYPE, INDEX_TYPE, ElEMENT_PER_LANE, 2, COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, SUBWARP_WIDTH>
            <<<gridSize, blockSize, 0, stream>>>(batchSz, indices, num_hot, table, cache, static_cast<int32_t>(numElements), output);
        break;
    case 1:
        EmbeddingBagMain<ELEMENT_TYPE, INDEX_TYPE, ElEMENT_PER_LANE, 1, COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, SUBWARP_WIDTH>
            <<<gridSize, blockSize, 0, stream>>>(batchSz, indices, num_hot, table, cache, static_cast<int32_t>(numElements), output);
        break;
    default:
        assert(0);
    }
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename ELEMENT_TYPE, typename INDEX_TYPE, bool COUNT_MISSES, uint32_t Y_BLOCK, uint32_t PARTIAL_UNROLL, class CacheDataT>
void callEmbeddingBagKernel(uint32_t batchSz, uint32_t nTables, uint32_t rowSizeInBytes,
        const INDEX_TYPE* indices, int32_t num_hot, const int8_t* const* table, CacheDataT cache, ELEMENT_TYPE* output, cudaStream_t stream)
{
    uint32_t subgroupWidth = std::min(nextPow2(rowSizeInBytes / sizeof(ELEMENT_TYPE)), 32u);
    constexpr auto elemSize = sizeof(output[0]);
    const auto numElements = rowSizeInBytes / elemSize;
    switch (subgroupWidth)
    {
    case 32:
        {
        using VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec4;
        callEmbeddingBagKernelFixedMaskSubwarp<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, 32, 4, VEC_TYPE, INDEX_TYPE>(
            batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, numElements,
            reinterpret_cast<VEC_TYPE*>(output), stream);
        }
        break;
    case 16:
        {
        using VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec4;
        callEmbeddingBagKernelFixedMaskSubwarp<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, 16, 4, VEC_TYPE, INDEX_TYPE>(
            batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, numElements,
            reinterpret_cast<VEC_TYPE*>(output), stream);
        }
        break;
    case 8:
        {
        using VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec4;
        callEmbeddingBagKernelFixedMaskSubwarp<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, 8, 4, VEC_TYPE, INDEX_TYPE>(
            batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, numElements,
            reinterpret_cast<VEC_TYPE*>(output), stream);
        }
        break;
    case 4:
        using VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec4;
        callEmbeddingBagKernelFixedMaskSubwarp<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, 4, 4, VEC_TYPE, INDEX_TYPE>(
            batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, numElements,
            reinterpret_cast<VEC_TYPE*>(output), stream);
        break;
    case 2:
        {
        using VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec2;
        callEmbeddingBagKernelFixedMaskSubwarp<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, 2, 2, VEC_TYPE, INDEX_TYPE>(
            batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, numElements,
            reinterpret_cast<VEC_TYPE*>(output), stream);
        }
        break;
    case 1:
        {
        using VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec1;
        callEmbeddingBagKernelFixedMaskSubwarp<COUNT_MISSES, Y_BLOCK, PARTIAL_UNROLL, CacheDataT, 1, 1, VEC_TYPE, INDEX_TYPE>(
            batchSz, nTables, rowSizeInBytes, indices, num_hot, table, cache, numElements,
            reinterpret_cast<VEC_TYPE*>(output), stream);
        }
        break;
    default:
        assert(0);
    }
}