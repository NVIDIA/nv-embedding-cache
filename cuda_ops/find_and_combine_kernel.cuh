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
#include <ecache/embedding_cache_combined.h>
#include "kernels_common.cuh"
#include "cuda_ops/cuda_common.h"

using namespace nve;

template<typename ELEMENT_TYPE, typename INDEX_TYPE, typename ACC_TYPE,
         typename ELEMENT_VEC_TYPE, typename ACC_VEC_TYPE, typename CacheDataT, 
         uint32_t SZ_ACCUM, bool FIXED_HOTNESS, bool SUM_POOLING, bool IS_WEIGHTED>
__launch_bounds__(128, 1)
__global__ void FindAndCombine(const uint32_t batchSz,
                               const int8_t* __restrict__ table,
                               const INDEX_TYPE* __restrict__ indices, 
                               const INDEX_TYPE* __restrict__ offsets,
                               const ELEMENT_TYPE* __restrict__ weights,
                               int32_t num_hot,  CacheDataT cache, 
                               int32_t rowSizeInElements,
                               ELEMENT_TYPE* pOutput)
{
    const int32_t sampleId = blockIdx.x;
    constexpr int32_t SUBWARP_WIDTH = 32;

    uint32_t hotness = num_hot;
    if constexpr (!FIXED_HOTNESS) {
        hotness = offsets[sampleId + 1] - offsets[sampleId];
    }

    const uint32_t hotnessTid = threadIdx.x;
    
    ACC_VEC_TYPE acc[SZ_ACCUM];
    ACC_TYPE acc_weight[SZ_ACCUM];

    for (uint32_t k = 0; k < SZ_ACCUM; k++)
    {
        InitAcc(acc[k]);
        if (!SUM_POOLING) {
          InitAcc(acc_weight[k]);
        }
    }

    const INDEX_TYPE sampleStart = FIXED_HOTNESS ? sampleId * hotness :  offsets[sampleId];
    const INDEX_TYPE sampleEnd = sampleStart + hotness;
    const INDEX_TYPE* currIdx = indices + sampleStart;

    for (int i = 0; i < hotness; i += SUBWARP_WIDTH)
    {
        // TODO: should be conditional load
        INDEX_TYPE laneIdx = ((i + hotnessTid) < hotness) ? __ldcv(currIdx + i + hotnessTid) : 0;
        uint64_t lanePtr = AddressFunctor<INDEX_TYPE, CacheDataT>::get_address(laneIdx, table, 0, cache);
        #pragma unroll 4
        for (int s = 0; s < SUBWARP_WIDTH; s++)
        {
            uint64_t ptr = __shfl_sync(0xffffffff, lanePtr, s, SUBWARP_WIDTH);
            if ((sampleStart + i + s) < sampleEnd) {
                for (uint32_t k = 0; k < SZ_ACCUM; k++)
                {
                    uint32_t offset = hotnessTid + k * SUBWARP_WIDTH;
                    if ( offset < rowSizeInElements)
                    {
                        ELEMENT_VEC_TYPE el_raw = *((ELEMENT_VEC_TYPE*)ptr + offset);
                        ACC_VEC_TYPE el_cast = Cast<ELEMENT_VEC_TYPE, ACC_VEC_TYPE>(el_raw);
                        if (SUM_POOLING) {
                            if (IS_WEIGHTED) {
                                ELEMENT_TYPE weight_raw = weights[sampleStart + i + s];
                                ACC_TYPE weight_cast = weight_raw;
                                
				                MulAccumulate(acc[k], el_cast, weight_cast);
                            } else {
                                Accumulate(acc[k], el_cast);
                            }
                        } else {
                            if (IS_WEIGHTED) {
                                ACC_TYPE weight_cast = weights[sampleStart + i + s];
                                MulAccumulate(acc[k], el_cast, weight_cast);
                                Accumulate(acc_weight[k], weight_cast);
                            } else {
                                Accumulate(acc[k], el_cast);
                                Accumulate(acc_weight[k], ACC_TYPE(1.0));
                            }
                        }         
                    }
                }
            }
        }
    }

    ELEMENT_VEC_TYPE* currOut = reinterpret_cast<ELEMENT_VEC_TYPE*>(pOutput + sampleId * rowSizeInElements);

    // TODO: try loop on acc size
    constexpr auto vecSizeInElements = sizeof(ELEMENT_VEC_TYPE) / sizeof(ELEMENT_TYPE);
    int32_t rowSizeInVecs = rowSizeInElements / vecSizeInElements;

    for (int32_t j = hotnessTid, k = 0; j < rowSizeInVecs; j += SUBWARP_WIDTH, ++k)
    {
        if (!SUM_POOLING) {
            // TODO: replace by inverse and mul?
            Div(acc[k], acc_weight[k]);
        }
        currOut[j] = Cast<ACC_VEC_TYPE, ELEMENT_VEC_TYPE>(acc[k]);
    }
}

template<typename ELEMENT_TYPE, typename INDEX_TYPE, typename ACC_TYPE, typename CacheDataT, 
         bool FIXED_HOTNESS, bool SUM_POOLING, bool IS_WEIGHTED>
void callFindAndCombineKernel(const uint32_t batchSz,
                             const int8_t* __restrict__ table,
                             const INDEX_TYPE* __restrict__ indices, 
                             const INDEX_TYPE* __restrict__ offsets,
                             const ELEMENT_TYPE* __restrict__ weights,
                             int32_t num_hot,  CacheDataT cache, 
                             int32_t rowSizeInElements,
                             ELEMENT_TYPE* output,
                             cudaStream_t stream)
{
    dim3 gridSize (batchSz, 1);
    dim3 blockSize (32, 1);

    switch (rowSizeInElements % 4) {
      case 0:
          {
            const int32_t SUBWARP_WIDTH = 32;
            uint32_t SZ_ACCUM = DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 4);
            using ELEMENT_VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec4;
            using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec4;

            switch (SZ_ACCUM)
            {
              case 2:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_VEC_TYPE, ACC_VEC_TYPE, CacheDataT,
                            2, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
              case 1:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_VEC_TYPE, ACC_VEC_TYPE, CacheDataT,
                            1, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
              default:
                assert(0);
            }
            break;
          }
      case 2:
          {
            const int32_t SUBWARP_WIDTH = 32;
            uint32_t SZ_ACCUM = DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 2);
            using ELEMENT_VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec2;
            using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec2;
            switch (SZ_ACCUM)
            {
              case 4:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_VEC_TYPE, ACC_VEC_TYPE, CacheDataT,
                            4, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
              case 2:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_VEC_TYPE, ACC_VEC_TYPE, CacheDataT,
                            2, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
              case 1:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_VEC_TYPE, ACC_VEC_TYPE, CacheDataT,
                            1, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
              default:
                assert(0);
            }
            break;
          }
          break;
      default:
          {
            const int32_t SUBWARP_WIDTH = 32;
            uint32_t SZ_ACCUM = DivRoundUp(rowSizeInElements, SUBWARP_WIDTH);

            switch (SZ_ACCUM)
            {
            case 8:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            8, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 7:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            7, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 6:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            6, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 5:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            5, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 4:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            4, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 3:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            3, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 2:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            2, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            case 1:
                FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE,
                            ELEMENT_TYPE, ACC_TYPE, CacheDataT,
                            1, FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED>
                    <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, num_hot, cache, static_cast<int32_t>(rowSizeInElements), output);
                break;
            default:
                assert(0);
            }
          }
    }
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}
