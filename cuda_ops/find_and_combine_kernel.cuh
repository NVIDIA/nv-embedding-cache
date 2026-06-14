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

    constexpr auto vecSizeInElements = sizeof(ELEMENT_VEC_TYPE) / sizeof(ELEMENT_TYPE);
    int32_t rowSizeInVecs = rowSizeInElements / vecSizeInElements;
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
                    if (offset < static_cast<uint32_t>(rowSizeInVecs))
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
    constexpr int32_t SUBWARP_WIDTH = 32;

    // Each thread holds SZ_ACCUM vector accumulators; with SUBWARP_WIDTH (=32) threads per row
    // this covers up to SUBWARP_WIDTH * SZ_ACCUM * vecSizeInElements row elements. Raising the
    // SZ_ACCUM cap costs register pressure / kernel-variant count, so we cap each vector-width path
    // independently. Current caps cover up to 1024 fp32 / 2048 fp16 row elements on the Vec4 path,
    // which is enough for common embedding dimensions.
#define NVE_FAC_LAUNCH(SZ_ACC_RUNTIME, ELEMENT_VEC_T, ACC_VEC_T) \
    FindAndCombine<ELEMENT_TYPE, INDEX_TYPE, ACC_TYPE, \
                   ELEMENT_VEC_T, ACC_VEC_T, CacheDataT, \
                   (SZ_ACC_RUNTIME), FIXED_HOTNESS, SUM_POOLING, IS_WEIGHTED> \
        <<<gridSize, blockSize, 0, stream>>>(batchSz, table, indices, offsets, weights, \
                                             num_hot, cache, rowSizeInElements, output)

#define NVE_FAC_DISPATCH(SZ_ACC_RUNTIME, ELEMENT_VEC_T, ACC_VEC_T, MAX_SZ)                          \
    do {                                                                                        \
        switch (SZ_ACC_RUNTIME) {                                                                   \
          case 1: NVE_FAC_LAUNCH(1, ELEMENT_VEC_T, ACC_VEC_T); break;                           \
          case 2: NVE_FAC_LAUNCH(2, ELEMENT_VEC_T, ACC_VEC_T); break;                           \
          case 3: if constexpr ((MAX_SZ) >= 3) { NVE_FAC_LAUNCH(3, ELEMENT_VEC_T, ACC_VEC_T); } \
                  else { NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME); } break;     \
          case 4: if constexpr ((MAX_SZ) >= 4) { NVE_FAC_LAUNCH(4, ELEMENT_VEC_T, ACC_VEC_T); } \
                  else { NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME); } break;     \
          case 5: if constexpr ((MAX_SZ) >= 5) { NVE_FAC_LAUNCH(5, ELEMENT_VEC_T, ACC_VEC_T); } \
                  else { NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME); } break;     \
          case 6: if constexpr ((MAX_SZ) >= 6) { NVE_FAC_LAUNCH(6, ELEMENT_VEC_T, ACC_VEC_T); } \
                  else { NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME); } break;     \
          case 7: if constexpr ((MAX_SZ) >= 7) { NVE_FAC_LAUNCH(7, ELEMENT_VEC_T, ACC_VEC_T); } \
                  else { NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME); } break;     \
          case 8: if constexpr ((MAX_SZ) >= 8) { NVE_FAC_LAUNCH(8, ELEMENT_VEC_T, ACC_VEC_T); } \
                  else { NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME); } break;     \
          default: NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME);                    \
        }                                                                                       \
    } while (0)

    switch (rowSizeInElements % 4) {
      case 0:
          {
            const uint32_t SZ_ACCUM = DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 4);
            using ELEMENT_VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec4;
            using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec4;
            NVE_FAC_DISPATCH(SZ_ACCUM, ELEMENT_VEC_TYPE, ACC_VEC_TYPE, 8);
            break;
          }
      case 2:
          {
            const uint32_t SZ_ACCUM = DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 2);
            using ELEMENT_VEC_TYPE = typename VecWidthHelper<ELEMENT_TYPE>::Vec2;
            using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec2;
            NVE_FAC_DISPATCH(SZ_ACCUM, ELEMENT_VEC_TYPE, ACC_VEC_TYPE, 8);
            break;
          }
      default:
          {
            const uint32_t SZ_ACCUM = DivRoundUp(rowSizeInElements, SUBWARP_WIDTH);
            NVE_FAC_DISPATCH(SZ_ACCUM, ELEMENT_TYPE, ACC_TYPE, 8);
            break;
          }
    }
#undef NVE_FAC_DISPATCH
#undef NVE_FAC_LAUNCH
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}
