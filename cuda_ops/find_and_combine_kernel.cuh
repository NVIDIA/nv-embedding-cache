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

#include <algorithm>
#include <type_traits>
#include <ecache/embedding_cache_combined.h>
#include <ecache/embedding_cache_combined.cuh>
#include "kernels_common.cuh"
#include "cuda_ops/cuda_common.h"

using namespace nve;

// Bit layout for the packed find_and_combine MASK template parameter. The kernel unpacks these and
// the dispatch packs them; cmake/gen_find_and_combine_instantiations.py enumerates MASK with this
// same layout, so all three must stay in sync. Fixed vs CSR hotness and index loading are always
// passed as kernel arguments too. With NVE_FAC_SPECIALIZE_FLAGS (set unless the CMake option
// NVE_OPTIMIZE_BIN_SIZE is ON) they are
// also MASK bits, so each combination gets its own branch-free kernel; without it MASK only has the
// pooling bits and the kernel branches on the arguments (4x fewer kernels, smaller library).
#ifndef NVE_FAC_SPECIALIZE_FLAGS
#define NVE_FAC_SPECIALIZE_FLAGS 0
#endif
constexpr uint32_t NVE_FAC_MASK_SUM_POOLING   = 1u << 0;
constexpr uint32_t NVE_FAC_MASK_IS_WEIGHTED   = 1u << 1;
constexpr uint32_t NVE_FAC_MASK_FIXED_HOTNESS = 1u << 2;  // NVE_FAC_SPECIALIZE_FLAGS only
constexpr uint32_t NVE_FAC_MASK_LOAD_INDICES  = 1u << 3;  // NVE_FAC_SPECIALIZE_FLAGS only

// Unweighted kernels never read the weights, so they are instantiated and launched with a float
// weight type whatever the caller's weight type is (one kernel instead of one per weight type).
// Keep in sync with UNWEIGHTED_WEIGHT_TYPE in cmake/gen_find_and_combine_instantiations.py.
template<typename WEIGHT_TYPE, uint32_t MASK>
using nve_fac_weight_t = std::conditional_t<(MASK & NVE_FAC_MASK_IS_WEIGHTED) != 0u, WEIGHT_TYPE, float>;

template<DataType_t ELEMENT_TYPE_ID, typename INDEX_TYPE, typename ACC_TYPE, typename WEIGHT_TYPE,
         typename OUTPUT_TYPE, typename INPUT_VEC_TYPE, typename OUT_VEC_TYPE,
         typename ELEMENT_VEC_TYPE, typename ACC_VEC_TYPE, typename CacheDataT,
         uint32_t SZ_ACCUM, uint32_t MASK>
__launch_bounds__(128, 1)
__global__ void find_and_combine(const uint32_t batchSz,
                               const int8_t* __restrict__ table,
                               const INDEX_TYPE* __restrict__ indices,
                               const INDEX_TYPE* __restrict__ offsets,
                               const WEIGHT_TYPE* __restrict__ weights,
                               int32_t num_hot,  CacheDataT cache,
                               int32_t rowSizeInElements,
                               OUTPUT_TYPE* pOutput,
                               bool is_fixed_hotness_arg,
                               bool load_indices_arg)
{
    constexpr bool SUM_POOLING = (MASK & NVE_FAC_MASK_SUM_POOLING) != 0u;
    constexpr bool IS_WEIGHTED = (MASK & NVE_FAC_MASK_IS_WEIGHTED) != 0u;
    // Compile-time constants in specialized builds, so the branches below fold away.
    const bool is_fixed_hotness =
        NVE_FAC_SPECIALIZE_FLAGS ? (MASK & NVE_FAC_MASK_FIXED_HOTNESS) != 0u : is_fixed_hotness_arg;
    const bool load_indices =
        NVE_FAC_SPECIALIZE_FLAGS ? (MASK & NVE_FAC_MASK_LOAD_INDICES) != 0u : load_indices_arg;
    using ELEMENT_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType;
    const int32_t sampleId = blockIdx.x;
    constexpr int32_t SUBWARP_WIDTH = 32;

    uint32_t hotness = num_hot;
    INDEX_TYPE sampleStart = sampleId * hotness;
    if (!is_fixed_hotness) {
        sampleStart = offsets[sampleId];
        hotness = offsets[sampleId + 1] - sampleStart;
    }

    const uint32_t hotnessTid = threadIdx.x;
    
    ACC_VEC_TYPE acc[SZ_ACCUM];
    ACC_TYPE acc_weight[SZ_ACCUM];

    constexpr auto vecSizeInElements = sizeof(ELEMENT_VEC_TYPE) / sizeof(ELEMENT_TYPE);
    int32_t rowSizeInVecs = rowSizeInElements / vecSizeInElements;
    const INDEX_TYPE sampleEnd = sampleStart + hotness;
    const INDEX_TYPE* currIdx = indices + sampleStart;
    for (int base_offset = 0; base_offset < rowSizeInVecs; base_offset += SUBWARP_WIDTH * SZ_ACCUM) {
      for (uint32_t k = 0; k < SZ_ACCUM; k++)
      {
          InitAcc(acc[k]);
          if (!SUM_POOLING) {
            InitAcc(acc_weight[k]);
          }
      }
      for (int i = 0; i < hotness; i += SUBWARP_WIDTH)
      {
          // TODO: should be conditional load
          INDEX_TYPE laneIdx = ((i + hotnessTid) < hotness)
              ? (load_indices ? __ldcv(currIdx + i + hotnessTid) : (sampleStart + i + hotnessTid))
              : 0;
          uint64_t lanePtr = AddressFunctor<INDEX_TYPE, CacheDataT>::get_address(laneIdx, table, 0, cache);
          #pragma unroll 4
          for (int s = 0; s < SUBWARP_WIDTH; s++)
          {
              uint64_t ptr = __shfl_sync(0xffffffff, lanePtr, s, SUBWARP_WIDTH);
              ELEMENT_TYPE qscale = 1;
              ELEMENT_TYPE qoffset = 0;
              if (QuantizationHelper<ELEMENT_TYPE_ID>::has_scale) {
                  memcpy(&qscale, (const char*)(ptr) + rowSizeInElements * QuantizationHelper<ELEMENT_TYPE_ID>::element_size, sizeof(ELEMENT_TYPE));
              }
              if (QuantizationHelper<ELEMENT_TYPE_ID>::has_offset) {
                  memcpy(&qoffset, (const char*)(ptr) + rowSizeInElements * QuantizationHelper<ELEMENT_TYPE_ID>::element_size + sizeof(ELEMENT_TYPE), sizeof(ELEMENT_TYPE));
              }

              if ((sampleStart + i + s) < sampleEnd) {
                  for (uint32_t k = 0; k < SZ_ACCUM; k++)
                  {
                      uint32_t offset = base_offset + hotnessTid + k * SUBWARP_WIDTH;
                      if (offset < static_cast<uint32_t>(rowSizeInVecs))
                      {
                          INPUT_VEC_TYPE el_quantized = *((INPUT_VEC_TYPE*)ptr + offset);
                          ELEMENT_VEC_TYPE el_raw = dequantize<nve::QuantizationHelper<ELEMENT_TYPE_ID>, ELEMENT_VEC_TYPE, INPUT_VEC_TYPE>(el_quantized, qscale, qoffset);
                          ACC_VEC_TYPE el_cast = Cast<ELEMENT_VEC_TYPE, ACC_VEC_TYPE>(el_raw);
                          if (SUM_POOLING) {
                              if (IS_WEIGHTED) {
                                  WEIGHT_TYPE weight_raw = weights[sampleStart + i + s];
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

      OUT_VEC_TYPE* currOut = reinterpret_cast<OUT_VEC_TYPE*>(pOutput + sampleId * rowSizeInElements);

      // TODO: try loop on acc size
      for (int32_t j = hotnessTid + base_offset, k = 0; k < SZ_ACCUM; j += SUBWARP_WIDTH, ++k)
      {
          if (j >= rowSizeInVecs) {
            break;
          }
          if (!SUM_POOLING) {
              // Match the CPU pooling kernels: a zero averaging denominator (an empty bag, or
              // weights summing to zero) yields a zero output row instead of dividing by zero.
              if (IsZero(acc_weight[k])) {
                  InitAcc(acc[k]);
              } else {
                  // TODO: replace by inverse and mul?
                  Div(acc[k], acc_weight[k]);
              }
          }
          currOut[j] = Cast<ACC_VEC_TYPE, OUT_VEC_TYPE>(acc[k]);
      }
    }
}

// Pointer to any find_and_combine instantiation with these parameter types.
template<typename INDEX_TYPE, typename WEIGHT_TYPE, typename OUTPUT_TYPE, typename CacheDataT>
using nve_fac_kernel_t = void (*)(uint32_t, const int8_t*, const INDEX_TYPE*, const INDEX_TYPE*,
                                  const WEIGHT_TYPE*, int32_t, CacheDataT, int32_t, OUTPUT_TYPE*,
                                  bool, bool);

// Address of one find_and_combine combination. Kept as a plain host function template so it can be
// explicitly instantiated in a separate translation unit and linked without relocatable device
// code: taking the kernel's address instantiates its device code in whichever TU instantiates this.
// The generated inst_*.cu files provide the definitions; the generated find_and_combine_dispatch.cu
// includes their extern declarations (find_and_combine_kernel_extern.inc) and is the only caller.
// Hidden: code outside nve-common calls the exported call_find_and_combine_kernel instantiations.
template<DataType_t ELEMENT_TYPE_ID, typename INDEX_TYPE, typename ACC_TYPE, typename WEIGHT_TYPE,
         typename OUTPUT_TYPE, typename INPUT_VEC_TYPE, typename OUT_VEC_TYPE,
         typename ELEMENT_VEC_TYPE, typename ACC_VEC_TYPE, typename CacheDataT,
         uint32_t SZ_ACCUM, uint32_t MASK>
__attribute__((visibility("hidden")))
nve_fac_kernel_t<INDEX_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT> nve_fac_kernel()
{
    return find_and_combine<ELEMENT_TYPE_ID, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE,
                            INPUT_VEC_TYPE, OUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE,
                            CacheDataT, SZ_ACCUM, MASK>;
}

// Type-checks a selected kernel against the dispatch's parameter types, then erases its type for
// nve_fac_launch. Unweighted kernels take a float weight pointer instead (nve_fac_weight_t).
template<typename INDEX_TYPE, typename WEIGHT_TYPE, typename OUTPUT_TYPE, typename CacheDataT,
         uint32_t MASK>
inline const void* nve_fac_erase(nve_fac_kernel_t<INDEX_TYPE, nve_fac_weight_t<WEIGHT_TYPE, MASK>,
                                                  OUTPUT_TYPE, CacheDataT> kernel)
{
    return reinterpret_cast<const void*>(kernel);
}

// Launches a find_and_combine kernel by its host address with cudaLaunchKernel (the <<<>>> stub
// launches through a cached cudaKernel_t instead). `kernel` comes from nve_fac_erase with these
// parameter types; unweighted kernels take a float weight pointer, which they never read, so the
// caller's weight pointer is passed to them as is. Out of line so the element-type dispatchers
// share one copy per parameter-type combination.
template<typename INDEX_TYPE, typename WEIGHT_TYPE, typename OUTPUT_TYPE, typename CacheDataT>
__attribute__((noinline, visibility("hidden")))
void nve_fac_launch(const void* kernel,
                    uint32_t batchSz,
                    const int8_t* table,
                    const INDEX_TYPE* indices,
                    const INDEX_TYPE* offsets,
                    const WEIGHT_TYPE* weights,
                    int32_t num_hot,
                    CacheDataT cache,
                    int32_t rowSizeInElements,
                    void* output,
                    bool is_fixed_hotness,
                    bool load_indices,
                    cudaStream_t stream)
{
    OUTPUT_TYPE* pOutput = reinterpret_cast<OUTPUT_TYPE*>(output);
    void* args[] = {&batchSz, &table, &indices, &offsets, &weights, &num_hot, &cache,
                    &rowSizeInElements, &pOutput, &is_fixed_hotness, &load_indices};
    static_assert(std::is_same_v<nve_fac_kernel_t<INDEX_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>,
                                 void (*)(decltype(batchSz), decltype(table), decltype(indices),
                                          decltype(offsets), decltype(weights), decltype(num_hot),
                                          decltype(cache), decltype(rowSizeInElements),
                                          decltype(pOutput), decltype(is_fixed_hotness),
                                          decltype(load_indices))>,
                  "args[] must match the find_and_combine parameters");
    // Like the <<<>>> stub, the result is discarded; the cudaGetLastError check after dispatch
    // reports launch errors.
    (void)cudaLaunchKernel(kernel, dim3(batchSz, 1), dim3(32, 1), args, 0, stream);
}

inline uint32_t get_mask(PoolingType_t pooling_type) {
    uint32_t mask = 0;
    switch (pooling_type) {
      case PoolingType_t::Sum:
        mask |= NVE_FAC_MASK_SUM_POOLING;
        break;
      case PoolingType_t::Mean:
        break;
      case PoolingType_t::WeightedSum:
        mask |= (NVE_FAC_MASK_SUM_POOLING | NVE_FAC_MASK_IS_WEIGHTED);
        break;
      case PoolingType_t::WeightedMean:
        mask |= NVE_FAC_MASK_IS_WEIGHTED;
        break;
      default:
        NVE_THROW_("Unsupported pooling type ", static_cast<uint32_t>(pooling_type));
    }
    return mask;
}

template<DataType_t ELEMENT_TYPE_ID, typename INDEX_TYPE, typename ACC_TYPE, typename WEIGHT_TYPE, typename OUTPUT_TYPE, typename CacheDataT>
void call_find_and_combine_kernel_types_resolved(const uint32_t batchSz,
                             const int8_t* __restrict__ table,
                             const INDEX_TYPE* __restrict__ indices,
                             const INDEX_TYPE* __restrict__ offsets,
                             const WEIGHT_TYPE* __restrict__ weights,
                             int32_t num_hot,  
                             CacheDataT cache,
                             int32_t rowSizeInElements,
                             SparseType_t hot_type,
                             PoolingType_t pooling_type,
                             bool load_indices,
                             void* output,
                             cudaStream_t stream)
{
    // call_find_and_combine_kernel has already rejected sparse types other than Fixed and CSR.
    const bool is_fixed_hotness = hot_type == SparseType_t::Fixed;
    uint32_t mask = get_mask(pooling_type);
    if constexpr (NVE_FAC_SPECIALIZE_FLAGS) {
        mask |= (is_fixed_hotness ? NVE_FAC_MASK_FIXED_HOTNESS : 0u) |
                (load_indices ? NVE_FAC_MASK_LOAD_INDICES : 0u);
    }

    // The dispatch below only selects the kernel; one launch after it serves every combination.
    const void* kernel = nullptr;
#define NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T, MASK) \
    kernel = nve_fac_erase<INDEX_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT, MASK>( \
        nve_fac_kernel<ELEMENT_TYPE_ID, INDEX_TYPE, ACC_TYPE, nve_fac_weight_t<WEIGHT_TYPE, MASK>, \
                       OUTPUT_TYPE, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T, \
                       CacheDataT, (SZ_ACC_RUNTIME), MASK>())

    constexpr int32_t SUBWARP_WIDTH = 32;

    // Each thread holds SZ_ACCUM vector accumulators; with SUBWARP_WIDTH (=32) threads per row this
    // covers SUBWARP_WIDTH * SZ_ACCUM * vecSizeInElements row elements per pass. The kernel loops
    // over the row in chunks of that size, so any row length is supported regardless of SZ_ACCUM.
    // SZ_ACCUM is therefore purely a performance knob: we pick the smallest SZ_ACCUM that covers the
    // row in a single pass, clamped per vector width to min(4, 8 / vecSizeInElements) so that
    // vecSizeInElements * SZ_ACCUM <= 8 (bounded register pressure) and SZ_ACCUM <= 4. The clamped
    // range instantiated per width must match WIDTH_ACCUM in
    // cmake/gen_find_and_combine_instantiations.py.
#define NVE_FAC_SELECT_MASK(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T)  \
    do {                                                                                     \
        switch (mask) {                                                                      \
          case 0b00: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                    ACC_VEC_T, 0b00); break;                                  \
          case 0b01: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                    ACC_VEC_T, 0b01); break;                                  \
          case 0b10: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                    ACC_VEC_T, 0b10); break;                                  \
          case 0b11: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                    ACC_VEC_T, 0b11); break;                                  \
          NVE_FAC_SELECT_FLAG_MASKS(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T,  \
                                    ACC_VEC_T)                                               \
          default:                                                                           \
            NVE_THROW_("Unsupported mask ", mask);                                           \
        }                                                                                    \
    } while (0)
// Masks with the hotness / index-loading bits exist only in specialized builds.
#if NVE_FAC_SPECIALIZE_FLAGS
#define NVE_FAC_SELECT_FLAG_MASKS(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T,    \
                                  ACC_VEC_T)                                                 \
          case 0b0100: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b0100); break;                                \
          case 0b0101: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b0101); break;                                \
          case 0b0110: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b0110); break;                                \
          case 0b0111: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b0111); break;                                \
          case 0b1000: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1000); break;                                \
          case 0b1001: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1001); break;                                \
          case 0b1010: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1010); break;                                \
          case 0b1011: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1011); break;                                \
          case 0b1100: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1100); break;                                \
          case 0b1101: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1101); break;                                \
          case 0b1110: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1110); break;                                \
          case 0b1111: NVE_FAC_SELECT(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, \
                                      ACC_VEC_T, 0b1111); break;
#else
#define NVE_FAC_SELECT_FLAG_MASKS(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T,    \
                                  ACC_VEC_T)
#endif

    // Dispatch the runtime-selected SZ_ACCUM. Each vector width only instantiates SZ_ACCUM up to its
    // per-width clamp (min(4, 8 / vecSizeInElements)), so Vec4 switches over {1,2} while Vec2/Vec1
    // switch over {1..4}; referencing an un-instantiated SZ_ACCUM here would be a link error.
#define NVE_FAC_DISPATCH_2(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T)   \
    do {                                                                                        \
        switch (SZ_ACC_RUNTIME) {                                                                \
          case 1: NVE_FAC_SELECT_MASK(1, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T); break; \
          case 2: NVE_FAC_SELECT_MASK(2, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T); break; \
          default: NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME);                 \
        }                                                                                       \
    } while (0)

#define NVE_FAC_DISPATCH_4(SZ_ACC_RUNTIME, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T)   \
    do {                                                                                        \
        switch (SZ_ACC_RUNTIME) {                                                                \
          case 1: NVE_FAC_SELECT_MASK(1, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T); break; \
          case 2: NVE_FAC_SELECT_MASK(2, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T); break; \
          case 3: NVE_FAC_SELECT_MASK(3, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T); break; \
          case 4: NVE_FAC_SELECT_MASK(4, INPUT_VEC_TYPE, OUT_VEC_T, ELEMENT_VEC_T, ACC_VEC_T); break; \
          default: NVE_THROW_("Unsupported kernel dimensions ", SZ_ACC_RUNTIME);                 \
        }                                                                                       \
    } while (0)

    const int64_t row_bytes = row_size_in_bytes(ELEMENT_TYPE_ID, rowSizeInElements);

    // Smallest SZ_ACCUM covering the row in one pass, clamped per width to min(4, 8 / vecWidth).
    // Keep the clamps in sync with WIDTH_ACCUM in gen_find_and_combine_instantiations.py.
    if (row_bytes % sizeof(typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec4) == 0 && rowSizeInElements % 4 == 0)
    {
        const uint32_t SZ_ACCUM = std::min<uint32_t>(DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 4), 2u);
        using INPUT_VEC_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec4;
        using ELEMENT_VEC_TYPE = typename VecWidthHelper<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType>::Vec4;
        using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec4;
        using OUTPUT_VEC_TYPE = typename VecWidthHelper<OUTPUT_TYPE>::Vec4;
        NVE_FAC_DISPATCH_2(SZ_ACCUM, INPUT_VEC_TYPE, OUTPUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE);
    }
    else if (row_bytes % sizeof(typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec2) == 0 && rowSizeInElements % 2 == 0)
    {
      const uint32_t SZ_ACCUM = std::min<uint32_t>(DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 2), 4u);
      using INPUT_VEC_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec2;
      using ELEMENT_VEC_TYPE = typename VecWidthHelper<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType>::Vec2;
      using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec2;
      using OUTPUT_VEC_TYPE = typename VecWidthHelper<OUTPUT_TYPE>::Vec2;
      NVE_FAC_DISPATCH_4(SZ_ACCUM, INPUT_VEC_TYPE, OUTPUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE);
    }
    else
    {
      const uint32_t SZ_ACCUM = std::min<uint32_t>(DivRoundUp(rowSizeInElements, SUBWARP_WIDTH * 1), 4u);
      using INPUT_VEC_TYPE = typename QuantizationHelper<ELEMENT_TYPE_ID>::Vec1;
      using ELEMENT_VEC_TYPE = typename VecWidthHelper<typename QuantizationHelper<ELEMENT_TYPE_ID>::ParamType>::Vec1;
      using ACC_VEC_TYPE = typename VecWidthHelper<ACC_TYPE>::Vec1;
      using OUTPUT_VEC_TYPE = typename VecWidthHelper<OUTPUT_TYPE>::Vec1;
      NVE_FAC_DISPATCH_4(SZ_ACCUM, INPUT_VEC_TYPE, OUTPUT_VEC_TYPE, ELEMENT_VEC_TYPE, ACC_VEC_TYPE);
    }
#undef NVE_FAC_DISPATCH_2
#undef NVE_FAC_DISPATCH_4
#undef NVE_FAC_SELECT
#undef NVE_FAC_SELECT_MASK
#undef NVE_FAC_SELECT_FLAG_MASKS
    nve_fac_launch<INDEX_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(
        kernel, batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements,
        output, is_fixed_hotness, load_indices, stream);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename ACC_TYPE, typename WEIGHT_TYPE, typename OUTPUT_TYPE, typename INDEX_TYPE, typename CacheDataT>
void call_find_and_combine_kernel_acc_weight_type(const uint32_t batchSz,
                              const int8_t* __restrict__ table,
                              const INDEX_TYPE* __restrict__ indices,
                              const INDEX_TYPE* __restrict__ offsets,
                              const WEIGHT_TYPE* __restrict__ weights,
                              int32_t num_hot,  
                              CacheDataT cache,
                              int32_t rowSizeInElements,
                              SparseType_t hot_type,
                              PoolingType_t pooling_type,
                              bool load_indices,
                              DataType_t element_type,
                              void* output,
                              cudaStream_t stream)
{
    switch (element_type) {
      case DataType_t::Float32:
        call_find_and_combine_kernel_types_resolved<DataType_t::Float32, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::Float16:
        call_find_and_combine_kernel_types_resolved<DataType_t::Float16, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QUint8RowwiseF32:
        call_find_and_combine_kernel_types_resolved<DataType_t::QUint8RowwiseF32, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QInt8RowwiseF32:
        call_find_and_combine_kernel_types_resolved<DataType_t::QInt8RowwiseF32, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QUint8RowwiseF16:
        call_find_and_combine_kernel_types_resolved<DataType_t::QUint8RowwiseF16, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      case DataType_t::QInt8RowwiseF16:
        call_find_and_combine_kernel_types_resolved<DataType_t::QInt8RowwiseF16, INDEX_TYPE, ACC_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, CacheDataT>(batchSz, table, indices, offsets, weights, num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, output, stream);
        break;
      default:
        NVE_THROW_("Unsupported element type ", element_type);
    }
}

template<typename INDEX_TYPE, typename CacheDataT>
void call_find_and_combine_kernel(const uint32_t num_keys,
                              const uint32_t num_offsets,
                              const int8_t* __restrict__ table,
                              const INDEX_TYPE* __restrict__ indices,
                              const INDEX_TYPE* __restrict__ offsets,
                              const void* __restrict__ weights,
                              int32_t num_hot,  
                              CacheDataT cache,
                              int32_t rowSizeInElements,
                              SparseType_t hot_type,
                              PoolingType_t pooling_type,
                              bool load_indices,
                              DataType_t element_type,
                              DataType_t weight_type,
                              DataType_t acc_type,
                              DataType_t output_type,
                              void* output,
                              cudaStream_t stream)
{
    uint32_t batchSz = 0;
    switch (hot_type) {
      case SparseType_t::Fixed:
        NVE_CHECK_(num_keys % num_hot == 0);
        batchSz = num_keys / num_hot;
        break;
      case SparseType_t::CSR:
        NVE_CHECK_(num_offsets > 0 && offsets != nullptr);
        batchSz = num_offsets;
        break;
      default:
        NVE_THROW_("Unsupported sparse type ", static_cast<uint32_t>(hot_type));
    }
    // Resolve the output type for a fixed (ACC_T, WEIGHT_T) pair. Output precision is independent of
    // acc/weight, so each supported (acc, weight) combination fans out over both output types.
#define FAC_DISPATCH_OUTPUT(ACC_T, WEIGHT_T)                                                                                                                                                                                                              \
    do {                                                                                                                                                                                                                                                  \
        switch (output_type) {                                                                                                                                                                                                                            \
          case DataType_t::Float32:                                                                                                                                                                                                                       \
            call_find_and_combine_kernel_acc_weight_type<ACC_T, WEIGHT_T, float, INDEX_TYPE, CacheDataT>(batchSz, table, indices, offsets, static_cast<const WEIGHT_T*>(weights), num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, element_type, output, stream); \
            break;                                                                                                                                                                                                                                        \
          case DataType_t::Float16:                                                                                                                                                                                                                       \
            call_find_and_combine_kernel_acc_weight_type<ACC_T, WEIGHT_T, __half, INDEX_TYPE, CacheDataT>(batchSz, table, indices, offsets, static_cast<const WEIGHT_T*>(weights), num_hot, cache, rowSizeInElements, hot_type, pooling_type, load_indices, element_type, output, stream); \
            break;                                                                                                                                                                                                                                        \
          default:                                                                                                                                                                                                                                        \
            NVE_THROW_("Unsupported output type ", output_type);                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                 \
    } while (0)

    switch (acc_type) {
      case DataType_t::Float32:
        switch (weight_type) {
          case DataType_t::Float32: FAC_DISPATCH_OUTPUT(float, float);   break;
          case DataType_t::Float16: FAC_DISPATCH_OUTPUT(float, __half);  break;
          default:
            NVE_THROW_("Unsupported weight type ", weight_type);
        }
        break;
      default:
        NVE_THROW_("Unsupported acc type ", acc_type, ": only Float32 accumulation is supported");
    }

#undef FAC_DISPATCH_OUTPUT
}

// extern template declarations of call_find_and_combine_kernel for the index/cache types nve-common
// instantiates (generated, see cmake/gen_find_and_combine_instantiations.py). The definitions live
// in the generated find_and_combine_dispatch.cu; call_find_and_combine_kernel is the only entry
// point usable outside nve-common, since the nve_fac_kernel functions it calls are hidden.
#include "find_and_combine_dispatch_extern.inc"
