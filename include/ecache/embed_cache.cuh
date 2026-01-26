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

#include <cuda_runtime.h>
#include "embed_cache.h"
namespace nve {

template<typename IndexT, typename CacheDataT>
class AddressFunctor
{
public:
    static __device__ inline uint64_t get_address(IndexT /*index*/, const int8_t* /*table*/, uint32_t /*curr_table*/, const CacheDataT /*data*/)
    {
        return 0;
    }
};

template<uint32_t SUBWARP_WIDTH, typename DataType>
__device__ void memcpy_warp(int8_t* pDst, const int8_t* __restrict__ pSrc, uint32_t sz)
{
    const auto offset_in_sample = threadIdx.x % SUBWARP_WIDTH;
    const uint32_t ELEMENT_SIZE = sizeof(DataType);
    for (uint32_t k = 0; k < sz; k += ELEMENT_SIZE*SUBWARP_WIDTH)
    {
        uint32_t offset = k + offset_in_sample * ELEMENT_SIZE;
        if (offset < sz)
        {
            DataType d = *(DataType*)((int8_t*)pSrc + offset);
            DataType* dst_ptr = (DataType*)((int8_t*)pDst + offset);
            *dst_ptr = d;
        }
    }
}

template<typename IndexT, uint32_t SUBWARP_WIDTH, uint32_t BLOCK_Y, typename DataType, typename CacheDataT>
__global__ void query_uvm(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* __restrict__ d_table,
    CacheDataT data, uint32_t curr_table, size_t stride)
{
    uint32_t tid_batch = blockIdx.x * SUBWARP_WIDTH * BLOCK_Y + threadIdx.y * SUBWARP_WIDTH;
    uint32_t tid = tid_batch + threadIdx.x; // each tid search for one index, and then we do a "transpose" and copy them out if needed
    uint64_t laneptr;
    
    if (tid >= len)
    {
        laneptr = 0;
    }
    else
    {
        IndexT lane_idx = d_keys[tid];
        laneptr = AddressFunctor<IndexT, CacheDataT>::get_address(lane_idx, d_table, curr_table, data);
    }

    __syncwarp();

    #pragma unroll
    for (uint32_t s = 0; s < SUBWARP_WIDTH; s++)
    {
        const uint32_t ELEMENT_SIZE = sizeof(DataType);
        uint64_t src_ptr = __shfl_sync(0xffffffff, laneptr, s, SUBWARP_WIDTH);
        if (src_ptr != 0) {
            for (uint32_t k = 0; k < data.row_size_in_bytes; k += ELEMENT_SIZE*SUBWARP_WIDTH)
            {
                uint32_t offset = k + threadIdx.x * ELEMENT_SIZE;
                if (offset < data.row_size_in_bytes)
                {
                    DataType d = __ldg((DataType*)(src_ptr + offset));
                    DataType* dst_ptr = (DataType*)(d_values + (tid_batch + s) * stride + offset);
                    __stcs(dst_ptr, d);
                }
            }
        }
    }
}

template<typename IndexT, typename CacheDataT>
cudaError_t call_cache_query_uvm(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t curr_table, size_t stride)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 4;
    const uint32_t blockSize = blockX * blockY;
    const uint32_t nBlock = static_cast<uint32_t>(len / blockSize + std::min(len % blockSize, (size_t)1));
    dim3 gridDims(nBlock);
    dim3 blockDims(blockX, blockY);
    if (data.row_size_in_bytes % sizeof(uint4) == 0)
    {
        query_uvm<IndexT, blockX, blockY, uint4><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, curr_table, stride);
    }
    else if (data.row_size_in_bytes % sizeof(uint32_t) == 0)
    {
        query_uvm<IndexT, blockX, blockY, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, curr_table, stride);
    }
    else
    {
        query_uvm<IndexT, blockX, blockY, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_keys, len, d_values, d_table, data, curr_table, stride);
    }
    return cudaGetLastError();
}

}
