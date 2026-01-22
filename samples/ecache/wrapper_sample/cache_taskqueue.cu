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

#include <bitset>
#include <cassert>
#include <cstring>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <random>
#include <unordered_set>
#include <cuda_support.hpp>
#include "cache_taskqueue.h"
#include <embedding_cache_combined.cuh>
#include <datagen.h>
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/cuda_utils.cuh"
#include "cuda_ops/dedup_grads_kernel.cuh"

using namespace nve;

template<uint32_t SubwarpWidth, typename DataType>
__global__ void EmbedScatter( const int8_t* __restrict__ src,
                              int8_t* __restrict__ dst,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const uint64_t* __restrict__ hit_mask,
                              const int num_indices)
{
    const int embed = blockIdx.x * blockDim.y + threadIdx.y;

    if (embed >= num_indices) {
      return;
    }

    const int mask_entry = embed / 64;
    const int mask_bit = embed % 64;
    uint64_t mask = 1;
    mask <<= mask_bit;

    if (hit_mask[mask_entry] & mask) {
      return;
    }

    const int8_t* embed_src = src + embed * embed_src_stride_in_bytes;
    int8_t* embed_dst = dst + embed * embed_dst_stride_in_bytes;

    nve::MemcpyWarp<SubwarpWidth, DataType>(embed_dst, embed_src, embed_width_in_bytes);
}

template<uint32_t SubwarpWidth, typename DataType>
void CallScatterKernelVecTypeSubwarp(
                              const int8_t* __restrict__ src,
                              int8_t* __restrict__ dst,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const uint64_t* __restrict__ hit_mask,
                              const int num_indices,
                              const cudaStream_t stream = 0)
{
    uint32_t indices_per_warp = 32 / SubwarpWidth;
    dim3 grid_size ((num_indices + indices_per_warp - 1) / indices_per_warp, 1);
    dim3 block_size (SubwarpWidth, indices_per_warp);
    EmbedScatter<SubwarpWidth, DataType><<<grid_size, block_size, 0, stream>>>(
        src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename DataType>
void CallScatterKernelVecType(
                              const int8_t* __restrict__ src,
                              int8_t* __restrict__ dst,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const uint64_t* __restrict__ hit_mask,
                              const int num_indices,
                              const cudaStream_t stream = 0)
{
    uint32_t subgroupWidth = std::min(nextPow2(embed_width_in_bytes / sizeof(DataType)), 32u);
    switch (subgroupWidth)
    {
    case 32:
        CallScatterKernelVecTypeSubwarp<32, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 16:
        CallScatterKernelVecTypeSubwarp<16, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 8:
        CallScatterKernelVecTypeSubwarp<8, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 4:
        CallScatterKernelVecTypeSubwarp<4, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 2:
        CallScatterKernelVecTypeSubwarp<2, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 1:
        CallScatterKernelVecTypeSubwarp<1, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    default:
        assert(0);
    }
}

template<typename VecType>
void CallPoolingDenseKernelVecType(
                              const VecType* __restrict__ src,
                              VecType* __restrict__ dst,
                              const uint32_t num_elements,
                              const uint32_t embed_src_stride,
                              const uint32_t embed_dst_stride,
                              const uint32_t hotness,
                              const uint32_t num_bags,
                              const cudaStream_t stream = 0)
{
    uint32_t subgroupWidth = std::min(nextPow2(num_elements), 32u);
    switch (subgroupWidth)
    {
    case 32:
        CallPoolingKernelVecTypeSubwarp<32, VecType>(src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags, stream);
        break;
    case 16:
        CallPoolingKernelVecTypeSubwarp<16, VecType>(src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags, stream);
        break;
    case 8:
        CallPoolingKernelVecTypeSubwarp<8, VecType>(src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags, stream);
        break;
    case 4:
        CallPoolingKernelVecTypeSubwarp<4, VecType>(src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags, stream);
        break;
    case 2:
        CallPoolingKernelVecTypeSubwarp<2, VecType>(src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags, stream);
        break;
    case 1:
        CallPoolingKernelVecTypeSubwarp<1, VecType>(src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags, stream);
        break;
    default:
        assert(0);
    }
}

void EmbeddingForwardScatter(const void* src,
                             void* dst,
                             const uint32_t embed_width_in_bytes,
                             const uint32_t embed_src_stride_in_bytes,
                             const uint32_t embed_dst_stride_in_bytes,
                             const uint64_t* hit_mask,
                             const int num_indices,
                             const cudaStream_t stream = 0)
{
    if ((embed_width_in_bytes % 16) == 0) {
      using Vec4 = typename VecWidthHelper<float>::Vec4;
      CallScatterKernelVecType<Vec4>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } else if ((embed_width_in_bytes % 8) == 0) {
      using Vec2 = typename VecWidthHelper<float>::Vec2;
      CallScatterKernelVecType<Vec2>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } else if ((embed_width_in_bytes % 4) == 0) {
      using Vec1 = typename VecWidthHelper<float>::Vec1;
      CallScatterKernelVecType<Vec1>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } else if ((embed_width_in_bytes % 2) == 0) {
      using Vec1 = typename VecWidthHelper<__half>::Vec1;
      CallScatterKernelVecType<Vec1>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } 
}

void EmbeddingPoolingDense(const void* src,
                           void* dst,
                           nve::DataTypeFormat element_format,
                           const uint32_t embed_width_in_bytes,
                           const uint32_t embed_src_stride_in_bytes,
                           const uint32_t embed_dst_stride_in_bytes,
                           const uint32_t hotness,
                           const uint32_t num_bags,
                           const cudaStream_t stream = 0)
{
    assert((element_format == nve::DATATYPE_FP16) || (element_format == nve::DATATYPE_FP32));
    const uint32_t num_elements = (element_format == nve::DATATYPE_FP16) ? (embed_width_in_bytes / sizeof(__half)) 
                                                                            : (embed_width_in_bytes / sizeof(float));

    if ((num_elements % 4) == 0) {
      if (element_format == nve::DATATYPE_FP16) {
        using Vec4 = typename VecWidthHelper<__half>::Vec4;
        CallPoolingDenseKernelVecType<Vec4>(reinterpret_cast<const Vec4*>(src), reinterpret_cast<Vec4*>(dst), num_elements / 4,
                                            embed_src_stride_in_bytes / (4 * sizeof(__half)), embed_dst_stride_in_bytes / (4 * sizeof(__half)), hotness, num_bags, stream);
      } else {
        using Vec4 = typename VecWidthHelper<float>::Vec4;
        CallPoolingDenseKernelVecType<Vec4>(reinterpret_cast<const Vec4*>(src), reinterpret_cast<Vec4*>(dst), num_elements / 4,
                                            embed_src_stride_in_bytes / (4 * sizeof(float)), embed_dst_stride_in_bytes / (4 * sizeof(float)), hotness, num_bags, stream);
      }
    } else if ((num_elements % 2) == 0) {
      if (element_format == nve::DATATYPE_FP16) {
        using Vec2 = typename VecWidthHelper<__half>::Vec2;
        CallPoolingDenseKernelVecType<Vec2>(reinterpret_cast<const Vec2*>(src), reinterpret_cast<Vec2*>(dst), num_elements / 2,
                                            embed_src_stride_in_bytes / (2 * sizeof(__half)), embed_dst_stride_in_bytes / (2 * sizeof(__half)), hotness, num_bags, stream);
      } else {
        using Vec2 = typename VecWidthHelper<float>::Vec2;
        CallPoolingDenseKernelVecType<Vec2>(reinterpret_cast<const Vec2*>(src), reinterpret_cast<Vec2*>(dst), num_elements / 2,
                                            embed_src_stride_in_bytes / (2 * sizeof(float)), embed_dst_stride_in_bytes / (2 * sizeof(float)), hotness, num_bags, stream);
      }
    } else {
      if (element_format == nve::DATATYPE_FP16) {
        using Vec1 = typename VecWidthHelper<__half>::Vec1;
        CallPoolingDenseKernelVecType<Vec1>(reinterpret_cast<const Vec1*>(src), reinterpret_cast<Vec1*>(dst), num_elements,
                                            embed_src_stride_in_bytes / sizeof(__half), embed_dst_stride_in_bytes / sizeof(__half), hotness, num_bags, stream);
      } else {
        using Vec1 = typename VecWidthHelper<float>::Vec1;
        CallPoolingDenseKernelVecType<Vec1>(reinterpret_cast<const Vec1*>(src), reinterpret_cast<Vec1*>(dst), num_elements,
                                            embed_src_stride_in_bytes / sizeof(float), embed_dst_stride_in_bytes / sizeof(float), hotness, num_bags, stream);
      }
    }
}

template<typename IndexType>
void CallDedupGradients(const void* __restrict__ src,
                        void* __restrict__ dst,
                        const IndexType* __restrict__ unique_keys,
                        const IndexType* __restrict__ inverse_buffer,
                        const IndexType* key_counts,
                        const IndexType* key_offsets,
                        nve::DataTypeFormat element_format,
                        const uint64_t num_unique_keys,                                                        
                        const uint32_t embed_width_in_bytes,
                        const cudaStream_t stream = 0)
{
    assert((element_format == nve::DATATYPE_FP16) || (element_format == nve::DATATYPE_FP32));

    IndexType num_unique_keys_h[2];
    num_unique_keys_h[0] = num_unique_keys_h[1] = static_cast<IndexType>(num_unique_keys);

    if (element_format == nve::DATATYPE_FP16) {
        CallComputeGradients<IndexType, __half>(
            reinterpret_cast<const __half*>(src),
            reinterpret_cast<__half*>(dst),
            unique_keys, inverse_buffer, key_counts, key_offsets,
            num_unique_keys_h, embed_width_in_bytes / sizeof(__half), stream);
    } else {
        CallComputeGradients<IndexType, float>(
            reinterpret_cast<const float*>(src),
            reinterpret_cast<float*>(dst),
            unique_keys, inverse_buffer, key_counts, key_offsets,
            num_unique_keys_h, embed_width_in_bytes / sizeof(float), stream);
    }
}

template<typename IndexT>
double HostResolve(IndexT* h_keys, uint64_t* h_hit_mask, uint64_t* d_hit_mask, 
    void* h_values, void* d_values, IndexT* h_freq,
    uint64_t num_keys, uint64_t embedding_width_in_bytes, cudaStream_t stream,
    std::shared_ptr<MockHostCache<IndexT>> host_cache,
    std::shared_ptr<MockParameterServer<IndexT>> ps,
    CacheIterationMetric* metric)
{
    // First calculate the hitrate (needed to drive insert heuristic later)
    uint64_t missCount = 0;
    const uint64_t element_bits = sizeof(uint64_t) * 8;
    const uint64_t mask_elements = (num_keys + element_bits - 1) / element_bits;
    uint64_t total_num_keys = 0; // number of original keys, as we can reach this function through dedup flow num_keys reference the number of unique keys
    for (uint64_t i=0 ; i<mask_elements ; i++)
    {
        auto mask = h_hit_mask[i];
        
        // Last mask can have less than 64 bits
        if ((i == (mask_elements-1)) && (num_keys % element_bits))
        {
            uint64_t rest = uint64_t(-1) << (num_keys % element_bits);
            mask |= rest;
        }
        if (h_freq)
        {
            for (uint64_t j = 0; j < element_bits; j++)
            {
                if ((mask & (1llu << j)) == 0)
                {
                    missCount += h_freq[i*element_bits + j];
                }
                if ((i < (mask_elements-1)) || ((i == (mask_elements-1)) && (j < (num_keys % element_bits))))
                {
                    //claculate the non deduped number of keys
                    total_num_keys += h_freq[i*element_bits + j];
                }
            }   
        }
        else
        {
            missCount += element_bits - std::bitset<element_bits>(mask).count();
            total_num_keys += element_bits;
        }
    }

    // Then resolve misses using mock host cache
    host_cache->Lookup(h_keys, num_keys, reinterpret_cast<int8_t*>(h_values), embedding_width_in_bytes, h_hit_mask, metric);
    
    // And access the PS to resolve the last cache misses (reading h_hit_mask and writing to h_values)
    ps->Lookup(h_keys, num_keys, reinterpret_cast<int8_t*>(h_values), embedding_width_in_bytes, h_hit_mask, metric);

    // Finally, handle scatter of the resolved vectors on host
    if (metric) {
        NVE_CHECK_(cudaEventRecord(metric->scatter.d_start, stream));
    }
    EmbeddingForwardScatter(h_values, d_values, static_cast<uint32_t>(embedding_width_in_bytes),
        static_cast<uint32_t>(embedding_width_in_bytes), static_cast<uint32_t>(embedding_width_in_bytes), d_hit_mask,
        static_cast<int>(num_keys), // TODO: change scatter num indices to 64bit?
        stream);
    const auto hitrate = (total_num_keys > 0) ? (1.0 - (static_cast<double>(missCount) / static_cast<double>(total_num_keys))) : 0.0;
    if (metric) {
        NVE_CHECK_(cudaEventRecord(metric->scatter.d_end, stream));
        metric->hitrate = hitrate;
    }

    return hitrate;
}

void PoolingDense(void* d_values, void* d_pooling, nve::DataTypeFormat element_format,
    uint64_t embedding_width_in_bytes, uint32_t hotness, uint32_t num_bags,
    cudaStream_t stream, CacheIterationMetric* metric)
{
    if (metric) {
        NVE_CHECK_(cudaEventRecord(metric->pooling.d_start, stream));
    }
    EmbeddingPoolingDense(d_values, d_pooling, element_format, static_cast<uint32_t>(embedding_width_in_bytes), 
        static_cast<uint32_t>(embedding_width_in_bytes), static_cast<uint32_t>(embedding_width_in_bytes), hotness, num_bags, stream);
    if (metric) {
        NVE_CHECK_(cudaEventRecord(metric->pooling.d_end, stream));
    }
}

template<typename IndexT>
void DedupGradients(const void* d_gradients, void* d_unique_gradients, nve::DataTypeFormat element_format,
    const IndexT* d_unique_keys, const IndexT* d_counts, const IndexT* d_offsets, const IndexT* d_inverse_buffer,
    uint64_t embedding_width_in_bytes, uint64_t num_unique_keys, cudaStream_t stream, CacheIterationMetric* metric)
{
    if (metric) {
        NVE_CHECK_(cudaEventRecord(metric->dedup_gradients.d_start, stream));
    }
    
    CallDedupGradients(d_gradients, d_unique_gradients, d_unique_keys, d_inverse_buffer, d_counts, d_offsets,
        element_format, num_unique_keys, static_cast<uint32_t>(embedding_width_in_bytes), stream);

    if (metric) {
        NVE_CHECK_(cudaEventRecord(metric->dedup_gradients.d_end, stream));
    }
}

bool InsertHeuristic(double cache_hr)
{
    static std::mt19937 gen(98437598437); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (cache_hr < 0.75)
        return (dis(gen) < (1 - cache_hr));
    else
        return dis(gen) < (1 - cache_hr)*(1-cache_hr)*(1-cache_hr);
}
template<typename IndexT>
CacheTaskQueue<IndexT>::CacheTaskQueue(
    std::shared_ptr<ECacheWrapper<IndexT>> cache,
    int device_id,
    uint64_t insert_key_buffer_size,
    uint64_t insert_val_buffer_size,
    uint64_t max_keys,
    std::shared_ptr<MockHostCache<IndexT>> host_cache,
    std::shared_ptr<MockParameterServer<IndexT>> ps)
        : m_cache(std::move(cache)), m_main_queue(device_id, 1), m_insert_queue(device_id, 1), m_host_cache(std::move(host_cache)), m_ps(std::move(ps))
{
    auto allocator = m_cache->GetAllocator();
    uint64_t unique_val_buffer_size = max_keys*m_cache->GetConfig().embedWidth;
    NVE_CHECK_(allocator->hostAllocate((void**)(&m_insert_h_keys), insert_key_buffer_size));
    NVE_CHECK_(allocator->deviceAllocate((void**)(&m_insert_d_values), insert_val_buffer_size));

    NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_unique_buffer), unique_val_buffer_size));
    NVE_CHECK_(allocator->hostAllocate((void**)(&m_priority), max_keys*sizeof(float)));
    m_deduper = std::make_shared<Deduper<IndexT>>();
}

template<typename IndexT>
std::future<void> CacheTaskQueue<IndexT>::LookupDeduped(
    IndexT* d_keys,
    IndexT* h_keys,
    uint64_t hotness,
    uint64_t batch_size,
    IndexT* d_unique_keys, 
    IndexT* h_unique_keys,
    IndexT* d_counts, 
    IndexT* h_counts,
    IndexT* d_inverse_buffer, 
    IndexT* d_offsets,
    IndexT* h_num_runs, 
    int8_t* d_values,
    int8_t* h_values,
    uint64_t* d_hit_mask,
    uint64_t* h_hit_mask,
    int8_t* d_pooling_values,
    bool host_input,
    CacheIterationMetric* metric)
    {
        IndexT num_keys = static_cast<IndexT>(hotness * batch_size);
        return m_main_queue.Submit([=]() {
            if (metric) {
                metric->lookup.h_start = CacheIterationMetric::Now();
            }
            // Copy input keys to device if needed (the data is already there, copying only to simulate the actual data flow)
            cudaStream_t stream = m_cache->GetLookupStreams()[0];
            if (host_input) {
                NVE_CHECK_(cudaMemcpyAsync(d_keys, h_keys, num_keys * sizeof(IndexT), cudaMemcpyDefault, stream));
            }

            size_t tmp_mem_size_device, tmp_mem_size_host;
            m_deduper->GetAllocRequirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
            auto allocator = m_cache->GetAllocator();
            char* tmp_device_mem;
            NVE_CHECK_(allocator->deviceAllocate((void**)(&tmp_device_mem), tmp_mem_size_device));
            char* tmp_host_mem;
            NVE_CHECK_(allocator->hostAllocate((void**)(&tmp_host_mem), tmp_mem_size_device));
            m_deduper->SetAndInitBuffers(num_keys, tmp_device_mem, tmp_host_mem);

            m_deduper->Dedup(d_keys, 
                    num_keys,
                    d_unique_keys, 
                    d_counts, 
                    nullptr, // no need for location map
                    h_num_runs, 
                    d_inverse_buffer, 
                    d_offsets,
                    stream);
            uint64_t num_unique_keys = *h_num_runs;
            // 1. Launch cache lookup
            size_t hit_mask_sz_in_bytes = ((num_unique_keys + 63)/64) * sizeof(uint64_t);
            const auto embed_width_in_bytes = m_cache->GetConfig().embedWidth;
            cudaEvent_t lookup_start = (metric ? metric->lookup.d_start : nullptr);
            cudaEvent_t lookup_end = (metric ? metric->lookup.d_end : nullptr);
            m_cache->Lookup(0, d_unique_keys, num_unique_keys, m_d_unique_buffer, embed_width_in_bytes, d_hit_mask, lookup_start, lookup_end);

            // 3. Copy hitmask (and keys if needed) to host (needed to resolve cache misses)
            NVE_CHECK_(cudaMemcpyAsync(h_hit_mask, d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, stream));
            NVE_CHECK_(cudaMemcpyAsync(h_unique_keys, d_unique_keys, num_unique_keys * sizeof(IndexT), cudaMemcpyDefault, stream));
            NVE_CHECK_(cudaMemcpyAsync(h_counts, d_counts, sizeof(IndexT)*num_unique_keys, cudaMemcpyDefault, stream));
            NVE_CHECK_(cudaStreamSynchronize(stream)); // Wait for copy to complete (used in next step)
            
            // 4. Resolve misses on host and launch scatter kernel to device buffer
            double hitrate = HostResolve(h_unique_keys, h_hit_mask, d_hit_mask, h_values, m_d_unique_buffer, h_counts, num_unique_keys, embed_width_in_bytes, stream, m_host_cache, m_ps, metric);
            
            // 5. Unpack the unique buffer with inverse buffer
            UnpackDedup(d_inverse_buffer, d_offsets, d_counts, m_d_unique_buffer, d_values, num_unique_keys, embed_width_in_bytes, stream);
            
            if (metric) {
                metric->lookup.h_end = CacheIterationMetric::Now();
            }

            if (d_pooling_values != nullptr) {
                nve::DataTypeFormat element_format = m_cache->GetElementFormat();
                PoolingDense(d_values, d_pooling_values, element_format, embed_width_in_bytes, static_cast<uint32_t>(hotness), static_cast<uint32_t>(batch_size), stream, metric);
            }
            // 6. Handle cache inserts
            if (InsertHeuristic(hitrate))
            {
                if (!m_insert_lock.test_and_set()) { // Acquire "single insert in-flight" lock
                    // 6.1 copy keys to host side temporary buffer
                    std::memcpy(m_insert_h_keys, h_unique_keys, num_unique_keys * sizeof(IndexT));
                    
                    // 6.2 copy device values to temporary buffer
                    // we assume single stream mode here. if modify stream != lookup stream, we need to add a cuda event for the insert to wait for the copy to finish
                    assert(stream == m_cache->GetModifyStream());
                    NVE_CHECK_(cudaMemcpyAsync(m_insert_d_values, m_d_unique_buffer, num_unique_keys * m_cache->GetConfig().embedWidth, cudaMemcpyDefault, stream));
                    for (uint64_t i = 0; i < num_unique_keys; i++)
                    {
                        m_priority[i] = (float)h_counts[i];
                    }
                    // 5.3 offload insert to second work queue
                    m_insert_queue.Submit([=](){
                        cudaEvent_t insert_start = (metric ? metric->insert.d_start : nullptr);
                        cudaEvent_t insert_end = (metric ? metric->insert.d_end : nullptr);
                        if (metric) {
                            metric->insert_done = true;
                            metric->insert.h_start = metric->Now();
                        }
                        std::vector<int8_t*> data_ptr;
                        for (uint64_t i = 0; i < num_unique_keys; i++)
                        {
                            data_ptr.push_back(m_insert_d_values + i * embed_width_in_bytes);
                        }
                        m_cache->Insert(m_insert_h_keys, num_unique_keys, m_priority, data_ptr.data(), insert_start, insert_end);
                        if (metric) {
                            metric->insert.h_end = metric->Now();
                        }
                        m_insert_lock.clear(); // Release "single insert in-flight" lock
                    });
                }
            }
        });
    }

template<typename IndexT>
std::future<void> CacheTaskQueue<IndexT>::Lookup(
    IndexT* d_keys,
    IndexT* h_keys,
    const uint64_t batch_size,
    const uint64_t hotness,
    int8_t* d_values,
    int8_t* h_values,
    uint64_t* d_hit_mask,
    uint64_t* h_hit_mask,
    int8_t* d_pooling_values,
    bool host_input,
    CacheIterationMetric* metric)
{
    const uint64_t num_keys = batch_size * hotness;

    return m_main_queue.Submit([=]() {
        if (metric) {
            metric->lookup.h_start = CacheIterationMetric::Now();
        }
        // 1. Copy input keys to device if needed (the data is already there, copying only to simulate the actual data flow)
        cudaStream_t stream = m_cache->GetLookupStreams()[0];
        // either copy inputs from host to device for lookup, or from device to host for the resolve later (parallelize with lookup)
        if (host_input) {
            NVE_CHECK_(cudaMemcpyAsync(d_keys, h_keys, num_keys * sizeof(IndexT), cudaMemcpyDefault, stream));
        } else {
            NVE_CHECK_(cudaMemcpyAsync(h_keys, d_keys, num_keys * sizeof(IndexT), cudaMemcpyDefault, stream));
        }
        // 2. Launch cache lookup
        size_t hit_mask_sz_in_bytes = ((num_keys + 63)/64) * sizeof(uint64_t);
        const auto embed_width_in_bytes = m_cache->GetConfig().embedWidth;
        cudaEvent_t lookup_start = (metric ? metric->lookup.d_start : nullptr);
        cudaEvent_t lookup_end = (metric ? metric->lookup.d_end : nullptr);
        m_cache->Lookup(0, d_keys, num_keys, d_values, embed_width_in_bytes, d_hit_mask, lookup_start, lookup_end);

        // 3. Copy hitmask to host (needed to resolve cache misses)
        NVE_CHECK_(cudaMemcpyAsync(h_hit_mask, d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, stream));
        NVE_CHECK_(cudaStreamSynchronize(stream)); // Wait for copy to complete (used in next step)
        
        // 4. Resolve misses on host and launch scatter kernel to device buffer
        // passing nullptr on the host resolve freq buffer as each hit as a freq of 1
        double hitrate = HostResolve(h_keys, h_hit_mask, d_hit_mask, h_values, d_values, (IndexT*)nullptr, num_keys, embed_width_in_bytes, stream, m_host_cache, m_ps, metric);
        if (metric) {
            metric->lookup.h_end = CacheIterationMetric::Now();
        }

        // 4a. Optional: perform pooling
        if (d_pooling_values != nullptr) {
          nve::DataTypeFormat element_format = m_cache->GetElementFormat();
          PoolingDense(d_values, d_pooling_values, element_format, embed_width_in_bytes, static_cast<uint32_t>(hotness), static_cast<uint32_t>(batch_size), stream, metric);
        }

        // 5. Handle cache inserts
        if (InsertHeuristic(hitrate))
        {
            if (!m_insert_lock.test_and_set()) { // Acquire "single insert in-flight" lock
                // 5.1 copy keys to host side temporary buffer
                std::memcpy(m_insert_h_keys, h_keys, num_keys * sizeof(IndexT));
                
                // 5.2 copy device values to temporary buffer
                // we assume single stream mode here. if modify stream != lookup stream, we need to add a cuda event for the insert to wait for the copy to finish
                assert(stream == m_cache->GetModifyStream());
                NVE_CHECK_(cudaMemcpyAsync(m_insert_d_values, d_values, num_keys * m_cache->GetConfig().embedWidth, cudaMemcpyDefault, stream));
                
                // 5.3 offload insert to second work queue
                m_insert_queue.Submit([=](){
                    cudaEvent_t insert_start = (metric ? metric->insert.d_start : nullptr);
                    cudaEvent_t insert_end = (metric ? metric->insert.d_end : nullptr);
                    if (metric) {
                        metric->insert_done = true;
                        metric->insert.h_start = metric->Now();
                    }
                    nve::DefaultHistogram histogram(m_insert_h_keys, num_keys, m_insert_d_values, embed_width_in_bytes, false);
                    m_cache->Insert(histogram.GetKeys(), histogram.GetNumBins(), histogram.GetPriority(), histogram.GetData(), insert_start, insert_end);
                    if (metric) {
                        metric->insert.h_end = metric->Now();
                    }
                    m_insert_lock.clear(); // Release "single insert in-flight" lock
                });
            }
        }
    });
}

template<typename IndexT>
std::future<void> CacheTaskQueue<IndexT>::GradientsDedup(
    const uint64_t num_keys,
    const int8_t* d_gradients,
    int8_t* d_accumulate,
    const IndexT* d_grads_unique_keys,
    const IndexT* d_grads_counts,
    const IndexT* d_grads_offsets,
    const IndexT* d_grads_inverse_buffer,
    CacheIterationMetric* metric)
{
    return m_main_queue.Submit([=]() {
        cudaStream_t stream = m_cache->GetLookupStreams()[0];
        const auto row_size_bytes = m_cache->GetConfig().embedWidth;
        nve::DataTypeFormat element_format = m_cache->GetElementFormat();

        DedupGradients<IndexT>(d_gradients, d_accumulate, element_format, d_grads_unique_keys, 
                               d_grads_counts, d_grads_offsets, d_grads_inverse_buffer,
                               row_size_bytes, num_keys, stream, metric);
    });  
}

template<typename IndexT>
std::future<void> CacheTaskQueue<IndexT>::Accumulate(
    const IndexT* d_keys,
    const uint64_t num_keys,
    int8_t* d_values,
    nve::DataTypeFormat value_format,
    const IndexT* h_keys,
    int8_t* h_values,
    CacheIterationMetric* metric)
{
    return m_main_queue.Submit([=]() {
        // Accumulate in GPU cache
        cudaEvent_t accumulate_start = (metric ? metric->accumulate.d_start : nullptr);
        cudaEvent_t accumulate_end =  (metric ? metric->accumulate.d_end : nullptr);
        const auto row_size_bytes = m_cache->GetConfig().embedWidth;
        m_cache->AccumulateNoSync(d_keys, num_keys, d_values, row_size_bytes, value_format, accumulate_start, accumulate_end);

        // Accumulate in host cache
        m_host_cache->Accumulate(h_keys, num_keys, h_values, value_format, metric);

        // Accumulate in the param server
        m_ps->Accumulate(h_keys, num_keys, h_values, value_format, metric);
    });
}

template class CacheTaskQueue<int32_t>;
template class CacheTaskQueue<int64_t>;
