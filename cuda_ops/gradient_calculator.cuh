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
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/kernels_common.cuh"
#include "cuda_ops/cuda_utils.cuh"
#include "include/nve_types.hpp"

template <typename IndexType, typename DataType, bool FIXED_HOTNESS = true>
__global__ void ComputeNormalizedWeights(const  DataType* __restrict__ input_weights,
                                         const uint32_t hotness,
                                         const  IndexType* __restrict__ csr_offsets,
                                         DataType* __restrict__ output_weights,
                                         const uint32_t num_bags) {
    uint32_t bag_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (bag_id >= num_bags) {
        return;
    }

    // compute sum of weights
    IndexType offset = FIXED_HOTNESS ? bag_id * hotness : csr_offsets[bag_id];
    IndexType count = FIXED_HOTNESS ? hotness : csr_offsets[bag_id + 1] - csr_offsets[bag_id];
    DataType acc = 0;

    for (IndexType w = 0; w < count; w += warpSize) {
        auto thread_idx = w + threadIdx.x;
        DataType curr_sum = (thread_idx < count) ? input_weights[offset + thread_idx] : DataType(0);
        uint32_t idx_mask = 1;
        // warp reduction on weights
        for (uint32_t i = 0; i < 5; i++) {
            DataType pair_weight = __shfl_sync(0xffffffff, curr_sum, threadIdx.x ^ idx_mask);
            curr_sum += pair_weight;
            idx_mask <<=1;
        }
        acc += curr_sum;
    }

    // normalize weights by the sum
    for (uint32_t w = threadIdx.x; w < count; w += warpSize) {
        if (w < count) {
            output_weights[offset + w] = input_weights[offset + w] / acc;
        }
    }
}

template<typename IndexType, bool FIXED_HOTNESS = true>
__global__ void ComputePoolingInverseMapping(
    const uint32_t hotness,
    const IndexType* __restrict__ key_offsets,
    IndexType* __restrict__ output_location_mapping,
    const uint64_t num_bags) {

    const uint32_t bag_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (bag_id < num_bags) {
        IndexType count = FIXED_HOTNESS ? hotness : key_offsets[bag_id + 1] - key_offsets[bag_id];
        IndexType pos = FIXED_HOTNESS ? bag_id * hotness : key_offsets[bag_id];
        auto location_map_p = output_location_mapping + pos;
        for (IndexType i = threadIdx.x; i<count; i+=warpSize) {
            location_map_p[i] = bag_id;
        }
    }
}

template<typename IndexType, typename DataType, typename WeightType>
__global__ void ComputePoolingGradients(
    const IndexType* __restrict__ inverse_weight_mapping,
    const IndexType* __restrict__ grad_mapping,
    const IndexType* __restrict__ offsets,
    const IndexType* __restrict__ output_loc_map,
    const DataType* __restrict__ gradients_in,
    const WeightType* __restrict__ normalized_weights,
    DataType* __restrict__ gradients_out,
    const IndexType num_keys,
    const uint32_t embedding_width) {
    const IndexType key_id = blockIdx.x * blockDim.y + threadIdx.y;
    const IndexType dst_id = output_loc_map == nullptr ? key_id : output_loc_map[key_id];

    if (key_id >= num_keys) {
        return;
    }

    const uint32_t count = offsets[key_id + 1] - offsets[key_id];

    for (uint32_t el = 0; el < embedding_width; el += warpSize) {
        DataType acc;
        nve::InitAcc(acc);

        for (uint32_t i = 0; i < count; i += warpSize) {
            IndexType weight_index = 0;
            IndexType grad_index = 0;
            WeightType weight = 0;

            if ((i + threadIdx.x) < count) {
                weight_index = inverse_weight_mapping[offsets[key_id] + i + threadIdx.x];
                grad_index = grad_mapping[weight_index];
                // TODO: always request weights, or check condition?
                weight = (normalized_weights == nullptr) ? WeightType(1.0f) : normalized_weights[weight_index];
            }

            for (uint32_t s = 0; s < warpSize; s++) {
                if ((i + s) < count) {
                    WeightType curr_weight = __shfl_sync(0xffffffff, weight, s);
                    IndexType curr_grad_index = __shfl_sync(0xffffffff, grad_index, s);

                    if ((el + threadIdx.x) < embedding_width) {
                        const DataType* src_ptr = gradients_in + curr_grad_index * embedding_width + el + threadIdx.x;
                        nve::MulAccumulate(acc, (*src_ptr), curr_weight);
                    }
                }
            }
        }
        if ((el + threadIdx.x) < embedding_width) {
            DataType* dst_ptr = gradients_out + dst_id * embedding_width + el + threadIdx.x;
            nve::AtomicAccumulate(dst_ptr, acc);
        }
    }
}

template<typename IndexType, typename DataType, typename WeightType>
void CallComputePoolingGradients(
    const IndexType* __restrict__ inverse_weight_mapping,
    const IndexType* __restrict__ grad_mapping,
    const IndexType* __restrict__ offsets,
    const IndexType* __restrict__ output_loc_map,
    const DataType* __restrict__ gradients_in,
    const WeightType* __restrict__ normalized_weights,
    DataType* __restrict__ gradients_out,
    const IndexType num_unique_keys,
    const uint32_t embedding_width,
    const cudaStream_t stream = 0)
{
    NVE_CHECK_(cudaMemsetAsync(gradients_out, 0, embedding_width * sizeof(DataType) * num_unique_keys, stream));

    const uint32_t WARPS_PER_SM = 4;
    const uint32_t SubwarpWidth = 32;
    uint32_t keys_per_warp = 1;
    uint32_t keys_per_sm = keys_per_warp * WARPS_PER_SM;
    dim3 grid_size (static_cast<uint32_t>((num_unique_keys + keys_per_sm - 1) / keys_per_sm), 1);
    dim3 block_size (SubwarpWidth, keys_per_sm);

    if ((embedding_width % 4) == 0) {
        using Vec4 = typename nve::VecWidthHelper<DataType>::Vec4;
        ComputePoolingGradients<IndexType, Vec4, WeightType><<<grid_size, block_size, 0, stream>>>(
            inverse_weight_mapping, grad_mapping, offsets, output_loc_map,
            reinterpret_cast<const Vec4*>(gradients_in), normalized_weights,
            reinterpret_cast<Vec4*>(gradients_out), num_unique_keys, embedding_width / 4);
    } else if ((embedding_width % 2) == 0) {
        using Vec2 = typename nve::VecWidthHelper<DataType>::Vec2;
        ComputePoolingGradients<IndexType, Vec2, WeightType><<<grid_size, block_size, 0, stream>>>(
            inverse_weight_mapping, grad_mapping, offsets, output_loc_map,
            reinterpret_cast<const Vec2*>(gradients_in), normalized_weights,
            reinterpret_cast<Vec2*>(gradients_out), num_unique_keys, embedding_width / 2);
    } else { 
        ComputePoolingGradients<IndexType, DataType, WeightType><<<grid_size, block_size, 0, stream>>>(
            inverse_weight_mapping, grad_mapping, offsets, output_loc_map,
            gradients_in, normalized_weights, gradients_out, num_unique_keys, embedding_width);
    }
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexT, int32_t MAX_RUN_SIZE = -1>
class GradientCalculator
{
public:
    GradientCalculator() {
        // allocate buffers
        NVE_CHECK_(cudaMallocHost(&h_num_runs_out_, sizeof(IndexT) * 2));
    }

    ~GradientCalculator() {
        NVE_CHECK_(cudaFreeHost(h_num_runs_out_));
    }

    void GetAllocRequirements(IndexT num_keys, size_t data_element_size, size_t& tmp_mem_size_device, size_t& tmp_mem_size_host) {
        deduper_.GetAllocRequirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 511) >> 9) << 9;
        size_t weights_buffer_size = data_element_size * (num_keys + 1);
        weights_buffer_size = ((weights_buffer_size + 511) >> 9) << 9;

        tmp_mem_size_device += index_buffer_size; //d_location_buffer_
        tmp_mem_size_device += weights_buffer_size; //d_normalized_weights_
        tmp_mem_size_device += index_buffer_size; //d_inverse_weight_mapping_
        tmp_mem_size_device += index_buffer_size; //d_grad_mapping_
        tmp_mem_size_device += index_buffer_size; //d_output_loc_map_
        tmp_mem_size_device += index_buffer_size; //d_dedup_offsets_
    }

    void SetAndInitBuffers(IndexT num_keys, size_t data_element_size, char* tmp_mem_device, char* tmp_mem_host) {
        char* curr_device_ptr = tmp_mem_device;
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 511) >> 9) << 9;
        size_t weights_buffer_size = data_element_size * (num_keys + 1);
        weights_buffer_size = ((weights_buffer_size + 511) >> 9) << 9;

        d_location_buffer_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        d_normalized_weights_ = reinterpret_cast<void*>(curr_device_ptr);
        curr_device_ptr += weights_buffer_size;
        d_inverse_weight_mapping_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        d_grad_mapping_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        d_output_loc_map_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        d_dedup_offsets_ = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        deduper_.SetAndInitBuffers(num_keys, curr_device_ptr, tmp_mem_host);
    }

    template <typename DataT, bool FIXED_HOTNESS = true>
    uint64_t ComputeGradients(
        const IndexT* __restrict__ keys,
        const IndexT* __restrict__ offsets,
        const DataT* __restrict__ gradients_in,
        const DataT* __restrict__ weights,
        DataT* __restrict__ gradients_out,
        IndexT* __restrict__ unique_keys_out,
        const uint32_t batch,
        const IndexT num_keys,
        const uint32_t hotness,
        const uint32_t embedding_width,
        nve::PoolingType_t pooling_type = nve::PoolingType_t::Concatenate,
        const cudaStream_t stream = 0)
    {
        // Call deduper to get unique keys and inverse weight mapping
        deduper_.Dedup(
            keys, num_keys, unique_keys_out, d_output_loc_map_, d_output_loc_map_,
            h_num_runs_out_, d_inverse_weight_mapping_, d_dedup_offsets_, stream);

        IndexT num_unique_keys = h_num_runs_out_[0];
        IndexT num_runs = h_num_runs_out_[1];

        // Compute grad mapping
        if (pooling_type != nve::PoolingType_t::Concatenate) {
            const uint32_t WARPS_PER_SM = 4;
            const uint32_t SubwarpWidth = 32;
            uint32_t grid_x = (batch + WARPS_PER_SM - 1) / WARPS_PER_SM;
            dim3 grid_size (grid_x, 1);
            dim3 block_size (SubwarpWidth, WARPS_PER_SM);

            bool normalize_weigts = (pooling_type == nve::PoolingType_t::Mean) || (pooling_type == nve::PoolingType_t::WeightedMean);
            if (normalize_weigts) {
                // TODO: need to init weights to all 1 for non weighted
                ComputeNormalizedWeights<IndexT, DataT, FIXED_HOTNESS><<<grid_size, block_size, 0, stream>>>(
                    weights, hotness, offsets, reinterpret_cast<DataT*>(d_normalized_weights_), batch);
                NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
            }

            ComputePoolingInverseMapping<IndexT, FIXED_HOTNESS><<<grid_size, block_size, 0, stream>>>(
                hotness, offsets, d_grad_mapping_, batch);
            NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error

            CallComputePoolingGradients<IndexT, DataT, DataT>(
                d_inverse_weight_mapping_, d_grad_mapping_,
                d_dedup_offsets_, 
                (MAX_RUN_SIZE == -1) ? nullptr : d_output_loc_map_,
                gradients_in,
                normalize_weigts ? reinterpret_cast<DataT*>(d_normalized_weights_) : weights,
                gradients_out, num_runs,
                embedding_width, stream);
        } else {
            CallComputeGradients<IndexT, DataT>(
                gradients_in, gradients_out, 
                reinterpret_cast<IndexT*>(unique_keys_out),
                d_inverse_weight_mapping_,
                (MAX_RUN_SIZE == -1) ? nullptr : d_output_loc_map_,
                d_dedup_offsets_,
                h_num_runs_out_,
                embedding_width,
                stream);
        }

        return num_unique_keys;
    }

private:
    IndexT* d_location_buffer_ = nullptr;
    void* d_normalized_weights_ = nullptr;
    IndexT* d_inverse_weight_mapping_ = nullptr;
    IndexT* d_output_loc_map_ = nullptr;
    IndexT* d_grad_mapping_ = nullptr;
    IndexT* d_dedup_offsets_ = nullptr;
    IndexT* h_num_runs_out_ = nullptr;

    Deduper<IndexT, MAX_RUN_SIZE> deduper_;
};
