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

#include "sph_check.h"
#include <cub/cub.cuh>
#include <cstdint>
#include <cstddef>

template<typename IndexType, int SPLIT_SIZE = 1024>
__global__ void compute_split_counts(const IndexType* __restrict__ key_counts,
                                   IndexType* __restrict__ num_split_chunks,
                                   const uint64_t num_unique_keys)
{
    const int key_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (key_id < num_unique_keys) {
        num_split_chunks[key_id] = (key_counts[key_id] + SPLIT_SIZE - 1) / SPLIT_SIZE;
    }
}

template<typename IndexT, int SPLIT_SIZE = 1024>
void call_compute_split_counts(const IndexT* key_counts, IndexT* num_split_chunks,
                            uint64_t num_counts, cudaStream_t stream)
{
    const uint32_t THREADS_PER_SM = 128;
    dim3 grid_size (static_cast<uint32_t>((num_counts + THREADS_PER_SM - 1) / THREADS_PER_SM), 1);
    dim3 block_size (THREADS_PER_SM, 1);
    compute_split_counts<IndexT, SPLIT_SIZE><<<grid_size, block_size, 0, stream>>>(
        key_counts, num_split_chunks, num_counts);
    SPH_CHECK(cudaGetLastError());
}

template<typename IndexType, int SPLIT_SIZE = 1024>
__global__ void compute_location_mapping(const IndexType* __restrict__ chunk_counts,
                                       const IndexType* __restrict__ chunk_offsets,
                                       const IndexType* __restrict__ key_offsets,
                                       IndexType* __restrict__ output_location_mapping,
                                       IndexType* __restrict__ split_key_offsets,
                                       const uint64_t num_unique_keys)
{
    const int key_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (key_id < num_unique_keys) {
        IndexType count = chunk_counts[key_id];
        auto location_map_p  = output_location_mapping + chunk_offsets[key_id];
        auto split_offsets_p = split_key_offsets + chunk_offsets[key_id];
        for (IndexType i = threadIdx.x; i < count; i += warpSize) {
            location_map_p[i]  = key_id;
            split_offsets_p[i] = key_offsets[key_id] + i * SPLIT_SIZE;
        }
        if ((key_id + 1) == num_unique_keys) {
            split_offsets_p[count] = key_offsets[key_id + 1];
        }
    }
}

template<typename IndexT, int SPLIT_SIZE = 1024>
void call_compute_location_mapping(const IndexT* chunk_counts,
                                const IndexT* chunk_offsets,
                                const IndexT* __restrict__ key_offsets,
                                IndexT* output_location_mapping,
                                IndexT* __restrict__ split_key_offsets,
                                uint64_t num_counts, cudaStream_t stream)
{
    const uint32_t WARPS_PER_SM = 4;
    dim3 grid_size (static_cast<uint32_t>((num_counts + WARPS_PER_SM - 1) / WARPS_PER_SM), 1);
    dim3 block_size (32, WARPS_PER_SM);
    compute_location_mapping<IndexT, SPLIT_SIZE><<<grid_size, block_size, 0, stream>>>(
        chunk_counts, chunk_offsets, key_offsets,
        output_location_mapping, split_key_offsets, num_counts);
    SPH_CHECK(cudaGetLastError());
}

template<typename IndexT, int32_t MAX_RUN_SIZE = -1>
class SphDeduper
{
public:
    SphDeduper() {
        SPH_CHECK(cudaMalloc(&m_d_num_runs_out, sizeof(IndexT)));
    }

    ~SphDeduper() {
        // SPH_CHECK would throw from a destructor; SPH_TEARDOWN logs instead.
        SPH_TEARDOWN(cudaFree(m_d_num_runs_out));
    }

    // end_bit: cap for cub::DeviceRadixSort. Defaults to full key width but
    //   callers should pass ceil_log2(num_buckets) when sorting bucket
    //   indices — radix sort cost is linear in (end_bit - begin_bit), so
    //   sorting 22 bits of a 64-bit key is ~3x faster than sorting all 64.
    void dedup(const IndexT* d_keys,
        const uint64_t num_keys,
        IndexT* d_unique_out,
        IndexT* d_counts_out,
        IndexT* d_loc_map_out,
        IndexT* h_num_runs_out,
        IndexT* d_inverse_buffer,
        IndexT* d_offsets,
        cudaStream_t stream,
        int end_bit = sizeof(IndexT) * 8)
    {
        SPH_CHECK(cub::DeviceRadixSort::SortPairs(
            m_d_temp_storage_sort, m_temp_storage_bytes_sort,
            d_keys, m_d_sorted_buffer, m_d_location_buffer, d_inverse_buffer,
            num_keys, 0, end_bit, stream));

        // (was: cudaMemsetAsync(d_counts_out, 0, num_keys * sizeof(IndexT)).)
        // RunLengthEncode::Encode WRITES counts at [0, num_unique_out) — no
        // accumulate, no read of pre-existing values. Memset removed.

        SPH_CHECK(cub::DeviceRunLengthEncode::Encode(
            m_d_temp_storage_encode, m_temp_storage_bytes_encode,
            m_d_sorted_buffer, d_unique_out, d_counts_out,
            m_d_num_runs_out, static_cast<int>(num_keys), stream));

        // Async copy of num_runs to host; the OUTER caller's stream sync
        // (e.g. bucketize_sph_v2b's final cudaStreamSynchronize) makes the
        // host value visible. We no longer sync inside dedup.
        SPH_CHECK(cudaMemcpyAsync(h_num_runs_out, m_d_num_runs_out,
                                   sizeof(IndexT), cudaMemcpyDefault, stream));

        IndexT* d_output_offset_buffer =
            (MAX_RUN_SIZE != -1) ? m_d_tmp_offset_buffer : d_offsets;

        // Over-scan: scan num_keys items (the upper bound on unique runs)
        // instead of num_unique_out+1, so we don't need to round-trip the
        // unique count to the host before this call. The tail of d_counts_out
        // beyond num_unique_out is uninitialized (no memset), so the
        // corresponding d_offsets tail will hold junk — but downstream only
        // reads d_offsets[0..num_unique_out), which depends solely on
        // d_counts_out[0..num_unique_out) and is computed correctly by the
        // prefix scan up to that point.
        SPH_CHECK(cub::DeviceScan::ExclusiveSum(
            m_d_temp_storage_ex_sum, m_temp_storage_bytes_ex_sum,
            d_counts_out, d_output_offset_buffer,
            static_cast<int>(num_keys), stream));

        if (MAX_RUN_SIZE != -1) {
            // This path requires num_unique_out on the host, so it has to
            // sync — kept self-contained for symmetry with the legacy API.
            // SPH instantiates SphDeduper with default MAX_RUN_SIZE = -1, so
            // this branch is never taken by hash_benchmarks.
            SPH_CHECK(cudaStreamSynchronize(stream));
            auto num_unique_out = static_cast<int>(*h_num_runs_out);

            call_compute_split_counts<IndexT, MAX_RUN_SIZE>(
                d_counts_out, m_d_split_count_buffer, num_unique_out, stream);

            SPH_CHECK(cub::DeviceScan::ExclusiveSum(
                m_d_temp_storage_ex_sum, m_temp_storage_bytes_ex_sum,
                m_d_split_count_buffer, m_d_split_offset_buffer,
                num_unique_out + 1, stream));

            SPH_CHECK(cudaMemcpyAsync(h_num_runs_out + 1,
                m_d_split_offset_buffer + num_unique_out,
                sizeof(IndexT), cudaMemcpyDefault, stream));

            call_compute_location_mapping<IndexT, MAX_RUN_SIZE>(
                m_d_split_count_buffer,
                m_d_split_offset_buffer,
                m_d_tmp_offset_buffer,
                d_loc_map_out,
                d_offsets,
                num_unique_out,
                stream);
        }
        // (was: h_num_runs_out[1] = h_num_runs_out[0]; — read of unsynced host
        // value, write OOB on the single-int64 SPH allocation. Drop entirely;
        // h_num_runs_out[1] is dead in MAX_RUN_SIZE=-1 callers.)
    }

    IndexT* get_sorted() const { return m_d_sorted_buffer; }

    void get_alloc_requirements(uint64_t num_keys,
                              size_t& tmp_mem_size_device,
                              size_t& tmp_mem_size_host)
    {
        tmp_mem_size_host   = 0;
        tmp_mem_size_device = 0;

        IndexT* p = nullptr;
        {
            SPH_CHECK(cub::DeviceRadixSort::SortPairs(p, m_temp_storage_bytes_sort,
                p, p, p, p, num_keys, 0, sizeof(IndexT) * 8));
            m_temp_storage_bytes_sort = ((m_temp_storage_bytes_sort + 511) >> 9) << 9;
            tmp_mem_size_device += m_temp_storage_bytes_sort;
        }
        {
            SPH_CHECK(cub::DeviceRunLengthEncode::Encode(
                p, m_temp_storage_bytes_encode,
                p, p, p, p, static_cast<int>(num_keys)));
            m_temp_storage_bytes_encode = ((m_temp_storage_bytes_encode + 511) >> 9) << 9;
            tmp_mem_size_device += m_temp_storage_bytes_encode;
        }
        {
            int num_keys_ = static_cast<int>(num_keys) + 1;
            SPH_CHECK(cub::DeviceScan::ExclusiveSum(
                p, m_temp_storage_bytes_ex_sum,
                p, p, num_keys_));
            m_temp_storage_bytes_ex_sum = ((m_temp_storage_bytes_ex_sum + 511) >> 9) << 9;
            tmp_mem_size_device += m_temp_storage_bytes_ex_sum;
        }
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 1023) >> 10) << 10;

        tmp_mem_size_device += index_buffer_size; // m_d_sorted_buffer
        tmp_mem_size_device += index_buffer_size; // m_d_location_buffer

        if (MAX_RUN_SIZE != -1) {
            tmp_mem_size_device += index_buffer_size; // m_d_split_count_buffer
            tmp_mem_size_device += index_buffer_size; // m_d_tmp_offset_buffer
            tmp_mem_size_device += index_buffer_size; // m_d_split_offset_buffer
        }
        tmp_mem_size_host += index_buffer_size;        // m_h_location_buffer
    }

    void set_and_init_buffers(uint64_t num_keys, char* tmp_mem_device, char* tmp_mem_host)
    {
        char* curr_device_ptr = tmp_mem_device;
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 511) >> 9) << 9;

        m_d_temp_storage_sort   = curr_device_ptr; curr_device_ptr += m_temp_storage_bytes_sort;
        m_d_temp_storage_encode = curr_device_ptr; curr_device_ptr += m_temp_storage_bytes_encode;
        m_d_temp_storage_ex_sum = curr_device_ptr; curr_device_ptr += m_temp_storage_bytes_ex_sum;
        m_d_sorted_buffer       = reinterpret_cast<IndexT*>(curr_device_ptr); curr_device_ptr += index_buffer_size;
        m_d_location_buffer     = reinterpret_cast<IndexT*>(curr_device_ptr); curr_device_ptr += index_buffer_size;

        if (MAX_RUN_SIZE != -1) {
            m_d_split_count_buffer  = reinterpret_cast<IndexT*>(curr_device_ptr); curr_device_ptr += index_buffer_size;
            m_d_tmp_offset_buffer   = reinterpret_cast<IndexT*>(curr_device_ptr); curr_device_ptr += index_buffer_size;
            m_d_split_offset_buffer = reinterpret_cast<IndexT*>(curr_device_ptr); curr_device_ptr += index_buffer_size;
        }

        m_h_location_buffer = reinterpret_cast<IndexT*>(tmp_mem_host);
        for (IndexT i = 0; i < static_cast<IndexT>(num_keys); i++) {
            m_h_location_buffer[i] = i;
        }
        SPH_CHECK(cudaMemcpy(m_d_location_buffer, m_h_location_buffer,
                              sizeof(IndexT) * num_keys, cudaMemcpyDefault));
    }

private:
    size_t   m_temp_storage_bytes_sort   = 0;
    size_t   m_temp_storage_bytes_encode = 0;
    size_t   m_temp_storage_bytes_ex_sum = 0;

    void* m_d_temp_storage_sort   = nullptr;
    void* m_d_temp_storage_encode = nullptr;
    void* m_d_temp_storage_ex_sum = nullptr;

    IndexT* m_d_location_buffer     = nullptr;
    IndexT* m_h_location_buffer     = nullptr;
    IndexT* m_d_sorted_buffer       = nullptr;
    IndexT* m_d_num_runs_out        = nullptr;
    IndexT* m_d_split_count_buffer  = nullptr;
    IndexT* m_d_tmp_offset_buffer   = nullptr;
    IndexT* m_d_split_offset_buffer = nullptr;
};
