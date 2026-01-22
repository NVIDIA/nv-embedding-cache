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

#include <vector>
#include <memory>
#include <mutex>
#include <nve_types.hpp>
#include <embedding_cache_combined.h>

template<typename IndexT>
class ECacheWrapper
{
public:
    /**
     * Constructor.
     * @param allocator - Allocator object. nullptr implies nve::DefaultAllocator
     * @param logger - Logger object. nullptr implies nve::Logger (empty log)
    */
    ECacheWrapper(nve::allocator_ptr_t allocator = {}, std::shared_ptr<nve::Logger> logger = {});
    /**
     * Destructor
    */
    ~ECacheWrapper();

    /**
     * Initialize the cache wrapper.
     * 
     * @param cache_size - Number of bytes to allocate to the cache (data vectors and metadata)
     * @param row_elements - Number of elements in each row (datavector)
     * @param element_format - Format of the datavector elements.
     * @param lookup_streams - CUDA streams to use during Lookup() calls.
     * @param modify_stream - CUDA stream to use for cache modifications (insert/update/accumulate)
     * @param max_modify_size - Maximal amount of entries to be handled by a single modify operation (insert/update/accumulate)
     * 
     * @note Using a single stream for lookup and modify will serialize all cache ops, removing sync overheads.
     *       In this case, parallelism can only be achieved between different caches (and not between lookups of the same cache).
     * 
     * @note On failure, std::runtime_error will be thrown.
    */
    void Init(
        uint64_t cache_size,
        uint64_t row_elements,
        nve::DataTypeFormat element_format,
        const std::vector<cudaStream_t>& lookup_streams,
        cudaStream_t modify_stream,
        uint64_t max_modify_size = (1u<<16)
    );

    /**
     * Get the lookup stream for a given index
     * 
     * @returns CUDA stream associated with the given index.
    */
    const std::vector<cudaStream_t>& GetLookupStreams() const { return m_lookup_streams; }
    /**
     * Get the modify stream
     * 
     * @returns CUDA stream.
    */
    cudaStream_t GetModifyStream() const { return m_modify_stream; }
    /**
     * Get the cache allocator
     * 
     * @returns allocator_ptr_t
    */
    nve::allocator_ptr_t GetAllocator() const { return m_allocator; }
    
    /**
     * Retrieve the data vectors from the cache. If a given key is not resident, no data will be written to it's position in "d_values"
     * 
     * @param stream_index - Index of the lookup stream to use.
     * @param d_keys - Array of keys of the data vectors to be retrieved.
     *                 This array should be in UVM accessible memory (preferably in GPU mem)
     * @param num_keys - Number of keys in "d_keys"
     * @param d_values - Pointer to the output buffer. The datavector for key 'i' will be written to d_values[stride * i].
     *                   Datavecotrs not resident in the cache will be left untouched in the buffer.
     *                   This buffer should be in UVM accessible memory (preferably in GPU mem)
     * @param stride - Stride to use when when writing to d_values, i.e. the offset in bytes between consecutive data vectors.
     * @param d_hit_mask - Bit array (in uint64_t elements) to signal which key was retrieved from the cache.
     *                     The value of the i'th bit will be 1 if the data vector for d_keys[i] was retrieved successfully and 0 otherwise.
     *                     Must be of size ((num_keys + 63) / 64) * sizeof(uint64_t).
     *                     This buffer should be in UVM accessible memory (preferably in GPU mem)
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note This function returns after a CUDA kernel was launched for the relevant lookup stream (check with GetLookupStream(stream_index)).
     *       The outputs can be read after said stream completed (check with cudaStreamSynchronize or a cudaEvent)
    */
    void Lookup(
        uint64_t stream_index,
        const IndexT* d_keys,
        const uint64_t num_keys,
        int8_t* d_values,
        uint64_t stride,
        uint64_t* d_hit_mask,
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    /**
     * Insert datavectors to the cache.
     * The function expects a representative set of indices (with repititions) that will be analized with the current state of the cache
     * to modify the cache state adding some datavectors and removing others.
     * 
     * The function does not guarantee all given keys will be stored in cache even if there is enough storage space.
     * Implementation details such as set size may prevent some data vectors from being stored.
     * 
     * @param h_keys - Array of keys to consider for insertion.
     *                 This array must be in host memory.
     * @param num_keys - Number of keys in "h_keys"
     * @param d_values - The datavectors for the given keys. The vector for key i should be in d_values[i * vector_size]
     *                   vector_size will be calculated as sizeof(element_type) * row_elements.
     *                   This buffer should be in UVM accessible memory (preferably in GPU mem)
     * @param stride - Stride to use when when reading from d_values, i.e. the offset in bytes between consecutive data vectors.
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note This function returns after a CUDA kernel was launched for the modify stream.
    */
    void Insert(
        const IndexT* h_keys,
        const uint64_t num_keys,
        const int8_t* d_values,
        uint64_t stride,
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    /**
     * Insert datavectors to the cache.
     * The function expects a triplets of (key, priority, ptr_to_data) that will be analized with the current state of the cache
     * to modify the cache state adding some datavectors and removing others.
     * The function expects the arrays to be sorted in a non descending orders according to the prioirty array
     * 
     * The function does not guarantee all given keys will be stored in cache even if there is enough storage space.
     * Implementation details such as set size may prevent some data vectors from being stored.
     * 
     * @param h_keys - Array of keys to consider for insertion.
     *                 This array must be in host memory.
     * @param num_keys - Number of keys in "h_keys" (also the size of h_priority and h_data_ptrs)
     * @param h_priority - Array of priorities for a the set of keys, which will be considered against the current cache, 
     *                   higher priority is more important.
     * @param h_data_ptrs - Array of pointers to the datavectors for the given keys. The pointer to a vector for key i should be in data_ptrs[i]
     *                      This buffer should be in host accessible memory, the values in the array should point to UVM accessible memory (preferably in GPU mem)
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note This function returns after a CUDA kernel was launched for the modify stream.
    */
    void Insert(
        const IndexT* h_keys, 
        const uint64_t num_keys,
        const float* h_priority, 
        const int8_t* const* h_data_ptrs, 
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);
    
    /**
     * Update datavectors already resident in the cache.
     * The function will overwrite the datavector for the given keys iff an older version was already resident in the cache.
     * If (h_keys[i] is in cache) values[i] = d_values[i];
     * 
     * @param h_keys - Array of keys to update.
     *                 This array must be in host memory.
     * @param num_keys - Number of keys in "h_keys"
     * @param d_values - The datavectors for the given keys. The vector for key i should be in d_values[i * vector_size]
     *                   vector_size will be calculated as sizeof(element_type) * row_elements.
     *                   This buffer should be in UVM accessible memory (preferably in GPU mem)
     * @param stride - Stride to use when when reading from d_values, i.e. the offset in bytes between consecutive data vectors.
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note When this function returns, a CUDA kernel was already launched for the modify stream.
    */
    void Update(
        const IndexT* h_keys,
        const uint64_t num_keys,
        const int8_t* d_values,
        uint64_t stride,
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    /**
     * Update datavectors already resident in the cache.
     * The function will overwrite the datavector for the given keys iff an older version was already resident in the cache.
     * If (h_keys[i] is in cache) values[i] = *data_ptrs[i];
     * 
     * @param h_keys - Array of keys to update.
     *                 This array must be in host memory.
     * @param num_keys - Number of keys in "h_keys" (also the size of h_data_ptrs)
     * @param h_data_ptrs - Array of pointers to the datavectors for the given keys. The pointer to vector for key i should be in data_ptrs[i * vector_size]
     *                      This buffer should be in host accessible memory, the values in the array should point to UVM accessible memory (preferably in GPU mem)
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note When this function returns, a CUDA kernel was already launched for the modify stream.
    */
    void Update(
        const IndexT* h_keys, 
        const uint64_t num_keys,
        const int8_t* const* h_data_ptrs, 
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    /**
     * Accumulate given datavectors into those already resident in the cache.
     * The function will accumulate the datavector for the given keys iff they are already resident in the cache.
     * 
     * @param h_keys - Array of keys to update.
     *                 This array must be in host memory.
     * @param num_keys - Number of keys in "h_keys"
     * @param d_values - The datavectors for the given keys. The vector for key i should be in d_values[i * vector_size]
     *                   vector_size will be calculated as sizeof(element_type) * row_elements.
     *                   This buffer should be in UVM accessible memory (preferably in GPU mem)
     * @param stride - Stride to use when when reading from d_values, i.e. the offset in bytes between consecutive data vectors.
     * @param value_format - The data format of d_values (can be different from the format of the datavectors in the cache),
     *                       e.g. value_format can be DATATYPE_FP16 while the cache was initialized with DATATYPE_FP32.
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note When this function returns, a CUDA kernel was already launched for the modify stream.
    */
    void Accumulate(
        const IndexT* h_keys,
        const uint64_t num_keys,
        const int8_t* d_values,
        uint64_t stride,
        nve::DataTypeFormat value_format,
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    /**
     * Accumulate given pointers to datavectors into those already resident in the cache.
     * The function will accumulate the datavector for the given keys iff they are already resident in the cache.
     * 
     * @param h_keys - Array of keys to update.
     *                 This array must be in host memory.
     * @param num_keys - Number of keys in "h_keys"
     * @param h_data_ptrs - Array of pointers to the datavectors for the given keys. The pointer to vector for key i should be in data_ptrs[i * vector_size]
     *                      This buffer should be in host accessible memory, the values in the array should point to UVM accessible memory (preferably in GPU mem)
     * @param value_format - The data format of d_values (can be different from the format of the datavectors in the cache),
     *                       e.g. value_format can be DATATYPE_FP16 while the cache was initialized with DATATYPE_FP32.
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note When this function returns, a CUDA kernel was already launched for the modify stream.
    */
    void Accumulate(
        const IndexT* h_keys, 
        const uint64_t num_keys,
        const int8_t* const* h_data_ptrs, 
        nve::DataTypeFormat value_format,
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    /**
     * Accumulate given datavectors into those already resident in the cache.
     * The function will accumulate the datavector for the given keys iff they are already resident in the cache.
     * 
     * @param d_keys - Array of keys to update.
     *                 This array should be in UVM accessible memory (preferably in GPU mem).
     * @param num_keys - Number of keys in "d_keys"
     * @param d_values - The datavectors for the given keys. The vector for key i should be in d_values[i * vector_size]
     *                   vector_size will be calculated as sizeof(element_type) * row_elements.
     *                   This buffer should be in UVM accessible memory (preferably in GPU mem)
     * @param stride - Stride to use when when reading from d_values, i.e. the offset in bytes between consecutive data vectors.
     * @param value_format - The data format of d_values (can be different from the format of the datavectors in the cache),
     *                       e.g. value_format can be DATATYPE_FP16 while the cache was initialized with DATATYPE_FP32.
     * @param start_event - Cuda event to signal before launching the GPU kernel
     * @param end_event - Cuda event to signal after launching the GPU kernel
     * 
     * @note When this function returns, a CUDA kernel was already launched for the modify stream.
     * @warning This function does not sync with inflight lookups (use this in single stream scenario)
    */
    void AccumulateNoSync(
        const IndexT* d_keys,
        const uint64_t num_keys,
        const int8_t* d_values,
        uint64_t stride,
        nve::DataTypeFormat value_format,
        cudaEvent_t start_event = nullptr,
        cudaEvent_t end_event = nullptr);

    using CacheType = nve::CacheSAHostModify<IndexT, IndexT>;
    typename CacheType::CacheConfig GetConfig() const { return m_cfg; }
    std::shared_ptr<CacheType> GetCache() const { return m_cache; }
    uint64_t GetMaxModifySize() const { return m_max_modify_size; }
    nve::DataTypeFormat GetElementFormat() const { return m_element_format; }

private:
    uint64_t m_row_elements{0};
    nve::DataTypeFormat m_element_format{nve::DataTypeFormat::NUM_DATA_TYPES_FORMATS};


    std::shared_ptr<CacheType> m_cache;
    typename CacheType::CacheConfig m_cfg;

    // CUDA streams
    std::vector<cudaStream_t> m_lookup_streams;
    cudaStream_t m_modify_stream;
    bool m_single_stream{false};

    // Cache contexts
    std::vector<nve::LookupContextHandle> m_lookup_contexts;
    nve::ModifyContextHandle m_modify_context;
    uint64_t m_max_modify_size{0};

    std::shared_ptr<nve::DefaultECEvent> m_modify_event;

    nve::allocator_ptr_t m_allocator;
    std::shared_ptr<nve::Logger> m_logger;
    
    // Lock to ensure only a single modify is in flight at any given point.
    std::mutex m_modify_lock;

};
