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

#include <sstream>
#include <stdexcept>
#include "cache_wrapper.h"
#include <embedding_cache_combined.cuh>
#include "cuda_ops/cuda_common.h"
#include <default_allocator.hpp>

template<typename IndexT>
ECacheWrapper<IndexT>::ECacheWrapper(nve::allocator_ptr_t allocator, std::shared_ptr<nve::Logger> logger) :
m_allocator(std::move(allocator)), m_logger(std::move(logger)), m_modify_stream(0)
{
    if (!m_allocator)
    {
        m_allocator = std::make_shared<nve::DefaultAllocator>(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD);
    }
    if (!m_logger)
    {
        m_logger = std::make_shared<nve::Logger>();
    }
    NVE_CHECK_(m_allocator && m_logger, "Failed to create cache allocator/logger");
}

template<typename IndexT>
ECacheWrapper<IndexT>::~ECacheWrapper()
{
    // wait on all streams
    NVE_CHECK_(cudaStreamSynchronize(m_modify_stream));
    for (auto s : m_lookup_streams)
    {
        NVE_CHECK_(cudaStreamSynchronize(s));
    }
    
    // destroy lookup/modify contexts (too late to check for errors here)
    m_cache->ModifyContextDestroy(m_modify_context);
    for (auto ctx : m_lookup_contexts)
    {
        m_cache->LookupContextDestroy(ctx);
    }
}

template<typename IndexT>
void ECacheWrapper<IndexT>::Init(
    uint64_t cache_size,
    uint64_t row_elements,
    nve::DataTypeFormat element_format,
    const std::vector<cudaStream_t>& lookup_streams,
    cudaStream_t modify_stream,
    uint64_t max_modify_size)
{
    // Init streams
    NVE_CHECK_(lookup_streams.size() > 0, "Missing lookup streams");
    m_lookup_streams = lookup_streams;
    m_modify_stream = modify_stream;
    m_single_stream = (m_lookup_streams.size() == 1) && (m_lookup_streams.at(0) == m_modify_stream);

    // Init cache
    NVE_CHECK_(cache_size > 0, "Invalid cache size");
    NVE_CHECK_(row_elements > 0, "Invalid row elements");
    NVE_CHECK_(max_modify_size > 0, "Invalid max modify size");
    m_max_modify_size = max_modify_size;
    m_row_elements = row_elements;
    m_element_format = element_format;
    m_cfg.cacheSzInBytes = cache_size;
    m_cfg.embedWidth = m_row_elements;
    switch (m_element_format)
    {
        case nve::DATATYPE_FP16:
            m_cfg.embedWidth *= sizeof(__half);
            break;
        case nve::DATATYPE_FP32:
            m_cfg.embedWidth *= sizeof(float);
            break;
        default:
            NVE_THROW_NOT_IMPLEMENTED_();
    }
    m_cfg.numTables = 1;  // Only supporting single table cache at this point
    m_cache = std::make_shared<CacheType>(m_allocator.get(), m_logger.get(), m_cfg);
    NVE_CHECK_(static_cast<bool>(m_cache), "Failed to create embedding cache");
    EC_CHECK(m_cache->Init());

    // Init lookup/modify contexts
    const auto num_lookups = m_lookup_streams.size();
    for (const auto& s : m_lookup_streams)
    {
        nve::LookupContextHandle ctx;
        EC_CHECK(m_cache->LookupContextCreate(ctx, nullptr, 0));
        m_lookup_contexts.push_back(ctx);
    }
    EC_CHECK(m_cache->ModifyContextCreate(m_modify_context, static_cast<uint32_t>(m_max_modify_size)));

    // Init modify event object
    m_modify_event = std::make_shared<nve::DefaultECEvent>( m_single_stream ? std::vector<cudaStream_t>() : m_lookup_streams);

}

template<typename IndexT>
void ECacheWrapper<IndexT>::Lookup(
    uint64_t stream_index,
    const IndexT* d_keys,
    const uint64_t num_keys,
    int8_t* d_values,
    uint64_t stride,
    uint64_t* d_hit_mask,
    cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    NVE_CHECK_(stream_index < m_lookup_streams.size(), "Invalid stream index");
    const auto stream = m_lookup_streams.at(stream_index);
    if (start_event) {
        NVE_CHECK_(cudaEventRecord(start_event, stream));
    }
    EC_CHECK(
        m_cache->Lookup(
            m_lookup_contexts.at(stream_index),
            d_keys,
            num_keys,
            d_values,
            d_hit_mask,
            0, // currTable
            stride,
            stream));
    if (end_event) {
        NVE_CHECK_(cudaEventRecord(end_event, stream));
    }
}

template<typename IndexT>
void ECacheWrapper<IndexT>::Insert(
    const IndexT* h_keys,
    const uint64_t num_keys,
    const int8_t* d_values,
    uint64_t stride,
    cudaEvent_t start_event,
    cudaEvent_t end_event)
    {
        nve::DefaultHistogram<IndexT> histogram(h_keys, num_keys, d_values, stride, false);
        Insert(histogram.GetKeys(), histogram.GetNumBins(), histogram.GetPriority(), histogram.GetData(), start_event, end_event);
    }

template<typename IndexT>
void ECacheWrapper<IndexT>::Insert(
    const IndexT* h_keys, 
    const uint64_t num_keys,
    const float* h_priority, 
    const int8_t* const* h_data_ptrs, 
    cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    std::lock_guard<std::mutex> lock(m_modify_lock); // Locking to guarantee single modify in flight
    if (start_event) {
        NVE_CHECK_(cudaEventRecord(start_event, m_modify_stream));
    }
    EC_CHECK(
        m_cache->Insert(
            m_modify_context,
            h_keys, 
            h_priority,
            h_data_ptrs,
            num_keys,
            0, // tableIndex
            m_modify_event.get(),
            m_modify_stream
        ));

    if (end_event) {
        NVE_CHECK_(cudaEventRecord(end_event, m_modify_stream));
    }
    NVE_CHECK_(cudaStreamSynchronize(m_modify_stream));
}

template<typename IndexT>
void ECacheWrapper<IndexT>::Update(
    const IndexT* h_keys, 
    const uint64_t num_keys,
    const int8_t* d_values,
    uint64_t stride,
    cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    std::lock_guard<std::mutex> lock(m_modify_lock); // Locking to guarantee single modify in flight
    if (start_event) {
        NVE_CHECK_(cudaEventRecord(start_event, m_modify_stream));
    }
    EC_CHECK(
        m_cache->Update(
            m_modify_context,
            h_keys, 
            d_values,
            stride,
            num_keys,
            0, // tableIndex
            m_modify_event.get(),
            m_modify_stream
            ));

    if (end_event) {
        NVE_CHECK_(cudaEventRecord(end_event, m_modify_stream));
    }
    NVE_CHECK_(cudaStreamSynchronize(m_modify_stream));
}

template<typename IndexT>
void ECacheWrapper<IndexT>::Accumulate(
    const IndexT* h_keys, 
    size_t num_keys,
    const int8_t* d_values, 
    uint64_t stride,
    nve::DataTypeFormat value_format,
    cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    std::lock_guard<std::mutex> lock(m_modify_lock); // Locking to guarantee single modify in flight
    if (start_event) {
        NVE_CHECK_(cudaEventRecord(start_event, m_modify_stream));
    }
    EC_CHECK(
        m_cache->UpdateAccumulate(
            m_modify_context,
            h_keys, 
            d_values,
            stride,
            num_keys,
            0, // tableIndex
            value_format,
            m_element_format,
            m_modify_event.get(),
            m_modify_stream));

    if (end_event) {
        NVE_CHECK_(cudaEventRecord(end_event, m_modify_stream));
    }
    NVE_CHECK_(cudaStreamSynchronize(m_modify_stream));
}

template<typename IndexT>
void ECacheWrapper<IndexT>::AccumulateNoSync(
    const IndexT* d_keys,
    const uint64_t num_keys,
    const int8_t* d_values,
    uint64_t stride,
    nve::DataTypeFormat value_format,
    cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    if (start_event) {
        NVE_CHECK_(cudaEventRecord(start_event, m_modify_stream));
    }
    EC_CHECK(m_cache->UpdateAccumulateNoSync(m_lookup_contexts.at(0), m_modify_context, d_keys, num_keys, d_values, 0, stride, value_format, value_format, m_modify_stream));
    if (end_event) {
        NVE_CHECK_(cudaEventRecord(end_event, m_modify_stream));
    }
}

template class ECacheWrapper<int32_t>;
template class ECacheWrapper<int64_t>;
