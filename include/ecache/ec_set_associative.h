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
#include "embed_cache.h"
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <memory>
#include <cassert>
#include <mutex>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <map>
#include <shared_mutex>
namespace nve {

template<typename IndexT, typename TagT>
class EmbedCacheSA;

template<typename IndexT, typename TagT>
cudaError_t call_cache_query(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t curr_table, size_t stride);

template<typename IndexT, typename TagT>
cudaError_t call_cache_query_hit_mask(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint32_t* d_hit_mask,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t curr_table, size_t stride);

template<typename IndexT, typename CacheDataT>
cudaError_t call_cache_query_uvm(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t curr_table, size_t stride);

template<typename IndexT, typename TagT>
cudaError_t call_mem_update_kernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* list, uint32_t nEnries, uint32_t row_size_in_bytes, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t call_tag_update_kernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* list, uint32_t nEnries, TagT* tags, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t call_tag_invalidate_kernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* list, uint32_t nEnries, TagT* tags, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t call_mem_update_accumulate_kernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* list, uint32_t nEnries, uint32_t row_size_in_bytes, DataTypeFormat input_format, DataTypeFormat output_format, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t call_mem_update_accumulate_no_sync_kernel(const IndexT* d_keys, const size_t len, const int8_t* d_values, typename EmbedCacheSA<IndexT, TagT>::CacheData data, uint32_t curr_table, size_t stride, DataTypeFormat input_format, DataTypeFormat output_format, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t call_update_accumulate_no_sync_fused_with_pipeline( const int8_t* grads, 
                    int8_t* dst, 
                    const IndexT* keys, 
                    IndexT num_keys,
                    uint32_t row_size_in_bytes,
                    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
                    cudaStream_t stream);


template<typename IndexT, typename TagT>
cudaError_t call_mem_update_accumulate_quantized_kernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* list, uint32_t row_size_in_bytes, DataTypeFormat input_format, DataTypeFormat output_format, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t call_sort_gather( const int8_t* uvm, 
                    int8_t* dst, 
                    const IndexT* keys, 
                    int8_t* auxBuf,
                    size_t num_keys,
                    size_t row_size_in_bytes,
                    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
                    cudaStream_t stream);

#define INVALID_IDX -1 // using define to be able to cast to whatever needed

#define LOG_ERROR_AND_RETURN(ex) { if (this->logger_) { this->logger_->log(LogLevel_t::Error, ex.what());} return ex.m_err;}

template<typename IndexT, typename TagT>
class EmbedCacheSA : public EmbedCacheBase<IndexT>
{
public:
    using CounterT = float;

    struct CacheData
    {
        int8_t* cache_ptr; 
        const int8_t* tags_ptr;
        uint32_t num_sets;
        uint64_t row_size_in_bytes;
        bool count_misses;
        int64_t* misses;
    };

    struct CacheConfig
    {
        // TRTREC-78 change cache_sz_in_bytes to two config ec device size and ec host size (when allocating on host, size in bytes is not well defined)
        size_t cache_sz_in_bytes = 0; // total size of storage, tags should not exceed this
        uint64_t embed_width_in_bytes = 0;
        uint64_t num_tables = 1;
        float decay_rate = 0.95f; 
        bool allocate_data_on_host = false;
    };

    struct ModifyEntry
    {
        const int8_t* src;
        int8_t* dst;
        uint32_t set;
        uint32_t way;
        TagT tag;
    };

    enum ModifyOpType : uint32_t
    {
        MODIFY_REPLACE,
        MODIFY_UPDATE,
        MODIFY_UPDATE_ACCUMLATE,
        MODIFY_INVALIDATE
    };
    
    struct ModifyList
    {
        ModifyEntry* entries;
        uint32_t num_entries;
    };

public:

    ECError calc_allocation_size(CacheAllocationSize& out_allocation_sz) const
    {
        uint64_t num_sets = calc_num_sets();
        
        if (num_sets == 0)
        {
            return ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL;
        }
        const size_t sz_tags_per_set = get_tag_size_per_set();
        const size_t sz_entry_per_set = get_entry_size_per_set();

        size_t sz_host_per_set = sz_tags_per_set;
        size_t sz_device_per_set = sz_tags_per_set;
        if (config_.allocate_data_on_host) { 
            sz_host_per_set += sz_entry_per_set;
        }
        else {
            sz_device_per_set += sz_entry_per_set;
        }
        
        out_allocation_sz.device_allocation_size = (num_sets * sz_device_per_set) * config_.num_tables + get_extra_device_alloc_size(config_.num_tables, num_sets);
        out_allocation_sz.host_allocation_size = (num_sets * sz_host_per_set ) * config_.num_tables + get_extra_host_alloc_size(config_.num_tables, num_sets);

        // if our calculation are correct this shouldn't happen
        assert(out_allocation_sz.device_allocation_size <= config_.cache_sz_in_bytes);
        assert((num_sets * sz_host_per_set) * config_.num_tables <= config_.cache_sz_in_bytes);
        
        return ECERROR_SUCCESS;
    }

    EmbedCacheSA(Allocator* allocator, Logger* logger, const CacheConfig& cfg, CACHE_IMPLEMENTATION_TYPE type)
        : EmbedCacheBase<IndexT>(allocator, logger, type), config_(cfg), num_sets_(0), h_pool_(nullptr), d_pool_(nullptr),
          cache_(nullptr), d_tags_(nullptr), h_tags_(nullptr), custom_flow_lock_(custom_flow_mutex_, std::defer_lock)
    {

    }

    ECError lookup_context_create(LookupContextHandle& out_handle, const PerformanceMetric* metrics, size_t num_metrics) const override
    {
        try 
        {
            EmbedCacheSA::CacheData* data;
            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&data, sizeof(EmbedCacheSA::CacheData)));
            memset(data, 0, sizeof(CacheData));
            data->row_size_in_bytes = config_.embed_width_in_bytes;
            data->cache_ptr = cache_;
            data->tags_ptr = (const int8_t*)d_tags_;
            data->num_sets = static_cast<decltype(data->num_sets)>(num_sets_);
            for (size_t i = 0; i < num_metrics; i++)
            {
                if (metrics->type == MERTIC_COUNT_MISSES)
                {
                    data->count_misses = true;
                    data->misses = metrics[i].d_val;
                }
            }
            out_handle.handle = (uint64_t)data;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }


    ECError lookup_context_destroy(LookupContextHandle& handle) const override
    {
        EmbedCacheSA::CacheData* p = (EmbedCacheSA::CacheData*)handle.handle;
        handle.handle = 0;

        ECError ret = this->allocator_->host_free(p);
        return ret;
    }

    // performance 
    ECError performance_metric_create(PerformanceMetric& out_metric, PerformanceMerticTypes type) const override
    {
        try 
        {
            switch (type)
            {
            case MERTIC_COUNT_MISSES:
            {
                CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&out_metric.d_val, sizeof(uint64_t)));
                out_metric.type = type;
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(out_metric.d_val, 0, sizeof(uint64_t)));
                return ECERROR_SUCCESS;
            }
            default:
                return ECERROR_NOT_IMPLEMENTED;
            }
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
        
    }
    
    ECError performance_metric_destroy(PerformanceMetric& metric) const override
    {
        try
        {
            switch (metric.type)
            {
            case MERTIC_COUNT_MISSES:
            {
                ECError ret = this->allocator_->device_free(metric.d_val);
                metric.d_val = nullptr;
                return ret;
            }
            default:
                return ECERROR_NOT_IMPLEMENTED;
            }
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError performance_metric_get_value(const PerformanceMetric& metric, int64_t* out_value, cudaStream_t stream) const override
    {
        try
        {
            switch (metric.type)
            {
            case MERTIC_COUNT_MISSES:
            {
                if (!metric.d_val)
                {
                    EC_THROW(ECERROR_INVALID_ARGUMENT);
                }
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(out_value, metric.d_val, sizeof(metric.d_val[0]), cudaMemcpyDefault, stream));
                return ECERROR_SUCCESS;
            }
            default:
                EC_THROW(ECERROR_NOT_IMPLEMENTED);
            }
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError performance_metric_reset(PerformanceMetric& metric, cudaStream_t stream) const override
    {
        try
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(metric.d_val, 0, sizeof(metric.d_val[0]), stream));
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    virtual ECError modify_context_create(ModifyContextHandle& out_handle, uint32_t max_update_size) const override = 0;

    virtual ECError modify_context_destroy(ModifyContextHandle& out_handle) const override = 0;

    virtual ECError invalidate(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) override = 0;

    virtual ~EmbedCacheSA() override
    {
        this->allocator_->host_free(h_pool_);
        this->allocator_->device_free(d_pool_);
    }
    
    virtual ECError init() override
    {
        try
        {
            if (!this->allocator_)
            {
                EC_THROW(ECERROR_BAD_ALLOCATOR);
            }

            if (!this->logger_)
            {
                EC_THROW(ECERROR_BAD_LOGGER);
            }

            CacheAllocationSize alloc_sz = {0, 0}; 
            calc_allocation_size(alloc_sz);

            num_sets_ = calc_num_sets();

            if (num_sets_ == 0)
            {
                EC_THROW(ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL);
            }
            
            // do a small check for KeyTag configuration and issue a warning 
            check_tag_key_config();

            const size_t sz_tags_per_set = get_tag_size_per_set(); // this already aligned to 16
            const size_t tag_size = sz_tags_per_set * num_sets_ * config_.num_tables;
            const size_t sz_entry_per_set = get_entry_size_per_set();
            const size_t data_size = sz_entry_per_set * num_sets_ * config_.num_tables;

            // allocate memory pools
            CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&d_pool_, alloc_sz.device_allocation_size));
            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&h_pool_, alloc_sz.host_allocation_size));

            int8_t* pd = d_pool_;
            const int8_t* ed = pd + alloc_sz.device_allocation_size;
            int8_t* ph = h_pool_;
            const int8_t* eh = ph + alloc_sz.host_allocation_size;

            // allocating pointers in memory pools
            // cactch bad alighments
            if(config_.allocate_data_on_host){
                // for gpu managed host cache config- allocate data on host. 
                cache_ = allocate_in_pool(ph, eh - ph, data_size, 16);
            }
            else{
                // for normal gpu cache config- allocate as usual.
                cache_ = allocate_in_pool(pd, ed - pd, data_size, 16);
            }
            d_tags_ = (TagT*)allocate_in_pool(pd, ed - pd, tag_size, 16);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(d_tags_, INVALID_IDX, tag_size));

            // always allocate host buffer for tags
            // in gpu_modify mode it is required for get_keys_stored_in_cache method
            h_tags_ = (TagT*)allocate_in_pool(ph, eh - ph, tag_size, 16);

            std::fill(h_tags_, h_tags_ + num_sets_ * NUM_WAYS * config_.num_tables, static_cast<TagT>(INVALID_IDX));
            
            init_extras_host(config_.num_tables, ph, eh - ph);
            init_extras_device(config_.num_tables, pd, ed - pd);

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            if(h_pool_){
                this->allocator_->host_free(h_pool_);
            }
            if(d_pool_){
                this->allocator_->device_free(d_pool_);
            }
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // return CacheData for Address Calcaulation each template should implement its own
    CacheData get_cache_data(const LookupContextHandle& handle) const
    {
        CacheData* cd = (CacheData*)handle.handle;
        // if cd is nullptr we have a problem, but i hate for this function to take a reference, it will create code ugly lines, please don't pass nullptr here
        return *cd;
    }

    ECError lookup(const LookupContextHandle& h_lookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t curr_table, size_t stride, cudaStream_t stream) override
    {
        try{
            auto data = get_cache_data(h_lookup);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(d_missing_len, 0, sizeof(*d_missing_len), stream));
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_cache_query<IndexT, TagT>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, stream, curr_table, stride)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError lookup(const LookupContextHandle& h_lookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_hit_mask,
                                            uint32_t curr_table, size_t stride, cudaStream_t stream) override
    {
        try 
        {
            auto data = get_cache_data(h_lookup);
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_cache_query_hit_mask<IndexT, TagT>(d_keys, len, d_values, reinterpret_cast<uint32_t*>(d_hit_mask), data, stream, curr_table, stride)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError lookup(const LookupContextHandle& h_lookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, uint32_t curr_table, 
                                            size_t stride, cudaStream_t stream) override
    {
        try 
        {
            auto data = get_cache_data(h_lookup);
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_cache_query_uvm<IndexT>(d_keys, len, d_values, d_table, data, stream, curr_table, stride)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError start_custom_flow() override {
        custom_flow_lock_.lock();
        return ECERROR_SUCCESS;
    }

    ECError end_custom_flow() override {
        custom_flow_lock_.unlock();
        return ECERROR_SUCCESS;
    }

    ECError lookup_sort_gather(const LookupContextHandle& h_lookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, int8_t* d_auxiliary_buffer, size_t& auxiliary_buffer_bytes, uint32_t /*curr_table*/, 
                                            size_t stride, cudaStream_t stream) override
    {
        try
        {
            auto data = get_cache_data(h_lookup);
            // first calculate the aux buf required size
            size_t required_bytes = 0;
            CACHE_CUDA_ERR_CHK_AND_THROW((call_sort_gather<IndexT, TagT>(d_table, 
                    d_values, 
                    d_keys, 
                    nullptr,
                    required_bytes,
                    len,
                    stride,
                    data,
                    stream)));
            
            // if aux is null return its required size
            if (d_auxiliary_buffer == nullptr)
            {
                auxiliary_buffer_bytes = required_bytes;
                return ECERROR_SUCCESS;
            }
            else
            {
                if (auxiliary_buffer_bytes < required_bytes)
                {
                    EC_THROW(ECERROR_INVALID_ARGUMENT);
                }
                
                if (len > 0)
                {
                    CHECK_ERR_AND_THROW(start_custom_flow());
                    CACHE_CUDA_ERR_CHK_AND_THROW((call_sort_gather<IndexT, TagT>( d_table, 
                    d_values, 
                    d_keys, 
                    d_auxiliary_buffer,
                    auxiliary_buffer_bytes,
                    len,
                    stride,
                    data,
                    stream)));
                    CHECK_ERR_AND_THROW(end_custom_flow());
                }

                return ECERROR_SUCCESS;
            }
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError update_accumulate_no_sync(const LookupContextHandle& h_lookup, ModifyContextHandle& /*hModify*/, const IndexT* d_keys, const size_t len, const int8_t* d_values, uint32_t curr_table, size_t stride, DataTypeFormat input_format, DataTypeFormat output_format, cudaStream_t stream) override
    {
        try
        {
            auto data = get_cache_data(h_lookup);
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_accumulate_no_sync_kernel<IndexT, TagT>(d_keys, len, d_values, data, curr_table, stride, input_format,output_format, stream)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError update_accumulate_no_sync_fused(const LookupContextHandle& h_lookup, ModifyContextHandle& /*hModify*/, const IndexT* d_keys, const size_t len, const int8_t* d_values, uint32_t /*curr_table*/, size_t stride, 
                                            DataTypeFormat input_format, DataTypeFormat output_format, int8_t* uvm, cudaStream_t stream) override
    {
        try
        {
            auto data = get_cache_data(h_lookup);
            if (input_format != output_format || input_format != DATATYPE_FP32 || stride != 512)
            {
                return ECERROR_NOT_IMPLEMENTED;
            }

            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_update_accumulate_no_sync_fused_with_pipeline<IndexT,TagT>(d_values, 
                        uvm, 
                        d_keys, 
                        (IndexT)len,
                        (uint32_t)stride,
                        data,
                        stream)));
            }
            return ECERROR_SUCCESS; 
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }


    size_t get_max_num_embedding_vectors_in_cache() const override
    {
        return num_sets_ * NUM_WAYS;
    }

    virtual ECError get_keys_stored_in_cache(const LookupContextHandle& /*lookup_context_handle*/, IndexT* out_keys, size_t& num_out_keys) const override = 0;
    
    CacheAllocationSize get_lookup_context_size() const override
    {
        CacheAllocationSize ret = {0, 0};
        ret.host_allocation_size = sizeof(CacheData);
        return ret;
    }

    virtual CacheAllocationSize get_modify_context_size(uint32_t max_update_size) const override = 0;

    virtual ECError clear_cache(cudaStream_t stream) override = 0;

    // assuming keys are either host or device accessiable
    // depending on cache type
    virtual ECError insert(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const float* priority,
        const int8_t* const* data,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) override = 0;

    // assuming keys are either host or device accessiable
    // depending on cache type
    virtual ECError update(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) override = 0;

    // assuming keys are either host or device accessiable
    // depending on cache type
    virtual ECError update_accumulate(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t num_keys,
        uint32_t table_index,
        DataTypeFormat update_format,
        DataTypeFormat cache_format,
        IECEvent* sync_event,
        cudaStream_t stream) override = 0;

private:

    void check_tag_key_config() const
    {
        constexpr IndexT max_key = std::numeric_limits<IndexT>::max();

        // currently handling float keys and float tags isn't supported and will cause problems
        if (!std::numeric_limits<TagT>::is_integer || !std::numeric_limits<IndexT>::is_integer)
        {
            EC_THROW(ECERROR_BAD_KEY_TAG_CONFIG);
        }

        // might have issues with sign mismatch between key and tag
        if (std::numeric_limits<TagT>::is_signed != std::numeric_limits<IndexT>::is_signed)
        {
            EC_THROW(ECERROR_BAD_KEY_TAG_CONFIG);
        }

        if (std::numeric_limits<TagT>::max() < max_key)
        {
            Log(LogLevel_t::Warning, "Maximum supported key is %llu", static_cast<uint64_t>(std::numeric_limits<TagT>::max())*num_sets_ + (num_sets_-1));
        }
    }

    void Log(LogLevel_t verbosity, const char* format, ...) const
    {
        char buf[EC_MAX_STRING_BUF] = {0};
        va_list args;
        va_start(args, format);
        std::vsnprintf(buf, EC_MAX_STRING_BUF, format, args);
        va_end(args);
        assert(this->logger_); // shouldn't be initilized if logger_ == nullptr
        this->logger_->log(verbosity, buf);
    }

    // Helper method to calculate tag size per set
    size_t get_tag_size_per_set() const {
        return CACHE_ALIGN(sizeof(TagT) * NUM_WAYS, 16);
    }

    // Helper method to calculate entry size per set
    size_t get_entry_size_per_set() const {
        return CACHE_ALIGN(config_.embed_width_in_bytes * NUM_WAYS, 16);
    }

    // Helper method to calculate counter size per set
    virtual size_t get_counter_size_per_set() const {
        return 0;
    }

    uint64_t calc_num_sets() const {
        const size_t sz_per_set = get_tag_size_per_set() + get_entry_size_per_set() + get_counter_size_per_set();
        uint64_t nSets = (config_.cache_sz_in_bytes / config_.num_tables ) / (sz_per_set);
        return nSets;
    }

protected:

    int8_t* allocate_in_pool(int8_t*& p, size_t space, size_t sz, size_t align = 0) const
    {
        int8_t* ret = nullptr;
        if (align > 0)
        {
            if (std::align(align, sz, (void*&)p, space))
            {
                ret = p;
                p += sz;
            }
            else
            {
                // TRTREC-79 handle error and check for failure in the caller
            }
        }
        else
        {
            // should it be <= or < ?
            if (sz <= space)
            {
                ret = p;
                p += sz;
            }
            else
            {
                // TRTREC-79 handle error and check for failure in the caller
            }
        }
        return ret;
    }

    virtual size_t get_extra_device_alloc_size(uint64_t /*num_tables*/, uint64_t /*numSets*/) const { return 0; }
    virtual size_t get_extra_host_alloc_size(uint64_t /*num_tables*/, uint64_t /*numSets*/) const { return 0; }
    virtual void init_extras_host(uint64_t /*num_tables*/, int8_t* /*pool*/, size_t /*space*/) {}
    virtual void init_extras_device(uint64_t /*num_tables*/, int8_t* /*pool*/, size_t /*space*/) {}

    using ReadWriteLock =  std::shared_mutex;
    using WriteLock =  std::unique_lock<ReadWriteLock>; 
    using ReadLock =  std::shared_lock<ReadWriteLock>;  

    CacheConfig config_;
    uint64_t num_sets_;
    int8_t* h_pool_; // host memory pool for cache internal buffers- one free to rule them all
    int8_t* d_pool_; // device memory pool for cache internal buffers- one free to rule them all
    int8_t* cache_; // cache data storage
    TagT* d_tags_; // device tags
    TagT* h_tags_; // host copy of tags
    ReadWriteLock custom_flow_mutex_; // mutex for custom flow allowing for read write lock
    ReadLock custom_flow_lock_; // dexplicit reader lock for defered locking via custom flow calls

public:
    static const uint32_t NUM_WAYS = 8;
};
}