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
#include "ec_set_associative.h"
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

namespace nve {

template<typename IndexT, typename TagT>
class CacheSADeviceModify : public EmbedCacheSA<IndexT, TagT>
{

public:
    using CounterT = float;
    using CacheConfig = typename EmbedCacheSA<IndexT, TagT>::CacheConfig;
    using ModifyEntry = typename EmbedCacheSA<IndexT, TagT>::ModifyEntry;
    using ModifyList = typename EmbedCacheSA<IndexT, TagT>::ModifyList;
    using ModifyOpType = typename EmbedCacheSA<IndexT, TagT>::ModifyOpType;
    static const uint32_t NUM_WAYS = EmbedCacheSA<IndexT, TagT>::NUM_WAYS;

    struct ModifyContext
    {
        ModifyList* d_list;
        ModifyEntry* d_entries; // device allocated entries to modify
        int8_t* d_tmp_storage;
        size_t tmp_storage_sz;
        uint32_t max_update_sz; // TODO: should be max num keys
        uint32_t table_index;
        DataTypeFormat input_type;
        DataTypeFormat output_type;
    };

public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY;

    CacheSADeviceModify(Allocator* allocator, Logger* logger, const CacheConfig& cfg) 
           : EmbedCacheSA<IndexT, TagT>(allocator, logger, cfg, TYPE)
    {}

    virtual ~CacheSADeviceModify() override = default;

    ECError modify_context_create(ModifyContextHandle& out_handle, uint32_t max_update_size) const override
    {
        ModifyContext* context = nullptr;
        try
        {
            // check erros
            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&context, sizeof(ModifyContext)));
            memset(context, 0, sizeof(ModifyContext));
            CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&context->d_list, sizeof(ModifyList)));
            CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&context->d_entries, max_update_size * sizeof(ModifyEntry)));

            context->max_update_sz = max_update_size;

            // calling ComputeSetReplaceData to calculate temp storage size
            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetReplaceData<IndexT, TagT, CounterT, NUM_WAYS, false>(
                nullptr,
                nullptr,
                nullptr,
                context->tmp_storage_sz,
                context->max_update_sz, // max number of keys
                nullptr,
                this->d_tags_,
                d_counters_,
                this->cache_,
                this->config_.embed_width_in_bytes,
                this->config_.decay_rate,
                this->num_sets_,
                context->max_update_sz,
                context->d_list,
                0)));

            CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&context->d_tmp_storage, context->tmp_storage_sz));

            out_handle.handle = (uint64_t)context;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            // deallocate everything
            if (context != nullptr) {
                if (context->d_entries != nullptr)
                    CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_entries));
                if (context->d_list != nullptr)
                    CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_list));
                if (context->d_tmp_storage != nullptr)
                    CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_tmp_storage));
                CHECK_ERR_AND_THROW(this->allocator_->host_free(context));
            }

            // can i do multiple catches
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError modify_context_destroy(ModifyContextHandle& out_handle) const override
    {
        try
        {
            CacheSADeviceModify::ModifyContext* context = (CacheSADeviceModify::ModifyContext*)out_handle.handle;
            if (!context)
            {
                return ECERROR_SUCCESS;
            }
            CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_entries));
            CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_list));
            CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_tmp_storage));
            CHECK_ERR_AND_THROW(this->allocator_->host_free(context));
            out_handle.handle = 0;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
           LOG_ERROR_AND_RETURN(e);
        }
        
    }

    ECError invalidate(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) override
    {
        try
        {
            std::lock_guard<std::mutex> lock(modify_mutex_);
            
            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto dst_tags = this->d_tags_ + table_index * this->num_sets_ * NUM_WAYS;

            ModifyList list;
            list.num_entries = 0;
            list.entries = context->d_entries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(context->d_list, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetInvalidateData<IndexT, TagT, NUM_WAYS>(
                keys, num_keys, this->num_sets_, dst_tags, context->max_update_sz,
                context->d_list, stream)));

            invalidate_tags_and_sync(context, num_keys, dst_tags, sync_event, stream);

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    CacheAllocationSize get_modify_context_size(uint32_t max_update_size) const override
    {
        constexpr size_t sz_counter_per_set = sizeof(CounterT) * NUM_WAYS;

        CacheAllocationSize ret = {0, 0};
        size_t host_size = 0;
        size_t device_size = 0;
        host_size += sizeof(ModifyContext);
        device_size += this->num_sets_ * sz_counter_per_set;
        device_size += max_update_size * sizeof(ModifyEntry);
        device_size += sizeof(uint32_t);
        ret.host_allocation_size = host_size;
        ret.device_allocation_size = device_size;
        return ret;
    }

    virtual ECError clear_cache(cudaStream_t stream) override
    {
        try 
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(this->d_tags_, INVALID_IDX, this->num_sets_*sizeof(TagT) * NUM_WAYS* this->config_.num_tables, stream));
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming keys are device accessiable
    ECError insert(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const float* priority,
        const int8_t* const* data_ptrs,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) override
    {
        try
        {
            std::lock_guard<std::mutex> lock(modify_mutex_);

            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto dst_tags = this->d_tags_ + table_index * this->num_sets_ * NUM_WAYS;
            int8_t* cache_ptr = this->cache_ + table_index * this->num_sets_ * NUM_WAYS * this->config_.embed_width_in_bytes;

            ModifyList list;
            list.num_entries = 0;
            list.entries = context->d_entries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(context->d_list, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetReplaceData<IndexT, TagT, CounterT, NUM_WAYS, false>(
                data_ptrs,
                keys,
                context->d_tmp_storage,
                context->tmp_storage_sz,
                num_keys,
                priority,
                dst_tags,
                d_counters_,
                cache_ptr,
                this->config_.embed_width_in_bytes,
                this->config_.decay_rate,
                this->num_sets_,
                context->max_update_sz,
                context->d_list,
                stream)));

            invalidate_tags_and_sync(context, num_keys, dst_tags, sync_event, stream);

            // copy data
            CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), static_cast<uint32_t>(this->config_.embed_width_in_bytes), stream)));
            // update tags
            CACHE_CUDA_ERR_CHK_AND_THROW((call_tag_update_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), dst_tags, stream)));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming keys are device accessiable
    ECError update(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t num_keys,
        uint32_t table_index,
        IECEvent* sync_event,
        cudaStream_t stream) override
    {
        try
        {   
            std::lock_guard<std::mutex> lock(modify_mutex_);

            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto dst_tags = this->d_tags_ + table_index * this->num_sets_ * NUM_WAYS;
            int8_t* cache_ptr = this->cache_ + table_index * this->num_sets_ * NUM_WAYS * this->config_.embed_width_in_bytes;

            ModifyList list;
            list.num_entries = 0;
            list.entries = context->d_entries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(context->d_list, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetUpdateData<IndexT, TagT, NUM_WAYS>(
                cache_ptr, d_values, keys, num_keys, this->num_sets_, stride, this->config_.embed_width_in_bytes, dst_tags,
                context->max_update_sz, context->d_list, stream)));

            invalidate_tags_and_sync(context, num_keys, dst_tags, sync_event, stream);

            // update data
            CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), static_cast<uint32_t>(this->config_.embed_width_in_bytes), stream)));
            // update tags
            CACHE_CUDA_ERR_CHK_AND_THROW((call_tag_update_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), dst_tags, stream)));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming keys are device accessiable
    ECError update_accumulate(
        ModifyContextHandle& modify_context_handle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t num_keys,
        uint32_t table_index,
        DataTypeFormat update_format,
        DataTypeFormat cache_format,
        IECEvent* sync_event,
        cudaStream_t stream) override
    {
        try
        {         
            std::lock_guard<std::mutex> lock(modify_mutex_);
            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto dst_tags = this->d_tags_ + table_index * this->num_sets_ * NUM_WAYS;
            int8_t* cache_ptr = this->cache_ + table_index * this->num_sets_ * NUM_WAYS * this->config_.embed_width_in_bytes;

            ModifyList list;
            list.num_entries = 0;
            list.entries = context->d_entries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(context->d_list, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetUpdateData<IndexT, TagT, NUM_WAYS>(
                cache_ptr, d_values, keys, num_keys, this->num_sets_, stride, this->config_.embed_width_in_bytes, dst_tags,
                context->max_update_sz, context->d_list, stream)));

            invalidate_tags_and_sync(context, num_keys, dst_tags, sync_event, stream);

            // update data
            if (context->input_type == nve::DATATYPE_INT8_SCALED) {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_accumulate_quantized_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), static_cast<uint32_t>(this->config_.embed_width_in_bytes), update_format, cache_format, stream)));
            } else {
                CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_accumulate_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), static_cast<uint32_t>(this->config_.embed_width_in_bytes), update_format, cache_format, stream)));
            }

            // update tags
            CACHE_CUDA_ERR_CHK_AND_THROW((call_tag_update_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_keys), dst_tags, stream)));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError get_keys_stored_in_cache(const LookupContextHandle& /*lookup_context_handle*/, IndexT* out_keys, size_t& num_out_keys) const override
    {
        try 
        {
            if (!out_keys || !this->h_tags_)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            CACHE_CUDA_ERR_CHK_AND_THROW(cudaDeviceSynchronize());
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpy(this->h_tags_, this->d_tags_, this->num_sets_ * NUM_WAYS * sizeof(TagT), cudaMemcpyDeviceToHost));
 
            num_out_keys = 0;
            for (size_t i = 0; i < this->num_sets_; i++)
            {
                for (size_t j = 0; j < NUM_WAYS; j++)
                {
                    TagT tag = this->h_tags_[i * NUM_WAYS + j];
                    if (tag == static_cast<TagT>(INVALID_IDX))
                    {
                        continue;
                    }
                    IndexT key = static_cast<IndexT>(tag * this->num_sets_ + i);
                    out_keys[num_out_keys++] = key;
                }
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

private:

    ECError invalidate_tags_and_sync(const ModifyContext* context,
                                  size_t num_entries,
                                  TagT* tags,
                                  IECEvent* sync_event,
                                  cudaStream_t stream) {
        try 
        {
            
            CACHE_CUDA_ERR_CHK_AND_THROW((call_tag_invalidate_kernel<IndexT, TagT>(context->d_list, static_cast<uint32_t>(num_entries), tags, stream)));

            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamSynchronize(stream));  
            {
                typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->custom_flow_mutex_); // this will prevent invalidation while custom flow is running and avoid multiple invalidations
                CHECK_ERR_AND_THROW(sync_event->event_record());
            }
            CHECK_ERR_AND_THROW(sync_event->event_wait_stream(stream));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    virtual size_t get_extra_device_alloc_size(uint64_t num_tables, uint64_t num_sets) const override 
    {
        size_t ctr_size = num_tables * num_sets * NUM_WAYS * sizeof(CounterT);
        return ctr_size;
    }

    virtual void init_extras_device(uint64_t num_tables, int8_t* pool, size_t space) override 
    {
        // we should have set the number of sets before calling this function
        assert(this->num_sets_ > 0);
        size_t sz = get_extra_device_alloc_size(num_tables, this->num_sets_);
        d_counters_ = (CounterT*)EmbedCacheSA<IndexT, TagT>::allocate_in_pool(pool, space, sz, 16);
        CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(d_counters_, 0, sz));
    }

    virtual size_t get_counter_size_per_set() const override {
        return CACHE_ALIGN(sizeof(CounterT) * NUM_WAYS, 16);
    }

    std::mutex modify_mutex_;
    CounterT* d_counters_; // device allocated counters to modify
};
}
