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
class CacheSAHostModify : public EmbedCacheSA<IndexT, TagT>
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
        ModifyList* list;
        ModifyList* d_list; // device allocated list of entries to modify
        ModifyEntry* d_entries;
        uint32_t max_update_sz;
        uint32_t table_index;
        ModifyOpType op;
        DataTypeFormat input_type;
        DataTypeFormat output_type;
    };

public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA;

    CacheSAHostModify(Allocator* allocator, Logger* logger, const CacheConfig& cfg) 
           : EmbedCacheSA<IndexT, TagT>(allocator, logger, cfg, TYPE), h_counters_(nullptr)
    {}

    virtual ~CacheSAHostModify() override = default;

    ECError modify_context_create(ModifyContextHandle& out_handle, uint32_t max_update_size) const override
    {
        ModifyContext* context = nullptr;
        try
        {
            // check erros
            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&context, sizeof(ModifyContext)));
            memset(context, 0, sizeof(ModifyContext));

            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&context->list, sizeof(ModifyList)));
            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&context->list->entries, max_update_size * sizeof(ModifyEntry)));

            CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&context->d_list, sizeof(ModifyList)));
            CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&context->d_entries, max_update_size * sizeof(ModifyEntry)));

            context->max_update_sz = max_update_size;
            context->list->num_entries = 0;

            out_handle.handle = (uint64_t)context;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            //deallocate everything
            if (context != nullptr) {
                if (context->list != nullptr) {
                    if (context->list->entries != nullptr)
                        CHECK_ERR_AND_THROW(this->allocator_->host_free(context->list->entries));
                    CHECK_ERR_AND_THROW(this->allocator_->host_free(context->list));
                }
                if (context->d_entries != nullptr)
                    CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_entries));
                if (context->d_list != nullptr)
                    CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_list));
                
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
            CacheSAHostModify::ModifyContext* context = (CacheSAHostModify::ModifyContext*)out_handle.handle;
            if (!context)
            {
                return ECERROR_SUCCESS;
            }
            CHECK_ERR_AND_THROW(this->allocator_->host_free(context->list->entries));
            CHECK_ERR_AND_THROW(this->allocator_->host_free(context->list));
            CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_entries));
            CHECK_ERR_AND_THROW(this->allocator_->device_free(context->d_list));
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
            // grab writer lock as invalidate changes tag storage
            typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->rw_lock_);

            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            context->op = ModifyOpType::MODIFY_INVALIDATE;
            auto curr_tags = this->h_tags_ + table_index * this->num_sets_ * NUM_WAYS;
            context->list->num_entries = 0;

            for (uint32_t i = 0; i < num_keys; i++)
            {
                auto input_key = keys[i];
                auto set = input_key % this->num_sets_;
                TagT* set_ways = curr_tags + set * NUM_WAYS;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    auto tag = set_ways[j];
                    IndexT stored_key = static_cast<IndexT>(tag * this->num_sets_ + set);
                    if (stored_key == input_key)
                    {
                        //hit
                        set_ways[j] = static_cast<TagT>(-1);
                        assert(context->list->num_entries <= context->max_update_sz);
                        context->list->entries[context->list->num_entries++] = {nullptr, nullptr, static_cast<uint32_t>(set), j, static_cast<TagT>(INVALID_IDX)};
                    }
                }
            }

            context->table_index = table_index;
            
            return modify(modify_context_handle, sync_event, stream);
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    CacheAllocationSize get_modify_context_size(uint32_t max_update_size) const override
    {
        CacheAllocationSize ret = {0, 0};
        size_t host_size = 0;
        host_size += sizeof(ModifyContext);
        host_size += max_update_size * sizeof(ModifyEntry);
        ret.host_allocation_size = host_size;
        ret.device_allocation_size = max_update_size * sizeof(ModifyEntry);
        return ret;
    }

    virtual ECError clear_cache(cudaStream_t stream) override
    {
        try 
        {
            std::fill(this->h_tags_, this->h_tags_ + this->num_sets_ * NUM_WAYS * this->config_.num_tables, static_cast<TagT>(INVALID_IDX));
            std::fill(this->h_counters_, this->h_counters_ + this->num_sets_ * NUM_WAYS * this->config_.num_tables, 0);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(this->d_tags_, this->h_tags_, this->num_sets_*sizeof(TagT) * NUM_WAYS* this->config_.num_tables, cudaMemcpyDefault, stream));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming indices is host accessiable
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
            // grab writer lock as insert changes tag storage and counters
            typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->rw_lock_);

            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            context->op = ModifyOpType::MODIFY_REPLACE;
            auto curr_tags = this->h_tags_ + table_index * this->num_sets_ * NUM_WAYS;
            auto curr_counters = this->h_counters_ + table_index * this->num_sets_ * NUM_WAYS;

            // decay all counters
            for (uint64_t i = 0; i < this->num_sets_; i++)
            {
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    curr_counters[i*NUM_WAYS + j] *= this->config_.decay_rate;
                }
            }

            std::unordered_map<uint64_t, ModifyEntry> insertMap;
            
            int8_t* curr_cache = this->cache_ + uint64_t(table_index) * this->num_sets_ * NUM_WAYS * this->config_.embed_width_in_bytes;
            context->list->num_entries = 0;
            for (size_t i = 0; (i < num_keys) && (insertMap.size() < context->max_update_sz); i++ )
            {
                IndexT index = keys[i];
                uint64_t set = index % this->num_sets_;
                TagT* set_ways = curr_tags + set * NUM_WAYS;
                bool found = false;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    auto tag = set_ways[j];
                    IndexT key = static_cast<IndexT>(tag * this->num_sets_ + set);
                    if (key == index)
                    {
                        //hit
                        found = true;
                        curr_counters[set*NUM_WAYS + j] += priority[i];
                        break;
                    }
                }

                if (!found)
                {
                    CounterT* set_counters = curr_counters + set*NUM_WAYS;
                    auto candidate = std::min_element( set_counters, set_counters + NUM_WAYS );
                    if (*candidate > priority[i])
                    {
                        continue;
                    }
                    auto candidate_index = std::distance(set_counters, candidate);
                    uint64_t hashkey = set * NUM_WAYS + static_cast<uint64_t>(candidate_index); // key to the unique map (candidate_index is guaranteed to be positive)
                    if (!insertMap.count(hashkey))
                    {
                        *candidate = priority[i];
                        set_ways[candidate_index] = static_cast<TagT>(index / this->num_sets_);
                        int8_t* dst = curr_cache + ( set * NUM_WAYS + candidate_index ) * this->config_.embed_width_in_bytes;
                        TagT tag = *(curr_tags + set * NUM_WAYS + candidate_index);
                        // the insert map make sure we will only put one item per (set,way) pair
                        insertMap[hashkey] = {data_ptrs[i], dst, static_cast<uint32_t>(set), static_cast<uint32_t>(candidate_index), tag};
                    }
                }
            }
            for (auto ii : insertMap)
            {
                assert(context->list->num_entries <= context->max_update_sz);
                context->list->entries[context->list->num_entries++] = ii.second;
            }
            context->table_index = table_index;

            return modify(modify_context_handle, sync_event, stream);
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming indices is host accessiable
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
            // grab reader lock as update does not change tag storage
            typename EmbedCacheSA<IndexT, TagT>::ReadLock lock(this->rw_lock_);

            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            context->op = ModifyOpType::MODIFY_UPDATE;
            auto curr_tags = this->h_tags_ + table_index * this->num_sets_ * NUM_WAYS;

            search_tag_storage_for_update(context, keys, d_values, stride, num_keys, table_index);

            context->table_index = table_index;
            
            return modify(modify_context_handle, sync_event, stream);
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming indices is host accessiable
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
            // grab reader lock as update does not change tag storage
            typename EmbedCacheSA<IndexT, TagT>::ReadLock lock(this->rw_lock_);

            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            context->op = ModifyOpType::MODIFY_UPDATE_ACCUMLATE;
            context->input_type = update_format;
            context->output_type = cache_format;
            auto curr_tags = this->h_tags_ + table_index * this->num_sets_ * NUM_WAYS;

            search_tag_storage_for_update(context, keys, d_values, stride, num_keys, table_index);

            context->table_index = table_index;
            
            return modify(modify_context_handle, sync_event, stream);
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
    // this function require synchronization with other worker threads needs to be atomic i.e no work that uses this cache can be called untill this function is returned and the event is waited
    // this code is non re-entrant
    ECError modify(const ModifyContextHandle& modify_context_handle, IECEvent* sync_event, cudaStream_t stream)
    {
        try
        {
            // this function should not be called without proper locking it assumed it was called first from the proper
            // modify operations
            ModifyContext* context = (ModifyContext*)modify_context_handle.handle;
            if (!context)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            if (context->list->num_entries == 0)
            {
                return ECERROR_SUCCESS;
            }
            uint64_t table_index = context->table_index;

            ModifyList list;
            list.num_entries = context->list->num_entries;
            list.entries = context->d_entries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(context->d_list, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(context->d_entries, context->list->entries, sizeof(ModifyEntry)*context->list->num_entries, cudaMemcpyHostToDevice, stream));
            
            auto dst_tags = this->d_tags_ + table_index * this->num_sets_ * NUM_WAYS;

            CACHE_CUDA_ERR_CHK_AND_THROW((call_tag_invalidate_kernel<IndexT, TagT>(context->d_list, context->list->num_entries, dst_tags, stream)));
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamSynchronize(stream));
            typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->custom_flow_mutex_);
            {
                CHECK_ERR_AND_THROW(sync_event->event_record());
            }
            CHECK_ERR_AND_THROW(sync_event->event_wait_stream(stream));
                
            if (context->op != ModifyOpType::MODIFY_INVALIDATE)
            {
                //invalidate doesn't require memory or tag update
                if (context->op == ModifyOpType::MODIFY_UPDATE_ACCUMLATE)
                {
                    if (context->input_type == nve::DATATYPE_INT8_SCALED) {
                        CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_accumulate_quantized_kernel<IndexT, TagT>(context->d_list, context->list->num_entries, static_cast<uint32_t>(this->config_.embed_width_in_bytes), context->input_type, context->output_type, stream)));
                    } else {
                        CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_accumulate_kernel<IndexT, TagT>(context->d_list, context->list->num_entries, static_cast<uint32_t>(this->config_.embed_width_in_bytes), context->input_type, context->output_type, stream)));
                    }
                }
                else
                {
                    CACHE_CUDA_ERR_CHK_AND_THROW((call_mem_update_kernel<IndexT, TagT>(context->d_list, context->list->num_entries, static_cast<uint32_t>(this->config_.embed_width_in_bytes), stream)));
                }
                CACHE_CUDA_ERR_CHK_AND_THROW((call_tag_update_kernel<IndexT, TagT>(context->d_list, context->list->num_entries, dst_tags, stream)));
            }
            
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // helper method to search the cache and build a list of tuples of src and dst, no asusmptions on uniqueness, this has been written to unify
    // code between update and update accumulate
    void search_tag_storage_for_update(ModifyContext* context, const IndexT* keys, const int8_t* d_values, int64_t stride, size_t num_keys, uint32_t table_index) const
    {
        int8_t* curr_cache = this->cache_ + table_index * this->num_sets_ * NUM_WAYS * this->config_.embed_width_in_bytes;
        auto curr_tags = this->h_tags_ + table_index * this->num_sets_ * NUM_WAYS;
        context->list->num_entries = 0;

        for (uint32_t i = 0; (i < num_keys) && (context->list->num_entries < context->max_update_sz); i++)
        {
            auto index = keys[i];
            IndexT set = static_cast<IndexT>(index % this->num_sets_);
            TagT* set_ways = curr_tags + set * NUM_WAYS;
            for (uint32_t j = 0; j < NUM_WAYS; j++)
            {
                auto tag = set_ways[j];
                IndexT key = static_cast<IndexT>(tag * this->num_sets_ + set);
                if (key == index)
                {
                    //hit
                    assert(context->list->num_entries <= context->max_update_sz);
                    int8_t* dst = curr_cache + ( set * NUM_WAYS + j ) * this->config_.embed_width_in_bytes;
                    TagT tag = *(curr_tags + set * NUM_WAYS + j);
                    context->list->entries[context->list->num_entries++] = {d_values + i * stride, dst, static_cast<uint32_t>(set), j, tag};
                    break;
                }
            }
        }
    }

private:

    virtual size_t get_extra_host_alloc_size(uint64_t num_tables, uint64_t num_sets) const override 
    {
        size_t ctr_size = num_tables * num_sets * NUM_WAYS * sizeof(CounterT);
        return ctr_size;
    }

    virtual void init_extras_host(uint64_t num_tables, int8_t* pool, size_t space) override 
    {
        // we should have set the number of sets before calling this function
        assert(this->num_sets_ > 0);
        h_counters_ = (CounterT*)EmbedCacheSA<IndexT, TagT>::allocate_in_pool(pool, space, sizeof(CounterT) * num_tables * this->num_sets_ * NUM_WAYS, 16);
        std::fill(h_counters_, h_counters_ + num_tables * this->num_sets_ * NUM_WAYS, 0);
    }

    CounterT* h_counters_; // per table counters
    typename EmbedCacheSA<IndexT, TagT>::ReadWriteLock rw_lock_;
};
}