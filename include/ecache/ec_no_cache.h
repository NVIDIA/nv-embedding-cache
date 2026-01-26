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
#include <numeric>
#include <vector>
#include <logging.hpp>

namespace nve {

template<typename IndexT, typename CacheDataT>
cudaError_t call_cache_query_uvm(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t curr_table, size_t stride);


template<typename IndexT>
class ECNoCache : public EmbedCacheBase<IndexT>
{
public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::API;
public:
    struct CacheConfig
    {
        uint32_t row_size_in_bytes;
    };

    struct CacheData
    {
        uint32_t row_size_in_bytes;
        bool count_misses;
        int64_t* misses;
    };

    struct ModifyContext
    {
    };

    ECNoCache(Allocator* allocator, Logger* pLogger, CacheConfig& cfg) : EmbedCacheBase<IndexT>(allocator, pLogger, API), config_(cfg)
    {

    }

    ECError lookup_context_create(LookupContextHandle& out_handle, const PerformanceMetric* metrics, size_t num_metrics) const override
    {
        try 
        {
            CacheData* p_data;
            CHECK_ERR_AND_THROW(this->allocator_->host_allocate((void**)&p_data, sizeof(CacheData)));
            memset(p_data, 0, sizeof(CacheData));
            p_data->row_size_in_bytes = config_.row_size_in_bytes;
            for (size_t i = 0; i < num_metrics; i++)
            {
                if (metrics->type == MERTIC_COUNT_MISSES)
                {
                    p_data->count_misses = true;
                    p_data->misses = metrics[i].d_val;
                }
            }
            out_handle.handle = (uint64_t)p_data;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            return e.m_err;
        }
    }


    ECError lookup_context_destroy(LookupContextHandle& handle) const override
    {
        CacheData* p = (CacheData*)handle.handle;
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
                CHECK_ERR_AND_THROW(this->allocator_->device_allocate((void**)&out_metric.d_val, sizeof(uint32_t)));
                out_metric.type = type;
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(out_metric.d_val, 0, sizeof(uint32_t)));
                return ECERROR_SUCCESS;
            }
            default:
                return ECERROR_NOT_IMPLEMENTED;
            }
        }
        catch(const ECException& e)
        {
            return e.m_err;
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
            return e.m_err;
        }
        
        
    }

    ECError performance_metric_get_value(const PerformanceMetric& metric, int64_t* p_out_value, cudaStream_t stream) const override
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
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(p_out_value, metric.d_val, sizeof(metric.d_val[0]), cudaMemcpyDefault, stream));
                return ECERROR_SUCCESS;
            }
            default:
                EC_THROW(ECERROR_NOT_IMPLEMENTED);
            }
        }
        catch(const ECException& e)
        {
            return e.m_err;
        }
        
    }

    ECError performance_metric_reset(PerformanceMetric& p_metric, cudaStream_t stream) const override
    {
        try
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(p_metric.d_val, 0, sizeof(p_metric.d_val[0]), stream));
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            return e.m_err;
        }
    }

    ECError modify_context_create(ModifyContextHandle& out_handle, uint32_t /*max_update_size*/) const override
    {
        out_handle.handle = 0;
        return ECERROR_SUCCESS;
    }

    ECError modify_context_destroy(ModifyContextHandle& /*out_handle*/) const override
    {
        return ECERROR_SUCCESS;
    }

    ECError invalidate(
        ModifyContextHandle& /*modify_context_handle*/,
        const IndexT* /*keys*/,
        size_t /*num_keys*/,
        uint32_t /*table_index*/,
        IECEvent* /*sync_event*/,
        cudaStream_t /*stream*/) override
    {
        return ECERROR_SUCCESS;
    }

    ~ECNoCache()
    {
    }
    
    ECError init() override
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
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            return e.m_err;
        }
    }

    // return CacheData for Address Calcaulation each template should implement its own
    CacheData get_cache_data(const LookupContextHandle& handle) const
    {
        CacheData* cd = (CacheData*)handle.handle;
        // if cd is nullptr we have a problem, but i hate for this function to take a reference, it will create code ugly lines, please don't pass nullptr here
        return *cd;
    }

    virtual ECError update_accumulate(
        ModifyContextHandle& /*modify_context_handle*/,
        const IndexT* /*keys*/,
        const int8_t* /*d_values*/,
        int64_t /*stride*/,
        size_t /*num_keys*/,
        uint32_t /*table_index*/,
        DataTypeFormat /*update_format*/,
        DataTypeFormat /*cache_format*/,
        IECEvent* /*sync_event*/,
        cudaStream_t /*stream*/) override
    {
        return ECERROR_SUCCESS;
    }

    ECError lookup(const LookupContextHandle& /*h_lookup*/, const IndexT* d_keys, const size_t len,
                                            int8_t* /*d_values*/, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t /*curr_table*/, size_t /*stride*/, cudaStream_t stream) override
    {
        try 
        {
            std::vector<uint64_t> indx(len);
            std::iota(indx.begin(), indx.end(), 0);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(d_missing_index, indx.data(), sizeof(uint64_t)*len, cudaMemcpyDefault, stream));
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(d_missing_keys, d_keys, sizeof(IndexT)*len, cudaMemcpyDefault, stream));
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(d_missing_len, &len, sizeof(size_t), cudaMemcpyDefault, stream));
            return ECERROR_SUCCESS;
        }
        catch (ECException& e)
        {
            return e.m_err;
        }
    }

    ECError lookup(const LookupContextHandle& /*h_lookup*/, const IndexT* /*d_keys*/, const size_t len,
                                            int8_t* /*d_values*/, uint64_t* d_hit_mask,
                                            uint32_t /*curr_table*/, size_t /*stride*/, cudaStream_t /*stream*/) override
    {
        try 
        {
            cudaMemsetAsync(d_hit_mask, 0, (len+7)/8);
            return ECERROR_SUCCESS;
        }
        catch (ECException& e)
        {
            return e.m_err;
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
                CACHE_CUDA_ERR_CHK_AND_THROW(call_cache_query_uvm<IndexT>(d_keys, len, d_values, d_table, data, stream, curr_table, stride));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError lookup_sort_gather(const LookupContextHandle& /*h_lookup*/, const IndexT* /*d_keys*/, const size_t /*len*/,
                                            int8_t* /*d_values*/, const int8_t* /*d_table*/, int8_t* /*d_auxiliary_buffer*/, size_t& /*auxiliary_buffer_bytes*/, 
                                            uint32_t /*curr_table*/, size_t /*stride*/, cudaStream_t /*stream*/)
    {
        return ECERROR_NOT_IMPLEMENTED;
    }

    ECError update_accumulate_no_sync(const LookupContextHandle& /*h_lookup*/, ModifyContextHandle& /*hModify*/, const IndexT* /*d_keys*/, const size_t /*len*/, const int8_t* /*d_values*/, uint32_t /*curr_table*/, size_t /*stride*/, 
                                            DataTypeFormat /*input_format*/, DataTypeFormat /*output_format*/, cudaStream_t /*stream*/) override
                                            {
        return ECERROR_SUCCESS;
    }

    ECError update_accumulate_no_sync_fused(const LookupContextHandle&, ModifyContextHandle&, const IndexT*, const size_t, const int8_t*, uint32_t, size_t, 
                                            DataTypeFormat, DataTypeFormat, int8_t*, cudaStream_t)
    {
        return ECERROR_NOT_IMPLEMENTED;
    }

    size_t get_max_num_embedding_vectors_in_cache() const override
    {
        return 0;
    }
    
    CacheAllocationSize get_lookup_context_size() const override
    {
        CacheAllocationSize ret = {0};
        ret.host_allocation_size = sizeof(CacheData);
        return ret;
    }

    CacheAllocationSize get_modify_context_size(uint32_t /*max_update_size*/) const override
    {
        CacheAllocationSize ret = {0};
        return ret;
    }

    ECError get_keys_stored_in_cache(const LookupContextHandle& /*lookup_context_handle*/, IndexT* /*out_keys*/, size_t& num_out_keys) const override
    {
        num_out_keys = 0;
        return ECERROR_SUCCESS;
    }

    ECError clear_cache(cudaStream_t /*stream*/) override
    {
        return ECERROR_SUCCESS;
    }

    // assuming indices is host accessiable
    ECError insert(
        ModifyContextHandle& /*modify_context_handle*/,
        const IndexT* /*keys*/,
        const float* /*priority*/,
        const int8_t* const* /*pp_data*/,
        size_t /*num_keys*/,
        uint32_t /*table_index*/,
        IECEvent* /*sync_event*/,
        cudaStream_t /*stream*/)
    {
        return ECERROR_SUCCESS;
    }

    // assuming indices is host accessiable
    ECError update(
        ModifyContextHandle& /*modify_context_handle*/,
        const IndexT* /*keys*/,
        const int8_t* /*d_values*/,
        int64_t /*stride*/,
        size_t /*num_keys*/,
        uint32_t /*table_index*/,
        IECEvent* /*sync_event*/,
        cudaStream_t /*stream*/)
    {
        return ECERROR_SUCCESS;
    }

    ECError start_custom_flow() override
    {
        return ECERROR_SUCCESS;
    }

    ECError end_custom_flow() override    
    {
        return ECERROR_SUCCESS;
    }
private:
    CacheConfig config_;
};
}