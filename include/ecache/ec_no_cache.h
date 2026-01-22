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
cudaError_t callCacheQueryUVM(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t currTable, size_t stride);


template<typename IndexT>
class ECNoCache : public EmbedCacheBase<IndexT>
{
public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::API;
public:
    struct CacheConfig
    {
        uint32_t rowSizeInBytes;
    };

    struct CacheData
    {
        uint32_t rowSizeInBytes;
        bool bCountMisses;
        int64_t* misses;
    };

    struct ModifyContext
    {
    };

    ECNoCache(Allocator* pAllocator, Logger* pLogger, CacheConfig& cfg) : EmbedCacheBase<IndexT>(pAllocator, pLogger, API), m_config(cfg)
    {

    }

    ECError LookupContextCreate(LookupContextHandle& outHandle, const PerformanceMetric* pMertics, size_t nMetrics) const override
    {
        try 
        {
            CacheData* pData;
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pData, sizeof(CacheData)));
            memset(pData, 0, sizeof(CacheData));
            pData->rowSizeInBytes = m_config.rowSizeInBytes;
            for (size_t i = 0; i < nMetrics; i++)
            {
                if (pMertics->type == MERTIC_COUNT_MISSES)
                {
                    pData->bCountMisses = true;
                    pData->misses = pMertics[i].p_dVal;
                }
            }
            outHandle.handle = (uint64_t)pData;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            return e.m_err;
        }
    }


    ECError LookupContextDestroy(LookupContextHandle& handle) const override
    {
        CacheData* p = (CacheData*)handle.handle;
        handle.handle = 0;

        ECError ret = this->m_pAllocator->hostFree(p);
        return ret;
    }

    // performance 
    ECError PerformanceMetricCreate(PerformanceMetric& outMetric, PerformanceMerticTypes type) const override
    {
        try 
        {
            switch (type)
            {
            case MERTIC_COUNT_MISSES:
            {
                CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&outMetric.p_dVal, sizeof(uint32_t)));
                outMetric.type = type;
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(outMetric.p_dVal, 0, sizeof(uint32_t)));
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
    
    ECError PerformanceMetricDestroy(PerformanceMetric& metric) const override
    {
        try
        {
            switch (metric.type)
            {
            case MERTIC_COUNT_MISSES:
            {
                ECError ret = this->m_pAllocator->deviceFree(metric.p_dVal);
                metric.p_dVal = nullptr;
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

    ECError PerformanceMetricGetValue(const PerformanceMetric& metric, int64_t* pOutValue, cudaStream_t stream) const override
    {
        try
        {
            switch (metric.type)
            {
            case MERTIC_COUNT_MISSES:
            {
                if (!metric.p_dVal)
                {
                    EC_THROW(ECERROR_INVALID_ARGUMENT);
                }
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pOutValue, metric.p_dVal, sizeof(metric.p_dVal[0]), cudaMemcpyDefault, stream));
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

    ECError PerformanceMetricReset(PerformanceMetric& pMetric, cudaStream_t stream) const override
    {
        try
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(pMetric.p_dVal, 0, sizeof(pMetric.p_dVal[0]), stream));
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            return e.m_err;
        }
    }

    ECError ModifyContextCreate(ModifyContextHandle& outHandle, uint32_t /*maxUpdateSize*/) const override
    {
        outHandle.handle = 0;
        return ECERROR_SUCCESS;
    }

    ECError ModifyContextDestroy(ModifyContextHandle& /*outHandle*/) const override
    {
        return ECERROR_SUCCESS;
    }

    ECError Invalidate(
        ModifyContextHandle& /*modifyContextHandle*/,
        const IndexT* /*keys*/,
        size_t /*numKeys*/,
        uint32_t /*tableIndex*/,
        IECEvent* /*syncEvent*/,
        cudaStream_t /*stream*/) override
    {
        return ECERROR_SUCCESS;
    }

    ~ECNoCache()
    {
    }
    
    ECError Init() override
    {
        try
        {
            if (!this->m_pAllocator)
            {
                EC_THROW(ECERROR_BAD_ALLOCATOR);
            }

            if (!this->m_pLogger)
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
    CacheData GetCacheData(const LookupContextHandle& handle) const
    {
        CacheData* cd = (CacheData*)handle.handle;
        // if cd is nullptr we have a problem, but i hate for this function to take a reference, it will create code ugly lines, please don't pass nullptr here
        return *cd;
    }

    virtual ECError UpdateAccumulate(
        ModifyContextHandle& /*modifyContextHandle*/,
        const IndexT* /*keys*/,
        const int8_t* /*d_values*/,
        int64_t /*stride*/,
        size_t /*numKeys*/,
        uint32_t /*tableIndex*/,
        DataTypeFormat /*updateFormat*/,
        DataTypeFormat /*cacheFormat*/,
        IECEvent* /*syncEvent*/,
        cudaStream_t /*stream*/) override
    {
        return ECERROR_SUCCESS;
    }

    ECError Lookup(const LookupContextHandle& /*hLookup*/, const IndexT* d_keys, const size_t len,
                                            int8_t* /*d_values*/, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t /*currTable*/, size_t /*stride*/, cudaStream_t stream) override
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

    ECError Lookup(const LookupContextHandle& /*hLookup*/, const IndexT* /*d_keys*/, const size_t len,
                                            int8_t* /*d_values*/, uint64_t* d_hit_mask,
                                            uint32_t /*currTable*/, size_t /*stride*/, cudaStream_t /*stream*/) override
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

    ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, uint32_t currTable, 
                                            size_t stride, cudaStream_t stream) override
    {
        try 
        {
            auto data = GetCacheData(hLookup);
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW(callCacheQueryUVM<IndexT>(d_keys, len, d_values, d_table, data, stream, currTable, stride));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError LookupSortGather(const LookupContextHandle& /*hLookup*/, const IndexT* /*d_keys*/, const size_t /*len*/,
                                            int8_t* /*d_values*/, const int8_t* /*d_table*/, int8_t* /*d_auxiliryBuffer*/, size_t& /*auxiliryBufferBytes*/, 
                                            uint32_t /*currTable*/, size_t /*stride*/, cudaStream_t /*stream*/)
    {
        return ECERROR_NOT_IMPLEMENTED;
    }

    ECError UpdateAccumulateNoSync(const LookupContextHandle& /*hLookup*/, ModifyContextHandle& /*hModify*/, const IndexT* /*d_keys*/, const size_t /*len*/, const int8_t* /*d_values*/, uint32_t /*currTable*/, size_t /*stride*/, 
                                            DataTypeFormat /*inputFormat*/, DataTypeFormat /*outputFormat*/, cudaStream_t /*stream*/) override
                                            {
        return ECERROR_SUCCESS;
    }

    ECError UpdateAccumulateNoSyncFused(const LookupContextHandle&, ModifyContextHandle&, const IndexT*, const size_t, const int8_t*, uint32_t, size_t, 
                                            DataTypeFormat, DataTypeFormat, int8_t*, cudaStream_t)
    {
        return ECERROR_NOT_IMPLEMENTED;
    }

    size_t GetMaxNumEmbeddingVectorsInCache() const override
    {
        return 0;
    }
    
    CacheAllocationSize GetLookupContextSize() const override
    {
        CacheAllocationSize ret = {0};
        ret.hostAllocationSize = sizeof(CacheData);
        return ret;
    }

    CacheAllocationSize GetModifyContextSize(uint32_t /*maxUpdateSize*/) const override
    {
        CacheAllocationSize ret = {0};
        return ret;
    }

    ECError GetKeysStoredInCache(const LookupContextHandle& /*lookupContextHandle*/, IndexT* /*outKeys*/, size_t& numOutKeys) const override
    {
        numOutKeys = 0;
        return ECERROR_SUCCESS;
    }

    ECError ClearCache(cudaStream_t /*stream*/) override
    {
        return ECERROR_SUCCESS;
    }

    // assuming indices is host accessiable
    ECError Insert(
        ModifyContextHandle& /*modifyContextHandle*/,
        const IndexT* /*keys*/,
        const float* /*priority*/,
        const int8_t* const* /*ppData*/,
        size_t /*numKeys*/,
        uint32_t /*tableIndex*/,
        IECEvent* /*syncEvent*/,
        cudaStream_t /*stream*/)
    {
        return ECERROR_SUCCESS;
    }

    // assuming indices is host accessiable
    ECError Update(
        ModifyContextHandle& /*modifyContextHandle*/,
        const IndexT* /*keys*/,
        const int8_t* /*d_values*/,
        int64_t /*stride*/,
        size_t /*numKeys*/,
        uint32_t /*tableIndex*/,
        IECEvent* /*syncEvent*/,
        cudaStream_t /*stream*/)
    {
        return ECERROR_SUCCESS;
    }

    ECError StartCustomFlow() override
    {
        return ECERROR_SUCCESS;
    }

    ECError EndCustomFlow() override    
    {
        return ECERROR_SUCCESS;
    }
private:
    CacheConfig m_config;
};
}