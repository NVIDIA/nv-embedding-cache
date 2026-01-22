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
cudaError_t callCacheQuery(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint64_t* d_missing_index,
    IndexT* d_missing_keys, size_t* d_missing_len,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t currTable, size_t stride);

template<typename IndexT, typename TagT>
cudaError_t callCacheQueryHitMask(const IndexT* d_keys, const size_t len,
    int8_t* d_values, uint32_t* d_hit_mask,
    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
    cudaStream_t stream, uint32_t currTable, size_t stride);

template<typename IndexT, typename CacheDataT>
cudaError_t callCacheQueryUVM(const IndexT* d_keys, const size_t len,
    int8_t* d_values, const int8_t* d_table,
    CacheDataT data, cudaStream_t stream, uint32_t currTable, size_t stride);

template<typename IndexT, typename TagT>
cudaError_t callMemUpdateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEnries, uint32_t rowSizeInBytes, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t callTagUpdateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEnries, TagT* pTags, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t callTagInvalidateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEnries, TagT* pTags, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t callMemUpdateAccumulateKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t nEnries, uint32_t rowSizeInBytes, DataTypeFormat inputFormat, DataTypeFormat outputFormat, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t callMemUpdateAccumulateNoSyncKernel(const IndexT* d_keys, const size_t len, const int8_t* d_values, typename EmbedCacheSA<IndexT, TagT>::CacheData data, uint32_t currTable, size_t stride, DataTypeFormat inputFormat, DataTypeFormat outputFormat, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t callUpdateAccumalateNoSyncFusedWithPipeline( const int8_t* grads, 
                    int8_t* dst, 
                    const IndexT* keys, 
                    IndexT num_keys,
                    uint32_t row_size_in_bytes,
                    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
                    cudaStream_t stream);


template<typename IndexT, typename TagT>
cudaError_t callMemUpdateAccumulateQuantizedKernel(typename EmbedCacheSA<IndexT, TagT>::ModifyList* pList, uint32_t rowSizeInBytes, DataTypeFormat inputFormat, DataTypeFormat outputFormat, cudaStream_t stream);

template<typename IndexT, typename TagT>
cudaError_t callSortGather( const int8_t* uvm, 
                    int8_t* dst, 
                    const IndexT* keys, 
                    int8_t* auxBuf,
                    size_t num_keys,
                    size_t row_size_in_bytes,
                    typename EmbedCacheSA<IndexT, TagT>::CacheData data,
                    cudaStream_t stream);

#define INVALID_IDX -1 // using define to be able to cast to whatever needed

#define LOG_ERROR_AND_RETURN(ex) { if (this->m_pLogger) { this->m_pLogger->log(LogLevel_t::Error, ex.what());} return ex.m_err;}

template<typename IndexT, typename TagT>
class EmbedCacheSA : public EmbedCacheBase<IndexT>
{
public:
    using CounterT = float;

    struct CacheData
    {
        int8_t* pCache; 
        const int8_t* pTags;
        uint32_t nSets;
        uint64_t rowSizeInBytes;
        bool bCountMisses;
        int64_t* misses;
    };

    struct CacheConfig
    {
        // TRTREC-78 change cacheSzInBytes to two config ec device size and ec host size (when allocating on host, size in bytes is not well defined)
        size_t cacheSzInBytes = 0; // total size of storage, tags should not exceed this
        uint64_t embedWidth = 0;
        uint64_t numTables = 1;
        float decayRate = 0.95f; 
        bool allocDataOnHost = false;
    };

    struct ModifyEntry
    {
        const int8_t* pSrc;
        int8_t* pDst;
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
        ModifyEntry* pEntries;
        uint32_t nEntries;
    };

public:

    ECError CalcAllocationSize(CacheAllocationSize& outAllocationSz) const
    {
        uint64_t nSets = CalcNumSets();
        
        if (nSets == 0)
        {
            return ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL;
        }
        const size_t szTagsPerSet = GetTagSizePerSet();
        const size_t szEntryPerSet = GetEntrySizePerSet();

        size_t szHostPerSet = szTagsPerSet;
        size_t szDevicePerSet = szTagsPerSet;
        if (m_config.allocDataOnHost) { 
            szHostPerSet += szEntryPerSet;
        }
        else {
            szDevicePerSet += szEntryPerSet;
        }
        
        outAllocationSz.deviceAllocationSize = (nSets * szDevicePerSet) * m_config.numTables + GetExtraDeviceAllocSize(m_config.numTables, nSets);
        outAllocationSz.hostAllocationSize = (nSets * szHostPerSet ) * m_config.numTables + GetExtraHostAllocSize(m_config.numTables, nSets);

        // if our calculation are correct this shouldn't happen
        assert(outAllocationSz.deviceAllocationSize <= m_config.cacheSzInBytes);
        assert((nSets * szHostPerSet) * m_config.numTables <= m_config.cacheSzInBytes);
        
        return ECERROR_SUCCESS;
    }

    EmbedCacheSA(Allocator* pAllocator, Logger* pLogger, const CacheConfig& cfg, CACHE_IMPLEMENTATION_TYPE type)
        : EmbedCacheBase<IndexT>(pAllocator, pLogger, type), m_config(cfg), m_nSets(0), m_hpPool(nullptr), m_dpPool(nullptr),
          m_pCache(nullptr), m_dpTags(nullptr), m_hpTags(nullptr), m_customFlowLock(m_customFlowMutex, std::defer_lock)
    {

    }

    ECError LookupContextCreate(LookupContextHandle& outHandle, const PerformanceMetric* pMertics, size_t nMetrics) const override
    {
        try 
        {
            EmbedCacheSA::CacheData* pData;
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pData, sizeof(EmbedCacheSA::CacheData)));
            memset(pData, 0, sizeof(CacheData));
            pData->rowSizeInBytes = m_config.embedWidth;
            pData->pCache = m_pCache;
            pData->pTags = (const int8_t*)m_dpTags;
            pData->nSets = static_cast<decltype(pData->nSets)>(m_nSets);
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
            LOG_ERROR_AND_RETURN(e);
        }
    }


    ECError LookupContextDestroy(LookupContextHandle& handle) const override
    {
        EmbedCacheSA::CacheData* p = (EmbedCacheSA::CacheData*)handle.handle;
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
                CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&outMetric.p_dVal, sizeof(uint64_t)));
                outMetric.type = type;
                CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(outMetric.p_dVal, 0, sizeof(uint64_t)));
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
            LOG_ERROR_AND_RETURN(e);
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
            LOG_ERROR_AND_RETURN(e);
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
            LOG_ERROR_AND_RETURN(e);
        }
    }

    virtual ECError ModifyContextCreate(ModifyContextHandle& outHandle, uint32_t maxUpdateSize) const override = 0;

    virtual ECError ModifyContextDestroy(ModifyContextHandle& outHandle) const override = 0;

    virtual ECError Invalidate(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        size_t numKeys,
        uint32_t tableIndex,
        IECEvent* syncEvent,
        cudaStream_t stream) override = 0;

    virtual ~EmbedCacheSA() override
    {
        this->m_pAllocator->hostFree(m_hpPool);
        this->m_pAllocator->deviceFree(m_dpPool);
    }
    
    virtual ECError Init() override
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

            CacheAllocationSize allocSz = {0, 0}; 
            CalcAllocationSize(allocSz);

            m_nSets = CalcNumSets();

            if (m_nSets == 0)
            {
                EC_THROW(ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL);
            }
            
            // do a small check for KeyTag configuration and issue a warning 
            CheckTagKeyConfig();

            const size_t szTagsPerSet = GetTagSizePerSet(); // this already aligned to 16
            const size_t tagSize = szTagsPerSet * m_nSets * m_config.numTables;
            const size_t szEntryPerSet = GetEntrySizePerSet();
            const size_t dataSize = szEntryPerSet * m_nSets * m_config.numTables;

            // allocate memory pools
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&m_dpPool, allocSz.deviceAllocationSize));
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&m_hpPool, allocSz.hostAllocationSize));

            int8_t* pd = m_dpPool;
            const int8_t* ed = pd + allocSz.deviceAllocationSize;
            int8_t* ph = m_hpPool;
            const int8_t* eh = ph + allocSz.hostAllocationSize;

            // allocating pointers in memory pools
            // cactch bad alighments
            if(m_config.allocDataOnHost){
                // for gpu managed host cache config- allocate data on host. 
                m_pCache = AllocateInPool(ph, eh - ph, dataSize, 16);
            }
            else{
                // for normal gpu cache config- allocate as usual.
                m_pCache = AllocateInPool(pd, ed - pd, dataSize, 16);
            }
            m_dpTags = (TagT*)AllocateInPool(pd, ed - pd, tagSize, 16);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(m_dpTags, INVALID_IDX, tagSize));

            // always allocate host buffer for tags
            // in gpu_modify mode it is required for GetKeysStoredInCache method
            m_hpTags = (TagT*)AllocateInPool(ph, eh - ph, tagSize, 16);

            std::fill(m_hpTags, m_hpTags + m_nSets * NUM_WAYS * m_config.numTables, static_cast<TagT>(INVALID_IDX));
            
            InitExtrasHost(m_config.numTables, ph, eh - ph);
            InitExtrasDevice(m_config.numTables, pd, ed - pd);

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            if(m_hpPool){
                this->m_pAllocator->hostFree(m_hpPool);
            }
            if(m_dpPool){
                this->m_pAllocator->deviceFree(m_dpPool);
            }
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // return CacheData for Address Calcaulation each template should implement its own
    CacheData GetCacheData(const LookupContextHandle& handle) const
    {
        CacheData* cd = (CacheData*)handle.handle;
        // if cd is nullptr we have a problem, but i hate for this function to take a reference, it will create code ugly lines, please don't pass nullptr here
        return *cd;
    }

    ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_missing_index,
                                            IndexT* d_missing_keys, size_t* d_missing_len,
                                            uint32_t currTable, size_t stride, cudaStream_t stream) override
    {
        try{
            auto data = GetCacheData(hLookup);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(d_missing_len, 0, sizeof(*d_missing_len), stream));
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((callCacheQuery<IndexT, TagT>(d_keys, len, d_values, d_missing_index, d_missing_keys, d_missing_len, data, stream, currTable, stride)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError Lookup(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, uint64_t* d_hit_mask,
                                            uint32_t currTable, size_t stride, cudaStream_t stream) override
    {
        try 
        {
            auto data = GetCacheData(hLookup);
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((callCacheQueryHitMask<IndexT, TagT>(d_keys, len, d_values, reinterpret_cast<uint32_t*>(d_hit_mask), data, stream, currTable, stride)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
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
                CACHE_CUDA_ERR_CHK_AND_THROW((callCacheQueryUVM<IndexT>(d_keys, len, d_values, d_table, data, stream, currTable, stride)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError StartCustomFlow() override {
        m_customFlowLock.lock();
        return ECERROR_SUCCESS;
    }

    ECError EndCustomFlow() override {
        m_customFlowLock.unlock();
        return ECERROR_SUCCESS;
    }

    ECError LookupSortGather(const LookupContextHandle& hLookup, const IndexT* d_keys, const size_t len,
                                            int8_t* d_values, const int8_t* d_table, int8_t* d_auxiliryBuffer, size_t& auxiliryBufferBytes, uint32_t /*currTable*/, 
                                            size_t stride, cudaStream_t stream) override
    {
        try
        {
            auto data = GetCacheData(hLookup);
            // first calculate the aux buf required size
            size_t requiredBytes = 0;
            CACHE_CUDA_ERR_CHK_AND_THROW((callSortGather<IndexT, TagT>(d_table, 
                    d_values, 
                    d_keys, 
                    nullptr,
                    requiredBytes,
                    len,
                    stride,
                    data,
                    stream)));
            
            // if aux is null return its required size
            if (d_auxiliryBuffer == nullptr)
            {
                auxiliryBufferBytes = requiredBytes;
                return ECERROR_SUCCESS;
            }
            else
            {
                if (auxiliryBufferBytes < requiredBytes)
                {
                    EC_THROW(ECERROR_INVALID_ARGUMENT);
                }
                
                if (len > 0)
                {
                    CHECK_ERR_AND_THROW(StartCustomFlow());
                    CACHE_CUDA_ERR_CHK_AND_THROW((callSortGather<IndexT, TagT>( d_table, 
                    d_values, 
                    d_keys, 
                    d_auxiliryBuffer,
                    auxiliryBufferBytes,
                    len,
                    stride,
                    data,
                    stream)));
                    CHECK_ERR_AND_THROW(EndCustomFlow());
                }

                return ECERROR_SUCCESS;
            }
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError UpdateAccumulateNoSync(const LookupContextHandle& hLookup, ModifyContextHandle& /*hModify*/, const IndexT* d_keys, const size_t len, const int8_t* d_values, uint32_t currTable, size_t stride, DataTypeFormat inputFormat, DataTypeFormat outputFormat, cudaStream_t stream) override
    {
        try
        {
            auto data = GetCacheData(hLookup);
            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateAccumulateNoSyncKernel<IndexT, TagT>(d_keys, len, d_values, data, currTable, stride, inputFormat,outputFormat, stream)));
            }
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError UpdateAccumulateNoSyncFused(const LookupContextHandle& hLookup, ModifyContextHandle& /*hModify*/, const IndexT* d_keys, const size_t len, const int8_t* d_values, uint32_t /*currTable*/, size_t stride, 
                                            DataTypeFormat inputFormat, DataTypeFormat outputFormat, int8_t* uvm, cudaStream_t stream) override
    {
        try
        {
            auto data = GetCacheData(hLookup);
            if (inputFormat != outputFormat || inputFormat != DATATYPE_FP32 || stride != 512)
            {
                return ECERROR_NOT_IMPLEMENTED;
            }

            if (len > 0)
            {
                CACHE_CUDA_ERR_CHK_AND_THROW((callUpdateAccumalateNoSyncFusedWithPipeline<IndexT,TagT>(d_values, 
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


    size_t GetMaxNumEmbeddingVectorsInCache() const override
    {
        return m_nSets * NUM_WAYS;
    }

    virtual ECError GetKeysStoredInCache(const LookupContextHandle& /*lookupContextHandle*/, IndexT* outKeys, size_t& numOutKeys) const override = 0;
    
    CacheAllocationSize GetLookupContextSize() const override
    {
        CacheAllocationSize ret = {0, 0};
        ret.hostAllocationSize = sizeof(CacheData);
        return ret;
    }

    virtual CacheAllocationSize GetModifyContextSize(uint32_t maxUpdateSize) const override = 0;

    virtual ECError ClearCache(cudaStream_t stream) override = 0;

    // assuming keys are either host or device accessiable
    // depending on cache type
    virtual ECError Insert(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        const float* priority,
        const int8_t* const* ppData,
        size_t numKeys,
        uint32_t tableIndex,
        IECEvent* syncEvent,
        cudaStream_t stream) override = 0;

    // assuming keys are either host or device accessiable
    // depending on cache type
    virtual ECError Update(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t numKeys,
        uint32_t tableIndex,
        IECEvent* syncEvent,
        cudaStream_t stream) override = 0;

    // assuming keys are either host or device accessiable
    // depending on cache type
    virtual ECError UpdateAccumulate(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t numKeys,
        uint32_t tableIndex,
        DataTypeFormat updateFormat,
        DataTypeFormat cacheFormat,
        IECEvent* syncEvent,
        cudaStream_t stream) override = 0;

private:

    void CheckTagKeyConfig() const
    {
        constexpr IndexT maxKey = std::numeric_limits<IndexT>::max();

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

        if (std::numeric_limits<TagT>::max() < maxKey)
        {
            Log(LogLevel_t::Warning, "Maximum supported key is %llu", static_cast<uint64_t>(std::numeric_limits<TagT>::max())*m_nSets + (m_nSets-1));
        }
    }

    void Log(LogLevel_t verbosity, const char* format, ...) const
    {
        char buf[EC_MAX_STRING_BUF] = {0};
        va_list args;
        va_start(args, format);
        std::vsnprintf(buf, EC_MAX_STRING_BUF, format, args);
        va_end(args);
        assert(this->m_pLogger); // shouldn't be initilized if m_pLogger == nullptr
        this->m_pLogger->log(verbosity, buf);
    }

    // Helper method to calculate tag size per set
    size_t GetTagSizePerSet() const {
        return CACHE_ALIGN(sizeof(TagT) * NUM_WAYS, 16);
    }

    // Helper method to calculate entry size per set
    size_t GetEntrySizePerSet() const {
        return CACHE_ALIGN(m_config.embedWidth * NUM_WAYS, 16);
    }

    // Helper method to calculate counter size per set
    virtual size_t GetCounterSizePerSet() const {
        return 0;
    }

    uint64_t CalcNumSets() const {
        const size_t szPerSet = GetTagSizePerSet() + GetEntrySizePerSet() + GetCounterSizePerSet();
        uint64_t nSets = (m_config.cacheSzInBytes / m_config.numTables ) / (szPerSet);
        return nSets;
    }

protected:

    int8_t* AllocateInPool(int8_t*& p, size_t space, size_t sz, size_t align = 0) const
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

    virtual size_t GetExtraDeviceAllocSize(uint64_t /*numTables*/, uint64_t /*numSets*/) const { return 0; }
    virtual size_t GetExtraHostAllocSize(uint64_t /*numTables*/, uint64_t /*numSets*/) const { return 0; }
    virtual void InitExtrasHost(uint64_t /*numTables*/, int8_t* /*pool*/, size_t /*space*/) {}
    virtual void InitExtrasDevice(uint64_t /*numTables*/, int8_t* /*pool*/, size_t /*space*/) {}

    using ReadWriteLock =  std::shared_mutex;
    using WriteLock =  std::unique_lock<ReadWriteLock>; 
    using ReadLock =  std::shared_lock<ReadWriteLock>;  

    CacheConfig m_config;
    uint64_t m_nSets;
    int8_t* m_hpPool; // host memory pool for cache internal buffers- one free to rule them all
    int8_t* m_dpPool; // device memory pool for cache internal buffers- one free to rule them all
    int8_t* m_pCache; // cache data storage
    TagT* m_dpTags; // device tags
    TagT* m_hpTags; // host copy of tags
    ReadWriteLock m_customFlowMutex; // mutex for custom flow allowing for read write lock
    ReadLock m_customFlowLock; // dexplicit reader lock for defered locking via custom flow calls

public:
    static const uint32_t NUM_WAYS = 8;
};
}