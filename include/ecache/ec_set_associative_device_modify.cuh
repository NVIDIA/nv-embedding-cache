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
        ModifyList* pdList;
        ModifyEntry* pdEntries; // device allocated entries to modify
        int8_t* pdTmpStorage;
        size_t tmpStorageSz;
        uint32_t maxUpdateSz; // TODO: should be max num keys
        uint32_t tableIndex;
        DataTypeFormat inputType;
        DataTypeFormat outputType;
    };

public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY;

    CacheSADeviceModify(Allocator* pAllocator, Logger* pLogger, const CacheConfig& cfg) 
           : EmbedCacheSA<IndexT, TagT>(pAllocator, pLogger, cfg, TYPE)
    {}

    virtual ~CacheSADeviceModify() override = default;

    ECError ModifyContextCreate(ModifyContextHandle& outHandle, uint32_t maxUpdateSize) const override
    {
        ModifyContext* pContext = nullptr;
        try
        {
            // check erros
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext, sizeof(ModifyContext)));
            memset(pContext, 0, sizeof(ModifyContext));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&pContext->pdList, sizeof(ModifyList)));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&pContext->pdEntries, maxUpdateSize * sizeof(ModifyEntry)));

            pContext->maxUpdateSz = maxUpdateSize;

            // calling ComputeSetReplaceData to calculate temp storage size
            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetReplaceData<IndexT, TagT, CounterT, NUM_WAYS, false>(
                nullptr,
                nullptr,
                nullptr,
                pContext->tmpStorageSz,
                pContext->maxUpdateSz, // max number of keys
                nullptr,
                this->m_dpTags,
                m_pdCounters,
                this->m_pCache,
                this->m_config.embedWidth,
                this->m_config.decayRate,
                this->m_nSets,
                pContext->maxUpdateSz,
                pContext->pdList,
                0)));

            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&pContext->pdTmpStorage, pContext->tmpStorageSz));

            outHandle.handle = (uint64_t)pContext;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            // deallocate everything
            if (pContext != nullptr) {
                if (pContext->pdEntries != nullptr)
                    CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdEntries));
                if (pContext->pdList != nullptr)
                    CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdList));
                if (pContext->pdTmpStorage != nullptr)
                    CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdTmpStorage));
                CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext));
            }

            // can i do multiple catches
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError ModifyContextDestroy(ModifyContextHandle& outHandle) const override
    {
        try
        {
            CacheSADeviceModify::ModifyContext* pContext = (CacheSADeviceModify::ModifyContext*)outHandle.handle;
            if (!pContext)
            {
                return ECERROR_SUCCESS;
            }
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdEntries));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdList));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdTmpStorage));
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext));
            outHandle.handle = 0;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
           LOG_ERROR_AND_RETURN(e);
        }
        
    }

    ECError Invalidate(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        size_t numKeys,
        uint32_t tableIndex,
        IECEvent* syncEvent,
        cudaStream_t stream) override
    {
        try
        {
            std::lock_guard<std::mutex> lock(m_modifyMutex);
            
            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto pDstDeviceCurrTags = this->m_dpTags + tableIndex * this->m_nSets * NUM_WAYS;

            ModifyList list;
            list.nEntries = 0;
            list.pEntries = pContext->pdEntries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdList, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetInvalidateData<IndexT, TagT, NUM_WAYS>(
                keys, numKeys, this->m_nSets, pDstDeviceCurrTags, pContext->maxUpdateSz,
                pContext->pdList, stream)));

            InvalidateTagsAndSync(pContext, numKeys, pDstDeviceCurrTags, syncEvent, stream);

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    CacheAllocationSize GetModifyContextSize(uint32_t maxUpdateSize) const override
    {
        constexpr size_t szCounterPerSet = sizeof(CounterT) * NUM_WAYS;

        CacheAllocationSize ret = {0, 0};
        size_t hostSize = 0;
        size_t deviceSize = 0;
        hostSize += sizeof(ModifyContext);
        deviceSize += this->m_nSets * szCounterPerSet;
        deviceSize += maxUpdateSize * sizeof(ModifyEntry);
        deviceSize += sizeof(uint32_t);
        ret.hostAllocationSize = hostSize;
        ret.deviceAllocationSize = deviceSize;
        return ret;
    }

    virtual ECError ClearCache(cudaStream_t stream) override
    {
        try 
        {
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(this->m_dpTags, INVALID_IDX, this->m_nSets*sizeof(TagT) * NUM_WAYS* this->m_config.numTables, stream));
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming keys are device accessiable
    ECError Insert(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        const float* priority,
        const int8_t* const* ppData,
        size_t numKeys,
        uint32_t tableIndex,
        IECEvent* syncEvent,
        cudaStream_t stream) override
    {
        try
        {
            std::lock_guard<std::mutex> lock(m_modifyMutex);

            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto pDstDeviceCurrTags = this->m_dpTags + tableIndex * this->m_nSets * NUM_WAYS;
            int8_t* pCurrCache = this->m_pCache + tableIndex * this->m_nSets * NUM_WAYS * this->m_config.embedWidth;

            ModifyList list;
            list.nEntries = 0;
            list.pEntries = pContext->pdEntries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdList, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetReplaceData<IndexT, TagT, CounterT, NUM_WAYS, false>(
                ppData,
                keys,
                pContext->pdTmpStorage,
                pContext->tmpStorageSz,
                numKeys,
                priority,
                pDstDeviceCurrTags,
                m_pdCounters,
                pCurrCache,
                this->m_config.embedWidth,
                this->m_config.decayRate,
                this->m_nSets,
                pContext->maxUpdateSz,
                pContext->pdList,
                stream)));

            InvalidateTagsAndSync(pContext, numKeys, pDstDeviceCurrTags, syncEvent, stream);

            // copy data
            CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), static_cast<uint32_t>(this->m_config.embedWidth), stream)));
            // update tags
            CACHE_CUDA_ERR_CHK_AND_THROW((callTagUpdateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), pDstDeviceCurrTags, stream)));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming keys are device accessiable
    ECError Update(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t numKeys,
        uint32_t tableIndex,
        IECEvent* syncEvent,
        cudaStream_t stream) override
    {
        try
        {   
            std::lock_guard<std::mutex> lock(m_modifyMutex);

            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto pDstDeviceCurrTags = this->m_dpTags + tableIndex * this->m_nSets * NUM_WAYS;
            int8_t* pCurrCache = this->m_pCache + tableIndex * this->m_nSets * NUM_WAYS * this->m_config.embedWidth;

            ModifyList list;
            list.nEntries = 0;
            list.pEntries = pContext->pdEntries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdList, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetUpdateData<IndexT, TagT, NUM_WAYS>(
                pCurrCache, d_values, keys, numKeys, this->m_nSets, stride, this->m_config.embedWidth, pDstDeviceCurrTags,
                pContext->maxUpdateSz, pContext->pdList, stream)));

            InvalidateTagsAndSync(pContext, numKeys, pDstDeviceCurrTags, syncEvent, stream);

            // update data
            CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), static_cast<uint32_t>(this->m_config.embedWidth), stream)));
            // update tags
            CACHE_CUDA_ERR_CHK_AND_THROW((callTagUpdateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), pDstDeviceCurrTags, stream)));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming keys are device accessiable
    ECError UpdateAccumulate(
        ModifyContextHandle& modifyContextHandle,
        const IndexT* keys,
        const int8_t* d_values,
        int64_t stride,
        size_t numKeys,
        uint32_t tableIndex,
        DataTypeFormat updateFormat,
        DataTypeFormat cacheFormat,
        IECEvent* syncEvent,
        cudaStream_t stream) override
    {
        try
        {         
            std::lock_guard<std::mutex> lock(m_modifyMutex);
            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            auto pDstDeviceCurrTags = this->m_dpTags + tableIndex * this->m_nSets * NUM_WAYS;
            int8_t* pCurrCache = this->m_pCache + tableIndex * this->m_nSets * NUM_WAYS * this->m_config.embedWidth;

            ModifyList list;
            list.nEntries = 0;
            list.pEntries = pContext->pdEntries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdList, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));

            CACHE_CUDA_ERR_CHK_AND_THROW((ComputeSetUpdateData<IndexT, TagT, NUM_WAYS>(
                pCurrCache, d_values, keys, numKeys, this->m_nSets, stride, this->m_config.embedWidth, pDstDeviceCurrTags,
                pContext->maxUpdateSz, pContext->pdList, stream)));

            InvalidateTagsAndSync(pContext, numKeys, pDstDeviceCurrTags, syncEvent, stream);

            // update data
            if (pContext->inputType == nve::DATATYPE_INT8_SCALED) {
                CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateAccumulateQuantizedKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), static_cast<uint32_t>(this->m_config.embedWidth), updateFormat, cacheFormat, stream)));
            } else {
                CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateAccumulateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), static_cast<uint32_t>(this->m_config.embedWidth), updateFormat, cacheFormat, stream)));
            }

            // update tags
            CACHE_CUDA_ERR_CHK_AND_THROW((callTagUpdateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numKeys), pDstDeviceCurrTags, stream)));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    ECError GetKeysStoredInCache(const LookupContextHandle& /*lookupContextHandle*/, IndexT* outKeys, size_t& numOutKeys) const override
    {
        try 
        {
            if (!outKeys || !this->m_hpTags)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }

            CACHE_CUDA_ERR_CHK_AND_THROW(cudaDeviceSynchronize());
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpy(this->m_hpTags, this->m_dpTags, this->m_nSets * NUM_WAYS * sizeof(TagT), cudaMemcpyDeviceToHost));
 
            numOutKeys = 0;
            for (size_t i = 0; i < this->m_nSets; i++)
            {
                for (size_t j = 0; j < NUM_WAYS; j++)
                {
                    TagT tag = this->m_hpTags[i * NUM_WAYS + j];
                    if (tag == static_cast<TagT>(INVALID_IDX))
                    {
                        continue;
                    }
                    IndexT key = static_cast<IndexT>(tag * this->m_nSets + i);
                    outKeys[numOutKeys++] = key;
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

    ECError InvalidateTagsAndSync(const ModifyContext* pContext,
                                  size_t numTags,
                                  TagT* tags,
                                  IECEvent* syncEvent,
                                  cudaStream_t stream) {
        try 
        {
            
            CACHE_CUDA_ERR_CHK_AND_THROW((callTagInvalidateKernel<IndexT, TagT>(pContext->pdList, static_cast<uint32_t>(numTags), tags, stream)));

            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamSynchronize(stream));  
            {
                typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->m_customFlowMutex); // this will prevent invalidation while custom flow is running and avoid multiple invalidations
                CHECK_ERR_AND_THROW(syncEvent->EventRecord());
            }
            CHECK_ERR_AND_THROW(syncEvent->EventWaitStream(stream));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    virtual size_t GetExtraDeviceAllocSize(uint64_t numTables, uint64_t numSets) const override 
    {
        size_t ctr_size = numTables * numSets * NUM_WAYS * sizeof(CounterT);
        return ctr_size;
    }

    virtual void InitExtrasDevice(uint64_t numTables, int8_t* pool, size_t space) override 
    {
        // we should have set the number of sets before calling this function
        assert(this->m_nSets > 0);
        size_t sz = GetExtraDeviceAllocSize(numTables, this->m_nSets);
        m_pdCounters = (CounterT*)EmbedCacheSA<IndexT, TagT>::AllocateInPool(pool, space, sz, 16);
        CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemset(m_pdCounters, 0, sz));
    }

    virtual size_t GetCounterSizePerSet() const override {
        return CACHE_ALIGN(sizeof(CounterT) * NUM_WAYS, 16);
    }

    std::mutex m_modifyMutex;
    CounterT* m_pdCounters; // device allocated counters to modify
};
}
