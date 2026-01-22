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
        ModifyList* pList;
        ModifyList* pdList; // device allocated list of entries to modify
        ModifyEntry* pdEntries;
        uint32_t maxUpdateSz;
        uint32_t tableIndex;
        ModifyOpType op;
        DataTypeFormat inputType;
        DataTypeFormat outputType;
    };

public:
    static constexpr CACHE_IMPLEMENTATION_TYPE TYPE = CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA;

    CacheSAHostModify(Allocator* pAllocator, Logger* pLogger, const CacheConfig& cfg) 
           : EmbedCacheSA<IndexT, TagT>(pAllocator, pLogger, cfg, TYPE), m_phCounters(nullptr)
    {}

    virtual ~CacheSAHostModify() override = default;

    ECError ModifyContextCreate(ModifyContextHandle& outHandle, uint32_t maxUpdateSize) const override
    {
        ModifyContext* pContext = nullptr;
        try
        {
            // check erros
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext, sizeof(ModifyContext)));
            memset(pContext, 0, sizeof(ModifyContext));

            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pList, sizeof(ModifyList)));
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostAllocate((void**)&pContext->pList->pEntries, maxUpdateSize * sizeof(ModifyEntry)));

            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&pContext->pdList, sizeof(ModifyList)));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceAllocate((void**)&pContext->pdEntries, maxUpdateSize * sizeof(ModifyEntry)));

            pContext->maxUpdateSz = maxUpdateSize;
            pContext->pList->nEntries = 0;

            outHandle.handle = (uint64_t)pContext;
            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            //deallocate everything
            if (pContext != nullptr) {
                if (pContext->pList != nullptr) {
                    if (pContext->pList->pEntries != nullptr)
                        CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pList->pEntries));
                    CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pList));
                }
                if (pContext->pdEntries != nullptr)
                    CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdEntries));
                if (pContext->pdList != nullptr)
                    CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdList));
                
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
            CacheSAHostModify::ModifyContext* pContext = (CacheSAHostModify::ModifyContext*)outHandle.handle;
            if (!pContext)
            {
                return ECERROR_SUCCESS;
            }
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pList->pEntries));
            CHECK_ERR_AND_THROW(this->m_pAllocator->hostFree(pContext->pList));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdEntries));
            CHECK_ERR_AND_THROW(this->m_pAllocator->deviceFree(pContext->pdList));
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
            // grab writer lock as invalidate changes tag storage
            typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->m_rwLock);

            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            pContext->op = ModifyOpType::MODIFY_INVALIDATE;
            auto pCurrTags = this->m_hpTags + tableIndex * this->m_nSets * NUM_WAYS;
            pContext->pList->nEntries = 0;

            for (uint32_t i = 0; i < numKeys; i++)
            {
                auto inputKey = keys[i];
                auto set = inputKey % this->m_nSets;
                TagT* pSetWays = pCurrTags + set * NUM_WAYS;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    auto tag = pSetWays[j];
                    IndexT storedKey = static_cast<IndexT>(tag * this->m_nSets + set);
                    if (storedKey == inputKey)
                    {
                        //hit
                        pSetWays[j] = static_cast<TagT>(-1);
                        assert(pContext->pList->nEntries <= pContext->maxUpdateSz);
                        pContext->pList->pEntries[pContext->pList->nEntries++] = {nullptr, nullptr, static_cast<uint32_t>(set), j, static_cast<TagT>(INVALID_IDX)};
                    }
                }
            }

            pContext->tableIndex = tableIndex;
            
            return Modify(modifyContextHandle, syncEvent, stream);
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    CacheAllocationSize GetModifyContextSize(uint32_t maxUpdateSize) const override
    {
        CacheAllocationSize ret = {0, 0};
        size_t hostSize = 0;
        hostSize += sizeof(ModifyContext);
        hostSize += maxUpdateSize * sizeof(ModifyEntry);
        ret.hostAllocationSize = hostSize;
        ret.deviceAllocationSize = maxUpdateSize * sizeof(ModifyEntry);
        return ret;
    }

    virtual ECError ClearCache(cudaStream_t stream) override
    {
        try 
        {
            std::fill(this->m_hpTags, this->m_hpTags + this->m_nSets * NUM_WAYS * this->m_config.numTables, static_cast<TagT>(INVALID_IDX));
            std::fill(this->m_phCounters, this->m_phCounters + this->m_nSets * NUM_WAYS * this->m_config.numTables, 0);
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(this->m_dpTags, this->m_hpTags, this->m_nSets*sizeof(TagT) * NUM_WAYS* this->m_config.numTables, cudaMemcpyDefault, stream));

            return ECERROR_SUCCESS;
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming indices is host accessiable
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
            // grab writer lock as insert changes tag storage and counters
            typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->m_rwLock);

            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            pContext->op = ModifyOpType::MODIFY_REPLACE;
            auto pCurrTags = this->m_hpTags + tableIndex * this->m_nSets * NUM_WAYS;
            auto pCurrCounters = this->m_phCounters + tableIndex * this->m_nSets * NUM_WAYS;

            // decay all counters
            for (uint64_t i = 0; i < this->m_nSets; i++)
            {
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    pCurrCounters[i*NUM_WAYS + j] *= this->m_config.decayRate;
                }
            }

            std::unordered_map<uint64_t, ModifyEntry> insertMap;
            
            int8_t* pCurrCache = this->m_pCache + uint64_t(tableIndex) * this->m_nSets * NUM_WAYS * this->m_config.embedWidth;
            pContext->pList->nEntries = 0;
            for (size_t i = 0; (i < numKeys) && (insertMap.size() < pContext->maxUpdateSz); i++ )
            {
                IndexT index = keys[i];
                uint64_t set = index % this->m_nSets;
                TagT* pSetWays = pCurrTags + set * NUM_WAYS;
                bool bFound = false;
                for (uint32_t j = 0; j < NUM_WAYS; j++)
                {
                    auto tag = pSetWays[j];
                    IndexT key = static_cast<IndexT>(tag * this->m_nSets + set);
                    if (key == index)
                    {
                        //hit
                        bFound = true;
                        pCurrCounters[set*NUM_WAYS + j] += priority[i];
                        break;
                    }
                }

                if (!bFound)
                {
                    CounterT* pSetCounters = pCurrCounters + set*NUM_WAYS;
                    auto candidate = std::min_element( pSetCounters, pSetCounters + NUM_WAYS );
                    if (*candidate > priority[i])
                    {
                        continue;
                    }
                    auto candidateIndex = std::distance(pSetCounters, candidate);
                    uint64_t hashkey = set * NUM_WAYS + static_cast<uint64_t>(candidateIndex); // key to the unique map (candidateIndex is guaranteed to be positive)
                    if (!insertMap.count(hashkey))
                    {
                        *candidate = priority[i];
                        pSetWays[candidateIndex] = static_cast<TagT>(index / this->m_nSets);
                        int8_t* dst = pCurrCache + ( set * NUM_WAYS + candidateIndex ) * this->m_config.embedWidth;
                        TagT tag = *(pCurrTags + set * NUM_WAYS + candidateIndex);
                        // the insert map make sure we will only put one item per (set,way) pair
                        insertMap[hashkey] = {ppData[i], dst, static_cast<uint32_t>(set), static_cast<uint32_t>(candidateIndex), tag};
                    }
                }
            }
            for (auto ii : insertMap)
            {
                assert(pContext->pList->nEntries <= pContext->maxUpdateSz);
                pContext->pList->pEntries[pContext->pList->nEntries++] = ii.second;
            }
            pContext->tableIndex = tableIndex;

            return Modify(modifyContextHandle, syncEvent, stream);
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming indices is host accessiable
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
            // grab reader lock as update does not change tag storage
            typename EmbedCacheSA<IndexT, TagT>::ReadLock lock(this->m_rwLock);

            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            pContext->op = ModifyOpType::MODIFY_UPDATE;
            auto pCurrTags = this->m_hpTags + tableIndex * this->m_nSets * NUM_WAYS;

            SearchTagStorageForUpdate(pContext, keys, d_values, stride, numKeys, tableIndex);

            pContext->tableIndex = tableIndex;
            
            return Modify(modifyContextHandle, syncEvent, stream);
        }
        catch(const ECException& e)
        {
            LOG_ERROR_AND_RETURN(e);
        }
    }

    // assuming indices is host accessiable
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
            // grab reader lock as update does not change tag storage
            typename EmbedCacheSA<IndexT, TagT>::ReadLock lock(this->m_rwLock);

            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            pContext->op = ModifyOpType::MODIFY_UPDATE_ACCUMLATE;
            pContext->inputType = updateFormat;
            pContext->outputType = cacheFormat;
            auto pCurrTags = this->m_hpTags + tableIndex * this->m_nSets * NUM_WAYS;

            SearchTagStorageForUpdate(pContext, keys, d_values, stride, numKeys, tableIndex);

            pContext->tableIndex = tableIndex;
            
            return Modify(modifyContextHandle, syncEvent, stream);
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
    // this function require synchronization with other worker threads needs to be atomic i.e no work that uses this cache can be called untill this function is returned and the event is waited
    // this code is non re-entrant
    ECError Modify(const ModifyContextHandle& modifyContextHandle, IECEvent* syncEvent, cudaStream_t stream)
    {
        try
        {
            // this function should not be called without proper locking it assumed it was called first from the proper
            // modify operations
            ModifyContext* pContext = (ModifyContext*)modifyContextHandle.handle;
            if (!pContext)
            {
                EC_THROW(ECERROR_INVALID_ARGUMENT);
            }
            if (pContext->pList->nEntries == 0)
            {
                return ECERROR_SUCCESS;
            }
            uint64_t tableIndex = pContext->tableIndex;

            ModifyList list;
            list.nEntries = pContext->pList->nEntries;
            list.pEntries = pContext->pdEntries;
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdList, &list, sizeof(ModifyList), cudaMemcpyHostToDevice, stream));
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemcpyAsync(pContext->pdEntries, pContext->pList->pEntries, sizeof(ModifyEntry)*pContext->pList->nEntries, cudaMemcpyHostToDevice, stream));
            
            auto pDstDeviceCurrTags = this->m_dpTags + tableIndex * this->m_nSets * NUM_WAYS;

            CACHE_CUDA_ERR_CHK_AND_THROW((callTagInvalidateKernel<IndexT, TagT>(pContext->pdList, pContext->pList->nEntries, pDstDeviceCurrTags, stream)));
            CACHE_CUDA_ERR_CHK_AND_THROW(cudaStreamSynchronize(stream));
            typename EmbedCacheSA<IndexT, TagT>::WriteLock lock(this->m_customFlowMutex);
            {
                CHECK_ERR_AND_THROW(syncEvent->EventRecord());
            }
            CHECK_ERR_AND_THROW(syncEvent->EventWaitStream(stream));
                
            if (pContext->op != ModifyOpType::MODIFY_INVALIDATE)
            {
                //invalidate doesn't require memory or tag update
                if (pContext->op == ModifyOpType::MODIFY_UPDATE_ACCUMLATE)
                {
                    if (pContext->inputType == nve::DATATYPE_INT8_SCALED) {
                        CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateAccumulateQuantizedKernel<IndexT, TagT>(pContext->pdList, pContext->pList->nEntries, static_cast<uint32_t>(this->m_config.embedWidth), pContext->inputType, pContext->outputType, stream)));
                    } else {
                        CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateAccumulateKernel<IndexT, TagT>(pContext->pdList, pContext->pList->nEntries, static_cast<uint32_t>(this->m_config.embedWidth), pContext->inputType, pContext->outputType, stream)));
                    }
                }
                else
                {
                    CACHE_CUDA_ERR_CHK_AND_THROW((callMemUpdateKernel<IndexT, TagT>(pContext->pdList, pContext->pList->nEntries, static_cast<uint32_t>(this->m_config.embedWidth), stream)));
                }
                CACHE_CUDA_ERR_CHK_AND_THROW((callTagUpdateKernel<IndexT, TagT>(pContext->pdList, pContext->pList->nEntries, pDstDeviceCurrTags, stream)));
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
    void SearchTagStorageForUpdate(ModifyContext* pContext, const IndexT* keys, const int8_t* d_values, int64_t stride, size_t num_keys, uint32_t tableIndex) const
    {
        int8_t* pCurrCache = this->m_pCache + tableIndex * this->m_nSets * NUM_WAYS * this->m_config.embedWidth;
        auto pCurrTags = this->m_hpTags + tableIndex * this->m_nSets * NUM_WAYS;
        pContext->pList->nEntries = 0;

        for (uint32_t i = 0; (i < num_keys) && (pContext->pList->nEntries < pContext->maxUpdateSz); i++)
        {
            auto index = keys[i];
            IndexT set = static_cast<IndexT>(index % this->m_nSets);
            TagT* pSetWays = pCurrTags + set * NUM_WAYS;
            for (uint32_t j = 0; j < NUM_WAYS; j++)
            {
                auto tag = pSetWays[j];
                IndexT key = static_cast<IndexT>(tag * this->m_nSets + set);
                if (key == index)
                {
                    //hit
                    assert(pContext->pList->nEntries <= pContext->maxUpdateSz);
                    int8_t* dst = pCurrCache + ( set * NUM_WAYS + j ) * this->m_config.embedWidth;
                    TagT tag = *(pCurrTags + set * NUM_WAYS + j);
                    pContext->pList->pEntries[pContext->pList->nEntries++] = {d_values + i * stride, dst, static_cast<uint32_t>(set), j, tag};
                    break;
                }
            }
        }
    }

private:

    virtual size_t GetExtraHostAllocSize(uint64_t numTables, uint64_t numSets) const override 
    {
        size_t ctr_size = numTables * numSets * NUM_WAYS * sizeof(CounterT);
        return ctr_size;
    }

    virtual void InitExtrasHost(uint64_t numTables, int8_t* pool, size_t space) override 
    {
        // we should have set the number of sets before calling this function
        assert(this->m_nSets > 0);
        m_phCounters = (CounterT*)EmbedCacheSA<IndexT, TagT>::AllocateInPool(pool, space, sizeof(CounterT) * numTables * this->m_nSets * NUM_WAYS, 16);
        std::fill(m_phCounters, m_phCounters + numTables * this->m_nSets * NUM_WAYS, 0);
    }

    CounterT* m_phCounters; // per table counters
    typename EmbedCacheSA<IndexT, TagT>::ReadWriteLock m_rwLock;
};
}