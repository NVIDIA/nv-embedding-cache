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

#include "gtest/gtest.h"
#include "embedding_cache_combined.h"
#include "embedding_cache_combined.cuh"
#include "datagen.h"
#include <algorithm>
#include "../common/check_error.h"
#include <default_allocator.hpp>
using namespace nve;

enum IndexFormat
{
    INDEX_FORMAT_UINT32_T,
};

struct TableConfig
{
    size_t numLines;
    size_t rowSizeInBytes;
    size_t rowStride;
    bool bLinearTable;
    uint32_t numTables;
};

struct SampleConfig
{
    size_t sampleSize;
    uint32_t numSamples;
    float alpha;
};

struct CacheConfig
{
    CACHE_IMPLEMENTATION_TYPE impl;
    uint32_t numSets;
    bool allocDataOnHost;
    float decayRate;
};

struct EnvConfig
{
    uint32_t numStreams;
};

// helper struct, can help debug by finiding a specifc test case by index. or some tests are agnostics to some of the config and can skip tests
// based on index. e.g a test testing the Key invalidate api doesn't depend on the sample, nor the table config and can skip all of those
struct TestIndex
{
    uint32_t tcIdx = 0;
    uint32_t scIdx = 0;
    uint32_t ecIdx = 0;
};

struct TestCase
{
    TestCase(const CacheConfig& _c, const TableConfig& _t, const SampleConfig& _s,  const EnvConfig& _e, const TestIndex& _i = {0}) : cache(_c), table(_t), sample(_s), env(_e), idx(_i) {}
    TestCase(std::tuple<CacheConfig,TableConfig,SampleConfig,EnvConfig,TestIndex> in) : cache(std::get<0>(in)), table(std::get<1>(in)), sample(std::get<2>(in)), env(std::get<3>(in)), idx(std::get<4>(in)) {}
    CacheConfig cache;
    TableConfig table;
    SampleConfig sample;
    EnvConfig env;
    TestIndex idx;
};

template<typename IndexT, typename TableDataType>
class ApiTest : public testing::TestWithParam<TestCase>
{
public:
    using IndexType = IndexT;
    using DataType = TableDataType;
    using checkResultFunc = std::function<void(uint32_t t)>;
    ApiTest() : m_allocator(DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {}
protected:
    void SetUp() override
    {
        TestCase t = GetParam();

        const size_t numIndices = t.sample.sampleSize * t.sample.numSamples;
        ASSERT_TRUE(t.table.rowSizeInBytes % sizeof(TableDataType) == 0);
        ASSERT_TRUE(t.table.rowSizeInBytes <= t.table.rowStride);
        const size_t tableSizeInBytes = t.table.numLines * t.table.rowSizeInBytes;
        const auto numTables = t.table.numTables;
        switch (t.cache.impl)
        {
        case CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA:
        {
            typename CacheSAHostModify<IndexT, IndexT>::CacheConfig config;
            config.cacheSzInBytes = t.cache.numSets * t.table.rowSizeInBytes * 8;
            config.embedWidth = t.table.rowSizeInBytes;
            config.numTables = t.table.numTables;
            config.allocDataOnHost = t.cache.allocDataOnHost;
            config.decayRate = t.cache.decayRate;
            m_pCache = std::make_shared<CacheSAHostModify<IndexT, IndexT>>(&m_allocator, &m_logger, config);
            break;
        }
        case CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY:
        {
            typename CacheSADeviceModify<IndexT, IndexT>::CacheConfig config;
            config.cacheSzInBytes = t.cache.numSets * t.table.rowSizeInBytes * 8;
            config.embedWidth = t.table.rowSizeInBytes;
            config.numTables = t.table.numTables;
            config.allocDataOnHost = t.cache.allocDataOnHost;
            config.decayRate = t.cache.decayRate;
            m_pCache = std::make_shared<CacheSADeviceModify<IndexT, IndexT>>(&m_allocator, &m_logger, config);
            break;
        }
        default:
            FAIL() << "Unkown Cache implemenation\n";
        }

        CHECK_EC(m_pCache->Init());
        
        m_dMissingIndex.resize(numTables);
        m_dMissingKeys.resize(numTables);
        m_dKeys.resize(numTables);
        m_dMissingLen.resize(numTables);
        m_dValues.resize(numTables);
        m_hMissingIndex.resize(numTables);
        m_hMissingKeys.resize(numTables);
        m_hKeys.resize(numTables);
        m_hMissingLen.resize(numTables);
        m_hValues.resize(numTables);
        m_pTable.resize(numTables);
        
        for (uint32_t i = 0; i < numTables; i++)
        {
            CHECK_CUDA_ERROR(cudaMallocHost(&m_pTable.at(i), tableSizeInBytes));

            CHECK_CUDA_ERROR(cudaMalloc(&m_dMissingIndex.at(i), numIndices*sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dMissingKeys.at(i), numIndices*sizeof(IndexT)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dKeys.at(i), numIndices*sizeof(IndexT)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dMissingLen.at(i), sizeof(size_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&m_dValues.at(i), numIndices*t.table.rowSizeInBytes));

            CHECK_CUDA_ERROR(cudaMallocHost(&m_hMissingIndex.at(i), numIndices*sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMallocHost(&m_hMissingKeys.at(i), numIndices*sizeof(IndexT)));
            CHECK_CUDA_ERROR(cudaMallocHost(&m_hKeys.at(i), numIndices*sizeof(IndexT)));
            CHECK_CUDA_ERROR(cudaMallocHost(&m_hMissingLen.at(i), sizeof(size_t)));
            CHECK_CUDA_ERROR(cudaMallocHost(&m_hValues.at(i), numIndices*t.table.rowSizeInBytes));
        }

        for (uint32_t i = 0; i < t.env.numStreams; i++)
        {
            cudaStream_t stream;
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            m_streams.push_back(stream);

            ModifyContextHandle hModify;
            LookupContextHandle hLookup;
            std::array<PerformanceMetric, NUM_PERFORMANCE_METRIC_TYPES> perfMetric;
            CHECK_EC(m_pCache->ModifyContextCreate(hModify, static_cast<uint32_t>(numIndices)));
            // needs to be expended when more metrics will be introduced
            CHECK_EC(m_pCache->PerformanceMetricCreate(perfMetric[MERTIC_COUNT_MISSES], MERTIC_COUNT_MISSES));
            CHECK_EC(m_pCache->LookupContextCreate(hLookup, perfMetric.data(), 1));
            m_hLookup.push_back(hLookup);
            m_hModify.push_back(hModify);
            m_perfMetric.push_back(perfMetric);
        }

        FillTable(t);
    }

    void FillTable(TestCase c)
    {
        TableConfig t = c.table;
        for (uint32_t i = 0; i < t.numTables; i++)
        {
            int8_t* m_pRaw = (int8_t*)m_pTable[i];
            for (uint32_t j = 0; j < t.numLines; j++)
            {
                size_t numRowElements = t.rowSizeInBytes / sizeof(TableDataType);
                for (size_t k = 0; k < numRowElements; k++)
                {
                    TableDataType* pElement = (TableDataType*)(m_pRaw + j * t.rowStride + k * sizeof(TableDataType));
                    *pElement = static_cast<TableDataType>(k == 0 ? i : j);
                }
            }
        }
    }

    void GenerateIndices(IndexT* h_indexSet, size_t numSamples, std::shared_ptr<FeatureGenerator<IndexT>>& sg)
    {
        for (size_t j = 0; j < numSamples; j++)
        {
            //m_mutex->lock();
            auto sample = sg->getCategoryIndices();
            //m_mutex->unlock();
            for (size_t i = 0; i < sample.size(); i++)
            {
                h_indexSet[j * sample.size() + i] =  sample[i];
            }
        }
    }

    void TearDown() override
    {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        TestCase t = GetParam();

        for (uint32_t i = 0; i < t.env.numStreams; i++)
        {
            CHECK_EC(m_pCache->PerformanceMetricDestroy(m_perfMetric.at(i).at(MERTIC_COUNT_MISSES)));
            CHECK_EC(m_pCache->LookupContextDestroy(m_hLookup.at(i)));
            CHECK_EC(m_pCache->ModifyContextDestroy(m_hModify.at(i)));
            CHECK_CUDA_ERROR(cudaStreamDestroy(m_streams.at(i)));
        }   

        for (uint32_t i = 0; i < t.table.numTables; i++)
        {
            CHECK_CUDA_ERROR(cudaFreeHost(m_pTable.at(i)));
            m_pTable.at(i) = nullptr;
            
            // Free device memory
            CHECK_CUDA_ERROR(cudaFree(m_dMissingIndex.at(i)));
            CHECK_CUDA_ERROR(cudaFree(m_dMissingKeys.at(i)));
            CHECK_CUDA_ERROR(cudaFree(m_dKeys.at(i)));
            CHECK_CUDA_ERROR(cudaFree(m_dMissingLen.at(i)));
            CHECK_CUDA_ERROR(cudaFree(m_dValues.at(i)));
            
            // Free host memory
            CHECK_CUDA_ERROR(cudaFreeHost(m_hMissingIndex.at(i)));
            CHECK_CUDA_ERROR(cudaFreeHost(m_hMissingKeys.at(i)));
            CHECK_CUDA_ERROR(cudaFreeHost(m_hKeys.at(i)));
            CHECK_CUDA_ERROR(cudaFreeHost(m_hMissingLen.at(i)));
            CHECK_CUDA_ERROR(cudaFreeHost(m_hValues.at(i)));
        }                   
    }

    void Lookup(const checkResultFunc& func, std::shared_ptr<FeatureGenerator<IndexT>>& sg, bool bInsertIndices, bool bUseUVM)
    {
        TestCase t = this->GetParam();
        for (uint32_t i = 0; i < t.table.numTables; i++)
        {
            this->GenerateIndices(this->m_hKeys[i], t.sample.numSamples, sg);
        }
        
        
        Lookup(func, m_hKeys, bInsertIndices, bUseUVM);
    }

    void Lookup(const checkResultFunc& func, std::vector<IndexT*>& hIdx, bool bInsertIndices, bool bUseUVM)
    {
        TestCase t = this->GetParam();
        const auto numIndices = t.sample.sampleSize * t.sample.numSamples;
        for (uint32_t i = 0; i < t.table.numTables; i++)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[i], hIdx[i], numIndices*sizeof(IndexT), cudaMemcpyDefault));
        }
        if (bInsertIndices)
        {
            DefaultECEvent syncEvent(m_streams);
            for (uint32_t i = 0; i < t.table.numTables; i++)
            {
                DefaultHistogram hist(hIdx[i], numIndices, (const int8_t*)m_pTable[i], t.table.rowStride, t.table.bLinearTable);
                
                CHECK_EC(m_pCache->Insert(m_hModify[0], hist.GetKeys(), hist.GetPriority(), hist.GetData(), hist.GetNumBins(), i, &syncEvent, m_streams[0]));
                CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
            }
        }

        for (uint32_t i = 0; i < t.table.numTables;)
        {
            // distribute tables across streams
            for (uint32_t s = 0; s < t.env.numStreams && i < t.table.numTables; s++, i++)
            {
                if (bUseUVM)
                {
                    this->m_pCache->Lookup(this->m_hLookup[s], this->m_dKeys[i], numIndices, (int8_t*)this->m_dValues[i], (const int8_t*)this->m_pTable[i], i, t.table.rowSizeInBytes, this->m_streams[s]);
                }
                else
                {
                    this->m_pCache->Lookup(this->m_hLookup[s], this->m_dKeys[i], numIndices, (int8_t*)this->m_dValues[i], this->m_dMissingIndex[i], this->m_dMissingKeys[i], this->m_dMissingLen[i], i, t.table.rowSizeInBytes, this->m_streams[s]);
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hMissingLen[i], this->m_dMissingLen[i], sizeof(size_t), cudaMemcpyDefault, this->m_streams[s]));
                    
                    // disptach rest of copies this intentaionally doesn't optimize copy size to check all stream parrallisem doesn't cause issues
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hMissingIndex[i], this->m_dMissingIndex[i], numIndices*sizeof(size_t), cudaMemcpyDefault, this->m_streams[s]));
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hMissingKeys[i], this->m_dMissingKeys[i], numIndices*sizeof(IndexT), cudaMemcpyDefault, this->m_streams[s]));
                }
                CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hValues[i], this->m_dValues[i], numIndices*t.table.rowSizeInBytes, cudaMemcpyDefault, this->m_streams[s]));
            }
        }

        // check results and synchronize 
        for (uint32_t s = 0; s < t.env.numStreams; s++)
        {
            CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[s]));
        }
        
        EXPECT_TRUE(t.table.numTables <= t.env.numStreams); // TRTREC-101: iterating over numTables below, might cause illegal access in performanceMetric array in func.
        for (uint32_t i = 0; i < t.table.numTables; i++)
        {
            func(i);
        }
    }

    void CheckKeysInCache(const std::set<IndexT> &keys, size_t missingTolerable)
    {
        std::vector<IndexT> outKeys(m_pCache->GetMaxNumEmbeddingVectorsInCache());
        size_t nKeys = 0;
        m_pCache->GetKeysStoredInCache(m_hLookup[0], outKeys.data(), nKeys);
        std::set<IndexT> cacheKeys(outKeys.begin(), outKeys.begin() + nKeys);
        size_t missingCount = 0;
        for (auto key : keys)
        {
            if (cacheKeys.find(key) == cacheKeys.end())
            {
                missingCount++;
            }
        }
        EXPECT_TRUE(missingCount <= missingTolerable);
    }

    void AssertCacheContainsExactly(const std::set<IndexT> &keys)
    {
        std::vector<IndexT> outKeys(m_pCache->GetMaxNumEmbeddingVectorsInCache());
        size_t nKeys = 0;
        m_pCache->GetKeysStoredInCache(m_hLookup[0], outKeys.data(), nKeys);
        std::set<IndexT> cacheKeys(outKeys.begin(), outKeys.begin() + nKeys);
        EXPECT_EQ(keys, cacheKeys);
        EXPECT_EQ(nKeys, keys.size());
    }

    void EmptyLookupInternal()
    {
        TestCase t = GetParam();
        auto sg = getSampleGenerator<IndexT>(t.sample.alpha, static_cast<uint32_t>(t.table.numLines), static_cast<uint32_t>(t.sample.sampleSize), 3458760);
        auto f = [&](uint32_t i)
        {
            auto numIndices = t.sample.sampleSize * t.sample.numSamples;
            EXPECT_EQ(*m_hMissingLen[i], numIndices);
            EXPECT_TRUE(std::equal(m_hKeys[i], m_hKeys[i] + numIndices, m_hMissingKeys[i], m_hMissingKeys[i] + numIndices));
        };
        Lookup(f, sg, false, false);
    }

    void SimpleLookupInternal()
    {
        TestCase t = GetParam();
        auto sg = getSampleGenerator<IndexT>(t.sample.alpha, static_cast<uint32_t>(t.table.numLines), static_cast<uint32_t>(t.sample.sampleSize), 3458760);
        auto f = [&](uint32_t i)
        {
            auto numIndices = t.sample.sampleSize * t.sample.numSamples;
            for (size_t j = 0; j < numIndices; j++)
            {
                size_t missingLen = *m_hMissingLen[i];
                if (std::find(m_hMissingIndex[i], m_hMissingIndex[i] + missingLen, j) != m_hMissingIndex[i] + missingLen)
                {
                    continue;
                }
                int8_t* pSrc = (int8_t*)m_hValues[i] + j * t.table.rowSizeInBytes;
                int8_t* pDst = (int8_t*)m_pTable[i] + m_hKeys[i][j] * t.table.rowStride;
                EXPECT_TRUE(std::equal(pSrc, pSrc + t.table.rowSizeInBytes, pDst));
            }
        };
        Lookup(f, sg, true, false);
    }

    void SimpleLookupUVMInternal()
    {
        TestCase t = GetParam();
        auto sg = getSampleGenerator<IndexT>(t.sample.alpha, static_cast<uint32_t>(t.table.numLines), static_cast<uint32_t>(t.sample.sampleSize), 3458760);
        auto f = [&](uint32_t i)
        {
            auto numIndices = t.sample.sampleSize * t.sample.numSamples;
            for (size_t j = 0; j < numIndices; j++)
            {
                int8_t* pSrc = (int8_t*)m_hValues[i] + j * t.table.rowSizeInBytes;
                int8_t* pDst = (int8_t*)m_pTable[i] + m_hKeys[i][j] * t.table.rowStride;
                EXPECT_TRUE(std::equal(pSrc, pSrc + t.table.rowSizeInBytes, pDst));
            }
        };
        Lookup(f, sg, true, true);
    }

    void CheckKeyRetrival()
    {
        constexpr size_t nIndices = 10;
        TestCase t = GetParam();
        if (t.idx.tcIdx != 0 || t.idx.scIdx != 0 || t.idx.ecIdx != 0)
        {
            return;
        }

        //insert indices
        DefaultECEvent syncEvent(m_streams);
        std::vector<IndexT> hIdx;
        for (size_t i = 0; i < nIndices; i++)
        {
            hIdx.push_back(static_cast<IndexT>(i));
        }
        DefaultHistogram hist(hIdx.data(), nIndices, (const int8_t*)m_pTable[0], t.table.rowStride, true);
        CHECK_EC(m_pCache->Insert(m_hModify[0], hist.GetKeys(), hist.GetPriority(), hist.GetData(), hist.GetNumBins(), 0, &syncEvent, m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        AssertCacheContainsExactly(std::set<IndexT>(hIdx.begin(), hIdx.end()));

        CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[0], hIdx.data(), hIdx.size()*sizeof(IndexT), cudaMemcpyDefault));
        this->m_pCache->Lookup(this->m_hLookup[0], this->m_dKeys[0], hIdx.size(), (int8_t*)this->m_dValues[0], this->m_dMissingIndex[0], this->m_dMissingKeys[0], this->m_dMissingLen[0], 0, t.table.rowSizeInBytes, this->m_streams[0]);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hMissingLen[0], this->m_dMissingLen[0], sizeof(size_t), cudaMemcpyDefault, this->m_streams[0]));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        EXPECT_EQ(*this->m_hMissingLen[0], 0);
    }

    void SimpleInvalidate()
    {
        // First put indices make sure those are in the cache
        constexpr size_t nIndices = 10;
        TestCase t = GetParam();
        if (t.idx.tcIdx != 0 || t.idx.scIdx != 0 || t.idx.ecIdx != 0)
        {
            return;
        }

        //insert indices
        DefaultECEvent syncEvent(m_streams);
        std::vector<IndexT> hIdx;
        for (size_t i = 0; i < nIndices; i++)
        {
            hIdx.push_back(static_cast<IndexT>(i));
        }
        std::set<IndexT> ref(hIdx.begin(), hIdx.end());
        DefaultHistogram hist(hIdx.data(), nIndices, (const int8_t*)m_pTable[0], t.table.rowStride, true);
        CHECK_EC(m_pCache->Insert(m_hModify[0], hist.GetKeys(), hist.GetPriority(), hist.GetData(), hist.GetNumBins(), 0, &syncEvent, m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        {
            AssertCacheContainsExactly(ref);
        }

        std::vector<IndexT> invIdx;
        invIdx.push_back(3);
        invIdx.push_back(7);
        if (t.cache.impl == CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY) {
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[0], invIdx.data(), invIdx.size()*sizeof(IndexT), cudaMemcpyDefault));
            CHECK_EC(m_pCache->Invalidate(m_hModify[0], this->m_dKeys[0], invIdx.size(), 0, &syncEvent, m_streams[0]));
        } else {
            CHECK_EC(m_pCache->Invalidate(m_hModify[0], invIdx.data(), invIdx.size(), 0, &syncEvent, m_streams[0]));
        }
        for (auto i : invIdx)
        {
            ref.erase(i);
        }

        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        {
            AssertCacheContainsExactly(ref);
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[0], invIdx.data(), invIdx.size()*sizeof(IndexT), cudaMemcpyDefault));
            this->m_pCache->Lookup(this->m_hLookup[0], this->m_dKeys[0], invIdx.size(), (int8_t*)this->m_dValues[0], this->m_dMissingIndex[0], this->m_dMissingKeys[0], this->m_dMissingLen[0], 0, t.table.rowSizeInBytes, this->m_streams[0]);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hMissingLen[0], this->m_dMissingLen[0], sizeof(size_t), cudaMemcpyDefault, this->m_streams[0]));
            
            CHECK_CUDA_ERROR(cudaMemcpyAsync(this->m_hMissingKeys[0], this->m_dMissingKeys[0], invIdx.size()*sizeof(IndexT), cudaMemcpyDefault, this->m_streams[0]));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            EXPECT_EQ(*this->m_hMissingLen[0], invIdx.size());
            std::vector<IndexT> missKeys(this->m_hMissingKeys[0], this->m_hMissingKeys[0] + *this->m_hMissingLen[0]);
            EXPECT_EQ(missKeys, invIdx);
        }
    }

    void SimpleClear()
    {
        // first put indices make sure those are in the cache
        constexpr size_t nIndices = 10;
        TestCase t = GetParam();
        if (t.idx.tcIdx != 0 || t.idx.scIdx != 0 || t.idx.ecIdx != 0)
        {
            return;
        }

        //insert indices
        DefaultECEvent syncEvent(m_streams);
        std::vector<IndexT> hIdx;
        for (size_t i = 0; i < nIndices; i++)
        {
            hIdx.push_back(static_cast<IndexT>(i));
        }
        DefaultHistogram hist(hIdx.data(), nIndices, (const int8_t*)m_pTable[0], t.table.rowStride, true);
        CHECK_EC(m_pCache->Insert(m_hModify[0], hist.GetKeys(), hist.GetPriority(), hist.GetData(), hist.GetNumBins(), 0, &syncEvent, m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        AssertCacheContainsExactly(std::set<IndexT>(hIdx.begin(), hIdx.end()));

        // clear cache
        CHECK_EC(m_pCache->ClearCache(m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));

        AssertCacheContainsExactly(std::set<IndexT>());
    }

    void CheckMetric()
    {
        TestCase t = GetParam();
        auto sg = getSampleGenerator<IndexT>(t.sample.alpha, static_cast<uint32_t>(t.table.numLines), static_cast<uint32_t>(t.sample.sampleSize), 3458760);
        auto f = [&](uint32_t i)
        {
            auto numIndices = t.sample.sampleSize * t.sample.numSamples;
            EXPECT_EQ(*m_hMissingLen[i], numIndices);
            int64_t missCount = 0;
            m_pCache->PerformanceMetricGetValue(m_perfMetric[i][MERTIC_COUNT_MISSES], &missCount, 0);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            EXPECT_EQ(missCount, numIndices);
        };
        Lookup(f, sg, false, false);
    }

    void SimpleUpdateAccumulate()
    {
        // first put indices make sure those are in the cache
        constexpr size_t nIndices = 10;
        TestCase t = GetParam();
        if (t.idx.tcIdx != 0 || t.idx.scIdx != 0 || t.idx.ecIdx != 0)
        {
            return;
        }

        //insert indices
        DefaultECEvent syncEvent(m_streams);
        std::vector<IndexT> hIdx;
        for (size_t i = 0; i < nIndices; i++)
        {
            hIdx.push_back(static_cast<IndexT>(i));
        }
        
        DefaultHistogram hist(hIdx.data(), nIndices, (const int8_t*)m_pTable[0], t.table.rowStride, true);
        CHECK_EC(m_pCache->Insert(m_hModify[0], hist.GetKeys(), hist.GetPriority(), hist.GetData(), hist.GetNumBins(), 0, &syncEvent, m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        
        // allocate and fill update data
        size_t rowSizeInElements = t.table.rowSizeInBytes/sizeof(DataType);
        float* gradUpdate = nullptr;
        CHECK_CUDA_ERROR(cudaMallocHost(&gradUpdate, t.table.rowSizeInBytes*nIndices));
        memset(gradUpdate, 0, t.table.rowSizeInBytes*nIndices);
        for (uint32_t i = 0; i < nIndices; i++)
        {
            gradUpdate[i*rowSizeInElements] = 1;
        }

        CHECK_CUDA_ERROR(cudaMemcpy(this->m_dValues[0], gradUpdate, hIdx.size()*t.table.rowSizeInBytes, cudaMemcpyDefault));
        if (t.cache.impl == CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY) {
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[0], hIdx.data(), hIdx.size()*sizeof(IndexT), cudaMemcpyDefault));
            CHECK_EC(m_pCache->UpdateAccumulate(m_hModify[0], this->m_dKeys[0], (const int8_t*)this->m_dValues[0], t.table.rowSizeInBytes, hIdx.size(), 0, nve::DATATYPE_FP32, nve::DATATYPE_FP32, &syncEvent, m_streams[0]));
        } else {
            CHECK_EC(m_pCache->UpdateAccumulate(m_hModify[0], hIdx.data(), (const int8_t*)this->m_dValues[0], t.table.rowSizeInBytes, hIdx.size(), 0, nve::DATATYPE_FP32, nve::DATATYPE_FP32, &syncEvent, m_streams[0]));
        }
        float* ref = nullptr;
        CHECK_CUDA_ERROR(cudaMallocHost(&ref, t.table.rowSizeInBytes*nIndices));

        for (uint32_t i = 0; i < nIndices; i++)
        {
            for (uint32_t j = 0; j < rowSizeInElements; j++)
            {
                ref[i * rowSizeInElements + j ] = gradUpdate[i*rowSizeInElements + j] + m_pTable[0][hIdx[i]*rowSizeInElements + j];
            }
        }

        float* test = nullptr;
        CHECK_CUDA_ERROR(cudaMallocHost(&test, t.table.rowSizeInBytes*nIndices));
        CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[0], hIdx.data(), hIdx.size()*sizeof(IndexT), cudaMemcpyDefault));
        this->m_pCache->Lookup(this->m_hLookup[0], this->m_dKeys[0], hIdx.size(), (int8_t*)test, this->m_dMissingIndex[0], this->m_dMissingKeys[0], this->m_dMissingLen[0], 0, t.table.rowSizeInBytes, this->m_streams[0]);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_streams[0]));
        for (uint32_t i = 0; i < nIndices*rowSizeInElements; i++) {
            EXPECT_EQ(ref[i], test[i]);
        }
    }

    void InsertInternal()
    {
        TestCase t = GetParam();
        // We'll define high priorities, low priorities, and original priorities.
        constexpr float HIGH_PRIORITY = 500000.0f; // 500 k
        constexpr float ORIGINAL_PRIORITY = 10000.0f; // 10k
        constexpr float LOW_PRIORITY = 0.00001f;
        // First populate the cache and make sure the keys are in the cache.
        size_t nSampleIndices = t.sample.numSamples * t.sample.sampleSize;
        size_t nMaxIndicesInCache = m_pCache->GetMaxNumEmbeddingVectorsInCache();
        size_t nChunksToInsert = (nMaxIndicesInCache + nSampleIndices - 1) / nSampleIndices;
        DefaultECEvent syncEvent(m_streams);
        std::vector<IndexT> hOriginalIdx;
        std::vector<float> originalPriorities(nMaxIndicesInCache, ORIGINAL_PRIORITY);
        std::vector<const int8_t*> pOriginalData(nMaxIndicesInCache, nullptr);
        for (size_t i = 0; i < nMaxIndicesInCache; i++)
        {
            hOriginalIdx.push_back(static_cast<IndexT>(i));
            pOriginalData[i] = (const int8_t*)m_pTable[0] + i * t.table.rowSizeInBytes;
        }
        // Insert of large number of keys, should be done in batches due to maxModifySize
        for(size_t i = 0; i < (nChunksToInsert != 0 ? nChunksToInsert : 1); i++)
        {
            size_t startIndex = i * nSampleIndices;
            size_t endIndex = std::min(startIndex + nSampleIndices, nMaxIndicesInCache);
            size_t chunkSize = endIndex - startIndex;
            CHECK_EC(m_pCache->Insert(m_hModify[0], hOriginalIdx.data() + startIndex, originalPriorities.data() + startIndex, pOriginalData.data() + startIndex, chunkSize, 0, &syncEvent, m_streams[0]));
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        std::set<IndexT> refOriginal(hOriginalIdx.begin(), hOriginalIdx.end());
        AssertCacheContainsExactly(refOriginal);

        // Now try to insert different set of keys with low priority, we'll expect no eviction and no replacing.
        size_t nOtherIndices = std::min({size_t(EmbedCacheSA<IndexT, IndexT>::NUM_WAYS), t.table.numLines - nMaxIndicesInCache, nSampleIndices}); //at least one set in the cache.
        std::vector<IndexT> hOtherIdx;
        std::vector<float> otherLowPriorities(nOtherIndices, LOW_PRIORITY);
        std::vector<const int8_t*> pOtherData(nOtherIndices, nullptr);
        for (size_t i = 0; i < nOtherIndices; i++)
        {
            size_t idx = i + nMaxIndicesInCache;
            hOtherIdx.push_back(static_cast<IndexT>(idx));
            pOtherData[i] = (const int8_t*)m_pTable[0] + idx * t.table.rowSizeInBytes;
        }
        CHECK_EC(m_pCache->Insert(m_hModify[0], hOtherIdx.data(), otherLowPriorities.data(), pOtherData.data(), nOtherIndices, 0, &syncEvent, m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        AssertCacheContainsExactly(refOriginal);

        // Try to insert again with high priority, we'll expect the new keys to be fully inserted, which will result in some eviction and replacing.
        std::vector<float> otherHighPriorities(nOtherIndices, HIGH_PRIORITY);
        CHECK_EC(m_pCache->Insert(m_hModify[0], hOtherIdx.data(), otherHighPriorities.data(), pOtherData.data(), nOtherIndices, 0, &syncEvent, m_streams[0]));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
        CheckKeysInCache(std::set<IndexT>(hOtherIdx.begin(), hOtherIdx.end()), 0);
        CheckKeysInCache(refOriginal, nOtherIndices);
    }

public:
    DefaultAllocator m_allocator; // Must precede m_pCache for destruction order
    std::shared_ptr<EmbedCacheBase<IndexT>> m_pCache;
    Logger m_logger;
    // per table
    std::vector<TableDataType*> m_pTable;
    std::vector<IndexT*> m_dKeys; 
    std::vector<IndexT*> m_hKeys; 
    std::vector<uint64_t*> m_dMissingIndex;
    std::vector<uint64_t*> m_hMissingIndex;
    std::vector<IndexT*> m_dMissingKeys;
    std::vector<IndexT*> m_hMissingKeys;
    std::vector<size_t*> m_dMissingLen;
    std::vector<size_t*> m_hMissingLen;
    std::vector<TableDataType*> m_dValues;
    std::vector<TableDataType*> m_hValues;

    // per stream
    std::vector<cudaStream_t> m_streams;

    std::vector<std::array<PerformanceMetric, NUM_PERFORMANCE_METRIC_TYPES>> m_perfMetric;
    std::vector<LookupContextHandle> m_hLookup;
    std::vector<ModifyContextHandle> m_hModify;
};

TEST(NegativeTests, InitZeroMemory)
{
    DefaultAllocator allocator(DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD);
    Logger logger;

    typename CacheSAHostModify<uint32_t, uint32_t>::CacheConfig config;

    config.cacheSzInBytes = 0;
    config.embedWidth = 128;
    config.numTables = 1;
    CacheSAHostModify<uint32_t, uint32_t> ec(&allocator, &logger, config);
    EXPECT_EQ(ec.Init() , ECERROR_MEMORY_ALLOCATED_TO_CACHE_TOO_SMALL);
}

using Test_UINT32_T_FLOAT = ApiTest<uint32_t, float>;
using Test_INT32_T_FLOAT = ApiTest<int32_t, float>;
using Test_INT64_T_FLOAT = ApiTest<int64_t, float>;
using Test_UINT64_T_FLOAT = ApiTest<uint64_t, float>;
using Test_UINT32_T_INT8 = ApiTest<uint32_t, int8_t>;

#define TEST_FORMAT(test_name, test_func) \
TEST_P(Test_UINT32_T_FLOAT, test_name)    \
{                                         \
    test_func();                          \
}                                         \
TEST_P(Test_INT32_T_FLOAT, test_name)     \
{                                         \
    test_func();                          \
}                                         \
TEST_P(Test_INT64_T_FLOAT, test_name)     \
{                                         \
    test_func();                          \
}                                         \
TEST_P(Test_UINT64_T_FLOAT, test_name)    \
{                                         \
    test_func();                          \
}

TEST_P(Test_UINT32_T_FLOAT, Initialization)
{
    // initialization should work here
    EXPECT_TRUE(true);
}

TEST_FORMAT(EmptyLookup, EmptyLookupInternal);
TEST_FORMAT(SimpleLookup, SimpleLookupInternal);
TEST_FORMAT(SimpleLookupUVM, SimpleLookupUVMInternal);
TEST_FORMAT(Insert, InsertInternal);

TEST_P(Test_UINT32_T_INT8, Initialization)
{
    // initialization should work here
    EXPECT_TRUE(true);
}

TEST_P(Test_UINT32_T_INT8, SimpleLookup)
{
    SimpleLookupInternal();
}

TEST_P(Test_UINT32_T_FLOAT, SimpleReplace)
{
    TestCase t = GetParam();
    auto sg = getSampleGenerator<uint32_t>(t.sample.alpha, static_cast<uint32_t>(t.table.numLines), static_cast<uint32_t>(t.sample.sampleSize), 3458760);
    auto f = [&](uint32_t i)
    {
        size_t missingLen = *m_hMissingLen[i];
        EXPECT_EQ(missingLen, 0);
    };
    Lookup(f, sg, true, false);
}

TEST_P(Test_UINT32_T_FLOAT, SimpleUpdate)
{
    TestCase t = GetParam();
    auto numIndices = t.sample.sampleSize * t.sample.numSamples;
    auto numElements = t.table.rowSizeInBytes / sizeof(DataType);
    DefaultECEvent syncEvent(m_streams);
    std::vector<DataType*> updateData(t.table.numTables);
    
    auto f = [&](uint32_t i)
    {
        auto numIndices = t.sample.sampleSize * t.sample.numSamples;
        size_t missingLen = *m_hMissingLen[i];
        for (size_t j = 0; j < numIndices; j++)
        {
            if (std::find(m_hMissingIndex[i], m_hMissingIndex[i] + missingLen, j) != m_hMissingIndex[i] + missingLen)
            {
                continue;
            }
            int8_t* pSrc = (int8_t*)m_hValues[i] + j * t.table.rowSizeInBytes;
            int8_t* pDst = (int8_t*)updateData[i] + m_hKeys[i][j] * t.table.rowSizeInBytes;
            EXPECT_TRUE(std::equal(pSrc, pSrc + t.table.rowSizeInBytes, pDst));
        }
    };

    for (uint32_t i = 0; i < t.table.numTables; i++)
    {
        IndexType* hIdx = m_hKeys[i];
        DataType* p = nullptr;
        CHECK_CUDA_ERROR(cudaMallocHost(&p, t.table.rowSizeInBytes * numIndices));
        updateData[i] = p;
        for (uint32_t j = 0; j < numIndices; j++)
        {
            hIdx[j] = j;
            for (uint32_t k = 0; k < numElements; k++)
            {
                *(p + numElements * j + k) = static_cast<DataType>(k * j + 5 * i);
            }
        }
        DefaultHistogram hist(m_hKeys[i], numIndices, (const int8_t*)m_pTable[i], t.table.rowStride, t.table.bLinearTable);
        CHECK_EC(m_pCache->Insert(m_hModify[0], hist.GetKeys(), hist.GetPriority(), hist.GetData(), hist.GetNumBins(), i, &syncEvent, m_streams[0]));

        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));

        CHECK_CUDA_ERROR(cudaMemcpy(this->m_dValues[i], updateData[i], numIndices*t.table.rowSizeInBytes, cudaMemcpyDefault));
        if (t.cache.impl == CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY) {
            CHECK_CUDA_ERROR(cudaMemcpy(this->m_dKeys[i], m_hKeys[i], numIndices*sizeof(IndexType), cudaMemcpyDefault));
            CHECK_EC(m_pCache->Update(m_hModify[0], m_dKeys[i], (const int8_t*)this->m_dValues[i], t.table.rowSizeInBytes, numIndices, i, &syncEvent, m_streams[0]));
        } else {
            CHECK_EC(m_pCache->Update(m_hModify[0], m_hKeys[i], (const int8_t*)this->m_dValues[i], t.table.rowSizeInBytes, numIndices, i, &syncEvent, m_streams[0]));
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_streams[0]));
    }

    Lookup(f, m_hKeys, false, false);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    for (auto p : updateData)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(p));
    }
}

TEST_P(Test_UINT32_T_FLOAT, CheckKeyRetrival)
{
    CheckKeyRetrival();
}

TEST_P(Test_UINT32_T_FLOAT, SimpleInvalidate)
{
    SimpleInvalidate();
}

TEST_P(Test_UINT32_T_FLOAT, SimpleClear)
{
    SimpleClear();
}

TEST_P(Test_UINT32_T_FLOAT, SimpleUpdateAccumulate)
{
    SimpleUpdateAccumulate();
}

TEST_P(Test_UINT32_T_FLOAT, CheckMetric)
{
    CheckMetric();
}

TEST_P(Test_UINT32_T_FLOAT, PerformanceTest)
{
    TestCase t = GetParam();
    auto sg = getSampleGenerator<uint32_t>(t.sample.alpha, static_cast<uint32_t>(t.table.numLines), static_cast<uint32_t>(t.sample.sampleSize), 3458760);
    auto f = [&](uint32_t i)
    {
        size_t missingLen = *m_hMissingLen[i];
        int64_t val = 0;
        CHECK_EC(m_pCache->PerformanceMetricGetValue(m_perfMetric[i][MERTIC_COUNT_MISSES], &val, 0));
        EXPECT_EQ(val, missingLen);
    };
    Lookup(f, sg, true, false);
}

static TestCase cases_int8[] =
{
    // CacheC                                                                       //TableC                //SampleC      //EnvC
    {{CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA, 32, false, 0.95f},  {1000, 7, 7, true, 1}, {32, 1, -1.0}, {1}, {0,0,0}},
    {{CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA, 32, true, 0.95f},  {1000, 7, 7, true, 1}, {32, 1, -1.0}, {1}, {0,0,0}},
    {{CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY, 32, false, 0.95f},  {1000, 7, 7, true, 1}, {32, 1, -1.0}, {1}, {0,0,0}},
    {{CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY, 32, true, 0.95f},  {1000, 7, 7, true, 1}, {32, 1, -1.0}, {1}, {0,0,0}},
    {{CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY, 31, true, 0.95f},  {1000, 1, 1, true, 1}, {32, 1, -1.0}, {1}, {0,0,0}},

};

static std::vector<CacheConfig> cc = 
{
    {CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA, 32, false, 0.55f},
    {CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA, 32, true, 0.75f},
    {CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY, 32, false, 0.85f},
    {CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY, 32, true, 0.95f},
    {CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_DEVICE_ONLY, 33, false, 0.95f},
    {CACHE_IMPLEMENTATION_TYPE::SET_ASSOCIATIVE_HOST_METADATA, 33, false, 0.95f},
};

static std::vector<TableConfig> tc_float =
{
    {1000, 512, 512, true, 1},
    {1000, 4, 4, true, 1},
};

static std::vector<SampleConfig> sc =
{
    {32, 1, -1.0}
};

static std::vector<EnvConfig> ec =
{
    {1}
};

#define INSTANTIATE_TEST_SUITE_P_FORMAT(name, values) \
INSTANTIATE_TEST_SUITE_P(name,                        \
                        Test_UINT32_T_FLOAT,          \
                        values);                      \
INSTANTIATE_TEST_SUITE_P(name,                        \
                        Test_INT32_T_FLOAT,           \
                        values);                      \
INSTANTIATE_TEST_SUITE_P(name,                        \
                        Test_INT64_T_FLOAT,           \
                        values);                      \
INSTANTIATE_TEST_SUITE_P(name,                        \
                        Test_UINT64_T_FLOAT,          \
                        values);                      
static std::vector<TestCase> genCase(const std::vector<CacheConfig>& _cc, const std::vector<TableConfig>& _tc, const std::vector<SampleConfig>& _sc, const std::vector<EnvConfig>& _ec) {
    std::vector<TestCase> ret;
    for (uint32_t c = 0; c < _cc.size(); c++)
    {
        for (uint32_t t = 0; t < _tc.size(); t++)
        {
            for (uint32_t s = 0; s < _sc.size(); s++)
            {
                for (uint32_t e = 0; e < _ec.size(); e++)
                {
                    TestIndex idx = {t, s, e};
                    ret.push_back(TestCase(_cc[c], _tc[t], _sc[s], _ec[e], idx));
                }
            }
        }
    }
    return ret;
}

INSTANTIATE_TEST_SUITE_P_FORMAT(OneTableFloat, testing::ValuesIn(genCase(cc, tc_float, sc, ec)));

INSTANTIATE_TEST_SUITE_P(OneTableInt8,
                        Test_UINT32_T_INT8,
                         testing::ValuesIn(
                            cases_int8
                         ));

TEST(ErrorMacros, BasicThrow) {
    int64_t caught_exceptions = 0;
    try {
        EC_THROW(ECERROR_INVALID_ARGUMENT);
    } catch (nve::ECException& e) {
        caught_exceptions++;
        NVE_DEBUG_PRINT_("Test Caught: " << e.what() << std::endl);
    }

    try {
        CACHE_CUDA_ERR_CHK_AND_THROW(cudaErrorInvalidValue);
    } catch (nve::ECException& e) {
        caught_exceptions++;
        NVE_DEBUG_PRINT_("Test Caught: " << e.what() << std::endl);
    }

    try {
        CHECK_ERR_AND_THROW(ECERROR_INVALID_ARGUMENT);
    } catch (nve::ECException& e) {
        caught_exceptions++;
        NVE_DEBUG_PRINT_("Test Caught: " << e.what() << std::endl);
    }
    EXPECT_EQ(caught_exceptions, 3);
}
