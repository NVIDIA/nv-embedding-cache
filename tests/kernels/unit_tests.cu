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
#include <datagen.h>
#include <memory>
#include <algorithm>
#include <thread>
#include <numeric>
#include "embedding_cache_combined.cuh"
#include "cuda_ops/scatter.cuh"
#include "cuda_ops/update_accumulate.cuh"
#include "cuda_ops/cuda_utils.cuh"
#include "cuda_ops/dedup_grads_kernel.cuh"
#include "cuda_ops/find_and_combine_kernel.cuh"
#include "cuda_ops/gradient_calculator.cuh"
#include "cuda_ops/gather_keys_data_ptrs.cuh"
#include "../common/buffer.h"
#include <default_allocator.hpp>

struct Near {
  float tolerance_;

  explicit Near(const float tol) : tolerance_(tol) {}

  __host__ __device__ bool operator()(const float& a, const float& b) const {
      return fabsf(a - b) <= tolerance_;
  }

  __host__ __device__ bool operator()(const __half& a, const __half& b) const {
      return fabsf(__half2float(a) - __half2float(b)) <= tolerance_;
  }
};

template <typename T>
class EmbeddingCacheRefTest : public ::testing::Test {
  public:
    typedef typename T::ElemType ElemType;
    typedef typename T::IndexType IndexType;

    using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
    using CacheDataType = typename CacheType::CacheData;

    EmbeddingCacheRefTest(): m_cache_allocator(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    }

    ~EmbeddingCacheRefTest() {
        CHECK_CUDA_ERROR(m_cache_ptr->lookup_context_destroy(m_handle_lookup));
        CHECK_CUDA_ERROR(m_cache_ptr->modify_context_destroy(m_handle_modify));
    }

    void LaunchTest(uint64_t num_rows, uint32_t num_elements, uint32_t batch, uint32_t hotness, bool allocDataOnHost, int64_t kernel_param = 0) {
        m_num_rows = num_rows;
        m_num_elements = num_elements;
        m_batch = batch;
        m_hotness = hotness;
        m_num_keys = static_cast<IndexType>(m_batch) * static_cast<IndexType>(m_hotness);
        m_allocDataOnHost = allocDataOnHost;
        m_kernel_param = kernel_param;
        AllocateTable();
        InitCache();
        InitTestExtras();
        AllocateKeys();
        CacheDataType cache_data = m_cache_ptr->get_cache_data(m_handle_lookup);
        std::vector<IndexType> cached_indices = 
            ComputeCachedIndices(cache_data.num_sets, CacheType::NUM_WAYS);
        PopulateCache(cached_indices);
        ComputeRefResults(m_table->ph, m_keys->ph, cached_indices);
        LaunchKernel(m_table->pd, m_keys->pd, cache_data);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CheckResult();
    }

    cudaStream_t m_stream;
    uint64_t m_num_rows{0};
    uint32_t m_num_elements{0}; 
    uint32_t m_batch{0};
    uint32_t m_hotness{0};
    IndexType m_num_keys{0};
    bool m_allocDataOnHost{false};
    int64_t m_kernel_param{0};
  protected:
  
    ElemType* syncTable() {
      m_table->DtoH(m_stream);
      return m_table->ph;
    }

    void GetCacheContent(ElemType* d_cache_content, std::shared_ptr<Buffer<IndexType>> keys_in_c, size_t& num_keys) 
    {
        
        m_cache_ptr->get_keys_stored_in_cache(m_handle_lookup, keys_in_c->ph, num_keys);
        keys_in_c->HtoD(m_stream);
        int64_t num_hitmask_elems = (num_keys + 31) / 32;
        auto hitmask = std::make_shared<Buffer<uint32_t>>(num_hitmask_elems * sizeof(uint32_t));
        CHECK_CUDA_ERROR(cudaMemset(hitmask->pd, 0, num_hitmask_elems * sizeof(uint32_t)));

        NVE_DEBUG_PRINTF_("launch lookup with hitmask kernel\n");

        m_cache_ptr->lookup(m_handle_lookup, keys_in_c->pd, num_keys, (int8_t*)d_cache_content, (uint64_t*)hitmask->pd, 0, m_num_elements*sizeof(ElemType), m_stream);
    }

    std::map<IndexType, std::vector<ElemType>> GetCacheContent()
    {
        auto num_vec_in_c = m_cache_ptr->get_max_num_embedding_vectors_in_cache();
        auto key_in_c = std::make_shared<Buffer<IndexType>>(sizeof(IndexType)*num_vec_in_c);
        auto cache_content = std::make_shared<Buffer<ElemType>>(sizeof(ElemType)*this->m_num_elements*num_vec_in_c);
        size_t num_keys_in_c = 0;
        this->GetCacheContent(cache_content->pd, key_in_c, num_keys_in_c);
        cache_content->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));
        std::map<IndexType, std::vector<ElemType>> mock_cache;
        for (size_t i = 0; i < num_keys_in_c; i++)
        {
            auto key = key_in_c->ph[i];
            mock_cache[key].resize(this->m_num_elements);
            memcpy(mock_cache[key].data(), cache_content->ph + i*this->m_num_elements, this->m_num_elements* sizeof(ElemType));
        }
        return mock_cache;
    }

    std::shared_ptr<Buffer<IndexType>> m_keys = nullptr;

  private:
    virtual void ComputeRefResults(const ElemType* /*table*/,
                                   const IndexType* /*indices*/,
                                   const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute baseline reference results\n");
    }

    virtual void LaunchKernel(const ElemType* /*table*/, 
                              const IndexType* /*indices*/,
                              CacheDataType& /*cache_data*/) {
    }

    virtual void CheckResult() {
    }
  
    virtual void InitTestExtras() {
    }

    void InitCache() {
        const float cache_ratio = 0.15f;
        const uint32_t num_rows_in_cache = static_cast<uint32_t>(cache_ratio * static_cast<float>(this->m_num_rows));

        typename CacheType::CacheConfig cfg;
        cfg.embed_width_in_bytes = this->m_num_elements * sizeof(ElemType);

        NVE_DEBUG_PRINTF_("cache of %d entries\n", int(cache_ratio * float(this->m_num_rows)));
        cfg.cache_sz_in_bytes = num_rows_in_cache * cfg.embed_width_in_bytes;
        cfg.num_tables = 1;
        cfg.allocate_data_on_host = m_allocDataOnHost;

        m_cache_ptr = std::make_shared<CacheType>(&m_cache_allocator, &m_cache_logger, cfg);
        m_cache_ptr->init();

        m_cache_ptr->lookup_context_create(m_handle_lookup, nullptr, 0);
    }

    void AllocateTable() {
        m_table = std::make_shared<Buffer<ElemType>>(this->m_num_rows * this->m_num_elements * sizeof(ElemType));
        std::mt19937 gen(0X814753);
        // init to [a / 2^b] when a and b are in specific range,
        // to minimize arithmetic error in half when doing weigthed math
        // inputs in the range of [-1.0f, 1.0f] accumulate larger errors
        // (tolerance of 1e-3f is required to pass the tests)
        std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
        std::uniform_int_distribution<uint32_t> dist_denom(3, 9);

        for (uint64_t i = 0; i < this->m_num_rows; i++)
        {
            ElemType* curr_row = reinterpret_cast<ElemType*>(m_table->ph + i * this->m_num_elements);
            for (uint64_t j = 0; j < this->m_num_elements; j++)
            {
                float nom = float(dist_nom(gen));
                float denom = float(1 << dist_denom(gen));
                curr_row[j] = nom / denom;
            }
        }

        m_table->HtoD(m_stream);
    }

    virtual void AllocateKeys() {
        m_keys = std::make_shared<Buffer<IndexType>>(m_num_keys * sizeof(IndexType));
        const float alpha = 1.05f;

        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->m_num_rows), this->m_hotness, 283982);
        for (uint32_t b = 0; b < this->m_batch; b++)
        {
            auto sample = sg->getCategoryIndices();
            std::copy(sample.begin(), sample.end(), m_keys->ph + b*this->m_hotness);
        }
        m_keys->HtoD(m_stream);
    }

    std::vector<IndexType> ComputeCachedIndices(const int num_sets, const int num_ways) {
        std::set<IndexType> cached_indices;
        std::vector<int> counters(num_sets, 0);

        uint64_t cache_capacity = static_cast<uint64_t>(static_cast<double>(this->m_num_rows) * 0.15);

        while (cache_capacity > static_cast<uint64_t>(this->m_num_keys)) {
            cache_capacity /= 2;
        }

        uint32_t step = static_cast<uint32_t>(this->m_num_keys / cache_capacity);
        EXPECT_TRUE(step > 0); // batch_size * hotness > cache_capcitiy 
        // for now: random indices
        for (IndexType i=0; i < this->m_num_keys; i+=step) {
            IndexType idx = m_keys->ph[i];
            if (++counters[idx % num_sets] <= num_ways) {
                cached_indices.insert(idx);
            }
        }

        std::vector<IndexType> idx_vec;
        for (typename std::set<IndexType>::iterator itr = cached_indices.begin(); itr != cached_indices.end(); ++itr) {
            idx_vec.push_back(*itr);
        }
        return idx_vec;
    }

    void PopulateCache(const std::vector<IndexType>& cached_indices) {
        NVE_DEBUG_PRINTF_("inserting %lu embeds to cache\n", cached_indices.size());

        std::vector<float> priorities(cached_indices.size(), 1.0f);
        m_cache_ptr->modify_context_create(m_handle_modify, static_cast<uint32_t>(cached_indices.size()));

        nve::DefaultHistogram<IndexType> hist(cached_indices.data(), cached_indices.size(),
                                              reinterpret_cast<const int8_t*>(m_table->ph),
                                              this->m_num_elements * sizeof(ElemType), true);
        cudaEvent_t wait;
        CHECK_CUDA_ERROR(cudaEventCreate(&wait));
        nve::DefaultECEvent ec_event(std::vector<cudaStream_t>{});

        m_cache_ptr->insert(
            m_handle_modify, hist.get_keys(), hist.get_priority(), hist.get_data(), hist.get_num_bins(), 0, &ec_event, m_stream);

        CHECK_CUDA_ERROR(cudaEventRecord(wait));
        CHECK_CUDA_ERROR(cudaEventSynchronize(wait));

    }

    std::shared_ptr<Buffer<ElemType>> m_table = nullptr;

    nve::DefaultAllocator m_cache_allocator;
    nve::Logger m_cache_logger;
    nve::PerformanceMetric m_miss_count;
    std::shared_ptr<CacheType> m_cache_ptr;
    nve::LookupContextHandle m_handle_lookup;
    nve::ModifyContextHandle m_handle_modify;
};

template <typename T>
class EmbeddingCacheLookupRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using AccumType = typename T::AccumType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:

    void InitTestExtras() {
        if (T::FIXED_HOTNESS_FLAG) {
            this->m_num_keys = this->m_batch * this->m_hotness;
        } else {
            // allocate and init offsets
            m_offsets = std::make_shared<Buffer<IndexType>>((this->m_batch + 1) * sizeof(IndexType));
            std::mt19937 gen(0X475381);
            std::uniform_int_distribution<uint32_t> dist_offset(1, 31);

            IndexType curr_offset = 0;
            for (uint32_t b = 0; b < this->m_batch; b++) {
                m_offsets->ph[b] = curr_offset;
                curr_offset += dist_offset(gen);
            }
            this->m_num_keys = curr_offset;
            m_offsets->ph[this->m_batch] = curr_offset;
            m_offsets->HtoD(this->m_stream);
        }

        if (T::IS_WEIGHTED_FLAG) {
            // allocate and init weights
            m_weights = std::make_shared<Buffer<ElemType>>(this->m_num_keys * sizeof(ElemType));
            std::mt19937 genr(0X753812);
            std::uniform_real_distribution<float> dist_float(0.1f, 1.0f);
            std::bernoulli_distribution distrib(0.5);
            for (IndexType i=0 ; i < this->m_num_keys; i++) {
                m_weights->ph[i] = distrib(genr) ? ElemType(0.5f) : ElemType(0.25f);
            }

            m_weights->HtoD(this->m_stream);
        }
    }

    void AllocateKeys() {
        this->m_keys = std::make_shared<Buffer<IndexType>>(this->m_num_keys * sizeof(IndexType));
        const float alpha = 1.05f;

        const size_t seed = 283982;
        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->m_num_rows),
                                                T::FIXED_HOTNESS_FLAG ? this->m_hotness : 32, seed);
        for (uint32_t b = 0; b < this->m_batch; b++)
        {
            auto sample = sg->getCategoryIndices();
            IndexType start_pos = T::FIXED_HOTNESS_FLAG ? b * this->m_hotness : this->m_offsets->ph[b];
            IndexType keys_to_copy = T::FIXED_HOTNESS_FLAG ? this->m_hotness :
                                                             (this->m_offsets->ph[b + 1] - this->m_offsets->ph[b]);
            std::copy(sample.begin(), sample.begin() + keys_to_copy, this->m_keys->ph + start_pos);
        }
        this->m_keys->HtoD(this->m_stream);
    }

    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute lookup reference results\n");
        m_ref_result.resize(0);
        for (uint32_t b = 0; b < this->m_batch; b++) {
            IndexType hotness;
            IndexType start_idx;
            if (T::FIXED_HOTNESS_FLAG) {
                hotness = this->m_hotness;
                start_idx = b * hotness;
            } else {
                hotness = m_offsets->ph[b+1] - m_offsets->ph[b];
                start_idx = m_offsets->ph[b];
            }

            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                AccumType acc = 0;
                AccumType weight_acc = 0;
                
                for (IndexType h = 0; h < hotness; h++) {
                    AccumType el_cast = table[indices[start_idx + h] * this->m_num_elements + el];
                    if (T::IS_WEIGHTED_FLAG) {
                        AccumType weight_cast = m_weights->ph[start_idx + h];
                        acc += el_cast * weight_cast;
                        weight_acc += weight_cast;
                    } else {
                        acc += el_cast;
                    }
                }
                if (T::SUM_POOLING_FLAG) {
                    m_ref_result.push_back(acc);
                } else {
                    if (T::IS_WEIGHTED_FLAG) {
                        m_ref_result.push_back(weight_acc != AccumType(0) ? acc / weight_acc : AccumType(1));
                    } else {
                        m_ref_result.push_back(acc / AccumType(hotness));
                    }
                }
            }
        }
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        m_result = std::make_shared<Buffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));
        NVE_DEBUG_PRINTF_("launch lookup kernel\n");

        int8_t** tables_d;
        std::vector<const int8_t*> tables_h;
        tables_h.push_back(reinterpret_cast <const int8_t*>(table));
        CHECK_CUDA_ERROR(cudaMalloc(&tables_d, sizeof(int8_t*)));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(tables_d, tables_h.data(), sizeof(int8_t*), cudaMemcpyDefault, this->m_stream));

        callFindAndCombineKernel<ElemType, IndexType, AccumType, CacheDataType,
                                 T::FIXED_HOTNESS_FLAG, T::SUM_POOLING_FLAG, T::IS_WEIGHTED_FLAG>(
            this->m_batch, reinterpret_cast<const int8_t*>(table), indices,
            m_offsets == nullptr ? nullptr : m_offsets->pd,
            m_weights == nullptr ? nullptr : m_weights->pd,
            this->m_hotness, cache_data, this->m_num_elements,
            m_result->pd, this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));
        CHECK_CUDA_ERROR(cudaFree(tables_d));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup results\n");
        m_result->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if ((!T::IS_WEIGHTED_FLAG) && T::SUM_POOLING_FLAG) {
            const float tolerance = 0;
            EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), m_result->ph, Near(tolerance)));
        } else {
            const float tolerance = 1e-7f;
            EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), m_result->ph, Near(tolerance)));
        }
    }

    std::shared_ptr<Buffer<ElemType>> m_result = nullptr;
    std::vector<ElemType> m_ref_result;
    std::shared_ptr<Buffer<ElemType>> m_weights = nullptr;
    std::shared_ptr<Buffer<IndexType>> m_offsets = nullptr;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupRefTest);

TYPED_TEST_P(EmbeddingCacheLookupRefTest, TestPoolingAgainstRefCpu) {
    // Removed "nice" hotness and num elements values in favour of multiple pooling modes
    // Try multiples of 32 in case of test failures
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {59}) {
            for (const auto num_elements : {132, 30, 63}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheLookupRefTest, TestPoolingAgainstRefCpu);

template <typename T>
class EmbeddingCacheLookupHitMaskRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& cache_data) {

        NVE_DEBUG_PRINTF_("compute lookup with hitmask reference results\n");

        uint32_t mask = 0;
        m_ref_result.resize(this->m_batch * this->m_hotness * this->m_num_elements);
        m_ref_hitmask.resize(0);
        int64_t i = 0;
        for (; i < static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness); i++) {
            IndexType idx = indices[i];

            bool in_cache = std::binary_search(cache_data.begin(), cache_data.end(), idx);
            if (in_cache) {
               for (uint32_t el = 0; el < this->m_num_elements; el++) {
                    m_ref_result[i * this->m_num_elements + el] = table[idx * this->m_num_elements + el];
               }
               mask |= 0x80000000;
            }

            if (((i + 1) % 32) == 0) {
                m_ref_hitmask.push_back(mask);
                mask = 0;
            } else {
                mask >>= 1;
            }

        }
        uint64_t set_bits = i % 32;
        if (set_bits != 0) {
            mask >>= (32-set_bits-1);
            m_ref_hitmask.push_back(mask);
        }
    }

    void LaunchKernel(const ElemType* /*table*/, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        int64_t num_hitmask_elems = (num_indices + 31) / 32;

        m_result = std::make_shared<Buffer<ElemType>>(num_indices *this->m_num_elements * sizeof(ElemType));
        m_hitmask = std::make_shared<Buffer<uint32_t>>(num_hitmask_elems * sizeof(uint32_t));

        std::memset(m_hitmask->ph, 0, num_hitmask_elems * sizeof(uint32_t));
        m_hitmask->HtoD(this->m_stream);

        NVE_DEBUG_PRINTF_("launch lookup with hitmask kernel\n");

        CHECK_CUDA_ERROR((nve::call_cache_query_hit_mask<IndexType, IndexType>(
            indices, this->m_batch * this->m_hotness,
            reinterpret_cast<int8_t*>(m_result->pd), m_hitmask->pd, cache_data,
            this->m_stream, 0, this->m_num_elements * sizeof(ElemType))));                         
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup with hitmask results\n");
        m_result->DtoH(this->m_stream);
        m_hitmask->DtoH(this->m_stream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        EXPECT_TRUE(std::equal(m_ref_hitmask.begin(), m_ref_hitmask.end(), m_hitmask->ph)); 

        int64_t curr_mask_idx = 0;
        int64_t curr_mask_el = 1;

        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        for (int64_t i = 0; i < num_indices; i++) {
            uint32_t curr_mask = m_ref_hitmask[curr_mask_idx];
            if ((curr_mask & curr_mask_el) == 0x1) {
                EXPECT_TRUE(std::equal(m_result->ph + i * this->m_num_elements,
                                       m_result->ph + (i + 1) * this->m_num_elements,
                                       m_ref_result.begin() + i * this->m_num_elements)); 
            }
            if (((i + 1) % 32) == 0) {
                ++curr_mask_idx;
                curr_mask_el = 1;
            } else {
                curr_mask_el <<= 1;
            }
        }
    }

    std::shared_ptr<Buffer<ElemType>> m_result = nullptr;
    std::shared_ptr<Buffer<uint32_t>> m_hitmask = nullptr;
    std::vector<ElemType> m_ref_result;
    std::vector<uint32_t> m_ref_hitmask;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupHitMaskRefTest);

TYPED_TEST_P(EmbeddingCacheLookupHitMaskRefTest, TestFixedHotnessAgainstRefCpu) {
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {13, 32}) {
            for (const auto num_elements : {53, 512}) {
                for(const auto allocOnHost: {false, true}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheLookupHitMaskRefTest, TestFixedHotnessAgainstRefCpu);

template <typename T>
class ScatterRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& cache_data) {
        NVE_DEBUG_PRINTF_("compute scatter reference results\n");

        // reference - just do gather of all indices
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        int64_t num_hitmask_elems = (num_indices + 63) / 64;
        m_hitmask = std::make_shared<Buffer<uint64_t>>(num_hitmask_elems * sizeof(uint64_t));
        m_input = std::make_shared<Buffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));

        uint64_t mask = 0;

        int64_t i = 0;
        for (; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_input->ph[i * this->m_num_elements + el] = table[idx * this->m_num_elements + el];
            }
            bool in_cache = std::binary_search(cache_data.begin(), cache_data.end(), idx);
            if (in_cache) {
               mask |= 0x80000000;
            }

            if (((i + 1) % 64) == 0) {
                m_hitmask->ph[i / 64] = mask;
                mask = 0;
            } else {
                mask >>= 1;
            }

        }
        if (( i % 64) != 0) {
            m_hitmask->ph[i / 64] = mask;
        }
    }

    void LaunchKernel(const ElemType* /*table*/,
                      const IndexType* /*indices*/,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch scatter kernel\n");

        m_hitmask->HtoD(this->m_stream);
        m_input->HtoD(this->m_stream);
        m_result = std::make_shared<Buffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));
        uint64_t num_indices = static_cast<uint64_t>(this->m_batch) * static_cast<uint64_t>(this->m_hotness);
        for (uint64_t i = 0; i < num_indices; i++) {
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_result->ph[i * this->m_num_elements + el] = ElemType(8);
            }
        }
        m_result->HtoD(this->m_stream);

        uint32_t embed_width_in_bytes = static_cast<uint32_t>(this->m_num_elements * sizeof(ElemType));
        nve::EmbeddingForwardScatter(m_input->pd, m_result->pd, embed_width_in_bytes,
                                            embed_width_in_bytes, embed_width_in_bytes,
                                            reinterpret_cast<uint64_t*>(m_hitmask->pd),
                                            static_cast<uint32_t>(num_indices), this->m_stream);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check scatter results\n");
        m_result->DtoH(this->m_stream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        int64_t curr_mask_idx = 0;
        int64_t curr_mask_el = 1;

        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        for (int64_t i = 0; i < num_indices; i++) {
            if ((m_hitmask->ph[curr_mask_idx] & curr_mask_el) == 0) {                
                EXPECT_TRUE(std::equal(m_result->ph + i * this->m_num_elements,
                                       m_result->ph + (i + 1) * this->m_num_elements,
                                       m_input->ph + i * this->m_num_elements));
            } else {
                EXPECT_FALSE(std::equal(m_result->ph + i * this->m_num_elements,
                                       m_result->ph + (i + 1) * this->m_num_elements,
                                       m_input->ph + i * this->m_num_elements)); 
            }
            if (((i + 1) % 64) == 0) {
                ++curr_mask_idx;
                curr_mask_el = 1;
            } else {
                curr_mask_el <<= 1;
            }
        }
    }

    std::shared_ptr<Buffer<ElemType>> m_result = nullptr;
    std::shared_ptr<Buffer<ElemType>> m_input = nullptr;
    std::shared_ptr<Buffer<uint64_t>> m_hitmask = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(ScatterRefTest);

TYPED_TEST_P(ScatterRefTest, TestScatterAgainstRefCpu) {
    for (const auto batch : {17, 2048}) {
        for (const auto hotness : {13, 32}) {
            for (const auto num_elements : {13, 512}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(ScatterRefTest, TestScatterAgainstRefCpu);

template <typename T>
class UpdateRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute update reference results\n");

        // reference - just do gather of all indices
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        m_input = std::make_shared<Buffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));

        m_ref_result.resize(this->m_num_elements * this->m_num_rows);
        memcpy(&m_ref_result[0], table, this->m_num_elements * this->m_num_rows * sizeof(ElemType));

        // generate inputs to update and update ref
        int64_t i = 0;
        for (; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_input->ph[i * this->m_num_elements + el] = table[idx * this->m_num_elements + el] + ElemType(i);
                m_ref_result[idx * this->m_num_elements + el] += ElemType(i);
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch update kernel\n");

        m_input->HtoD(this->m_stream);

        uint32_t embed_width_in_bytes = static_cast<uint32_t>(this->m_num_elements * sizeof(ElemType));
        uint64_t num_indices = static_cast<uint64_t>(this->m_batch) * static_cast<uint64_t>(this->m_hotness);
        nve::UpdateTable(m_input->pd, indices, const_cast<ElemType*>(table), 
                         embed_width_in_bytes, embed_width_in_bytes, embed_width_in_bytes,
                         static_cast<uint32_t>(num_indices), this->m_stream);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update results\n");

        ElemType* curr_table = this->syncTable();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), curr_table));
    }

    std::shared_ptr<Buffer<ElemType>> m_input = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(UpdateRefTest);

// this test should get unique keys, therefore using batch size of 1
TYPED_TEST_P(UpdateRefTest, TestUpdateAgainstRefCpu) {
    for (const auto hotness : {17, 2048}) {
        for (const auto num_elements : {53, 512}) {
            for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", 1, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, 1, hotness, allocOnHost);
                }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(UpdateRefTest, TestUpdateAgainstRefCpu);

template <typename T>
class UpdateAccumulateRefTest : public EmbeddingCacheRefTest<T> {
  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute update accumulate reference results\n");

        // reference - just do gather of all indices
        int64_t num_indices = static_cast<int64_t>(this->m_batch) * static_cast<int64_t>(this->m_hotness);
        m_input = std::make_shared<Buffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));

        m_ref_result.resize(this->m_num_elements * this->m_num_rows);
        memcpy(&m_ref_result[0], table, this->m_num_elements * this->m_num_rows * sizeof(ElemType));

        // generate inputs to update and update ref
        std::mt19937 genr(0X753812);
        std::uniform_int_distribution<int32_t> dist_nom(-3, 3);
        std::uniform_int_distribution<uint32_t> dist_denom(5, 9);

        for (int64_t i = 0; i < num_indices; i++) {
            IndexType idx = indices[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                float val = float(dist_nom(genr)) / float(1 << dist_denom(genr));
                m_input->ph[i * this->m_num_elements + el] = ElemType(val);
                m_ref_result[idx * this->m_num_elements + el] += ElemType(val);
            }
        }
    }

    void LaunchKernel(const ElemType* table,
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch update accumulate kernel\n");

        m_input->HtoD(this->m_stream);

        uint64_t num_indices = static_cast<uint64_t>(this->m_batch) * static_cast<uint64_t>(this->m_hotness);
        nve::UpdateAccumulateTable(m_input->pd, indices, const_cast<ElemType*>(table), 
                                   this->m_num_elements, this->m_num_elements, this->m_num_elements,
                                   static_cast<uint32_t>(num_indices), this->m_stream);
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update accumulate results\n");

        ElemType* curr_table = this->syncTable();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        const float tolerance = 1e-5f;
        EXPECT_TRUE(std::equal(m_ref_result.begin(), m_ref_result.end(), curr_table, Near(tolerance)));
    }

    std::shared_ptr<Buffer<ElemType>> m_input = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(UpdateAccumulateRefTest);

TYPED_TEST_P(UpdateAccumulateRefTest, TestUpdateAccumulateAgainstRefCpu) {
    for (const auto batch : {171}) {
        for (const auto hotness : {16, 111}) {
            for (const auto num_elements : {53, 512}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                        this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
            }
                }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(UpdateAccumulateRefTest, TestUpdateAccumulateAgainstRefCpu);

template <typename T>
class EmbeddingCacheFusedUpdateAccumlateTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute table and cache results\n");
        NVE_DEBUG_PRINTF_("compute update reference results\n");

        // reference - just do gather of all indices
        IndexType num_keys = this->m_batch * this->m_hotness;

        // first compute cache content
        m_ref_cache = this->GetCacheContent();
        
        // second dedup keys
        m_unique_keys = std::make_shared<Buffer<IndexType>>(num_keys * sizeof(IndexType));
        // allocate 1 extra element for counts, because we call cub exclusive prefix sum kernel
        // to compute n + 1 outputs, and it reads the last input element even though it is not used
        // in the output
        auto counts_out = std::make_shared<Buffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
        
        auto inverse_buffer = std::make_shared<Buffer<IndexType>>(num_keys * sizeof(IndexType));
        auto offsets = std::make_shared<Buffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
        
        std::shared_ptr<Deduper<IndexType, -1>> deduper_ = std::make_shared<Deduper<IndexType, -1>>();
        size_t tmp_mem_size_device, tmp_mem_size_host;
        deduper_->GetAllocRequirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_device_mem;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
        char* tmp_host_mem;
        CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
        deduper_->SetAndInitBuffers(num_keys, tmp_device_mem, tmp_host_mem);

        CHECK_CUDA_ERROR(cudaMallocHost(&m_h_num_runs_out, sizeof(IndexType) * 2));

        deduper_->Dedup(reinterpret_cast<const IndexType*>(indices),
                        num_keys, m_unique_keys->pd, 
                        counts_out->pd,
                        nullptr, 
                        m_h_num_runs_out, 
                        inverse_buffer->pd, 
                        offsets->pd,
                        this->m_stream);

        m_unique_keys->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        m_input = std::make_shared<Buffer<ElemType>>(*m_h_num_runs_out * this->m_num_elements * sizeof(ElemType));

        m_ref_result.resize(this->m_num_elements * this->m_num_rows);
        memcpy(&m_ref_result[0], table, this->m_num_elements * this->m_num_rows * sizeof(ElemType));

        // generate inputs to update and update ref
        for (int64_t i = 0; i < *m_h_num_runs_out; i++) {
            IndexType idx = m_unique_keys->ph[i];
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                m_input->ph[i * this->m_num_elements + el] = ElemType(i);
                m_ref_result[idx * this->m_num_elements + el] += ElemType(i);
                if (m_ref_cache.count(idx))
                    m_ref_cache[idx][el] += ElemType(i);
            }
        }

        CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* /*indices*/,
                      CacheDataType& cache_data) {
        NVE_DEBUG_PRINTF_("launch update kernel\n");

        m_input->HtoD(this->m_stream);

        uint64_t num_indices = *m_h_num_runs_out;
        CHECK_CUDA_ERROR((nve::call_update_accumulate_no_sync_fused_with_pipeline<IndexType,IndexType>(reinterpret_cast<int8_t*>(m_input->pd), 
                                   (int8_t*)(table), 
                                   m_unique_keys->pd, 
                                   static_cast<IndexType>(num_indices),
                                   this->m_num_elements * sizeof(ElemType),
                                   cache_data,
                                   this->m_stream)));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check update results\n");

        ElemType* curr_table = this->syncTable();
        auto test_cache = this->GetCacheContent();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (uint32_t j = 0; j < this->m_num_rows; j ++ ) {
            for (uint32_t i = 0; i < this->m_num_elements; i++)
            {
                auto ref_val = m_ref_result[j*this->m_num_elements + i];
                auto test_val = curr_table[j*this->m_num_elements + i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }
        }

        // compare cache content
        ASSERT_EQ(test_cache.size(), m_ref_cache.size());
        for (const auto& ent : test_cache)
        {
            auto key = ent.first;
            auto test_vec = ent.second;
            auto ref_it = m_ref_cache.find(key);
            ASSERT_TRUE(ref_it != m_ref_cache.end());
            auto ref_vec = ref_it->second;
            for (uint32_t i = 0; i < this->m_num_elements; i++)
            {
                auto ref_val = ref_vec[i];
                auto test_val = test_vec[i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }

        }

    }

    std::shared_ptr<Buffer<ElemType>> m_input = nullptr;
    std::vector<ElemType> m_ref_result;
    std::shared_ptr<Buffer<IndexType>> m_unique_keys = nullptr;
    IndexType* m_h_num_runs_out = nullptr;
    std::map<IndexType, std::vector<ElemType>> m_ref_cache;
};

TYPED_TEST_SUITE_P(EmbeddingCacheFusedUpdateAccumlateTest);

TYPED_TEST_P(EmbeddingCacheFusedUpdateAccumlateTest, TestFusedAccumulateRefCpu) {
    for (const auto batch : {32*15*8, 12, 348989}) {
        for (const auto hotness : {1}) {
            for (const auto num_elements : {128}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheFusedUpdateAccumlateTest, TestFusedAccumulateRefCpu);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class EmbeddingCacheLookupSortGatherRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:
    void ComputeRefResults(const ElemType* table,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute lookup reference results\n");
        m_ref_result.resize(0);
        ASSERT_TRUE(this->m_hotness == 1);
        for (uint32_t b = 0; b < this->m_batch; b++) {
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                ElemType acc = 0;
                acc = table[indices[b] * this->m_num_elements + el];
                m_ref_result.push_back(acc);
            }
        }
    }

    void LaunchKernel(const ElemType* table, 
                      const IndexType* indices,
                      CacheDataType& cache_data) {
        m_result = std::make_shared<Buffer<ElemType>>(this->m_batch * this->m_hotness * this->m_num_elements * sizeof(ElemType));
        NVE_DEBUG_PRINTF_("launch sort gather lookup kernel\n");

        // first calc required aux size
        size_t bytes = 0;
        CHECK_CUDA_ERROR((nve::call_sort_gather<IndexType, IndexType>(
            reinterpret_cast<const int8_t*>(table), 
            reinterpret_cast<int8_t*>(m_result->pd), 
            indices, 
            nullptr, 
            bytes,
            this->m_batch * this->m_hotness, 
            this->m_num_elements * sizeof(ElemType), 
            this->m_kernel_param,
            cache_data, 
            this->m_stream)));
        ASSERT_TRUE(bytes > 0 && bytes != static_cast<size_t>(-1));

        // alloc aux
        int8_t* aux = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&aux, bytes));

        // launch kernel
        CHECK_CUDA_ERROR((nve::call_sort_gather<IndexType, IndexType>(
            reinterpret_cast<const int8_t*>(table), 
            reinterpret_cast<int8_t*>(m_result->pd), 
            indices, 
            aux, 
            bytes,
            this->m_batch * this->m_hotness, 
            this->m_num_elements * sizeof(ElemType), 
            1024,
            cache_data, 
            this->m_stream)));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        //clean up
        CHECK_CUDA_ERROR(cudaFree(aux));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check lookup results\n");
        m_result->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (uint32_t j = 0; j < this->m_hotness * this->m_batch; j ++ ) {
            for (uint32_t i = 0; i < this->m_num_elements; i++)
            {
                auto ref_val = m_ref_result[j*this->m_num_elements + i];
                auto test_val = m_result->ph[j*this->m_num_elements + i];
                ASSERT_FLOAT_EQ(ref_val, test_val);
            }
        }
    }

    std::shared_ptr<Buffer<ElemType>> m_result = nullptr;
    std::vector<ElemType> m_ref_result;
};

TYPED_TEST_SUITE_P(EmbeddingCacheLookupSortGatherRefTest);

TYPED_TEST_P(EmbeddingCacheLookupSortGatherRefTest, TestSortGather) {
    for (const auto batch : {4096, 1, 33, 4097, 16*1024, 3*4096+1, 2*4096+1}) {
        for (const auto hotness : {1}) {
            for (const auto num_elements : {128, 1, 31}) {
                for(const auto allocOnHost: {false}){
                    for (const auto kernel_param : {1, 17, 32, 4096}) {
                        NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s kernel_param %d\n", batch, hotness, num_elements, allocOnHost?"host":"device", kernel_param);
                        this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost, kernel_param);
                    }
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(EmbeddingCacheLookupSortGatherRefTest, TestSortGather);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElemT, typename IndexT>
struct EmbedTestTypeCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
};

template <typename ElemT, typename IndexT, typename AccumT,
          bool FIXED_HOTNESS, bool SUM_POOILING, bool IS_WEIGHTED>
struct EmbedTestTypePoolingCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
  typedef AccumT AccumType;
  static bool constexpr FIXED_HOTNESS_FLAG = FIXED_HOTNESS;
  static bool constexpr SUM_POOLING_FLAG = SUM_POOILING;
  static bool constexpr IS_WEIGHTED_FLAG = IS_WEIGHTED;
};

template <typename ElemT, typename IndexT,
          bool FIXED_HOTNESS, PoolingType_t POOILING_TYPE>
struct EmbedTestTypeBackpropCombo {
  typedef ElemT ElemType;
  typedef IndexT IndexType;
  
  static bool constexpr FIXED_HOTNESS_FLAG = FIXED_HOTNESS;
  static PoolingType_t constexpr POOLING_TYPE = POOILING_TYPE;
};

// Testing only float so far to avoid accuracy issues with weight normalization 
typedef ::testing::Types<
                         // Concat
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::Concatenate>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::Concatenate>,
                         // SUM
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::Sum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::Sum>,
                         // MEAN
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::Mean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::Mean>,
                         // CSR 
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::Sum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::Sum>,
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::Mean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::Mean>,
                         // weighted
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int32_t, true, PoolingType_t::WeightedMean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, true, PoolingType_t::WeightedMean>,
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::WeightedSum>,
                         EmbedTestTypeBackpropCombo<float, int32_t, false, PoolingType_t::WeightedMean>,
                         EmbedTestTypeBackpropCombo<float, int64_t, false, PoolingType_t::WeightedMean>>
    EmbedTestBackpropTypes;

typedef ::testing::Types<
                        // SUM
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, true, false>,
                         // MEAN
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, false, false>,
                         // CSR 
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, true, false>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, false>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, false, false>,
                         // weighted
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, true, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, true, false, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, true, true>,
                         EmbedTestTypePoolingCombo<float, int32_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<float, int64_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, __half, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, __half, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int32_t, float, false, false, true>,
                         EmbedTestTypePoolingCombo<__half, int64_t, float, false, false, true>>
    EmbedTestPoolingTypes;

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>,
                         EmbedTestTypeCombo<float, int64_t>,
                         EmbedTestTypeCombo<__half, int32_t>,
                         EmbedTestTypeCombo<__half, int64_t>>
    EmbedTestTypes;

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>,
                         EmbedTestTypeCombo<__half, int64_t>>
    EmbedElementAgnosticTestTypes;

typedef ::testing::Types<EmbedTestTypeCombo<float, int32_t>>
    EmbedBasicTestTypes;

class EmbeddingCacheRefTestNames {
 public:
  template <typename T>
  static std::string GetName(const std::string& base_name, int i) {
    typedef typename T::ElemType ElemType;
    typedef typename T::IndexType IndexType;

    std::string test_name = base_name;
    if (std::is_same_v<ElemType, float>) {
      test_name += std::string("Elem[float]_");
    }
    if (std::is_same_v<ElemType, __half>) {
      test_name += std::string("Elem[half]_");
    }
    if (std::is_same_v<IndexType, int32_t>) {
      test_name += std::string("Index[int32]_");
    }
    if (std::is_same_v<IndexType, int64_t>) {
      test_name += std::string("Index[int64]_");
    }
    test_name += ::testing::PrintToString(i);
    return test_name;
  }
};

class EmbeddingCacheLookupRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("EmbeddingCacheLookupRefTest_", i);
  }
};

class EmbeddingCacheLookupHitMaskRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("EmbeddingCacheLookupHitMaskRefTest_", i);
  }
};

class ScatterRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("ScatterRefTest_", i);
  }
};

class UpdateRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("UpdateRefTest_", i);
  }
};

class UpdateAccumulateRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("UpdateAccumulateRefTest_", i);
  }
};

class FusedUpdateAccumulateRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("FusedUpdateAccumulateRefTest_", i);
  }
};

class EmbeddingCacheLookupSortGatherRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("LookupSortGatherRefTest_", i);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingLookup,    // TRTREC-44
                               EmbeddingCacheLookupRefTest,
                               EmbedTestPoolingTypes,
                               EmbeddingCacheLookupRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingLookupHitMask,
                               EmbeddingCacheLookupHitMaskRefTest,
                               EmbedTestTypes,
                               EmbeddingCacheLookupHitMaskRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(Scatter,
                               ScatterRefTest,
                               EmbedElementAgnosticTestTypes,
                               ScatterRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(Update,
                               UpdateRefTest,
                               EmbedElementAgnosticTestTypes,
                               UpdateRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(UpdateAccumulate,
                               UpdateAccumulateRefTest,
                               EmbedElementAgnosticTestTypes,
                               UpdateAccumulateRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(FusedUpdateAccumulate,
                               EmbeddingCacheFusedUpdateAccumlateTest,
                               EmbedBasicTestTypes,
                               FusedUpdateAccumulateRefTestNames);

INSTANTIATE_TYPED_TEST_SUITE_P(CacheLookupSortGather,
                               EmbeddingCacheLookupSortGatherRefTest,
                               EmbedBasicTestTypes,
                               EmbeddingCacheLookupSortGatherRefTestNames);


TEST(dedupgrad, dedupgrad)
{
    using IndexType = int64_t; //int32_t;
    using DataType = float;

    const size_t embedding_size = 212;
    const size_t m_num_rows = 1000000;
    const size_t m_row_size = sizeof(DataType)*embedding_size;
    
    const size_t m_batch = 512;
    const size_t m_hotness = 4096;
    const auto num_keys = m_batch * m_hotness;
    auto keys = std::make_shared<Buffer<IndexType>>(num_keys * sizeof(IndexType));
    auto unique_keys = std::make_shared<Buffer<IndexType>>(num_keys * sizeof(IndexType));
    // allocate an extra element for loc_map, because it is reused for counters, and the
    // way we use counters in Dedup may lead to a read of extra element in one of cub kernels 
    auto loc_map = std::make_shared<Buffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
    IndexType* h_num_runs_out;
    auto inverse_buffer = std::make_shared<Buffer<IndexType>>(num_keys * sizeof(IndexType));
    auto offsets = std::make_shared<Buffer<IndexType>>((num_keys + 1) * sizeof(IndexType));
    auto grads = std::make_shared<Buffer<DataType>>(num_keys * m_row_size);
    auto unique_grads = std::make_shared<Buffer<DataType>>(num_keys * m_row_size);
    auto ref = std::make_shared<Buffer<DataType>>(num_keys * m_row_size);

    cudaStream_t m_stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    
    const float alpha = 1.05f;
    auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(m_num_rows), m_hotness, 283982);
    
    // make sure we have keys appearing more than SPLIT_LIMIT times
    // change approx half of batches to contain approx half identical keys
    std::mt19937 genr(0X753812);
    std::bernoulli_distribution distrib(0.5);
    std::uniform_int_distribution<uint32_t> dist_index(0, m_hotness-1);

    for (uint32_t b = 0; b < m_batch; b++)
    {
        auto sample = sg->getCategoryIndices();
        if (distrib(genr)) {
            auto duplicate_idx = dist_index(genr);
            IndexType key_to_duplicate = sample[duplicate_idx];
            for (uint32_t h = 0; h < m_hotness; h++) {
                if (distrib(genr)) sample[h] = key_to_duplicate;
            }
        }
        std::copy(sample.begin(), sample.end(), keys->ph + b*m_hotness);
    }

    keys->HtoD(m_stream);
    
    constexpr int32_t MAX_RUN_SIZE = 1024;
    std::shared_ptr<Deduper<IndexType, MAX_RUN_SIZE>> deduper_ = std::make_shared<Deduper<IndexType, MAX_RUN_SIZE>>();
    size_t tmp_mem_size_device, tmp_mem_size_host;
    deduper_->GetAllocRequirements(num_keys, tmp_mem_size_device, tmp_mem_size_host);
    char* tmp_device_mem;
    CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
    char* tmp_host_mem;
    CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
    deduper_->SetAndInitBuffers(num_keys, tmp_device_mem, tmp_host_mem);

    CHECK_CUDA_ERROR(cudaMallocHost(&h_num_runs_out, sizeof(IndexType) * 2));

    memset(grads->ph, 0, grads->m_size);
    std::uniform_real_distribution<float> dist_float(-1.0f, 1.0f);
    for (uint32_t i = 0; i < num_keys; i++) {
        for (uint32_t j = 0; j < embedding_size; j++) {
            (grads->ph)[i*embedding_size + j] = static_cast<DataType>(dist_float(genr));
        }
    }
    grads->HtoD(m_stream);

    memset(unique_grads->ph, 0, unique_grads->m_size);
    unique_grads->HtoD(m_stream);

    deduper_->Dedup(reinterpret_cast<const IndexType*>(keys->pd),
                    num_keys, unique_keys->pd, 
                    loc_map->pd, // reuse loc_map buffer for counters
                    loc_map->pd, 
                    h_num_runs_out, 
                    inverse_buffer->pd, 
                    offsets->pd,
                    m_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
    
    DedupGradients<IndexType>(reinterpret_cast<const void*>(grads->pd), (void*)unique_grads->pd, 
                    DataType_t::Float32,
                    unique_keys->pd, 
                    (MAX_RUN_SIZE == -1) ? nullptr : loc_map->pd,
                    offsets->pd,
                    inverse_buffer->pd,
                    m_row_size,
                    h_num_runs_out,
                    m_stream);

    // calculate ref
    std::map<IndexType, DataType*> ref_grads;
    for (uint32_t i = 0; i < num_keys; i++) 
    {
        auto key = keys->ph[i];
        if (ref_grads.count(key) == 0) {
            DataType* p = new DataType[embedding_size];
            memset(p, 0, m_row_size);
            ref_grads[key] = p;
        }
        auto ref_grad = ref_grads[key];
        for (uint32_t j = 0; j < embedding_size; j++) {
            ref_grad[j] += grads->ph[i * embedding_size + j];
        }
    }

    // check results
    unique_grads->DtoH(m_stream);
    unique_keys->DtoH(m_stream);
    inverse_buffer->DtoH(m_stream);
    loc_map->DtoH(m_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

    for (IndexType i = 0; i < h_num_runs_out[0]; i++) 
    {
        auto unq_key = unique_keys->ph[i];
        DataType* kernel_ptr = unique_grads->ph + i * embedding_size;
        DataType* ref_ptr = ref_grads[unq_key];
        const float tolerance = (MAX_RUN_SIZE == -1) ? 0 : 1e-3f;
        EXPECT_TRUE(std::equal(ref_ptr, ref_ptr + embedding_size, kernel_ptr, Near(tolerance)));
    }

    CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
    CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
}

template <typename T>
class PoolingBackPropRefTest : public EmbeddingCacheRefTest<T> {
  public:

  using ElemType = typename EmbeddingCacheRefTest<T>::ElemType;
  using IndexType = typename EmbeddingCacheRefTest<T>::IndexType;
  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using CacheDataType = typename CacheType::CacheData;

  private:

    void InitTestExtras() {
        if (T::FIXED_HOTNESS_FLAG) {
            this->m_num_keys = this->m_batch * this->m_hotness;
        } else {
            // allocate and init offsets
            m_offsets = std::make_shared<Buffer<IndexType>>((this->m_batch + 1) * sizeof(IndexType));
            std::mt19937 gen(0X475381);
            std::uniform_int_distribution<uint32_t> dist_offset(1, 31);

            IndexType curr_offset = 0;
            for (uint32_t b = 0; b < this->m_batch; b++) {
                m_offsets->ph[b] = curr_offset;
                curr_offset += dist_offset(gen);
            }
            this->m_num_keys = curr_offset;
            m_offsets->ph[this->m_batch] = curr_offset;
            m_offsets->HtoD(this->m_stream);
        }

        // allocate and init weights
        m_weights = std::make_shared<Buffer<ElemType>>(this->m_num_keys * sizeof(ElemType));
        const bool is_weighted = (T::POOLING_TYPE == PoolingType_t::WeightedSum) ||
                                 (T::POOLING_TYPE == PoolingType_t::WeightedMean);
        if (is_weighted) {
            std::mt19937 genr(0X753812);
            std::uniform_real_distribution<float> dist_float(0.1f, 1.0f);
            std::bernoulli_distribution distrib(0.5);
            for (IndexType i=0 ; i < this->m_num_keys; i++) {
                m_weights->ph[i] = distrib(genr) ? ElemType(0.5f) : ElemType(0.25f);
            }
        } else {
            for (IndexType i=0 ; i < this->m_num_keys; i++) {
                m_weights->ph[i] = ElemType(1.0f);
            }
        }
        m_weights->HtoD(this->m_stream);

        uint32_t batch_ = this->m_batch;
        if (T::POOLING_TYPE == PoolingType_t::Concatenate) {
            batch_ *= this->m_hotness;
        }
        m_grads_in = std::make_shared<Buffer<ElemType>>(batch_ * this->m_num_elements * sizeof(ElemType));
        
        std::mt19937 genr(0X387512);
        std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
        std::uniform_int_distribution<uint32_t> dist_denom(3, 9);
        for (uint32_t b = 0; b < batch_; b++) {
            for (uint32_t el = 0; el < this->m_num_elements; el++) {
                float nom = float(dist_nom(genr));
                float denom = float(1 << dist_denom(genr));
                m_grads_in->ph[b * this->m_num_elements + el] = nom / denom;
            }
        }
        m_grads_in->HtoD(this->m_stream);
    }

    void AllocateKeys() {
        this->m_keys = std::make_shared<Buffer<IndexType>>(this->m_num_keys * sizeof(IndexType));
        const float alpha = 1.05f;

        const size_t seed = 283982;
        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->m_num_rows),
                                                T::FIXED_HOTNESS_FLAG ? this->m_hotness : 32, seed);
        for (uint32_t b = 0; b < this->m_batch; b++)
        {
            auto sample = sg->getCategoryIndices();
            IndexType start_pos = T::FIXED_HOTNESS_FLAG ? b * this->m_hotness : this->m_offsets->ph[b];
            IndexType keys_to_copy = T::FIXED_HOTNESS_FLAG ? this->m_hotness :
                                                             (this->m_offsets->ph[b + 1] - this->m_offsets->ph[b]);
            std::copy(sample.begin(), sample.begin() + keys_to_copy, this->m_keys->ph + start_pos);
        }
        this->m_keys->HtoD(this->m_stream);
    }

    void ComputeRefResults(const ElemType* /*table*/,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute pooling backprop reference results\n");

        m_ref_grads.clear();
        for (uint32_t b = 0; b < this->m_batch; b++) {
            IndexType hotness;
            IndexType start_idx;
            if (T::FIXED_HOTNESS_FLAG) {
                hotness = this->m_hotness;
                start_idx = b * hotness;
            } else {
                hotness = m_offsets->ph[b+1] - m_offsets->ph[b];
                start_idx = m_offsets->ph[b];
            }

            ElemType sum_weights = 0;
            switch (T::POOLING_TYPE) {
                case PoolingType_t::WeightedMean:
                    for (IndexType h = 0; h < hotness; h++) {
                        sum_weights += ElemType(m_weights->ph[start_idx + h]);
                    }
                    break;
                case PoolingType_t::Mean:
                    sum_weights = ElemType(hotness);
                    break;
                default:
                    sum_weights = 1.0f;
                    break;
            }

            for (IndexType h = 0; h < hotness; h++) {
                auto key = indices[start_idx + h];
                if (m_ref_grads.count(key) == 0) {
                    ElemType* p = new ElemType[this->m_num_elements];
                    memset(p, 0, this->m_num_elements * sizeof(ElemType));
                    m_ref_grads[key] = p;
                }
                auto ref_grad = m_ref_grads[key];
                ElemType* input_grad = m_grads_in->ph + b * this->m_num_elements;

                if (T::POOLING_TYPE == PoolingType_t::Concatenate) {
                    input_grad = m_grads_in->ph + b * hotness * this->m_num_elements + h * this->m_num_elements;
                }

                for (uint32_t el = 0; el < this->m_num_elements; el++) {
                    ElemType el_cast = input_grad[el];
                    ElemType weight_cast = 1.0f;

                    if ((T::POOLING_TYPE == PoolingType_t::WeightedSum) ||
                        (T::POOLING_TYPE == PoolingType_t::WeightedMean)) {
                        weight_cast = m_weights->ph[start_idx + h];
                    }
                    ref_grad[el] += el_cast * (weight_cast / sum_weights);
                }
            }
        }
    }

    void LaunchKernel(const ElemType* /*table*/, 
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch pooling backprop kernel %lu keys\n", static_cast<uint64_t>(this->m_num_keys));

        m_unique_keys = std::make_shared<Buffer<IndexType>>(this->m_num_keys * sizeof(IndexType));
        m_grads_out = std::make_shared<Buffer<ElemType>>(this->m_num_keys * this->m_num_elements * sizeof(ElemType));
    
        const int32_t MAX_RUN_SIZE = 32;
        m_split_grads = MAX_RUN_SIZE != -1;

        GradientCalculator<IndexType, MAX_RUN_SIZE> pool_backprop;

        size_t tmp_mem_size_device, tmp_mem_size_host;
        size_t element_size = sizeof(ElemType);
        pool_backprop.GetAllocRequirements(this->m_num_keys, element_size, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_device_mem;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
        char* tmp_host_mem;
        CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
        pool_backprop.SetAndInitBuffers(this->m_num_keys, element_size, tmp_device_mem, tmp_host_mem);

        PoolingType_t pooling_type = T::POOLING_TYPE;

        pool_backprop.template ComputeGradients<ElemType, T::FIXED_HOTNESS_FLAG>(
            indices,
            T::FIXED_HOTNESS_FLAG ? nullptr : m_offsets->pd,
            m_grads_in->pd,
            m_weights->pd,
            m_grads_out->pd,
            m_unique_keys->pd,
            this->m_batch,
            this->m_num_keys,
            this->m_hotness,
            this->m_num_elements,
            pooling_type,
            this->m_stream);

        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));

        CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check pooling backprop results\n");
        
        // check results
        m_unique_keys->DtoH(this->m_stream);
        m_grads_out->DtoH(this->m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->m_stream));

        const float tolerance = (m_split_grads ||
                                (T::POOLING_TYPE == PoolingType_t::Mean) ||
                                (T::POOLING_TYPE == PoolingType_t::WeightedMean)) ? 1e-5f : 0;

        for (unsigned long i = 0; i < m_ref_grads.size(); i++) 
        {
            auto unq_key = m_unique_keys->ph[i];
            ElemType* kernel_ptr = m_grads_out->ph + i * this->m_num_elements;
            ElemType* ref_ptr = m_ref_grads[unq_key];
            EXPECT_TRUE(std::equal(ref_ptr, ref_ptr + this->m_num_elements, kernel_ptr, Near(tolerance)));
        }
    }

    std::vector<ElemType> m_ref_result;
    std::shared_ptr<Buffer<ElemType>> m_weights = nullptr;
    std::shared_ptr<Buffer<IndexType>> m_offsets = nullptr;
    std::shared_ptr<Buffer<ElemType>> m_grads_in = nullptr;
    std::shared_ptr<Buffer<ElemType>> m_grads_out = nullptr;
    std::shared_ptr<Buffer<IndexType>> m_unique_keys = nullptr;

    std::map<IndexType, ElemType*> m_ref_grads;

    bool m_split_grads = false;
};

TYPED_TEST_SUITE_P(PoolingBackPropRefTest);

TYPED_TEST_P(PoolingBackPropRefTest, TestPoolingBackpropAgainstRefCpu) {
    // Removed "nice" hotness and num elements values in favour of multiple pooling modes
    // Try multiples of 32 in case of test failures
    for (const auto batch : {171}) {
        for (const auto hotness : {61}) {
            for (const auto num_elements : {53, 128, 510}) {
                for(const auto allocOnHost: {true, false}){
                    NVE_DEBUG_PRINTF_("running test on batch %d hotness %d num_elements %d allocateDataOn: %s\n", batch, hotness, num_elements, allocOnHost?"host":"device");
                    this->LaunchTest(262144, num_elements, batch, hotness, allocOnHost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(PoolingBackPropRefTest, TestPoolingBackpropAgainstRefCpu);

class PoolingBackpropRefTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    return EmbeddingCacheRefTestNames::GetName<T>("PoolingBackpropRefTest_", i);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(PoolingBackprop,    // TRTREC-44
                               PoolingBackPropRefTest,
                               EmbedTestBackpropTypes,
                               PoolingBackpropRefTestNames);


template <typename KeyT, bool gpu_histogram_flag>
struct HistogramTestType {
  typedef KeyT KeyType;
  static bool constexpr runOnGPU = gpu_histogram_flag;
};

template <typename T>
class HistogramTest : public ::testing::Test {
  public:
    using KeyType = typename T::KeyType;

    void launchTest(KeyType num_keys) {
        const size_t embedding_size = 512;

        std::mt19937 genr(0X753812);
        std::uniform_int_distribution<KeyType> dist_keys(1, num_keys * 200);

        std::set<KeyType> unique_keys;
        while (unique_keys.size() < static_cast<size_t>(num_keys)) {
            unique_keys.insert(dist_keys(genr));
        }

        auto keys = std::make_shared<Buffer<KeyType>>(num_keys * (num_keys + 1) * sizeof(KeyType) / 2);
        KeyType idx = 0;
        std::vector<KeyType> keys_(unique_keys.begin(), unique_keys.end());

        for (KeyType i = 0 ; i < num_keys; i++) {
            for (int32_t j = 0; j <= i; j++) {
                keys->ph[idx++] = keys_[j];
            }
        }
        auto total_num_keys = idx;

        std::map<KeyType, int32_t> histogram;
        for (KeyType i = 0 ; i < total_num_keys; i++) {
            KeyType key = keys->ph[i];
            if (histogram.find(key) != histogram.end()) {
                ++histogram[key];
            } else {
                histogram[key] = 1;
            }
        }

        std::vector<KeyType> unique_keys_h;
        std::vector<float> priority_h;
        std::vector<int8_t*> data_ptrs_h;
        KeyType* d_hist_storage = nullptr;
        KeyType num_unique_keys = 0;

        if (T::runOnGPU) {
            keys->HtoD(0);

            // run the test
            nve::DefaultGPUHistogram<KeyType> histogramGPU(total_num_keys);
            size_t histAllocSize = histogramGPU.getAllocSize();
            
            CHECK_CUDA_ERROR(cudaMalloc(&d_hist_storage, histAllocSize));

            histogramGPU.computeHistogram(reinterpret_cast<const KeyType*>(keys->pd), total_num_keys,
                                        reinterpret_cast<const int8_t*>(keys->ph),
                                        embedding_size, d_hist_storage, 0);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            num_unique_keys = static_cast<KeyType>(histogramGPU.getNumBins());

            unique_keys_h.resize(num_unique_keys);
            priority_h.resize(num_unique_keys);
            data_ptrs_h.resize(num_unique_keys);

            KeyType* unique_keys_d = histogramGPU.getKeys();
            float* priority_d = histogramGPU.getPriority();
            const int8_t* const* data_ptrs_d = histogramGPU.getData();

            cudaMemcpy(&unique_keys_h[0], unique_keys_d, num_unique_keys * sizeof(KeyType), cudaMemcpyDeviceToHost);
            cudaMemcpy(&priority_h[0], priority_d, num_unique_keys * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&data_ptrs_h[0], data_ptrs_d, num_unique_keys * sizeof(int8_t*), cudaMemcpyDeviceToHost);
        } else {
            nve::DefaultHistogram<KeyType> histogramCPU(
                reinterpret_cast<const KeyType*>(keys->ph),
                static_cast<size_t>(total_num_keys),
                reinterpret_cast<const int8_t*>(keys->ph),
                static_cast<size_t>(embedding_size),
                false);
            
            num_unique_keys = static_cast<KeyType>(histogramCPU.get_num_bins());
            unique_keys_h.resize(num_unique_keys);
            priority_h.resize(num_unique_keys);
            data_ptrs_h.resize(num_unique_keys);

            memcpy(&unique_keys_h[0], histogramCPU.get_keys(), num_unique_keys * sizeof(KeyType));
            memcpy(&priority_h[0], histogramCPU.get_priority(), num_unique_keys * sizeof(float));
            memcpy(&data_ptrs_h[0], histogramCPU.get_data(), num_unique_keys * sizeof(int8_t*));
        }

        // get ref histogram sorted by priority
        std::map<int32_t, KeyType> histogram_by_count;
        for (typename std::map<KeyType, int32_t>::iterator itr = histogram.begin(); itr != histogram.end(); itr++) {
            histogram_by_count[(*itr).second] = (*itr).first;
        }

        EXPECT_TRUE (histogram_by_count.size() == num_unique_keys);
        EXPECT_TRUE (num_keys == num_unique_keys);

        idx = num_unique_keys - 1;
        for (typename std::map<int32_t, KeyType>::iterator itr = histogram_by_count.begin(); itr != histogram_by_count.end(); itr++) {
            KeyType key = (*itr).second;
            float priority = float((*itr).first) / float(total_num_keys);
            EXPECT_TRUE (unique_keys_h[idx] == key);
            EXPECT_TRUE (priority_h[idx] == priority);
            for (KeyType i = 0; i < num_keys; i++) {
                if (keys->ph[i] == key) {
                    const int8_t* data_ptr_ref = reinterpret_cast<const int8_t*>(keys->ph) + i * embedding_size;
                    const int8_t* data_ptr = data_ptrs_h[idx];
                    EXPECT_TRUE (data_ptr == data_ptr_ref);
                    break;
                }
            }
            --idx;
        }

        if (T::runOnGPU) {
            CHECK_CUDA_ERROR(cudaFree(d_hist_storage));
        }
    }
};

typedef ::testing::Types<
                         HistogramTestType<int32_t, true>,
                         HistogramTestType<int64_t, true>,
                         HistogramTestType<int32_t, false>,
                         HistogramTestType<int64_t, false>> HistogramTestTypes;

class HistogramTestNames {
 public:
  template <typename T>
  static std::string GetName(int i) {
    std::string test_name = "HistogramTest_";
    if (std::is_same_v<typename T::KeyType, int32_t>) {
      test_name += std::string("Keys[int32]_");
    }
    if (std::is_same_v<typename T::KeyType, int64_t>) {
      test_name += std::string("Keys[int64]_");
    }
    if (T::runOnGPU) {
      test_name += std::string("GPU_");
    } else {
      test_name += std::string("CPU_");
    }
    test_name += ::testing::PrintToString(i);
    return test_name;
  }
};

TYPED_TEST_SUITE_P(HistogramTest);

TYPED_TEST_P(HistogramTest, HistogramTestSuite) {
    this->launchTest(17);
}

REGISTER_TYPED_TEST_SUITE_P(HistogramTest, HistogramTestSuite);

INSTANTIATE_TYPED_TEST_SUITE_P(Histogram,
                               HistogramTest,
                               HistogramTestTypes,
                               HistogramTestNames);
