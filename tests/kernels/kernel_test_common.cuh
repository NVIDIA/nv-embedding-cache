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

// Shared fixtures and helpers for the split GPU kernel unit tests
// (find_and_combine_lookup_tests.cu, quant_lookup_tests.cu, pooling_backprop_tests.cu,
//  find_and_dequant_tests.cu, misc_kernel_tests.cu).

#pragma once

#include "gtest/gtest.h"
#include <bit_ops.hpp>
#include <datagen.h>
#include <memory>
#include <algorithm>
#include <thread>
#include <numeric>
#include <cmath>
#include <type_traits>
#include "embedding_cache_combined.cuh"
#include "cuda_ops/scatter.cuh"
#include "cuda_ops/update_accumulate.cuh"
#include "cuda_ops/cuda_utils.cuh"
#include "cuda_ops/dedup_grads_kernel.cuh"
#include "cuda_ops/find_and_combine_kernel.cuh"
#include "cuda_ops/find_and_dequant.cuh"
#include "cuda_ops/find_and_pool.cuh"
#include "cuda_ops/gradient_calculator.cuh"
#include "cuda_ops/gather_keys_data_ptrs.cuh"
#include "cuda_ops/sanitize_keys.cuh"
#include "../common/test_buffer.h"
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

    EmbeddingCacheRefTest(): cache_allocator_(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
    }

    ~EmbeddingCacheRefTest() {
        CHECK_CUDA_ERROR(cache_ptr_->lookup_context_destroy(handle_lookup_));
        CHECK_CUDA_ERROR(cache_ptr_->modify_context_destroy(handle_modify_));
    }

    void LaunchTest(uint64_t num_rows, uint32_t num_elements, uint32_t batch, uint32_t hotness, bool allocDataOnHost, int64_t kernel_param = 0) {
        num_rows_ = num_rows;
        num_elements_ = num_elements;
        batch_ = batch;
        hotness_ = hotness;
        num_keys_ = static_cast<IndexType>(batch_) * static_cast<IndexType>(hotness_);
        allocDataOnHost_ = allocDataOnHost;
        kernel_param_ = kernel_param;
        AllocateTable();
        InitCache();
        InitTestExtras();
        AllocateKeys();
        CacheDataType cache_data = cache_ptr_->get_cache_data(handle_lookup_);
        std::vector<IndexType> cached_indices = 
            ComputeCachedIndices(cache_data.num_sets, CacheType::NUM_WAYS);
        PopulateCache(cached_indices);
        ComputeRefResults(table_->ph, keys_->ph, cached_indices);
        LaunchKernel(table_->pd, keys_->pd, cache_data);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CheckResult();
    }

    cudaStream_t stream_;
    uint64_t num_rows_{0};
    uint32_t num_elements_{0};
    uint32_t batch_{0};
    uint32_t hotness_{0};
    IndexType num_keys_{0};
    bool allocDataOnHost_{false};
    int64_t kernel_param_{0};
  protected:
  
    ElemType* syncTable() {
      table_->DtoH(stream_);
      return table_->ph;
    }

    void GetCacheContent(ElemType* d_cache_content, std::shared_ptr<TestBuffer<IndexType>> keys_in_c, size_t& num_keys) 
    {
        
        cache_ptr_->get_keys_stored_in_cache(handle_lookup_, keys_in_c->ph, cache_ptr_->get_max_num_embedding_vectors_in_cache(), num_keys, stream_);
        keys_in_c->HtoD(stream_);
        int64_t num_hitmask_elems = nve::ceil_div(num_keys, static_cast<size_t>(32));
        auto hitmask = std::make_shared<TestBuffer<uint32_t>>(num_hitmask_elems * sizeof(uint32_t));
        CHECK_CUDA_ERROR(cudaMemset(hitmask->pd, 0, num_hitmask_elems * sizeof(uint32_t)));

        NVE_DEBUG_PRINTF_("launch lookup with hitmask kernel\n");

        cache_ptr_->lookup(handle_lookup_, keys_in_c->pd, num_keys, (int8_t*)d_cache_content, (uint64_t*)hitmask->pd, 0, num_elements_*sizeof(ElemType), stream_);
    }

    std::map<IndexType, std::vector<ElemType>> GetCacheContent()
    {
        auto num_vec_in_c = cache_ptr_->get_max_num_embedding_vectors_in_cache();
        auto key_in_c = std::make_shared<TestBuffer<IndexType>>(sizeof(IndexType)*num_vec_in_c);
        auto cache_content = std::make_shared<TestBuffer<ElemType>>(sizeof(ElemType)*this->num_elements_*num_vec_in_c);
        size_t num_keys_in_c = 0;
        this->GetCacheContent(cache_content->pd, key_in_c, num_keys_in_c);
        cache_content->DtoH(this->stream_);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream_));
        std::map<IndexType, std::vector<ElemType>> mock_cache;
        for (size_t i = 0; i < num_keys_in_c; i++)
        {
            auto key = key_in_c->ph[i];
            mock_cache[key].resize(this->num_elements_);
            memcpy(mock_cache[key].data(), cache_content->ph + i*this->num_elements_, this->num_elements_* sizeof(ElemType));
        }
        return mock_cache;
    }

    std::shared_ptr<TestBuffer<IndexType>> keys_ = nullptr;

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
        const uint32_t num_rows_in_cache = static_cast<uint32_t>(cache_ratio * static_cast<float>(this->num_rows_));

        typename CacheType::CacheConfig cfg;
        cfg.embed_width_in_bytes = this->num_elements_ * sizeof(ElemType);

        NVE_DEBUG_PRINTF_("cache of %d entries\n", int(cache_ratio * float(this->num_rows_)));
        cfg.cache_sz_in_bytes = num_rows_in_cache * cfg.embed_width_in_bytes;
        cfg.num_tables = 1;
        cfg.allocate_data_on_host = allocDataOnHost_;

        cache_ptr_ = std::make_shared<CacheType>(&cache_allocator_, &cache_logger_, cfg);
        cache_ptr_->init();

        cache_ptr_->lookup_context_create(handle_lookup_, nullptr, 0);
    }

    void AllocateTable() {
        table_ = std::make_shared<TestBuffer<ElemType>>(this->num_rows_ * this->num_elements_ * sizeof(ElemType));
        std::mt19937 gen(0X814753);
        // init to [a / 2^b] when a and b are in specific range,
        // to minimize arithmetic error in half when doing weigthed math
        // inputs in the range of [-1.0f, 1.0f] accumulate larger errors
        // (tolerance of 1e-3f is required to pass the tests)
        std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
        std::uniform_int_distribution<uint32_t> dist_denom(3, 9);

        for (uint64_t i = 0; i < this->num_rows_; i++)
        {
            ElemType* curr_row = reinterpret_cast<ElemType*>(table_->ph + i * this->num_elements_);
            for (uint64_t j = 0; j < this->num_elements_; j++)
            {
                float nom = float(dist_nom(gen));
                float denom = float(1 << dist_denom(gen));
                curr_row[j] = nom / denom;
            }
        }

        table_->HtoD(stream_);
    }

    virtual void AllocateKeys() {
        keys_ = std::make_shared<TestBuffer<IndexType>>(num_keys_ * sizeof(IndexType));
        const float alpha = 1.05f;

        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->num_rows_), this->hotness_, 283982);
        for (uint32_t b = 0; b < this->batch_; b++)
        {
            auto sample = sg->getCategoryIndices();
            std::copy(sample.begin(), sample.end(), keys_->ph + b*this->hotness_);
        }
        keys_->HtoD(stream_);
    }

    std::vector<IndexType> ComputeCachedIndices(const int num_sets, const int num_ways) {
        std::set<IndexType> cached_indices;
        std::vector<int> counters(num_sets, 0);

        uint64_t cache_capacity = static_cast<uint64_t>(static_cast<double>(this->num_rows_) * 0.15);

        while (cache_capacity > static_cast<uint64_t>(this->num_keys_)) {
            cache_capacity /= 2;
        }

        uint32_t step = static_cast<uint32_t>(this->num_keys_ / cache_capacity);
        EXPECT_TRUE(step > 0); // batch_size * hotness > cache_capcitiy 
        // for now: random indices
        for (IndexType i=0; i < this->num_keys_; i+=step) {
            IndexType idx = keys_->ph[i];
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
        cache_ptr_->modify_context_create(handle_modify_, static_cast<uint32_t>(cached_indices.size()));

        nve::DefaultHistogram<IndexType> hist(cached_indices.data(), cached_indices.size(),
                                              reinterpret_cast<const int8_t*>(table_->ph),
                                              this->num_elements_ * sizeof(ElemType), true);
        cudaEvent_t wait;
        CHECK_CUDA_ERROR(cudaEventCreate(&wait));
        nve::DefaultECEvent ec_event(std::vector<cudaStream_t>{});

        cache_ptr_->insert(
            handle_modify_, hist.get_keys(), hist.get_priority(), hist.get_data(), hist.get_num_bins(), 0, &ec_event, stream_);

        CHECK_CUDA_ERROR(cudaEventRecord(wait));
        CHECK_CUDA_ERROR(cudaEventSynchronize(wait));

    }

    std::shared_ptr<TestBuffer<ElemType>> table_ = nullptr;

    nve::DefaultAllocator cache_allocator_;
    nve::Logger cache_logger_;
    nve::PerformanceMetric miss_count_;
    std::shared_ptr<CacheType> cache_ptr_;
    nve::LookupContextHandle handle_lookup_;
    nve::ModifyContextHandle handle_modify_;
};


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
