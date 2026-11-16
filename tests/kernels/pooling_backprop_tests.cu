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

#include "kernel_test_common.cuh"

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
            this->num_keys_ = this->batch_ * this->hotness_;
        } else {
            // allocate and init offsets
            offsets_ = std::make_shared<TestBuffer<IndexType>>((this->batch_ + 1) * sizeof(IndexType));
            std::mt19937 gen(0X475381);
            std::uniform_int_distribution<uint32_t> dist_offset(1, 31);

            IndexType curr_offset = 0;
            for (uint32_t b = 0; b < this->batch_; b++) {
                offsets_->ph[b] = curr_offset;
                curr_offset += dist_offset(gen);
            }
            this->num_keys_ = curr_offset;
            offsets_->ph[this->batch_] = curr_offset;
            offsets_->HtoD(this->stream_);
        }

        // allocate and init weights
        weights_ = std::make_shared<TestBuffer<ElemType>>(this->num_keys_ * sizeof(ElemType));
        const bool is_weighted = is_weighted_pooling(T::POOLING_TYPE);
        if (is_weighted) {
            std::mt19937 genr(0X753812);
            std::uniform_real_distribution<float> dist_float(0.1f, 1.0f);
            std::bernoulli_distribution distrib(0.5);
            for (IndexType i=0 ; i < this->num_keys_; i++) {
                weights_->ph[i] = distrib(genr) ? ElemType(0.5f) : ElemType(0.25f);
            }
        } else {
            for (IndexType i=0 ; i < this->num_keys_; i++) {
                weights_->ph[i] = ElemType(1.0f);
            }
        }
        weights_->HtoD(this->stream_);

        uint32_t batch = this->batch_;
        if (T::POOLING_TYPE == PoolingType_t::Concatenate) {
            batch *= this->hotness_;
        }
        grads_in_ = std::make_shared<TestBuffer<ElemType>>(batch * this->num_elements_ * sizeof(ElemType));
        
        std::mt19937 genr(0X387512);
        std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
        std::uniform_int_distribution<uint32_t> dist_denom(3, 9);
        for (uint32_t b = 0; b < batch; b++) {
            for (uint32_t el = 0; el < this->num_elements_; el++) {
                float nom = float(dist_nom(genr));
                float denom = float(1 << dist_denom(genr));
                grads_in_->ph[b * this->num_elements_ + el] = nom / denom;
            }
        }
        grads_in_->HtoD(this->stream_);
    }

    void AllocateKeys() {
        this->keys_ = std::make_shared<TestBuffer<IndexType>>(this->num_keys_ * sizeof(IndexType));
        const float alpha = 1.05f;

        const size_t seed = 283982;
        auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(this->num_rows_),
                                                T::FIXED_HOTNESS_FLAG ? this->hotness_ : 32, seed);
        for (uint32_t b = 0; b < this->batch_; b++)
        {
            auto sample = sg->getCategoryIndices();
            IndexType start_pos = T::FIXED_HOTNESS_FLAG ? b * this->hotness_ : this->offsets_->ph[b];
            IndexType keys_to_copy = T::FIXED_HOTNESS_FLAG ? this->hotness_ :
                                                             (this->offsets_->ph[b + 1] - this->offsets_->ph[b]);
            std::copy(sample.begin(), sample.begin() + keys_to_copy, this->keys_->ph + start_pos);
        }
        this->keys_->HtoD(this->stream_);
    }

    void ComputeRefResults(const ElemType* /*table*/,
                           const IndexType* indices,    
                           const std::vector<IndexType>& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("compute pooling backprop reference results\n");

        ref_grads_.clear();
        for (uint32_t b = 0; b < this->batch_; b++) {
            IndexType hotness;
            IndexType start_idx;
            if (T::FIXED_HOTNESS_FLAG) {
                hotness = this->hotness_;
                start_idx = b * hotness;
            } else {
                hotness = offsets_->ph[b+1] - offsets_->ph[b];
                start_idx = offsets_->ph[b];
            }

            // Using constexpr if-else instead of switch to avoid Coverity false positive (PW.SWITCH_SELECTOR_EXPR_IS_CONSTANT)
            ElemType sum_weights = 1.0f;
            if constexpr (T::POOLING_TYPE == PoolingType_t::WeightedMean) {
                sum_weights = 0.f;
                for (IndexType h = 0; h < hotness; h++) {
                    sum_weights += ElemType(weights_->ph[start_idx + h]);
                }
            } else if constexpr (T::POOLING_TYPE == PoolingType_t::Mean) {
                sum_weights = ElemType(hotness);
            }

            for (IndexType h = 0; h < hotness; h++) {
                auto key = indices[start_idx + h];
                if (ref_grads_.count(key) == 0) {
                    ElemType* p = new ElemType[this->num_elements_];
                    memset(p, 0, this->num_elements_ * sizeof(ElemType));
                    ref_grads_[key] = p;
                }
                auto ref_grad = ref_grads_[key];
                ElemType* input_grad = grads_in_->ph + b * this->num_elements_;

                if (T::POOLING_TYPE == PoolingType_t::Concatenate) {
                    input_grad = grads_in_->ph + b * hotness * this->num_elements_ + h * this->num_elements_;
                }

                for (uint32_t el = 0; el < this->num_elements_; el++) {
                    ElemType el_cast = input_grad[el];
                    ElemType weight_cast = 1.0f;

                    if (is_weighted_pooling(T::POOLING_TYPE)) {
                        weight_cast = weights_->ph[start_idx + h];
                    }
                    ref_grad[el] += el_cast * (weight_cast / sum_weights);
                }
            }
        }
    }

    void LaunchKernel(const ElemType* /*table*/, 
                      const IndexType* indices,
                      CacheDataType& /*cache_data*/) {
        NVE_DEBUG_PRINTF_("launch pooling backprop kernel %lu keys\n", static_cast<uint64_t>(this->num_keys_));

        unique_keys_ = std::make_shared<TestBuffer<IndexType>>(this->num_keys_ * sizeof(IndexType));
        grads_out_ = std::make_shared<TestBuffer<ElemType>>(this->num_keys_ * this->num_elements_ * sizeof(ElemType));
    
        const int32_t MAX_RUN_SIZE = 32;
        split_grads_ = MAX_RUN_SIZE != -1;

        nve::GradientCalculator<IndexType, MAX_RUN_SIZE> pool_backprop(nve::GetDefaultAllocator());

        size_t tmp_mem_size_device, tmp_mem_size_host;
        size_t element_size = sizeof(ElemType);
        pool_backprop.get_alloc_requirements(this->num_keys_, element_size, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_device_mem;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_device_mem, tmp_mem_size_device));
        char* tmp_host_mem;
        CHECK_CUDA_ERROR(cudaMallocHost(&tmp_host_mem, tmp_mem_size_host));
        pool_backprop.set_and_init_buffers(this->num_keys_, element_size, tmp_device_mem, tmp_host_mem);

        PoolingType_t pooling_type = T::POOLING_TYPE;

        pool_backprop.template compute_gradients<ElemType, T::FIXED_HOTNESS_FLAG>(
            indices,
            T::FIXED_HOTNESS_FLAG ? nullptr : offsets_->pd,
            grads_in_->pd,
            // Unweighted pooling gets no weights, as from the Python bindings
            is_weighted_pooling(T::POOLING_TYPE) ? weights_->pd : nullptr,
            grads_out_->pd,
            unique_keys_->pd,
            this->batch_,
            this->num_keys_,
            this->hotness_,
            this->num_elements_,
            pooling_type,
            this->stream_);

        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream_));

        CHECK_CUDA_ERROR(cudaFree(tmp_device_mem));
        CHECK_CUDA_ERROR(cudaFreeHost(tmp_host_mem));
    }

    void CheckResult() {
        NVE_DEBUG_PRINTF_("check pooling backprop results\n");
        
        // check results
        unique_keys_->DtoH(this->stream_);
        grads_out_->DtoH(this->stream_);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream_));

        const float tolerance = (split_grads_ ||
                                (T::POOLING_TYPE == PoolingType_t::Mean) ||
                                (T::POOLING_TYPE == PoolingType_t::WeightedMean)) ? 1e-5f : 0;

        for (unsigned long i = 0; i < ref_grads_.size(); i++)
        {
            auto unq_key = unique_keys_->ph[i];
            ElemType* kernel_ptr = grads_out_->ph + i * this->num_elements_;
            ElemType* ref_ptr = ref_grads_[unq_key];
            EXPECT_TRUE(std::equal(ref_ptr, ref_ptr + this->num_elements_, kernel_ptr, Near(tolerance)));
        }
    }

    std::vector<ElemType> ref_result_;
    std::shared_ptr<TestBuffer<ElemType>> weights_ = nullptr;
    std::shared_ptr<TestBuffer<IndexType>> offsets_ = nullptr;
    std::shared_ptr<TestBuffer<ElemType>> grads_in_ = nullptr;
    std::shared_ptr<TestBuffer<ElemType>> grads_out_ = nullptr;
    std::shared_ptr<TestBuffer<IndexType>> unique_keys_ = nullptr;

    std::map<IndexType, ElemType*> ref_grads_;

    bool split_grads_ = false;
};

TYPED_TEST_SUITE_P(PoolingBackPropRefTest);

TYPED_TEST_P(PoolingBackPropRefTest, TestPoolingBackpropAgainstRefCpu) {
    // Removed "nice" hotness and num elements values in favour of multiple pooling modes
    // Try multiples of 32 in case of test failures
    for (const auto batch : {171}) {
        for (const auto hotness : {61}) {
            for (const auto num_elements : {53, 128, 510}) {
                for (const auto allocOnHost: {true, false}){
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
