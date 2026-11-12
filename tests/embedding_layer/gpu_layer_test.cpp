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

#include <gtest/gtest.h>

#include <buffer_wrapper.hpp>
#include <gpu_embedding_layer.hpp>
#include <key_utils.hpp>
#include "cuda_ptr.hpp"
#include "emb_layer_utils.hpp"
#include <thread>
#include <cuda_support.hpp>
#include "mock_host_table.hpp"

namespace nve {

// Fixture for functional tests
struct GPUTestCase {
  int64_t gpu_table_size;
  int64_t row_size;
  int64_t test_keys;
  DataType_t data_type;
  bool    do_pooling = false;
  int64_t hotness = 1;
  PoolingType_t pooling_type = PoolingType_t::Concatenate;
  SparseType_t  offsets_layout = SparseType_t::Fixed;
};

class GPU : public ::testing::TestWithParam<GPUTestCase> {};
class GPU_UP_ACC : public ::testing::TestWithParam<GPUTestCase> {};

template <typename IndexT>
class GPULayerLookupTest {
 public:
  using layer_type = GPUEmbeddingLayer<IndexT>;
  static constexpr int DEVICE_ID = 0;

  GPULayerLookupTest(int64_t row_size, int64_t gpu_table_size, DataType_t data_type, size_t seed = 31337)
    : row_size_(row_size), rows_(gpu_table_size / row_size) {
    // init linear table
    {
      NVE_CHECK_(cudaMalloc(&gpu_table_, static_cast<size_t>(row_size_ * rows_)));
      NVE_CHECK_(cudaMallocHost(&host_table_, static_cast<size_t>(row_size_ * rows_)));
      NVE_CHECK_(row_size_ % dtype_size(data_type) == 0); // cannot use ASSERT_EQ in c'tor

      // Init table values with multiple threads to save on test time
      std::vector<std::shared_ptr<std::thread>> input_gen_threads;

      constexpr int64_t num_threads = 32;
      for (int64_t t=0 ; t<num_threads ; t++) {
        auto start_row = t * rows_ / num_threads;
        auto end_row = std::min<int64_t>((t + 1) * rows_ / num_threads, rows_);

        input_gen_threads.push_back(std::make_shared<std::thread>(
          InitTableRows, host_table_, row_size_, start_row, end_row, data_type, seed + static_cast<size_t>(t)
        ));
      }
      for (auto& t : input_gen_threads) {
          t->join();
      }

      HostTableConfig mock_cfg;
      mock_cfg.value_dtype = data_type;
      mock_cfg.max_value_size = static_cast<int64_t>(row_size_);
      ref_tab_ = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/, host_table_);
    }

    // init layer
    {
      NVE_CHECK_(cudaMemcpy(gpu_table_, host_table_, static_cast<size_t>(row_size_ * rows_), cudaMemcpyHostToDevice));
      GPUEmbeddingLayerConfig embedding_table_cfg;
      embedding_table_cfg.device_id = DEVICE_ID;
      embedding_table_cfg.num_embeddings = static_cast<int64_t>(rows_);
      embedding_table_cfg.embedding_width_in_bytes = static_cast<int64_t>(row_size);
      embedding_table_cfg.embedding_table = gpu_table_;
      embedding_table_cfg.value_dtype = data_type;
      layer_ = std::make_shared<GPUEmbeddingLayer<IndexT>>(embedding_table_cfg);
    }
    // init context
    ctx_ = layer_->create_execution_context(0, 0, nullptr, nullptr);
  }

  GPULayerLookupTest(GPUTestCase tc) : GPULayerLookupTest<IndexT>(tc.row_size, tc.gpu_table_size, tc.data_type) {}

  ~GPULayerLookupTest() {
    NVE_CHECK_(cudaFree(gpu_table_));
    NVE_CHECK_(cudaFreeHost(host_table_));
  }

  void LookupAndCheck(const GPUTestCase& tc, std::vector<IndexT>& keys, uint64_t start_key = 0,
                      uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto output_bags = num_keys;

    uint64_t output_size = static_cast<size_t>(num_keys * row_size_);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);

    std::vector<float> hitrates(3);

    if (tc.do_pooling) {
      EmbeddingLayerBase::PoolingParams pp;
      pp.pooling_type = tc.pooling_type;
      pp.sparse_type = tc.offsets_layout;
      pp.output_type = tc.data_type;

      SetupCSROffsets<IndexT> offsets_setup(
          tc.offsets_layout == SparseType_t::CSR ? static_cast<IndexT>(num_keys) : IndexT{1},
          static_cast<IndexT>(std::max<int64_t>(tc.hotness, 1)));
      const IndexT fixed_hotness =
          (num_keys == 1) ? IndexT{1} : static_cast<IndexT>(tc.hotness);
      if (tc.offsets_layout == SparseType_t::CSR) {
        pp.csr_offsets = offsets_setup.offsets_buffer;
        pp.num_csr_offsets = static_cast<int64_t>(offsets_setup.num_offsets);
        output_bags = static_cast<int64_t>(offsets_setup.num_offsets) - 1;
      } else {
        pp.fixed_hotness = fixed_hotness;
        if (tc.pooling_type != PoolingType_t::Concatenate) {
          output_bags = num_keys / fixed_hotness;
        }
      }

      std::vector<int8_t> weights;
      if (is_weighted_pooling(tc.pooling_type)) {
        GenerateWeights(weights, static_cast<uint64_t>(num_keys), tc.data_type);
        pp.weights = weights.data();
        // for now, weight type (pp.weight_type) is assumed to be data type
        pp.weight_type = tc.data_type;
      } else {
        pp.weights = nullptr;
      }

      layer_->lookup(ctx_, num_keys, keys_buffer, output, row_size_, nullptr /*hitmask*/,
                     &pp /*pool_params*/, hitrates.data());

      std::vector<int8_t> find_output(static_cast<size_t>(num_keys * row_size_));
      {
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
        auto values_bw = std::make_shared<BufferWrapper<void>>(
            ctx_, "values", find_output.data(),
            static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
        ref_tab_->find(ctx_, num_keys, std::move(keys_bw), nullptr /*hitmask*/, row_size_,
                           std::move(values_bw), nullptr /*value_sizes*/);
      }

      ref_output_.resize(static_cast<size_t>(output_bags * row_size_));
      ref_tab_->combine(find_output.data(), num_keys, pp.pooling_type, pp.sparse_type,
                         pp.csr_offsets, pp.num_csr_offsets, pp.fixed_hotness, pp.weights,
                         pp.weight_type, ref_output_.data());
    } else {
      layer_->lookup(ctx_, num_keys, keys_buffer, output, row_size_, nullptr /*hitmask*/,
                      nullptr, hitrates.data());

      ref_output_.resize(static_cast<size_t>(num_keys * row_size_));
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto values_bw = std::make_shared<BufferWrapper<void>>(
          ctx_, "values", ref_output_.data(),
          static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
      ref_tab_->find(ctx_, num_keys, std::move(keys_bw), nullptr /*hitmask*/, row_size_,
                         std::move(values_bw), nullptr /*value_sizes*/);
    }
    
    NVE_CHECK_(cudaDeviceSynchronize());

    // compare hitrates
    ASSERT_EQ(hitrates[0], 1.0f);

    const int64_t output_elements = static_cast<int64_t>(ref_output_.size()) / dtype_size(tc.data_type);

    float tolerance = 0.f;
    if (tc.do_pooling && tc.pooling_type != PoolingType_t::Concatenate) {
      switch (tc.data_type) {
        case DataType_t::Float32:
          tolerance = 1e-5f;
          break;
        case DataType_t::Float16:
          tolerance = 1e-3f;
          break;
        default:
          throw std::runtime_error("Invalid datatype");
      }
    }

    for (int64_t i=0 ; i<output_elements ; i++) {
      ASSERT_NEAR(
        load_as_float(output, i, tc.data_type),
        load_as_float(ref_output_.data(), i, tc.data_type),
        tolerance
      );

    }
    NVE_CHECK_(cudaFreeHost(output));
  }

  void ExpectPoolingTypesRejected(DataType_t output_type, DataType_t weight_type,
                                  bool use_weights) {
    const IndexT key = 0;
    const IndexT hotness = 1;
    std::vector<int8_t> output(static_cast<size_t>(row_size_));
    std::vector<int8_t> weights;

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = use_weights ? PoolingType_t::WeightedSum : PoolingType_t::Sum;
    pp.sparse_type = SparseType_t::Fixed;
    pp.fixed_hotness = hotness;
    pp.output_type = output_type;
    if (use_weights) {
      weights.resize(static_cast<size_t>(dtype_size(weight_type)));
      pp.weights = weights.data();
      pp.weight_type = weight_type;
    }

    EXPECT_THROW(layer_->lookup(ctx_, 1, &key, output.data(), row_size_,
                                 /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr),
                 Exception);
  }

  void Update(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
              uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }

    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    // 1. lookup these keys to ref
    auto output_size = static_cast<size_t>(num_keys * row_size_);
    int8_t* lookup_output{nullptr};
    NVE_CHECK_(cudaMallocHost(&lookup_output, output_size));
    NVE_CHECK_(lookup_output != 0);
    std::vector<uint8_t> res_lookup_output(output_size);
    std::vector<float> hitrates(3);

    // 3. call accumulate 
    layer_->update(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer, -1);

    NVE_CHECK_(cudaDeviceSynchronize());

    // 4. call lookup
    layer_->lookup(ctx_, num_keys, keys_buffer, lookup_output, row_size_, nullptr /*hitmask*/,
                    nullptr, hitrates.data());

    NVE_CHECK_(cudaDeviceSynchronize());

    NVE_CHECK_(cudaMemcpy(&res_lookup_output[0], lookup_output, output_size, cudaMemcpyDeviceToHost));
    NVE_CHECK_(cudaDeviceSynchronize());

    // 5. compare to ref
    for (int64_t i = 0; i < num_keys; i++) {
      for (int64_t j = 0; j < row_size_; j++) {
        ASSERT_EQ(res_lookup_output[static_cast<size_t>((i * row_size_) + j)], data_buffer[static_cast<size_t>((i * row_size_) + j)]);
      }
    }

    NVE_CHECK_(cudaFreeHost(lookup_output));
  }

  void Accumulate(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
                  DataType_t value_type, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
  
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    // 1. lookup these keys to ref
    auto output_size = static_cast<size_t>(num_keys * row_size_);
    int8_t* lookup_output{nullptr};
    NVE_CHECK_(cudaMallocHost(&lookup_output, output_size));
    NVE_CHECK_(lookup_output != 0);
    std::vector<int8_t> ref_lookup_output(static_cast<size_t>(num_keys * row_size_));
    std::vector<int8_t> res_lookup_output(static_cast<size_t>(num_keys * row_size_));
    std::vector<float> hitrates(3);

    layer_->lookup(ctx_, num_keys, keys_buffer, lookup_output, row_size_, nullptr /*hitmask*/,
                    nullptr, hitrates.data());

    NVE_CHECK_(cudaDeviceSynchronize());

    NVE_CHECK_(cudaMemcpy(&ref_lookup_output[0], lookup_output, output_size, cudaMemcpyDeviceToHost));
    NVE_CHECK_(cudaDeviceSynchronize());

    // 2. add datavectors
    switch (value_type)
    {
      case DataType_t::Float32:
        {
          float* ref = reinterpret_cast<float*>(ref_lookup_output.data());
          float* acc = reinterpret_cast<float*>(setup.data_buffer);
          const auto num_elements = ref_lookup_output.size() / sizeof(float);
          for (uint64_t i = 0; i < num_elements; i++) {
            ref[i] += acc[i];
          }
        }
        break;
      case DataType_t::Float16:
        {
          half* ref = reinterpret_cast<half*>(ref_lookup_output.data());
          half* acc = reinterpret_cast<half*>(setup.data_buffer);
          const auto num_elements = ref_lookup_output.size() / sizeof(half);
          for (uint64_t i = 0; i < num_elements; i++) {
            ref[i] = __float2half(__half2float(ref[i]) + __half2float(acc[i]));
          }
        }
        break;
      default:
        throw std::runtime_error("Not implemented!");
    }

    // 3. call accumulate 
    layer_->accumulate(ctx_, num_keys, keys_buffer, row_size_, row_size_, setup.data_buffer, value_type, -1);

    NVE_CHECK_(cudaDeviceSynchronize());

    // 4. call lookup
    layer_->lookup(ctx_, num_keys, keys_buffer, lookup_output, row_size_, nullptr /*hitmask*/,
                    nullptr, hitrates.data());

    NVE_CHECK_(cudaDeviceSynchronize());

    NVE_CHECK_(cudaMemcpy(&res_lookup_output[0], lookup_output, output_size, cudaMemcpyDeviceToHost));
    NVE_CHECK_(cudaDeviceSynchronize());

    // 5. compare to ref (bitwise comparison for now)
    for (uint64_t i = 0; i < res_lookup_output.size(); i++) {
      ASSERT_EQ(res_lookup_output[i], ref_lookup_output[i]);
    }

    NVE_CHECK_(cudaFreeHost(lookup_output));
  }

 public:
  const int64_t row_size_;
  const int64_t rows_;
 private:
  std::vector<int8_t> ref_output_;
  int8_t* gpu_table_{nullptr};
  int8_t* host_table_{nullptr};
  std::shared_ptr<layer_type> layer_{nullptr};
  context_ptr_t ctx_;
  std::shared_ptr<MockHostTable<IndexT>> ref_tab_{nullptr};
};

TEST(GPUValidation, RejectsUnsupportedTableValueType) {
  GPUEmbeddingLayerConfig config;
  config.embedding_table = &config;  // Only required to be non-null before dtype validation.
  config.num_embeddings = 1;
  config.embedding_width_in_bytes = sizeof(double);
  config.value_dtype = DataType_t::Float64;

  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
}

TEST(GPUValidation, RejectsNullEmbeddingTable) {
  GPUEmbeddingLayerConfig config;
  config.embedding_table = nullptr;
  config.num_embeddings = 1;
  config.embedding_width_in_bytes = sizeof(float);
  config.value_dtype = DataType_t::Float32;

  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
}

TEST(GPUValidation, RejectsZeroNumEmbeddings) {
  GPUEmbeddingLayerConfig config;
  config.embedding_table = &config;  // Only required to be non-null before the row count check.
  config.num_embeddings = 0;
  config.embedding_width_in_bytes = sizeof(float);
  config.value_dtype = DataType_t::Float32;

  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
}

// update/accumulate must ignore keys outside [0, num_embeddings) rather than write past the table.
// The device buffer holds `guard_rows` extra rows past num_embeddings: out of range keys are aimed
// at them, so an unchecked write shows up as a modified guard row.
static void RunGpuLayerOutOfRangeKeysTest(bool accumulate) {
  using IndexT = int64_t;
  constexpr int64_t num_rows = 64;
  constexpr int64_t guard_rows = 8;
  constexpr int64_t row_size = 128;  // bytes == 32 floats
  constexpr size_t row_floats = static_cast<size_t>(row_size) / sizeof(float);
  constexpr size_t total_floats = static_cast<size_t>(num_rows + guard_rows) * row_floats;

  std::vector<float> ref_table(total_floats);
  for (size_t i = 0; i < total_floats; ++i) {
    ref_table[i] = static_cast<float>(i);
  }
  float* raw_table = nullptr;
  NVE_CHECK_(cudaMalloc(&raw_table, total_floats * sizeof(float)));
  DevicePtr<float> d_table(raw_table);
  NVE_CHECK_(cudaMemcpy(d_table.get(), ref_table.data(), total_floats * sizeof(float), cudaMemcpyHostToDevice));

  GPUEmbeddingLayerConfig cfg;
  cfg.device_id = 0;
  cfg.num_embeddings = num_rows;
  cfg.embedding_width_in_bytes = row_size;
  cfg.embedding_table = d_table.get();
  cfg.value_dtype = DataType_t::Float32;
  auto layer = std::make_shared<GPUEmbeddingLayer<IndexT>>(cfg);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  // Every 7th key is out of range: negative, or aimed at a guard row past the end of the table.
  constexpr int64_t num_keys = num_rows;
  std::vector<IndexT> keys(static_cast<size_t>(num_keys));
  for (int64_t i = 0; i < num_keys; ++i) {
    if ((i % 7) == 1) {
      keys[static_cast<size_t>(i)] = -(i + 1);
    } else if ((i % 7) == 3) {
      keys[static_cast<size_t>(i)] = num_rows + (i % guard_rows);
    } else {
      keys[static_cast<size_t>(i)] = i;
    }
  }

  std::vector<float> updates(static_cast<size_t>(num_keys) * row_floats);
  for (int64_t i = 0; i < num_keys; ++i) {
    for (size_t j = 0; j < row_floats; ++j) {
      updates[static_cast<size_t>(i) * row_floats + j] = static_cast<float>(1000 + i) + static_cast<float>(j);
    }
  }

  // Reference: apply the in-range keys only
  for (int64_t i = 0; i < num_keys; ++i) {
    const IndexT key = keys[static_cast<size_t>(i)];
    if (key < 0 || key >= num_rows) {
      continue;
    }
    for (size_t j = 0; j < row_floats; ++j) {
      float& dst = ref_table[static_cast<size_t>(key) * row_floats + j];
      const float src = updates[static_cast<size_t>(i) * row_floats + j];
      dst = accumulate ? (dst + src) : src;
    }
  }

  if (accumulate) {
    layer->accumulate(ctx, num_keys, keys.data(), row_size, row_size, updates.data(),
                      DataType_t::Float32, -1 /*table_id*/);
  } else {
    layer->update(ctx, num_keys, keys.data(), row_size, row_size, updates.data(), -1 /*table_id*/);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<float> result(total_floats);
  NVE_CHECK_(cudaMemcpy(result.data(), d_table.get(), total_floats * sizeof(float), cudaMemcpyDeviceToHost));
  for (int64_t r = 0; r < num_rows + guard_rows; ++r) {
    for (size_t j = 0; j < row_floats; ++j) {
      const size_t idx = static_cast<size_t>(r) * row_floats + j;
      ASSERT_FLOAT_EQ(ref_table[idx], result[idx])
          << (r >= num_rows ? "guard row " : "row ") << r << " float " << j;
    }
  }
}

// Lookups of keys outside [0, num_embeddings) must resolve to the configured default row instead of
// reading past the table. The device buffer holds `guard_rows` extra rows past num_embeddings with
// values no valid row can produce, so out of range keys aimed at them are visible in the output.
static void RunGpuLayerDefaultRowLookupTest(bool pooled) {
  using IndexT = int64_t;
  constexpr int64_t num_rows = 64;
  constexpr int64_t guard_rows = 8;
  constexpr int64_t row_size = 128;  // bytes == 32 floats
  constexpr int64_t default_row_index = 7;
  constexpr int64_t hotness = 2;
  constexpr size_t row_floats = static_cast<size_t>(row_size) / sizeof(float);
  constexpr size_t total_floats = static_cast<size_t>(num_rows + guard_rows) * row_floats;

  std::vector<float> host_table(total_floats);
  for (int64_t r = 0; r < num_rows + guard_rows; ++r) {
    for (size_t j = 0; j < row_floats; ++j) {
      const float value = static_cast<float>(r * 100) + static_cast<float>(j);
      host_table[static_cast<size_t>(r) * row_floats + j] = (r < num_rows) ? value : -value;
    }
  }
  float* raw_table = nullptr;
  NVE_CHECK_(cudaMalloc(&raw_table, total_floats * sizeof(float)));
  DevicePtr<float> d_table(raw_table);
  NVE_CHECK_(cudaMemcpy(d_table.get(), host_table.data(), total_floats * sizeof(float), cudaMemcpyHostToDevice));

  GPUEmbeddingLayerConfig cfg;
  cfg.device_id = 0;
  cfg.num_embeddings = num_rows;
  cfg.embedding_width_in_bytes = row_size;
  cfg.embedding_table = d_table.get();
  cfg.value_dtype = DataType_t::Float32;
  cfg.default_row_index = default_row_index;
  auto layer = std::make_shared<GPUEmbeddingLayer<IndexT>>(cfg);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  // Every 3rd key is out of range: negative, or aimed at a guard row past the end of the table.
  constexpr int64_t num_keys = 32;
  std::vector<IndexT> keys(static_cast<size_t>(num_keys));
  std::vector<IndexT> resolved(static_cast<size_t>(num_keys));  // row each key must read
  for (int64_t i = 0; i < num_keys; ++i) {
    const auto idx = static_cast<size_t>(i);
    if ((i % 3) == 1) {
      keys[idx] = -(i + 1);
    } else if ((i % 3) == 2) {
      keys[idx] = num_rows + (i % guard_rows);
    } else {
      keys[idx] = i;
    }
    resolved[idx] = key_in_range(keys[idx], num_rows) ? keys[idx] : default_row_index;
  }

  const int64_t output_rows = pooled ? (num_keys / hotness) : num_keys;
  std::vector<float> output(static_cast<size_t>(output_rows) * row_floats, 0.f);

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::Fixed;
  pp.fixed_hotness = hotness;
  pp.output_type = DataType_t::Float32;
  layer->lookup(ctx, num_keys, keys.data(), output.data(), row_size, nullptr /*hitmask*/,
                pooled ? &pp : nullptr, nullptr /*hitrates*/);
  ctx->wait();

  // Reference: out of range keys contribute the default row, in a bag like anywhere else
  for (int64_t out = 0; out < output_rows; ++out) {
    for (size_t j = 0; j < row_floats; ++j) {
      float expected = 0.f;
      if (pooled) {
        for (int64_t k = 0; k < hotness; ++k) {
          expected += host_table[static_cast<size_t>(resolved[static_cast<size_t>(out * hotness + k)]) * row_floats + j];
        }
      } else {
        expected = host_table[static_cast<size_t>(resolved[static_cast<size_t>(out)]) * row_floats + j];
      }
      ASSERT_FLOAT_EQ(expected, output[static_cast<size_t>(out) * row_floats + j])
          << "output row " << out << " float " << j;
    }
  }
}

TEST(GPUValidation, RejectsDefaultRowOutsideTable) {
  GPUEmbeddingLayerConfig config;
  config.embedding_table = &config;  // Only required to be non-null before the default row check.
  config.num_embeddings = 16;
  config.embedding_width_in_bytes = sizeof(float);
  config.value_dtype = DataType_t::Float32;
  config.default_row_index = config.num_embeddings;

  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
}

TEST(GPUValidation, RejectsNumEmbeddingsOutsideKeyType) {
  GPUEmbeddingLayerConfig config;
  config.embedding_table = &config;  // Only required to be non-null before the row count check.
  config.num_embeddings = int64_t{1} << 31;
  config.embedding_width_in_bytes = sizeof(float);
  config.value_dtype = DataType_t::Float32;

  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int32_t>>(config), Exception);
}

TEST(GPUValidation, RejectsUnsupportedPoolingTypeConversions) {
  int num_devices = 0;
  if (cudaGetDeviceCount(&num_devices) != cudaSuccess || num_devices == 0) {
    cudaGetLastError();  // Clear the error reported while probing for a CUDA device.
    GTEST_SKIP() << "No CUDA device available";
  }

  constexpr int64_t row_size = 16;
  constexpr int64_t table_size = row_size * 4;
  GPULayerLookupTest<int64_t> elt(row_size, table_size, DataType_t::Float32);

  elt.ExpectPoolingTypesRejected(DataType_t::Float16, DataType_t::Unknown,
                                 /*use_weights=*/false);
  elt.ExpectPoolingTypesRejected(DataType_t::Float32, DataType_t::Float16,
                                 /*use_weights=*/true);
}

namespace {

bool gpu_test_device_available() {
  int num_devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&num_devices);
  if (status != cudaSuccess) cudaGetLastError();
  return status == cudaSuccess && num_devices > 0;
}

TEST(GPUOutOfRangeKeys, UpdateIgnoresOutOfRangeKeys) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  RunGpuLayerOutOfRangeKeysTest(false /*accumulate*/);
}

TEST(GPUOutOfRangeKeys, AccumulateIgnoresOutOfRangeKeys) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  RunGpuLayerOutOfRangeKeysTest(true /*accumulate*/);
}

TEST(GPUOutOfRangeKeys, LookupReturnsDefaultRow) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  RunGpuLayerDefaultRowLookupTest(false /*pooled*/);
}

TEST(GPUOutOfRangeKeys, PooledLookupPoolsDefaultRow) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  RunGpuLayerDefaultRowLookupTest(true /*pooled*/);
}

template <typename IndexT>
class QuantGpuLayerTest {
 public:
  QuantGpuLayerTest(DataType_t dtype, int64_t value_count = 64, int64_t default_row_index = -1)
      : dtype_(dtype), value_count_(value_count),
        row_bytes_(row_size_in_bytes(dtype, value_count)),
        output_dtype_(quant_rowwise_output_dtype(dtype)),
        output_stride_(value_count * dtype_size(output_dtype_)),
        host_table_(static_cast<size_t>(kNumRows * row_bytes_)) {
    InitTableRowsQuant(host_table_.data(), row_bytes_, value_count_, 0, kNumRows, dtype_, 9417);
    NVE_CHECK_(cudaMalloc(&device_table_, host_table_.size()));
    NVE_CHECK_(cudaMemcpy(device_table_, host_table_.data(), host_table_.size(),
                          cudaMemcpyHostToDevice));

    GPUEmbeddingLayerConfig config;
    config.device_id = 0;
    config.embedding_table = device_table_;
    config.num_embeddings = kNumRows;
    config.embedding_width_in_bytes = row_bytes_;
    config.value_dtype = dtype_;
    config.default_row_index = default_row_index;
    layer_ = std::make_shared<GPUEmbeddingLayer<IndexT>>(config);
    ctx_ = layer_->create_execution_context(0, 0, nullptr, nullptr);
  }

  ~QuantGpuLayerTest() {
    if (ctx_) ctx_->wait();
    ctx_.reset();
    layer_.reset();
    if (device_table_) cudaFree(device_table_);
  }

  std::vector<IndexT> keys() const {
    std::vector<IndexT> result(16);
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = static_cast<IndexT>(i + 1);
    }
    return result;
  }

  void CheckRawLookup(bool device_output, bool concatenate = false) {
    auto lookup_keys = keys();
    std::vector<int8_t> result(lookup_keys.size() * static_cast<size_t>(row_bytes_));
    void* output = result.data();
    DevicePtr<void> device_result;
    if (device_output) {
      void* ptr = nullptr;
      NVE_CHECK_(cudaMalloc(&ptr, result.size()));
      device_result.reset(ptr);
      output = ptr;
    }

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = PoolingType_t::Concatenate;
    pp.output_type = dtype_;
    layer_->lookup(ctx_, static_cast<int64_t>(lookup_keys.size()), lookup_keys.data(), output,
                   row_bytes_, nullptr, concatenate ? &pp : nullptr, nullptr);
    ctx_->wait();
    if (device_output) {
      NVE_CHECK_(cudaMemcpy(result.data(), output, result.size(), cudaMemcpyDeviceToHost));
    }

    for (size_t i = 0; i < lookup_keys.size(); ++i) {
      const auto* expected = host_table_.data() + static_cast<int64_t>(lookup_keys[i]) * row_bytes_;
      EXPECT_EQ(std::memcmp(result.data() + static_cast<int64_t>(i) * row_bytes_, expected,
                            static_cast<size_t>(row_bytes_)),
                0)
          << "raw row mismatch at key index " << i;
    }
  }

  void CheckPooling(PoolingType_t pooling_type, SparseType_t sparse_type,
                    DataType_t weight_type = DataType_t::Unknown,
                    bool device_output = false) {
    auto lookup_keys = keys();
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling_type;
    pp.output_type = output_dtype_;

    IndexT fixed_hotness = 4;
    std::vector<IndexT> csr_offsets{0, 3, 7, static_cast<IndexT>(lookup_keys.size())};
    double ignored_weight = 0.0;
    int64_t output_rows = static_cast<int64_t>(lookup_keys.size());
    if (pooling_type == PoolingType_t::Concatenate) {
      // Concatenate must ignore every bag/weight field.
      pp.sparse_type = static_cast<SparseType_t>(2);
      pp.csr_offsets = nullptr;
      pp.num_csr_offsets = -1;
      pp.fixed_hotness = -1;
      pp.weights = &ignored_weight;
      pp.weight_type = DataType_t::Float64;
    } else if (sparse_type == SparseType_t::Fixed) {
      pp.sparse_type = SparseType_t::Fixed;
      pp.fixed_hotness = fixed_hotness;
      output_rows = static_cast<int64_t>(lookup_keys.size()) / fixed_hotness;
    } else {
      pp.sparse_type = SparseType_t::CSR;
      pp.csr_offsets = csr_offsets.data();
      pp.num_csr_offsets = static_cast<int64_t>(csr_offsets.size());
      output_rows = static_cast<int64_t>(csr_offsets.size()) - 1;
    }

    std::vector<int8_t> weights;
    if (is_weighted_pooling(pooling_type)) {
      GenerateWeights(weights, lookup_keys.size(), weight_type, 4141);
      pp.weights = weights.data();
      pp.weight_type = weight_type;
    }

    const size_t output_bytes =
        static_cast<size_t>(output_rows) * static_cast<size_t>(output_stride_);
    std::vector<int8_t> result(output_bytes);
    void* output = result.data();
    DevicePtr<void> device_result;
    if (device_output) {
      void* ptr = nullptr;
      NVE_CHECK_(cudaMalloc(&ptr, output_bytes));
      device_result.reset(ptr);
      output = ptr;
    }

    layer_->lookup(ctx_, static_cast<int64_t>(lookup_keys.size()), lookup_keys.data(), output,
                   output_stride_, nullptr, &pp, nullptr);
    ctx_->wait();
    if (device_output) {
      NVE_CHECK_(cudaMemcpy(result.data(), output, output_bytes, cudaMemcpyDeviceToHost));
    }

    std::vector<int8_t> gathered(
        static_cast<size_t>(lookup_keys.size()) * static_cast<size_t>(row_bytes_));
    for (size_t i = 0; i < lookup_keys.size(); ++i) {
      std::memcpy(gathered.data() + static_cast<int64_t>(i) * row_bytes_,
                  host_table_.data() + static_cast<int64_t>(lookup_keys[i]) * row_bytes_,
                  static_cast<size_t>(row_bytes_));
    }

    HostTableConfig ref_config;
    ref_config.value_dtype = dtype_;
    ref_config.max_value_size = row_bytes_;
    MockHostTable<IndexT> reference(ref_config, true /*functional_ref*/, host_table_.data());
    std::vector<int8_t> expected(output_bytes);
    const bool concat = pooling_type == PoolingType_t::Concatenate;
    const bool ref_is_csr = !concat && pp.sparse_type == SparseType_t::CSR;
    const int64_t ref_fixed_hotness = concat ? 1 : pp.fixed_hotness;
    reference.combine(
        gathered.data(), static_cast<int64_t>(lookup_keys.size()), pooling_type,
        concat ? SparseType_t::Fixed : pp.sparse_type,
        ref_is_csr ? pp.csr_offsets : nullptr, ref_is_csr ? pp.num_csr_offsets : 0,
        ref_fixed_hotness,
        concat ? nullptr : pp.weights,
        concat ? DataType_t::Unknown : pp.weight_type,
        expected.data(), output_dtype_);

    float tolerance = output_dtype_ == DataType_t::Float16 ? 1e-3f : 1e-5f;
    if (pooling_type == PoolingType_t::Sum || pooling_type == PoolingType_t::WeightedSum) {
      tolerance *= 16.0f;
    }
    for (int64_t i = 0; i < output_rows * value_count_; ++i) {
      ASSERT_NEAR(load_as_float(result.data(), i, output_dtype_),
                  load_as_float(expected.data(), i, output_dtype_), tolerance)
          << "pooled element " << i;
    }
  }

  void CheckUpdateRawAndDequantized() {
    std::vector<IndexT> update_keys{2, 7, 11};
    std::vector<int8_t> replacements(
        static_cast<size_t>(update_keys.size()) * static_cast<size_t>(row_bytes_));
    InitTableRowsQuant(replacements.data(), row_bytes_, value_count_, 0, update_keys.size(), dtype_,
                       7777);

    layer_->update(ctx_, static_cast<int64_t>(update_keys.size()), update_keys.data(), row_bytes_,
                   row_bytes_, replacements.data(), -1);
    ctx_->wait();

    std::vector<int8_t> raw(replacements.size());
    layer_->lookup(ctx_, static_cast<int64_t>(update_keys.size()), update_keys.data(), raw.data(),
                   row_bytes_, nullptr, nullptr, nullptr);
    ctx_->wait();
    ASSERT_EQ(raw, replacements);

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = PoolingType_t::Concatenate;
    pp.output_type = output_dtype_;
    std::vector<int8_t> dequantized(
        static_cast<size_t>(update_keys.size()) * static_cast<size_t>(output_stride_));
    layer_->lookup(ctx_, static_cast<int64_t>(update_keys.size()), update_keys.data(),
                   dequantized.data(), output_stride_, nullptr, &pp, nullptr);
    ctx_->wait();

    for (size_t row = 0; row < update_keys.size(); ++row) {
      for (int64_t e = 0; e < value_count_; ++e) {
        const float expected = load_quant_row_element_as_float(
            replacements.data() + static_cast<int64_t>(row) * row_bytes_, e, value_count_, dtype_);
        ASSERT_NEAR(load_as_float(dequantized.data(),
                                  static_cast<int64_t>(row) * value_count_ + e, output_dtype_),
                    expected, output_dtype_ == DataType_t::Float16 ? 1e-3f : 1e-6f);
      }
    }
  }

  // Out of range keys must read the default row, both as stored bytes and dequantized: the row's
  // trailing scale/offset metadata has to come along for the dequantized values to match.
  void CheckDefaultRowLookup(int64_t default_row_index) {
    const std::vector<IndexT> lookup_keys{0, -1, kNumRows, kNumRows + 5};
    const auto num_keys = static_cast<int64_t>(lookup_keys.size());
    const auto* default_row = host_table_.data() + default_row_index * row_bytes_;

    std::vector<int8_t> raw(static_cast<size_t>(num_keys) * static_cast<size_t>(row_bytes_));
    layer_->lookup(ctx_, num_keys, lookup_keys.data(), raw.data(), row_bytes_, nullptr, nullptr,
                   nullptr);
    ctx_->wait();

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = PoolingType_t::Concatenate;
    pp.output_type = output_dtype_;
    std::vector<int8_t> dequantized(static_cast<size_t>(num_keys) *
                                    static_cast<size_t>(output_stride_));
    layer_->lookup(ctx_, num_keys, lookup_keys.data(), dequantized.data(), output_stride_, nullptr,
                   &pp, nullptr);
    ctx_->wait();

    for (int64_t i = 0; i < num_keys; ++i) {
      const IndexT key = lookup_keys[static_cast<size_t>(i)];
      const auto* expected_row =
          key_in_range(key, kNumRows) ? (host_table_.data() + key * row_bytes_) : default_row;
      EXPECT_EQ(std::memcmp(raw.data() + i * row_bytes_, expected_row,
                            static_cast<size_t>(row_bytes_)),
                0)
          << "raw row mismatch at key index " << i;
      for (int64_t e = 0; e < value_count_; ++e) {
        const float expected =
            load_quant_row_element_as_float(expected_row, e, value_count_, dtype_);
        ASSERT_NEAR(load_as_float(dequantized.data(), i * value_count_ + e, output_dtype_), expected,
                    output_dtype_ == DataType_t::Float16 ? 1e-3f : 1e-6f)
            << "dequantized element " << e << " at key index " << i;
      }
    }
  }

  std::shared_ptr<GPUEmbeddingLayer<IndexT>>& layer() { return layer_; }
  context_ptr_t& context() { return ctx_; }
  DataType_t output_dtype() const { return output_dtype_; }
  int64_t output_stride() const { return output_stride_; }
  int64_t row_bytes() const { return row_bytes_; }

 private:
  static constexpr int64_t kNumRows = 64;
  DataType_t dtype_;
  int64_t value_count_;
  int64_t row_bytes_;
  DataType_t output_dtype_;
  int64_t output_stride_;
  std::vector<int8_t> host_table_;
  void* device_table_{nullptr};
  std::shared_ptr<GPUEmbeddingLayer<IndexT>> layer_;
  context_ptr_t ctx_;
};

#define GPU_QUANT_RAW_TEST(test_name, index_type, value_type) \
  TEST(GPUQuantRaw, test_name) {                              \
    if (!gpu_test_device_available()) GTEST_SKIP();           \
    QuantGpuLayerTest<index_type> test(value_type);           \
    test.CheckRawLookup(false);                               \
    test.CheckRawLookup(false, true);                         \
  }

GPU_QUANT_RAW_TEST(QInt8F32_Int64, int64_t, DataType_t::QInt8RowwiseF32)
GPU_QUANT_RAW_TEST(QInt8F16_Int64, int64_t, DataType_t::QInt8RowwiseF16)
GPU_QUANT_RAW_TEST(QUint8F32_Int64, int64_t, DataType_t::QUint8RowwiseF32)
GPU_QUANT_RAW_TEST(QUint8F16_Int64, int64_t, DataType_t::QUint8RowwiseF16)
GPU_QUANT_RAW_TEST(QInt8F32_Int32, int32_t, DataType_t::QInt8RowwiseF32)
GPU_QUANT_RAW_TEST(QInt8F16_Int32, int32_t, DataType_t::QInt8RowwiseF16)
GPU_QUANT_RAW_TEST(QUint8F32_Int32, int32_t, DataType_t::QUint8RowwiseF32)
GPU_QUANT_RAW_TEST(QUint8F16_Int32, int32_t, DataType_t::QUint8RowwiseF16)

#undef GPU_QUANT_RAW_TEST

TEST(GPUQuantRaw, DeviceOutput) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> test(DataType_t::QUint8RowwiseF32);
  test.CheckRawLookup(true);
  test.CheckRawLookup(true, true);
}

TEST(GPUQuantRaw, OutOfRangeKeysReturnDefaultRow) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  constexpr int64_t default_row_index = 5;
  QuantGpuLayerTest<int64_t> test(DataType_t::QInt8RowwiseF32, 64, default_row_index);
  test.CheckDefaultRowLookup(default_row_index);
}

TEST(GPUQuantPool, ConcatF32HostOutput) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> test(DataType_t::QInt8RowwiseF32);
  test.CheckPooling(PoolingType_t::Concatenate, SparseType_t::Fixed);
}

TEST(GPUQuantPool, ConcatF16DeviceOutput) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int32_t> test(DataType_t::QUint8RowwiseF16);
  test.CheckPooling(PoolingType_t::Concatenate, SparseType_t::Fixed,
                    DataType_t::Unknown, true);
}

TEST(GPUQuantPool, FixedSumAndMean) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> f32_test(DataType_t::QInt8RowwiseF32);
  f32_test.CheckPooling(PoolingType_t::Sum, SparseType_t::Fixed);
  QuantGpuLayerTest<int64_t> f16_test(DataType_t::QUint8RowwiseF16);
  f16_test.CheckPooling(PoolingType_t::Mean, SparseType_t::Fixed);
}

TEST(GPUQuantPool, CSRSumBothKeyWidths) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> int64_test(DataType_t::QUint8RowwiseF32);
  int64_test.CheckPooling(PoolingType_t::Sum, SparseType_t::CSR);
  QuantGpuLayerTest<int32_t> int32_test(DataType_t::QUint8RowwiseF32);
  int32_test.CheckPooling(PoolingType_t::Sum, SparseType_t::CSR);
}

TEST(GPUQuantPool, WeightedSumBothWeightTypes) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> test(DataType_t::QInt8RowwiseF32);
  test.CheckPooling(PoolingType_t::WeightedSum, SparseType_t::Fixed,
                    DataType_t::Float32);
  test.CheckPooling(PoolingType_t::WeightedSum, SparseType_t::Fixed,
                    DataType_t::Float16);
}

TEST(GPUQuantPool, WeightedMeanBothWeightTypes) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int32_t> test(DataType_t::QUint8RowwiseF16);
  test.CheckPooling(PoolingType_t::WeightedMean, SparseType_t::CSR,
                    DataType_t::Float32);
  test.CheckPooling(PoolingType_t::WeightedMean, SparseType_t::CSR,
                    DataType_t::Float16);
}

TEST(GPUQuantModify, UpdatePreservesRawAndDequantizedRows) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> test(DataType_t::QInt8RowwiseF32);
  test.CheckUpdateRawAndDequantized();
}

TEST(GPUQuantValidation, RejectsInvalidStoredWidths) {
  GPUEmbeddingLayerConfig config;
  config.embedding_table = &config;
  config.num_embeddings = 1;
  config.value_dtype = DataType_t::QInt8RowwiseF16;

  config.embedding_width_in_bytes = 3;
  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
  config.embedding_width_in_bytes = 2;  // Metadata only: no quantized values.
  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
  config.embedding_width_in_bytes = 0;
  EXPECT_THROW((void)std::make_shared<GPUEmbeddingLayer<int64_t>>(config), Exception);
}

TEST(GPUQuantValidation, RejectsInvalidPoolingRequests) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> test(DataType_t::QInt8RowwiseF32);
  const int64_t key = 1;
  std::vector<int8_t> output(static_cast<size_t>(test.output_stride()));

  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.row_bytes() - 1, nullptr, nullptr, nullptr),
               Exception);

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Concatenate;
  pp.output_type = DataType_t::Unknown;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  pp.output_type = DataType_t::Float16;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  pp.output_type = test.output_dtype();
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride() - 1, nullptr, &pp, nullptr),
               Exception);

  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::Fixed;
  int64_t hotness = 0;
  pp.fixed_hotness = hotness;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  hotness = 2;
  pp.fixed_hotness = hotness;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  pp.sparse_type = SparseType_t::CSR;
  pp.csr_offsets = &hotness;
  pp.num_csr_offsets = 1;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  pp.num_csr_offsets = static_cast<int64_t>(INT32_MAX) + 2;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  pp.pooling_type = PoolingType_t::Concatenate;
  EXPECT_THROW(test.layer()->lookup(test.context(), static_cast<int64_t>(INT_MAX) + 1, &key,
                                    output.data(), test.output_stride(), nullptr, &pp, nullptr),
               Exception);

  pp.pooling_type = PoolingType_t::WeightedSum;
  pp.sparse_type = SparseType_t::Fixed;
  hotness = 1;
  pp.fixed_hotness = hotness;
  double weight = 1.0;
  pp.weights = &weight;
  pp.weight_type = DataType_t::Float64;
  EXPECT_THROW(test.layer()->lookup(test.context(), 1, &key, output.data(),
                                    test.output_stride(), nullptr, &pp, nullptr),
               Exception);
}

TEST(GPUQuantValidation, RejectsAccumulate) {
  if (!gpu_test_device_available()) GTEST_SKIP();
  QuantGpuLayerTest<int64_t> test(DataType_t::QInt8RowwiseF32);
  const int64_t key = 1;
  std::vector<int8_t> values(static_cast<size_t>(test.row_bytes()));
  EXPECT_THROW(test.layer()->accumulate(test.context(), 1, &key, test.row_bytes(),
                                        test.row_bytes(), values.data(), DataType_t::Float32, -1),
               Exception);
}

}  // namespace

// [Sanity] Init the layer
TEST_P(GPU, Init) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
}

// [Sanity] lookup 1key
TEST_P(GPU, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, 1, tc.row_size, 0, static_cast<int64_t>(elt.rows_), tc.data_type);
  elt.LookupAndCheck(tc, keys);
}

// [Lookup]
TEST_P(GPU, Lookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, static_cast<size_t>(tc.test_keys), tc.row_size, 0, static_cast<int64_t>(elt.rows_), tc.data_type);
  elt.LookupAndCheck(tc, keys);
}

TEST_P(GPU_UP_ACC, Update) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, static_cast<size_t>(tc.test_keys), tc.row_size, 0, static_cast<int64_t>(elt.rows_), tc.data_type);
  elt.Update(keys, data);
}

TEST_P(GPU_UP_ACC, Accumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, static_cast<size_t>(tc.test_keys), tc.row_size, 0, static_cast<int64_t>(elt.rows_), tc.data_type);
  elt.Accumulate(keys, data, tc.data_type);
}

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    GPU,
    ::testing::Values(
        //  TestCase: gpu_table_Size,   row_size,         test_keys         data_type
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16}),  // Single key
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16}),   // Medium
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::Concatenate, SparseType_t::Fixed}), // fixed hotness concat
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::Sum, SparseType_t::Fixed}), // fixed hotness sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::Sum, SparseType_t::CSR}), // CSR sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::WeightedSum, SparseType_t::Fixed}), // fixed hotness weighted sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::WeightedSum, SparseType_t::CSR}), // CSR weighted sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::Mean, SparseType_t::Fixed}), // fixed hotness mean
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::Mean, SparseType_t::CSR}), // CSR mean
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::WeightedMean, SparseType_t::Fixed}), // fixed hotness weighted mean
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16,
                      true, 32, PoolingType_t::WeightedMean, SparseType_t::CSR}), // CSR weighted mean

        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float32}),  // Single key
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32}),   // Medium
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::Concatenate, SparseType_t::Fixed}), // fixed hotness concat
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::Sum, SparseType_t::Fixed}), // fixed hotness sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::Sum, SparseType_t::CSR}), // CSR sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::WeightedSum, SparseType_t::Fixed}), // fixed hotness weighted sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::WeightedSum, SparseType_t::CSR}), // CSR weighted sum
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::Mean, SparseType_t::Fixed}), // fixed hotness mean
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::Mean, SparseType_t::CSR}), // CSR mean
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::WeightedMean, SparseType_t::Fixed}), // fixed hotness weighted mean
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32,
                      true, 32, PoolingType_t::WeightedMean, SparseType_t::CSR}) // CSR weighted mean
      ));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_EmbLayer_Large,
    GPU,
    ::testing::Values(
        //  TestCase: gpu_table_Size,   row_size,         test_keys         data_type
        GPUTestCase({ int64_t(1) << 32, int64_t(1) << 10, int64_t(1) << 16, DataType_t::Float32})  // Large 
      ));


INSTANTIATE_TEST_SUITE_P(
    EmbLayerUpAcc,
    GPU_UP_ACC,
    ::testing::Values(
        //  TestCase: gpu_table_Size,   row_size,         test_keys         data_type
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16}),  // Single key
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16}),  // Single key
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32}),  // Medium
        GPUTestCase({ int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32})   // Medium
      ));

}  // namespace nve
