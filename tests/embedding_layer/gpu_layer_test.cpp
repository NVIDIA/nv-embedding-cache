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

#include <gpu_embedding_layer.hpp>
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
    : m_row_size(row_size), m_rows(gpu_table_size / row_size) {
    // init linear table
    {
      NVE_CHECK_(cudaMalloc(&m_gpu_table, static_cast<size_t>(m_row_size * m_rows)));
      NVE_CHECK_(cudaMallocHost(&m_host_table, static_cast<size_t>(m_row_size * m_rows)));
      NVE_CHECK_(m_row_size % dtype_size(data_type) == 0); // cannot use ASSERT_EQ in c'tor

      // Init table values with multiple threads to save on test time
      std::vector<std::shared_ptr<std::thread>> input_gen_threads;

      constexpr int64_t num_threads = 32;
      for (int64_t t=0 ; t<num_threads ; t++) {
        auto start_row = t * m_rows / num_threads;
        auto end_row = std::min<int64_t>((t + 1) * m_rows / num_threads, m_rows);

        input_gen_threads.push_back(std::make_shared<std::thread>(
          InitTableRows, m_host_table, m_row_size, start_row, end_row, data_type, seed + static_cast<size_t>(t)
        ));
      }
      for (auto& t : input_gen_threads) {
          t->join();
      }

      HostTableConfig mock_cfg;
      mock_cfg.value_dtype = data_type;
      mock_cfg.max_value_size = static_cast<int64_t>(m_row_size);
      m_ref_tab = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/, m_host_table);
    }

    // init layer
    {
      NVE_CHECK_(cudaMemcpy(m_gpu_table, m_host_table, static_cast<size_t>(m_row_size * m_rows), cudaMemcpyHostToDevice));
      GPUEmbeddingLayerConfig embedding_table_cfg;
      embedding_table_cfg.device_id = DEVICE_ID;
      embedding_table_cfg.num_embeddings = static_cast<int64_t>(m_rows);
      embedding_table_cfg.embedding_width_in_bytes = static_cast<int64_t>(row_size);
      embedding_table_cfg.embedding_table = m_gpu_table;
      embedding_table_cfg.value_dtype = data_type;
      m_layer = std::make_shared<GPUEmbeddingLayer<IndexT>>(embedding_table_cfg);
    }
    // init context
    m_ctx = m_layer->create_execution_context(0, 0, nullptr, nullptr);
  }

  GPULayerLookupTest(GPUTestCase tc) : GPULayerLookupTest<IndexT>(tc.row_size, tc.gpu_table_size, tc.data_type) {}

  ~GPULayerLookupTest() {
    NVE_CHECK_(cudaFree(m_gpu_table));
    NVE_CHECK_(cudaFreeHost(m_host_table));
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

    uint64_t output_size = static_cast<size_t>(num_keys * m_row_size);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);

    std::vector<float> hitrates(3);

    if (tc.do_pooling) {
      EmbeddingLayerBase::PoolingParams pp;
      pp.pooling_type = tc.pooling_type;
      pp.sparse_type = tc.offsets_layout;

      SetupCSROffsets<int64_t> offsets_setup(tc.offsets_layout == SparseType_t::CSR ? static_cast<int64_t>(num_keys) : 1);
      if (tc.offsets_layout == SparseType_t::CSR) {
        pp.key_indices = offsets_setup.offsets_buffer;
        pp.num_key_indices = static_cast<int64_t>(offsets_setup.num_offsets);
        output_bags = static_cast<int64_t>(offsets_setup.num_offsets) - 1;
      } else {
        const int64_t hotness = (num_keys == 1) ? 1 : tc.hotness; // special handling for single key test
        offsets_setup.offsets_buffer[0] = hotness; 
        pp.key_indices = offsets_setup.offsets_buffer;
        pp.num_key_indices = 1;
        if (tc.pooling_type != PoolingType_t::Concatenate) {
          output_bags = num_keys / hotness;
        }
      }

      std::vector<int8_t> weights;
      if ((tc.pooling_type == PoolingType_t::WeightedSum) || (tc.pooling_type == PoolingType_t::WeightedMean)) {
        GenerateWeights(weights, static_cast<uint64_t>(num_keys), tc.data_type);
        pp.sparse_weights = weights.data();
        // for now, weight type (pp.weight_type) is assumed to be data type
        pp.weight_type = tc.data_type;
      } else {
        pp.sparse_weights = nullptr;
      }

      m_layer->lookup(m_ctx, num_keys, keys_buffer, output, m_row_size, nullptr /*hitmask*/,
                      &pp /*pool_params*/, hitrates.data());

      std::vector<int8_t> find_output(static_cast<size_t>(num_keys * m_row_size));
      m_ref_tab->find(m_ctx, num_keys, keys_buffer, nullptr /*hitmask*/, m_row_size,
                                  find_output.data(), nullptr /*value_sizes*/);

      m_ref_output.resize(static_cast<size_t>(output_bags * m_row_size));
      m_ref_tab->combine(find_output.data(), num_keys, pp.pooling_type, pp.sparse_type, pp.key_indices,
                         pp.num_key_indices, pp.sparse_weights, pp.weight_type, m_ref_output.data());
    } else {
      m_layer->lookup(m_ctx, num_keys, keys_buffer, output, m_row_size, nullptr /*hitmask*/,
                      nullptr, hitrates.data());

      m_ref_output.resize(static_cast<size_t>(num_keys * m_row_size));
      m_ref_tab->find(m_ctx, num_keys, keys_buffer, nullptr /*hitmask*/, m_row_size,
                      m_ref_output.data(), nullptr /*value_sizes*/);
    }
    
    NVE_CHECK_(cudaDeviceSynchronize());

    // compare hitrates
    ASSERT_EQ(hitrates[0], 1.0f);

    const int64_t output_elements = static_cast<int64_t>(m_ref_output.size()) / dtype_size(tc.data_type);

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
        load_as_float(m_ref_output.data(), i, tc.data_type),
        tolerance
      );

    }
    NVE_CHECK_(cudaFreeHost(output));
  }

  void Update(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
              uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }

    SetupKeys setup(keys, start_key, end_key, datavectors, m_row_size);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    // 1. lookup these keys to ref
    auto output_size = static_cast<size_t>(num_keys * m_row_size);
    int8_t* lookup_output{nullptr};
    NVE_CHECK_(cudaMallocHost(&lookup_output, output_size));
    NVE_CHECK_(lookup_output != 0);
    std::vector<uint8_t> res_lookup_output(output_size);
    std::vector<float> hitrates(3);

    // 3. call accumulate 
    m_layer->update(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer);

    NVE_CHECK_(cudaDeviceSynchronize());

    // 4. call lookup
    m_layer->lookup(m_ctx, num_keys, keys_buffer, lookup_output, m_row_size, nullptr /*hitmask*/,
                    nullptr, hitrates.data());

    NVE_CHECK_(cudaDeviceSynchronize());

    NVE_CHECK_(cudaMemcpy(&res_lookup_output[0], lookup_output, output_size, cudaMemcpyDeviceToHost));
    NVE_CHECK_(cudaDeviceSynchronize());

    // 5. compare to ref
    for (int64_t i = 0; i < num_keys; i++) {
      for (int64_t j = 0; j < m_row_size; j++) {
        ASSERT_EQ(res_lookup_output[static_cast<size_t>((i * m_row_size) + j)], data_buffer[static_cast<size_t>((i * m_row_size) + j)]);
      }
    }

    NVE_CHECK_(cudaFreeHost(lookup_output));
  }

  void Accumulate(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
                  DataType_t value_type, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
  
    SetupKeys setup(keys, start_key, end_key, datavectors, m_row_size);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    // 1. lookup these keys to ref
    auto output_size = static_cast<size_t>(num_keys * m_row_size);
    int8_t* lookup_output{nullptr};
    NVE_CHECK_(cudaMallocHost(&lookup_output, output_size));
    NVE_CHECK_(lookup_output != 0);
    std::vector<int8_t> ref_lookup_output(static_cast<size_t>(num_keys * m_row_size));
    std::vector<int8_t> res_lookup_output(static_cast<size_t>(num_keys * m_row_size));
    std::vector<float> hitrates(3);

    m_layer->lookup(m_ctx, num_keys, keys_buffer, lookup_output, m_row_size, nullptr /*hitmask*/,
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
    m_layer->accumulate(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, setup.data_buffer, value_type);

    NVE_CHECK_(cudaDeviceSynchronize());

    // 4. call lookup
    m_layer->lookup(m_ctx, num_keys, keys_buffer, lookup_output, m_row_size, nullptr /*hitmask*/,
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
  const int64_t m_row_size;
  const int64_t m_rows;
 private:
  std::vector<int8_t> m_ref_output;
  int8_t* m_gpu_table{nullptr};
  int8_t* m_host_table{nullptr};
  std::shared_ptr<layer_type> m_layer{nullptr};
  context_ptr_t m_ctx;
  std::shared_ptr<MockHostTable<IndexT>> m_ref_tab{nullptr};
};

// [Sanity] Init the layer
TEST_P(GPU, Init) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
}

// [Sanity] lookup 1key
TEST_P(GPU, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, 1, tc.row_size, 0, static_cast<int64_t>(elt.m_rows), tc.data_type);
  elt.LookupAndCheck(tc, keys);
}

// [Lookup]
TEST_P(GPU, Lookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, static_cast<size_t>(tc.test_keys), tc.row_size, 0, static_cast<int64_t>(elt.m_rows), tc.data_type);
  elt.LookupAndCheck(tc, keys);
}

TEST_P(GPU_UP_ACC, Update) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, static_cast<size_t>(tc.test_keys), tc.row_size, 0, static_cast<int64_t>(elt.m_rows), tc.data_type);
  elt.Update(keys, data);
}

TEST_P(GPU_UP_ACC, Accumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  GPULayerLookupTest<int64_t> elt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, static_cast<size_t>(tc.test_keys), tc.row_size, 0, static_cast<int64_t>(elt.m_rows), tc.data_type);
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
