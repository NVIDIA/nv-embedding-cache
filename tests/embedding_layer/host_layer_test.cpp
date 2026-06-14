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

// C++ functional tests for HostEmbeddingLayer wrapping a LinearHostTable.
// Mirrors the patterns in linear_host_table_test.cpp but exercises the layer
// surface (lookup / update / accumulate / clear / erase, hitmask & default-
// embedding paths) instead of poking the table directly.

#include <gtest/gtest.h>

#include "cpu_ops/cpu_pooling.h"
#include "include/buffer_wrapper.hpp"
#include "include/common.hpp"
#include "include/host_embedding_layer.hpp"
#include "include/linear_host_table.hpp"
#include "include/nve_types.hpp"

#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

namespace nve {

struct HostLayerTestParams {
  int64_t row_size_bytes;
  int64_t num_rows;
  DataType_t value_dtype;
};

template <typename KeyType>
class HostLayerTest : public testing::TestWithParam<HostLayerTestParams> {
public:
  using LayerType = HostEmbeddingLayer<KeyType>;
  using TableType = LinearHostTable<KeyType>;
  using DataType = float;

  HostLayerTest() { Init(); }

  ~HostLayerTest() {
    ctx_.reset();
    layer_.reset();
    table_.reset();
    if (h_table_) {
      NVE_CHECK_(cudaFreeHost(h_table_));
    }
  }

  // ---------- helpers --------------------------------------------------------

  size_t data_elements() const {
    return static_cast<size_t>(GetParam().row_size_bytes) / sizeof(DataType);
  }

  std::vector<DataType> lookup(const std::vector<KeyType>& keys,
                               max_bitmask_repr_t* hitmask = nullptr,
                               float* hitrates = nullptr) {
    const auto& params = GetParam();
    std::vector<DataType> out(keys.size() * data_elements(), DataType{0});
    layer_->lookup(ctx_, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                   params.row_size_bytes, hitmask, /*pool_params=*/nullptr, hitrates);
    return out;
  }

  void update(KeyType key, const std::vector<DataType>& row) {
    const auto& params = GetParam();
    layer_->update(ctx_, /*num_keys=*/1, &key, params.row_size_bytes, params.row_size_bytes,
                   row.data(), /*table_id=*/0);
  }

  void accumulate(KeyType key, const std::vector<DataType>& row) {
    const auto& params = GetParam();
    layer_->accumulate(ctx_, /*num_keys=*/1, &key, params.row_size_bytes, params.row_size_bytes,
                       row.data(), params.value_dtype, /*table_id=*/0);
  }

  // Read back a row directly from the backing buffer for ground-truth checks.
  std::vector<DataType> read_backing_row(KeyType key) const {
    const size_t elements = data_elements();
    return std::vector<DataType>(h_table_ + static_cast<size_t>(key) * elements,
                                 h_table_ + (static_cast<size_t>(key) + 1) * elements);
  }

  // ---------- members --------------------------------------------------------

  std::shared_ptr<TableType> table_;
  std::shared_ptr<LayerType> layer_;
  context_ptr_t ctx_;
  DataType* h_table_{nullptr};

  static constexpr KeyType kTestKey = static_cast<KeyType>(1337);

private:
  void Init() {
    const auto& params = GetParam();

    LinearHostTableConfig cfg;
    cfg.value_dtype = params.value_dtype;
    cfg.max_threads = 64;
    cfg.max_value_size = params.row_size_bytes;

    const size_t table_size = static_cast<size_t>(params.row_size_bytes * params.num_rows);
    NVE_CHECK_(cudaMallocHost(&h_table_, table_size));
    const size_t elements = table_size / sizeof(DataType);
    for (size_t i = 0; i < elements; i++) {
      h_table_[i] = static_cast<DataType>(10000 + i);
    }
    cfg.emb_table = h_table_;

    table_ = std::make_shared<TableType>(cfg);

    typename LayerType::Config layer_cfg;
    layer_cfg.layer_name = "host_layer_test";
    layer_ = std::make_shared<LayerType>(layer_cfg, table_);

    // Null stream/allocator/threadpool: the layer delegates context creation to
    // the host table and defaults to GetDefaultAllocator() / the default thread pool.
    ctx_ = layer_->create_execution_context(/*lookup_stream=*/0, /*modify_stream=*/0,
                                            /*thread_pool=*/nullptr, /*allocator=*/nullptr);
  }
};

// ---------------------------- test bodies ------------------------------------

template <typename KeyType>
void test_lookup_no_hitmask(HostLayerTest<KeyType>* t) {
  const auto out = t->lookup({HostLayerTest<KeyType>::kTestKey});
  const auto expected = t->read_backing_row(HostLayerTest<KeyType>::kTestKey);
  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_FLOAT_EQ(out[i], expected[i]);
  }
}

template <typename KeyType>
void test_lookup_with_hitmask(HostLayerTest<KeyType>* t) {
  const std::vector<KeyType> keys{static_cast<KeyType>(0), static_cast<KeyType>(5),
                                  static_cast<KeyType>(42), static_cast<KeyType>(1337)};
  const size_t mask_words = (keys.size() + 63) / 64;
  std::vector<max_bitmask_repr_t> hitmask(mask_words, 0xdeadbeefdeadbeefULL);  // garbage on entry
  float hitrates[1] = {-1.0f};
  const auto out = t->lookup(keys, hitmask.data(), hitrates);

  // Every key in a LinearHostTable resolves — the kernel sets all-ones.
  for (size_t i = 0; i < keys.size(); i++) {
    const auto bit = (hitmask[i / 64] >> (i % 64)) & max_bitmask_repr_t{1};
    EXPECT_EQ(bit, max_bitmask_repr_t{1}) << "key index " << i << " missing in hitmask";
  }
  EXPECT_FLOAT_EQ(hitrates[0], 1.0f);

  for (size_t k = 0; k < keys.size(); k++) {
    const auto expected = t->read_backing_row(keys[k]);
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_FLOAT_EQ(out[k * t->data_elements() + i], expected[i]);
    }
  }
}

// Regression for an out-of-bounds hitmask write when num_keys is an exact
// multiple of 64: the hitmask buffer is exactly ceil(n/64) words, and the
// gather kernel must not write a trailing partial word past the end.
// Run under compute-sanitizer in CI to catch the OOB.
template <typename KeyType>
void test_lookup_hitmask_multiple_of_64(HostLayerTest<KeyType>* t) {
  constexpr int64_t kNumKeys = 128;  // exact multiple of 64
  std::vector<KeyType> keys(kNumKeys);
  for (int64_t i = 0; i < kNumKeys; i++) {
    keys[static_cast<size_t>(i)] = static_cast<KeyType>(i);
  }
  const size_t mask_words = (keys.size() + 63) / 64;  // == kNumKeys/64, no extra word
  std::vector<max_bitmask_repr_t> hitmask(mask_words, 0);
  const auto out = t->lookup(keys, hitmask.data(), /*hitrates=*/nullptr);

  // Every key resolves -> all bits in every word set.
  for (size_t w = 0; w < mask_words; w++) {
    EXPECT_EQ(hitmask[w], ~max_bitmask_repr_t{0}) << "word " << w << " not all-ones";
  }
  for (size_t k = 0; k < keys.size(); k++) {
    const auto expected = t->read_backing_row(keys[k]);
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_FLOAT_EQ(out[k * t->data_elements() + i], expected[i]);
    }
  }
}

template <typename KeyType>
void test_update_then_lookup(HostLayerTest<KeyType>* t) {
  const auto elements = t->data_elements();
  std::vector<typename HostLayerTest<KeyType>::DataType> row(elements);
  for (size_t i = 0; i < elements; i++) {
    row[i] = static_cast<float>(-static_cast<int64_t>(i));
  }
  t->update(HostLayerTest<KeyType>::kTestKey, row);

  const auto out = t->lookup({HostLayerTest<KeyType>::kTestKey});
  ASSERT_EQ(out.size(), row.size());
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_FLOAT_EQ(out[i], row[i]);
  }
}

template <typename KeyType>
void test_accumulate(HostLayerTest<KeyType>* t) {
  const auto elements = t->data_elements();
  const auto before = t->read_backing_row(HostLayerTest<KeyType>::kTestKey);
  std::vector<typename HostLayerTest<KeyType>::DataType> delta(elements);
  for (size_t i = 0; i < elements; i++) {
    delta[i] = static_cast<float>(i);
  }
  t->accumulate(HostLayerTest<KeyType>::kTestKey, delta);

  const auto out = t->lookup({HostLayerTest<KeyType>::kTestKey});
  ASSERT_EQ(out.size(), before.size());
  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_FLOAT_EQ(out[i], before[i] + delta[i]);
  }
}

template <typename KeyType>
void test_lookup_gpu_input(HostLayerTest<KeyType>* t) {
  // Same data path as the python binding's `cuda` device-id flow: caller
  // supplies device-resident keys and output. Layer + BufferWrapper transit
  // the data through host scratch + a cudaMemcpyAsync back to GPU output.
  const auto& params = t->GetParam();
  std::vector<KeyType> h_keys{static_cast<KeyType>(0), static_cast<KeyType>(5),
                              static_cast<KeyType>(1337)};
  const size_t key_bytes = h_keys.size() * sizeof(KeyType);
  const size_t output_bytes = h_keys.size() * static_cast<size_t>(params.row_size_bytes);

  KeyType* d_keys = nullptr;
  void* d_output = nullptr;
  ASSERT_EQ(cudaMalloc(&d_keys, key_bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_output, output_bytes), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_keys, h_keys.data(), key_bytes, cudaMemcpyHostToDevice), cudaSuccess);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto ctx = t->layer_->create_execution_context(stream, stream, nullptr, nullptr);

  t->layer_->lookup(ctx, static_cast<int64_t>(h_keys.size()), d_keys, d_output,
                    params.row_size_bytes, /*hitmask=*/nullptr, /*pool_params=*/nullptr,
                    /*hitrates=*/nullptr);

  std::vector<float> h_output(h_keys.size() * t->data_elements(), 0.0f);
  ASSERT_EQ(cudaMemcpyAsync(h_output.data(), d_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  for (size_t k = 0; k < h_keys.size(); k++) {
    const auto expected = t->read_backing_row(h_keys[k]);
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_FLOAT_EQ(h_output[k * t->data_elements() + i], expected[i])
          << "mismatch at key " << static_cast<int64_t>(h_keys[k]) << " elem " << i;
    }
  }

  ctx.reset();
  cudaStreamDestroy(stream);
  cudaFree(d_keys);
  cudaFree(d_output);
}

// A layer configured with a default_embedding but called without a caller-
// supplied hitmask must allocate one internally rather than failing. (With a
// LinearHostTable every key resolves, so the default fill itself is a no-op,
// but the internal-hitmask allocation path still runs and must be safe.)
template <typename KeyType>
void test_default_embedding_no_hitmask(HostLayerTest<KeyType>* t) {
  const auto& params = t->GetParam();
  using LayerType = typename HostLayerTest<KeyType>::LayerType;

  typename LayerType::Config cfg;
  cfg.layer_name = "host_layer_default_emb";
  cfg.default_embedding.assign(static_cast<size_t>(params.row_size_bytes),
                               static_cast<uint8_t>(0));
  auto layer = std::make_shared<LayerType>(cfg, t->table_);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  const std::vector<KeyType> keys{static_cast<KeyType>(0), static_cast<KeyType>(1337)};
  std::vector<float> out(keys.size() * t->data_elements(), 0.0f);
  EXPECT_NO_THROW(layer->lookup(ctx, static_cast<int64_t>(keys.size()), keys.data(),
                                out.data(), params.row_size_bytes,
                                /*hitmask=*/nullptr, /*pool_params=*/nullptr,
                                /*hitrates=*/nullptr));

  for (size_t k = 0; k < keys.size(); k++) {
    const auto expected = t->read_backing_row(keys[k]);
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_FLOAT_EQ(out[k * t->data_elements() + i], expected[i]);
    }
  }
  ctx.reset();
}

// LinearHostTable's insert / clear / erase are documented no-ops — the layer
// should forward without throwing.
template <typename KeyType>
void test_noop_modifications(HostLayerTest<KeyType>* t) {
  const auto& params = t->GetParam();
  std::vector<KeyType> keys{HostLayerTest<KeyType>::kTestKey};
  std::vector<float> row(t->data_elements(), 0.0f);

  EXPECT_NO_THROW(t->layer_->insert(t->ctx_, 1, keys.data(), params.row_size_bytes,
                                    params.row_size_bytes, row.data(), /*table_id=*/0));
  EXPECT_NO_THROW(t->layer_->clear(t->ctx_));
  EXPECT_NO_THROW(t->layer_->erase(t->ctx_, 1, keys.data(), /*table_id=*/0));
}

// ----------------------------- pooling tests ----------------------------------

// Serial CPU reference: pool the backing rows of `keys` into `num_bags` bags as
// defined by `bag_offsets` (size num_bags+1), matching cpu_kernel_pooling.
template <typename KeyType>
std::vector<float> reference_pool(HostLayerTest<KeyType>* t, const std::vector<KeyType>& keys,
                                  const std::vector<KeyType>& bag_offsets, PoolingType_t pooling,
                                  const std::vector<float>& weights) {
  const size_t elements = t->data_elements();
  const size_t num_bags = bag_offsets.size() - 1;
  const bool weighted =
      (pooling == PoolingType_t::WeightedSum) || (pooling == PoolingType_t::WeightedMean);
  const bool mean = (pooling == PoolingType_t::Mean) || (pooling == PoolingType_t::WeightedMean);
  std::vector<float> expected(num_bags * elements, 0.0f);
  for (size_t b = 0; b < num_bags; b++) {
    const int64_t start = bag_offsets[b];
    const int64_t end = bag_offsets[b + 1];
    const int64_t count = end - start;
    // WeightedMean divides by the sum of weights; Mean by the element count.
    float inv = 1.0f;
    if (mean) {
      float denom = static_cast<float>(count);
      if (weighted) {
        denom = 0.0f;
        for (int64_t k = start; k < end; k++) {
          denom += weights[static_cast<size_t>(k)];
        }
      }
      inv = (denom != 0.0f) ? 1.0f / denom : 0.0f;
    }
    for (size_t e = 0; e < elements; e++) {
      float acc = 0.0f;
      for (int64_t k = start; k < end; k++) {
        float v = t->read_backing_row(keys[static_cast<size_t>(k)])[e];
        if (weighted) {
          v *= weights[static_cast<size_t>(k)];
        }
        acc += v;
      }
      expected[b * elements + e] = acc * inv;
    }
  }
  return expected;
}

template <typename KeyType>
std::vector<KeyType> make_keys(std::initializer_list<int64_t> ids) {
  std::vector<KeyType> keys;
  keys.reserve(ids.size());
  for (int64_t id : ids) {
    keys.push_back(static_cast<KeyType>(id));
  }
  return keys;
}

// Fixed-hotness pooling: 6 keys, hotness 3 -> 2 bags. Exercises every pooling type.
template <typename KeyType>
void test_pooling_fixed(HostLayerTest<KeyType>* t) {
  const auto& params = t->GetParam();
  const size_t elements = t->data_elements();
  const auto keys = make_keys<KeyType>({0, 1, 2, 5, 42, 1337});
  const KeyType hotness = static_cast<KeyType>(3);
  const auto bag_offsets = make_keys<KeyType>({0, 3, 6});
  const int64_t num_bags = 2;
  const std::vector<float> weights{0.5f, 1.5f, 2.0f, 1.0f, 0.25f, 4.0f};

  for (const auto pooling : {PoolingType_t::Sum, PoolingType_t::Mean, PoolingType_t::WeightedSum,
                             PoolingType_t::WeightedMean}) {
    const bool weighted =
        (pooling == PoolingType_t::WeightedSum) || (pooling == PoolingType_t::WeightedMean);
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling;
    pp.sparse_type = SparseType_t::Fixed;
    pp.key_indices = &hotness;
    pp.num_key_indices = 1;
    if (weighted) {
      pp.sparse_weights = weights.data();
      pp.weight_type = DataType_t::Float32;
    }

    std::vector<float> out(static_cast<size_t>(num_bags) * elements, 0.0f);
    t->layer_->lookup(t->ctx_, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                      params.row_size_bytes, /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

    const auto expected = reference_pool(t, keys, bag_offsets, pooling, weights);
    ASSERT_EQ(out.size(), expected.size());
    for (size_t i = 0; i < out.size(); i++) {
      EXPECT_FLOAT_EQ(out[i], expected[i]) << "pooling " << static_cast<int>(pooling) << " elem " << i;
    }
  }
}

// CSR pooling: 5 keys -> bags of size {2, 3} via offsets {0, 2, 5}.
template <typename KeyType>
void test_pooling_csr(HostLayerTest<KeyType>* t) {
  const auto& params = t->GetParam();
  const size_t elements = t->data_elements();
  const auto keys = make_keys<KeyType>({0, 5, 42, 1337, 7});
  // key_indices must be of the same type as the layer's key type.
  const auto offsets = make_keys<KeyType>({0, 2, 5});
  const int64_t num_bags = 2;
  const std::vector<float> weights{1.0f, 2.0f, 0.5f, 3.0f, 1.0f};

  for (const auto pooling : {PoolingType_t::Sum, PoolingType_t::Mean, PoolingType_t::WeightedSum,
                             PoolingType_t::WeightedMean}) {
    const bool weighted =
        (pooling == PoolingType_t::WeightedSum) || (pooling == PoolingType_t::WeightedMean);
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling;
    pp.sparse_type = SparseType_t::CSR;
    pp.key_indices = offsets.data();
    pp.num_key_indices = static_cast<int64_t>(offsets.size());
    if (weighted) {
      pp.sparse_weights = weights.data();
      pp.weight_type = DataType_t::Float32;
    }

    std::vector<float> out(static_cast<size_t>(num_bags) * elements, 0.0f);
    t->layer_->lookup(t->ctx_, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                      params.row_size_bytes, /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

    const auto expected = reference_pool(t, keys, offsets, pooling, weights);
    ASSERT_EQ(out.size(), expected.size());
    for (size_t i = 0; i < out.size(); i++) {
      EXPECT_FLOAT_EQ(out[i], expected[i]) << "pooling " << static_cast<int>(pooling) << " elem " << i;
    }
  }
}

// fp32 table values -> fp16 output via output_type. Verifies the output dtype
// conversion path; compares with fp16 rounding tolerance.
template <typename KeyType>
void test_pooling_fp16_output(HostLayerTest<KeyType>* t) {
  const size_t elements = t->data_elements();
  const auto keys = make_keys<KeyType>({0, 1, 2, 5});
  const KeyType hotness = static_cast<KeyType>(2);
  const auto bag_offsets = make_keys<KeyType>({0, 2, 4});
  const int64_t num_bags = 2;

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Mean;
  pp.sparse_type = SparseType_t::Fixed;
  pp.key_indices = &hotness;
  pp.num_key_indices = 1;
  pp.output_type = DataType_t::Float16;

  const int64_t out_stride = static_cast<int64_t>(elements * sizeof(__half));
  std::vector<__half> out(static_cast<size_t>(num_bags) * elements, __float2half(0.0f));
  t->layer_->lookup(t->ctx_, static_cast<int64_t>(keys.size()), keys.data(), out.data(), out_stride,
                    /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

  const auto expected = reference_pool(t, keys, bag_offsets, PoolingType_t::Mean, {});
  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); i++) {
    const float got = __half2float(out[i]);
    // fp16 has ~3 decimal digits; values are ~1e4 so allow a relative tolerance.
    EXPECT_NEAR(got, expected[i], std::abs(expected[i]) * 1e-2f + 1.0f) << "elem " << i;
  }
}

// Output stride must be large enough for the effective output dtype, not just the
// table dtype. Here fp32 table rows are pooled into fp16 rows and the stride is
// one byte too small for a full fp16 output row.
template <typename KeyType>
void test_pooling_output_stride_too_small(HostLayerTest<KeyType>* t) {
  const size_t elements = t->data_elements();
  const auto keys = make_keys<KeyType>({0, 1, 2, 5});
  const KeyType hotness = static_cast<KeyType>(2);
  const int64_t num_bags = 2;

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Mean;
  pp.sparse_type = SparseType_t::Fixed;
  pp.key_indices = &hotness;
  pp.num_key_indices = 1;
  pp.output_type = DataType_t::Float16;

  const int64_t min_out_stride = static_cast<int64_t>(elements * sizeof(__half));
  const int64_t short_stride = min_out_stride - 1;
  std::vector<uint8_t> out(static_cast<size_t>(num_bags * min_out_stride), 0);
  EXPECT_THROW(t->layer_->lookup(t->ctx_, static_cast<int64_t>(keys.size()), keys.data(),
                                 out.data(), short_stride, /*hitmask=*/nullptr, &pp,
                                 /*hitrates=*/nullptr),
               Exception);
}

// --------------------- quantized-input (QInt8/QUint8 Rowwise) pooling ----------

// Build a host table whose rows are per-row quantized (symmetric int8:
// [int8 values][scale], dequant value*scale; affine uint8: [uint8 values][scale][offset],
// dequant value*scale + offset) and verify the pooling op dequantizes before combining.
// Scale/offset are chosen exactly representable in fp16 so the fp32-output path matches
// exactly; the reference reads back the stored metadata to account for fp16 rounding.
template <typename KeyType>
void test_quant_pooling(DataType_t qtype) {
  const int64_t num_values = 16;
  const int64_t num_rows = 32;
  const int64_t row_stride = num_values + quant_rowwise_meta_bytes(qtype);
  const bool uint_quant =
      (qtype == DataType_t::QUint8RowwiseF32 || qtype == DataType_t::QUint8RowwiseF16);
  const bool scale_f16 =
      (qtype == DataType_t::QInt8RowwiseF16 || qtype == DataType_t::QUint8RowwiseF16);

  int8_t* buf = nullptr;
  ASSERT_EQ(cudaMallocHost(&buf, static_cast<size_t>(row_stride * num_rows)), cudaSuccess);

  // deq[r][e] is the dequantized value the layer should produce for that element.
  std::vector<std::vector<float>> deq(static_cast<size_t>(num_rows),
                                      std::vector<float>(static_cast<size_t>(num_values)));
  for (int64_t r = 0; r < num_rows; r++) {
    int8_t* row = buf + r * row_stride;
    for (int64_t e = 0; e < num_values; e++) {
      if (uint_quant) {
        reinterpret_cast<uint8_t*>(row)[e] = static_cast<uint8_t>((r * 7 + e * 3) % 251);  // [0,250]
      } else {
        row[e] = static_cast<int8_t>(((r * 5 + e * 3) % 21) - 10);  // [-10, 10]
      }
    }
    const float s = 0.5f * static_cast<float>(1 + (r % 4));  // 0.5 .. 2.0 (exact in fp16)
    const float o = uint_quant ? static_cast<float>((r % 7) - 3) : 0.0f;  // affine offset only
    int8_t* meta = row + num_values;
    float scale, offset = 0.0f;
    if (!scale_f16) {
      reinterpret_cast<float*>(meta)[0] = s;
      scale = reinterpret_cast<float*>(meta)[0];
      if (uint_quant) {
        reinterpret_cast<float*>(meta)[1] = o;
        offset = reinterpret_cast<float*>(meta)[1];
      }
    } else {
      reinterpret_cast<__half*>(meta)[0] = __float2half(s);
      scale = __half2float(reinterpret_cast<__half*>(meta)[0]);
      if (uint_quant) {
        reinterpret_cast<__half*>(meta)[1] = __float2half(o);
        offset = __half2float(reinterpret_cast<__half*>(meta)[1]);
      }
    }
    for (int64_t e = 0; e < num_values; e++) {
      const float base = uint_quant ? static_cast<float>(reinterpret_cast<uint8_t*>(row)[e])
                                    : static_cast<float>(row[e]);
      deq[static_cast<size_t>(r)][static_cast<size_t>(e)] = base * scale + offset;
    }
  }

  LinearHostTableConfig cfg;
  cfg.value_dtype = qtype;
  cfg.max_threads = 16;
  cfg.max_value_size = row_stride;
  cfg.emb_table = buf;
  auto table = std::make_shared<LinearHostTable<KeyType>>(cfg);
  typename HostEmbeddingLayer<KeyType>::Config lcfg;
  lcfg.layer_name = "host_layer_quant";
  auto layer = std::make_shared<HostEmbeddingLayer<KeyType>>(lcfg, table);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  const auto keys = make_keys<KeyType>({0, 1, 2, 5, 6, 7, 10, 11, 12, 20, 21, 22});
  const KeyType hotness = static_cast<KeyType>(3);
  const int64_t num_bags = static_cast<int64_t>(keys.size()) / hotness;
  std::vector<float> weights(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    weights[i] = 0.5f * static_cast<float>(i % 4) + 0.5f;
  }

  for (const auto pooling : {PoolingType_t::Sum, PoolingType_t::Mean, PoolingType_t::WeightedSum,
                             PoolingType_t::WeightedMean}) {
    const bool weighted =
        (pooling == PoolingType_t::WeightedSum) || (pooling == PoolingType_t::WeightedMean);
    const bool mean =
        (pooling == PoolingType_t::Mean) || (pooling == PoolingType_t::WeightedMean);
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling;
    pp.sparse_type = SparseType_t::Fixed;
    pp.key_indices = &hotness;
    pp.num_key_indices = 1;
    if (weighted) {
      pp.sparse_weights = weights.data();
      pp.weight_type = DataType_t::Float32;
    }

    std::vector<float> out(static_cast<size_t>(num_bags * num_values), 0.0f);
    const int64_t out_stride = num_values * static_cast<int64_t>(sizeof(float));
    layer->lookup(ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(), out_stride,
                  /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

    for (int64_t b = 0; b < num_bags; b++) {
      // WeightedMean divides by the sum of weights; Mean by the element count.
      float inv = 1.0f;
      if (mean) {
        float denom = static_cast<float>(hotness);
        if (weighted) {
          denom = 0.0f;
          for (int64_t h = 0; h < hotness; h++) {
            denom += weights[static_cast<size_t>(b * hotness + h)];
          }
        }
        inv = (denom != 0.0f) ? 1.0f / denom : 0.0f;
      }
      for (int64_t e = 0; e < num_values; e++) {
        float acc = 0.0f;
        for (int64_t h = 0; h < hotness; h++) {
          const size_t k = static_cast<size_t>(b * hotness + h);
          float v = deq[static_cast<size_t>(keys[k])][static_cast<size_t>(e)];
          if (weighted) {
            v *= weights[k];
          }
          acc += v;
        }
        const float expected = acc * inv;
        EXPECT_FLOAT_EQ(out[static_cast<size_t>(b * num_values + e)], expected)
            << "qtype " << static_cast<int>(dtype_id(qtype)) << " pooling "
            << static_cast<int>(pooling) << " bag " << b << " elem " << e;
      }
    }
  }

  ctx.reset();
  layer.reset();
  table.reset();
  cudaFreeHost(buf);
}

TEST(HostLayerQuantInt32, pooling_qint8_f32) {
  test_quant_pooling<int32_t>(DataType_t::QInt8RowwiseF32);
}
TEST(HostLayerQuantInt32, pooling_qint8_f16) {
  test_quant_pooling<int32_t>(DataType_t::QInt8RowwiseF16);
}
TEST(HostLayerQuantInt64, pooling_qint8_f32) {
  test_quant_pooling<int64_t>(DataType_t::QInt8RowwiseF32);
}
TEST(HostLayerQuantInt64, pooling_qint8_f16) {
  test_quant_pooling<int64_t>(DataType_t::QInt8RowwiseF16);
}
TEST(HostLayerQuantInt32, pooling_quint8_f32) {
  test_quant_pooling<int32_t>(DataType_t::QUint8RowwiseF32);
}
TEST(HostLayerQuantInt32, pooling_quint8_f16) {
  test_quant_pooling<int32_t>(DataType_t::QUint8RowwiseF16);
}
TEST(HostLayerQuantInt64, pooling_quint8_f32) {
  test_quant_pooling<int64_t>(DataType_t::QUint8RowwiseF32);
}
TEST(HostLayerQuantInt64, pooling_quint8_f16) {
  test_quant_pooling<int64_t>(DataType_t::QUint8RowwiseF16);
}

// ------------------- generic dtype/pooling/sparse combinations -----------------

// Row stride in bytes for a value table of the given input dtype.
inline int64_t pooling_row_stride(DataType_t in_dtype, int64_t num_values) {
  if (is_quant_rowwise(in_dtype)) {
    return num_values + quant_rowwise_meta_bytes(in_dtype);
  }
  const int64_t elem = (in_dtype == DataType_t::Float16) ? static_cast<int64_t>(sizeof(__half))
                                                         : static_cast<int64_t>(sizeof(float));
  return num_values * elem;
}

// Fill `buf` (size row_stride*num_rows) with rows of the given input dtype and
// record the effective (stored / dequantized) value of each element in val[r][e].
// Input values / scales are chosen exactly representable in fp16 and the recorded
// value is read back from the stored bytes, so an fp32-output path matches exactly.
// Quant layout matches the kernel: symmetric int8 stores [scale] (dequant value*scale);
// affine uint8 stores [scale][offset] (dequant value*scale + offset).
inline void fill_value_table(DataType_t in_dtype, int64_t num_values, int64_t num_rows,
                             int8_t* buf, int64_t row_stride,
                             std::vector<std::vector<float>>& val) {
  const bool quant = is_quant_rowwise(in_dtype);
  const bool in_half = (in_dtype == DataType_t::Float16);
  const bool uint_quant =
      (in_dtype == DataType_t::QUint8RowwiseF32 || in_dtype == DataType_t::QUint8RowwiseF16);
  const bool scale_f16 =
      (in_dtype == DataType_t::QInt8RowwiseF16 || in_dtype == DataType_t::QUint8RowwiseF16);
  val.assign(static_cast<size_t>(num_rows), std::vector<float>(static_cast<size_t>(num_values)));
  for (int64_t r = 0; r < num_rows; r++) {
    int8_t* row = buf + r * row_stride;
    if (quant) {
      for (int64_t e = 0; e < num_values; e++) {
        if (uint_quant) {
          reinterpret_cast<uint8_t*>(row)[e] = static_cast<uint8_t>((r * 7 + e * 3) % 251);  // [0,250]
        } else {
          row[e] = static_cast<int8_t>(((r * 5 + e * 3) % 21) - 10);  // [-10, 10]
        }
      }
      const float s = 0.5f * static_cast<float>(1 + (r % 4));  // 0.5 .. 2.0 (exact in fp16)
      // Affine uint8 carries an offset; symmetric int8 does not.
      const float o = uint_quant ? static_cast<float>((r % 7) - 3) : 0.0f;  // small ints
      int8_t* meta = row + num_values;
      float scale, offset = 0.0f;
      if (!scale_f16) {
        reinterpret_cast<float*>(meta)[0] = s;
        scale = reinterpret_cast<float*>(meta)[0];
        if (uint_quant) {
          reinterpret_cast<float*>(meta)[1] = o;
          offset = reinterpret_cast<float*>(meta)[1];
        }
      } else {
        reinterpret_cast<__half*>(meta)[0] = __float2half(s);
        scale = __half2float(reinterpret_cast<__half*>(meta)[0]);
        if (uint_quant) {
          reinterpret_cast<__half*>(meta)[1] = __float2half(o);
          offset = __half2float(reinterpret_cast<__half*>(meta)[1]);
        }
      }
      for (int64_t e = 0; e < num_values; e++) {
        const float base = uint_quant ? static_cast<float>(reinterpret_cast<uint8_t*>(row)[e])
                                      : static_cast<float>(row[e]);
        val[static_cast<size_t>(r)][static_cast<size_t>(e)] = base * scale + offset;
      }
    } else {
      for (int64_t e = 0; e < num_values; e++) {
        const float x = static_cast<float>(((r * 5 + e * 3) % 21) - 10);  // [-10, 10]
        if (in_half) {
          reinterpret_cast<__half*>(row)[e] = __float2half(x);
          val[static_cast<size_t>(r)][static_cast<size_t>(e)] =
              __half2float(reinterpret_cast<__half*>(row)[e]);
        } else {
          reinterpret_cast<float*>(row)[e] = x;
          val[static_cast<size_t>(r)][static_cast<size_t>(e)] = x;
        }
      }
    }
  }
}


// Self-contained pooling test covering any combination of:
//   - input table dtype: Float32, Float16, QInt8Rowwise{F32,F16}, QUint8Rowwise{F32,F16}
//   - output dtype:      Float32, Float16
//   - weight dtype:      Float32, Float16 (only used for weighted pooling)
//   - sparse layout:     Fixed (hotness 3) or CSR (variable bag sizes)
// Runs all four pooling types. Input values / scales / weights are chosen exactly
// representable in fp16, and the reference reads back the stored bytes, so the
// fp32-output path matches exactly; the fp16-output path uses a rounding tolerance.
template <typename KeyType>
void test_pooling_combo(DataType_t in_dtype, DataType_t out_dtype, DataType_t weight_dtype,
                        SparseType_t sparse) {
  const bool out_half = (out_dtype == DataType_t::Float16);
  const bool w_half = (weight_dtype == DataType_t::Float16);

  const int64_t num_values = 16;
  const int64_t num_rows = 32;
  const int64_t row_stride = pooling_row_stride(in_dtype, num_values);

  int8_t* buf = nullptr;
  ASSERT_EQ(cudaMallocHost(&buf, static_cast<size_t>(row_stride * num_rows)), cudaSuccess);

  // val[r][e] is the effective (dequantized / stored) value the layer should see.
  std::vector<std::vector<float>> val;
  fill_value_table(in_dtype, num_values, num_rows, buf, row_stride, val);

  LinearHostTableConfig cfg;
  cfg.value_dtype = in_dtype;
  cfg.max_threads = 16;
  cfg.max_value_size = row_stride;
  cfg.emb_table = buf;
  auto table = std::make_shared<LinearHostTable<KeyType>>(cfg);
  typename HostEmbeddingLayer<KeyType>::Config lcfg;
  lcfg.layer_name = "host_layer_combo";
  auto layer = std::make_shared<HostEmbeddingLayer<KeyType>>(lcfg, table);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  // Key layout + bag offsets per sparse type. bag_off holds the reference bag
  // boundaries and, for CSR, doubles as the key_indices (which must be of the
  // same type as the layer's key type).
  const KeyType hotness = static_cast<KeyType>(3);
  std::vector<KeyType> keys;
  std::vector<KeyType> bag_off;
  if (sparse == SparseType_t::Fixed) {
    keys = make_keys<KeyType>({0, 1, 2, 5, 6, 7, 10, 11, 12, 20, 21, 22});
    for (int64_t b = 0; b <= static_cast<int64_t>(keys.size()) / hotness; b++) {
      bag_off.push_back(static_cast<KeyType>(b * hotness));
    }
  } else {  // CSR: bags of sizes {2, 3, 3}
    keys = make_keys<KeyType>({0, 1, 2, 5, 6, 7, 10, 11});
    bag_off = make_keys<KeyType>({0, 2, 5, 8});
  }
  const int64_t num_bags = static_cast<int64_t>(bag_off.size()) - 1;

  // Weights stored in the requested weight dtype; reference reads them back.
  std::vector<float> wf(keys.size());
  std::vector<__half> wh(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    const float w = 0.5f * static_cast<float>(i % 4) + 0.5f;  // {0.5,1.0,1.5,2.0}
    if (w_half) {
      wh[i] = __float2half(w);
      wf[i] = __half2float(wh[i]);
    } else {
      wf[i] = w;
    }
  }

  const int64_t out_stride =
      num_values * static_cast<int64_t>(out_half ? sizeof(__half) : sizeof(float));

  for (const auto pooling : {PoolingType_t::Sum, PoolingType_t::Mean, PoolingType_t::WeightedSum,
                             PoolingType_t::WeightedMean}) {
    const bool weighted =
        (pooling == PoolingType_t::WeightedSum) || (pooling == PoolingType_t::WeightedMean);
    const bool mean =
        (pooling == PoolingType_t::Mean) || (pooling == PoolingType_t::WeightedMean);

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling;
    pp.sparse_type = sparse;
    pp.output_type = out_dtype;
    if (sparse == SparseType_t::Fixed) {
      pp.key_indices = &hotness;
      pp.num_key_indices = 1;
    } else {
      pp.key_indices = bag_off.data();
      pp.num_key_indices = static_cast<int64_t>(bag_off.size());
    }
    if (weighted) {
      pp.sparse_weights = w_half ? static_cast<const void*>(wh.data())
                                 : static_cast<const void*>(wf.data());
      pp.weight_type = weight_dtype;
    }

    std::vector<uint8_t> out(static_cast<size_t>(num_bags * out_stride), 0);
    layer->lookup(ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(), out_stride,
                  /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

    for (int64_t b = 0; b < num_bags; b++) {
      const int64_t start = bag_off[static_cast<size_t>(b)];
      const int64_t end = bag_off[static_cast<size_t>(b) + 1];
      const int64_t count = end - start;
      // WeightedMean divides by the sum of weights; Mean by the element count.
      float inv = 1.0f;
      if (mean) {
        float denom = static_cast<float>(count);
        if (weighted) {
          denom = 0.0f;
          for (int64_t k = start; k < end; k++) {
            denom += wf[static_cast<size_t>(k)];
          }
        }
        inv = (denom != 0.0f) ? 1.0f / denom : 0.0f;
      }
      for (int64_t e = 0; e < num_values; e++) {
        float acc = 0.0f;
        for (int64_t k = start; k < end; k++) {
          float v = val[static_cast<size_t>(keys[static_cast<size_t>(k)])][static_cast<size_t>(e)];
          if (weighted) {
            v *= wf[static_cast<size_t>(k)];
          }
          acc += v;
        }
        const float expected = acc * inv;
        const size_t idx = static_cast<size_t>(b * num_values + e);
        const float got = out_half ? __half2float(reinterpret_cast<__half*>(out.data())[idx])
                                   : reinterpret_cast<float*>(out.data())[idx];
        if (out_half) {
          EXPECT_NEAR(got, expected, std::abs(expected) * 1e-2f + 0.05f)
              << "pooling " << static_cast<int>(pooling) << " bag " << b << " elem " << e;
        } else {
          EXPECT_FLOAT_EQ(got, expected)
              << "pooling " << static_cast<int>(pooling) << " bag " << b << " elem " << e;
        }
      }
    }
  }

  ctx.reset();
  layer.reset();
  table.reset();
  cudaFreeHost(buf);
}

#define COMBO_TEST(name, in_dt, out_dt, w_dt, sparse)                        \
  TEST(HostLayerComboInt32, name) {                                          \
    test_pooling_combo<int32_t>(in_dt, out_dt, w_dt, sparse);                \
  }                                                                          \
  TEST(HostLayerComboInt64, name) {                                          \
    test_pooling_combo<int64_t>(in_dt, out_dt, w_dt, sparse);                \
  }

// fp16 input (was previously untested — fixture only stored fp32).
COMBO_TEST(fp16_in_fp32_out_fixed, DataType_t::Float16, DataType_t::Float32, DataType_t::Float32,
           SparseType_t::Fixed)
COMBO_TEST(fp16_in_fp32_out_csr, DataType_t::Float16, DataType_t::Float32, DataType_t::Float32,
           SparseType_t::CSR)
COMBO_TEST(fp16_in_fp16_out_csr, DataType_t::Float16, DataType_t::Float16, DataType_t::Float32,
           SparseType_t::CSR)
// fp16 output for CSR / float input.
COMBO_TEST(fp32_in_fp16_out_csr, DataType_t::Float32, DataType_t::Float16, DataType_t::Float32,
           SparseType_t::CSR)
// fp16 weights.
COMBO_TEST(fp32_in_fp16_weights_fixed, DataType_t::Float32, DataType_t::Float32, DataType_t::Float16,
           SparseType_t::Fixed)
COMBO_TEST(fp16_in_fp16_weights_csr, DataType_t::Float16, DataType_t::Float32, DataType_t::Float16,
           SparseType_t::CSR)
// Symmetric int8 input gaps: CSR layout and fp16 output (Fixed/fp32 already covered above).
COMBO_TEST(qint8_f32_fp16_out_fixed, DataType_t::QInt8RowwiseF32, DataType_t::Float16,
           DataType_t::Float32, SparseType_t::Fixed)
COMBO_TEST(qint8_f32_fp32_out_csr, DataType_t::QInt8RowwiseF32, DataType_t::Float32,
           DataType_t::Float32, SparseType_t::CSR)
COMBO_TEST(qint8_f16_fp16_out_csr_fp16w, DataType_t::QInt8RowwiseF16, DataType_t::Float16,
           DataType_t::Float16, SparseType_t::CSR)
// Affine uint8 input: CSR and fp16-output paths (Fixed/fp32 covered by test_quant_pooling).
COMBO_TEST(quint8_f32_fp16_out_fixed, DataType_t::QUint8RowwiseF32, DataType_t::Float16,
           DataType_t::Float32, SparseType_t::Fixed)
COMBO_TEST(quint8_f32_fp32_out_csr, DataType_t::QUint8RowwiseF32, DataType_t::Float32,
           DataType_t::Float32, SparseType_t::CSR)
COMBO_TEST(quint8_f16_fp16_out_csr_fp16w, DataType_t::QUint8RowwiseF16, DataType_t::Float16,
           DataType_t::Float16, SparseType_t::CSR)

#undef COMBO_TEST

// Concatenate = no arithmetic reduction: each key produces one output row, only
// converting/dequantizing the raw gathered data to the output type. Verifies the
// pure data-conversion path (notably QInt8Rowwise -> fp16/fp32). The sparse layout
// and weights are ignored, and key_indices may be null.
template <typename KeyType>
void test_pooling_concat_convert(DataType_t in_dtype, DataType_t out_dtype) {
  const bool out_half = (out_dtype == DataType_t::Float16);
  const int64_t num_values = 16;
  const int64_t num_rows = 32;
  const int64_t row_stride = pooling_row_stride(in_dtype, num_values);

  int8_t* buf = nullptr;
  ASSERT_EQ(cudaMallocHost(&buf, static_cast<size_t>(row_stride * num_rows)), cudaSuccess);
  std::vector<std::vector<float>> val;
  fill_value_table(in_dtype, num_values, num_rows, buf, row_stride, val);

  LinearHostTableConfig cfg;
  cfg.value_dtype = in_dtype;
  cfg.max_threads = 16;
  cfg.max_value_size = row_stride;
  cfg.emb_table = buf;
  auto table = std::make_shared<LinearHostTable<KeyType>>(cfg);
  typename HostEmbeddingLayer<KeyType>::Config lcfg;
  lcfg.layer_name = "host_layer_concat";
  auto layer = std::make_shared<HostEmbeddingLayer<KeyType>>(lcfg, table);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  const auto keys = make_keys<KeyType>({0, 1, 2, 5, 6, 7, 10, 11, 12, 20, 21, 22});

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Concatenate;
  pp.output_type = out_dtype;
  // key_indices / sparse_type / weights are intentionally left unset — Concatenate
  // ignores them.

  const int64_t out_stride =
      num_values * static_cast<int64_t>(out_half ? sizeof(__half) : sizeof(float));
  std::vector<uint8_t> out(static_cast<size_t>(static_cast<int64_t>(keys.size()) * out_stride), 0);
  layer->lookup(ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(), out_stride,
                /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

  // One output row per key, each element just converted from the (dequantized) input.
  for (size_t k = 0; k < keys.size(); k++) {
    for (int64_t e = 0; e < num_values; e++) {
      const float expected = val[static_cast<size_t>(keys[k])][static_cast<size_t>(e)];
      const size_t idx = k * static_cast<size_t>(num_values) + static_cast<size_t>(e);
      const float got = out_half ? __half2float(reinterpret_cast<__half*>(out.data())[idx])
                                 : reinterpret_cast<float*>(out.data())[idx];
      if (out_half) {
        EXPECT_NEAR(got, expected, std::abs(expected) * 1e-2f + 0.05f)
            << "key " << k << " elem " << e;
      } else {
        EXPECT_FLOAT_EQ(got, expected) << "key " << k << " elem " << e;
      }
    }
  }

  ctx.reset();
  layer.reset();
  table.reset();
  cudaFreeHost(buf);
}

#define CONCAT_TEST(name, in_dt, out_dt)                              \
  TEST(HostLayerConcatInt32, name) {                                  \
    test_pooling_concat_convert<int32_t>(in_dt, out_dt);             \
  }                                                                   \
  TEST(HostLayerConcatInt64, name) {                                  \
    test_pooling_concat_convert<int64_t>(in_dt, out_dt);             \
  }

// int8 (symmetric) -> fp16/fp32 pure conversion (no arithmetic pooling).
CONCAT_TEST(qint8_f32_to_fp32, DataType_t::QInt8RowwiseF32, DataType_t::Float32)
CONCAT_TEST(qint8_f32_to_fp16, DataType_t::QInt8RowwiseF32, DataType_t::Float16)
CONCAT_TEST(qint8_f16_to_fp32, DataType_t::QInt8RowwiseF16, DataType_t::Float32)
CONCAT_TEST(qint8_f16_to_fp16, DataType_t::QInt8RowwiseF16, DataType_t::Float16)
// uint8 (affine) -> fp16/fp32 pure conversion (dequant value*scale + offset, no pooling).
CONCAT_TEST(quint8_f32_to_fp32, DataType_t::QUint8RowwiseF32, DataType_t::Float32)
CONCAT_TEST(quint8_f32_to_fp16, DataType_t::QUint8RowwiseF32, DataType_t::Float16)
CONCAT_TEST(quint8_f16_to_fp32, DataType_t::QUint8RowwiseF16, DataType_t::Float32)
CONCAT_TEST(quint8_f16_to_fp16, DataType_t::QUint8RowwiseF16, DataType_t::Float16)
// float dtype conversions via concat.
CONCAT_TEST(fp32_to_fp16, DataType_t::Float32, DataType_t::Float16)
CONCAT_TEST(fp16_to_fp32, DataType_t::Float16, DataType_t::Float32)

#undef CONCAT_TEST

// WeightedMean with per-bag weights summing to zero must not divide by zero: the
// kernel should emit a finite zero row instead. Exercises both the float and the
// quantized kernels via in_dtype.
template <typename KeyType>
void test_weighted_mean_zero_weight_sum(DataType_t in_dtype) {
  const int64_t num_values = 8;
  const int64_t num_rows = 16;
  const int64_t row_stride = pooling_row_stride(in_dtype, num_values);

  int8_t* buf = nullptr;
  ASSERT_EQ(cudaMallocHost(&buf, static_cast<size_t>(row_stride * num_rows)), cudaSuccess);
  std::vector<std::vector<float>> val;
  fill_value_table(in_dtype, num_values, num_rows, buf, row_stride, val);

  LinearHostTableConfig cfg;
  cfg.value_dtype = in_dtype;
  cfg.max_threads = 8;
  cfg.max_value_size = row_stride;
  cfg.emb_table = buf;
  auto table = std::make_shared<LinearHostTable<KeyType>>(cfg);
  typename HostEmbeddingLayer<KeyType>::Config lcfg;
  lcfg.layer_name = "host_layer_zero_w";
  auto layer = std::make_shared<HostEmbeddingLayer<KeyType>>(lcfg, table);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  // 2 bags of hotness 2; each bag's weights sum to zero.
  const auto keys = make_keys<KeyType>({0, 1, 2, 3});
  const KeyType hotness = static_cast<KeyType>(2);
  const int64_t num_bags = 2;
  const std::vector<float> weights{1.0f, -1.0f, 2.0f, -2.0f};

  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::WeightedMean;
  pp.sparse_type = SparseType_t::Fixed;
  pp.key_indices = &hotness;
  pp.num_key_indices = 1;
  pp.sparse_weights = weights.data();
  pp.weight_type = DataType_t::Float32;

  std::vector<float> out(static_cast<size_t>(num_bags * num_values), 123.0f);
  layer->lookup(ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                num_values * static_cast<int64_t>(sizeof(float)), /*hitmask=*/nullptr, &pp,
                /*hitrates=*/nullptr);

  for (size_t i = 0; i < out.size(); i++) {
    EXPECT_TRUE(std::isfinite(out[i])) << "non-finite output at " << i;
    EXPECT_FLOAT_EQ(out[i], 0.0f) << "expected zero row for zero weight sum at " << i;
  }

  ctx.reset();
  layer.reset();
  table.reset();
  cudaFreeHost(buf);
}

TEST(HostLayerWeightedMeanInt32, zero_weight_sum_fp32) {
  test_weighted_mean_zero_weight_sum<int32_t>(DataType_t::Float32);
}
TEST(HostLayerWeightedMeanInt32, zero_weight_sum_qint8) {
  test_weighted_mean_zero_weight_sum<int32_t>(DataType_t::QInt8RowwiseF32);
}
TEST(HostLayerWeightedMeanInt32, zero_weight_sum_quint8) {
  test_weighted_mean_zero_weight_sum<int32_t>(DataType_t::QUint8RowwiseF32);
}
TEST(HostLayerWeightedMeanInt64, zero_weight_sum_fp32) {
  test_weighted_mean_zero_weight_sum<int64_t>(DataType_t::Float32);
}
TEST(HostLayerWeightedMeanInt64, zero_weight_sum_qint8) {
  test_weighted_mean_zero_weight_sum<int64_t>(DataType_t::QInt8RowwiseF32);
}
TEST(HostLayerWeightedMeanInt64, zero_weight_sum_quint8) {
  test_weighted_mean_zero_weight_sum<int64_t>(DataType_t::QUint8RowwiseF32);
}

// -------------------------- test instantiation --------------------------------

const HostLayerTestParams kHostLayerParams[] = {
    {/*row_size_bytes=*/128, /*num_rows=*/10000, DataType_t::Float32},
};

using HostLayerFixture_INT32 = HostLayerTest<int32_t>;
using HostLayerFixture_INT64 = HostLayerTest<int64_t>;

#define HOST_LAYER_TEST(name, fn)               \
  TEST_P(HostLayerFixture_INT32, name) { fn(this); } \
  TEST_P(HostLayerFixture_INT64, name) { fn(this); }

HOST_LAYER_TEST(lookup_no_hitmask, test_lookup_no_hitmask)
HOST_LAYER_TEST(lookup_with_hitmask, test_lookup_with_hitmask)
HOST_LAYER_TEST(lookup_hitmask_multiple_of_64, test_lookup_hitmask_multiple_of_64)
HOST_LAYER_TEST(update_then_lookup, test_update_then_lookup)
HOST_LAYER_TEST(accumulate, test_accumulate)
HOST_LAYER_TEST(lookup_gpu_input, test_lookup_gpu_input)
HOST_LAYER_TEST(default_embedding_no_hitmask, test_default_embedding_no_hitmask)
HOST_LAYER_TEST(noop_modifications, test_noop_modifications)
HOST_LAYER_TEST(pooling_fixed, test_pooling_fixed)
HOST_LAYER_TEST(pooling_csr, test_pooling_csr)
HOST_LAYER_TEST(pooling_fp16_output, test_pooling_fp16_output)
HOST_LAYER_TEST(pooling_output_stride_too_small, test_pooling_output_stride_too_small)

#undef HOST_LAYER_TEST

INSTANTIATE_TEST_SUITE_P(HostLayerInt32, HostLayerFixture_INT32, testing::ValuesIn(kHostLayerParams));
INSTANTIATE_TEST_SUITE_P(HostLayerInt64, HostLayerFixture_INT64, testing::ValuesIn(kHostLayerParams));

}  // namespace nve
