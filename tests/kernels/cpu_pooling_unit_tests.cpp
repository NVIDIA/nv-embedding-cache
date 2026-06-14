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
#include "cpu_ops/cpu_pooling.h"
#include "include/thread_pool.hpp"
#include "include/nve_types.hpp"
#include "mock_host_table.hpp"
#include "emb_layer_utils.hpp"
#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace nve {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Input row stride in bytes for a given dtype and row_elements (value elements).
static int64_t input_row_stride(DataType_t dtype, int64_t row_elements) {
  switch (dtype) {
    case DataType_t::Float32:         return row_elements * static_cast<int64_t>(sizeof(float));
    case DataType_t::Float16:         return row_elements * static_cast<int64_t>(sizeof(__half));
    case DataType_t::QInt8RowwiseF32: return row_elements + static_cast<int64_t>(sizeof(float));
    case DataType_t::QInt8RowwiseF16: return row_elements + static_cast<int64_t>(sizeof(uint16_t));
    case DataType_t::QUint8RowwiseF32:return row_elements + 2 * static_cast<int64_t>(sizeof(float));
    case DataType_t::QUint8RowwiseF16:return row_elements + 2 * static_cast<int64_t>(sizeof(uint16_t));
    default: throw std::runtime_error("Unsupported dtype in input_row_stride");
  }
}

// Output row stride in bytes.
static int64_t output_row_stride(DataType_t out_type, int64_t row_elements) {
  switch (out_type) {
    case DataType_t::Float32: return row_elements * static_cast<int64_t>(sizeof(float));
    case DataType_t::Float16: return row_elements * static_cast<int64_t>(sizeof(__half));
    default: throw std::runtime_error("Unsupported out_type in output_row_stride");
  }
}

// Generate a contiguous embedding table for the given dtype.
// For float/half: random values in (-1, 1).
// For quantized: random int8/uint8 values in [−100,100]/[0,200], scale in (0.01,0.1), offset (affine only).
static std::vector<int8_t> make_table(int64_t num_rows, int64_t row_elements, DataType_t dtype,
                                      int64_t row_stride, uint32_t seed) {
  std::vector<int8_t> table(static_cast<size_t>(num_rows * row_stride), 0);
  std::mt19937 rng(seed);

  if (dtype == DataType_t::Float32 || dtype == DataType_t::Float16) {
    // InitTableRows handles Float32/Float16
    InitTableRows(table.data(), static_cast<uint64_t>(row_stride),
                  0, static_cast<uint64_t>(num_rows), dtype, seed);
    return table;
  }

  // Quantized: fill value bytes then append metadata
  std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
  std::uniform_real_distribution<float> offset_dist(-0.5f, 0.5f);

  const bool is_uint8 = (dtype == DataType_t::QUint8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF16);
  const bool fp32_meta = (dtype == DataType_t::QInt8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF32);

  for (int64_t r = 0; r < num_rows; r++) {
    int8_t* row = table.data() + r * row_stride;
    if (is_uint8) {
      std::uniform_int_distribution<int> val_dist(0, 200);
      for (int64_t e = 0; e < row_elements; e++) {
        reinterpret_cast<uint8_t*>(row)[e] = static_cast<uint8_t>(val_dist(rng));
      }
    } else {
      std::uniform_int_distribution<int> val_dist(-100, 100);
      for (int64_t e = 0; e < row_elements; e++) {
        row[e] = static_cast<int8_t>(val_dist(rng));
      }
    }
    // Write scale
    const float scale = scale_dist(rng);
    int8_t* meta = row + row_elements;
    if (fp32_meta) {
      std::memcpy(meta, &scale, sizeof(float));
      meta += sizeof(float);
    } else {
      const __half h = __float2half(scale);
      std::memcpy(meta, &h, sizeof(__half));
      meta += sizeof(__half);
    }
    // Write offset (affine only)
    if (is_uint8) {
      const float off = offset_dist(rng);
      if (fp32_meta) {
        std::memcpy(meta, &off, sizeof(float));
      } else {
        const __half h = __float2half(off);
        std::memcpy(meta, &h, sizeof(__half));
      }
    }
  }
  return table;
}

// Build a CSR offsets array with exactly num_bags bags and roughly avg_hotness keys each.
// All bags have at least 1 key.
static std::vector<int64_t> make_csr_offsets(int64_t num_bags, int64_t avg_hotness, uint32_t seed) {
  std::mt19937 rng(seed);
  const int64_t lo = std::max<int64_t>(1, avg_hotness / 2);
  const int64_t hi = avg_hotness * 2;
  std::uniform_int_distribution<int64_t> dist(lo, hi);
  std::vector<int64_t> offsets(static_cast<size_t>(num_bags + 1));
  offsets[0] = 0;
  for (int64_t b = 0; b < num_bags; b++) {
    offsets[static_cast<size_t>(b + 1)] = offsets[static_cast<size_t>(b)] + dist(rng);
  }
  return offsets;
}

// Pre-gather: build a contiguous input buffer where row i holds the embedding
// for the i-th key in the key sequence. For Fixed layout the sequence is
// simply 0..total_keys-1 (identity mapping); keys are consecutive indices into
// the table.  For CSR, same: row k in the gathered buffer is table row k.
// (cpu_kernel_pooling operates on pre-gathered rows, so the identity mapping
// is the minimal correct setup.)
static std::vector<int8_t> gather_table(const std::vector<int8_t>& table, int64_t total_keys,
                                        int64_t row_stride) {
  // Identity gather: key k → table row k
  return std::vector<int8_t>(table.begin(),
                              table.begin() + total_keys * row_stride);
}

// Tolerance for EXPECT_NEAR based on in/out types.
// fp16 output uses 1e-2: the kernel uses multiply-by-inverse for mean/weighted-mean while the
// reference divides by the denominator; these differ by ~1 fp16 ULP (~0.004 for values of ~5).
static float get_tolerance(DataType_t in_type, DataType_t out_type) {
  if (out_type == DataType_t::Float16 ||
      in_type == DataType_t::Float16 ||
      in_type == DataType_t::QInt8RowwiseF16 ||
      in_type == DataType_t::QUint8RowwiseF16) {
    return 1e-2f;
  }
  return 1e-5f;
}

// Compare two output buffers element-wise.
static void compare_outputs(const int8_t* ref, const int8_t* got, int64_t num_elems,
                             DataType_t out_type, float tol, const std::string& ctx) {
  for (int64_t i = 0; i < num_elems; i++) {
    float ref_val, got_val;
    if (out_type == DataType_t::Float32) {
      ref_val = reinterpret_cast<const float*>(ref)[i];
      got_val = reinterpret_cast<const float*>(got)[i];
    } else {
      ref_val = __half2float(reinterpret_cast<const __half*>(ref)[i]);
      got_val = __half2float(reinterpret_cast<const __half*>(got)[i]);
    }
    EXPECT_NEAR(ref_val, got_val, tol) << ctx << " element " << i;
  }
}

// ---------------------------------------------------------------------------
// Parameterized test
// ---------------------------------------------------------------------------

struct PoolingTestParams {
  int64_t       num_bags;
  int64_t       row_elements;    // value elements per row (not bytes)
  SparseType_t  sparse_type;
  int64_t       avg_hotness;
  PoolingType_t pooling_type;
  DataType_t    in_type;
  DataType_t    out_type;
  DataType_t    weight_type;  // used only for WeightedSum/WeightedMean
  int64_t       num_workers;
  uint32_t      seed;
};

class CpuPoolingTest : public ::testing::TestWithParam<PoolingTestParams> {
 protected:
  void SetUp() override { thread_pool_ = default_thread_pool(); }

  void run_test() {
    const auto& p = GetParam();

    const int64_t in_stride  = input_row_stride(p.in_type, p.row_elements);
    const int64_t out_stride = output_row_stride(p.out_type, p.row_elements);
    const bool weighted = (p.pooling_type == PoolingType_t::WeightedSum ||
                           p.pooling_type == PoolingType_t::WeightedMean);

    // Build CSR offsets or derive total_keys for Fixed
    std::vector<int64_t> csr_offsets;
    int64_t total_keys;
    if (p.sparse_type == SparseType_t::CSR) {
      csr_offsets = make_csr_offsets(p.num_bags, p.avg_hotness, p.seed);
      total_keys = csr_offsets.back();
    } else {
      total_keys = p.num_bags * p.avg_hotness;
    }

    // Build embedding table and pre-gather
    const auto table = make_table(total_keys, p.row_elements, p.in_type, in_stride, p.seed + 1);
    const auto gathered = gather_table(table, total_keys, in_stride);

    // Generate per-key weights
    std::vector<int8_t> weights_buf;
    if (weighted) {
      GenerateWeights(weights_buf, static_cast<uint64_t>(total_keys), p.weight_type, p.seed + 2);
    }
    const void* weights_ptr = weighted ? weights_buf.data() : nullptr;

    // Concatenate outputs one row per input key (total_keys); other modes output one per bag.
    const bool concat = (p.pooling_type == PoolingType_t::Concatenate);
    const int64_t out_rows = concat ? total_keys : p.num_bags;

    // Reference: MockHostTable::combine()
    HostTableConfig cfg;
    cfg.max_value_size = in_stride;
    cfg.value_dtype    = p.in_type;
    MockHostTable<int64_t> mock(cfg, /*functional_ref=*/false);

    std::vector<int8_t> ref_out(static_cast<size_t>(out_rows * out_stride), 0);
    if (p.sparse_type == SparseType_t::Fixed) {
      const int64_t key_idx = p.avg_hotness;
      mock.combine(gathered.data(), total_keys, p.pooling_type, p.sparse_type,
                   &key_idx, 1, weights_ptr, p.weight_type, ref_out.data(), p.out_type);
    } else {
      mock.combine(gathered.data(), total_keys, p.pooling_type, p.sparse_type,
                   csr_offsets.data(), static_cast<int64_t>(csr_offsets.size()),
                   weights_ptr, p.weight_type, ref_out.data(), p.out_type);
    }

    // Run cpu_kernel_pooling_dispatch
    std::vector<int8_t> got_out(static_cast<size_t>(out_rows * out_stride), 0);
    const int64_t* offsets_ptr = (p.sparse_type == SparseType_t::CSR) ? csr_offsets.data() : nullptr;
    cpu_kernel_pooling_dispatch(
        thread_pool_, p.num_bags, p.row_elements,
        gathered.data(), in_stride,
        got_out.data(), out_stride,
        p.sparse_type, p.avg_hotness, offsets_ptr,
        p.pooling_type,
        weights_ptr, p.weight_type, p.in_type, p.out_type,
        p.num_workers);

    // Compare
    const float tol = get_tolerance(p.in_type, p.out_type);
    const int64_t total_elems = out_rows * p.row_elements;
    compare_outputs(ref_out.data(), got_out.data(), total_elems, p.out_type, tol,
                    "PoolingVsReference");
  }

  std::shared_ptr<nve::ThreadPool> thread_pool_;
};

TEST_P(CpuPoolingTest, PoolingVsReference) { run_test(); }

// ---------------------------------------------------------------------------
// Parameter generation
// ---------------------------------------------------------------------------

static std::vector<PoolingTestParams> genPoolingCases() {
  std::vector<PoolingTestParams> cases;
  uint32_t seed = 0;

  const std::vector<int64_t>       num_bags_vals     = {1, 16, 512};
  const std::vector<int64_t>       row_elements_vals    = {1, 8, 64};
  const std::vector<int64_t>       hotness_vals      = {1, 4, 16};
  const std::vector<SparseType_t>  sparse_vals       = {SparseType_t::Fixed, SparseType_t::CSR};
  const std::vector<PoolingType_t> pooling_vals      = {PoolingType_t::Sum, PoolingType_t::Mean,
                                                        PoolingType_t::WeightedSum,
                                                        PoolingType_t::WeightedMean,
                                                        PoolingType_t::Concatenate};
  const std::vector<DataType_t>    in_type_vals      = {DataType_t::Float32, DataType_t::Float16,
                                                        DataType_t::QInt8RowwiseF32,
                                                        DataType_t::QUint8RowwiseF32};
  const std::vector<DataType_t>    out_type_vals     = {DataType_t::Float32, DataType_t::Float16};
  const std::vector<int64_t>       worker_vals       = {1, 4};

  for (auto num_bags : num_bags_vals)
  for (auto row_elements : row_elements_vals)
  for (auto avg_hotness : hotness_vals)
  for (auto sparse_type : sparse_vals)
  for (auto pooling_type : pooling_vals)
  for (auto in_type : in_type_vals)
  for (auto out_type : out_type_vals)
  for (auto num_workers : worker_vals) {
    cases.push_back({num_bags, row_elements, sparse_type, avg_hotness,
                     pooling_type, in_type, out_type,
                     DataType_t::Float32,  // weight_type
                     num_workers, seed++});
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    PoolingTests,
    CpuPoolingTest,
    ::testing::ValuesIn(genPoolingCases()));

// ---------------------------------------------------------------------------
// Edge-case tests
// ---------------------------------------------------------------------------

class CpuPoolingEdgeTest : public ::testing::Test {
 protected:
  void SetUp() override { thread_pool_ = default_thread_pool(); }

  // Run the dispatch and the reference, compare outputs. OffsetT selects the
  // CSR offset element type (int32_t or int64_t) to exercise both dispatch
  // instantiations; it defaults to int64_t for existing callers.
  template <typename OffsetT = int64_t>
  void check(int64_t num_bags, int64_t row_elements, DataType_t in_type, DataType_t out_type,
             SparseType_t sparse_type, int64_t fixed_hotness,
             const std::vector<OffsetT>& csr_offsets,
             PoolingType_t pooling_type,
             const std::vector<int8_t>& gathered,
             const void* weights, DataType_t weight_type,
             float tol) {
    const int64_t in_stride  = input_row_stride(in_type, row_elements);
    const int64_t out_stride = output_row_stride(out_type, row_elements);
    const int64_t total_keys = sparse_type == SparseType_t::Fixed
                                   ? num_bags * fixed_hotness
                                   : static_cast<int64_t>(csr_offsets.back());
    const bool concat = (pooling_type == PoolingType_t::Concatenate);
    const int64_t out_rows = concat ? total_keys : num_bags;

    HostTableConfig cfg;
    cfg.max_value_size = in_stride;
    cfg.value_dtype    = in_type;
    // The reference reads key_indices as OffsetT, matching what we pass to the dispatch.
    MockHostTable<OffsetT> mock(cfg, /*functional_ref=*/false);

    std::vector<int8_t> ref_out(static_cast<size_t>(out_rows * out_stride), 0);
    if (sparse_type == SparseType_t::Fixed) {
      const OffsetT fh = static_cast<OffsetT>(fixed_hotness);
      mock.combine(gathered.data(), total_keys, pooling_type, sparse_type,
                   &fh, 1, weights, weight_type, ref_out.data(), out_type);
    } else {
      mock.combine(gathered.data(), total_keys, pooling_type, sparse_type,
                   csr_offsets.data(), static_cast<int64_t>(csr_offsets.size()),
                   weights, weight_type, ref_out.data(), out_type);
    }

    std::vector<int8_t> got_out(static_cast<size_t>(out_rows * out_stride), 0);
    const OffsetT* offsets_ptr = (sparse_type == SparseType_t::CSR) ? csr_offsets.data() : nullptr;
    cpu_kernel_pooling_dispatch(
        thread_pool_, num_bags, row_elements,
        gathered.data(), in_stride, got_out.data(), out_stride,
        sparse_type, fixed_hotness, offsets_ptr,
        pooling_type, weights, weight_type, in_type, out_type,
        /*num_workers=*/4);

    compare_outputs(ref_out.data(), got_out.data(), out_rows * row_elements, out_type, tol,
                    "EdgeCase");
  }

  std::shared_ptr<nve::ThreadPool> thread_pool_;
};

// Empty bags (CSR): bags with zero keys must produce zero output rows.
TEST_F(CpuPoolingEdgeTest, EmptyBagsCsr) {
  const int64_t row_elements = 8;
  const DataType_t in_type = DataType_t::Float32;
  const DataType_t out_type = DataType_t::Float32;
  // 4 bags: sizes 0, 3, 0, 2
  std::vector<int64_t> offsets = {0, 0, 3, 3, 5};
  const int64_t total_keys = 5;
  const int64_t in_stride = input_row_stride(in_type, row_elements);
  auto gathered = make_table(total_keys, row_elements, in_type, in_stride, /*seed=*/42);

  check(4, row_elements, in_type, out_type, SparseType_t::CSR, 0, offsets,
        PoolingType_t::Sum, gathered, nullptr, DataType_t::Float32, 1e-5f);
}

// All-empty bags.
TEST_F(CpuPoolingEdgeTest, AllEmptyBags) {
  const int64_t row_elements = 4;
  const DataType_t in_type = DataType_t::Float32;
  const DataType_t out_type = DataType_t::Float32;
  const int64_t num_bags = 8;
  std::vector<int64_t> offsets(static_cast<size_t>(num_bags + 1), 0);
  const std::vector<int8_t> gathered;  // no keys

  // With no input rows, pass a dummy non-null pointer of zero size.
  const int8_t dummy = 0;
  std::vector<int8_t> got_out(static_cast<size_t>(num_bags * row_elements * sizeof(float)), 0);
  cpu_kernel_pooling_dispatch(
      thread_pool_, num_bags, row_elements,
      &dummy, input_row_stride(in_type, row_elements), got_out.data(),
      output_row_stride(out_type, row_elements),
      SparseType_t::CSR, 0, offsets.data(),
      PoolingType_t::Sum, nullptr, DataType_t::Float32,
      in_type, out_type, /*num_workers=*/1);

  // All output rows must be zero.
  const float* out_f = reinterpret_cast<const float*>(got_out.data());
  for (int64_t i = 0; i < num_bags * row_elements; i++) {
    EXPECT_EQ(out_f[i], 0.f) << "element " << i;
  }
}

// Zero weights with WeightedMean: output must be zero.
TEST_F(CpuPoolingEdgeTest, ZeroWeightsWeightedMean) {
  const int64_t num_bags = 4;
  const int64_t row_elements = 8;
  const int64_t hotness = 3;
  const DataType_t in_type = DataType_t::Float32;
  const DataType_t out_type = DataType_t::Float32;
  const int64_t in_stride = input_row_stride(in_type, row_elements);
  const int64_t total_keys = num_bags * hotness;

  auto gathered = make_table(total_keys, row_elements, in_type, in_stride, /*seed=*/7);
  std::vector<float> zero_weights(static_cast<size_t>(total_keys), 0.f);

  check(num_bags, row_elements, in_type, out_type, SparseType_t::Fixed, hotness,
        std::vector<int64_t>{},
        PoolingType_t::WeightedMean, gathered,
        zero_weights.data(), DataType_t::Float32, 0.f);
}

// Single-key bags (hotness = 1): output must equal the dequantized input row.
TEST_F(CpuPoolingEdgeTest, SingleKeyBagsQuantized) {
  const int64_t num_bags = 8;
  const int64_t row_elements = 16;
  const DataType_t in_type = DataType_t::QInt8RowwiseF32;
  const DataType_t out_type = DataType_t::Float32;
  const int64_t in_stride = input_row_stride(in_type, row_elements);

  auto gathered = make_table(num_bags, row_elements, in_type, in_stride, /*seed=*/99);
  check(num_bags, row_elements, in_type, out_type, SparseType_t::Fixed, /*hotness=*/1,
        std::vector<int64_t>{},
        PoolingType_t::Sum, gathered, nullptr, DataType_t::Float32, 1e-5f);
}

// int32 CSR offsets must produce the same result as the int64 path. This
// exercises the OffsetT=int32_t instantiation of cpu_kernel_pooling_dispatch.
TEST_F(CpuPoolingEdgeTest, Int32OffsetsCsr) {
  const int64_t row_elements = 8;
  const DataType_t in_type = DataType_t::Float32;
  const DataType_t out_type = DataType_t::Float32;
  // 3 bags of sizes 2, 3, 1 -> 6 keys.
  const std::vector<int32_t> offsets = {0, 2, 5, 6};
  const int64_t total_keys = 6;
  const int64_t in_stride = input_row_stride(in_type, row_elements);
  auto gathered = make_table(total_keys, row_elements, in_type, in_stride, /*seed=*/123);

  for (const auto pooling : {PoolingType_t::Sum, PoolingType_t::Mean,
                             PoolingType_t::Concatenate}) {
    check<int32_t>(3, row_elements, in_type, out_type, SparseType_t::CSR, 0, offsets,
                   pooling, gathered, nullptr, DataType_t::Float32, 1e-5f);
  }
}

// int32 CSR offsets with quantized input and weighted pooling.
TEST_F(CpuPoolingEdgeTest, Int32OffsetsCsrQuantizedWeighted) {
  const int64_t row_elements = 16;
  const DataType_t in_type = DataType_t::QInt8RowwiseF32;
  const DataType_t out_type = DataType_t::Float32;
  const std::vector<int32_t> offsets = {0, 3, 4, 7};
  const int64_t total_keys = 7;
  const int64_t in_stride = input_row_stride(in_type, row_elements);
  auto gathered = make_table(total_keys, row_elements, in_type, in_stride, /*seed=*/321);
  std::vector<int8_t> weights_buf;
  GenerateWeights(weights_buf, static_cast<uint64_t>(total_keys), DataType_t::Float32, /*seed=*/9);

  check<int32_t>(3, row_elements, in_type, out_type, SparseType_t::CSR, 0, offsets,
                 PoolingType_t::WeightedSum, gathered, weights_buf.data(),
                 DataType_t::Float32, 1e-5f);
}

// int32 CSR offsets with empty bags.
TEST_F(CpuPoolingEdgeTest, Int32EmptyBagsCsr) {
  const int64_t row_elements = 8;
  const DataType_t in_type = DataType_t::Float32;
  const DataType_t out_type = DataType_t::Float32;
  // 4 bags: sizes 0, 3, 0, 2.
  const std::vector<int32_t> offsets = {0, 0, 3, 3, 5};
  const int64_t total_keys = 5;
  const int64_t in_stride = input_row_stride(in_type, row_elements);
  auto gathered = make_table(total_keys, row_elements, in_type, in_stride, /*seed=*/42);

  check<int32_t>(4, row_elements, in_type, out_type, SparseType_t::CSR, 0, offsets,
                 PoolingType_t::Sum, gathered, nullptr, DataType_t::Float32, 1e-5f);
}

// Invalid out_type throws.
TEST_F(CpuPoolingEdgeTest, InvalidOutTypeThrows) {
  const int64_t row_elements = 4;
  std::vector<int8_t> in_buf(static_cast<size_t>(row_elements * sizeof(float)), 0);
  std::vector<int8_t> out_buf(static_cast<size_t>(row_elements * sizeof(float)), 0);
  const int64_t offsets[2] = {0, 1};
  EXPECT_THROW(
      cpu_kernel_pooling_dispatch(
          thread_pool_, 1, row_elements,
          in_buf.data(), row_elements * static_cast<int64_t>(sizeof(float)),
          out_buf.data(), row_elements * static_cast<int64_t>(sizeof(float)),
          SparseType_t::CSR, 0, offsets,
          PoolingType_t::Sum, nullptr, DataType_t::Float32,
          DataType_t::Float32,
          DataType_t::QInt8RowwiseF32,  // invalid out_type
          1),
      std::exception);
}

// Invalid in_type throws.
TEST_F(CpuPoolingEdgeTest, InvalidInTypeThrows) {
  const int64_t row_elements = 4;
  std::vector<int8_t> in_buf(static_cast<size_t>(row_elements * sizeof(float)), 0);
  std::vector<int8_t> out_buf(static_cast<size_t>(row_elements * sizeof(float)), 0);
  const int64_t offsets[2] = {0, 1};
  EXPECT_THROW(
      cpu_kernel_pooling_dispatch(
          thread_pool_, 1, row_elements,
          in_buf.data(), row_elements * static_cast<int64_t>(sizeof(float)),
          out_buf.data(), row_elements * static_cast<int64_t>(sizeof(float)),
          SparseType_t::CSR, 0, offsets,
          PoolingType_t::Sum, nullptr, DataType_t::Float32,
          DataType_t::Unknown,  // invalid in_type
          DataType_t::Float32,
          1),
      std::exception);
}

}  // namespace nve
