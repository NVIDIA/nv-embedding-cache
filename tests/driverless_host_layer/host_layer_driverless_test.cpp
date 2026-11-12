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

// Driverless host-layer smoke test — the C++ analogue of tests/python_binding/test_host_cpu.py.
//
// Exercises a HostEmbeddingLayer over a LinearHostTable using plain malloc-backed storage
// (std::vector or HostMemBlock, NOT cudaMallocHost) and the default allocator/thread pool. On a
// driverless host GetDefaultAllocator() falls back to malloc and the execution context skips
// cudaStreamSynchronize, so this runs end-to-end with no CUDA driver / GPU. Built into its own
// executable that links only nve-common (driver-free in a NVE_DRIVERLESS_BUILD), so it stays
// loadable on a driverless machine.

#include <gtest/gtest.h>

#include "include/common.hpp"
#include "include/host_embedding_layer.hpp"
#include "include/linear_host_table.hpp"
#include "include/memblock.hpp"
#include "include/nve_types.hpp"
#include "include/serialization.hpp"
#include "tests/host_layer_test_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace nve {
namespace {

using KeyType = int64_t;
constexpr int64_t kRowBytes = 16;  // 4 x float32
constexpr int64_t kNumRows = 256;
constexpr size_t kElems = static_cast<size_t>(kRowBytes) / sizeof(float);

// HostEmbeddingLayer + LinearHostTable over a caller-owned host buffer.
struct HostLayerHarness : nve_test::HostLayerHandles<KeyType> {
  HostLayerHarness(void* emb_table, std::vector<uint8_t> default_embedding = {})
      : nve_test::HostLayerHandles<KeyType>(nve_test::make_host_layer<KeyType>(
            emb_table, kRowBytes, kNumRows, DataType_t::Float32, /*max_threads=*/8,
            "driverless_host_layer", std::move(default_embedding))) {}

  std::vector<float> lookup(const std::vector<KeyType>& keys) {
    std::vector<float> out(keys.size() * kElems, 0.0f);
    layer->lookup(ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(), kRowBytes,
                  /*hitmask=*/nullptr, /*pool_params=*/nullptr, /*hitrates=*/nullptr);
    return out;
  }
};

// Table storage with distinct, recognizable per-element values. Kept in its own base so it is
// constructed before the harness that points at it.
struct BackingStorage {
  std::vector<float> backing;
  BackingStorage() : backing(static_cast<size_t>(kNumRows) * kElems) {
    std::iota(backing.begin(), backing.end(), 1000.0f);
  }
};

struct HostLayerFixture : BackingStorage, HostLayerHarness {
  explicit HostLayerFixture(std::vector<uint8_t> default_embedding = {})
      : BackingStorage(), HostLayerHarness(backing.data(), std::move(default_embedding)) {}
};

TEST(DriverlessHostLayer, LookupReturnsBackingRows) {
  HostLayerFixture f;
  const std::vector<KeyType> keys{0, 5, 42, 200};
  const auto out = f.lookup(keys);

  for (size_t k = 0; k < keys.size(); ++k) {
    for (size_t e = 0; e < kElems; ++e) {
      const float expected = f.backing[static_cast<size_t>(keys[k]) * kElems + e];
      EXPECT_FLOAT_EQ(out[k * kElems + e], expected) << "key=" << keys[k] << " elem=" << e;
    }
  }
}

TEST(DriverlessHostLayer, UpdateThenLookup) {
  HostLayerFixture f;
  constexpr KeyType key = 77;
  std::vector<float> row(kElems);
  std::iota(row.begin(), row.end(), -5.0f);  // {-5, -4, -3, -2}

  f.layer->update(f.ctx, /*num_keys=*/1, &key, /*value_stride=*/kRowBytes,
                  /*value_size=*/kRowBytes, row.data(), /*table_id=*/0);

  const auto out = f.lookup({key});
  for (size_t e = 0; e < kElems; ++e) {
    EXPECT_FLOAT_EQ(out[e], row[e]) << "elem=" << e;
  }
}

TEST(DriverlessHostLayer, AccumulateAddsToBackingRow) {
  HostLayerFixture f;
  constexpr KeyType key = 33;
  const std::vector<float> before = f.lookup({key});
  const std::vector<float> delta{1.0f, 2.0f, 3.0f, 4.0f};

  f.layer->accumulate(f.ctx, /*num_keys=*/1, &key, kRowBytes, kRowBytes, delta.data(),
                      DataType_t::Float32, /*table_id=*/0);

  const auto out = f.lookup({key});
  for (size_t e = 0; e < kElems; ++e) {
    EXPECT_FLOAT_EQ(out[e], before[e] + delta[e]) << "elem=" << e;
  }
}

// CSR pooling (torch include_last_offset convention): 6 keys -> bags {3, 1, 2}.
TEST(DriverlessHostLayer, PooledLookupCsr) {
  HostLayerFixture f;
  const std::vector<KeyType> keys{0, 1, 2, 5, 10, 20};
  const std::vector<KeyType> offsets{0, 3, 4, 6};
  const std::vector<float> weights{0.5f, 1.5f, 2.0f, 1.0f, 0.25f, 4.0f};
  const size_t num_bags = offsets.size() - 1;

  for (const auto pooling : {PoolingType_t::Sum, PoolingType_t::Mean, PoolingType_t::WeightedSum,
                             PoolingType_t::WeightedMean}) {
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling;
    pp.sparse_type = SparseType_t::CSR;
    pp.output_type = DataType_t::Float32;
    pp.csr_offsets = offsets.data();
    pp.num_csr_offsets = static_cast<int64_t>(offsets.size());
    if (is_weighted_pooling(pooling)) {
      pp.weights = weights.data();
      pp.weight_type = DataType_t::Float32;
    }

    std::vector<float> out(num_bags * kElems, 0.0f);
    f.layer->lookup(f.ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(), kRowBytes,
                    /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);

    const auto expected = nve_test::reference_pool(f.backing.data(), kElems, keys, offsets, pooling, weights);
    ASSERT_EQ(out.size(), expected.size());
    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_FLOAT_EQ(out[i], expected[i]) << "pooling=" << static_cast<int>(pooling) << " elem=" << i;
    }
  }
}

// An all-empty-bags batch zeroes every output row instead of throwing (torch.nn.EmbeddingBag
// semantics); the non-pooled empty lookup is a no-op.
TEST(DriverlessHostLayer, EmptyLookup) {
  HostLayerFixture f;
  f.layer->lookup(f.ctx, 0, /*keys=*/nullptr, /*output=*/nullptr, kRowBytes,
                  /*hitmask=*/nullptr, /*pool_params=*/nullptr, /*hitrates=*/nullptr);

  const std::vector<KeyType> offsets{0, 0, 0};
  EmbeddingLayerBase::PoolingParams pp;
  pp.pooling_type = PoolingType_t::Sum;
  pp.sparse_type = SparseType_t::CSR;
  pp.output_type = DataType_t::Float32;
  pp.csr_offsets = offsets.data();
  pp.num_csr_offsets = static_cast<int64_t>(offsets.size());

  std::vector<float> out(2 * kElems, -99.0f);
  f.layer->lookup(f.ctx, 0, /*keys=*/nullptr, out.data(), kRowBytes,
                  /*hitmask=*/nullptr, &pp, /*hitrates=*/nullptr);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], 0.0f) << "elem=" << i;
  }
}

// Config::default_embedding fills the output rows of missing keys.
TEST(DriverlessHostLayer, DefaultEmbeddingFillsMisses) {
  const std::vector<float> default_row{7.0f, 8.0f, 9.0f, 10.0f};
  std::vector<uint8_t> default_bytes(kRowBytes);
  std::memcpy(default_bytes.data(), default_row.data(), default_bytes.size());
  HostLayerFixture f(default_bytes);

  const std::vector<KeyType> keys{-1, 5, kNumRows};
  const auto out = f.lookup(keys);
  for (size_t e = 0; e < kElems; ++e) {
    EXPECT_FLOAT_EQ(out[e], default_row[e]) << "elem=" << e;
    EXPECT_FLOAT_EQ(out[kElems + e], f.backing[5 * kElems + e]) << "elem=" << e;
    EXPECT_FLOAT_EQ(out[2 * kElems + e], default_row[e]) << "elem=" << e;
  }
}

TEST(DriverlessHostLayer, OutOfRangeKeysMissAndMutationsAreIgnored) {
  HostLayerFixture f;
  const std::vector<KeyType> keys{-1, kNumRows, 5};
  constexpr float untouched = -99.0f;
  std::vector<float> out(keys.size() * kElems, untouched);
  bitmask64_t hitmask = 0;

  f.layer->lookup(f.ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                  kRowBytes, &hitmask, /*pool_params=*/nullptr, /*hitrates=*/nullptr);

  EXPECT_EQ(hitmask, bitmask64::single(2));
  for (size_t e = 0; e < kElems; ++e) {
    EXPECT_FLOAT_EQ(out[e], untouched);
    EXPECT_FLOAT_EQ(out[kElems + e], untouched);
    EXPECT_FLOAT_EQ(out[2 * kElems + e], f.backing[5 * kElems + e]);
  }

  std::fill(out.begin(), out.end(), untouched);
  float hitrates[1] = {-1.0f};
  f.layer->lookup(f.ctx, static_cast<int64_t>(keys.size()), keys.data(), out.data(),
                  kRowBytes, /*hitmask=*/nullptr, /*pool_params=*/nullptr, hitrates);
  EXPECT_FLOAT_EQ(hitrates[0], 1.0f / 3.0f);
  for (size_t e = 0; e < kElems; ++e) {
    EXPECT_FLOAT_EQ(out[e], untouched);
    EXPECT_FLOAT_EQ(out[kElems + e], untouched);
    EXPECT_FLOAT_EQ(out[2 * kElems + e], f.backing[5 * kElems + e]);
  }

  const std::vector<float> original = f.backing;
  const std::vector<KeyType> invalid_keys{-1, kNumRows};
  std::vector<float> values(invalid_keys.size() * kElems, 7.0f);
  f.layer->update(f.ctx, static_cast<int64_t>(invalid_keys.size()), invalid_keys.data(),
                  kRowBytes, kRowBytes, values.data(), /*table_id=*/0);
  f.layer->accumulate(f.ctx, static_cast<int64_t>(invalid_keys.size()), invalid_keys.data(),
                      kRowBytes, kRowBytes, values.data(), DataType_t::Float32,
                      /*table_id=*/0);
  EXPECT_EQ(f.backing, original);
}

// In-memory StreamWrapperBase so the .nve round trip needs no filesystem.
class MemoryStream : public StreamWrapperBase {
 public:
  void write(const void* data, size_t size) override {
    if (pos_ + size > buf_.size()) buf_.resize(pos_ + size);
    std::memcpy(buf_.data() + pos_, data, size);
    pos_ += size;
  }
  uint64_t seek(uint64_t offset) override { return pos_ = std::min<uint64_t>(offset, buf_.size()); }
  uint64_t tell() override { return pos_; }
  void flush() override {}
  uint64_t read(void* data, size_t size) override {
    const size_t n = std::min<size_t>(size, buf_.size() - pos_);
    std::memcpy(data, buf_.data() + pos_, n);
    pos_ += n;
    return n;
  }
  uint64_t size() override { return buf_.size(); }

 private:
  std::vector<uint8_t> buf_;
  uint64_t pos_{0};
};

// The Python export path: a HostLayer over a malloc-backed HostMemBlock, serialized with
// TensorFileFormat (weights/*.nve) and loaded into a fresh HostMemBlock-backed layer.
TEST(DriverlessHostLayer, HostMemBlockSerializationRoundTrip) {
  auto src_block = std::make_shared<HostMemBlock>(kElems, kNumRows, DataType_t::Float32);
  ASSERT_EQ(src_block->get_type(), MemBlockType::HOST);
  ASSERT_EQ(src_block->get_size_in_bytes(), static_cast<size_t>(kNumRows * kRowBytes));
  auto* src = static_cast<float*>(src_block->get_ptr());
  std::iota(src, src + static_cast<size_t>(kNumRows) * kElems, 0.0f);

  const auto make_wrapper = [](const std::shared_ptr<HostMemBlock>& block) {
    TensorWrapper t;
    t.data = block->get_ptr();
    t.row = static_cast<uint64_t>(kNumRows);
    t.col = kElems;
    t.element_size_in_bytes = sizeof(float);
    return t;
  };

  constexpr uint64_t kLayerId = 42;
  MemoryStream stream;
  TensorFileFormat fmt;
  fmt.write_table_file_header(stream);
  fmt.write_tensor_to_stream(stream, kLayerId, make_wrapper(src_block));

  auto dst_block = std::make_shared<HostMemBlock>(kElems, kNumRows, DataType_t::Float32);
  TensorWrapper dst = make_wrapper(dst_block);
  stream.seek(0);
  fmt.load_tensor_from_stream(stream, kLayerId, dst);
  EXPECT_EQ(std::memcmp(src_block->get_ptr(), dst_block->get_ptr(), src_block->get_size_in_bytes()), 0);

  HostLayerHarness loaded(dst_block->get_ptr());
  const std::vector<KeyType> keys{0, 7, kNumRows - 1};
  const auto out = loaded.lookup(keys);
  for (size_t k = 0; k < keys.size(); ++k) {
    for (size_t e = 0; e < kElems; ++e) {
      EXPECT_FLOAT_EQ(out[k * kElems + e], src[static_cast<size_t>(keys[k]) * kElems + e])
          << "key=" << keys[k] << " elem=" << e;
    }
  }
}

TEST(QuantRowGeometry, RowSizeInBytes) {
  for (const DataType_t dt : {DataType_t::Float32, DataType_t::Float16, DataType_t::BFloat,
                              DataType_t::E4M3, DataType_t::E5M2, DataType_t::Float64}) {
    EXPECT_EQ(row_size_in_bytes(dt, 10), 10 * dtype_size(dt)) << dt;
  }
  // 1-byte codes + scale [+ offset] in the type's F32/F16 precision.
  EXPECT_EQ(row_size_in_bytes(DataType_t::QInt8RowwiseF32, 10), 10 + 4);
  EXPECT_EQ(row_size_in_bytes(DataType_t::QInt8RowwiseF16, 10), 10 + 2);
  EXPECT_EQ(row_size_in_bytes(DataType_t::QUint8RowwiseF32, 10), 10 + 8);
  EXPECT_EQ(row_size_in_bytes(DataType_t::QUint8RowwiseF16, 10), 10 + 4);
  // Corrupt sizes (e.g. from hand-edited metadata) throw instead of wrapping.
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(row_size_in_bytes(DataType_t::Float32, -1), std::exception);
  EXPECT_THROW(row_size_in_bytes(DataType_t::Float64, kMax / 4), std::exception);
  EXPECT_THROW(row_size_in_bytes(DataType_t::QUint8RowwiseF32, kMax - 4), std::exception);
}

TEST(QuantRowGeometry, RowNumElements) {
  for (const DataType_t dt : {DataType_t::Float32, DataType_t::Float16, DataType_t::Float64,
                              DataType_t::QInt8RowwiseF32, DataType_t::QInt8RowwiseF16,
                              DataType_t::QUint8RowwiseF32, DataType_t::QUint8RowwiseF16}) {
    EXPECT_EQ(row_num_elements(dt, row_size_in_bytes(dt, 10)), 10) << dt;
  }
  // Rows with no values beyond metadata, or a partial element, throw.
  EXPECT_THROW(row_num_elements(DataType_t::Float32, 0), std::exception);
  EXPECT_THROW(row_num_elements(DataType_t::Float32, 6), std::exception);
  EXPECT_THROW(row_num_elements(DataType_t::QUint8RowwiseF32, 8), std::exception);
}

TEST(QuantRowGeometry, HostMemBlockHoldsFullQuantRows) {
  for (const DataType_t dt : {DataType_t::QInt8RowwiseF32, DataType_t::QInt8RowwiseF16,
                              DataType_t::QUint8RowwiseF32, DataType_t::QUint8RowwiseF16}) {
    const auto row_bytes = static_cast<size_t>(row_size_in_bytes(dt, kElems));
    EXPECT_EQ(memblock_size_in_bytes(kElems, kNumRows, dt), kNumRows * row_bytes) << dt;
    HostMemBlock block(kElems, kNumRows, dt);
    EXPECT_EQ(block.get_size_in_bytes(), kNumRows * row_bytes) << dt;
  }
}

// Quantized tables serialize as raw bytes (col = stored row bytes, element size 1), keeping
// the trailing scale/offset of every row.
TEST(QuantRowGeometry, RawRowSerializationRoundTrip) {
  constexpr auto dt = DataType_t::QUint8RowwiseF32;
  const auto row_bytes = static_cast<uint64_t>(row_size_in_bytes(dt, kElems));
  auto src_block = std::make_shared<HostMemBlock>(kElems, kNumRows, dt);
  auto* src = static_cast<uint8_t*>(src_block->get_ptr());
  for (size_t i = 0; i < src_block->get_size_in_bytes(); ++i) src[i] = static_cast<uint8_t>(i * 7 + 3);

  const auto make_wrapper = [row_bytes](const std::shared_ptr<HostMemBlock>& block) {
    TensorWrapper t;
    t.data = block->get_ptr();
    t.row = static_cast<uint64_t>(kNumRows);
    t.col = row_bytes;
    t.element_size_in_bytes = 1;
    return t;
  };

  constexpr uint64_t kLayerId = 7;
  MemoryStream stream;
  TensorFileFormat fmt;
  fmt.write_table_file_header(stream);
  fmt.write_tensor_to_stream(stream, kLayerId, make_wrapper(src_block));

  auto dst_block = std::make_shared<HostMemBlock>(kElems, kNumRows, dt);
  TensorWrapper dst = make_wrapper(dst_block);
  stream.seek(0);
  fmt.load_tensor_from_stream(stream, kLayerId, dst);
  EXPECT_EQ(std::memcmp(src_block->get_ptr(), dst_block->get_ptr(), src_block->get_size_in_bytes()), 0);
}

}  // namespace
}  // namespace nve
