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

// ---------------------------------------------------------------------------
// find_and_dequant tests (cuda_ops/find_and_dequant.cuh).
//
// find_and_dequant is the hotness-1 / no-pooling gather: each key produces one output row of
// dequantized QuantizationHelper<DTYPE>::ParamType values (float for *F32, __half for *F16),
// reconstructed as float(q) * scale [+ offset]. Rows are stored exactly like the FindAndCombine
// quant tests above: [ q[num_elements] ][ scale ][ offset? ], with the stride padded to 4 bytes.
//
// This fixture mirrors EmbeddingCacheQuantLookupRefTest's data generation but exercises the
// dequant-only kernel across:
//   * both CacheData variants -- the set-associative cache (USE_CACHE) and ECNoCache (direct
//     table reads), to cover both AddressFunctor<IndexT, CacheDataT> specializations; and
//   * load_indices=true (resolve the row from the keys buffer) and false (positional row = tid).
// The dequantized bytes are identical whether a lookup hits the cache or falls through to the
// table, so the CPU reference is the same for every variant.
// ---------------------------------------------------------------------------
template <DataType_t VALUE_TYPE_ID_, typename StorageT, typename IndexT,
          bool USE_CACHE, bool LOAD_INDICES,
          typename OutputT = typename QuantizationHelper<VALUE_TYPE_ID_>::ParamType>
struct DequantTestType {
  static constexpr DataType_t VALUE_TYPE_ID = VALUE_TYPE_ID_;
  using StorageType = StorageT;
  using IndexType = IndexT;
  // Type written by find_and_dequant. Defaults to the stored ParamType (dequant type == output, the
  // common case); a distinct OutputType exercises the OUTPUT_TYPE template parameter (e.g. __half
  // storage dequantized to a float output, or float storage to a __half output).
  using OutputType = OutputT;
  static constexpr bool USE_CACHE_FLAG = USE_CACHE;
  static constexpr bool LOAD_INDICES_FLAG = LOAD_INDICES;
};

template <typename T>
class FindAndDequantRefTest : public ::testing::Test {
 public:
  using StorageType = typename T::StorageType;
  using IndexType = typename T::IndexType;
  static constexpr DataType_t VALUE_TYPE_ID = T::VALUE_TYPE_ID;
  // ParamType is the scale/offset and dequantized (pre-Cast) type (float for *F32, __half for *F16).
  using ParamType = typename QuantizationHelper<VALUE_TYPE_ID>::ParamType;
  // OutputType is the type find_and_dequant Casts the dequantized value to before storing.
  using OutputType = typename T::OutputType;
  static constexpr bool HAS_OFFSET = QuantizationHelper<VALUE_TYPE_ID>::has_offset;
  static constexpr bool USE_CACHE = T::USE_CACHE_FLAG;
  static constexpr bool LOAD_INDICES = T::LOAD_INDICES_FLAG;

  using CacheType = nve::CacheSAHostModify<IndexType, IndexType>;
  using SACacheData = typename CacheType::CacheData;
  using NoCacheData = typename nve::ECNoCache<IndexType>::CacheData;
  using CacheDataType = std::conditional_t<USE_CACHE, SACacheData, NoCacheData>;

  static ParamType from_float(float v) {
    if constexpr (std::is_same_v<ParamType, __half>) return __float2half(v);
    else return static_cast<ParamType>(v);
  }

  // Model the kernel's final Cast<DequantType, OutputType> so a __half output is checked against a
  // __half-rounded reference (and a float output against the un-rounded value).
  static OutputType to_output(float v) {
    if constexpr (std::is_same_v<OutputType, __half>) return __float2half(v);
    else return static_cast<OutputType>(v);
  }
  static float to_float(OutputType v) {
    if constexpr (std::is_same_v<OutputType, __half>) return __half2float(v);
    else return static_cast<float>(v);
  }

  FindAndDequantRefTest()
      : cache_allocator_(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD) {
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
  }
  ~FindAndDequantRefTest() {
    if constexpr (USE_CACHE) {
      if (cache_ptr_) {
        CHECK_CUDA_ERROR(cache_ptr_->lookup_context_destroy(handle_lookup_));
        CHECK_CUDA_ERROR(cache_ptr_->modify_context_destroy(handle_modify_));
      }
    }
  }

  void LaunchTest(uint64_t num_rows, uint32_t num_elements, uint64_t num_keys, bool allocOnHost) {
    num_rows_ = num_rows;
    num_elements_ = num_elements;
    num_keys_ = static_cast<IndexType>(num_keys);
    allocOnHost_ = allocOnHost;
    row_bytes_ = num_elements_ * sizeof(StorageType)
                + (HAS_OFFSET ? 2 : 1) * sizeof(ParamType);  // q + scale [+ offset]
    // Pad the row stride to 4 bytes so every row base stays aligned for the Vec4 (char4) load path.
    row_bytes_ = (row_bytes_ + 3u) & ~static_cast<uint64_t>(3u);
    // load_indices=false reads rows positionally over [0, num_keys); load_indices=true uses keys in
    // [1, num_rows). Either way every accessed row must be < num_rows.
    ASSERT_LT(static_cast<uint64_t>(num_keys_), num_rows_)
        << "num_keys must be < table rows so positional (load_indices=false) reads stay in-bounds";
    AllocateTable();
    AllocateKeys();

    CacheDataType cache_data = MakeCacheData();
    ComputeRefResults();
    LaunchKernel(cache_data);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CheckResult();
  }

 private:
  // Quantize one row of floats into StorageType q[], producing scale/offset such that
  // float(q[j]) * scale + offset reconstructs v[j]. Mirrors EmbeddingCacheQuantLookupRefTest.
  void Quantize(const std::vector<float>& v, StorageType* q, ParamType& scale_out, ParamType& offset_out) {
    float scale = 1.0f, offset = 0.0f;
    if constexpr (VALUE_TYPE_ID == DataType_t::QInt8RowwiseF32 ||
                  VALUE_TYPE_ID == DataType_t::QInt8RowwiseF16) {
      // Symmetric, pure scale (QInt8 variants set has_offset=false). The HAS_OFFSET branch is
      // dead for these types but kept structurally identical to the FindAndCombine quant fixture.
      if constexpr (HAS_OFFSET) {
        double sum = 0.0;
        for (float x : v) sum += x;
        offset = static_cast<float>(sum / static_cast<double>(v.size()));
      }
      float amax = 0.0f;
      for (float x : v) amax = std::max(amax, std::fabs(x - offset));
      scale = amax > 0.0f ? amax / 127.0f : 1.0f;
      for (size_t j = 0; j < v.size(); ++j) {
        long qi = std::lround((v[j] - offset) / scale);
        qi = std::min<long>(127, std::max<long>(-127, qi));
        q[j] = static_cast<StorageType>(qi);
      }
    } else {  // QUint8Rowwise* : min/max affine
      float lo = v[0], hi = v[0];
      for (float x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
      scale = hi > lo ? (hi - lo) / 255.0f : 1.0f;
      offset = lo;
      for (size_t j = 0; j < v.size(); ++j) {
        long qi = std::lround((v[j] - offset) / scale);
        qi = std::min<long>(255, std::max<long>(0, qi));
        q[j] = static_cast<StorageType>(qi);
      }
    }
    scale_out = from_float(scale);
    offset_out = from_float(offset);
  }

  // Dequantize one element, mirroring the kernel's per-type arithmetic so the only divergence is
  // float rounding (kept inside the tolerance).
  float Dequantize(StorageType q, ParamType scale, ParamType offset) const {
    if constexpr (std::is_same_v<ParamType, __half>) {
      float prod = static_cast<float>(q) * __half2float(scale);
      if constexpr (HAS_OFFSET) prod += __half2float(offset);
      return __half2float(__float2half(prod));  // kernel stores the dequantized value as __half
    } else {
      float r = static_cast<float>(q) * scale;
      if constexpr (HAS_OFFSET) r += offset;
      return r;
    }
  }

  void AllocateTable() {
    table_ = std::make_shared<TestBuffer<int8_t>>(num_rows_ * row_bytes_);
    std::mt19937 gen(0X814753);
    // Values in [-1, 1] expressed as a / 2^b to keep float rounding small.
    std::uniform_int_distribution<int32_t> dist_nom(-8, 8);
    std::uniform_int_distribution<uint32_t> dist_denom(3, 9);

    std::vector<float> v(num_elements_);
    for (uint64_t i = 0; i < num_rows_; i++) {
      for (uint32_t j = 0; j < num_elements_; j++) {
        v[j] = float(dist_nom(gen)) / float(1 << dist_denom(gen));
      }
      int8_t* row = table_->ph + i * row_bytes_;
      ParamType scale{}, offset{};
      Quantize(v, reinterpret_cast<StorageType*>(row), scale, offset);
      int8_t* meta = row + num_elements_ * sizeof(StorageType);
      memcpy(meta, static_cast<const void*>(&scale), sizeof(ParamType));
      if (HAS_OFFSET) memcpy(meta + sizeof(ParamType), static_cast<const void*>(&offset), sizeof(ParamType));
    }
    table_->HtoD(stream_);
  }

  void AllocateKeys() {
    keys_ = std::make_shared<TestBuffer<IndexType>>(num_keys_ * sizeof(IndexType));
    const float alpha = 1.05f;
    const size_t seed = 283982;
    auto sg = getSampleGenerator<IndexType>(alpha, static_cast<IndexType>(num_rows_),
                                            static_cast<uint32_t>(num_keys_), seed);
    auto sample = sg->getCategoryIndices();
    std::copy(sample.begin(), sample.begin() + num_keys_, keys_->ph);
    keys_->HtoD(stream_);
  }

  // Builds the CacheData passed to the kernel: a populated set-associative cache when USE_CACHE,
  // otherwise a trivial ECNoCache descriptor that makes the kernel read straight from the table.
  CacheDataType MakeCacheData() {
    if constexpr (USE_CACHE) {
      InitCache();
      std::vector<IndexType> cached = ComputeCachedIndices(cache_data_.num_sets, CacheType::NUM_WAYS);
      PopulateCache(cached);
      cache_data_ = cache_ptr_->get_cache_data(handle_lookup_);
      return cache_data_;
    } else {
      NoCacheData data{};
      data.row_size_in_bytes = static_cast<uint32_t>(row_bytes_);
      data.count_misses = false;
      data.misses = nullptr;
      return data;
    }
  }

  void InitCache() {
    const float cache_ratio = 0.15f;
    const uint32_t num_rows_in_cache = static_cast<uint32_t>(cache_ratio * static_cast<float>(num_rows_));
    typename CacheType::CacheConfig cfg;
    cfg.embed_width_in_bytes = row_bytes_;
    cfg.cache_sz_in_bytes = num_rows_in_cache * cfg.embed_width_in_bytes;
    cfg.num_tables = 1;
    cfg.allocate_data_on_host = allocOnHost_;
    cache_ptr_ = std::make_shared<CacheType>(&cache_allocator_, &cache_logger_, cfg);
    cache_ptr_->init();
    cache_ptr_->lookup_context_create(handle_lookup_, nullptr, 0);
    cache_data_ = cache_ptr_->get_cache_data(handle_lookup_);
  }

  std::vector<IndexType> ComputeCachedIndices(const int num_sets, const int num_ways) {
    std::set<IndexType> cached_indices;
    std::vector<int> counters(num_sets, 0);
    uint64_t cache_capacity = static_cast<uint64_t>(static_cast<double>(num_rows_) * 0.15);
    while (cache_capacity > static_cast<uint64_t>(num_keys_)) {
      cache_capacity /= 2;
    }
    uint32_t step = static_cast<uint32_t>(num_keys_ / cache_capacity);
    EXPECT_TRUE(step > 0);
    for (IndexType i = 0; i < num_keys_; i += step) {
      IndexType idx = keys_->ph[i];
      if (++counters[idx % num_sets] <= num_ways) {
        cached_indices.insert(idx);
      }
    }
    return std::vector<IndexType>(cached_indices.begin(), cached_indices.end());
  }

  void PopulateCache(const std::vector<IndexType>& cached_indices) {
    cache_ptr_->modify_context_create(handle_modify_, static_cast<uint32_t>(cached_indices.size()));
    nve::DefaultHistogram<IndexType> hist(cached_indices.data(), cached_indices.size(),
                                          table_->ph, row_bytes_, true);
    cudaEvent_t wait;
    CHECK_CUDA_ERROR(cudaEventCreate(&wait));
    nve::DefaultECEvent ec_event(std::vector<cudaStream_t>{});
    cache_ptr_->insert(handle_modify_, hist.get_keys(), hist.get_priority(), hist.get_data(),
                        hist.get_num_bins(), 0, &ec_event, stream_);
    CHECK_CUDA_ERROR(cudaEventRecord(wait));
    CHECK_CUDA_ERROR(cudaEventSynchronize(wait));
  }

  void ComputeRefResults() {
    // Store the dequantized value (as float, already reflecting the kernel's ParamType rounding);
    // CheckResult rounds it through OutputType to model the kernel's final Cast.
    ref_result_.assign(static_cast<size_t>(num_keys_) * num_elements_, 0.0f);
    for (uint64_t i = 0; i < static_cast<uint64_t>(num_keys_); i++) {
      // load_indices=true resolves the row from the keys buffer; false reads positionally (tid).
      IndexType idx = LOAD_INDICES ? keys_->ph[i] : static_cast<IndexType>(i);
      const int8_t* row = table_->ph + static_cast<uint64_t>(idx) * row_bytes_;
      const StorageType* q = reinterpret_cast<const StorageType*>(row);
      const int8_t* meta = row + num_elements_ * sizeof(StorageType);
      ParamType scale{}, offset{};
      memcpy(static_cast<void*>(&scale), meta, sizeof(ParamType));
      if (HAS_OFFSET) memcpy(static_cast<void*>(&offset), meta + sizeof(ParamType), sizeof(ParamType));
      for (uint32_t el = 0; el < num_elements_; el++) {
        ref_result_[i * num_elements_ + el] = Dequantize(q[el], scale, offset);
      }
    }
  }

  void LaunchKernel(CacheDataType& cache_data) {
    result_ = std::make_shared<TestBuffer<OutputType>>(
        static_cast<size_t>(num_keys_) * num_elements_ * sizeof(OutputType));
    const size_t stride = num_elements_ * sizeof(OutputType);
    CHECK_CUDA_ERROR((nve::call_find_and_dequant<IndexType, CacheDataType>(
        keys_->pd, static_cast<size_t>(num_keys_), reinterpret_cast<int8_t*>(result_->pd),
        table_->pd, VALUE_TYPE_ID, data_type<OutputType>(), num_elements_, cache_data, stream_,
        stride, /*load_indices=*/LOAD_INDICES, /*curr_table=*/0)));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  }

  void CheckResult() {
    result_->DtoH(stream_);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Single dequant per element (no accumulation): a float output matches the reference exactly bar
    // rounding; a __half (ParamType or output) is limited by half granularity.
    const float tolerance =
        (std::is_same_v<ParamType, __half> || std::is_same_v<OutputType, __half>) ? 1e-2f : 1e-4f;
    Near near(tolerance);
    bool all_near = true;
    for (size_t i = 0; i < ref_result_.size(); ++i) {
      const OutputType expected = to_output(ref_result_[i]);
      all_near = all_near && near(to_float(expected), to_float(result_->ph[i]));
    }
    EXPECT_TRUE(all_near);
  }

  cudaStream_t stream_;
  uint64_t num_rows_{0};
  uint32_t num_elements_{0};
  bool allocOnHost_{false};
  uint64_t row_bytes_{0};
  IndexType num_keys_{0};

  std::shared_ptr<TestBuffer<int8_t>> table_ = nullptr;
  std::shared_ptr<TestBuffer<IndexType>> keys_ = nullptr;
  std::shared_ptr<TestBuffer<OutputType>> result_ = nullptr;
  std::vector<float> ref_result_;

  nve::DefaultAllocator cache_allocator_;
  nve::Logger cache_logger_;
  std::shared_ptr<CacheType> cache_ptr_;
  nve::LookupContextHandle handle_lookup_;
  nve::ModifyContextHandle handle_modify_;
  SACacheData cache_data_{};
};

TYPED_TEST_SUITE_P(FindAndDequantRefTest);

TYPED_TEST_P(FindAndDequantRefTest, DequantAgainstRefCpu) {
  // num_elements 132 / 30 / 63 select the Vec4 / Vec2 / Vec1 dequant paths (row layout % 4).
  // num_keys 1000 / 4099 cover a non-multiple-of-block tail.
  for (const auto num_keys : {1000, 4099}) {
    for (const auto num_elements : {132, 30, 63}) {
      for (const auto allocOnHost : {true, false}) {
        this->LaunchTest(262144, num_elements, num_keys, allocOnHost);
      }
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(FindAndDequantRefTest, DequantAgainstRefCpu);

// 4 quant dtypes x {set-associative cache, no-cache} x {load_indices true, false}.
// QInt8 storage is int8_t (plain char is unsigned on aarch64); QUint8 storage is unsigned char
// (matching QuantizationHelper::Vec1).
typedef ::testing::Types<
    // QInt8RowwiseF32 (symmetric, pure float scale, no offset).
    DequantTestType<DataType_t::QInt8RowwiseF32,  int8_t,        int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF32,  int8_t,        int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QInt8RowwiseF32,  int8_t,        int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF32,  int8_t,        int64_t, /*cache=*/false, /*load_idx=*/false>,
    // QUint8RowwiseF32 (min/max affine, float scale + offset).
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, /*cache=*/false, /*load_idx=*/false>,
    // QInt8RowwiseF16 (symmetric, pure-scale / no offset; __half scale + output).
    DequantTestType<DataType_t::QInt8RowwiseF16,  int8_t,        int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF16,  int8_t,        int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QInt8RowwiseF16,  int8_t,        int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QInt8RowwiseF16,  int8_t,        int64_t, /*cache=*/false, /*load_idx=*/false>,
    // QUint8RowwiseF16 (affine; __half scale + offset + output).
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int32_t, /*cache=*/true,  /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int64_t, /*cache=*/true,  /*load_idx=*/false>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int32_t, /*cache=*/false, /*load_idx=*/true>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int64_t, /*cache=*/false, /*load_idx=*/false>,
    // Mismatched OUTPUT_TYPE (dequant type != output): exercise the OUTPUT_TYPE template parameter.
    // Trailing arg is the output type. float-dequant storage -> __half output:
    DequantTestType<DataType_t::QInt8RowwiseF32,  int8_t,        int32_t, /*cache=*/true,  /*load_idx=*/true,  __half>,
    DequantTestType<DataType_t::QUint8RowwiseF32, unsigned char, int64_t, /*cache=*/false, /*load_idx=*/false, __half>,
    // __half-dequant storage -> float output:
    DequantTestType<DataType_t::QInt8RowwiseF16,  int8_t,        int64_t, /*cache=*/true,  /*load_idx=*/false, float>,
    DequantTestType<DataType_t::QUint8RowwiseF16, unsigned char, int32_t, /*cache=*/false, /*load_idx=*/true,  float>>
    DequantTypes;

INSTANTIATE_TYPED_TEST_SUITE_P(FindAndDequant,
                               FindAndDequantRefTest,
                               DequantTypes);
