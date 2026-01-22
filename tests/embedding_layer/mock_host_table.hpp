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

#pragma once
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cuda_fp16.h>
#include "host_table.hpp"
#include "emb_layer_utils.hpp"

namespace nve {

template <typename IndexT>
class MockHostTable final : public HostTable<HostTableConfig> {
 public:
  using base_type = HostTable<HostTableConfig>;

  // sleep_ns is amount of nanoseconds to sleep for every key during find
  MockHostTable(const HostTableConfig& cfg, bool functional_ref, int8_t* linear_data = nullptr, uint64_t sleep_ns = 0)
      : base_type(0, cfg), functional_ref_(functional_ref), linear_data_(linear_data), sleep_ns_(sleep_ns) {}

  ~MockHostTable() override = default;

  int64_t size(context_ptr_t&, bool) const override { return static_cast<int64_t>(data_.size()) * config_.max_value_size; }

  void clear(context_ptr_t&) override { data_.clear(); }

  void erase(context_ptr_t&, int64_t n, const void* keys) override {
    const IndexT* typed_keys = reinterpret_cast<const IndexT*>(keys);
    for (int64_t i = 0; i < n; i++) {
      data_.erase(typed_keys[i]);
    }
  }

  void find(context_ptr_t& ctx, int64_t n, const void* keys, max_bitmask_repr_t* hit_mask,
            int64_t value_stride, void* values, int64_t* value_sizes) const override {
    constexpr auto mask_elements = sizeof(max_bitmask_repr_t) * 8;
    const IndexT* typed_keys = reinterpret_cast<const IndexT*>(keys);
    NVE_CHECK_(value_stride > 0, "Invalid stride");
    const uint64_t stride = static_cast<uint64_t>(value_stride);
    int64_t total_hits = 0;
    for (uint64_t i = 0; i < static_cast<uint64_t>(n); i++) {
      const uint64_t bit = 1ul << (i % mask_elements);
      if (hit_mask && (hit_mask[i / mask_elements] & bit)) {
        continue;  // datavector was already hit before
      }
      auto* dst = reinterpret_cast<uint8_t*>(values) + (i * stride);
      if (functional_ref_) {
        auto it = data_.find(typed_keys[i]);
        if (it != data_.end()) {
          total_hits++;
          std::memcpy(dst, it->second.data(), it->second.size());
          if (hit_mask) {
            hit_mask[i / mask_elements] |= bit;
          }
          if (value_sizes) {
            value_sizes[i] = static_cast<int64_t>(it->second.size());
          }
        } else if (linear_data_) {
          // fetch the line from linear data buffer (uvm), but don't update the hitmask (counted as a miss)
          auto* src = linear_data_ + (typed_keys[i] * config_.max_value_size);
          std::memcpy(dst, src, static_cast<size_t>(config_.max_value_size));
        }
      } else {
        total_hits++;
        std::memset(dst, 0xDB, stride);
        if (hit_mask) {
          hit_mask[i / mask_elements] |= bit;
        }
        if (value_sizes) {
          value_sizes[i] = value_stride;
        }
      }
    }
    if (sleep_ns_) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ns_ * static_cast<uint64_t>(total_hits)));
    }
    auto counter = this->get_internal_counter(ctx);
    NVE_CHECK_(counter != nullptr, "Invalid key counter");
    *counter += total_hits;
  }

  void insert(context_ptr_t&, int64_t n, const void* keys, int64_t value_stride, int64_t value_size,
              const void* values) override {
    const IndexT* typed_keys = reinterpret_cast<const IndexT*>(keys);
    NVE_CHECK_(value_size > 0);
    const uint64_t vsize = static_cast<uint64_t>(value_size);
    if (functional_ref_) {
      for (int64_t i = 0; i < n; i++) {
        if (data_.empty() || (data_.find(typed_keys[i]) == data_.end())) {
          auto& vec = data_[typed_keys[i]];
          vec.resize(vsize);
          auto* src = reinterpret_cast<const uint8_t*>(values) + (i * value_stride);
          std::memcpy(vec.data(), src, vsize);
        }
      }
    } else {
      // only read keys and values
      IndexT tmp_index = 0;
      for (int64_t i = 0; i < n; i++) {
        tmp_index += typed_keys[i];
        auto* src = reinterpret_cast<const uint8_t*>(values) + (i * value_stride);
        NVE_ASSERT_((vsize % sizeof(float)) == 0);
        auto typed_src = reinterpret_cast<const float*>(src);
        for (uint64_t j = 0; j < (vsize / sizeof(float)); j++) {
          tmp_val_ += typed_src[j];
        }
      }
    }
  }

  void update(context_ptr_t&, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override {
    const IndexT* typed_keys = reinterpret_cast<const IndexT*>(keys);
    NVE_CHECK_(value_size > 0);
    const uint64_t vsize = static_cast<uint64_t>(value_size);
    if (functional_ref_) {
      for (int64_t i = 0; i < n; i++) {
        auto it = data_.find(typed_keys[i]);
        if (it != data_.end()) {
          auto& vec = it->second;
          vec.resize(vsize);
          auto* src = reinterpret_cast<const uint8_t*>(values) + (i * value_stride);
          std::memcpy(vec.data(), src, vsize);
        }
      }
    } else {
      // only read keys and values
      IndexT tmp_index = 0;
      for (int64_t i = 0; i < n; i++) {
        tmp_index += typed_keys[i];
        auto* src = reinterpret_cast<const uint8_t*>(values) + (i * value_stride);
        NVE_ASSERT_((value_size % sizeof(float)) == 0);
        auto typed_src = reinterpret_cast<const float*>(src);
        for (uint64_t j = 0; j < vsize / sizeof(float); j++) {
          tmp_val_ += typed_src[j];
        }
      }
    }
  }

  void update_accumulate(context_ptr_t&, int64_t n, const void* keys, int64_t update_stride,
                         int64_t update_size, const void* updates,
                         DataType_t update_dtype) override {
    const IndexT* typed_keys = reinterpret_cast<const IndexT*>(keys);
    const auto update_element_size = dtype_size(update_dtype);
    if (update_size % update_element_size) {
      throw std::runtime_error("Invalid update size!");
    }
    const int64_t update_elements = update_size / update_element_size;
    if (functional_ref_) {
      for (int64_t i = 0; i < n; i++) {
        auto it = data_.find(typed_keys[i]);
        if (it != data_.end()) {
          auto& vec = it->second;
          auto* src_start = reinterpret_cast<const uint8_t*>(updates) + (i * update_stride);
          auto* dst_start = reinterpret_cast<uint8_t*>(vec.data());
          if (static_cast<int64_t>(vec.size())/config_.value_dtype_size() != update_elements) {
            throw std::runtime_error("Update size mismatch!");
          }
          accumulate_internal(dst_start, config_.value_dtype, src_start, update_dtype, static_cast<size_t>(update_elements));
        }
      }
    } else {
      // only read keys and values
      IndexT tmp_index = 0;
      for (int64_t i = 0; i < n; i++) {
        tmp_index += typed_keys[i];
        auto* src = reinterpret_cast<const uint8_t*>(updates) + (i * update_stride);
        NVE_ASSERT_((update_size % sizeof(float)) == 0);
        auto typed_src = reinterpret_cast<const float*>(src);
        for (uint64_t j = 0; j < static_cast<uint64_t>(update_size) / sizeof(float); j++) {
          tmp_val_ += typed_src[j];
        }
      }
    }
  }

  // Auxiliary function to support pooling (mostly for testing purposes)
  // Typically would be called on the output of find()
  void combine(
    const void* input,
    int64_t num_rows,
    PoolingType_t pooling_type,
    SparseType_t sparse_type,
    const int64_t* key_indices,
    int64_t num_key_indices,
    const void* weights,
    DataType_t weight_type,
    void* output
  ) {
    NVE_ASSERT_(num_key_indices > 0);
    const auto fixed_hotness = key_indices[0];
    int64_t num_bags;
    switch (sparse_type)
    {
      case SparseType_t::Fixed:
        NVE_ASSERT_(num_key_indices == 1);
        NVE_ASSERT_((num_rows % fixed_hotness) == 0);
        num_bags = num_rows / fixed_hotness;
        break;
      case SparseType_t::CSR:
        num_bags = num_key_indices - 1;
        break;
      default:
        throw std::runtime_error("Invalid sparse type");
    }
    
    const auto row_size = config_.max_value_size;
    NVE_ASSERT_((row_size % dtype_size(config_.value_dtype)) == 0);
    const uint64_t row_elements = static_cast<uint64_t>(row_size / dtype_size(config_.value_dtype));
    std::vector<float> bag_result(row_elements);
    int8_t* output_row = reinterpret_cast<int8_t*>(output);

    // loop over bags
    for (int64_t b=0 ; b<num_bags ; b++) {
      int64_t bag_start;
      int64_t bag_end;
      switch (sparse_type)
      {
        case SparseType_t::Fixed:
          bag_start = b * fixed_hotness;
          bag_end = (b+1) * fixed_hotness;
          break;
        case SparseType_t::CSR:
          bag_start = key_indices[b];
          bag_end = key_indices[b+1];
          break;
        default:
          throw std::runtime_error("Not implemented");
      }

      // loop over rows in a bag
      float weights_sum = 0.f;
      memset(bag_result.data(), 0, row_elements * sizeof(float));
      for (int64_t r = bag_start ; r < bag_end ; r++) {
        // sum row 'r' into bag_result
        const int8_t* row_start = reinterpret_cast<const int8_t*>(input) + (r * row_size);
        const float row_weight = weights ? load_as_float(weights, r, weight_type) : 1.f;
        weights_sum += row_weight;

        // loop over elements in the row
        for (uint64_t e=0 ; e<row_elements ; e++) {
          float val = load_as_float(row_start, static_cast<int64_t>(e), config_.value_dtype);
          switch (pooling_type) {
            case PoolingType_t::Concatenate:
              store_as_dtype(output_row, static_cast<int64_t>(e), config_.value_dtype, val); // for concat can store result now.
              break;
            case PoolingType_t::Sum:
            case PoolingType_t::Mean:
            case PoolingType_t::WeightedSum:
            case PoolingType_t::WeightedMean:
              bag_result[e] += val * row_weight;
              break;
            default:
              throw std::runtime_error("Not implemented");
          }
        }

        if (pooling_type == PoolingType_t::Concatenate) {
          // in concat need to advance output row after every input row
          output_row += row_size;
        }
      }

      // store bag_result to output (if not concat, once per bag)
      if (pooling_type != PoolingType_t::Mean && pooling_type != PoolingType_t::WeightedMean) {
        weights_sum = 1.f;
      }
      if (pooling_type != PoolingType_t::Concatenate) {
        for (uint64_t e=0 ; e<row_elements ; e++) {
          store_as_dtype(output_row, static_cast<int64_t>(e), config_.value_dtype, bag_result[e] / weights_sum);
        }
        output_row += row_size;
      }
    }
  }

 private:
  bool functional_ref_;
  int8_t* linear_data_;
  const uint64_t sleep_ns_;
  std::unordered_map<IndexT, std::vector<uint8_t>> data_;
  float tmp_val_{0.f}; // temp member we use to read values to so the compiler won't optimize out the reads as unused.

  void accumulate_internal(uint8_t* dst, DataType_t dst_type, const uint8_t* src, DataType_t src_type, size_t row_elements) {
    for (size_t i=0 ; i<row_elements ; i++) {
      float val = 0.f;
      switch (src_type) {
        case DataType_t::Float32:
          val = *reinterpret_cast<const float*>(src + i*sizeof(float));
          break;
        case DataType_t::Float16:
          val = __half2float(*reinterpret_cast<const half*>(src + i*sizeof(half)));
          break;
        default:
          throw std::runtime_error("Not implemented!");
      }
      switch (dst_type) {
        case DataType_t::Float32: {
            float* d = reinterpret_cast<float*>(dst + i*sizeof(float));
            *d += val;
          }
          break;
        case DataType_t::Float16: {
            half* d = reinterpret_cast<half*>(dst + i*sizeof(half));
            *d = __float2half(val + __half2float(*d));
          }
          break;
        default:
          throw std::runtime_error("Not implemented!");
      }
    }
  }
};

}  // namespace nve
