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

#include <cstring>
#include <vector>
#include <limits>
#include <random>
#include <embedding_layer.hpp>
#include <insert_heuristic.hpp>

namespace nve {

template <typename IndexT>
class SetupKeys {
public:
  SetupKeys(std::vector<IndexT>& keys, uint64_t& start_key, uint64_t& end_key) {
    if (!keys.empty()) {
      start_key = std::min(start_key, keys.size() - 1);
      end_key = std::max(std::min(end_key, keys.size()), uint64_t(1));
      num_keys = static_cast<int64_t>(end_key - start_key);
      keys_buffer = keys.data() + start_key;
    } else {
      start_key = 0;
      end_key = 0;
      num_keys = 0;
      keys_buffer = nullptr;
    }
  }
  SetupKeys(std::vector<IndexT>& keys, uint64_t start_key, uint64_t end_key, std::vector<uint8_t>& datavectors, int64_t row_size)
    : SetupKeys(keys, start_key, end_key) {
    if (!datavectors.empty()) {
      data_buffer = datavectors.data() + (start_key * static_cast<uint64_t>(row_size));
    } else {
      data_buffer = nullptr;
    }
  }
  int64_t num_keys{0};
  IndexT* keys_buffer{nullptr};
  uint8_t* data_buffer{nullptr};
};

// Generate uniform keys and data
template <typename IndexT>
void GenerateData(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
                          size_t num_keys, int64_t row_size, IndexT min_index = 0,
                          IndexT max_index = std::numeric_limits<IndexT>::max(),
                          nve::DataType_t dtype = nve::DataType_t::Float32,
                          bool unique_values = true, size_t seed = 31337) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<IndexT> dist_key(min_index, max_index - 1); // max index is exclusive so that we can just use num rows in the test
  std::uniform_real_distribution<float> dist_float(0.f, 1.f);
  keys.resize(num_keys);
  for (size_t i = 0; i < num_keys; i++) {
    do { 
        keys[i] = dist_key(gen);
    } while (unique_values && (std::count(keys.begin(), keys.end(), keys.at(i)) > 1));
  }
  const auto data_bytes = num_keys * static_cast<size_t>(row_size);
  datavectors.resize(data_bytes);
  const auto num_elements = data_bytes / static_cast<size_t>(dtype_size(dtype));
  auto data = datavectors.data();
  switch (dtype) {
    case DataType_t::Float32:
      for (size_t i=0 ; i < num_elements ; i++) {
        (reinterpret_cast<float*>(data))[i] = dist_float(gen);
      }
      break;
    case DataType_t::Float16:
      for (size_t i=0 ; i < num_elements ; i++) {
        (reinterpret_cast<half*>(data))[i] = __float2half(dist_float(gen));
      }
      break;
    default:
      throw std::runtime_error("Not implemented");
  }
  const auto remainder = data_bytes % sizeof(float);
  if (remainder) {
    std::memset(datavectors.data() + data_bytes - 1 - remainder, 0, remainder);
  }
}

class TestInsertHeuristic : public InsertHeuristic {
public:
  TestInsertHeuristic(std::vector<bool> results = {true, true, true}) : results_(results) {}
  ~TestInsertHeuristic() override = default;
  bool insert_needed(const float /*hitrate*/, const size_t threshold_id) override {
    return results_.at(threshold_id);
  }
  std::vector<bool> results_;
};

template <typename OffsetT>
class SetupCSROffsets {
public:
  SetupCSROffsets(OffsetT num_keys, size_t seed = 59371) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<OffsetT> dist_offset(1, 64);

    OffsetT curr_offset = 0;
    while (curr_offset < num_keys) {
      offsets_buffer_.push_back(curr_offset);
      OffsetT next_step = dist_offset(gen);
      curr_offset += next_step;
    }

    offsets_buffer_.push_back(num_keys);
    num_offsets = offsets_buffer_.size();
    offsets_buffer = offsets_buffer_.data(); 
  }

  uint64_t num_offsets{0};
  OffsetT* offsets_buffer{nullptr};
  std::vector<OffsetT> offsets_buffer_;
};

float load_as_float(const void* ptr, int64_t idx, DataType_t dtype);
void store_as_dtype(void* ptr, int64_t idx, DataType_t dtype, float val);
void GenerateWeights(std::vector<int8_t>& weights, uint64_t num_weights, DataType_t dtype, size_t seed = 31337);
void InitTableRows(int8_t* table, uint64_t row_size, uint64_t start_row, uint64_t end_row, DataType_t dtype, size_t seed = 1337);

}  // namespace nve
