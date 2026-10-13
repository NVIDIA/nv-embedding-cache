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

// Shared helpers for HostEmbeddingLayer tests (GPU-capable and driverless builds alike):
// building a layer over a caller-owned host buffer, key-list construction, and a serial
// pooling reference. Depends only on nve-common headers.

#pragma once

#include "include/host_embedding_layer.hpp"
#include "include/linear_host_table.hpp"
#include "include/nve_types.hpp"
#include "include/thread_pool.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nve_test {

// Thread pool wrapper that records the task count of every submit/submit_n call, so tests can
// check how CPU work was split (e.g. the task_size_threshold bound).
class TaskCountingThreadPool final : public nve::ThreadPool {
 public:
  explicit TaskCountingThreadPool(int64_t num_workers)
      : nve::ThreadPool(nve::ThreadPoolConfig{}), inner_(make_inner(num_workers)) {}

  int64_t num_workers() const noexcept override { return inner_->num_workers(); }
  int64_t num_workgroups() const noexcept override { return inner_->num_workgroups(); }

  result_type submit(task_type task, int64_t workgroup) override {
    record(1);
    return inner_->submit(std::move(task), workgroup);
  }

  int64_t submit_n(int64_t task_idx, int64_t num_tasks, const indexed_task_type& task,
                   result_type* results, int64_t workgroup) override {
    record(num_tasks);
    return inner_->submit_n(task_idx, num_tasks, task, results, workgroup);
  }

  // Task counts of all submissions so far, in call order.
  std::vector<int64_t> task_counts() const {
    std::lock_guard<std::mutex> lock(guard_);
    return task_counts_;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(guard_);
    task_counts_.clear();
  }

 private:
  static std::shared_ptr<nve::SimpleThreadPool> make_inner(int64_t num_workers) {
    nve::SimpleThreadPoolConfig cfg;
    cfg.num_workers = num_workers;
    return std::make_shared<nve::SimpleThreadPool>(cfg);
  }

  void record(int64_t num_tasks) {
    std::lock_guard<std::mutex> lock(guard_);
    task_counts_.push_back(num_tasks);
  }

  std::shared_ptr<nve::SimpleThreadPool> inner_;
  mutable std::mutex guard_;
  std::vector<int64_t> task_counts_;
};

template <typename KeyType>
std::vector<KeyType> make_keys(std::initializer_list<int64_t> ids) {
  std::vector<KeyType> keys;
  keys.reserve(ids.size());
  for (int64_t id : ids) {
    keys.push_back(static_cast<KeyType>(id));
  }
  return keys;
}

// LinearHostTable + HostEmbeddingLayer + execution context over a caller-owned host buffer.
template <typename KeyType>
struct HostLayerHandles {
  std::shared_ptr<nve::LinearHostTable<KeyType>> table;
  std::shared_ptr<nve::HostEmbeddingLayer<KeyType>> layer;
  nve::context_ptr_t ctx;
};

// Null stream/allocator/thread pool: the layer defaults to GetDefaultAllocator() and the
// default thread pool, which is the driverless-safe path.
template <typename KeyType>
HostLayerHandles<KeyType> make_host_layer(void* emb_table, int64_t row_bytes, int64_t num_rows,
                                          nve::DataType_t value_dtype, int64_t max_threads,
                                          const std::string& layer_name,
                                          std::vector<uint8_t> default_embedding = {}) {
  nve::LinearHostTableConfig cfg;
  cfg.value_dtype = value_dtype;
  cfg.max_threads = max_threads;
  cfg.max_value_size = row_bytes;
  cfg.num_rows = num_rows;
  cfg.emb_table = emb_table;

  HostLayerHandles<KeyType> h;
  h.table = std::make_shared<nve::LinearHostTable<KeyType>>(cfg);

  typename nve::HostEmbeddingLayer<KeyType>::Config layer_cfg;
  layer_cfg.layer_name = layer_name;
  layer_cfg.default_embedding = std::move(default_embedding);
  h.layer = std::make_shared<nve::HostEmbeddingLayer<KeyType>>(layer_cfg, h.table);
  h.ctx = h.layer->create_execution_context(/*lookup_stream=*/0, /*modify_stream=*/0,
                                            /*thread_pool=*/nullptr, /*allocator=*/nullptr);
  return h;
}

// Serial CPU reference: pool the fp32 rows of `keys` (`elements` per row in `rows`) into bags
// defined by `bag_offsets` (size num_bags+1), matching cpu_kernel_pooling. WeightedMean divides
// by the sum of weights; Mean by the bag's key count.
template <typename KeyType>
std::vector<float> reference_pool(const float* rows, size_t elements, const std::vector<KeyType>& keys,
                                  const std::vector<KeyType>& bag_offsets, nve::PoolingType_t pooling,
                                  const std::vector<float>& weights) {
  const size_t num_bags = bag_offsets.size() - 1;
  const bool weighted = nve::is_weighted_pooling(pooling);
  const bool mean = pooling == nve::PoolingType_t::Mean || pooling == nve::PoolingType_t::WeightedMean;
  std::vector<float> expected(num_bags * elements, 0.0f);
  for (size_t b = 0; b < num_bags; b++) {
    const int64_t start = bag_offsets[b];
    const int64_t end = bag_offsets[b + 1];
    float denom = 0.0f;
    for (int64_t k = start; k < end; k++) {
      const float w = weighted ? weights[static_cast<size_t>(k)] : 1.0f;
      denom += w;
      const float* row = rows + static_cast<size_t>(keys[static_cast<size_t>(k)]) * elements;
      for (size_t e = 0; e < elements; e++) {
        expected[b * elements + e] += w * row[e];
      }
    }
    if (mean && denom != 0.0f) {
      for (size_t e = 0; e < elements; e++) {
        expected[b * elements + e] /= denom;
      }
    }
  }
  return expected;
}

}  // namespace nve_test
