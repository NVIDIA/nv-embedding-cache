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

#include <regex>
#include <thread_pool.hpp>

namespace nve {

void ThreadPoolConfig::check() const {
  static const std::regex name_pattern{R"(^[\w\d]+$)"};
  NVE_CHECK_(std::regex_match(name, name_pattern));
}

ThreadPool::ThreadPool(const ThreadPoolConfig& config) : name{config.name} {}

int64_t ThreadPool::submit_n(int64_t task_idx, const int64_t num_tasks,
                             const indexed_task_type& task, result_type* results,
                             const int64_t workgroup) {
  if (results) {
    for (int64_t i{}; i < num_tasks; ++i) {
      results[i] = submit(std::bind(task, task_idx++), workgroup);
    }
  } else {
    for (int64_t i{}; i < num_tasks; ++i) {
      submit(std::bind(task, task_idx++), workgroup);
    }
  }
  return task_idx;
}

}  // namespace nve
