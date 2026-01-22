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

#include <nve_types.hpp>
#include <queue>
#include <thread_pool_base.hpp>

namespace nve {

void from_json(const nlohmann::json& json, ThreadPoolConfig& conf);

void to_json(nlohmann::json& json, const ThreadPoolConfig& conf);

thread_pool_ptr_t create_thread_pool(const nlohmann::json& json);

void configure_default_thread_pool(const nlohmann::json& json);

thread_pool_ptr_t default_thread_pool();

struct SimpleThreadPoolConfig : public ThreadPoolConfig {
  using base_type = ThreadPoolConfig;

  int64_t num_workers{};

  void check() const;
};

void from_json(const nlohmann::json& json, SimpleThreadPoolConfig& conf);

void to_json(nlohmann::json& json, const SimpleThreadPoolConfig& conf);

/**
 * A straightforward thread pool implementation that creates a caters all task to single workgroup.
 */
class SimpleThreadPool final : public ThreadPool {
 public:
  using base_type = ThreadPool;
  using task_queue_type = std::queue<packaged_task_type>;

  NVE_PREVENT_COPY_AND_MOVE_(SimpleThreadPool);

  SimpleThreadPool() = delete;

  SimpleThreadPool(const SimpleThreadPoolConfig& config);

  ~SimpleThreadPool() override;

  int64_t num_workers() const noexcept override { return static_cast<int64_t>(workers_.size()); }

  int64_t num_workgroups() const noexcept override { return 1; }

  result_type submit(task_type task, int64_t workgroup) override;

  int64_t submit_n(int64_t task_idx, int64_t num_tasks, const indexed_task_type& task,
                   result_type* results, int64_t workgroup) override;

 private:
  std::mutex tasks_guard_;
  task_queue_type tasks_;
  std::condition_variable on_submit_;

  std::vector<std::thread> workers_;

  void worker_main_(int64_t worker_idx);
};

/**
 * NUMA workgroup configuration. Use the `show_numa_config` tool to determine your system's NUMA
 * configuration.
 */
struct NumaWorkgroupConfig {
  int64_t cpu_socket_index{-1};  // CPU socket index. Either -1 or the index of the CPU socket.
  int64_t numa_node_index{};  // NUMA node index. If CPU socket is -1, this is the global NUMA node
                              // index. Otherwise, the nodex index refers to the NUMA nodes
                              // associated with the CPU socket. Use the `show_numa_config` tool to
                              // determine your current system's NUMA configuration.
  int64_t num_workers{};  // Number of workers to allocate to this workgroup. If zero, we set this
                          // to the number of logical cores present in the NUMA node.

  void check() const;
};

void from_json(const nlohmann::json& json, NumaWorkgroupConfig& conf);

void to_json(nlohmann::json& json, const NumaWorkgroupConfig& conf);

struct NumaThreadPoolConfig : public ThreadPoolConfig {
  using base_type = ThreadPoolConfig;

  std::vector<NumaWorkgroupConfig> workgroups{};  // Configure workgroups. Empty will result yield
                                                  // one workgroup for each NUMA node.

  void check() const;
};

void from_json(const nlohmann::json& json, NumaThreadPoolConfig& conf);

void to_json(nlohmann::json& json, const NumaThreadPoolConfig& conf);

/**
 * A thread-pool implementation organizes threads in workgroups which can in turn be bound to
 * specific NUMA nodes.
 */
class NumaThreadPool final : public ThreadPool {
 public:
  using base_type = ThreadPool;
  using task_queue_type = std::queue<packaged_task_type>;

  NVE_PREVENT_COPY_AND_MOVE_(NumaThreadPool);

  NumaThreadPool() = delete;

  NumaThreadPool(const NumaThreadPoolConfig& config);

  ~NumaThreadPool() override;

  int64_t num_workers() const noexcept override { return static_cast<int64_t>(workers_.size()); }

  int64_t num_workgroups() const noexcept override { return static_cast<int64_t>(tasks_.size()); }

  result_type submit(task_type task, int64_t workgroup) override;

  int64_t submit_n(int64_t task_idx, int64_t num_tasks, const indexed_task_type& task,
                   result_type* results, int64_t workgroup) override;

 private:
  std::vector<std::mutex> tasks_guards_;
  std::vector<task_queue_type> tasks_;
  std::vector<std::condition_variable> on_submits_;

  std::vector<std::thread> workers_;

  void worker_main_(uint64_t workgroup_idx, int64_t worker_idx, int64_t numa_node_idx);
};

}  // namespace nve
