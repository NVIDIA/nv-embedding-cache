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

#include <numa_support.hpp>
#include <regex>
#include <thread_pool.hpp>

namespace nve {

void from_json(const nlohmann::json& json, ThreadPoolConfig& conf) { NVE_READ_JSON_FIELD_(name); }

void to_json(nlohmann::json& json, const ThreadPoolConfig& conf) {
  json = json.object();

  NVE_WRITE_JSON_FIELD_(name);
}

static thread_pool_ptr_t create_simple_thread_pool(const nlohmann::json& json) {
  const auto config{static_cast<SimpleThreadPoolConfig>(json)};
  config.check();
  return std::make_shared<SimpleThreadPool>(config);
}

static thread_pool_ptr_t create_numa_thread_pool(const nlohmann::json& json) {
  const auto config{static_cast<NumaThreadPoolConfig>(json)};
  config.check();
  return std::make_shared<NumaThreadPool>(config);
}

using create_thread_pool_t = thread_pool_ptr_t (*)(const nlohmann::json& json);

static std::unordered_map<std::string, create_thread_pool_t> tp_impls{
    {"simple", create_simple_thread_pool}, {"numa", create_numa_thread_pool}};

create_thread_pool_t resolve_thread_pool_implementation(const std::string& impl_name) {
  auto it{tp_impls.find(impl_name)};
  NVE_CHECK_(it != tp_impls.end(), "Thread pool implementation '", impl_name, "' not found!");
  return it->second;
}

thread_pool_ptr_t create_thread_pool(const nlohmann::json& json) {
  // Determine what implementation should be used.
  std::string impl_name{"simple"};
  auto it{json.find("implementation")};
  if (it != json.end()) {
    it->get_to(impl_name);
  }
  create_thread_pool_t create_tp{resolve_thread_pool_implementation(impl_name)};

  NVE_LOG_INFO_("Creating thread pool via implementation '", impl_name, "\'.");
  return create_tp(json);
}

static nlohmann::json default_tp_config{R"({"name": "nve_default_tp"})"};
static thread_pool_ptr_t default_tp;

void configure_default_thread_pool(const nlohmann::json& json) {
  NVE_CHECK_(!default_tp, "Cannot reconfigure the default thread pool after it was created.");
  default_tp_config = json;
}

thread_pool_ptr_t default_thread_pool() {
  if (!default_tp) {
    NVE_LOG_VERBOSE_("Creating default thread pool...");
    default_tp = create_thread_pool(default_tp_config);
  }
  return default_tp;
}

void SimpleThreadPoolConfig::check() const {
  base_type::check();

  NVE_CHECK_(num_workers >= 0);
}

void from_json(const nlohmann::json& json, SimpleThreadPoolConfig& conf) {
  using base_type = SimpleThreadPoolConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(num_workers);
}

void to_json(nlohmann::json& json, const SimpleThreadPoolConfig& conf) {
  using base_type = SimpleThreadPoolConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(num_workers);
}

SimpleThreadPool::SimpleThreadPool(const SimpleThreadPoolConfig& config) : base_type(config) {
  int64_t num_workers{config.num_workers};
  if (num_workers <= 0) {
    num_workers = std::thread::hardware_concurrency();
  }
  NVE_CHECK_(num_workers > 0, "ThreadPool must have at least one worker!");

  workers_.reserve(static_cast<size_t>(num_workers));
  for (int64_t i{}; i < num_workers; ++i) {
    workers_.emplace_back(&SimpleThreadPool::worker_main_, this, i);
  }

  NVE_LOG_VERBOSE_("Created thread pool '", name, "' with ", workers_.size(), " workers.");
}

SimpleThreadPool::~SimpleThreadPool() {
  NVE_LOG_VERBOSE_("Shutting down thread pool '", name, "'.");
  if (workers_.empty()) {
    NVE_LOG_WARNING_("Threadpool `", name, "` double shutdown!");
    return;
  }

  // Queue termination tasks.
  {
    std::lock_guard lk(tasks_guard_);
    for (size_t i{}; i < workers_.size(); ++i) {
      tasks_.emplace();
    }
  }
  on_submit_.notify_all();

  // Await workers to exit.
  await(workers_.begin(), workers_.end());
  workers_.clear();
  NVE_LOG_VERBOSE_("Thread pool '", name, "' synchronized.");
}

ThreadPool::result_type SimpleThreadPool::submit(task_type task, const int64_t workgroup) {
  NVE_CHECK_(workgroup == 0, "`workgroup` is out of range!");
  NVE_CHECK_(!workers_.empty(), "ThreadPool `", name, "` is shutting down.");

  result_type res;
  {
    std::lock_guard lk(tasks_guard_);
    res = tasks_.emplace(std::move(task)).get_future();
  }
  NVE_LOG_VERBOSE_("Thread pool '", name, "'; submited 1 task to workgroup #", workgroup, '.');

  on_submit_.notify_one();
  return res;
}

int64_t SimpleThreadPool::submit_n(int64_t task_idx, const int64_t num_tasks,
                                   const indexed_task_type& task, result_type* results,
                                   const int64_t workgroup) {
  if (num_tasks <= 0) return task_idx;
  NVE_CHECK_(workgroup == 0, "`workgroup` is out of range!");
  NVE_CHECK_(!workers_.empty(), "ThreadPool `", name, "` is shutting down.");

  {
    std::lock_guard lk(tasks_guard_);
    auto& tasks{tasks_};

    if (results) {
      for (int64_t i{}; i < num_tasks; ++i) {
        results[i] = tasks.emplace(std::bind(task, task_idx++)).get_future();
      }
    } else {
      for (int64_t i{}; i < num_tasks; ++i) {
        tasks.emplace(std::bind(task, task_idx++));
      }
    }
  }
  NVE_LOG_VERBOSE_("Thread pool '", name, "'; submited ", num_tasks, " task to workgroup #",
                   workgroup, '.');

  on_submit_.notify_all();
  return task_idx;
}

void SimpleThreadPool::worker_main_(const int64_t worker_idx) {
  // Make thread identifyable at OS level.
  this_thread_name() = to_string(name, "/w", worker_idx);

  auto& tasks_guard{tasks_guard_};
  auto& tasks{tasks_};
  auto& on_submit{on_submit_};

  while (true) {
    NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Worker ", this_thread_name(), " enters 'idle' state."));
    packaged_task_type task;

    // Await next task submission.
    {
      std::unique_lock lk(tasks_guard);
      while (tasks.empty()) {
        on_submit.wait(lk, [&tasks] { return !tasks.empty(); });
      }
      task = std::move(tasks.front());
      tasks.pop();
    }

    // Invalid tasks are used to signal the worker to exit.
    if (!task.valid()) {
      NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Worker ", this_thread_name(), " exits."));
      return;
    }

    // Execute the task.
    NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Worker ", this_thread_name(), " enters 'busy' state."));
    task();
  }
}

void NumaWorkgroupConfig::check() const {
  NVE_CHECK_(cpu_socket_index >= -1, "Invalid value for `cpu_socket_index`!");
  NVE_CHECK_(cpu_socket_index < num_cpu_sockets(), "`cpu_socket_index` is out of range!");

  NVE_CHECK_(numa_node_index >= 0, "Invalid value for `numa_node_index`!");
  int64_t numa_nodes;
  if (cpu_socket_index < 0) {
    numa_nodes = num_numa_nodes();
  } else {
    numa_nodes = cpu_socket_num_numa_nodes(cpu_socket_index);
  }
  NVE_CHECK_(numa_node_index < numa_nodes, "`numa_node_index` is out of range!");

  NVE_CHECK_(num_workers >= 0, "Invalid value for `num_workers`!");
}

void from_json(const nlohmann::json& json, NumaWorkgroupConfig& conf) {
  NVE_READ_JSON_FIELD_(cpu_socket_index);
  NVE_READ_JSON_FIELD_(numa_node_index);
  NVE_READ_JSON_FIELD_(num_workers);
}

void to_json(nlohmann::json& json, const NumaWorkgroupConfig& conf) {
  json = json.object();

  NVE_WRITE_JSON_FIELD_(cpu_socket_index);
  NVE_WRITE_JSON_FIELD_(numa_node_index);
  NVE_WRITE_JSON_FIELD_(num_workers);
}

void NumaThreadPoolConfig::check() const {
  base_type::check();

  for (auto& workgroup : workgroups) {
    workgroup.check();
  }
}

void from_json(const nlohmann::json& json, NumaThreadPoolConfig& conf) {
  using base_type = NumaThreadPoolConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(workgroups);
}

void to_json(nlohmann::json& json, const NumaThreadPoolConfig& conf) {
  using base_type = NumaThreadPoolConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(workgroups);
}

NumaThreadPool::NumaThreadPool(const NumaThreadPoolConfig& config) : base_type(config) {
  std::vector<NumaWorkgroupConfig> workgroups{config.workgroups};
  if (workgroups.empty()) {
    const int64_t cpu_sockets{num_cpu_sockets()};
    workgroups.resize(static_cast<uint64_t>(cpu_sockets));
    for (int64_t i{}; i < cpu_sockets; ++i) {
      workgroups[static_cast<uint64_t>(i)] = {i, cpu_socket_numa_nodes(i).front(), 0};
    }
  }

  for (NumaWorkgroupConfig& w : workgroups) {
    if (w.num_workers == 0) {
      w.num_workers = numa_node_logical_cores(w.numa_node_index);
    }
  }

  tasks_guards_ = std::vector<std::mutex>(workgroups.size());
  tasks_ = std::vector<task_queue_type>(workgroups.size());
  on_submits_ = std::vector<std::condition_variable>(workgroups.size());

  int64_t num_workers{};
  for (const auto& workgroup : workgroups) {
    num_workers += workgroup.num_workers;
  }

  workers_.reserve(static_cast<uint64_t>(num_workers));
  for (uint64_t i{}; i < workgroups.size(); ++i) {
    const auto& workgroup{workgroups[i]};
    for (int64_t j{}; j < workgroup.num_workers; ++j) {
      workers_.emplace_back(&NumaThreadPool::worker_main_, this, i, j, workgroup.numa_node_index);
    }
    NVE_LOG_VERBOSE_("Workgroup #", i, " has ", workgroup.num_workers, " workers @ NUMA node #",
                     workgroup.numa_node_index, ".");
  }

  NVE_LOG_VERBOSE_("Created thread pool '", name, "' with ", workers_.size(), " workers across ",
                   tasks_.size(), " workgroups.");
}

NumaThreadPool::~NumaThreadPool() {
  NVE_LOG_VERBOSE_("Shutting down thread pool '", name, "'.");
  if (workers_.empty()) {
    NVE_LOG_WARNING_("Threadpool `", name, "` double shutdown!");
    return;
  }

  // Queue termination tasks.
  const uint64_t num_workers_per_node{workers_.size() / tasks_.size()};
  for (uint64_t j{}; j < tasks_.size(); ++j) {
    {
      std::lock_guard lk(tasks_guards_[j]);
      for (uint64_t i{}; i < num_workers_per_node; ++i) {
        tasks_[j].emplace();
      }
    }
    on_submits_[j].notify_all();
  }

  // Await workers to exit.
  await(workers_.begin(), workers_.end());
  workers_.clear();
  NVE_LOG_VERBOSE_("Thread pool '", name, "' synchronized.");
}

ThreadPool::result_type NumaThreadPool::submit(task_type task, int64_t workgroup) {
  NVE_CHECK_(workgroup >= 0 && workgroup < num_workgroups(), "`workgroup` is out of range!");
  NVE_CHECK_(!workers_.empty(), "ThreadPool `", name, "` is shutting down.");

  result_type res;
  {
    std::lock_guard lk(tasks_guards_[static_cast<uint64_t>(workgroup)]);
    res = tasks_[static_cast<uint64_t>(workgroup)].emplace(std::move(task)).get_future();
  }
  NVE_LOG_VERBOSE_("Thread pool '", name, "'; submitted single task to workgroup #", workgroup,
                   '.');

  on_submits_[static_cast<uint64_t>(workgroup)].notify_one();
  return res;
}

int64_t NumaThreadPool::submit_n(int64_t task_idx, const int64_t num_tasks,
                                 const indexed_task_type& task, result_type* results,
                                 const int64_t workgroup) {
  if (num_tasks <= 0) return task_idx;
  NVE_CHECK_(workgroup >= 0 && workgroup < num_workgroups(), "`workgroup` is out of range!");
  NVE_CHECK_(!workers_.empty(), "ThreadPool `", name, "` is shutting down.");

  {
    std::lock_guard lk(tasks_guards_[static_cast<uint64_t>(workgroup)]);
    auto& tasks{tasks_[static_cast<uint64_t>(workgroup)]};
    if (results) {
      for (int64_t i{}; i < num_tasks; ++i) {
        results[i] = tasks.emplace(std::bind(task, task_idx++)).get_future();
      }
    } else {
      for (int64_t i{}; i < num_tasks; ++i) {
        tasks.emplace(std::bind(task, task_idx++));
      }
    }
  }

  NVE_LOG_VERBOSE_("Thread pool '", name, "'; submitted batch ", num_tasks, " tasks to workgroup #",
                   workgroup, '.');
  on_submits_[static_cast<uint64_t>(workgroup)].notify_all();
  return task_idx;
}

void NumaThreadPool::worker_main_(const uint64_t workgroup_idx, const int64_t worker_idx,
                                  const int64_t numa_node_idx) {
  // Select desired NUMA node, and make thread identifyable at OS level.
  bind_thread_to_numa_node(numa_node_idx);
  this_thread_name() = to_string(name, "/g", workgroup_idx, "/w", worker_idx, "/n", numa_node_idx);

  auto& tasks_guard{tasks_guards_[workgroup_idx]};
  auto& tasks{tasks_[workgroup_idx]};
  auto& on_submit{on_submits_[workgroup_idx]};

  while (true) {
    NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Worker ", this_thread_name(), " enters 'idle' state."));
    packaged_task_type task;

    // Await next task submission.
    {
      std::unique_lock lk(tasks_guard);
      while (tasks.empty()) {
        on_submit.wait(lk, [&tasks] { return !tasks.empty(); });
      }
      task = std::move(tasks.front());
      tasks.pop();
    }

    // Invalid tasks are used to signal the worker to exit.
    if (!task.valid()) {
      NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Worker ", this_thread_name(), " exits."));
      return;
    }

    // Execute the task.
    NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Worker ", this_thread_name(), " enters 'busy' state."));
    task();
  }
}

}  // namespace nve
