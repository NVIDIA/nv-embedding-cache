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

#include <array>
#include <common.hpp>
#include <future>
#include <string>
#include <vector>

namespace nve {

/**
 * Block until the result is available. This will re-throw any exceptions.
 *
 * @param result The future to await.
 * @return The result embedded in the future.
 */
template <typename T>
inline T await(std::future<T>& future) {
  NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Awaiting future result."));
  return future.get();
}

inline void await(std::thread& thread) {
  NVE_IF_DEBUG_(NVE_LOG_VERBOSE_("Awaiting thread (joinable=", thread.joinable(), ")."));
  thread.join();
}

/**
 * Block until all results are availble. This will re-throw any exceptions.
 */
template <typename It>
inline void await(It first, It last) {
  for (; first != last; ++first) {
    await(*first);
  }
}

template <typename It>
inline void await_n(It first, const uint64_t n) {
  await(first, first + n);
}

template <typename It>
inline void await_n(It first, const int64_t n) {
  NVE_ASSERT_(n >= 0);
  await(first, first + n);
}

struct ThreadPoolConfig {
  std::string name{"nve_tp"};

  void check() const;
};

/**
 * A thread pool is a collection of worker threads that can execute tasks.
 *
 * Thread pools may group workers into workgroups. How they do that is implementation-dependent.
 * Workgroups are numbered from 0 to num_workgroups() - 1.
 */
class ThreadPool {
 public:
  using callable_type = void();
  using task_type = std::function<callable_type>;
  using indexed_task_type = std::function<void(int64_t)>;
  using packaged_task_type = std::packaged_task<callable_type>;
  using result_type = std::future<void>;

  const std::string name;

  NVE_PREVENT_COPY_AND_MOVE_(ThreadPool);

  ThreadPool() = delete;

  ThreadPool(const ThreadPoolConfig& config);

  virtual ~ThreadPool() = default;

  /**
   * @return Total number of worker threads in the thread pool.
   */
  virtual int64_t num_workers() const noexcept = 0;

  /**
   * @return Number of distinct groups of workers.
   */
  virtual int64_t num_workgroups() const noexcept = 0;

  /**
   * Schedule a task for execution in the thread pool.
   *
   * @param task The task.
   * @param workgroup Workgroup to submit the task to.
   * @return An "await"-able object to keep track of the thread-pool's progress.
   */
  virtual result_type submit(task_type task, int64_t workgroup = 0) = 0;

  /**
   * Schedule a task for execution by the thread pool, and block until the task is complete.
   *
   * @param task The task.
   * @param workgroup Workgroup to submit the task to.
   */
  inline void execute(task_type task, int64_t workgroup = 0) {
    result_type res{submit(std::move(task), workgroup)};
    await(res);
  }

  /**
   * Schedule a number of tasks for execution in the thread pool.
   *
   * @param task_idx Index, from which to start assigning task numbers.
   * @param num_tasks Number of tasks to execute.
   * @param task Indexed task to execute.
   * @param results An array to fill with "await"-able objects that keep track of the thread-pool's
   * progress.
   * @param workgroup Workgroup to submit the tasks to.
   * @return Next `task_idx`.
   */
  virtual int64_t submit_n(int64_t task_idx, int64_t num_tasks, const indexed_task_type& task,
                           result_type* results = nullptr, int64_t workgroup = 0);

  /**
   * Schedule a number of tasks, and await their conclusion.
   *
   * @param task_idx Index, from which to start assigning task numbers.
   * @param num_tasks Number of tasks to execute.
   * @param task Indexed task to execute.
   * @param workgroup Workgroup to submit the tasks to.
   * @return Next `task_idx`.
   */
  inline int64_t execute_n(int64_t task_idx, const int64_t num_tasks, const indexed_task_type& task,
                           const int64_t workgroup = 0) {
    if (num_tasks <= 0) return task_idx;

    static constexpr int64_t N{1024 / sizeof(result_type)};
    static_assert(N > 0);

    if (num_tasks < N) {
      std::array<result_type, N> results;
      task_idx = submit_n(task_idx, num_tasks, task, results.data(), workgroup);
      await_n(results.data(), num_tasks);
    } else {
      std::vector<result_type> results(static_cast<size_t>(num_tasks));
      task_idx = submit_n(task_idx, num_tasks, task, results.data(), workgroup);
      await(results.begin(), results.end());
    }

    return task_idx;
  }

  /**
   * Schedule a number of tasks for execution by the thread pool, and return immediately.
   *
   * @param task_idx Index, from which to start assigning task numbers.
   * @param num_tasks Number of tasks to execute.
   * @param task Indexed task to execute.
   * @param results An array to fill with "await"-able objects that keep track of the thread-pool's
   * progress.
   * @param first_workgroup First workgroup to submit the tasks to.
   * @param last_workgroup Akin to std-lib, the last workgroup (not inclusive).
   * @param tasks_per_workgroup Number of tasks to submit in bursts per workgroup.
   * @return Next `task_idx`.
   */
  template <typename It>
  inline int64_t submit_n(int64_t task_idx, int64_t num_tasks, const indexed_task_type& task,
                          result_type* results, const It& first_workgroup, const It& last_workgroup,
                          int64_t tasks_per_workgroup) {
    NVE_CHECK_(first_workgroup != last_workgroup, "Must select at least one workgroup!");
    tasks_per_workgroup = std::max<int64_t>(tasks_per_workgroup, 1);

    const int64_t num_workgroups{this->num_workgroups()};
    for (It it{first_workgroup}; num_tasks > 0; num_tasks -= tasks_per_workgroup) {
      const int64_t wg{*it++ % num_workgroups};
      if (it == last_workgroup) {
        it = first_workgroup;
      }

      const int64_t n{std::min(num_tasks, tasks_per_workgroup)};
      task_idx = submit_n(task_idx, n, task, results, wg);
      results += tasks_per_workgroup;
    }

    return task_idx;
  }

  /**
   * Schedule a number of tasks, and await their conclusion.
   *
   * @param task_idx Index, from which to start assigning task numbers.
   * @param num_tasks Number of tasks to execute.
   * @param task Indexed task to execute.
   * @param first_workgroup First workgroup to submit the tasks to.
   * @param last_workgroup Akin to std-lib, the last workgroup (not inclusive).
   * @param tasks_per_workgroup Number of tasks to submit to each workgroup.
   * @return Next `task_idx`.
   */
  template <typename It>
  inline int64_t execute_n(int64_t task_idx, const int64_t num_tasks, const indexed_task_type& task,
                           const It& first_workgroup, const It& last_workgroup,
                           int64_t tasks_per_workgroup) {
    if (num_tasks <= 0) return task_idx;

    static constexpr int64_t N{1024 / sizeof(result_type)};
    static_assert(N > 0);

    if (num_tasks < N) {
      std::array<result_type, N> results;
      task_idx = submit_n(task_idx, num_tasks, task, results.data(), first_workgroup,
                          last_workgroup, tasks_per_workgroup);
      await_n(results.data(), num_tasks);
    } else {
      std::vector<result_type> results(static_cast<size_t>(num_tasks));
      task_idx = submit_n(task_idx, num_tasks, task, results.data(), first_workgroup,
                          last_workgroup, tasks_per_workgroup);
      await(results.begin(), results.end());
    }

    return task_idx;
  }

  template <typename Collection>
  inline int64_t submit_n(const int64_t task_idx, const int64_t num_tasks,
                          const indexed_task_type& task, result_type* const results,
                          const Collection& workgroups, const int64_t tasks_per_workgroup) {
    return submit_n(task_idx, num_tasks, task, results, workgroups.begin(), workgroups.end(),
                    tasks_per_workgroup);
  }

  template <typename Collection>
  inline int64_t execute_n(const int64_t task_idx, const int64_t num_tasks,
                           const indexed_task_type& task, const Collection& workgroups,
                           const int64_t tasks_per_workgroup) {
    return execute_n(task_idx, num_tasks, task, workgroups.begin(), workgroups.end(),
                     tasks_per_workgroup);
  }
};

}  // namespace nve
