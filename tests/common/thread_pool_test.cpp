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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <thread_pool.hpp>

using namespace std::chrono_literals;
using namespace nve;

TEST(SimpleThreadPool, Init) {
  thread_pool_ptr_t tp{create_thread_pool(R"({"name": "tp", "num_workers": 1})"_json)};
  EXPECT_EQ(1, tp->num_workers());
}

TEST(SimpleThreadPool, SingleTask) {
  thread_pool_ptr_t tp{create_thread_pool(R"({"name": "tp"})")};

  std::atomic<int64_t> tmp{0};
  auto res = tp->submit([&tmp]() { tmp++; });
  EXPECT_TRUE(res.valid());
  res.get();
  EXPECT_EQ(tmp, 1);
}

TEST(SimpleThreadPool, ManyTasks) {
  thread_pool_ptr_t tp{create_thread_pool(R"({"name": "tp"})")};

  constexpr size_t NUM_TASKS = 10000;
  bool tmp_vec[NUM_TASKS];
  for (size_t i = 0; i < NUM_TASKS; i++) {
    tmp_vec[i] = false;
  }

  tp->execute_n(0, NUM_TASKS, [&tmp_vec](size_t i) { tmp_vec[i] = true; });
  for (size_t i = 0; i < NUM_TASKS; i++) {
    EXPECT_TRUE(tmp_vec[i]);
  }
}

TEST(SimpleThreadPool, ParallelSubmit) {
  thread_pool_ptr_t tp{create_thread_pool(R"({"name": "tp"})")};

  constexpr size_t NUM_TASKS = 1234;
  constexpr size_t NUM_WAVES = 7;
  std::atomic<int64_t> tmp_vec[NUM_TASKS];
  for (size_t i = 0; i < NUM_TASKS; i++) {
    tmp_vec[i] = 0;
  }

  // Launch multiple waves to submit at the same time
  std::vector<std::shared_ptr<std::thread>> wave_threads;
  wave_threads.reserve(NUM_WAVES);

  const std::function<void(size_t)>& task = [&tmp_vec](size_t idx) { tmp_vec[idx]++; };

  for (size_t i = 0; i < NUM_WAVES; i++) {
    wave_threads.emplace_back(std::make_shared<std::thread>([task, &tp]() {
      std::vector<std::future<void>> results;
      results.reserve(NUM_TASKS);
      for (size_t j = 0; j < NUM_TASKS; j++) {
        results.emplace_back(tp->submit(std::bind(task, j)));
      }
      for (auto& r : results) {
        r.get();
      }
    }));
  }

  for (auto t : wave_threads) {
    t->join();
  }

  for (size_t i = 0; i < NUM_TASKS; i++) {
    EXPECT_EQ(tmp_vec[i], NUM_WAVES);
  }
}

TEST(SimpleThreadPool, ParallelExecute) {
  thread_pool_ptr_t tp{create_thread_pool(R"({"name": "tp"})")};

  constexpr size_t NUM_TASKS = 1234;
  constexpr size_t NUM_WAVES = 7;
  std::atomic<int64_t> tmp_vec[NUM_TASKS];
  for (size_t i = 0; i < NUM_TASKS; i++) {
    tmp_vec[i] = 0;
  }

  // Launch multiple waves to execute_n at the same time
  std::vector<std::shared_ptr<std::thread>> wave_threads;
  wave_threads.reserve(NUM_WAVES);

  const std::function<void(size_t)>& task = [&tmp_vec](size_t idx) { tmp_vec[idx]++; };

  for (size_t i = 0; i < NUM_WAVES; i++) {
    wave_threads.emplace_back(
        std::make_shared<std::thread>([task, &tp]() { tp->execute_n(0, NUM_TASKS, task); }));
  }

  for (auto t : wave_threads) {
    t->join();
  }

  for (size_t i = 0; i < NUM_TASKS; i++) {
    EXPECT_EQ(tmp_vec[i], NUM_WAVES);
  }
}

TEST(NumaThreadPool, Init) {
  thread_pool_ptr_t tp{create_thread_pool(R"({
    "implementation": "numa",
    "workgroups": [
      {"cpu_socket_index": 0, "numa_node_index": 0, "num_workers": 1},
      {"cpu_socket_index": -1, "numa_node_index": 0, "num_workers": 1}
    ]
  })"_json)};

  EXPECT_EQ(2, tp->num_workgroups());
  EXPECT_EQ(2, tp->num_workers());
}

const nlohmann::json tp_conf{R"({
  "implementation": "numa",
  "workgroups": [
    {"cpu_socket_index": 0, "numa_node_index": 0},
    {"cpu_socket_index": -1, "numa_node_index": 0}
  ]
})"_json};

TEST(NumaThreadPool, SingleTask) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  std::atomic<int64_t> tmp{0};
  auto res{tp->submit([&tmp]() { tmp.fetch_add(1, std::memory_order_relaxed); })};
  EXPECT_TRUE(res.valid());
  await(res);
  EXPECT_EQ(tmp.load(std::memory_order_relaxed), 1);
}

TEST(NumaThreadPool, ManyTasks) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  constexpr int64_t num_tasks{10000};
  bool tmp[num_tasks]{};

  tp->execute_n(0, num_tasks, [&tmp](int64_t i) { tmp[i] = true; });

  for (bool t : tmp) {
    EXPECT_TRUE(t);
  }
}

TEST(NumaThreadPool, ParallelSubmit) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  constexpr int64_t num_tasks{1234};
  std::atomic_int64_t acc[num_tasks];
  for (std::atomic<int64_t>& a : acc) {
    a.store(0, std::memory_order_relaxed);
  }
  const std::function<void(int64_t)>& task = [&acc](int64_t idx) {
    acc[idx].fetch_add(1, std::memory_order_relaxed);
  };

  // Launch multiple waves with submit_n at the same time.
  constexpr int64_t num_threads{7};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::atomic_int64_t ready{0};
  std::vector<ThreadPool::result_type> res(num_tasks * num_threads);

  for (int64_t i{}; i < num_threads; ++i) {
    threads.emplace_back([i, &tp, &task, &ready, &res]() {
      // Spinlock until all submission threads have started up.
      ready.fetch_add(1, std::memory_order_relaxed);
      while (ready.load(std::memory_order_relaxed) < num_threads) {
        std::this_thread::yield();
      }

      tp->submit_n(0, num_tasks, task, &res[static_cast<uint64_t>(num_tasks * i)]);
      ready.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // Ensure sub-threaads have finished submitting.
  while (ready.load(std::memory_order_relaxed) < num_threads * 2) {
    std::this_thread::yield();
  }
  await(res.begin(), res.end());

  // Check results.
  for (std::atomic_int64_t& a : acc) {
    EXPECT_EQ(a.load(std::memory_order_relaxed), num_threads);
  }

  // Ensure sub-threads are joined, before we destroy the thread objects.
  await(threads.begin(), threads.end());
}

TEST(NumaThreadPool, ParallelExecute) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  constexpr int64_t num_tasks{1234};
  std::atomic_int64_t acc[num_tasks];
  for (std::atomic<int64_t>& a : acc) {
    a.store(0, std::memory_order_relaxed);
  }
  const std::function<void(int64_t)>& task = [&acc](int64_t idx) {
    acc[idx].fetch_add(1, std::memory_order_relaxed);
  };

  // Launch multiple waves to execute_n at the same time.
  constexpr int64_t num_threads{7};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::atomic_int64_t ready{0};

  for (int64_t i{}; i < num_threads; ++i) {
    threads.emplace_back([&tp, &task, &ready]() {
      // Spinlock until all submission threads have started up.
      ready.fetch_add(1, std::memory_order_relaxed);
      while (ready.load(std::memory_order_relaxed) < num_threads) {
        std::this_thread::yield();
      }

      tp->execute_n(0, num_tasks, task);
    });
  }
  await(threads.begin(), threads.end());

  for (std::atomic_int64_t& a : acc) {
    EXPECT_EQ(a.load(std::memory_order_relaxed), num_threads);
  }
}

TEST(NumaThreadPool, ManualScheduleRoundRobin) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  constexpr int64_t wave_size{37};
  int64_t tmp[wave_size]{};

  ThreadPool::result_type res[wave_size];
  for (int64_t i{}; i != wave_size; ++i) {
    res[i] = tp->submit(
        [i, &tmp]() {
          std::this_thread::sleep_for(0.1s);
          NVE_LOG_INFO_("Workload processed: ", i);
          tmp[i] += 101;
        },
        i % tp->num_workgroups());
  }
  await_n(res, wave_size);

  for (int64_t t : tmp) {
    EXPECT_EQ(t, 101);
  }
};

TEST(NumaThreadPool, ExecuteInWaves) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  constexpr int64_t wave_size{37};
  int64_t tmp[wave_size]{};

  constexpr int64_t num_waves{11};
  for (int64_t i{}; i != num_waves; ++i) {
    tp->execute_n(
        0, wave_size,
        [i, &tmp](const int64_t j) {
          std::this_thread::sleep_for(0.1s);
          tmp[static_cast<size_t>(j)] += j;
          NVE_LOG_INFO_("Workload processed: ", i, '-', j);
        },
        i % tp->num_workgroups());
  }

  for (size_t j{}; j != wave_size; ++j) {
    EXPECT_EQ(tmp[j], j * num_waves);
  }
}

TEST(NumaThreadPool, MassiveQueuing) {
  thread_pool_ptr_t tp{create_thread_pool(tp_conf)};

  constexpr int64_t wave_size{37};
  std::atomic_int64_t acc[wave_size];
  for (std::atomic_int64_t& a : acc) {
    a.store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> feeders;
  std::mutex results_access;
  std::vector<ThreadPool::result_type> results;

  constexpr int64_t num_feeders{3};
  constexpr int64_t num_waves{5};

  for (int64_t i{}; i != num_waves; ++i) {
    feeders.clear();
    results.clear();

    for (int64_t k{}; k != num_feeders; ++k) {
      feeders.emplace_back([i, k, &tp, &acc, &results_access, &results]() {
        this_thread_name() = to_string("feeder ", i, '/', k);

        for (int64_t j{}; j != wave_size; ++j) {
          ThreadPool::result_type result{tp->submit(
              [i, j, k, &acc]() {
                std::this_thread::sleep_for(0.1s);
                acc[j].fetch_add(j, std::memory_order_relaxed);
                NVE_LOG_INFO_("Workload processed: ", k, '/', i, '-', j);
              },
              j % tp->num_workgroups())};

          std::lock_guard lock(results_access);
          results.emplace_back(std::move(result));
        }
      });
    }

    await(feeders.begin(), feeders.end());
    await(results.begin(), results.end());
  }

  for (size_t j{}; j != wave_size; ++j) {
    EXPECT_EQ(acc[j], j * num_waves * num_feeders);
  }
}
