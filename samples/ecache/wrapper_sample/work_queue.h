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
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <deque>
#include "cuda_ops/cuda_common.h"

class WorkQueue {
public:
    WorkQueue(int device_id, size_t num_threads) : m_device_id(device_id), m_shutdown(false) {
        for (size_t i=0 ; i<num_threads ; i++) {
            m_threads.emplace_back(std::make_shared<std::thread>(&WorkQueue::ThreadMainLoop, this));
        }
    }
    ~WorkQueue() {
        Shutdown();
    }

    std::future<void> Submit(std::function<void()> task) {
        // Package task
        std::packaged_task<void()> package(std::move(task));
        std::future<void> res = package.get_future();
        
        // Add to queue
        {
            std::unique_lock lk(m_mutex);
            m_tasks.emplace_back(std::move(package));
        }

        // Signal a worker thread
        m_cv.notify_one();

        return res;
    }
    void Shutdown()
    {
        const auto num_threads = m_threads.size();
        for (size_t t=0 ; t<num_threads ; t++) {
            Submit([this]() { m_shutdown = true; });
        }
        for (auto t : m_threads) {
            t->join();
        }
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<std::shared_ptr<std::thread>> m_threads;
    std::deque<std::packaged_task<void()>> m_tasks;
    int m_device_id;
    bool m_shutdown;

    void ThreadMainLoop() {
        NVE_CHECK_(cudaSetDevice(m_device_id));
        while (true) {
            std::packaged_task<void ()> task;
            {
                std::unique_lock lk(m_mutex);
                if (m_shutdown) {
                    return;
                }
                while (m_tasks.empty()) {
                    m_cv.wait(lk, [this]{ return !(m_tasks.empty()); });

                    if (m_shutdown) {
                        return;
                    }
                }
                task = std::move(m_tasks.front());
                m_tasks.pop_front();
            }
            task();
        }
    }
};
