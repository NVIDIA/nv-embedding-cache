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

#include <unordered_set>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <cuda_support.hpp>
#include <thread_pool.hpp>
#include <execution_context.hpp>
#include <ecache/embed_cache.h>

namespace nve {

class DefaultECEvent;

template <>
constexpr bool is_success(const ECError& result) noexcept {
  return result == ECERROR_SUCCESS;
}

template <>
class RuntimeError<ECError> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[],
                      const ECError& error, const std::string& hint) noexcept
      : base_type(file, line, expr, hint), error_{error} {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that), error_{that.error_} {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    error_ = that.error_;
    return *this;
  }

  inline ECError error() const noexcept { return error_; }

  virtual std::string to_string() const override;

 private:
  ECError error_;
};

/**
 * Auxiliary class to keep track of all existing execution contexts so modify ops can sync with them.
 */
class ContextRegistry {
public:
  ContextRegistry() = default;
  NVE_PREVENT_COPY_AND_MOVE_(ContextRegistry);
  void add_context(const ExecutionContext* ctx);

  void remove_context(const ExecutionContext* ctx);

  std::shared_ptr<DefaultECEvent> create_sync_event();

  inline bool empty() const { return contexts_.empty(); }
private:
  std::mutex mutex_;
  std::unordered_set<cudaStream_t> lookup_streams_;
  std::unordered_set<const ExecutionContext*> contexts_; // Holding raw pointers instead of shared_ptr to avoid circular dependency
  void update_streams();
};

/**
 * StreamCoordinator object will set the correct cuda event dependencies between cuda streams during construction and destruction.
 * In between c'tor and d'tor, queue kernels to queue_stream
 */
class StreamCoordinator {
public:
  NVE_PREVENT_COPY_AND_MOVE_(StreamCoordinator);
  StreamCoordinator(const cudaStream_t& caller_stream, const cudaStream_t& run_stream) :
    queue_stream(run_stream ? run_stream : caller_stream), caller_stream_(caller_stream)
  {
    CreateStreamDependency(caller_stream_, queue_stream);
  }
  ~StreamCoordinator() {
    CreateStreamDependency(queue_stream, caller_stream_);
  }

  /**
   * Create a cuda event dependency between src and dst.
   * Specifically, commands queued for dst after this call will wait for all commands queued for src before this call.
   * An ad-hoc cudaEvent will be created for this call (this will cause a small overhead, measured 0.21us on my machine)
   */
  static inline void CreateStreamDependency(const cudaStream_t& src, const cudaStream_t& dst) {
    if (src != dst) {
      cudaEvent_t e;
      NVE_CHECK_(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      NVE_CHECK_(cudaEventRecord(e, src));
      NVE_CHECK_(cudaStreamWaitEvent(dst, e));
      NVE_CHECK_(cudaEventDestroy(e));
    }
  }

  const cudaStream_t queue_stream;
private:
  const cudaStream_t caller_stream_;
};

class LayerExecutionContext: public ExecutionContext {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(LayerExecutionContext);
  
  LayerExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator,
    const std::vector<context_ptr_t>& table_contexts)
    : ExecutionContext(lookup_stream, modify_stream, std::move(thread_pool), std::move(allocator)),
      table_contexts_(std::move(table_contexts)), parallel_task_res_(table_contexts_.size()) {}

  virtual ~LayerExecutionContext() { wait(); }

  virtual void wait() override {
    for (auto& res : parallel_task_res_) {
      if (res.valid()) {
        res.get(); // This needs to precede the stream sync since it queues work on the streams
      }
    }
    ExecutionContext::wait();
  }

  void submit_parallel_task(ThreadPool::task_type task, int64_t table_id) {
    parallel_task_res_.at(static_cast<size_t>(table_id)) = thread_pool_->submit(std::move(task));
  }

  // Public members since only this file can access them (the actual class isn't exposed)
  std::vector<context_ptr_t> table_contexts_;
  // For now only auto-insert can be offloaded and only one can exist at a given time for every table
  // todo: extend to multiple futures when needed.
  std::vector<ThreadPool::result_type> parallel_task_res_;
};

/**
 * Synchronization primitivaes to allow for try_lock w/o the wierd restrictions of c++;
 * mutex can fail on try lock w/o reason, while the wrapper class have same restriction as mutex while 
 * throwing exception when try_lock a locked object
 */
class TestableMutex {
public:
    TestableMutex() : flag(false) {}

    // check whether the lock is free if true, performs a lock
    bool try_lock() {
        std::unique_lock l(m);
        if (!flag)
        {
            flag = true;
            return true;
        }
        else
        {
          return false;
        }
    }

    bool is_locked() {
      std::unique_lock l(m);
      return flag;
    }

    void lock() {
        std::unique_lock l(m);
        cv.wait(l, [this]{ return !this->flag; });
        flag = true;
    }

    void unlock() {
        std::unique_lock l(m);
        flag = false;
        cv.notify_all();
    }
private:
    bool flag;
    std::mutex m;
    std::condition_variable cv;
};

class InsertHeuristic;
class Table;
template<typename T>
class BufferWrapper;

class AutoInsertHandler final {
public:
  AutoInsertHandler(
    std::shared_ptr<InsertHeuristic> heuristic,
    table_ptr_t table,
    const int64_t table_id,
    allocator_ptr_t allocator,
    const int64_t min_insert_wait,
    const int64_t min_insert_size,
    const int64_t key_size,
    const int32_t layer_gpu_device);

  ~AutoInsertHandler();

  void auto_insert(
    std::shared_ptr<LayerExecutionContext> layer_ctx,
    std::shared_ptr<BufferWrapper<const void>>& keys_bw,
    std::shared_ptr<BufferWrapper<void>>& output_bw,
    const float hitrate,
    const int64_t num_keys,
    const int64_t output_stride);
  void lock_modify();
  void unlock_modify();

private:
  TestableMutex insert_lock_; // lock to prevent parallel auto-insert handling and modify parallel to auto-insert
  std::shared_ptr<InsertHeuristic> heuristic_;
  table_ptr_t table_;
  const int64_t table_id_;
  allocator_ptr_t allocator_;
  const int64_t min_insert_wait_;
  const int64_t min_insert_size_;
  const int64_t key_size_;
  const int32_t layer_gpu_device_;

  int64_t insert_freq_cnt_{0};
  int64_t collected_keys_{0};
  int64_t collected_output_stride_{0};

  // Handler keeps it own keys/data buffers for accumulation of small key sets
  std::unique_ptr<ResizeableBuffer> insert_keys_;
  std::unique_ptr<ResizeableBuffer> insert_data_;

  void collect_keys_and_data(
    std::shared_ptr<BufferWrapper<const void>>& keys_bw,
    std::shared_ptr<BufferWrapper<void>>& output_bw,
    cudaStream_t lookup_stream,
    const int64_t num_keys);
  void launch_insert(
    std::shared_ptr<LayerExecutionContext> layer_ctx,
    const int64_t output_stride);
};

}  // namespace nve
