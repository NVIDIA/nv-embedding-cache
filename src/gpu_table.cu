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

#include <gpu_table.hpp>
#include <json_support.hpp>
#include <ecache/embedding_cache_combined.cuh>
#include <execution_context.hpp>
#include <layer_utils.hpp>
#include "cuda_ops/update_accumulate.cuh"
#include "cuda_ops/find_and_combine_kernel.cuh"
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/pipeline_gather.cuh"
#include "cuda_ops/gather_keys_data_ptrs.cuh"
#include "buffer_wrapper.hpp"

namespace nve {

static DataTypeFormat TRSDataToEC(DataType_t dt) {
  switch (dt) {
    case DataType_t::Float32:
      return DataTypeFormat::DATATYPE_FP32;
    case DataType_t::Float16:
      return DataTypeFormat::DATATYPE_FP16;
    default:
      NVE_THROW_NOT_IMPLEMENTED_();
  }
  return DataTypeFormat::NUM_DATA_TYPES_FORMATS;  // unreachable
}

// uvm_table and private_stream not supported in json initialization
// These should be handles created by the application at runtime and cannot be exported to text json files.
void from_json(const nlohmann::json& json, GPUTableConfig& conf) {
  NVE_READ_JSON_FIELD_(device_id);
  NVE_READ_JSON_FIELD_(cache_size);
  NVE_READ_JSON_FIELD_(row_size_in_bytes);
  NVE_READ_JSON_FIELD_(max_modify_size);
  NVE_READ_JSON_FIELD_(value_dtype);
  NVE_READ_JSON_FIELD_(count_misses);
  NVE_READ_JSON_FIELD_(disable_uvm_update);
  NVE_READ_JSON_FIELD_(uvm_cpu_accumulate);
  NVE_READ_JSON_FIELD_(data_storage_on_host);
  NVE_READ_JSON_FIELD_(modify_on_gpu);
  NVE_READ_JSON_FIELD_(kernel_mode_type);
  NVE_READ_JSON_FIELD_(kernel_mode_value);
}
void to_json(nlohmann::json& json, const GPUTableConfig& conf) {
  NVE_WRITE_JSON_FIELD_(device_id);
  NVE_WRITE_JSON_FIELD_(cache_size);
  NVE_WRITE_JSON_FIELD_(row_size_in_bytes);
  NVE_WRITE_JSON_FIELD_(max_modify_size);
  NVE_WRITE_JSON_FIELD_(value_dtype);
  NVE_WRITE_JSON_FIELD_(count_misses);
  NVE_WRITE_JSON_FIELD_(disable_uvm_update);
  NVE_WRITE_JSON_FIELD_(uvm_cpu_accumulate);
  NVE_WRITE_JSON_FIELD_(data_storage_on_host);
  NVE_WRITE_JSON_FIELD_(modify_on_gpu);
  NVE_WRITE_JSON_FIELD_(kernel_mode_type);
  NVE_WRITE_JSON_FIELD_(kernel_mode_value);
}

std::string RuntimeError<nve::ECError>::to_string() const {
  std::ostringstream o;

  const char* const what{this->what()};
  o << "EC runtime error " << static_cast<uint32_t>(error()) << " = '" << what << "' @ " << file()
    << ':' << line();
  const std::string& thread{thread_name()};
  if (!thread.empty()) {
    o << " in thread: '" << thread << '\'';
  }
  const char* const expr{expression()};
  if (what != expr) {
    o << ", expression: '" << expr << '\'';
  }
  o << '.';

  return o.str();
}

template <typename KeyType>
class GPUTableExecutionContext: public ExecutionContext {
 public:
  using CacheType = nve::EmbedCacheSA<KeyType, KeyType>;
  using cache_type = CacheType;
  using cache_ptr_type = std::shared_ptr<cache_type>;

  NVE_PREVENT_COPY_AND_MOVE_(GPUTableExecutionContext);
  
  GPUTableExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t& thread_pool,
    allocator_ptr_t& allocator,
    cache_ptr_type& cache,
    int64_t max_modify_size,
    bool count_misses,
    std::shared_ptr<ContextRegistry>& context_registry)
    : ExecutionContext(lookup_stream, modify_stream, thread_pool, allocator),
      max_modify_size_{max_modify_size}, count_misses_(count_misses), cache_{cache}, context_registry_(context_registry)
  {
    if (count_misses_) {
      NVE_CHECK_(cache_->performance_metric_create(miss_metric_, nve::PerformanceMerticTypes::MERTIC_COUNT_MISSES));
      NVE_CHECK_(cache_->performance_metric_reset(miss_metric_, lookup_stream_));
    } 
    NVE_CHECK_(cache_->lookup_context_create(lookup_context_, (count_misses_ ? &miss_metric_ : nullptr), (count_misses_ ? 1 : 0)),
              "Failed to create embedding cache lookup context.");
    if (max_modify_size_ < 1) {
      modify_context_.handle = 0;
    } else {
      NVE_CHECK_(cache_->modify_context_create(modify_context_, static_cast<uint32_t>(max_modify_size_)),
                "Failed to create embedding cache modification context.");
    }
    NVE_CHECK_(context_registry != nullptr, "Invalid context registry");
    context_registry_->add_context(this);
  }

  virtual ~GPUTableExecutionContext() {
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream_));
    NVE_CHECK_(cudaStreamSynchronize(modify_stream_));

    NVE_CHECK_(cache_->lookup_context_destroy(lookup_context_),
              "Failed to destroy embedding cache lookup context.");
    if (modify_context_.handle) {
      NVE_CHECK_(cache_->modify_context_destroy(modify_context_),
                          "Failed to destroy embedding cache modification context.");
    }
    if (count_misses_) {
      NVE_CHECK_(cache_->performance_metric_destroy(miss_metric_));
    }
    context_registry_->remove_context(this);
  }

  nve::LookupContextHandle lookup_context() { return lookup_context_; }
  nve::ModifyContextHandle modify_context() { return modify_context_; }
  nve::PerformanceMetric miss_metric() { return miss_metric_; }

 public:
  const int64_t max_modify_size_;
  const bool count_misses_;

 private:
  cache_ptr_type cache_;
  nve::LookupContextHandle lookup_context_;
  nve::ModifyContextHandle modify_context_;
  nve::PerformanceMetric miss_metric_;
  std::shared_ptr<ContextRegistry> context_registry_;
};

template <typename KeyType>
GpuTable<KeyType>::GpuTable(const GPUTableConfig& config, allocator_ptr_t allocator)
    : config_(config) {
  using CacheTypeDevice = nve::CacheSADeviceModify<KeyType, KeyType>;
  using CacheTypeHost = nve::CacheSAHostModify<KeyType, KeyType>;

  NVE_CHECK_((config.row_size_in_bytes % 2) == 0, "Invalid cache row size (must divide by 2)");
  allocator_ = allocator ? allocator : GetDefaultAllocator();
  NVE_CHECK_(allocator_ != nullptr, "Failed to get default allocator");

  // Create cache
  typename CacheType::CacheConfig cfg;
  cfg.embed_width_in_bytes = static_cast<uint64_t>(config_.row_size_in_bytes);
  cfg.cache_sz_in_bytes = static_cast<size_t>(config_.cache_size);
  cfg.num_tables = 1;  // Only supporting single table cache at this point
  cfg.allocate_data_on_host = config_.data_storage_on_host;

  if (config.modify_on_gpu) {
    cache_ = std::make_shared<CacheTypeDevice>(allocator_.get(), GetGlobalLogger(), cfg);
  } else {
    cache_ = std::make_shared<CacheTypeHost>(allocator_.get(), GetGlobalLogger(), cfg);
  }

  NVE_CHECK_(cache_->init(), "Failed to Init cache");
  NVE_CHECK_(static_cast<bool>(cache_), "Failed to create embedding cache");

  // Initialize context registry
  contexts_ = std::make_shared<ContextRegistry>();
  NVE_CHECK_(contexts_ != nullptr, "Failed to create context registry");
}

template <typename KeyType>
GpuTable<KeyType>::~GpuTable() {
  NVE_CHECK_(contexts_->empty(), "GPUTable destroyed with existing execution contexts");
}

template <typename KeyType>
void GpuTable<KeyType>::clear(context_ptr_t& ctx) {
  NVE_NVTX_SCOPED_FUNCTION_COL5_();
  ScopedDevice scope_device(config_.device_id);
  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");
  StreamCoordinator sc(gpu_table_ctx->get_modify_stream(), config_.private_stream);
  NVE_CHECK_(cache_->clear_cache(sc.queue_stream));
}

template <typename KeyType>
void GpuTable<KeyType>::erase(context_ptr_t& ctx, int64_t num_keys, const void* keys) {
  NVE_NVTX_SCOPED_FUNCTION_COL6_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_(keys != nullptr, "Invalid cache erase params");

  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto mod_ctx{gpu_table_ctx->modify_context()};

  auto ec_event = create_sync_event();
  StreamCoordinator sc(gpu_table_ctx->get_modify_stream(), config_.private_stream);

  NVE_CHECK_(cache_->invalidate(mod_ctx, static_cast<const key_type*>(keys),
                                num_keys, 0, ec_event.get(), sc.queue_stream),
             "Failed to call cache invalidate");
}

KernelType get_kernel_mode(const GPUTableConfig& config, int64_t num_keys) {
  switch (config.kernel_mode_type) {
    case 0: {
      int64_t constexpr default_threshold = 1 << 20;
      auto threshold = (config.kernel_mode_value == 0) ? default_threshold : static_cast<int64_t>(config.kernel_mode_value);
      return (num_keys < threshold) ? KernelType::LookupUVM : KernelType::SortGather;
    }
    case 1:
      return KernelType::LookupUVM;
    case 2:
      return KernelType::SortGather;
    case 3:
      return KernelType::PipelineGather;
    default:
      NVE_THROW_NOT_IMPLEMENTED_();
  }
}

template <typename KeyType, typename CacheType>
static void run_find_uvm(const GPUTableConfig& config, std::shared_ptr<CacheType> cache, context_ptr_t& ctx, int64_t num_keys, const void* keys,
                                     void* values, int64_t value_stride, cudaStream_t stream) {
  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");
  
  auto lookup_ctx{gpu_table_ctx->lookup_context()};
  KernelType mode = get_kernel_mode(config, num_keys);
  
  switch (mode) {
    case KernelType::DynamicKernel: 
    {
      NVE_ASSERT_(0);
      break;
    }
    case KernelType::LookupUVM:
    {
      NVE_CHECK_(cache->lookup(
        lookup_ctx,
        reinterpret_cast<const KeyType*>(keys),
        static_cast<size_t>(num_keys),
        reinterpret_cast<int8_t*>(values),
        static_cast<const int8_t*>(config.uvm_table),
        0, /*currTable*/
        value_stride,
        stream));
      break;
    }
    case KernelType::SortGather:
    {
      size_t aux_buffer_size = 0;
      NVE_CHECK_(cache->lookup_sort_gather(
        lookup_ctx,
        reinterpret_cast<const KeyType*>(keys),
        static_cast<size_t>(num_keys),
        reinterpret_cast<int8_t*>(values),
        static_cast<const int8_t*>(config.uvm_table),
        nullptr,
        aux_buffer_size,
        0, /*currTable*/
        value_stride,
        stream));
      int8_t* aux_buffer = (int8_t*)gpu_table_ctx->get_buffer("d_aux_buffer", aux_buffer_size, false);
      NVE_CHECK_(cache->lookup_sort_gather(
        lookup_ctx,
        reinterpret_cast<const KeyType*>(keys),
        static_cast<size_t>(num_keys),
        reinterpret_cast<int8_t*>(values),
        static_cast<const int8_t*>(config.uvm_table),
        aux_buffer, 
        aux_buffer_size, 
        0, /*currTable*/
        value_stride, 
        stream));
      break;
    }
    case KernelType::PipelineGather:
    {
      GatherKernelPipelineParams* params = reinterpret_cast<GatherKernelPipelineParams*>(config.kernel_mode_value);
      GatherKernelPipelineParams params_default{1024, 16};
      if (params == nullptr) {
        params = &params_default;
      }
      gather_flow_pipeline<KeyType, KeyType>(ctx, 
        cache,
        lookup_ctx,
        num_keys, 
        reinterpret_cast<const KeyType*>(keys), 
        reinterpret_cast<int8_t*>(values), 
        value_stride, 
        reinterpret_cast<int8_t*>(config.uvm_table), 
        config.row_size_in_bytes, 
        stream, 
        config.device_id,
        *params);
      break;
    }
    default:
      NVE_THROW_NOT_IMPLEMENTED_();
  }
}

template <typename KeyType>
void GpuTable<KeyType>::find(context_ptr_t& ctx, int64_t num_keys, const void* keys,
                             max_bitmask_repr_t* hit_mask, int64_t value_stride,
                             void* values, int64_t* value_sizes) const {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_(value_sizes == nullptr, "value_sizes must be nullptr for GPU table");

  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto lookup_ctx{gpu_table_ctx->lookup_context()};
  StreamCoordinator sc(gpu_table_ctx->get_lookup_stream(), config_.private_stream);
  if (config_.uvm_table) {
    std::unique_lock uvm_lock(uvm_table_mutex_, std::defer_lock); // don't lock yet, only lock when not using private stream
    if (!config_.private_stream) {
      uvm_lock.lock();
    }
    run_find_uvm<KeyType, CacheType >(config_, cache_, ctx, num_keys, keys, values, value_stride, sc.queue_stream);
  } else {
    NVE_CHECK_(cache_->lookup(
      lookup_ctx,
      reinterpret_cast<const KeyType*>(keys),
      static_cast<size_t>(num_keys),
      reinterpret_cast<int8_t*>(values),
      hit_mask,
      0, /*currTable*/
      value_stride,
      sc.queue_stream));
  }
}

template <typename KeyType>
void GpuTable<KeyType>::insert(context_ptr_t& ctx, int64_t num_keys, const void* keys,
                               int64_t value_stride, int64_t value_size, const void* values) {
  NVE_NVTX_SCOPED_FUNCTION_COL2_();
  ScopedDevice scope_device(config_.device_id);  
  NVE_CHECK_(keys && values, "Invalid cache insert params");
  NVE_CHECK_(value_size == config_.row_size_in_bytes, "Unsupported value_size");

  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto mod_ctx{gpu_table_ctx->modify_context()};

  if (num_keys > gpu_table_ctx->max_modify_size_) {
    NVE_LOG_WARNING_("Cache insert exceeds allowed size, partial insert will be performed");
  }

  if (config_.modify_on_gpu) {

    nve::DefaultGPUHistogram<KeyType> histogram(num_keys);
    size_t histAllocSize = histogram.getAllocSize();
    void* d_hist_storage = ctx->get_buffer("d_hist_storage", histAllocSize, false);
    cudaStream_t mod_stream = gpu_table_ctx->get_modify_stream();

    histogram.computeHistogram(reinterpret_cast<const KeyType*>(keys), num_keys,
                               reinterpret_cast<const int8_t*>(values),
                               value_stride, d_hist_storage, mod_stream);

    auto ec_event = create_sync_event();
    StreamCoordinator sc(mod_stream, config_.private_stream);

    auto num_keys_ = std::min(histogram.getNumBins(), static_cast<int64_t>(gpu_table_ctx->max_modify_size_));

    NVE_CHECK_(cache_->insert(
      mod_ctx,
      histogram.getKeys(),
      histogram.getPriority(),
      histogram.getData(),
      num_keys_,
      0, // tableIndex
      ec_event.get(),
      sc.queue_stream
    ), "Failed to call cache insert");
  } else {

    nve::DefaultHistogram histogram(
    static_cast<const key_type*>(keys),
    static_cast<size_t>(num_keys),
    static_cast<const int8_t*>(values),
    static_cast<size_t>(value_stride),
    false);

    auto ec_event = create_sync_event();
    StreamCoordinator sc(gpu_table_ctx->get_modify_stream(), config_.private_stream);

    NVE_CHECK_(cache_->insert(
      mod_ctx,
      histogram.get_keys(), 
      histogram.get_priority(),
      histogram.get_data(),
      histogram.get_num_bins(),
      0, // tableIndex
      ec_event.get(),
      sc.queue_stream
    ), "Failed to call cache insert");
  }
}

template <typename KeyType>
void GpuTable<KeyType>::update(context_ptr_t& ctx, int64_t num_keys, const void* keys,
                               int64_t value_stride, int64_t value_size, const void* values) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_(keys && values, "Invalid cache update params");
  NVE_CHECK_(value_size == config_.row_size_in_bytes, "Unsupported value_size");

  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto mod_ctx{gpu_table_ctx->modify_context()};
  const auto modify_stream = gpu_table_ctx->get_modify_stream();

  const auto max_modify_size = gpu_table_ctx->max_modify_size_;
  if (num_keys > max_modify_size) {
    NVE_LOG_WARNING_("Cache update exceeds configured size, update will be performed in parts (slower)");
  }

  for (int64_t k_start=0 ; k_start<num_keys ; k_start+=max_modify_size) {
    const auto k_end = std::min(k_start + max_modify_size, num_keys);

    auto ec_event = create_sync_event();
    StreamCoordinator sc(modify_stream, config_.private_stream);

    NVE_CHECK_(cache_->update(
      mod_ctx,
      static_cast<const key_type*>(keys) + k_start,
      static_cast<const int8_t*>(values) + (k_start * value_stride),
      value_stride,
      k_end - k_start,
      0, // tableIndex
      ec_event.get(),
      sc.queue_stream
    ), "Failed to call cache update");

    if (k_end != num_keys) {
      // Between consecutive updates we need to synchronize,
      // otherwise we overwrite the modify context state while the update is in progress.
      NVE_CHECK_(cudaStreamSynchronize(sc.queue_stream));
    }
  }

  // Update the UVM table
  if (config_.uvm_table && !config_.disable_uvm_update) {
    const auto keys_buffer_size = sizeof(KeyType) * num_keys;
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
    const void* d_keys = keys_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);

    std::unique_lock uvm_lock(uvm_table_mutex_, std::defer_lock); // don't lock yet, only lock when not using private stream
    cudaStream_t update_stream = modify_stream;
    std::shared_ptr<nve::DefaultECEvent> sync;

    if (config_.private_stream) {
      update_stream = config_.private_stream;
    } else {
      // lock mutex so we can gurantee no other kernel will be queued while we're setting up CUDA events
      uvm_lock.lock();
      // Set event on every other lookup stream
      // Not including modify streams since they will be also be sync'ed with the lookup streams thus implicitly synced among themeselves
      sync = contexts_->create_sync_event();
      sync->event_record();
      sync->event_wait_stream(update_stream);
    }

    // Set dependency on the modify stream (if different from update stream)
    StreamCoordinator::create_stream_dependency(modify_stream, update_stream);

    // Launch the update kernel
    UpdateTable<KeyType>(values, reinterpret_cast<const KeyType*>(d_keys),
                        config_.uvm_table,
                        static_cast<uint32_t>(config_.row_size_in_bytes),
                        static_cast<uint32_t>(value_stride),
                        static_cast<uint32_t>(config_.row_size_in_bytes),
                        static_cast<int32_t>(num_keys),
                        update_stream);

    cudaEvent_t uvm_update_event;
    NVE_CHECK_(cudaEventCreateWithFlags(&uvm_update_event, cudaEventDisableTiming));
    NVE_CHECK_(cudaEventRecord(uvm_update_event, update_stream));

    // If the modify stream is different from the update stream, it should wait for the update to complete
    if (modify_stream != update_stream) {
      NVE_CHECK_(cudaStreamWaitEvent(modify_stream, uvm_update_event));
    }

    if (!config_.private_stream) {
      // Have all lookup streams wait for the update to complete before starting
      for (auto s : sync->get_streams()) {
        NVE_CHECK_(cudaStreamWaitEvent(s, uvm_update_event));
      }
    }
    // if we needed to lock 'uvm_lock' - it will be released here.
  }
}

template <typename KeyType>
void GpuTable<KeyType>::update_accumulate(context_ptr_t& ctx, int64_t num_keys, const void* keys,
                                          int64_t update_stride, int64_t update_size,
                                          const void* updates, DataType_t update_dtype) {
  NVE_NVTX_SCOPED_FUNCTION_COL4_();
  ScopedDevice scope_device(config_.device_id);
  if (num_keys <= 0) {
    return;
  }
  NVE_CHECK_(keys && updates, "Invalid cache accumulate params");
  NVE_CHECK_(update_size == config_.row_size_in_bytes, "Row/update size mismatch");

  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto update_ec_type = TRSDataToEC(update_dtype);
  auto table_ec_type = TRSDataToEC(config_.value_dtype);
  auto mod_ctx{gpu_table_ctx->modify_context()};
  const auto modify_stream = gpu_table_ctx->get_modify_stream();
  StreamCoordinator sc(modify_stream, config_.private_stream);

  // First update the cache
  if (config_.private_stream) {
    auto lookup_ctx{gpu_table_ctx->lookup_context()};

    NVE_CHECK_(cache_->update_accumulate_no_sync(
      lookup_ctx,
      mod_ctx,
      static_cast<const key_type*>(keys),
      static_cast<size_t>(num_keys),
      static_cast<const int8_t*>(updates),
      0 /* currTable */,
      static_cast<size_t>(update_stride),
      update_ec_type,
      table_ec_type,
      sc.queue_stream
    ));
  } else {
    const auto max_modify_size = gpu_table_ctx->max_modify_size_;
    if (num_keys > max_modify_size) {
      NVE_LOG_WARNING_("Cache accumulate exceeds configured size, update will be performed in parts (slower)");
    }
    for (int64_t k_start=0 ; k_start<num_keys ; k_start+=max_modify_size) {
      const auto k_end = std::min(k_start + max_modify_size, num_keys);

      auto ec_event = create_sync_event();

      NVE_CHECK_(cache_->update_accumulate(
        mod_ctx,
        static_cast<const key_type*>(keys) + k_start,
        static_cast<const int8_t*>(updates) + (k_start * update_stride),
        update_stride,
        k_end - k_start,
        0, // tableIndex
        update_ec_type,
        table_ec_type,
        ec_event.get(),
        sc.queue_stream
      ), "Failed to call cache update");

      if (k_end != num_keys) {
        // Between consecutive updates (excluding update_accumulate_no_sync) we need to synchronize,
        // otherwise we overwrite the modify context state while the update is in progress.
        NVE_CHECK_(cudaStreamSynchronize(sc.queue_stream));
      }
    }
  }

  // Update the UVM table
  if (config_.uvm_table && !config_.disable_uvm_update) {
    const auto keys_buffer_size = sizeof(KeyType) * num_keys;
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);

    std::unique_lock uvm_lock(uvm_table_mutex_, std::defer_lock); // don't lock yet, only lock when not using private stream
    cudaStream_t update_stream = sc.queue_stream;
    std::shared_ptr<nve::DefaultECEvent> sync;

    if (!config_.private_stream) {
      // lock mutex so we can gurantee no other kernel will be queued while we're setting up CUDA events
      uvm_lock.lock();
      // Set event on every other lookup stream
      // Not including modify streams since they will be also be sync'ed with the lookup streams thus implicitly synced among themeselves
      sync = contexts_->create_sync_event();
      sync->event_record();
      sync->event_wait_stream(update_stream);
    }

    // Set dependency on the modify stream (if different from update stream)
    StreamCoordinator::create_stream_dependency(modify_stream, update_stream);

    if (config_.uvm_cpu_accumulate) {
      const void* h_keys = keys_bw->access_buffer(cudaMemoryTypeHost, true /*copy_content*/, sc.queue_stream);

      // copy updates to host
      NVE_CHECK_(update_size == update_stride, "Assuming updates are tightly packed");
      const auto update_buffer_size = num_keys * update_stride;
      void* h_updates = ctx->get_buffer("h_updates_uvm_accumulate", update_buffer_size, true); // Create a temporary buffer for host updates (not using wrapper so we can manually copy updates in parts)

      const int64_t num_threads{ctx->get_thread_pool()->num_workers()};
      constexpr int64_t keys_per_task = 512;
      const int64_t parallel_update_threshold = num_threads * keys_per_task / 2; // should be at least fill half a "wave"
      if (num_keys >= parallel_update_threshold) {
        // Use multiple cudaMemcpy calls to save latency of the CPU work
        const int64_t tasks_per_copy = num_threads;
        const int64_t keys_per_copy = keys_per_task * tasks_per_copy;
        const int64_t num_copies = (num_keys + keys_per_copy - 1) / keys_per_copy;
        std::vector<cudaEvent_t> copy_events(num_copies);

        // Launch copies
        for (int64_t i=0 ; i<num_copies ; i++) {
          int8_t* h_copy_start = static_cast<int8_t*>(h_updates) + (i * keys_per_copy * update_stride);
          const int8_t* d_copy_start = static_cast<const int8_t*>(updates) + (i * keys_per_copy * update_stride);
          const int64_t used_keys = std::min<int64_t>((i+1) * keys_per_copy, num_keys) - (i * keys_per_copy);
          const auto copy_size = used_keys * update_stride;
          NVE_CHECK_(cudaMemcpyAsync(h_copy_start, d_copy_start, copy_size, cudaMemcpyDefault, update_stream));
          NVE_CHECK_(cudaEventCreateWithFlags(&copy_events.at(i), cudaEventDisableTiming));
          NVE_CHECK_(cudaEventRecord(copy_events.at(i), update_stream));
        }

        // Now launch meta task to wait on cuda events then submit additional work to threadpool
        // Don't want to block many threads in the pool on cudaEventSynchronize
        std::atomic<int64_t> remaining_tasks(num_copies * tasks_per_copy);
        std::promise<void> barrier;
        auto future = barrier.get_future();

        int8_t* i8_table = reinterpret_cast<int8_t*>(config_.uvm_table);
        const int8_t* i8_h_updates = reinterpret_cast<const int8_t*>(h_updates);
        const KeyType* typed_keys = reinterpret_cast<const KeyType*>(h_keys);
        const auto row_size = config_.row_size_in_bytes;
        NVE_CHECK_(update_size % dtype_size(update_dtype) == 0);
        const auto elements_per_row = update_size / dtype_size(update_dtype);

        const auto update_launcher_task{[=, &barrier, &remaining_tasks]() {
          for (int64_t i=0 ; i<num_copies ; i++) {
            // First wait for the copy to complete
            NVE_CHECK_(cudaEventSynchronize(copy_events.at(i)));

            // Define the update tasks
            const auto update_task_fp16{[=, &barrier, &remaining_tasks] (const size_t idx) {
              const auto base_key = i * keys_per_copy;
              const int64_t start_key = base_key + (idx * keys_per_task);
              const int64_t end_key = std::min<int64_t>(start_key + keys_per_task, num_keys);
              const int64_t local_updates = end_key-start_key;

              std::vector<half*> dst(keys_per_task);
              std::vector<const half*> src(keys_per_task);
              for (int64_t i=0 ; i<local_updates ; i++) {
                dst[i] = reinterpret_cast<half*>(i8_table + (typed_keys[i+start_key] * row_size));
                src[i] = reinterpret_cast<const half*>(i8_h_updates + ((i+start_key) * update_stride));
              }
              for (int64_t i=0 ; i<local_updates ; i++) {
                for (int64_t j=0 ; j<elements_per_row ; j++) {
                  dst[i][j] += src[i][j];
                }
              }
              if ((--remaining_tasks) == 0) {
                barrier.set_value(); // release the future
              }
            }};
            const auto update_task_fp32{[=, &barrier, &remaining_tasks] (const int64_t idx) {
              const auto base_key = i * keys_per_copy;
              const int64_t start_key = base_key + (idx * keys_per_task);
              const int64_t end_key = std::min<int64_t>(start_key + keys_per_task, num_keys);
              const int64_t local_updates = end_key-start_key;

              std::vector<float*> dst(keys_per_task);
              std::vector<const float*> src(keys_per_task);
              for (int64_t i=0 ; i<local_updates ; i++) {
                dst[i] = reinterpret_cast<float*>(i8_table + (typed_keys[i+start_key] * row_size));
                src[i] = reinterpret_cast<const float*>(i8_h_updates + ((i+start_key) * update_stride));
              }
              for (int64_t i=0 ; i<local_updates ; i++) {
                for (int64_t j=0 ; j<elements_per_row ; j++) {
                  dst[i][j] += src[i][j];
                }
              }
              if ((--remaining_tasks) == 0) {
                barrier.set_value(); // release the future
              }
            }};

            // Then submit the CPU work for this part of the data
            switch (update_dtype) {
              case DataType_t::Float16:
                ctx->get_thread_pool()->submit_n(0, tasks_per_copy, update_task_fp16);
                break;
              case DataType_t::Float32:
                ctx->get_thread_pool()->submit_n(0, tasks_per_copy, update_task_fp32);
                break;
              default:
                NVE_LOG_ERROR_("Unsupported Data type");
                throw std::invalid_argument(std::string("Unsupported Data type"));
            }
          }
        }};

        ctx->get_thread_pool()->submit(update_launcher_task);
        future.wait(); // wait on all tasks to complete
        for (int64_t i=0 ; i<num_copies ; i++) {
          NVE_CHECK_(cudaEventDestroy(copy_events.at(i))); // todo: keep events in the context and reuse them
        }
      } else {
        // Use a single cudaMemcpy
        NVE_CHECK_(cudaMemcpyAsync(h_updates, updates, update_buffer_size, cudaMemcpyDefault, update_stream));
        NVE_CHECK_(cudaStreamSynchronize(update_stream));

        // launch update jobs on threadpool
        int8_t* i8_table = reinterpret_cast<int8_t*>(config_.uvm_table);
        const int8_t* i8_updates = reinterpret_cast<const int8_t*>(h_updates);
        const KeyType* typed_keys = reinterpret_cast<const KeyType*>(h_keys);

        const int64_t num_threads{ctx->get_thread_pool()->num_workers()};
        const int64_t updates_per_task = std::min<size_t>(1<<10, (num_keys + num_threads -1) / num_threads);

        const auto num_tasks = (num_keys + updates_per_task - 1) / updates_per_task;
        const auto row_size = config_.row_size_in_bytes;
        
        const auto update_task_fp16{[&](const size_t idx) {
          const int64_t start_key = idx * updates_per_task;
          const int64_t end_key = std::min<int64_t>((idx + 1) * updates_per_task, num_keys);
          const auto local_updates = end_key-start_key;
          NVE_CHECK_(update_size % sizeof(half) == 0);
          const auto elements_per_row = update_size / sizeof(half);

          std::vector<half*> dst(updates_per_task);
          std::vector<const half*> src(updates_per_task);
          for (int64_t i=0 ; i<local_updates ; i++) {
            dst[i] = reinterpret_cast<half*>(i8_table + (typed_keys[i+start_key] * row_size));
            src[i] = reinterpret_cast<const half*>(i8_updates + ((i+start_key) * update_stride));
          }

          for (int64_t i=0 ; i<local_updates ; i++) {
            for (size_t j=0 ; j<elements_per_row ; j++) {
              dst[i][j] += src[i][j];
            }
          }
        }};

        const auto update_task_fp32{[&](const size_t idx) {
          const int64_t start_key = idx * updates_per_task;
          const int64_t end_key = std::min<int64_t>((idx + 1) * updates_per_task, num_keys);
          const auto local_updates = end_key-start_key;
          NVE_CHECK_(update_size % sizeof(float) == 0);
          const auto elements_per_row = update_size / sizeof(float);

          std::vector<float*> dst(updates_per_task);
          std::vector<const float*> src(updates_per_task);
          for (int64_t i=0 ; i<local_updates ; i++) {
            dst[i] = reinterpret_cast<float*>(i8_table + (typed_keys[i+start_key] * row_size));
            src[i] = reinterpret_cast<const float*>(i8_updates + ((i+start_key) * update_stride));
          }
          for (int64_t i=0 ; i<local_updates ; i++) {
            for (size_t j=0 ; j<elements_per_row ; j++) {
              dst[i][j] += src[i][j];
            }
          }
        }};

        NVE_CHECK_(config_.value_dtype == update_dtype, "Unsupported update type combination"); // TODO: support other type combinations
        switch (update_dtype) {
          case DataType_t::Float16:
            ctx->get_thread_pool()->execute_n(0, num_tasks, update_task_fp16);
            break;
          case DataType_t::Float32:
            ctx->get_thread_pool()->execute_n(0, num_tasks, update_task_fp32);
            break;
          default:
            NVE_LOG_ERROR_("Unsupported Data type");
            throw std::invalid_argument(std::string("Unsupported Data type"));
        }
      }
    } else {
      // use gpu kernel to update uvm buffer
      const void* d_keys = keys_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, sc.queue_stream);

      // Launch the accumulate kernel
      switch (update_dtype) {
          case DataType_t::Float32:
            {
              NVE_CHECK_(config_.value_dtype == DataType_t::Float32);
              uint32_t embedding_width = static_cast<uint32_t>(config_.row_size_in_bytes / sizeof(float));
              uint32_t value_stride_elements = static_cast<uint32_t>(update_stride / sizeof(float));
              UpdateAccumulateTable<KeyType, float>(
                  reinterpret_cast<const float*>(updates),
                  reinterpret_cast<const KeyType*>(d_keys),
                  reinterpret_cast<float*>(config_.uvm_table),
                  embedding_width,
                  value_stride_elements,
                  embedding_width,
                  static_cast<int32_t>(num_keys),
                  update_stream);
            }
            break;
          case DataType_t::Float16:
            {
              NVE_CHECK_(config_.value_dtype == DataType_t::Float16);
              uint32_t embedding_width = static_cast<uint32_t>(config_.row_size_in_bytes / sizeof(__half));
              uint32_t value_stride_elements = static_cast<uint32_t>(update_stride / sizeof(__half));
              UpdateAccumulateTable<KeyType, __half>(
                  reinterpret_cast<const __half*>(updates),
                  reinterpret_cast<const KeyType*>(d_keys),
                  reinterpret_cast<__half*>(config_.uvm_table),
                  embedding_width,
                  value_stride_elements,
                  embedding_width,
                  static_cast<int32_t>(num_keys),
                  update_stream);
            }
            break;
        default:
              NVE_LOG_ERROR_("Unsupported Data type");
              throw std::invalid_argument(std::string("Unsupported Data type"));
      }
    }

    cudaEvent_t uvm_update_event;
    NVE_CHECK_(cudaEventCreateWithFlags(&uvm_update_event, cudaEventDisableTiming));
    NVE_CHECK_(cudaEventRecord(uvm_update_event, update_stream));

    // If the modify stream is different from the update stream, it should wait for the update to complete
    if (modify_stream != update_stream) {
      NVE_CHECK_(cudaStreamWaitEvent(modify_stream, uvm_update_event));
    }

    if (!config_.private_stream) {
      // Have all lookup streams wait for the update to complete before starting
      for (auto s : sync->get_streams()) {
        NVE_CHECK_(cudaStreamWaitEvent(s, uvm_update_event));
      }
    }
    // if we needed to lock 'uvm_lock' - it will be released here.
  }
}

template <typename KeyType>
template <typename OffsetType, typename ValueType, typename OutputType, typename WeightType>
void GpuTable<KeyType>::find_and_combine(context_ptr_t& ctx, int64_t num_keys, const void* keys,
                                         SparseType_t hot_type, int64_t num_offsets,
                                         const OffsetType* offsets, int64_t fixed_hotness,
                                         PoolingType_t pooling_type, const WeightType* weights,
                                         int64_t value_stride, OutputType* values) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_(value_stride == config_.row_size_in_bytes,
             "Output stride must be the same as cache row size");

  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");
  StreamCoordinator sc(gpu_table_ctx->get_lookup_stream(), config_.private_stream);

  std::unique_lock uvm_lock(uvm_table_mutex_, std::defer_lock); // don't lock yet, only lock when not using private stream
  if (config_.uvm_table && !config_.private_stream) {
    uvm_lock.lock();
  }

  auto lookup_ctx{gpu_table_ctx->lookup_context()};
  using CacheDataType = typename CacheType::CacheData;

  int32_t num_elements = static_cast<int32_t>(value_stride / sizeof(ValueType));

  switch (hot_type) {
    case SparseType_t::Fixed:
      if (num_keys % fixed_hotness > 0) {
        NVE_LOG_ERROR_("Number of keys doesn't divide by fixed hotness");
        throw std::invalid_argument("Invalid number of keys for fixed hotness");
      }
      {
        uint32_t batch = static_cast<uint32_t>(num_keys / fixed_hotness);
        switch (pooling_type) {
          case PoolingType_t::Sum:
              callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, true, true, false>
                  (batch, static_cast<const int8_t*>(config_.uvm_table), reinterpret_cast<const KeyType*>(keys),
                  offsets, weights, static_cast<int32_t>(fixed_hotness), cache_->get_cache_data(lookup_ctx),
                  num_elements, reinterpret_cast<ValueType*>(values), sc.queue_stream);
              break;
          case PoolingType_t::Mean:
              callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, true, false, false>
                  (batch, static_cast<const int8_t*>(config_.uvm_table), reinterpret_cast<const KeyType*>(keys),
                  offsets, weights, static_cast<int32_t>(fixed_hotness), cache_->get_cache_data(lookup_ctx),
                  num_elements, reinterpret_cast<ValueType*>(values), sc.queue_stream);
              break;
          case PoolingType_t::WeightedSum:
              NVE_CHECK_(weights != nullptr,
                        "Weights must be provided for weighted sum pooling");
              callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, true, true, true>
                  (batch, static_cast<const int8_t*>(config_.uvm_table), reinterpret_cast<const KeyType*>(keys),
                  offsets, weights, static_cast<int32_t>(fixed_hotness), cache_->get_cache_data(lookup_ctx),
                  num_elements, reinterpret_cast<ValueType*>(values), sc.queue_stream);
              break;
          case PoolingType_t::WeightedMean:
              NVE_CHECK_(weights != nullptr,
                        "Weights must be provided for weighted mean pooling");
              callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, true, false, true>
                  (batch, static_cast<const int8_t*>(config_.uvm_table), reinterpret_cast<const KeyType*>(keys),
                  offsets, weights, static_cast<int32_t>(fixed_hotness), cache_->get_cache_data(lookup_ctx),
                  num_elements, reinterpret_cast<ValueType*>(values), sc.queue_stream);
              break;
          default:
              NVE_THROW_NOT_IMPLEMENTED_();
        }
      }
      break;
    case SparseType_t::CSR:
      NVE_CHECK_(offsets != nullptr,
                 "Offsets must be provided for CSR layout");
      switch (pooling_type) {
        case PoolingType_t::Sum:
            callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, false, true, false>
                (static_cast<uint32_t>(num_offsets), static_cast<const int8_t*>(config_.uvm_table),
                 reinterpret_cast<const KeyType*>(keys), offsets, weights, static_cast<int32_t>(fixed_hotness),
                 cache_->get_cache_data(lookup_ctx), num_elements,
                 reinterpret_cast<ValueType*>(values), sc.queue_stream);
            break;
        case PoolingType_t::Mean:
            callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, false, false, false>
                (static_cast<uint32_t>(num_offsets), static_cast<const int8_t*>(config_.uvm_table),
                 reinterpret_cast<const KeyType*>(keys), offsets, weights, static_cast<int32_t>(fixed_hotness),
                 cache_->get_cache_data(lookup_ctx), num_elements,
                 reinterpret_cast<ValueType*>(values), sc.queue_stream);
            break;
        case PoolingType_t::WeightedSum:
            NVE_CHECK_(weights != nullptr,
                       "Weights must be provided for weighted sum pooling");
            callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, false, true, true>
                (static_cast<uint32_t>(num_offsets), static_cast<const int8_t*>(config_.uvm_table),
                 reinterpret_cast<const KeyType*>(keys), offsets, weights, static_cast<int32_t>(fixed_hotness),
                 cache_->get_cache_data(lookup_ctx), num_elements,
                 reinterpret_cast<ValueType*>(values), sc.queue_stream);
            break;
        case PoolingType_t::WeightedMean:
            NVE_CHECK_(weights != nullptr,
                       "Weights must be provided for weighted mean pooling");
            callFindAndCombineKernel<ValueType, KeyType, ValueType, CacheDataType, false, false, true>
                (static_cast<uint32_t>(num_offsets), static_cast<const int8_t*>(config_.uvm_table),
                 reinterpret_cast<const KeyType*>(keys), offsets, weights, static_cast<int32_t>(fixed_hotness),
                 cache_->get_cache_data(lookup_ctx), num_elements,
                 reinterpret_cast<ValueType*>(values), sc.queue_stream);
            break;
        default:
          NVE_THROW_NOT_IMPLEMENTED_();
      } break;
    case SparseType_t::COO:
      NVE_THROW_NOT_IMPLEMENTED_();
      break;
    default:
      NVE_LOG_ERROR_("Invalid Hotness type");
      throw std::invalid_argument(std::string("Invalid Hotness type"));
  }
}

template <typename KeyType>
std::shared_ptr<nve::DefaultECEvent> GpuTable<KeyType>::create_sync_event() {
  // When using private stream mode there's no need for any streams in the sync event
  // since all modify and lookup ops will use the same stream and are implicitly synchronized by CUDA
  if (config_.private_stream) {
    return std::make_shared<nve::DefaultECEvent>(std::unordered_set<cudaStream_t>());
  } else {
    return contexts_->create_sync_event();
  }
}

template <typename KeyType>
context_ptr_t GpuTable<KeyType>::create_execution_context(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator) 
{
  ScopedDevice scoped_device(config_.device_id);
  if (config_.private_stream &&
      ((config_.private_stream != lookup_stream) || (config_.private_stream != modify_stream))) {
    NVE_LOG_WARNING_("Creating an execution context with different streams than the private stream incurs a small synchronization overhead.");
  }
  return std::make_shared<GPUTableExecutionContext<KeyType>>(
    lookup_stream,
    modify_stream,
    thread_pool,
    allocator ? allocator : allocator_,
    cache_,
    config_.max_modify_size,
    config_.count_misses,
    contexts_);
}

template <typename KeyType>
void GpuTable<KeyType>::reset_lookup_counter(context_ptr_t& ctx) {
  if (!config_.count_misses) {
    NVE_LOG_WARNING_("GPUTable was configured without a key counter");
    return;
  }
  ScopedDevice scope_device(config_.device_id);
  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto miss_metric = gpu_table_ctx->miss_metric();
  StreamCoordinator sc(gpu_table_ctx->get_lookup_stream(), config_.private_stream);
  NVE_CHECK_(cache_->performance_metric_reset(miss_metric, sc.queue_stream));
}

template <typename KeyType>
void GpuTable<KeyType>::get_lookup_counter(context_ptr_t& ctx, int64_t* counter) {
  if (!config_.count_misses) {
    NVE_LOG_WARNING_("GPUTable was configured without a key counter");
    *counter = 0;
    return;
  }
  NVE_CHECK_(counter != nullptr, "Invalid counter ptr");
  ScopedDevice scope_device(config_.device_id);
  auto gpu_table_ctx = std::dynamic_pointer_cast<GPUTableExecutionContext<KeyType>>(ctx);
  NVE_CHECK_(gpu_table_ctx != nullptr, "Invalid GPU table context");

  auto miss_metric = gpu_table_ctx->miss_metric();
  StreamCoordinator sc(gpu_table_ctx->get_lookup_stream(), config_.private_stream);
  NVE_CHECK_(cache_->performance_metric_get_value(miss_metric, counter, sc.queue_stream));
}

template <typename KeyType>
bool GpuTable<KeyType>::lookup_counter_hits() {
  return false;
}

template <typename KeyType>
int32_t GpuTable<KeyType>::get_device_id() const { 
  return config_.device_id;
}

template <typename KeyType>
int64_t GpuTable<KeyType>::get_max_row_size() const { 
  return config_.row_size_in_bytes;
}

template <typename KeyType>
void GpuTable<KeyType>::erase_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) {
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  auto keys_buf = config_.modify_on_gpu ? 
                  keys->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream) :
                  keys->access_buffer(cudaMemoryTypeHost, true /*copy_content*/, modify_stream);
  erase(ctx, n, keys_buf);
}

template <typename KeyType>
void GpuTable<KeyType>::find_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
                                int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> /*value_sizes*/) const {
  ScopedDevice scope_device(config_.device_id);
  auto lookup_stream = ctx->get_lookup_stream();
  auto keys_buf = keys->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);
  max_bitmask_repr_t* hit_mask_buf = hit_mask ? hit_mask->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream) : nullptr;
  auto values_buf = values->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, lookup_stream);
  find(ctx, n, keys_buf, hit_mask_buf, value_stride, values_buf, nullptr);
}

template <typename KeyType>
void GpuTable<KeyType>::insert_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                                  int64_t value_size, buffer_ptr<const void> values) {
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  auto keys_buf = keys->access_buffer(config_.modify_on_gpu ? cudaMemoryTypeDevice : cudaMemoryTypeHost, true /*copy_content*/, modify_stream);
  auto values_buf = values->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  insert(ctx, n, keys_buf, value_stride, value_size, values_buf);
}

template <typename KeyType>
void GpuTable<KeyType>::update_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                                  int64_t value_size, buffer_ptr<const void> values) {
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  auto keys_buf = keys->access_buffer(config_.modify_on_gpu ? cudaMemoryTypeDevice : cudaMemoryTypeHost, true /*copy_content*/, modify_stream);
  auto values_buf = values->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  update(ctx, n, keys_buf, value_stride, value_size, values_buf);
}

template <typename KeyType>
void GpuTable<KeyType>::update_accumulate_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                int64_t update_stride, int64_t update_size, buffer_ptr<const void> updates,
                                DataType_t update_dtype) {
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  auto keys_buf = keys->access_buffer(config_.modify_on_gpu ? cudaMemoryTypeDevice : cudaMemoryTypeHost, true /*copy_content*/, modify_stream);
  auto updates_buf = updates->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  update_accumulate(ctx, n, keys_buf, update_stride, update_size, updates_buf, update_dtype);
}

template <typename KeyType>
template <typename OffsetType, typename ValueType, typename OutputType, typename WeightType>
void GpuTable<KeyType>::find_and_combine_bw(
  context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, SparseType_t sparse_type, int64_t num_offsets,
  buffer_ptr<const OffsetType> offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
  buffer_ptr<const WeightType> weights, int64_t value_stride, buffer_ptr<void> values) {
  ScopedDevice scope_device(config_.device_id);
  auto lookup_stream = ctx->get_lookup_stream();
  auto keys_buf = keys->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);
  auto offsets_buf = offsets->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);
  const WeightType* weights_buf =
    (weights != nullptr) ?
    weights->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream) :
    nullptr;
  auto values_buf = values->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, lookup_stream);
  find_and_combine<OffsetType, ValueType, OutputType, WeightType>(
    ctx, num_keys, keys_buf, sparse_type, num_offsets, offsets_buf,
    fixed_hotness, pooling_type, weights_buf, value_stride, reinterpret_cast<OutputType*>(values_buf));
}

// TODO: add more type combinations
template class GpuTable<int32_t>;
template class GpuTable<int64_t>;

template void GpuTable<int64_t>::find_and_combine<int64_t, float, float, float>(
    context_ptr_t& ctx, int64_t num_keys, const void* keys, SparseType_t sparse_type,
    int64_t num_offsets, const int64_t* offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
    const float* weights, int64_t value_stride, float* values);
template void GpuTable<int32_t>::find_and_combine<int32_t, float, float, float>(
    context_ptr_t& ctx, int64_t num_keys, const void* keys, SparseType_t sparse_type,
    int64_t num_offsets, const int32_t* offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
    const float* weights, int64_t value_stride, float* values);
template void GpuTable<int64_t>::find_and_combine<int64_t, __half, __half, __half>(
    context_ptr_t& ctx, int64_t num_keys, const void* keys, SparseType_t sparse_type,
    int64_t num_offsets, const int64_t* offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
    const __half* weights, int64_t value_stride, __half* values);
template void GpuTable<int32_t>::find_and_combine<int32_t, __half, __half, __half>(
    context_ptr_t& ctx, int64_t num_keys, const void* keys, SparseType_t sparse_type,
    int64_t num_offsets, const int32_t* offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
    const __half* weights, int64_t value_stride, __half* values);

template void GpuTable<int64_t>::find_and_combine_bw<int64_t, float, float, float>(
  context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, SparseType_t sparse_type, int64_t num_offsets,
  buffer_ptr<const int64_t> offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
  buffer_ptr<const float> weights, int64_t value_stride, buffer_ptr<void> values);
template void GpuTable<int32_t>::find_and_combine_bw<int32_t, float, float, float>(
  context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, SparseType_t sparse_type, int64_t num_offsets,
  buffer_ptr<const int32_t> offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
  buffer_ptr<const float> weights, int64_t value_stride, buffer_ptr<void> values);
template void GpuTable<int64_t>::find_and_combine_bw<int64_t, __half, __half, __half>(
  context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, SparseType_t sparse_type, int64_t num_offsets,
  buffer_ptr<const int64_t> offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
  buffer_ptr<const __half> weights, int64_t value_stride, buffer_ptr<void> values);
template void GpuTable<int32_t>::find_and_combine_bw<int32_t, __half, __half, __half>(
  context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, SparseType_t sparse_type, int64_t num_offsets,
  buffer_ptr<const int32_t> offsets, int64_t fixed_hotness, PoolingType_t pooling_type,
  buffer_ptr<const __half> weights, int64_t value_stride, buffer_ptr<void> values);

}  // namespace nve
