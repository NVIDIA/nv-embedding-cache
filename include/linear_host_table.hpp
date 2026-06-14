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

#include <host_table.hpp>
#include <buffer_wrapper.hpp>
#include <thread_pool.hpp>
#include <nve_types.hpp>
#include <execution_context.hpp>
#include <common.hpp>
#include "cpu_ops/cpu_gather.h"
#include "cpu_ops/cpu_update.h"
#include <vector>
#include <memory>
#include <limits>

namespace nve {

struct LinearHostTableConfig : public HostTableConfig {
    using base_type = HostTableConfig;

    void* emb_table{nullptr};                               // Pointer to host accessible buffer of the embedding table (preferably cudaMallocHost)
                                                            // Caller is responsible for allocating/freeing this buffer
                                                            // Caller is responsible for not accessing indices beyond the allocated size
    int64_t max_threads{std::numeric_limits<int64_t>::max()}; // Max amount of threads used (up to #threads in the machine)

    void check() const;
};

void from_json(const nlohmann::json& json, LinearHostTableConfig& conf);

void to_json(nlohmann::json& json, const LinearHostTableConfig& conf);

template<typename KeyType>
class LinearHostTable : public HostTable<LinearHostTableConfig> {
public:
    using base_type = HostTable<LinearHostTableConfig>;
    using config_type = typename base_type::config_type;
    using key_type = KeyType;
    
    NVE_PREVENT_COPY_AND_MOVE_(LinearHostTable);

    LinearHostTable() = delete;

    LinearHostTable(const LinearHostTableConfig& config)
        : base_type(0, config) {}

    ~LinearHostTable() override = default;

    void clear(context_ptr_t& /*ctx*/) override {
        NVE_NVTX_SCOPED_FUNCTION_COL5_();
        NVE_LOG_INFO_("Clearing linear host table has no effect");
    }

    int64_t size(context_ptr_t&, bool) const override { return -1; }

    void erase(context_ptr_t& /*ctx*/, int64_t /*num_keys*/, buffer_ptr<const void> /*keys*/) override {
        NVE_NVTX_SCOPED_FUNCTION_COL5_();
        NVE_LOG_INFO_("Erasing from linear host table has no effect");
    }

    void find(context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
              int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> /*value_sizes*/) const override {
        NVE_NVTX_SCOPED_FUNCTION_COL1_();
        NVE_CHECK_(ctx != nullptr, "Invalid context");
        auto lookup_stream = ctx->get_lookup_stream();
        const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, lookup_stream) : nullptr;
        // hit_mask is optional — callers that don't need per-key hit info pass null,
        // and the kernel below treats null as "no pre-existing hits, don't record hits".
        max_bitmask_repr_t* hit_mask_buf = hit_mask ? hit_mask->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, lookup_stream) : nullptr;
        void* values_buf = values ? values->access_buffer(cudaMemoryTypeUnregistered, false /*copy_content*/, lookup_stream) : nullptr;
        NVE_CHECK_(keys_buf != nullptr, "Invalid keys");
        NVE_CHECK_(values_buf != nullptr, "Invalid values");
        NVE_CHECK_(num_keys > 0, "Invalid number of keys");
        const auto* typed_keys = reinterpret_cast<const key_type*>(keys_buf);
        const int64_t num_threads{std::min(ctx->get_thread_pool()->num_workers(), config_.max_threads)};
        NVE_CHECK_(cpu_kernel_gather<key_type>(
            ctx->get_thread_pool(),
            static_cast<size_t>(num_keys),
            typed_keys,
            hit_mask_buf,
            static_cast<size_t>(value_stride),
            values_buf,
            static_cast<int8_t*>(config_.emb_table),
            static_cast<uint64_t>(config_.max_value_size),
            static_cast<uint64_t>(num_threads)
        ) == 0);
        auto ctx_counter = lookup_counter_storage(ctx);
        NVE_CHECK_(ctx_counter != nullptr, "Invalid key counter");
        *ctx_counter += num_keys;
    }

    void insert(context_ptr_t& /*ctx*/, int64_t /*num_keys*/, buffer_ptr<const void> /*keys*/, int64_t /*value_stride*/,
                int64_t /*value_size*/, buffer_ptr<const void> /*values*/) override {
        NVE_NVTX_SCOPED_FUNCTION_COL2_();
        NVE_LOG_INFO_("Inserting into linear host table has no effect");
    }

    void update(context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, int64_t value_stride,
                int64_t value_size, buffer_ptr<const void> values) override {
        NVE_NVTX_SCOPED_FUNCTION_COL3_();
        NVE_CHECK_(ctx != nullptr, "Invalid context");
        auto modify_stream = ctx->get_modify_stream();
        const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream) : nullptr;
        const void* values_buf = values ? values->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream) : nullptr;
        NVE_CHECK_(keys_buf != nullptr, "Invalid keys");
        NVE_CHECK_(values_buf != nullptr, "Invalid values");
        NVE_CHECK_(value_size == config_.max_value_size, "Invalid value size");
        NVE_CHECK_(num_keys > 0, "Invalid number of keys");
        const auto* typed_keys = reinterpret_cast<const key_type*>(keys_buf);
        const auto* typed_values = reinterpret_cast<const int8_t*>(values_buf);
        const int64_t num_threads{std::min(ctx->get_thread_pool()->num_workers(), config_.max_threads)};

        NVE_CHECK_(cpu_kernel_update<key_type>(
            ctx->get_thread_pool(),
            static_cast<size_t>(num_keys),
            typed_keys,
            static_cast<size_t>(value_stride),
            typed_values,
            static_cast<int8_t*>(config_.emb_table),
            static_cast<uint64_t>(config_.max_value_size),
            static_cast<uint64_t>(num_threads)
        ) == 0);
    }

    void update_accumulate(context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, int64_t update_stride,
                           int64_t update_size, buffer_ptr<const void> updates,
                           DataType_t update_dtype) override {
        NVE_NVTX_SCOPED_FUNCTION_COL4_();
        NVE_CHECK_(ctx != nullptr, "Invalid context");
        auto modify_stream = ctx->get_modify_stream();
        const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream) : nullptr;
        const void* updates_buf = updates ? updates->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream) : nullptr;
        NVE_CHECK_(keys_buf != nullptr, "Invalid keys");
        NVE_CHECK_(updates_buf != nullptr, "Invalid updates");
        NVE_CHECK_(update_size == config_.max_value_size, "Invalid update size");
        NVE_CHECK_(num_keys > 0, "Invalid number of keys");
        NVE_CHECK_(config_.value_dtype == DataType_t::Float32, "Unsupported value type");
        const auto* typed_keys = reinterpret_cast<const key_type*>(keys_buf);
        const int64_t num_threads{std::min(ctx->get_thread_pool()->num_workers(), config_.max_threads)};
        cpu_kernel_update_accumulate_dispatch(
            ctx->get_thread_pool(),
            static_cast<size_t>(num_keys),
            typed_keys,
            static_cast<size_t>(update_stride),
            updates_buf,
            static_cast<int8_t*>(config_.emb_table),
            static_cast<uint64_t>(config_.max_value_size),
            update_dtype,
            static_cast<uint64_t>(num_threads)
        );
    }

protected:
  int64_t* lookup_counter_storage(context_ptr_t& ctx) const override {
    static constexpr char buffer_name[]{"linear_host_table_key_counter"};

    NVE_CHECK_(ctx != nullptr, "Invalid context");
    void* buffer = ctx->get_buffer(buffer_name, sizeof(int64_t), true);
    NVE_CHECK_(buffer != nullptr, "Failed to get counter buffer");
    return reinterpret_cast<int64_t*>(buffer);
  }
};

} // namespace nve
