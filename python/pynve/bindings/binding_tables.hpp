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

#include <table.hpp>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

#include "include/common.hpp"  // NVE_CHECK_

namespace nve {

class StreamWrapperBase;
class TensorFileFormatBase;

// wrapper class for nve::Table to allow for late binding of the "real" parameter server.
//
class PyNVETable : public Table {
public:
    PyNVETable(table_ptr_t table) : table_(std::move(table)) {}

    void clear(context_ptr_t& ctx) override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->clear(ctx);
    }
    virtual void set_table(table_ptr_t table) {
        NVE_CHECK_(table_ == nullptr, "Table is already initialized");
        table_ = std::move(table);
    }
    int32_t get_device_id() const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        return table_->get_device_id();
    }
    int64_t get_max_row_size() const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        return table_->get_max_row_size();
    }
    int64_t get_key_size() const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        return table_->get_key_size();
    }
    int64_t get_invalid_key() const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        return table_->get_invalid_key();
    }
    DataType_t get_value_type() const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        return table_->get_value_type();
    }
    void reset_lookup_counter(context_ptr_t& ctx) override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->reset_lookup_counter(ctx);
    }
    void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->get_lookup_counter(ctx, counter);
    }
    bool lookup_counter_hits() const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        return table_->lookup_counter_hits();
    }
    void erase(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->erase(ctx, n, std::move(keys));
    }
    void find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                 buffer_ptr<max_bitmask_repr_t> hit_mask, int64_t value_stride,
                 buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->find(ctx, n, std::move(keys), std::move(hit_mask), value_stride, std::move(values), std::move(value_sizes));
    }
    void insert(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                   int64_t value_stride, int64_t value_size,
                   buffer_ptr<const void> values) override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->insert(ctx, n, std::move(keys), value_stride, value_size, std::move(values));
    }
    void update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                   int64_t value_stride, int64_t value_size,
                   buffer_ptr<const void> values) override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->update(ctx, n, std::move(keys), value_stride, value_size, std::move(values));
    }
    void update_accumulate(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                              int64_t update_stride, int64_t update_size,
                              buffer_ptr<const void> updates,
                              DataType_t update_dtype) override {
        NVE_CHECK_(table_ != nullptr, "Table is not initialized");
        table_->update_accumulate(ctx, n, std::move(keys), update_stride, update_size, std::move(updates), update_dtype);
    }

protected:
    // Subclasses pass this to the free utility functions in <table_utils.hpp>.
    const table_ptr_t& inner_table() const { return table_; }

private:
    table_ptr_t table_;
};

class ParameterServerTable final : public nve::PyNVETable {
public:
    enum class PSType_t : uint64_t {
        NVHashMap,
        Abseil,
        ParallelHash,
        Redis,
    };
    using KeyType = int64_t; // For now assuming all keys are int64

    // enum-based ctor. Internally translates
    // (ps_type, extra_params) into the (plugin shared object, factory_config,
    // table_config) triple expected by the plugin-based ctor below.
    // num_rows = 0, means no eviction
    ParameterServerTable(uint64_t num_rows, uint64_t row_elements, nve::DataType_t data_type, uint64_t initial_size, PSType_t ps_type, const std::string& extra_params);

    // Plugin-based ctor: plugin_name is a shared object name/path.
    ParameterServerTable(uint64_t row_elements,
                         nve::DataType_t data_type,
                         std::string plugin_name,
                         nlohmann::json factory_config,
                         nlohmann::json table_config,
                         uint64_t num_rows = 0,
                         table_id_t table_id = 1000);

    ~ParameterServerTable();
    void insert_keys(size_t num_keys, uintptr_t keys, uintptr_t values);
    void erase_keys(size_t num_keys, uintptr_t keys);
    void clear_keys();
    void insert_keys_from_filepath(const std::string& keys_path, const std::string& values_path, uint64_t batch_size);
    uint64_t get_row_size_in_bytes() const;
    void insert_keys_from_tensor_file(std::shared_ptr<TensorFileFormatBase> keys_file_reader, std::shared_ptr<TensorFileFormatBase> values_file_reader, uint64_t batch_size);

    // Returns a JSON object describing how to recreate this PS:
    //   { "remote_ps_type": "plugin",
    //     "plugin_name": "...",
    //     "factory_config": {...},
    //     "table_config":   {...},
    //     "row_elements":   <int>,
    //     "num_rows":       <int>,
    //     "data_type":      "Float32" | "Float16" | ... }
    // The shape is the same regardless of which ctor was used, so consumers
    // (Python load, C++ nve_loader) only need to handle one shape.
    nlohmann::json export_config() const;

private:
    // Shared body for both ctors: dlopens the plugin, creates the factory from
    void init_from_plugin(table_id_t table_id);

    // enum-based ctor helper: 
    table_id_t resolve_ps_type(PSType_t ps_type,
                                      uint64_t initial_size,
                                      const std::string& extra_params);

    uint64_t num_rows_;
    uint64_t row_elements_;
    uint64_t row_bytes_;
    DataType_t data_type_;
    context_ptr_t ctx_;
    std::string plugin_name_;
    nlohmann::json factory_config_;
    nlohmann::json table_config_;
};

} // namespace nve
