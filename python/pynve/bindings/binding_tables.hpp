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
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include <memory>

namespace py = pybind11;

namespace nve {

class StreamWrapperBase;
class TensorFileFormatBase;
    
// wrapper class for nve::Table to allow for late binding of the "real" parameter server
class PyNVETable : public Table {
public:
    PyNVETable(table_ptr_t table);
    
    void clear(context_ptr_t& ctx) override;
    void erase(context_ptr_t& ctx, int64_t n, const void* keys) override;
    void find(context_ptr_t& ctx, 
              int64_t n,
              const void* keys, 
              max_bitmask_repr_t* hit_mask,
              int64_t value_stride, 
              void* values, 
              int64_t* value_sizes) const override;
    void insert(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
                int64_t value_size, const void* values) override;
    void update(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
                   int64_t value_size, const void* values) override;
    void update_accumulate(context_ptr_t& ctx,
                            int64_t n,
                            const void* keys,
                            int64_t update_stride,
                            int64_t update_size,
                            const void* updates,
                            DataType_t update_dtype) override;
    void set_table(table_ptr_t table);
    
    int32_t get_device_id() const override;
    int64_t get_max_row_size() const override;
    void reset_lookup_counter(context_ptr_t& ctx) override;
    void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) override;
    bool lookup_counter_hits() override;

    void erase_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override;
    void find_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
                 int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override;
    void insert_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                   int64_t value_size, buffer_ptr<const void> values) override;
    void update_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                   int64_t value_size, buffer_ptr<const void> values) override;
    void update_accumulate_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                              int64_t update_stride, int64_t update_size, buffer_ptr<const void> updates,
                              DataType_t update_dtype) override;

private:
    table_ptr_t table_;
};

class LocalParameterServer final : public nve::PyNVETable {
public:
    enum class PSType_t : uint64_t {
        NVHashMap,
        Abseil,
        ParallelHash,
    };

    // num_rows = 0, means no eviction
    LocalParameterServer(uint64_t num_rows, uint64_t row_elements, nve::DataType_t data_type, uint64_t initial_size, PSType_t ps_type);
    ~LocalParameterServer();
    void insert_keys(size_t num_keys, uintptr_t keys, uintptr_t values);
    void erase_keys(size_t num_keys, uintptr_t keys);
    void clear_keys();
    void insert_keys_from_numpy_file(py::object keys_stream, py::object values_stream, uint64_t batch_size);
    void insert_keys_from_binary_file(py::object keys_stream, py::object values_stream, uint64_t batch_size);
    void insert_keys_from_filepath(const std::string& keys_path, const std::string& values_path, uint64_t batch_size);
    
private:
    // internal function to insert keys from tensor files
    void insert_keys_from_tensor_file(std::shared_ptr<TensorFileFormatBase> keys_file_reader, std::shared_ptr<TensorFileFormatBase> values_file_reader, uint64_t batch_size);
private:
    using KeyType = int64_t; // For now assuming all keys are int64
    uint64_t num_rows_;
    uint64_t row_elements_;
    uint64_t row_bytes_;
    DataType_t data_type_;
    context_ptr_t ctx_;
};

} // namespace nve
