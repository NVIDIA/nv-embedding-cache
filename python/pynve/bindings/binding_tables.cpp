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
 
#include "binding_tables.hpp"
#include "include/common.hpp"
#include "include/host_table.hpp"
#include "binding_serialization.hpp"
#include "include/buffer_wrapper.hpp"

namespace nve {

PyNVETable::PyNVETable(table_ptr_t table) : table_(table)
{
}

void PyNVETable::clear(context_ptr_t& ctx)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->clear(ctx);
}

void PyNVETable::erase(context_ptr_t& ctx, int64_t n, const void* keys)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->erase(ctx, n, keys);
}

void PyNVETable::find(context_ptr_t& ctx, 
                      int64_t n,
                      const void* keys, 
                      max_bitmask_repr_t* hit_mask,
                      int64_t value_stride, 
                      void* values, 
                      int64_t* value_sizes) const
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->find(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
}

void PyNVETable::insert(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
                        int64_t value_size, const void* values)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->insert(ctx, n, keys, value_stride, value_size, values);
}

void PyNVETable::update(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
                        int64_t value_size, const void* values)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->update(ctx, n, keys, value_stride, value_size, values);
}

void PyNVETable::update_accumulate(context_ptr_t& ctx,
                                   int64_t n,
                                   const void* keys,
                                   int64_t update_stride,
                                   int64_t update_size,
                                   const void* updates,
                                   DataType_t update_dtype)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->update_accumulate(ctx, n, keys, update_stride, update_size, updates, update_dtype);
}

void PyNVETable::set_table(table_ptr_t table)
{
    NVE_CHECK_(table_ == nullptr, "Table is already initialized");
    table_ = table;
}

int32_t PyNVETable::get_device_id() const
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    return table_->get_device_id();
}

int64_t PyNVETable::get_max_row_size() const
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    return table_->get_max_row_size();
}

void PyNVETable::reset_lookup_counter(context_ptr_t& ctx)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->reset_lookup_counter(ctx);
}

void PyNVETable::get_lookup_counter(context_ptr_t& ctx, int64_t* counter)
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->get_lookup_counter(ctx, counter);
}

bool PyNVETable::lookup_counter_hits()
{
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    return table_->lookup_counter_hits();
}

void PyNVETable::erase_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) {
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->erase_bw(ctx, n, keys);
}

void PyNVETable::find_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
             int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const {
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->find_bw(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
}

void PyNVETable::insert_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                           int64_t value_size, buffer_ptr<const void> values) {
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->insert_bw(ctx, n, keys, value_stride, value_size, values);
}

void PyNVETable::update_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                           int64_t value_size, buffer_ptr<const void> values) {
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->update_bw(ctx, n, keys, value_stride, value_size, values);
}

void PyNVETable::update_accumulate_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                      int64_t update_stride, int64_t update_size, buffer_ptr<const void> updates,
                                      DataType_t update_dtype) {
    NVE_CHECK_(table_ != nullptr, "Table is not initialized");
    table_->update_accumulate_bw(ctx, n, keys, update_stride, update_size, updates, update_dtype);
}

LocalParameterServer::LocalParameterServer(
    uint64_t num_rows,
    uint64_t row_elements,
    nve::DataType_t data_type,
    uint64_t initial_size,
    LocalParameterServer::PSType_t ps_type) :
    PyNVETable({}), num_rows_(num_rows), row_elements_(row_elements), data_type_(data_type) {
    row_bytes_ = row_elements_ * static_cast<uint64_t>(dtype_size(data_type_));
    const int64_t num_partitions = 1; // Single partition is better for inference
    const int64_t keys_per_partition = static_cast<int64_t>(num_rows_) / num_partitions;
    initial_size = std::max<uint64_t>(initial_size, 1024);

    nlohmann::json cfg = {
        {"key_size", sizeof(KeyType)},
        {"initial_capacity", initial_size / static_cast<uint64_t>(num_partitions)},
        {"max_value_size", row_bytes_},
        {"num_partitions", num_partitions},
        {"value_dtype", to_string(data_type)},
        {"value_alignment", 32},
        {"overflow_policy",
        {
            // Using random eviction, assuming gpu cache handles all host keys. Otherwise, replace with "evict_lru".
            {"handler", "evict_random"},
            // (num_rows_ == 0) implies unlimited size (grow until OOM). Use erase_keys() to remove data manually.
            {"overflow_margin", (num_rows_ > 0) ? keys_per_partition : INT64_MAX},
            {"resolution_margin", 0.95}
        }
        },
        {"max_num_keys_per_task", 4*1024},
    };
    if (num_partitions == 1){
        cfg["partitioner"] = "always_zero";
    }

    switch (ps_type) {
        default:
        {
            NVE_LOG_WARNING_("Invalid ps_type, defaulting to NVHashMap");
            [[fallthrough]];
        }
        case LocalParameterServer::PSType_t::NVHashMap:
        {
            load_host_table_plugin("nvhm");
            auto mw_fac = nve::create_host_table_factory(R"({"implementation": "nvhm_map"})"_json);
            NVE_CHECK_(mw_fac != nullptr, "Failed to initialize MW factory");
            auto mw_tab = mw_fac->produce(0, cfg);
            NVE_CHECK_(mw_tab != nullptr, "Failed to produce MW table");
            set_table(std::dynamic_pointer_cast<nve::Table>(mw_tab));
        }
        break;
        case LocalParameterServer::PSType_t::Abseil:
        {
            load_host_table_plugin("abseil");
            auto abs_fac = nve::create_host_table_factory(R"({"implementation": "abseil_flat_map"})"_json);
            NVE_CHECK_(abs_fac != nullptr, "Failed to initialize Abseil factory");
            auto abs_tab = abs_fac->produce(0, cfg);
            NVE_CHECK_(abs_tab != nullptr, "Failed to produce Abseil table");
            set_table(std::dynamic_pointer_cast<nve::Table>(abs_tab));
        }
        break;
        case LocalParameterServer::PSType_t::ParallelHash:
        {
            load_host_table_plugin("phmap");
            auto ph_fac = nve::create_host_table_factory(R"({"implementation": "phmap_flat_map"})"_json);
            NVE_CHECK_(ph_fac != nullptr, "Failed to initialize PH factory");
            auto ph_tab = ph_fac->produce(0, cfg);
            NVE_CHECK_(ph_tab != nullptr, "Failed to produce PH table");
            set_table(std::dynamic_pointer_cast<nve::Table>(ph_tab));
        }
        break;
    }

    // initialize an execution context
    ctx_ = create_execution_context(0, 0, nullptr, nullptr);
}

LocalParameterServer::~LocalParameterServer() {
    ctx_.reset();
}

void LocalParameterServer::insert_keys(size_t num_keys, uintptr_t keys, uintptr_t values) {
    NVE_CHECK_(keys != 0, "Invalid Keys tensor");
    NVE_CHECK_(values != 0, "Invalid Values tensor");
    const auto keys_buffer_size = num_keys * sizeof(KeyType);
    const auto values_buffer_size = num_keys * row_bytes_;
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "keys", reinterpret_cast<const void*>(keys), keys_buffer_size);
    auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "values", reinterpret_cast<const void*>(values), values_buffer_size);
    insert_bw(  ctx_,
                static_cast<int64_t>(num_keys),
                keys_bw,
                static_cast<int64_t>(row_bytes_),
                static_cast<int64_t>(row_bytes_),
                values_bw);
}

void LocalParameterServer::insert_keys_from_tensor_file(std::shared_ptr<TensorFileFormatBase> keys_file_reader, std::shared_ptr<TensorFileFormatBase> values_file_reader, uint64_t batch_size) {
    auto keys_num_rows = keys_file_reader->get_num_rows();
    auto values_num_rows = values_file_reader->get_num_rows();
    NVE_CHECK_(keys_num_rows == values_num_rows, "Keys/Values number of rows mismatch");
    std::vector<int8_t> value_buffer(row_bytes_ * batch_size);
    std::vector<KeyType> keys(batch_size);

    uint64_t num_batches = keys_num_rows / batch_size;
    uint64_t rem = keys_num_rows % batch_size;
    for (size_t i = 0; i < num_batches; i++) {
        keys_file_reader->load_batch(batch_size, keys.data());
        values_file_reader->load_batch(batch_size, value_buffer.data());
        insert_keys(batch_size, reinterpret_cast<uintptr_t>(keys.data()), reinterpret_cast<uintptr_t>(value_buffer.data()));
    }
    if (rem > 0) {
        keys_file_reader->load_batch(rem, keys.data());
        values_file_reader->load_batch(rem, value_buffer.data());
        insert_keys(rem, reinterpret_cast<uintptr_t>(keys.data()), reinterpret_cast<uintptr_t>(value_buffer.data()));
    }
}

std::string getFileExtension(const std::string& filepath) {
    size_t pos = filepath.find_last_of('.');
    if (pos == std::string::npos) {
        return ""; // No extension
    }
    
    std::string extension = filepath.substr(pos + 1);
    
    // Convert to lowercase
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension;
}

void LocalParameterServer::insert_keys_from_numpy_file(py::object keys_stream, py::object values_stream, uint64_t batch_size) {
    std::shared_ptr<StreamWrapperBase> keys_stream_wrapper = std::make_shared<PyStreamWrapper>(keys_stream);
    std::shared_ptr<StreamWrapperBase> values_stream_wrapper = std::make_shared<PyStreamWrapper>(values_stream);

    std::shared_ptr<NumpyTensorFileFormat> keys_file_reader = std::make_shared<NumpyTensorFileFormat>(keys_stream_wrapper);
    std::shared_ptr<NumpyTensorFileFormat> values_file_reader = std::make_shared<NumpyTensorFileFormat>(values_stream_wrapper);
    NVE_CHECK_(keys_file_reader->get_shape().size() == 1, "Invalid keys shape");
    NVE_CHECK_(keys_file_reader->get_shape()[0] == values_file_reader->get_shape()[0], "Values/Keys shape mismatch");
    NVE_CHECK_(values_file_reader->get_row_size_in_bytes() == row_bytes_, "Values row size mismatch");
    NVE_CHECK_(keys_file_reader->get_row_size_in_bytes() == sizeof(KeyType), "Key size mismatch");
    insert_keys_from_tensor_file(keys_file_reader, values_file_reader, batch_size);
}

void LocalParameterServer::insert_keys_from_binary_file(py::object keys_stream, py::object values_stream, uint64_t batch_size) {
    std::shared_ptr<StreamWrapperBase> keys_stream_wrapper = std::make_shared<PyStreamWrapper>(keys_stream);
    std::shared_ptr<StreamWrapperBase> values_stream_wrapper = std::make_shared<PyStreamWrapper>(values_stream);
    std::shared_ptr<BinaryTensorFileFormat> keys_file_reader = std::make_shared<BinaryTensorFileFormat>(keys_stream_wrapper, sizeof(KeyType));
    std::shared_ptr<BinaryTensorFileFormat> values_file_reader = std::make_shared<BinaryTensorFileFormat>(values_stream_wrapper, row_bytes_);
    insert_keys_from_tensor_file(keys_file_reader, values_file_reader, batch_size);
}

void LocalParameterServer::insert_keys_from_filepath(const std::string& keys_path, const std::string& values_path, uint64_t batch_size) {
    std::shared_ptr<StreamWrapperBase> keys_stream_wrapper = std::make_shared<InputFileStreamWrapper>(keys_path);
    std::shared_ptr<StreamWrapperBase> values_stream_wrapper = std::make_shared<InputFileStreamWrapper>(values_path);
    std::string keys_extension = getFileExtension(keys_path);
    std::string values_extension = getFileExtension(values_path);
    NVE_CHECK_(keys_extension == values_extension, "Keys/Values file format mismatch");
    if (keys_extension == "npy") {
        std::shared_ptr<NumpyTensorFileFormat> keys_file_reader = std::make_shared<NumpyTensorFileFormat>(keys_stream_wrapper);
        std::shared_ptr<NumpyTensorFileFormat> values_file_reader = std::make_shared<NumpyTensorFileFormat>(values_stream_wrapper);
        NVE_CHECK_(keys_file_reader->get_shape().size() == 1, "Invalid keys shape");
        NVE_CHECK_(keys_file_reader->get_shape()[0] == values_file_reader->get_shape()[0], "Values/Keys shape mismatch");
        NVE_CHECK_(values_file_reader->get_row_size_in_bytes() == row_bytes_, "Values row size mismatch");
        NVE_CHECK_(keys_file_reader->get_row_size_in_bytes() == sizeof(KeyType), "Key size mismatch");
        insert_keys_from_tensor_file(keys_file_reader, values_file_reader, batch_size);
    }
    else if (keys_extension == "dyn") {
        // dynemb bin file
        std::shared_ptr<BinaryTensorFileFormat> keys_file_reader = std::make_shared<BinaryTensorFileFormat>(keys_stream_wrapper, sizeof(KeyType));
        std::shared_ptr<BinaryTensorFileFormat> values_file_reader = std::make_shared<BinaryTensorFileFormat>(values_stream_wrapper, row_bytes_);
        insert_keys_from_tensor_file(keys_file_reader, values_file_reader, batch_size);
    }
    else {
        NVE_LOG_ERROR_("Unsupported file extension: " + keys_extension);
        throw std::runtime_error("Unsupported file extension");
    }
}

void LocalParameterServer::erase_keys(size_t num_keys, uintptr_t keys) {
    NVE_CHECK_(keys != 0, "Invalid Keys tensor");
    const auto keys_buffer_size = num_keys * sizeof(KeyType);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "keys", reinterpret_cast<const void*>(keys), keys_buffer_size);
    erase_bw(ctx_, static_cast<int64_t>(num_keys), keys_bw);
}

void LocalParameterServer::clear_keys() {
    clear(ctx_);
}

} // namespace nve
