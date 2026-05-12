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

#include "include/table_utils.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

#include "include/buffer_wrapper.hpp"
#include "include/host_table.hpp"

namespace nve {

namespace {

// Lowercase file extension (without the dot), or empty if there is no dot.
std::string get_file_extension(const std::string& filepath) {
    auto pos = filepath.find_last_of('.');
    if (pos == std::string::npos) {
        return "";
    }
    std::string ext = filepath.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ext;
}

}  // namespace

host_table_ptr_t create_table_from_plugin(const std::string& plugin_name,
                                          const nlohmann::json& factory_config,
                                          const nlohmann::json& table_config,
                                          table_id_t table_id)
{
    NVE_CHECK_(!plugin_name.empty(),
               "create_table_from_plugin: plugin shared object must not be empty");
    NVE_CHECK_(factory_config.is_object() && factory_config.contains("implementation"),
               "create_table_from_plugin: factory_config must be a JSON object containing an 'implementation' key");
    NVE_CHECK_(table_config.is_object(),
               "create_table_from_plugin: table_config must be a JSON object");

    load_host_table_plugin(plugin_name);
    auto factory = create_host_table_factory(factory_config);
    NVE_CHECK_(factory != nullptr,
               "create_table_from_plugin: failed to create host-table factory for plugin '" + plugin_name + "'");
    auto host_tab = factory->produce(table_id, table_config);
    NVE_CHECK_(host_tab != nullptr,
               "create_table_from_plugin: failed to produce host table for plugin '" + plugin_name + "'");
    return host_tab;
}

void insert_keys_from_tensor_file(table_ptr_t table,
                                  context_ptr_t ctx,
                                  std::shared_ptr<TensorFileFormatBase> keys_reader,
                                  std::shared_ptr<TensorFileFormatBase> values_reader,
                                  uint64_t row_bytes,
                                  uint64_t batch_size)
{
    NVE_CHECK_(table != nullptr, "insert_keys_from_tensor_file: table is null");
    NVE_CHECK_(ctx != nullptr, "insert_keys_from_tensor_file: ctx is null");

    using KeyType = int64_t;

    const uint64_t keys_num_rows = keys_reader->get_num_rows();
    const uint64_t values_num_rows = values_reader->get_num_rows();
    NVE_CHECK_(keys_num_rows == values_num_rows,
               "insert_keys_from_tensor_file: keys/values row count mismatch");

    std::vector<int8_t> value_buffer(row_bytes * batch_size);
    std::vector<KeyType> keys(batch_size);

    auto submit = [&](uint64_t n) {
        const uint64_t keys_buffer_size = n * sizeof(KeyType);
        const uint64_t values_buffer_size = n * row_bytes;
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
            ctx, "keys", static_cast<const void*>(keys.data()), keys_buffer_size);
        auto values_bw = std::make_shared<BufferWrapper<const void>>(
            ctx, "values", static_cast<const void*>(value_buffer.data()), values_buffer_size);
        table->insert_bw(ctx,
                         static_cast<int64_t>(n),
                         std::move(keys_bw),
                         static_cast<int64_t>(row_bytes),
                         static_cast<int64_t>(row_bytes),
                         std::move(values_bw));
    };

    const uint64_t num_batches = keys_num_rows / batch_size;
    const uint64_t rem = keys_num_rows % batch_size;
    for (uint64_t i = 0; i < num_batches; ++i) {
        keys_reader->load_batch(batch_size, keys.data());
        values_reader->load_batch(batch_size, value_buffer.data());
        submit(batch_size);
    }
    if (rem > 0) {
        keys_reader->load_batch(rem, keys.data());
        values_reader->load_batch(rem, value_buffer.data());
        submit(rem);
    }
}

void insert_keys_from_filepath(table_ptr_t table,
                               context_ptr_t ctx,
                               const std::string& keys_path,
                               const std::string& values_path,
                               uint64_t row_bytes,
                               uint64_t batch_size)
{
    using KeyType = int64_t;

    std::shared_ptr<StreamWrapperBase> keys_stream =
        std::make_shared<InputFileStreamWrapper>(keys_path);
    std::shared_ptr<StreamWrapperBase> values_stream =
        std::make_shared<InputFileStreamWrapper>(values_path);

    const std::string keys_ext = get_file_extension(keys_path);
    const std::string values_ext = get_file_extension(values_path);
    NVE_CHECK_(keys_ext == values_ext,
               "insert_keys_from_filepath: keys/values file format mismatch");

    if (keys_ext == "npy") {
        auto keys_reader = std::make_shared<NumpyTensorFileFormat>(keys_stream);
        auto values_reader = std::make_shared<NumpyTensorFileFormat>(values_stream);
        NVE_CHECK_(keys_reader->get_shape().size() == 1, "Invalid keys shape");
        NVE_CHECK_(keys_reader->get_shape()[0] == values_reader->get_shape()[0],
                   "values/keys shape mismatch");
        NVE_CHECK_(values_reader->get_row_size_in_bytes() == row_bytes,
                   "values row size mismatch");
        NVE_CHECK_(keys_reader->get_row_size_in_bytes() == sizeof(KeyType),
                   "key size mismatch");
        insert_keys_from_tensor_file(std::move(table), std::move(ctx),
                                     keys_reader, values_reader,
                                     row_bytes, batch_size);
    } else if (keys_ext == "dyn") {
        auto keys_reader =
            std::make_shared<BinaryTensorFileFormat>(keys_stream, sizeof(KeyType));
        auto values_reader =
            std::make_shared<BinaryTensorFileFormat>(values_stream, row_bytes);
        insert_keys_from_tensor_file(std::move(table), std::move(ctx),
                                     keys_reader, values_reader,
                                     row_bytes, batch_size);
    } else {
        throw std::runtime_error(
            "insert_keys_from_filepath: unsupported file extension '" + keys_ext + "'");
    }
}

}  // namespace nve
