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

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "include/common.hpp"
#include "include/host_table.hpp"   // host_table_ptr_t
#include "include/nve_types.hpp"
#include "include/table.hpp"
#include "include/serialization.hpp"

namespace nve {

// Loads the plugin shared object named by plugin_name, builds the host-table
// factory from factory_config (must contain "implementation"), and produces
// the underlying host table with table_config.
host_table_ptr_t create_table_from_plugin(const std::string& plugin_name,
                                          const nlohmann::json& factory_config,
                                          const nlohmann::json& table_config,
                                          table_id_t table_id = 1000);

// Streams (key, value) pairs from already-opened tensor-file readers into
// `table` in batches of `batch_size` rows. row_bytes is the size of a single
// value row.
//
// Operations are submitted asynchronously on `ctx`'s stream; the caller is
// responsible for ensuring `ctx` outlives in-flight work — typically by
// calling `ctx->wait()` before letting a one-shot context go out of scope.
//
// `table` and `ctx` are passed by value (not const&) — these functions
// mutate both the underlying nve::Table and ctx-bound buffers.
void insert_keys_from_tensor_file(table_ptr_t table,
                                  context_ptr_t ctx,
                                  std::shared_ptr<TensorFileFormatBase> keys_reader,
                                  std::shared_ptr<TensorFileFormatBase> values_reader,
                                  uint64_t row_bytes,
                                  uint64_t batch_size);

// Convenience wrapper: open the keys/values files, dispatch on extension
// (.npy → NumpyTensorFileFormat, .dyn → BinaryTensorFileFormat), and call
// insert_keys_from_tensor_file. Same async/wait contract as
// insert_keys_from_tensor_file. Keys are assumed to be int64.
void insert_keys_from_filepath(table_ptr_t table,
                               context_ptr_t ctx,
                               const std::string& keys_path,
                               const std::string& values_path,
                               uint64_t row_bytes,
                               uint64_t batch_size);

}  // namespace nve
