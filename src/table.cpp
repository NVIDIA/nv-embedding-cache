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

#include <execution_context.hpp>
#include <json_support.hpp>
#include <table.hpp>
#include <unordered_map>
namespace nve {

class TableExecutionContext: public ExecutionContext {
 public:
  using base_type = ExecutionContext;

  NVE_PREVENT_COPY_AND_MOVE_(TableExecutionContext);

  TableExecutionContext() = delete;

  TableExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator)
    : base_type(lookup_stream, modify_stream, std::move(thread_pool), std::move(allocator)) {}

  ~TableExecutionContext() override = default;
};


NVE_DEFINE_JSON_ENUM_CONVERSION_(DataType_t, DataType_t::Unknown, DataType_t::Float32,
                                 DataType_t::Float16, DataType_t::BFloat, DataType_t::E4M3,
                                 DataType_t::E5M2, DataType_t::Float64,
                                 DataType_t::QInt8RowwiseF32, DataType_t::QInt8RowwiseF16,
                                 DataType_t::QUint8RowwiseF32, DataType_t::QUint8RowwiseF16)

Table::Table() { NVE_LOG_VERBOSE_("Constructing table #", this); }

Table::~Table() { NVE_LOG_VERBOSE_("Destructing table #", this); }

context_ptr_t Table::create_execution_context(
  cudaStream_t lookup_stream,
  cudaStream_t modify_stream,
  thread_pool_ptr_t thread_pool,
  allocator_ptr_t allocator) {
  return std::make_shared<TableExecutionContext>(lookup_stream, modify_stream, thread_pool, allocator);
}


}  // namespace nve
