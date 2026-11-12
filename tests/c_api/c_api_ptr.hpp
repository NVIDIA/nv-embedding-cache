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

#include <nve_c_api.h>

#include <memory>
#include <type_traits>

// Owning handles for C API objects, in the spirit of cuda_ptr.hpp: an ASSERT_* failure
// returns from the test body and would skip a manual destroy, leaking whatever the
// handle owns (a GPU table holds its device-side cache). Tests that assert on the
// destroy status call the C API directly on .release() instead.
namespace nve {

struct TableDeleter {
  void operator()(nve_table_t table) const noexcept { nve_table_destroy(table); }
};
struct ContextDeleter {
  void operator()(nve_context_t ctx) const noexcept { nve_context_destroy(ctx); }
};
struct TableFactoryDeleter {
  void operator()(nve_table_factory_t factory) const noexcept { nve_table_factory_destroy(factory); }
};
struct ThreadPoolDeleter {
  void operator()(nve_thread_pool_t pool) const noexcept { nve_thread_pool_destroy(pool); }
};

using TablePtr = std::unique_ptr<std::remove_pointer_t<nve_table_t>, TableDeleter>;
using ContextPtr = std::unique_ptr<std::remove_pointer_t<nve_context_t>, ContextDeleter>;
using TableFactoryPtr = std::unique_ptr<std::remove_pointer_t<nve_table_factory_t>, TableFactoryDeleter>;
using ThreadPoolPtr = std::unique_ptr<std::remove_pointer_t<nve_thread_pool_t>, ThreadPoolDeleter>;

}  // namespace nve
