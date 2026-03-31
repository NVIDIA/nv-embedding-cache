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

#include <embedding_layer.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <insert_heuristic.hpp>
#include <thread_pool.hpp>
#include <allocator.hpp>

#include <memory>
#include <string>

/* ============================================================================
 * Thread-local error state
 * ============================================================================ */

nve_status_t nve_set_error(nve_status_t status, const char* message);
nve_status_t nve_set_error(nve_status_t status, const std::string& message);

/* ============================================================================
 * Exception-catching macros
 * ============================================================================ */

#define NVE_C_TRY try {

#define NVE_C_CATCH                                                 \
  }                                                                 \
  catch (const std::bad_alloc&) {                                   \
    return nve_set_error(NVE_ERROR_OUT_OF_MEMORY, "Out of memory"); \
  }                                                                 \
  catch (const std::invalid_argument& e) {                          \
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, e.what());     \
  }                                                                 \
  catch (const std::exception& e) {                                 \
    return nve_set_error(NVE_ERROR_RUNTIME, e.what());              \
  }                                                                 \
  catch (...) {                                                     \
    return nve_set_error(NVE_ERROR_RUNTIME, "Unknown error");       \
  }

/* ============================================================================
 * Handle struct definitions (opaque to consumers)
 * ============================================================================ */

struct nve_context_s {
  nve::context_ptr_t ptr;
};

struct nve_table_s {
  nve::table_ptr_t ptr;
  nve_key_type_t key_type;
};

struct nve_layer_s {
  std::shared_ptr<nve::EmbeddingLayerBase> ptr;
  nve_key_type_t key_type;
};

struct nve_thread_pool_s {
  nve::thread_pool_ptr_t ptr;
};

struct nve_allocator_s {
  nve::allocator_ptr_t ptr;
};

struct nve_heuristic_s {
  std::shared_ptr<nve::InsertHeuristic> ptr;
};

struct nve_host_factory_s {
  nve::host_table_factory_ptr_t ptr;
};

/* ============================================================================
 * Enum conversion helpers
 * ============================================================================ */

nve::DataType_t convert_dtype(nve_data_type_t dt);
nve::SparseType_t convert_sparse_type(nve_sparse_type_t st);
nve::PoolingType_t convert_pooling_type(nve_pooling_type_t pt);
nve::Partitioner_t convert_partitioner(nve_partitioner_t p);
nve::OverflowHandler_t convert_overflow_handler(nve_overflow_handler_t oh);

/* ============================================================================
 * Handle unwrapping helpers
 * ============================================================================ */

inline nve::allocator_ptr_t unwrap_allocator(nve_allocator_t a) {
  return a ? a->ptr : nve::allocator_ptr_t{};
}

inline nve::thread_pool_ptr_t unwrap_thread_pool(nve_thread_pool_t tp) {
  return tp ? tp->ptr : nve::thread_pool_ptr_t{};
}

inline std::shared_ptr<nve::InsertHeuristic> unwrap_heuristic(nve_heuristic_t h) {
  return h ? h->ptr : nullptr;
}
