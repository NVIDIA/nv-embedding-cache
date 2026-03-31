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

#include "nve_c_api_internal.hpp"
#include <gpu_embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <hierarchical_embedding_layer.hpp>

extern "C" {

/* ============================================================================
 * GPU Embedding Layer
 * ============================================================================ */

nve_status_t nve_gpu_embedding_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_gpu_embedding_layer_config_t* config, nve_allocator_t allocator) {
  if (!out || !config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out and config must not be NULL");
  }
  NVE_C_TRY
    nve::GPUEmbeddingLayerConfig cfg;
    cfg.layer_name = config->layer_name ? config->layer_name : "";
    cfg.device_id = config->device_id;
    cfg.embedding_table = config->embedding_table;
    cfg.num_embeddings = config->num_embeddings;
    cfg.embedding_width_in_bytes = config->embedding_width_in_bytes;
    cfg.value_dtype = convert_dtype(config->value_dtype);

    auto alloc = unwrap_allocator(allocator);
    std::shared_ptr<nve::EmbeddingLayerBase> layer;
    switch (key_type) {
      case NVE_KEY_INT32:
        layer = std::make_shared<nve::GPUEmbeddingLayer<int32_t>>(cfg, alloc);
        break;
      case NVE_KEY_INT64:
        layer = std::make_shared<nve::GPUEmbeddingLayer<int64_t>>(cfg, alloc);
        break;
      default:
        return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Unsupported key type");
    }
    *out = new nve_layer_s{std::move(layer), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Linear UVM Embedding Layer
 * ============================================================================ */

nve_status_t nve_linear_uvm_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_linear_uvm_layer_config_t* config,
    nve_table_t gpu_table, nve_allocator_t allocator) {
  if (!out || !config || !gpu_table || !gpu_table->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out, config, and gpu_table must not be NULL");
  }
  if (gpu_table->key_type != key_type) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "gpu_table key type must match layer key type");
  }
  NVE_C_TRY
    auto alloc = unwrap_allocator(allocator);
    std::shared_ptr<nve::EmbeddingLayerBase> layer;

    switch (key_type) {
      case NVE_KEY_INT32: {
        typename nve::LinearUVMEmbeddingLayer<int32_t>::Config cfg;
        cfg.layer_name = config->layer_name ? config->layer_name : "";
        cfg.insert_heuristic = unwrap_heuristic(config->insert_heuristic);
        cfg.min_insert_freq_gpu = config->min_insert_freq_gpu;
        cfg.min_insert_size_gpu = config->min_insert_size_gpu;
        auto gpu_tbl = std::static_pointer_cast<nve::GpuTable<int32_t>>(gpu_table->ptr);
        layer = std::make_shared<nve::LinearUVMEmbeddingLayer<int32_t>>(cfg, gpu_tbl, alloc);
        break;
      }
      case NVE_KEY_INT64: {
        typename nve::LinearUVMEmbeddingLayer<int64_t>::Config cfg;
        cfg.layer_name = config->layer_name ? config->layer_name : "";
        cfg.insert_heuristic = unwrap_heuristic(config->insert_heuristic);
        cfg.min_insert_freq_gpu = config->min_insert_freq_gpu;
        cfg.min_insert_size_gpu = config->min_insert_size_gpu;
        auto gpu_tbl = std::static_pointer_cast<nve::GpuTable<int64_t>>(gpu_table->ptr);
        layer = std::make_shared<nve::LinearUVMEmbeddingLayer<int64_t>>(cfg, gpu_tbl, alloc);
        break;
      }
      default:
        return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Unsupported key type");
    }
    *out = new nve_layer_s{std::move(layer), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Hierarchical Embedding Layer
 * ============================================================================ */

nve_status_t nve_hierarchical_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_hierarchical_layer_config_t* config,
    const nve_table_t* tables, int64_t num_tables, nve_allocator_t allocator) {
  if (!out || !config || !tables || num_tables <= 0) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  NVE_C_TRY
    std::vector<nve::table_ptr_t> cpp_tables;
    cpp_tables.reserve(static_cast<size_t>(num_tables));
    for (int64_t i = 0; i < num_tables; ++i) {
      if (!tables[i] || !tables[i]->ptr) {
        return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Table handle at index is NULL");
      }
      cpp_tables.push_back(tables[i]->ptr);
    }

    auto alloc = unwrap_allocator(allocator);
    std::shared_ptr<nve::EmbeddingLayerBase> layer;

    switch (key_type) {
      case NVE_KEY_INT32: {
        typename nve::HierarchicalEmbeddingLayer<int32_t>::Config cfg;
        cfg.layer_name = config->layer_name ? config->layer_name : "";
        cfg.insert_heuristic = unwrap_heuristic(config->insert_heuristic);
        cfg.min_insert_freq_gpu = config->min_insert_freq_gpu;
        cfg.min_insert_freq_host = config->min_insert_freq_host;
        cfg.min_insert_size_gpu = config->min_insert_size_gpu;
        cfg.min_insert_size_host = config->min_insert_size_host;
        layer = std::make_shared<nve::HierarchicalEmbeddingLayer<int32_t>>(cfg, cpp_tables, alloc);
        break;
      }
      case NVE_KEY_INT64: {
        typename nve::HierarchicalEmbeddingLayer<int64_t>::Config cfg;
        cfg.layer_name = config->layer_name ? config->layer_name : "";
        cfg.insert_heuristic = unwrap_heuristic(config->insert_heuristic);
        cfg.min_insert_freq_gpu = config->min_insert_freq_gpu;
        cfg.min_insert_freq_host = config->min_insert_freq_host;
        cfg.min_insert_size_gpu = config->min_insert_size_gpu;
        cfg.min_insert_size_host = config->min_insert_size_host;
        layer = std::make_shared<nve::HierarchicalEmbeddingLayer<int64_t>>(cfg, cpp_tables, alloc);
        break;
      }
      default:
        return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Unsupported key type");
    }
    *out = new nve_layer_s{std::move(layer), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Layer lifecycle
 * ============================================================================ */

nve_status_t nve_layer_destroy(nve_layer_t layer) {
  if (!layer) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer must not be NULL");
  }
  delete layer;
  return NVE_SUCCESS;
}

/* ============================================================================
 * Layer operations (polymorphic via EmbeddingLayerBase)
 * ============================================================================ */

nve_status_t nve_layer_lookup(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask, float* hitrates) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    layer->ptr->lookup(ctx->ptr, num_keys, keys, output, output_stride,
                       hitmask, nullptr, hitrates);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_lookup_pooled(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask,
    nve_pooling_type_t pooling_type, nve_sparse_type_t sparse_type,
    const int64_t* key_indices, int64_t num_key_indices,
    void* default_values, const void* sparse_weights,
    nve_data_type_t weight_type, float* hitrates) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    nve::EmbeddingLayerBase::PoolingParams params;
    params.pooling_type = convert_pooling_type(pooling_type);
    params.sparse_type = convert_sparse_type(sparse_type);
    params.key_indices = key_indices;
    params.num_key_indices = num_key_indices;
    params.default_values = default_values;
    params.sparse_weights = sparse_weights;
    params.weight_type = convert_dtype(weight_type);

    layer->ptr->lookup(ctx->ptr, num_keys, keys, output, output_stride,
                       hitmask, &params, hitrates);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_insert(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    int64_t table_id) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    layer->ptr->insert(ctx->ptr, num_keys, keys, value_stride, value_size, values, table_id);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_update(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    layer->ptr->update(ctx->ptr, num_keys, keys, value_stride, value_size, values);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_accumulate(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    nve_data_type_t value_type) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    layer->ptr->accumulate(ctx->ptr, num_keys, keys, value_stride, value_size,
                           values, convert_dtype(value_type));
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_clear(nve_layer_t layer, nve_context_t ctx) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    layer->ptr->clear(ctx->ptr);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_erase(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys, int64_t table_id) {
  if (!layer || !layer->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and ctx must not be NULL");
  }
  NVE_C_TRY
    layer->ptr->erase(ctx->ptr, num_keys, keys, table_id);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_layer_create_execution_context(
    nve_layer_t layer, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator) {
  if (!layer || !layer->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "layer and out must not be NULL");
  }
  NVE_C_TRY
    auto ctx = layer->ptr->create_execution_context(
        static_cast<cudaStream_t>(lookup_stream),
        static_cast<cudaStream_t>(modify_stream),
        unwrap_thread_pool(thread_pool),
        unwrap_allocator(allocator));
    *out = new nve_context_s{std::move(ctx)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

}  // extern "C"
