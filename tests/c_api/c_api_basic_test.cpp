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

#include <gtest/gtest.h>
#include <nve_c_api.h>
#include <gpu_embedding_layer.hpp>
#include <gpu_table.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <host_embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <plugin/external_plugin_table.hpp>

#include "test_utils.hpp"

TEST(NveCApiBasic, Version) {
  int32_t major, minor, patch;
  EXPECT_EQ(NVE_SUCCESS, nve_version(&major, &minor, &patch));
  EXPECT_GE(major, 26);
}

TEST(NveCApiBasic, VersionNullArgs) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_version(nullptr, nullptr, nullptr));
  const char* msg = nullptr;
  nve_get_last_error(&msg);
  EXPECT_NE(nullptr, msg);
  EXPECT_STRNE("", msg);
}

TEST(NveCApiBasic, AbiVersion) {
  int32_t major, minor;
  EXPECT_EQ(NVE_SUCCESS, nve_abi_version(&major, &minor));
  EXPECT_EQ(NVE_C_ABI_VERSION_MAJOR, major);
  EXPECT_EQ(NVE_C_ABI_VERSION_MINOR, minor);
}

TEST(NveCApiBasic, AbiVersionNullArgs) {
  int32_t major, minor;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_abi_version(nullptr, nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_abi_version(&major, nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_abi_version(nullptr, &minor));
  const char* msg = nullptr;
  nve_get_last_error(&msg);
  EXPECT_NE(nullptr, msg);
  EXPECT_STRNE("", msg);
}

TEST(NveCApiBasic, GetLastErrorEmpty) {
  const char* msg = nullptr;
  EXPECT_EQ(NVE_SUCCESS, nve_get_last_error(&msg));
  EXPECT_NE(nullptr, msg);
}

TEST(NveCApiBasic, ConfigDefaults) {
  nve_gpu_table_config_t gpu_cfg;
  const nve::GPUTableConfig gpu_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_config_init(&gpu_cfg, sizeof(gpu_cfg)));
  EXPECT_EQ(gpu_defaults.device_id, gpu_cfg.device_id);
  EXPECT_EQ(gpu_defaults.count_misses, gpu_cfg.count_misses);
  EXPECT_EQ(gpu_defaults.max_modify_size, gpu_cfg.max_modify_size);
  EXPECT_EQ(gpu_defaults.uvm_cpu_accumulate, gpu_cfg.uvm_cpu_accumulate);
  EXPECT_EQ(gpu_defaults.modify_on_gpu, gpu_cfg.modify_on_gpu);
  EXPECT_EQ(gpu_defaults.value_dtype, nve::convert_external_dtype(gpu_cfg.value_dtype));
  EXPECT_EQ(gpu_defaults.invalid_key, gpu_cfg.invalid_key);

  nve_gpu_embedding_layer_config_t emb_cfg;
  const nve::GPUEmbeddingLayerConfig emb_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_embedding_layer_config_init(&emb_cfg, sizeof(emb_cfg)));
  EXPECT_EQ(emb_defaults.device_id, emb_cfg.device_id);
  EXPECT_EQ(emb_defaults.value_dtype, nve::convert_external_dtype(emb_cfg.value_dtype));
  EXPECT_EQ(emb_defaults.default_row_index, emb_cfg.default_row_index);

  nve_linear_uvm_layer_config_t uvm_cfg;
  const nve::LinearUVMEmbeddingLayer<int64_t>::Config uvm_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_linear_uvm_layer_config_init(&uvm_cfg, sizeof(uvm_cfg)));
  EXPECT_EQ(uvm_defaults.min_insert_freq_gpu, uvm_cfg.min_insert_freq_gpu);
  EXPECT_EQ(uvm_defaults.min_insert_size_gpu, uvm_cfg.min_insert_size_gpu);
  EXPECT_EQ(nullptr, uvm_cfg.insert_heuristic);
  EXPECT_EQ(uvm_defaults.default_row_index, uvm_cfg.default_row_index);

  nve_hierarchical_layer_config_t hier_cfg;
  const nve::HierarchicalEmbeddingLayer<int64_t>::Config hier_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_hierarchical_layer_config_init(&hier_cfg, sizeof(hier_cfg)));
  EXPECT_EQ(hier_defaults.min_insert_freq_gpu, hier_cfg.min_insert_freq_gpu);
  EXPECT_EQ(hier_defaults.min_insert_freq_host, hier_cfg.min_insert_freq_host);
  EXPECT_EQ(hier_defaults.min_insert_size_gpu, hier_cfg.min_insert_size_gpu);
  EXPECT_EQ(hier_defaults.min_insert_size_host, hier_cfg.min_insert_size_host);

  nve_host_embedding_layer_config_t host_layer_cfg;
  const nve::HostEmbeddingLayer<int64_t>::Config host_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&host_layer_cfg, sizeof(host_layer_cfg)));
  EXPECT_STREQ(host_defaults.layer_name.c_str(), host_layer_cfg.layer_name);
  EXPECT_EQ(host_defaults.default_embedding.empty(), host_layer_cfg.default_embedding == nullptr);
  EXPECT_EQ(static_cast<int64_t>(host_defaults.default_embedding.size()),
            host_layer_cfg.default_embedding_size);
  EXPECT_EQ(host_defaults.task_size_threshold, host_layer_cfg.task_size_threshold);
  EXPECT_EQ(host_defaults.disable_threadpool, host_layer_cfg.disable_threadpool);
}

TEST(NveCApiBasic, DestroyNull) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_layer_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_context_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_heuristic_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_thread_pool_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_factory_destroy(nullptr));
}

TEST(NveCApiBasic, HeuristicCreateDestroy) {
  float thresholds[] = {0.8f, 0.9f};

  nve_heuristic_t h = nullptr;
  EXPECT_EQ(NVE_SUCCESS, nve_heuristic_create_default(&h, thresholds, 2));
  EXPECT_NE(nullptr, h);
  EXPECT_EQ(NVE_SUCCESS, nve_heuristic_destroy(h));

  h = nullptr;
  EXPECT_EQ(NVE_SUCCESS, nve_heuristic_create_never(&h));
  EXPECT_NE(nullptr, h);
  EXPECT_EQ(NVE_SUCCESS, nve_heuristic_destroy(h));
}

TEST(NveCApiBasic, HeuristicInvalidArgs) {
  nve_heuristic_t h = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_heuristic_create_default(nullptr, nullptr, 0));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_heuristic_create_default(&h, nullptr, 1));
}

TEST(NveCApiBasic, CreateTableFactoryInvalidArgs) {
  nve_table_factory_t factory = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_create_table_factory(nullptr, "libnve-plugin-stl-map.so", "{}"));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_create_table_factory(&factory, nullptr, "{}"));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_create_table_factory(&factory, "", "{}"));
  // Loading a non-existent shared object should produce a runtime error.
  EXPECT_NE(NVE_SUCCESS, nve_create_table_factory(&factory, "nonexistent_plugin_xyz.so", "{}"));
}

// A bare soname resolves through the loader's fallback beside libnve-common.so.
TEST(NveCApiBasic, CreateTableFactoryBareSoname) {
  nve_table_factory_t factory = nullptr;
  EXPECT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));
  EXPECT_NE(nullptr, factory);
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(factory));
}

TEST(NveCApiBasic, CreateTableFactoryFullPath) {
  nve_table_factory_t factory = nullptr;
  const std::string path = nve_test::plugin_full_path("stl-map");
  EXPECT_EQ(NVE_SUCCESS, nve_create_table_factory(&factory, path.c_str(), "{}"));
  EXPECT_NE(nullptr, factory);
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(factory));
}
