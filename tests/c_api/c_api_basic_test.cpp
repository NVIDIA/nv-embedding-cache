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

TEST(NveCApiBasic, GetLastErrorEmpty) {
  const char* msg = nullptr;
  EXPECT_EQ(NVE_SUCCESS, nve_get_last_error(&msg));
  EXPECT_NE(nullptr, msg);
}

TEST(NveCApiBasic, ConfigDefaults) {
  auto gpu_cfg = nve_gpu_table_config_default();
  EXPECT_EQ(0, gpu_cfg.device_id);
  EXPECT_EQ(1, gpu_cfg.count_misses);
  EXPECT_EQ(1 << 20, gpu_cfg.max_modify_size);
  EXPECT_EQ(1, gpu_cfg.uvm_cpu_accumulate);
  EXPECT_EQ(1, gpu_cfg.modify_on_gpu);
  EXPECT_EQ(NVE_DTYPE_UNKNOWN, gpu_cfg.value_dtype);

  auto emb_cfg = nve_gpu_embedding_layer_config_default();
  EXPECT_EQ(0, emb_cfg.device_id);
  EXPECT_EQ(NVE_DTYPE_UNKNOWN, emb_cfg.value_dtype);

  auto uvm_cfg = nve_linear_uvm_layer_config_default();
  EXPECT_EQ(0, uvm_cfg.min_insert_freq_gpu);
  EXPECT_EQ(1 << 16, uvm_cfg.min_insert_size_gpu);
  EXPECT_EQ(nullptr, uvm_cfg.insert_heuristic);

  auto hier_cfg = nve_hierarchical_layer_config_default();
  EXPECT_EQ(0, hier_cfg.min_insert_freq_gpu);
  EXPECT_EQ(0, hier_cfg.min_insert_freq_host);
  EXPECT_EQ(1 << 16, hier_cfg.min_insert_size_gpu);
  EXPECT_EQ(0, hier_cfg.min_insert_size_host);

  auto overflow_cfg = nve_overflow_policy_config_default();
  EXPECT_EQ(NVE_OVERFLOW_EVICT_RANDOM, overflow_cfg.handler);
  EXPECT_DOUBLE_EQ(0.8, overflow_cfg.resolution_margin);

  auto host_cfg = nve_host_table_config_default();
  EXPECT_EQ(8, host_cfg.max_value_size);
  EXPECT_EQ(NVE_DTYPE_UNKNOWN, host_cfg.value_dtype);
}

TEST(NveCApiBasic, DestroyNull) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_layer_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_context_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_heuristic_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_thread_pool_destroy(nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_host_factory_destroy(nullptr));
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

TEST(NveCApiBasic, LoadPluginInvalidName) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_load_host_table_plugin(nullptr));
  // Loading a non-existent plugin should produce a runtime error
  EXPECT_NE(NVE_SUCCESS, nve_load_host_table_plugin("nonexistent_plugin_xyz"));
}
