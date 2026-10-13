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

/*
 * Consumer for the sample external (pure-C ABI) plugin. Creates the factory
 * from the plugin SO through the C API, exercises every table operation
 * directly, then uses the same table behind a hierarchical embedding layer —
 * exactly like any in-tree table.
 */

#include <nve_c_api.h>

#include <bit_ops.hpp>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_NVE(expr)                                                        \
  do {                                                                         \
    nve_status_t status_ = (expr);                                             \
    if (status_ != NVE_SUCCESS) {                                              \
      const char* msg_ = "";                                                   \
      nve_get_last_error(&msg_);                                               \
      std::fprintf(stderr, "%s:%d: %s failed (%d): %s\n", __FILE__, __LINE__,  \
                   #expr, (int)status_, msg_);                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,    \
                   #cond);                                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace {

constexpr int64_t kNumKeys = 256;
constexpr int64_t kRowFloats = 16;
constexpr int64_t kRowBytes = kRowFloats * static_cast<int64_t>(sizeof(float));

}  // namespace

int main() {
  /* One call selects the plugin SO and creates its factory. */
  nve_table_factory_t factory = NULL;
  CHECK_NVE(nve_create_table_factory(&factory, "libnve-plugin-external-map.so",
                                     "{\"num_buckets\": 512}"));

  nve_table_t table = NULL;
  CHECK_NVE(nve_table_factory_produce(factory, 1000, "{\"max_value_size\": 64}", &table));

  /* The factory may be destroyed early; the table stays valid. */
  CHECK_NVE(nve_table_factory_destroy(factory));

  int64_t key_size = 0;
  CHECK_NVE(nve_table_get_key_size(table, &key_size));
  CHECK(key_size == 8);
  int32_t device_id = 0;
  CHECK_NVE(nve_table_get_device_id(table, &device_id));
  CHECK(device_id == -1);

  nve_context_t ctx = NULL;
  CHECK_NVE(nve_table_create_execution_context(table, &ctx, NULL, NULL, NULL, NULL));

  std::vector<int64_t> keys(kNumKeys);
  std::vector<float> values(kNumKeys * kRowFloats);
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = 100 + static_cast<int64_t>(i);
    for (size_t j = 0; j < static_cast<size_t>(kRowFloats); ++j) {
      values[i * static_cast<size_t>(kRowFloats) + j] = static_cast<float>(i);
    }
  }

  /* insert + find (with output values) */
  CHECK_NVE(nve_table_insert(table, ctx, kNumKeys, keys.data(), kRowBytes, kRowBytes,
                             values.data()));
  CHECK_NVE(nve_table_reset_lookup_counter(table, ctx));
  std::vector<float> found(kNumKeys * kRowFloats, -1.0f);
  std::vector<uint64_t> hit_mask(nve::ceil_div(kNumKeys, static_cast<int64_t>(64)), 0);
  CHECK_NVE(nve_table_find(table, ctx, kNumKeys, keys.data(), hit_mask.data(), kRowBytes,
                           found.data(), NULL));
  for (size_t i = 0; i < keys.size(); ++i) {
    CHECK((hit_mask[i / 64] >> (i % 64)) & 1u);
    CHECK(found[i * static_cast<size_t>(kRowFloats)] == static_cast<float>(i));
  }
  int64_t counter = 0;
  CHECK_NVE(nve_table_get_lookup_counter(table, ctx, &counter));
  CHECK(counter == kNumKeys); /* external table counts hits */

  /* count-only find: NULL values is an exists check */
  std::vector<uint64_t> exists_mask(nve::ceil_div(kNumKeys, static_cast<int64_t>(64)), 0);
  std::vector<int64_t> miss_keys = {1, 2, 3};
  CHECK_NVE(nve_table_find(table, ctx, 3, miss_keys.data(), exists_mask.data(), 0, NULL, NULL));
  CHECK(exists_mask[0] == 0);

  /* update overwrites existing keys only */
  for (auto& v : values) v += 1000.0f;
  CHECK_NVE(nve_table_update(table, ctx, kNumKeys, keys.data(), kRowBytes, kRowBytes,
                             values.data()));

  /* update_accumulate adds float32 updates into existing rows */
  std::vector<float> updates(kNumKeys * kRowFloats, 1.0f);
  CHECK_NVE(nve_table_update_accumulate(table, ctx, kNumKeys, keys.data(), kRowBytes,
                                        kRowBytes, updates.data(), NVE_DTYPE_FLOAT32));
  std::fill(found.begin(), found.end(), -1.0f);
  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  CHECK_NVE(nve_table_find(table, ctx, kNumKeys, keys.data(), hit_mask.data(), kRowBytes,
                           found.data(), NULL));
  CHECK(found[0] == 1001.0f);

  /* erase + clear */
  CHECK_NVE(nve_table_erase(table, ctx, 1, keys.data()));
  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  CHECK_NVE(nve_table_find(table, ctx, 1, keys.data(), hit_mask.data(), 0, NULL, NULL));
  CHECK(hit_mask[0] == 0);
  CHECK_NVE(nve_table_clear(table, ctx));

  CHECK_NVE(nve_context_destroy(ctx));

  /* The same external table drives a hierarchical embedding layer. */
  nve_hierarchical_layer_config_t layer_cfg;
  CHECK_NVE(nve_hierarchical_layer_config_init(&layer_cfg, sizeof(layer_cfg)));
  layer_cfg.layer_name = "external_plugin_sample";
  nve_layer_t layer = NULL;
  CHECK_NVE(nve_hierarchical_layer_create(&layer, NVE_KEY_INT64, &layer_cfg, &table, 1, NULL));

  nve_context_t layer_ctx = NULL;
  CHECK_NVE(nve_layer_create_execution_context(layer, &layer_ctx, NULL, NULL, NULL, NULL));

  CHECK_NVE(nve_layer_insert(layer, layer_ctx, kNumKeys, keys.data(), kRowBytes, kRowBytes,
                             values.data(), 0));
  std::vector<float> output(kNumKeys * kRowFloats, -1.0f);
  std::vector<uint64_t> layer_mask(nve::ceil_div(kNumKeys, static_cast<int64_t>(64)), 0);
  CHECK_NVE(nve_layer_lookup(layer, layer_ctx, kNumKeys, keys.data(), output.data(), kRowBytes,
                             layer_mask.data(), NULL));
  CHECK_NVE(nve_context_wait(layer_ctx));
  for (size_t i = 0; i < keys.size(); ++i) {
    CHECK((layer_mask[i / 64] >> (i % 64)) & 1u);
    CHECK(output[i * static_cast<size_t>(kRowFloats)] ==
          values[i * static_cast<size_t>(kRowFloats)]);
  }

  CHECK_NVE(nve_context_destroy(layer_ctx));
  CHECK_NVE(nve_layer_destroy(layer));
  CHECK_NVE(nve_table_destroy(table));

  std::printf("external plugin sample: OK\n");
  return 0;
}
