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
 * NVE Sample: Custom Remote Host Table (plugin-based)
 *
 * This sample shows how to use a custom "remote" host table plugin with
 * NVE's three-tier hierarchical embedding layer:
 *
 *   GPU table  (L1, ~microseconds  -- GPU SRAM / HBM cache)
 *     -> phmap host table  (L2, ~microseconds  -- CPU RAM cache, loaded via plugin)
 *       -> custom_remote   (L3, custom        -- your parameter server / DB)
 *
 * The custom_remote plugin (plugins/custom_remote/) implements a simple
 * std::map-backed host table.  A real remote table would replace the std::map
 * with network I/O to a parameter server or distributed KV store.
 *
 * The custom table implementation lives in a separate shared object
 * (libnve-plugin-custom_remote.so) and is loaded at runtime via
 * nve_load_host_table_plugin().  This sample uses the C API exclusively.
 */

#include <nve_c_api.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

/* ============================================================================
 * Error-handling helpers
 * ============================================================================ */

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_err));                    \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define CHECK_NVE(call)                                                         \
  do {                                                                          \
    nve_status_t _st = (call);                                                  \
    if (_st != NVE_SUCCESS) {                                                   \
      const char* _msg = nullptr;                                               \
      nve_get_last_error(&_msg);                                                \
      fprintf(stderr, "NVE error at %s:%d — %s\n", __FILE__, __LINE__, _msg);  \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

/* ============================================================================
 * Main — Wire everything together using the C API
 * ============================================================================ */

int main() {
  // Print NVE library version
  int32_t major, minor, patch;
  CHECK_NVE(nve_version(&major, &minor, &patch));
  printf("NVE version: %d.%d.%d\n\n", major, minor, patch);

  /* Configuration ---------------------------------------------------------- */
  const int     device_id      = 0;
  const int64_t embedding_dim  = 32;    // number of floats per embedding row
  const int64_t row_size       = embedding_dim * static_cast<int64_t>(sizeof(float));
  const int64_t num_embeddings = 1000;  // total embeddings loaded into remote table
  const int64_t lookup_batch   = 16;    // number of keys per lookup call

  CHECK_CUDA(cudaSetDevice(device_id));

  /* -- Step 1: GPU table (L1 cache) ---------------------------------------- */
  printf("[1] Creating GPU table (L1 cache)...\n");

  nve_gpu_table_config_t gpu_cfg = nve_gpu_table_config_default();
  gpu_cfg.device_id         = device_id;
  gpu_cfg.cache_size        = 4 * 1024 * 1024;  // 4 MB GPU cache
  gpu_cfg.row_size_in_bytes = row_size;
  gpu_cfg.count_misses      = 1;
  gpu_cfg.value_dtype       = NVE_DTYPE_FLOAT32;

  nve_table_t gpu_table = nullptr;
  CHECK_NVE(nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &gpu_cfg,
                                 /*allocator=*/nullptr));
  printf("  GPU table created (4 MB cache, %ld-byte rows)\n", row_size);

  /* -- Step 2: phmap host table (L2 cache) --------------------------------- */
  printf("\n[2] Creating phmap host table (L2 cache)...\n");

  CHECK_NVE(nve_load_host_table_plugin("phmap"));

  nve_host_factory_t phmap_factory = nullptr;
  CHECK_NVE(nve_create_host_table_factory(
      &phmap_factory, R"({"implementation": "phmap_flat_map"})"));

  nve_table_t phmap_table = nullptr;
  CHECK_NVE(nve_host_factory_produce(
      phmap_factory, /*table_id=*/1,
      R"({
        "mask_size"        : 8,
        "key_size"         : 8,
        "max_value_size"   : 128,
        "value_dtype"      : "float32",
        "num_partitions"   : 4,
        "initial_capacity" : 4096,
        "value_alignment"  : 32
      })",
      &phmap_table));

  printf("  phmap host table created\n");

  /* -- Step 3: Custom remote table (L3) via plugin ------------------------- */
  printf("\n[3] Creating custom remote table (L3) via custom_remote plugin...\n");

  // Load the custom_remote plugin — this is the custom remote table
  // implementation that lives in libnve-plugin-custom_remote.so.
  // See plugins/custom_remote/ for the source.
  CHECK_NVE(nve_load_host_table_plugin("custom_remote"));

  nve_host_factory_t remote_factory = nullptr;
  CHECK_NVE(nve_create_host_table_factory(
      &remote_factory, R"({"implementation": "custom_remote"})"));

  nve_table_t remote_table = nullptr;
  CHECK_NVE(nve_host_factory_produce(
      remote_factory, /*table_id=*/2,
      R"({
        "mask_size"        : 8,
        "key_size"         : 8,
        "max_value_size"   : 128,
        "value_dtype"      : "float32"
      })",
      &remote_table));

  printf("  custom_remote host table created (%ld-byte rows)\n", row_size);

  /* -- Step 4: Build the hierarchical layer -------------------------------- */
  printf("\n[4] Building hierarchical embedding layer...\n");

  nve_hierarchical_layer_config_t layer_cfg =
      nve_hierarchical_layer_config_default();
  layer_cfg.layer_name           = "custom_remote_sample";
  layer_cfg.min_insert_size_gpu  = 16;
  layer_cfg.min_insert_size_host = 16;

  nve_table_t tables[3] = {gpu_table, phmap_table, remote_table};

  nve_layer_t layer = nullptr;
  CHECK_NVE(nve_hierarchical_layer_create(
      &layer, NVE_KEY_INT64, &layer_cfg, tables, 3, /*allocator=*/nullptr));

  nve_context_t ctx = nullptr;
  CHECK_NVE(nve_layer_create_execution_context(
      layer, &ctx,
      /*lookup_stream=*/nullptr,
      /*modify_stream=*/nullptr,
      /*thread_pool=*/nullptr,
      /*allocator=*/nullptr));

  printf("  Hierarchical layer: GPU cache -> phmap host table -> custom_remote\n");

  /* -- Step 5: Populate the remote table ----------------------------------- */
  printf("\n[5] Inserting %ld embeddings into the remote table (L3)...\n",
         num_embeddings);

  std::vector<int64_t> all_keys(static_cast<size_t>(num_embeddings));
  std::vector<float>   all_values(
      static_cast<size_t>(num_embeddings * embedding_dim));

  for (int64_t i = 0; i < num_embeddings; ++i) {
    all_keys[static_cast<size_t>(i)] = i;
    const float val = static_cast<float>(i) * 0.01f;
    for (int64_t d = 0; d < embedding_dim; ++d)
      all_values[static_cast<size_t>(i * embedding_dim + d)] = val;
  }

  // table_id 2 targets the custom_remote table directly.
  CHECK_NVE(nve_layer_insert(layer, ctx, num_embeddings, all_keys.data(),
                             row_size, row_size, all_values.data(),
                             /*table_id=*/2));
  CHECK_NVE(nve_context_wait(ctx));
  printf("  Inserted %ld embeddings into custom_remote table\n", num_embeddings);

  /* -- Step 6: Lookup — observe three-tier traversal ----------------------- */
  printf("\n[6] Looking up %ld keys (expect GPU miss -> phmap miss -> remote hit)...\n",
         lookup_batch);

  std::vector<int64_t> lookup_keys(static_cast<size_t>(lookup_batch));
  for (int64_t i = 0; i < lookup_batch; ++i)
    lookup_keys[static_cast<size_t>(i)] = i * (num_embeddings / lookup_batch);

  void* output = nullptr;
  CHECK_CUDA(cudaMallocHost(&output, static_cast<size_t>(lookup_batch) * static_cast<size_t>(row_size)));

  // hitrates[0] = GPU tier hit rate, [1] = phmap tier, [2] = remote tier
  float hitrates[3] = {0.0f, 0.0f, 0.0f};
  CHECK_NVE(nve_layer_lookup(layer, ctx, lookup_batch, lookup_keys.data(),
                             output, row_size,
                             /*hitmask=*/nullptr, hitrates));
  CHECK_NVE(nve_context_wait(ctx));

  printf("  Hit rates — GPU: %.1f%%  phmap: %.1f%%  remote: %.1f%%\n",
         hitrates[0] * 100.0f,
         hitrates[1] * 100.0f,
         hitrates[2] * 100.0f);

  /* -- Step 7: Verify results ---------------------------------------------- */
  printf("\n[7] Verifying results...\n");

  const auto* out_floats = static_cast<const float*>(output);
  int errors = 0;
  for (int64_t i = 0; i < lookup_batch; ++i) {
    const float expected = static_cast<float>(lookup_keys[static_cast<size_t>(i)]) * 0.01f;
    const float actual   = out_floats[static_cast<size_t>(i) * static_cast<size_t>(embedding_dim)];
    const bool  ok       = (actual == expected);
    if (!ok) ++errors;

    if (i < 5 || !ok) {
      printf("  key=%4ld  expected=%.4f  got=%.4f  %s\n",
             lookup_keys[static_cast<size_t>(i)], expected, actual,
             ok ? "OK" : "MISMATCH");
    }
  }
  if (errors == 0) {
    printf("  All %ld lookups verified successfully!\n", lookup_batch);
  } else {
    printf("  %d/%ld lookups had mismatches\n", errors, lookup_batch);
  }

  /* -- Cleanup ------------------------------------------------------------- */
  printf("\n[8] Cleaning up...\n");
  CHECK_CUDA(cudaFreeHost(output));
  CHECK_NVE(nve_context_destroy(ctx));
  CHECK_NVE(nve_layer_destroy(layer));
  CHECK_NVE(nve_table_destroy(remote_table));
  CHECK_NVE(nve_table_destroy(phmap_table));
  CHECK_NVE(nve_table_destroy(gpu_table));
  CHECK_NVE(nve_host_factory_destroy(remote_factory));
  CHECK_NVE(nve_host_factory_destroy(phmap_factory));
  printf("  Done.\n");

  return errors == 0 ? 0 : 1;
}
