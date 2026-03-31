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
 * NVE C API Sample: Hierarchical Embedding Layer
 *
 * Demonstrates creating a hierarchical embedding layer with GPU cache + NVHM
 * host table, inserting embeddings, and performing lookups via the C API.
 */

#include <nve_c_api.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                       \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

#define CHECK_NVE(call)                                                      \
  do {                                                                       \
    nve_status_t st = (call);                                                \
    if (st != NVE_SUCCESS) {                                                 \
      const char* msg = nullptr;                                             \
      nve_get_last_error(&msg);                                              \
      fprintf(stderr, "NVE error at %s:%d: %s\n", __FILE__, __LINE__, msg);  \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

int main() {
  // Print NVE version
  int32_t major, minor, patch;
  CHECK_NVE(nve_version(&major, &minor, &patch));
  printf("NVE version: %d.%d.%d\n", major, minor, patch);

  // Configuration
  const int device_id = 0;
  const int64_t embedding_dim = 32;                    // 32 floats per embedding
  const int64_t row_size = embedding_dim * sizeof(float);  // 128 bytes
  const int64_t num_embeddings = 1000;                 // total embeddings to insert
  const int64_t lookup_batch = 16;                     // keys per lookup

  CHECK_CUDA(cudaSetDevice(device_id));

  // ── Step 1: Create GPU table (cache layer) ────────────────────────────
  printf("\n[1] Creating GPU table (cache)...\n");

  auto gpu_cfg = nve_gpu_table_config_default();
  gpu_cfg.device_id = device_id;
  gpu_cfg.cache_size = 4 * 1024 * 1024;  // 4 MB GPU cache
  gpu_cfg.row_size_in_bytes = row_size;
  gpu_cfg.uvm_table = nullptr;            // no UVM backing — misses go to host
  gpu_cfg.count_misses = 1;
  gpu_cfg.value_dtype = NVE_DTYPE_FLOAT32;

  nve_table_t gpu_table = nullptr;
  CHECK_NVE(nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &gpu_cfg, nullptr));
  printf("  GPU table created (4 MB cache, %ld-byte rows)\n", row_size);

  // ── Step 2: Create NVHM host table ────────────────────────────────────
  printf("\n[2] Creating NVHM host table...\n");

  CHECK_NVE(nve_load_host_table_plugin("nvhm"));

  nve_host_factory_t factory = nullptr;
  CHECK_NVE(nve_create_host_table_factory(&factory, R"({"implementation": "nvhm_map"})"));

  const char* host_table_config = R"({
    "mask_size": 8,
    "key_size": 8,
    "max_value_size": 128,
    "value_dtype": "float32",
    "num_partitions": 4,
    "initial_capacity": 4096,
    "value_alignment": 32
  })";

  nve_table_t host_table = nullptr;
  CHECK_NVE(nve_host_factory_produce(factory, 0, host_table_config, &host_table));
  printf("  NVHM host table created\n");

  // ── Step 3: Create hierarchical layer (GPU cache → host table) ────────
  printf("\n[3] Creating hierarchical embedding layer...\n");

  auto hier_cfg = nve_hierarchical_layer_config_default();
  hier_cfg.layer_name = "sample_hier_layer";
  hier_cfg.min_insert_size_gpu = 16;
  hier_cfg.min_insert_size_host = 16;

  nve_table_t tables[] = {gpu_table, host_table};
  nve_layer_t layer = nullptr;
  CHECK_NVE(nve_hierarchical_layer_create(&layer, NVE_KEY_INT64, &hier_cfg, tables, 2, nullptr));

  nve_context_t ctx = nullptr;
  CHECK_NVE(nve_layer_create_execution_context(layer, &ctx, nullptr, nullptr, nullptr, nullptr));
  printf("  Hierarchical layer created: GPU cache -> NVHM host table\n");

  // ── Step 4: Insert embeddings ─────────────────────────────────────────
  printf("\n[4] Inserting %ld embeddings...\n", num_embeddings);

  std::vector<int64_t> all_keys(num_embeddings);
  std::vector<float> all_values(num_embeddings * embedding_dim);

  for (int64_t i = 0; i < num_embeddings; ++i) {
    all_keys[static_cast<size_t>(i)] = i;
    // Each embedding row: [key_id * 0.01, key_id * 0.01, ...]
    for (int64_t d = 0; d < embedding_dim; ++d) {
      all_values[static_cast<size_t>(i * embedding_dim + d)] = static_cast<float>(i) * 0.01f;
    }
  }

  // Insert some keys to the GPU table
  int64_t embedding_subset = num_embeddings/4;
  CHECK_NVE(nve_layer_insert(layer, ctx, embedding_subset,
                              all_keys.data(),
                              row_size, row_size,
                              all_values.data(), 0));
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("  Inserted %ld embeddings to the GPU table\n", embedding_subset);

  // Insert in batches to the NVHM table
  const int64_t insert_batch = 256;
  for (int64_t offset = 0; offset < num_embeddings; offset += insert_batch) {
    int64_t n = std::min(insert_batch, num_embeddings - offset);
    CHECK_NVE(nve_layer_insert(layer, ctx, n,
                               all_keys.data() + offset,
                               row_size, row_size,
                               all_values.data() + offset * embedding_dim, 1));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("  Inserted %ld embeddings to the host table\n", num_embeddings);

  // ── Step 5: Lookup embeddings ─────────────────────────────────────────
  printf("\n[5] Looking up %ld keys...\n", lookup_batch);

  // Pick keys spread across the range
  std::vector<int64_t> lookup_keys(lookup_batch);
  for (int64_t i = 0; i < lookup_batch; ++i) {
    lookup_keys[static_cast<size_t>(i)] = i * (num_embeddings / lookup_batch);
  }

  void* output = nullptr;
  CHECK_CUDA(cudaMallocHost(&output, lookup_batch * row_size));

  float hitrates[2] = {0.0f, 0.0f};
  CHECK_NVE(nve_layer_lookup(layer, ctx, lookup_batch, lookup_keys.data(),
                             output, row_size, nullptr, hitrates));
  CHECK_CUDA(cudaDeviceSynchronize());

  printf("  Hit rates: GPU=%.1f%%, Host=%.1f%%\n",
         hitrates[0] * 100.0f, hitrates[1] * 100.0f);

  // Verify results
  auto* out_floats = static_cast<float*>(output);
  printf("\n  Sample results:\n");
  int errors = 0;
  for (uint64_t i = 0; i < lookup_batch; ++i) {
    float expected = static_cast<float>(lookup_keys[i]) * 0.01f;
    float actual = out_floats[i * embedding_dim];
    bool match = (actual == expected);
    if (!match) errors++;

    if (i < 5 || !match) {
      printf("    key=%4ld  expected=%.4f  got=%.4f  %s\n",
             lookup_keys[i], expected, actual, match ? "OK" : "MISMATCH");
    }
  }
  if (errors == 0) {
    printf("  All %ld lookups verified successfully!\n", lookup_batch);
  } else {
    printf("  %d/%ld lookups had mismatches\n", errors, lookup_batch);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────
  printf("\n[6] Cleaning up...\n");
  cudaFreeHost(output);
  nve_context_wait(ctx);
  nve_context_destroy(ctx);
  nve_layer_destroy(layer);
  nve_table_destroy(gpu_table);
  nve_table_destroy(host_table);
  nve_host_factory_destroy(factory);
  printf("  Done.\n");

  return 0;
}
