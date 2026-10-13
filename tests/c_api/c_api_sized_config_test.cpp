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
#include <linear_host_table.hpp>
#include <plugin/external_plugin_table.hpp>

#include "../../src/nve_c_api_config.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <sys/mman.h>
#include <unistd.h>

// Defined in c_api_abi_test.c, compiled as C against frozen ABI 2.0 declarations.
extern "C" int nve_frozen_abi_2_0_client(void);

namespace {

nve_key_type_t InvalidKeyType() {
  nve_key_type_t key_type;
  const int32_t raw = 99;
  static_assert(sizeof(key_type) == sizeof(raw));
  std::memcpy(&key_type, &raw, sizeof(raw));
  return key_type;
}

// Tests inspect the library-private header encoding.
template <typename Config>
uint64_t ConfigSize(const Config& config) {
  return config.header & nve_config_size_mask;
}

template <typename Config>
uint64_t ConfigTag(const Config& config) {
  return config.header >> nve_config_tag_shift;
}

// Replaces the size bits of an initialized config, keeping its type tag.
template <typename Config>
void SetConfigSize(Config& config, uint64_t size) {
  config.header = (config.header & ~nve_config_size_mask) | size;
}

template <typename Config>
struct ConfigApi;

template <>
struct ConfigApi<nve_gpu_table_config_t> {
  using Config = nve_gpu_table_config_t;
  using Handle = nve_table_t;
  inline static constexpr auto init = nve_gpu_table_config_init;
  static constexpr size_t prefix_size = offsetof(Config, uvm_num_rows);
  static constexpr size_t partial_size = offsetof(Config, device_id) + 1;
  static nve_status_t create(Handle* out, const Config* config) {
    return nve_gpu_table_create(out, InvalidKeyType(), config, nullptr);
  }
};

template <>
struct ConfigApi<nve_linear_host_table_config_t> {
  using Config = nve_linear_host_table_config_t;
  using Handle = nve_table_t;
  inline static constexpr auto init = nve_linear_host_table_config_init;
  static constexpr size_t prefix_size = offsetof(Config, max_threads);
  static constexpr size_t partial_size = offsetof(Config, emb_table) + 1;
  static nve_status_t create(Handle* out, const Config* config) {
    return nve_linear_host_table_create(out, InvalidKeyType(), config);
  }
};

template <>
struct ConfigApi<nve_gpu_embedding_layer_config_t> {
  using Config = nve_gpu_embedding_layer_config_t;
  using Handle = nve_layer_t;
  inline static constexpr auto init = nve_gpu_embedding_layer_config_init;
  static constexpr size_t prefix_size = offsetof(Config, default_row_index);
  static constexpr size_t partial_size = offsetof(Config, layer_name) + 1;
  static nve_status_t create(Handle* out, const Config* config) {
    return nve_gpu_embedding_layer_create(out, InvalidKeyType(), config, nullptr);
  }
};

template <>
struct ConfigApi<nve_linear_uvm_layer_config_t> {
  using Config = nve_linear_uvm_layer_config_t;
  using Handle = nve_layer_t;
  inline static constexpr auto init = nve_linear_uvm_layer_config_init;
  static constexpr size_t prefix_size = offsetof(Config, default_row_index);
  static constexpr size_t partial_size = offsetof(Config, layer_name) + 1;
  static nve_status_t create(Handle* out, const Config* config) {
    return nve_linear_uvm_layer_create(out, NVE_KEY_INT64, config, nullptr, nullptr);
  }
};

template <>
struct ConfigApi<nve_hierarchical_layer_config_t> {
  using Config = nve_hierarchical_layer_config_t;
  using Handle = nve_layer_t;
  inline static constexpr auto init = nve_hierarchical_layer_config_init;
  static constexpr size_t prefix_size = offsetof(Config, default_embedding_size);
  static constexpr size_t partial_size = offsetof(Config, layer_name) + 1;
  static nve_status_t create(Handle* out, const Config* config) {
    return nve_hierarchical_layer_create(out, NVE_KEY_INT64, config, nullptr, 0, nullptr);
  }
};

template <>
struct ConfigApi<nve_host_embedding_layer_config_t> {
  using Config = nve_host_embedding_layer_config_t;
  using Handle = nve_layer_t;
  inline static constexpr auto init = nve_host_embedding_layer_config_init;
  static constexpr size_t prefix_size = offsetof(Config, task_size_threshold);
  static constexpr size_t partial_size = offsetof(Config, layer_name) + 1;
  static nve_status_t create(Handle* out, const Config* config) {
    return nve_host_embedding_layer_create(out, NVE_KEY_INT64, config, nullptr, nullptr);
  }
};

template <typename Config>
class NveCApiSizedConfig : public ::testing::Test {};

using SizedConfigs = ::testing::Types<
    nve_gpu_table_config_t, nve_linear_host_table_config_t, nve_gpu_embedding_layer_config_t,
    nve_linear_uvm_layer_config_t, nve_hierarchical_layer_config_t,
    nve_host_embedding_layer_config_t>;
TYPED_TEST_SUITE(NveCApiSizedConfig, SizedConfigs);

TYPED_TEST(NveCApiSizedConfig, InvalidInitializersLeaveStorageUnchanged) {
  using Api = ConfigApi<TypeParam>;
  alignas(TypeParam) std::array<unsigned char, sizeof(TypeParam) + 8> storage;
  storage.fill(0xa5);
  const auto before = storage;
  auto* config = reinterpret_cast<TypeParam*>(storage.data());
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::init(nullptr, sizeof(TypeParam)));
  for (const auto size : {size_t{0}, sizeof(uint64_t) - 1, Api::partial_size}) {
    EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::init(config, size));
    EXPECT_EQ(before, storage);
  }
  EXPECT_EQ(NVE_ERROR_NOT_IMPLEMENTED, Api::init(config, sizeof(TypeParam) + 8));
  EXPECT_EQ(before, storage);
  EXPECT_EQ(NVE_ERROR_NOT_IMPLEMENTED, Api::init(config, std::numeric_limits<size_t>::max()));
  EXPECT_EQ(before, storage);
}

TYPED_TEST(NveCApiSizedConfig, InitializerPreservesBytesBeyondCallerSize) {
  using Api = ConfigApi<TypeParam>;
  alignas(TypeParam) std::array<unsigned char, sizeof(TypeParam) + 8> storage;
  for (const auto size : {sizeof(uint64_t), Api::prefix_size, sizeof(TypeParam)}) {
    storage.fill(0xa5);
    auto* config = reinterpret_cast<TypeParam*>(storage.data());
    ASSERT_EQ(NVE_SUCCESS, Api::init(config, size));
    EXPECT_EQ(size, ConfigSize(*config));
    for (size_t i = size; i < storage.size(); ++i) {
      EXPECT_EQ(0xa5, storage[i]) << "byte " << i;
    }
  }
}

TYPED_TEST(NveCApiSizedConfig, InvalidCreationClearsOutput) {
  using Api = ConfigApi<TypeParam>;
  typename Api::Handle out = reinterpret_cast<typename Api::Handle>(uintptr_t{1});
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::create(&out, nullptr));
  EXPECT_EQ(nullptr, out);
  TypeParam config;
  ASSERT_EQ(NVE_SUCCESS, Api::init(&config, sizeof(config)));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::create(nullptr, &config));
  for (const auto size : {uint64_t{0}, uint64_t{7}, uint64_t{Api::partial_size}}) {
    SetConfigSize(config, size);
    out = reinterpret_cast<typename Api::Handle>(uintptr_t{1});
    EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::create(&out, &config));
    EXPECT_EQ(nullptr, out);
  }
  // Untagged sizes, as left by uninitialized storage or a hand-set header, are rejected.
  for (const auto size : {uint64_t{sizeof(uint64_t)}, uint64_t{Api::prefix_size}, uint64_t{sizeof(config)},
                          std::numeric_limits<uint64_t>::max()}) {
    config.header = size;
    out = reinterpret_cast<typename Api::Handle>(uintptr_t{1});
    EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::create(&out, &config));
    EXPECT_EQ(nullptr, out);
  }
  ASSERT_EQ(NVE_SUCCESS, Api::init(&config, sizeof(config)));
  SetConfigSize(config, 0xFFFFFFFF);
  out = reinterpret_cast<typename Api::Handle>(uintptr_t{1});
  EXPECT_EQ(NVE_ERROR_NOT_IMPLEMENTED, Api::create(&out, &config));
  EXPECT_EQ(nullptr, out);
}

// Put each short config immediately before an inaccessible page: even a one-byte
// read or write past the caller's size fails, rather than landing in a larger object.
TYPED_TEST(NveCApiSizedConfig, ShortAllocationAtGuardPage) {
  using Api = ConfigApi<TypeParam>;
  const auto page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  void* memory = mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(MAP_FAILED, memory);
  auto* guard = static_cast<unsigned char*>(memory) + page_size;
  ASSERT_EQ(0, mprotect(guard, page_size, PROT_NONE));
  for (const auto size : {sizeof(uint64_t), Api::prefix_size}) {
    auto* config = reinterpret_cast<TypeParam*>(guard - size);
    EXPECT_EQ(NVE_SUCCESS, Api::init(config, size));
    typename Api::Handle out = nullptr;
    // Config is valid; deliberately invalid resources/key type prevent GPU work.
    EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, Api::create(&out, config));
    EXPECT_EQ(nullptr, out);
  }
  EXPECT_EQ(0, munmap(memory, 2 * page_size));
}

TEST(NveCApiSizedDefaults, LibraryDefaults) {
  nve_gpu_table_config_t gpu;
  const nve::GPUTableConfig gpu_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_config_init(&gpu, sizeof(gpu)));
  EXPECT_EQ(sizeof(gpu), ConfigSize(gpu));
  EXPECT_EQ(gpu_defaults.device_id, gpu.device_id);
  EXPECT_EQ(gpu_defaults.cache_size, gpu.cache_size);
  EXPECT_EQ(gpu_defaults.row_size_in_bytes, gpu.row_size_in_bytes);
  EXPECT_EQ(gpu_defaults.uvm_table, gpu.uvm_table);
  EXPECT_EQ(gpu_defaults.uvm_num_rows, gpu.uvm_num_rows);
  EXPECT_EQ(gpu_defaults.count_misses, gpu.count_misses);
  EXPECT_EQ(gpu_defaults.max_modify_size, gpu.max_modify_size);
  EXPECT_EQ(gpu_defaults.value_dtype, nve::convert_external_dtype(gpu.value_dtype));
  EXPECT_EQ(gpu_defaults.private_stream, gpu.private_stream);
  EXPECT_EQ(gpu_defaults.disable_uvm_update, gpu.disable_uvm_update);
  EXPECT_EQ(gpu_defaults.uvm_cpu_accumulate, gpu.uvm_cpu_accumulate);
  EXPECT_EQ(gpu_defaults.data_storage_on_host, gpu.data_storage_on_host);
  EXPECT_EQ(gpu_defaults.modify_on_gpu, gpu.modify_on_gpu);
  EXPECT_EQ(gpu_defaults.kernel_mode_type, gpu.kernel_mode_type);
  EXPECT_EQ(gpu_defaults.kernel_mode_value, gpu.kernel_mode_value);
  EXPECT_EQ(gpu_defaults.invalid_key, gpu.invalid_key);

  nve_linear_host_table_config_t linear;
  const nve::LinearHostTableConfig linear_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_linear_host_table_config_init(&linear, sizeof(linear)));
  EXPECT_EQ(linear_defaults.emb_table, linear.emb_table);
  EXPECT_EQ(linear_defaults.num_rows, linear.num_rows);
  EXPECT_EQ(linear_defaults.max_value_size, linear.row_size_in_bytes);
  EXPECT_EQ(linear_defaults.value_dtype, nve::convert_external_dtype(linear.value_dtype));
  EXPECT_EQ(linear_defaults.max_threads, linear.max_threads);
  EXPECT_EQ(linear_defaults.task_size_threshold, linear.task_size_threshold);

  nve_gpu_embedding_layer_config_t emb;
  const nve::GPUEmbeddingLayerConfig emb_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_embedding_layer_config_init(&emb, sizeof(emb)));
  EXPECT_STREQ(emb_defaults.layer_name.c_str(), emb.layer_name);
  EXPECT_EQ(emb_defaults.device_id, emb.device_id);
  EXPECT_EQ(emb_defaults.embedding_table, emb.embedding_table);
  EXPECT_EQ(emb_defaults.num_embeddings, emb.num_embeddings);
  EXPECT_EQ(emb_defaults.embedding_width_in_bytes, emb.embedding_width_in_bytes);
  EXPECT_EQ(emb_defaults.value_dtype, nve::convert_external_dtype(emb.value_dtype));
  EXPECT_EQ(emb_defaults.default_row_index, emb.default_row_index);

  nve_linear_uvm_layer_config_t uvm;
  const nve::LinearUVMEmbeddingLayer<int64_t>::Config uvm_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_linear_uvm_layer_config_init(&uvm, sizeof(uvm)));
  EXPECT_STREQ(uvm_defaults.layer_name.c_str(), uvm.layer_name);
  EXPECT_EQ(nullptr, uvm.insert_heuristic);
  EXPECT_EQ(uvm_defaults.min_insert_freq_gpu, uvm.min_insert_freq_gpu);
  EXPECT_EQ(uvm_defaults.min_insert_size_gpu, uvm.min_insert_size_gpu);
  EXPECT_EQ(uvm_defaults.default_row_index, uvm.default_row_index);

  nve_hierarchical_layer_config_t hier;
  const nve::HierarchicalEmbeddingLayer<int64_t>::Config hier_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_hierarchical_layer_config_init(&hier, sizeof(hier)));
  EXPECT_STREQ(hier_defaults.layer_name.c_str(), hier.layer_name);
  EXPECT_EQ(nullptr, hier.insert_heuristic);
  EXPECT_EQ(hier_defaults.min_insert_freq_gpu, hier.min_insert_freq_gpu);
  EXPECT_EQ(hier_defaults.min_insert_freq_host, hier.min_insert_freq_host);
  EXPECT_EQ(hier_defaults.min_insert_size_gpu, hier.min_insert_size_gpu);
  EXPECT_EQ(hier_defaults.min_insert_size_host, hier.min_insert_size_host);
  EXPECT_EQ(hier_defaults.default_embedding.empty(), hier.default_embedding == nullptr);
  EXPECT_EQ(static_cast<int64_t>(hier_defaults.default_embedding.size()), hier.default_embedding_size);
  if (!hier_defaults.default_embedding.empty()) {
    ASSERT_NE(nullptr, hier.default_embedding);
    EXPECT_EQ(0, std::memcmp(hier_defaults.default_embedding.data(), hier.default_embedding,
                            hier_defaults.default_embedding.size()));
  }

  nve_host_embedding_layer_config_t host;
  const nve::HostEmbeddingLayer<int64_t>::Config host_defaults{};
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&host, sizeof(host)));
  EXPECT_STREQ(host_defaults.layer_name.c_str(), host.layer_name);
  EXPECT_EQ(host_defaults.default_embedding.empty(), host.default_embedding == nullptr);
  EXPECT_EQ(static_cast<int64_t>(host_defaults.default_embedding.size()), host.default_embedding_size);
  EXPECT_EQ(host_defaults.task_size_threshold, host.task_size_threshold);
  EXPECT_EQ(host_defaults.disable_threadpool, host.disable_threadpool);
  if (!host_defaults.default_embedding.empty()) {
    ASSERT_NE(nullptr, host.default_embedding);
    EXPECT_EQ(0, std::memcmp(host_defaults.default_embedding.data(), host.default_embedding,
                            host_defaults.default_embedding.size()));
  }
}

TEST(NveCApiSizedDefaults, ConfigTypesHaveDistinctTags) {
  nve_gpu_table_config_t gpu;
  nve_linear_host_table_config_t linear;
  nve_gpu_embedding_layer_config_t emb;
  nve_linear_uvm_layer_config_t uvm;
  nve_hierarchical_layer_config_t hier;
  nve_host_embedding_layer_config_t host;
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_config_init(&gpu, sizeof(gpu)));
  ASSERT_EQ(NVE_SUCCESS, nve_linear_host_table_config_init(&linear, sizeof(linear)));
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_embedding_layer_config_init(&emb, sizeof(emb)));
  ASSERT_EQ(NVE_SUCCESS, nve_linear_uvm_layer_config_init(&uvm, sizeof(uvm)));
  ASSERT_EQ(NVE_SUCCESS, nve_hierarchical_layer_config_init(&hier, sizeof(hier)));
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&host, sizeof(host)));
  const std::set<uint64_t> tags = {ConfigTag(gpu), ConfigTag(linear), ConfigTag(emb),
                                   ConfigTag(uvm), ConfigTag(hier), ConfigTag(host)};
  EXPECT_EQ(6u, tags.size());
  EXPECT_EQ(0u, tags.count(0));

  // A config carrying another type's tag is rejected even when its size is valid.
  host.header = (gpu.header & ~nve_config_size_mask) | ConfigSize(host);
  nve_layer_t layer = reinterpret_cast<nve_layer_t>(uintptr_t{1});
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_host_embedding_layer_create(&layer, NVE_KEY_INT64, &host, nullptr, nullptr));
  EXPECT_EQ(nullptr, layer);
}

TEST(NveCApiSizedDefaults, HostKnobSizesMustEndBetweenFields) {
  nve_host_embedding_layer_config_t cfg;
  for (const auto offset : {offsetof(nve_host_embedding_layer_config_t, task_size_threshold),
                            offsetof(nve_host_embedding_layer_config_t, disable_threadpool)}) {
    ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&cfg, sizeof(cfg)));
    std::array<unsigned char, sizeof(cfg)> before;
    std::memcpy(before.data(), &cfg, sizeof(cfg));
    EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_host_embedding_layer_config_init(&cfg, offset + 1));
    EXPECT_EQ(0, std::memcmp(before.data(), &cfg, sizeof(cfg)));
    SetConfigSize(cfg, offset + 1);
    nve_layer_t layer = reinterpret_cast<nve_layer_t>(uintptr_t{1});
    EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
              nve_host_embedding_layer_create(&layer, NVE_KEY_INT64, &cfg, nullptr, nullptr));
    EXPECT_EQ(nullptr, layer);
    EXPECT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&cfg, offset));
  }
}

class NveCApiSizedLayer : public ::testing::TestWithParam<nve_key_type_t> {
 protected:
  void SetUp() override {
    ASSERT_EQ(NVE_SUCCESS, nve_create_table_factory(&factory_, "libnve-plugin-stl-map.so", "{}"));
    const char* json = GetParam() == NVE_KEY_INT32
        ? R"({"key_size":4,"max_value_size":4,"value_dtype":"float32"})"
        : R"({"key_size":8,"max_value_size":4,"value_dtype":"float32"})";
    ASSERT_EQ(NVE_SUCCESS, nve_table_factory_produce(factory_, 0, json, &table_));
  }
  void TearDown() override {
    if (ctx_) nve_context_destroy(ctx_);
    if (layer_) nve_layer_destroy(layer_);
    if (pool_) nve_thread_pool_destroy(pool_);
    if (table_) nve_table_destroy(table_);
    if (factory_) nve_table_factory_destroy(factory_);
  }
  void CheckLookup() {
    ASSERT_EQ(NVE_SUCCESS, nve_layer_create_execution_context(
        layer_, &ctx_, nullptr, nullptr, pool_, nullptr));
    const int32_t keys32[] = {3, 7};
    const int64_t keys64[] = {3, 7};
    const void* keys = GetParam() == NVE_KEY_INT32
        ? static_cast<const void*>(keys32) : static_cast<const void*>(keys64);
    const float inserted = 12.5f;
    ASSERT_EQ(NVE_SUCCESS, nve_layer_insert(layer_, ctx_, 1, keys, 4, 4, &inserted, 0));
    ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));
    float output[] = {0.f, 0.f};
    uint64_t hitmask = 0;
    ASSERT_EQ(NVE_SUCCESS, nve_layer_lookup(layer_, ctx_, 2, keys, output, 4, &hitmask, nullptr));
    ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));
    EXPECT_FLOAT_EQ(inserted, output[0]);
    EXPECT_FLOAT_EQ(7.5f, output[1]);
    EXPECT_EQ(1u, hitmask & 3u);
  }
  void CheckHostConfig(int64_t threshold, bool disable_threadpool) {
    const auto check = [&](const auto& layer) {
      ASSERT_NE(nullptr, layer);
      EXPECT_EQ(threshold, layer->get_config().task_size_threshold);
      EXPECT_EQ(disable_threadpool, layer->get_config().disable_threadpool);
    };
    if (GetParam() == NVE_KEY_INT32) {
      check(std::dynamic_pointer_cast<nve::HostEmbeddingLayer<int32_t>>(layer_->ptr));
    } else {
      check(std::dynamic_pointer_cast<nve::HostEmbeddingLayer<int64_t>>(layer_->ptr));
    }
  }
  nve_table_factory_t factory_ = nullptr;
  nve_table_t table_ = nullptr;
  nve_layer_t layer_ = nullptr;
  nve_context_t ctx_ = nullptr;
  nve_thread_pool_t pool_ = nullptr;
};

TEST_P(NveCApiSizedLayer, HostLookupAndDefaultEmbedding) {
  nve_host_embedding_layer_config_t cfg;
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&cfg, sizeof(cfg)));
  const float default_embedding = 7.5f;
  cfg.layer_name = "sized_host";
  cfg.default_embedding = &default_embedding;
  cfg.default_embedding_size = sizeof(default_embedding);
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_create(&layer_, GetParam(), &cfg, table_, nullptr));
  CheckLookup();
}

TEST_P(NveCApiSizedLayer, HostKnobsControlThreadPool) {
  nve_host_embedding_layer_config_t cfg;
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&cfg, sizeof(cfg)));
  const float default_embedding = 7.5f;
  cfg.default_embedding = &default_embedding;
  cfg.default_embedding_size = sizeof(default_embedding);
  cfg.task_size_threshold = 17;
  cfg.disable_threadpool = 2;  // Any nonzero value enables the boolean knob.
  ASSERT_EQ(NVE_SUCCESS, nve_thread_pool_create(&pool_, R"({"name":"simple","num_workers":2})"));
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_create(&layer_, GetParam(), &cfg, table_, nullptr));
  CheckHostConfig(cfg.task_size_threshold, true);
  CheckLookup();
  ASSERT_NE(nullptr, ctx_);
  EXPECT_EQ(nve::InlineThreadPool::instance(), ctx_->ptr->get_thread_pool());
}

TEST_P(NveCApiSizedLayer, OriginalHostConfigUsesCppKnobDefaults) {
  nve_host_embedding_layer_config_t cfg;
  const size_t original_size = offsetof(nve_host_embedding_layer_config_t, task_size_threshold);
  std::memset(&cfg, 0xa5, sizeof(cfg));
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&cfg, original_size));
  const float default_embedding = 7.5f;
  cfg.default_embedding = &default_embedding;
  cfg.default_embedding_size = sizeof(default_embedding);
  ASSERT_EQ(NVE_SUCCESS, nve_thread_pool_create(&pool_, R"({"name":"simple","num_workers":2})"));
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_create(&layer_, GetParam(), &cfg, table_, nullptr));
  const nve::HostEmbeddingLayer<int64_t>::Config defaults{};
  CheckHostConfig(defaults.task_size_threshold, defaults.disable_threadpool);
  CheckLookup();
  ASSERT_NE(nullptr, ctx_);
  EXPECT_EQ(pool_->ptr, ctx_->ptr->get_thread_pool());
}

TEST_P(NveCApiSizedLayer, HierarchicalLookupAndDefaultEmbedding) {
  nve_hierarchical_layer_config_t cfg;
  ASSERT_EQ(NVE_SUCCESS, nve_hierarchical_layer_config_init(&cfg, sizeof(cfg)));
  const float default_embedding = 7.5f;
  cfg.layer_name = "sized_hierarchical";
  cfg.min_insert_size_gpu = 17;
  cfg.min_insert_size_host = 19;
  cfg.default_embedding = &default_embedding;
  cfg.default_embedding_size = sizeof(default_embedding);
  ASSERT_EQ(NVE_SUCCESS, nve_heuristic_create_never(&cfg.insert_heuristic));
  const auto status = nve_hierarchical_layer_create(&layer_, GetParam(), &cfg, &table_, 1, nullptr);
  EXPECT_EQ(NVE_SUCCESS, nve_heuristic_destroy(cfg.insert_heuristic));
  ASSERT_EQ(NVE_SUCCESS, status);
  CheckLookup();
}

TEST_P(NveCApiSizedLayer, MissingDefaultEmbeddingSizeDisablesDefault) {
  nve_host_embedding_layer_config_t cfg;
  const size_t prefix = offsetof(nve_host_embedding_layer_config_t, default_embedding_size);
  std::memset(&cfg, 0xa5, sizeof(cfg));
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_config_init(&cfg, prefix));
  cfg.layer_name = "short_host";
  // An absent size must default to zero, so creation must not dereference this pointer.
  cfg.default_embedding = reinterpret_cast<const void*>(uintptr_t{1});
  ASSERT_EQ(NVE_SUCCESS, nve_host_embedding_layer_create(&layer_, GetParam(), &cfg, table_, nullptr));
  int64_t num_tables = 0;
  EXPECT_EQ(NVE_SUCCESS, nve_layer_get_num_tables(layer_, &num_tables));
  EXPECT_EQ(1, num_tables);
}

INSTANTIATE_TEST_SUITE_P(KeyTypes, NveCApiSizedLayer,
                        ::testing::Values(NVE_KEY_INT32, NVE_KEY_INT64));

TEST(NveCApiAbi, FrozenAbi2_0Client) {
  EXPECT_EQ(0, nve_frozen_abi_2_0_client());
}

}  // namespace
