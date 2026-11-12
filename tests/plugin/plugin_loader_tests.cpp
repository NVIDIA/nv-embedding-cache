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

#include <algorithm>
#include <bit_ops.hpp>
#include <buffer_wrapper.hpp>
#include <cstdint>
#include <cstring>
#include <json_support.hpp>
#include <memory>
#include <nve_c_api.h>
#include <plugin/external_plugin_table.hpp>
#include <plugin/plugin_loader.hpp>
#include <vector>

using namespace nve;

namespace {

constexpr int64_t kRowBytes = 32;
constexpr int64_t kRowFloats = kRowBytes / static_cast<int64_t>(sizeof(float));

nlohmann::json host_table_config() {
  return {{"key_size", 8}, {"max_value_size", kRowBytes}, {"value_dtype", "float32"}};
}

Table::buffer_ptr<const void> wrap_const(context_ptr_t& ctx, const std::string& name,
                                         const void* ptr, size_t bytes) {
  return std::make_shared<BufferWrapper<const void>>(ctx, name, ptr, bytes);
}

}  // namespace

/* ============================================================================
 * Internal mode (stl_map plugin)
 * ============================================================================ */

TEST(plugin_loader, rejects_missing_so) {
  // A relative name is retried beside nve-common; the error reports both attempts.
  try {
    Plugin plugin{"libnve-plugin-does-not-exist.so"};
    FAIL() << "expected load failure";
  } catch (const std::exception& e) {
    EXPECT_NE(nullptr, std::strstr(e.what(), "fallback")) << e.what();
  }
  // An absolute path has no fallback location; the error reports one attempt.
  try {
    Plugin plugin{"/nonexistent-dir/libnve-plugin-does-not-exist.so"};
    FAIL() << "expected load failure";
  } catch (const std::exception& e) {
    EXPECT_EQ(nullptr, std::strstr(e.what(), "fallback")) << e.what();
  }
}

TEST(plugin_loader, internal_metadata) {
  Plugin plugin(NVE_STL_MAP_PLUGIN_PATH);
  EXPECT_EQ(plugin.mode(), PluginMode::Internal);
  EXPECT_EQ(plugin.name(), "STL unordered_map hashtable plugin");
  EXPECT_EQ(plugin.author(), "NVIDIA Corporation");
  EXPECT_EQ(plugin.abi_version().first, NVE_PLUGIN_ABI_VERSION_MAJOR);
  EXPECT_LE(plugin.abi_version().second, NVE_PLUGIN_ABI_VERSION_MINOR);
}

// The factory and its tables stay valid after the stack Plugin (and even the
// factory, for tables) are gone — the library lease keeps the SO mapped.
TEST(plugin_loader, internal_lease_outlives_plugin_and_factory) {
  table_ptr_t table;
  {
    table_factory_ptr_t factory;
    {
      Plugin plugin(NVE_STL_MAP_PLUGIN_PATH);
      factory = plugin.create_table_factory(nlohmann::json::object());
    }
    table = factory->produce(/*id=*/1, host_table_config());
  }
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->get_device_id(), -1);
  EXPECT_EQ(table->get_key_size(), 8);

  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);
  std::vector<int64_t> keys{1, 2, 3, 4};
  std::vector<float> values(keys.size() * kRowFloats, 7.0f);
  table->insert(ctx, static_cast<int64_t>(keys.size()),
                wrap_const(ctx, "keys", keys.data(), keys.size() * sizeof(int64_t)), kRowBytes,
                kRowBytes, wrap_const(ctx, "vals", values.data(), values.size() * sizeof(float)));

  std::vector<float> out(values.size(), -1.0f);
  std::vector<uint64_t> mask(1, 0);
  auto out_buf = std::make_shared<BufferWrapper<void>>(ctx, "out", out.data(),
                                                       out.size() * sizeof(float));
  auto mask_buf = std::make_shared<BufferWrapper<bitmask64_t>>(ctx, "mask", mask.data(),
                                                                      sizeof(uint64_t));
  table->find(ctx, static_cast<int64_t>(keys.size()),
              wrap_const(ctx, "keys2", keys.data(), keys.size() * sizeof(int64_t)), mask_buf,
              kRowBytes, out_buf, nullptr);
  EXPECT_EQ(mask[0], 0xfu);
  EXPECT_EQ(out[0], 7.0f);
}

/* ============================================================================
 * External mode (sample pure-C map plugin)
 * ============================================================================ */

namespace {

table_ptr_t make_external_table() {
  Plugin plugin(NVE_EXTERNAL_MAP_PLUGIN_PATH);
  EXPECT_EQ(plugin.mode(), PluginMode::External);
  EXPECT_EQ(plugin.name(), "Sample External Map");
  auto factory = plugin.create_table_factory({{"num_buckets", 128}});
  return factory->produce(/*id=*/7, {{"max_value_size", kRowBytes}});
}

}  // namespace

TEST(plugin_loader, external_capabilities) {
  auto table = make_external_table();
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->get_device_id(), -1);
  EXPECT_EQ(table->get_key_size(), 8);
  EXPECT_EQ(table->get_max_row_size(), kRowBytes);
  EXPECT_EQ(table->get_invalid_key(), -1);
  EXPECT_EQ(table->get_value_type(), DataType_t::Float32);
  EXPECT_EQ(dtype_size(table->get_value_type()), 4);  // external FLOAT32 keeps its size
  EXPECT_TRUE(table->lookup_counter_hits());
}

TEST(plugin_loader, external_data_ops_roundtrip) {
  auto table = make_external_table();
  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);

  constexpr int64_t n = 200;  // large enough to take the execute_n path
  std::vector<int64_t> keys(n);
  std::vector<float> values(n * kRowFloats);
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = 1000 + static_cast<int64_t>(i);
    for (size_t j = 0; j < static_cast<size_t>(kRowFloats); ++j) {
      values[i * static_cast<size_t>(kRowFloats) + j] = static_cast<float>(i);
    }
  }
  auto keys_buf = wrap_const(ctx, "keys", keys.data(), n * sizeof(int64_t));

  table->reset_lookup_counter(ctx);
  table->insert(ctx, n, keys_buf, kRowBytes, kRowBytes,
                wrap_const(ctx, "vals", values.data(), values.size() * sizeof(float)));

  std::vector<float> out(values.size(), -1.0f);
  std::vector<uint64_t> mask(nve::ceil_div(n, static_cast<int64_t>(64)), 0);
  std::vector<int64_t> sizes(n, 0);
  auto out_buf = std::make_shared<BufferWrapper<void>>(ctx, "out", out.data(),
                                                       out.size() * sizeof(float));
  auto mask_buf = std::make_shared<BufferWrapper<bitmask64_t>>(
      ctx, "mask", mask.data(), mask.size() * sizeof(uint64_t));
  auto sizes_buf = std::make_shared<BufferWrapper<int64_t>>(ctx, "sizes", sizes.data(),
                                                            sizes.size() * sizeof(int64_t));
  table->find(ctx, n, keys_buf, mask_buf, kRowBytes, out_buf, sizes_buf);
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_TRUE((mask[i / 64] >> (i % 64)) & 1u) << i;
    ASSERT_EQ(out[i * static_cast<size_t>(kRowFloats)], static_cast<float>(i)) << i;
    ASSERT_EQ(sizes[i], kRowBytes) << i;
  }
  int64_t counter = 0;
  table->get_lookup_counter(ctx, &counter);
  EXPECT_EQ(counter, n);

  // Count-only find: null values is an exists check.
  std::vector<int64_t> misses{1, 2, 3};
  std::vector<uint64_t> miss_mask(1, 0);
  auto miss_mask_buf = std::make_shared<BufferWrapper<bitmask64_t>>(
      ctx, "miss_mask", miss_mask.data(), sizeof(uint64_t));
  table->find(ctx, static_cast<int64_t>(misses.size()),
              wrap_const(ctx, "miss_keys", misses.data(), misses.size() * sizeof(int64_t)),
              miss_mask_buf, 0, nullptr, nullptr);
  EXPECT_EQ(miss_mask[0], 0u);

  // update overwrites, update_accumulate adds float32.
  for (auto& v : values) v = 100.0f;
  table->update(ctx, n, keys_buf, kRowBytes, kRowBytes,
                wrap_const(ctx, "upd", values.data(), values.size() * sizeof(float)));
  std::vector<float> deltas(values.size(), 1.5f);
  table->update_accumulate(ctx, n, keys_buf, kRowBytes, kRowBytes,
                           wrap_const(ctx, "acc", deltas.data(), deltas.size() * sizeof(float)),
                           DataType_t::Float32);
  std::fill(mask.begin(), mask.end(), 0);
  table->find(ctx, n, keys_buf, mask_buf, kRowBytes, out_buf, nullptr);
  EXPECT_EQ(out[0], 101.5f);

  // size + get_keys (ABI 3.0): the full set, then a clipped read.
  EXPECT_EQ(table->size(ctx), n);
  std::vector<int64_t> stored(n + 1, -1);
  EXPECT_EQ(table->get_keys(ctx, static_cast<int64_t>(stored.size()), stored.data()), n);
  std::vector<int64_t> sorted(stored.begin(), stored.begin() + n);
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted, keys);
  EXPECT_EQ(stored.back(), -1);
  EXPECT_EQ(table->get_keys(ctx, 5, stored.data()), 5);
  EXPECT_THROW(table->get_keys(ctx, -1, stored.data()), nve::Exception);

  // erase + clear.
  table->erase(ctx, 1, wrap_const(ctx, "erase", keys.data(), sizeof(int64_t)));
  std::fill(mask.begin(), mask.end(), 0);
  table->find(ctx, 1, wrap_const(ctx, "erased", keys.data(), sizeof(int64_t)), mask_buf, 0,
              nullptr, nullptr);
  EXPECT_EQ(mask[0] & 1u, 0u);
  table->clear(ctx);
}

// Failure statuses round-trip through the adapter as ExternalPluginError with
// the plugin's exact category; unknown dtypes for accumulate surface
// NOT_IMPLEMENTED from the plugin.
TEST(plugin_loader, external_status_fidelity) {
  Plugin plugin(NVE_EXTERNAL_MAP_PLUGIN_PATH);
  auto factory = plugin.create_table_factory(nlohmann::json::object());

  try {
    factory->produce(/*id=*/8, {{"max_value_size", -1}});
    FAIL() << "expected ExternalPluginError";
  } catch (const ExternalPluginError& e) {
    EXPECT_EQ(e.status(), NVE_ERROR_INVALID_ARGUMENT);
  }

  auto table = factory->produce(/*id=*/9, {{"max_value_size", kRowBytes}});
  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);
  std::vector<int64_t> keys{1};
  std::vector<double> update(kRowBytes / sizeof(double), 1.0);
  try {
    table->update_accumulate(ctx, 1, wrap_const(ctx, "k", keys.data(), sizeof(int64_t)),
                             kRowBytes, kRowBytes,
                             wrap_const(ctx, "u", update.data(), kRowBytes),
                             DataType_t::Float64);
    FAIL() << "expected ExternalPluginError";
  } catch (const ExternalPluginError& e) {
    EXPECT_EQ(e.status(), NVE_ERROR_NOT_IMPLEMENTED);
  }
}

// Unknown numeric statuses from external code sanitize to NVE_ERROR_RUNTIME.
TEST(plugin_loader, sanitize_unknown_status) {
  EXPECT_EQ(sanitize_external_status(static_cast<nve_status_t>(1234)), NVE_ERROR_RUNTIME);
  EXPECT_EQ(sanitize_external_status(NVE_ERROR_OUT_OF_MEMORY), NVE_ERROR_OUT_OF_MEMORY);
  EXPECT_EQ(sanitize_external_status(NVE_SUCCESS), NVE_SUCCESS);
}

/* ============================================================================
 * Dtype conversion fidelity
 * ============================================================================ */

TEST(plugin_loader, dtype_conversion_roundtrip) {
  const DataType_t all_dtypes[] = {
      DataType_t::Unknown,          DataType_t::Float32,          DataType_t::BFloat,
      DataType_t::Float16,          DataType_t::E4M3,             DataType_t::E5M2,
      DataType_t::Float64,          DataType_t::QInt8RowwiseF32,  DataType_t::QInt8RowwiseF16,
      DataType_t::QUint8RowwiseF32, DataType_t::QUint8RowwiseF16,
  };
  for (DataType_t dtype : all_dtypes) {
    EXPECT_EQ(convert_external_dtype(convert_external_dtype(dtype)), dtype);
  }
  // External FLOAT32 must map to a DataType_t whose size survives (a packed
  // DataType_t truncated through a 32-bit field would report size 0).
  EXPECT_EQ(convert_external_dtype(NVE_DTYPE_FLOAT32), DataType_t::Float32);
  EXPECT_EQ(dtype_size(convert_external_dtype(NVE_DTYPE_FLOAT32)), 4);
  // Out-of-range IDs map to Unknown, never throw.
  EXPECT_EQ(convert_external_dtype(static_cast<nve_data_type_t>(9999)), DataType_t::Unknown);
  EXPECT_EQ(convert_external_dtype(static_cast<nve_data_type_t>(-1)), DataType_t::Unknown);
}

/* ============================================================================
 * Fault-injection plugin: status fidelity and worker-task bridging
 * ============================================================================ */

namespace {

constexpr nve_status_t kFailureStatuses[] = {
    NVE_ERROR_INVALID_ARGUMENT, NVE_ERROR_CUDA, NVE_ERROR_RUNTIME,
    NVE_ERROR_NOT_IMPLEMENTED,  NVE_ERROR_OUT_OF_MEMORY,
};

std::string inject(const char* key, int32_t value) {
  return nlohmann::json{{key, value}}.dump();
}

// Own the C handles so a failing ASSERT cannot leak them (and the plugin
// library lease they hold). Tests that assert on the destroy status call
// nve_*_destroy(handle.release()) instead of letting the deleter run.
using FactoryPtr = std::unique_ptr<nve_table_factory_s, decltype(&nve_table_factory_destroy)>;
using TablePtr   = std::unique_ptr<nve_table_s, decltype(&nve_table_destroy)>;
using ContextPtr = std::unique_ptr<nve_context_s, decltype(&nve_context_destroy)>;

void expect_last_error_contains(const char* needle) {
  const char* message = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_get_last_error(&message));
  ASSERT_NE(nullptr, message);
  EXPECT_NE(nullptr, std::strstr(message, needle)) << message;
}

}  // namespace

// Factory-creation failures round-trip status and message through the C API,
// and the output handle is pre-nulled on entry.
TEST(plugin_loader, c_api_factory_creation_status_fidelity) {
  for (nve_status_t status : kFailureStatuses) {
    auto factory = reinterpret_cast<nve_table_factory_t>(0x1);
    EXPECT_EQ(status, nve_create_table_factory(&factory, NVE_FAULT_PLUGIN_PATH,
                                               inject("factory_status", status).c_str()));
    EXPECT_EQ(nullptr, factory);
    expect_last_error_contains("injected factory-creation failure");
  }
  // Unknown numeric statuses sanitize to RUNTIME.
  nve_table_factory_t factory = nullptr;
  EXPECT_EQ(NVE_ERROR_RUNTIME,
            nve_create_table_factory(&factory, NVE_FAULT_PLUGIN_PATH,
                                     inject("factory_status", 999).c_str()));
  EXPECT_EQ(nullptr, factory);
}

// Produce failures round-trip status and message through the C API.
TEST(plugin_loader, c_api_produce_status_fidelity) {
  nve_table_factory_t raw_factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_create_table_factory(&raw_factory, NVE_FAULT_PLUGIN_PATH, "{}"));
  FactoryPtr factory(raw_factory, nve_table_factory_destroy);
  for (nve_status_t status : kFailureStatuses) {
    auto table = reinterpret_cast<nve_table_t>(0x1);
    EXPECT_EQ(status, nve_table_factory_produce(factory.get(), /*table_id=*/1,
                                                inject("produce_status", status).c_str(), &table));
    EXPECT_EQ(nullptr, table);
    expect_last_error_contains("injected produce failure");
  }
  nve_table_t table = nullptr;
  EXPECT_EQ(NVE_ERROR_RUNTIME,
            nve_table_factory_produce(factory.get(), /*table_id=*/2,
                                      inject("produce_status", 999).c_str(), &table));
  EXPECT_EQ(nullptr, table);
  // Non-object produce configs are rejected as invalid arguments.
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_factory_produce(factory.get(), /*table_id=*/3, "5", &table));
  EXPECT_EQ(nullptr, table);
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(factory.release()));
}

// Table-operation failures round-trip status and message through the C API.
TEST(plugin_loader, c_api_table_op_status_fidelity) {
  nve_table_factory_t raw_factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_create_table_factory(&raw_factory, NVE_FAULT_PLUGIN_PATH, "{}"));
  FactoryPtr factory(raw_factory, nve_table_factory_destroy);
  int32_t injected[] = {NVE_ERROR_INVALID_ARGUMENT, NVE_ERROR_CUDA, NVE_ERROR_RUNTIME,
                        NVE_ERROR_NOT_IMPLEMENTED,  NVE_ERROR_OUT_OF_MEMORY, 999};
  for (int32_t status : injected) {
    nve_table_t raw_table = nullptr;
    ASSERT_EQ(NVE_SUCCESS,
              nve_table_factory_produce(factory.get(), /*table_id=*/1,
                                        inject("operation_status", status).c_str(), &raw_table));
    ASSERT_NE(nullptr, raw_table);
    TablePtr table(raw_table, nve_table_destroy);
    nve_context_t raw_ctx = nullptr;
    ASSERT_EQ(NVE_SUCCESS, nve_table_create_execution_context(table.get(), &raw_ctx, nullptr,
                                                              nullptr, nullptr, nullptr));
    ContextPtr ctx(raw_ctx, nve_context_destroy);
    const nve_status_t expected =
        sanitize_external_status(static_cast<nve_status_t>(status));
    EXPECT_EQ(expected, nve_table_clear(table.get(), ctx.get()));
    expect_last_error_contains("injected operation failure");
    int64_t out = 0;
    EXPECT_EQ(expected, nve_table_size(table.get(), ctx.get(), &out));
    expect_last_error_contains("injected operation failure");
    EXPECT_EQ(expected, nve_table_get_keys(table.get(), ctx.get(), 1, &out, &out));
    expect_last_error_contains("injected operation failure");
    EXPECT_EQ(NVE_SUCCESS, nve_context_destroy(ctx.release()));
    EXPECT_EQ(NVE_SUCCESS, nve_table_destroy(table.release()));
  }
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(factory.release()));
}

// size/get_keys are mandatory: a table that leaves them NULL is rejected at produce time.
TEST(plugin_loader, c_api_rejects_table_without_key_ops) {
  nve_table_factory_t raw_factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_create_table_factory(&raw_factory, NVE_FAULT_PLUGIN_PATH, "{}"));
  FactoryPtr factory(raw_factory, nve_table_factory_destroy);
  nve_table_t raw_table = nullptr;
  EXPECT_EQ(NVE_ERROR_RUNTIME, nve_table_factory_produce(factory.get(), /*table_id=*/3,
                                                         inject("omit_key_ops", 1).c_str(), &raw_table));
  EXPECT_EQ(nullptr, raw_table);
  expect_last_error_contains("missing mandatory callbacks");
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(factory.release()));
}

// The produced table's key size is authoritative for the C handle key type;
// conflicting plugin JSON is not consulted, and unsupported sizes are rejected
// before a handle is published.
TEST(plugin_loader, c_api_key_type_from_produced_table) {
  nve_table_factory_t raw_factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&raw_factory, NVE_EXTERNAL_MAP_PLUGIN_PATH, "{}"));
  FactoryPtr factory(raw_factory, nve_table_factory_destroy);
  nve_table_t raw_table = nullptr;
  // The sample map ignores "key_size" and always produces 8-byte keys.
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_factory_produce(factory.get(), /*table_id=*/1,
                                      R"({"max_value_size": 32, "key_size": 4})", &raw_table));
  TablePtr table(raw_table, nve_table_destroy);
  int64_t key_size = 0;
  EXPECT_EQ(NVE_SUCCESS, nve_table_get_key_size(table.get(), &key_size));
  EXPECT_EQ(8, key_size);
  EXPECT_EQ(NVE_SUCCESS, nve_table_destroy(table.release()));
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(factory.release()));

  nve_table_factory_t raw_fault = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_create_table_factory(&raw_fault, NVE_FAULT_PLUGIN_PATH, "{}"));
  FactoryPtr fault(raw_fault, nve_table_factory_destroy);
  auto bad = reinterpret_cast<nve_table_t>(0x1);
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_factory_produce(fault.get(), /*table_id=*/2,
                                      R"({"report_key_size": 3})", &bad));
  EXPECT_EQ(nullptr, bad);
  EXPECT_EQ(NVE_SUCCESS, nve_table_factory_destroy(fault.release()));
}

// A table reporting an out-of-range value dtype is rejected during adoption.
TEST(plugin_loader, external_unknown_dtype_rejected_at_adoption) {
  Plugin plugin(NVE_FAULT_PLUGIN_PATH);
  auto factory = plugin.create_table_factory(nlohmann::json::object());
  try {
    factory->produce(/*id=*/1, {{"value_dtype", 9999}});
    FAIL() << "expected ExternalPluginError";
  } catch (const ExternalPluginError& e) {
    EXPECT_EQ(e.status(), NVE_ERROR_RUNTIME);
  }
}

// Worker tasks run through the workgroup-aware execute_n_wg host service; a
// failing task's status is folded at the barrier, sanitized, and surfaced with
// the generic barrier message on the calling thread.
TEST(plugin_loader, external_worker_task_status_through_barrier) {
  Plugin plugin(NVE_FAULT_PLUGIN_PATH);
  auto factory = plugin.create_table_factory(nlohmann::json::object());

  // Success path: each execute_n_wg task marks its private hit-mask word.
  auto table = factory->produce(/*id=*/1, nlohmann::json::object());
  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);
  constexpr int64_t n = 130;  // three mask words -> three tasks
  std::vector<int64_t> keys(n, 5);
  std::vector<uint64_t> mask(nve::ceil_div(n, static_cast<int64_t>(64)), 0);
  auto keys_buf = wrap_const(ctx, "keys", keys.data(), n * sizeof(int64_t));
  auto mask_buf = std::make_shared<BufferWrapper<bitmask64_t>>(
      ctx, "mask", mask.data(), mask.size() * sizeof(uint64_t));
  table->find(ctx, n, keys_buf, mask_buf, 0, nullptr, nullptr);
  EXPECT_EQ(mask[0], ~0ull);
  EXPECT_EQ(mask[1], ~0ull);
  EXPECT_EQ(mask[2], (1ull << (n - 128)) - 1u);

  // Failing tasks keep their category through the barrier.
  auto cuda_table = factory->produce(/*id=*/2, {{"task_status", NVE_ERROR_CUDA}});
  auto cuda_ctx = cuda_table->create_execution_context(0, 0, nullptr, nullptr);
  auto cuda_keys = wrap_const(cuda_ctx, "keys", keys.data(), n * sizeof(int64_t));
  try {
    cuda_table->find(cuda_ctx, n, cuda_keys, nullptr, 0, nullptr, nullptr);
    FAIL() << "expected ExternalPluginError";
  } catch (const ExternalPluginError& e) {
    EXPECT_EQ(e.status(), NVE_ERROR_CUDA);
    EXPECT_NE(nullptr, std::strstr(e.what(), "task returned an error")) << e.what();
  }

  // Unknown numeric task statuses become RUNTIME at the barrier.
  auto odd_table = factory->produce(/*id=*/3, {{"task_status", 999}});
  auto odd_ctx = odd_table->create_execution_context(0, 0, nullptr, nullptr);
  auto odd_keys = wrap_const(odd_ctx, "keys", keys.data(), n * sizeof(int64_t));
  try {
    odd_table->find(odd_ctx, n, odd_keys, nullptr, 0, nullptr, nullptr);
    FAIL() << "expected ExternalPluginError";
  } catch (const ExternalPluginError& e) {
    EXPECT_EQ(e.status(), NVE_ERROR_RUNTIME);
  }
}
