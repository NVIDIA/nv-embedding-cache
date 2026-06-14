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

#include <buffer_wrapper.hpp>
#include <execution_context.hpp>
#include <host_table.hpp>
#include <json_support.hpp>
#include <numeric>
#include <redis_cluster_table.hpp>
#include <vector>

using namespace nve;
using namespace nve::plugin;
using namespace nlohmann::literals;

TEST(redis_plugin, parse_table_conf) {
  const RedisClusterTableConfig conf{{4, sizeof(int32_t), 16, DataType_t::BFloat},
                                     1024,
                                     2,
                                     Partitioner_t::AlwaysZero,
                                     {1},
                                     "whatever",
                                     7,
                                     {1024, OverflowHandler_t::EvictLRU, 0.5}};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"(
    {
      "mask_size": 4,
      "key_size": 4,
      "max_value_size": 16,
      "value_dtype": "bfloat",

      "max_batch_size": 1024,

      "num_partitions": 2,
      "partitioner": "always_zero",
      "workgroups": [1],
      "hash_key": "whatever",
      "string_namespace_id": 7,

      "overflow_policy": {
        "overflow_margin": 1024,
        "handler": "evict_lru",
        "resolution_margin": 0.5
      }
    }
  )"_json);
  ASSERT_TRUE(is_json_subset(json, static_cast<nlohmann::json>(conf)));

  RedisClusterTableConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}

TEST(redis_plugin, parse_factory_conf) {
  const RedisClusterTableFactoryConfig conf{
      {},    "localhost:7000", "admin",       "12345",      true /*single_node*/, false,   3,
      true,  "my_bundle.crt",  "my_cert.pem", "my_key.pem", "my_sni"};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"({
    "address": "localhost:7000",
    "user_name": "admin",
    "password": "12345",
    "single_node": true,

    "keep_alive": false,
    "connections_per_node": 3,

    "use_tls": true,
    "ca_certificate": "my_bundle.crt",
    "client_certificate": "my_cert.pem",
    "client_key": "my_key.pem",
    "server_name_identification": "my_sni"
  })"_json);
  ASSERT_TRUE(is_json_subset(json, static_cast<nlohmann::json>(conf)));

  RedisClusterTableFactoryConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}

// End-to-end CRUD against a live standalone Redis in string mode (`num_partitions == 0`). Uses a
// `string_namespace_id` so the test namespaces its own keys (and `clear()` uses SCAN+DEL rather than wiping
// the whole DB). Fails when no Redis server is reachable.
TEST(redis_plugin, single_node_string_crud) {
  using key_type = int64_t;

  RedisClusterTableFactoryConfig fac_conf;
  fac_conf.address = "localhost:6379";
  fac_conf.single_node = true;

  RedisClusterTableFactory fac{fac_conf};

  // String mode: unpartitioned, no shared hash key, namespaced by `string_namespace_id`.
  const int64_t max_value_size{16};
  nlohmann::json table_conf{{"mask_size", sizeof(max_bitmask_repr_t)},
                            {"key_size", sizeof(key_type)},
                            {"max_value_size", max_value_size},
                            {"value_dtype", "float32"},
                            {"num_partitions", 0},
                            {"string_namespace_id", 987654321}};
  host_table_ptr_t tab{fac.produce(7, table_conf)};
  auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);

  // Probe connectivity; redis-plus-plus connects lazily, so the first command reveals an outage.
  try {
    tab->clear(ctx);
  } catch (const std::exception& e) {
    GTEST_FAIL() << "No standalone Redis reachable at localhost:6379: " << e.what();
  }

  const int64_t n{2000};
  const int64_t value_stride{max_value_size};

  // Unique keys so every lookup resolves to a distinct entry.
  std::vector<key_type> keys(static_cast<size_t>(n));
  std::iota(keys.begin(), keys.end(), key_type{1});

  // Deterministic per-key value payloads.
  std::vector<char> values(static_cast<size_t>(n * value_stride));
  for (int64_t i{}; i != n; ++i) {
    for (int64_t j{}; j != max_value_size; ++j) {
      values[static_cast<size_t>(i * value_stride + j)] = static_cast<char>((i * 31 + j) & 0xff);
    }
  }

  std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)));
  std::vector<char> out_values(static_cast<size_t>(n * value_stride));
  std::vector<int64_t> out_value_sizes(static_cast<size_t>(n));

  const size_t keys_bytes{static_cast<size_t>(n) * sizeof(key_type)};
  const size_t hit_mask_bytes{static_cast<size_t>(max_bitmask_t::mask_size(n)) *
                              sizeof(max_bitmask_repr_t)};

  auto keys_bw = [&](const key_type* p) {
    return std::make_shared<BufferWrapper<const void>>(ctx, "keys", p, keys_bytes);
  };
  auto hit_mask_bw = [&]() {
    return std::make_shared<BufferWrapper<max_bitmask_repr_t>>(ctx, "hit_mask", hit_mask.data(),
                                                               hit_mask_bytes);
  };
  auto values_in_bw = [&]() {
    return std::make_shared<BufferWrapper<const void>>(ctx, "values_in", values.data(),
                                                       values.size());
  };
  auto values_out_bw = [&]() {
    return std::make_shared<BufferWrapper<void>>(ctx, "values_out", out_values.data(),
                                                 out_values.size());
  };
  auto value_sizes_bw = [&]() {
    return std::make_shared<BufferWrapper<int64_t>>(ctx, "value_sizes", out_value_sizes.data(),
                                                    out_value_sizes.size() * sizeof(int64_t));
  };

  int64_t cnt{};

  // Empty table -> no hits.
  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  tab->reset_lookup_counter(ctx);
  tab->find(ctx, n, keys_bw(keys.data()), hit_mask_bw(), value_stride, nullptr, nullptr);
  tab->get_lookup_counter(ctx, &cnt);
  ASSERT_EQ(cnt, 0);
  ASSERT_EQ(tab->size(ctx, true), 0);

  // Insert -> find returns every key with the original payload.
  tab->insert(ctx, n, keys_bw(keys.data()), value_stride, max_value_size, values_in_bw());
  ASSERT_EQ(tab->size(ctx, true), n);

  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  std::fill(out_values.begin(), out_values.end(), 0);
  tab->reset_lookup_counter(ctx);
  tab->find(ctx, n, keys_bw(keys.data()), hit_mask_bw(), value_stride, values_out_bw(),
            value_sizes_bw());
  tab->get_lookup_counter(ctx, &cnt);
  ASSERT_EQ(cnt, n);
  for (int64_t i{}; i != n; ++i) {
    ASSERT_EQ(out_value_sizes[static_cast<size_t>(i)], max_value_size) << "i=" << i;
    for (int64_t j{}; j != max_value_size; ++j) {
      ASSERT_EQ(out_values[static_cast<size_t>(i * value_stride + j)],
                values[static_cast<size_t>(i * value_stride + j)])
          << "mismatch at key " << i << " byte " << j;
    }
  }

  // Erase the first half; only the second half should remain.
  const int64_t half{n / 2};
  tab->erase(ctx, half, keys_bw(keys.data()));
  ASSERT_EQ(tab->size(ctx, true), n - half);

  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  tab->reset_lookup_counter(ctx);
  tab->find(ctx, n, keys_bw(keys.data()), hit_mask_bw(), value_stride, nullptr, nullptr);
  tab->get_lookup_counter(ctx, &cnt);
  ASSERT_EQ(cnt, n - half);

  // Clear (SCAN+DEL by prefix) removes the rest.
  tab->clear(ctx);
  ASSERT_EQ(tab->size(ctx, true), 0);
}

// Two string-mode tables sharing one standalone Redis but with distinct `string_namespace_id` values must be
// fully isolated: identical keys map to disjoint Redis keys, `size()` is prefix-scoped, and clearing
// one table leaves the other untouched. Fails when no Redis server is reachable.
TEST(redis_plugin, single_node_string_namespace_id_isolation) {
  using key_type = int64_t;

  RedisClusterTableFactoryConfig fac_conf;
  fac_conf.address = "localhost:6379";
  fac_conf.single_node = true;
  RedisClusterTableFactory fac{fac_conf};

  const int64_t max_value_size{16};
  const int64_t value_stride{max_value_size};
  auto make_table_conf = [&](int64_t string_namespace_id) {
    return nlohmann::json{{"mask_size", sizeof(max_bitmask_repr_t)},
                          {"key_size", sizeof(key_type)},
                          {"max_value_size", max_value_size},
                          {"value_dtype", "float32"},
                          {"num_partitions", 0},
                          {"string_namespace_id", string_namespace_id}};
  };

  // Two tables that differ only by their key prefix (table id is irrelevant to key naming in string
  // mode, so isolation must come from the prefix alone).
  host_table_ptr_t tab_a{fac.produce(1, make_table_conf(111))};
  host_table_ptr_t tab_b{fac.produce(2, make_table_conf(222))};
  auto ctx_a = tab_a->create_execution_context(0, 0, nullptr, nullptr);
  auto ctx_b = tab_b->create_execution_context(0, 0, nullptr, nullptr);

  // Probe connectivity; redis-plus-plus connects lazily, so the first command reveals an outage.
  try {
    tab_a->clear(ctx_a);
    tab_b->clear(ctx_b);
  } catch (const std::exception& e) {
    GTEST_FAIL() << "No standalone Redis reachable at localhost:6379: " << e.what();
  }

  const int64_t n{1500};
  std::vector<key_type> keys(static_cast<size_t>(n));
  std::iota(keys.begin(), keys.end(), key_type{1});

  // Distinct payloads per table so a cross-read would be detectable.
  auto make_values = [&](uint8_t salt) {
    std::vector<char> v(static_cast<size_t>(n * value_stride));
    for (int64_t i{}; i != n; ++i) {
      for (int64_t j{}; j != max_value_size; ++j) {
        v[static_cast<size_t>(i * value_stride + j)] = static_cast<char>((i + j + salt) & 0xff);
      }
    }
    return v;
  };
  const std::vector<char> values_a{make_values(0x10)};
  const std::vector<char> values_b{make_values(0x20)};

  auto do_insert = [&](host_table_ptr_t& tab, context_ptr_t& ctx, const std::vector<char>& vals) {
    auto kbw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(),
                                                           static_cast<size_t>(n) * sizeof(key_type));
    auto vbw = std::make_shared<BufferWrapper<const void>>(ctx, "values", vals.data(), vals.size());
    tab->insert(ctx, n, std::move(kbw), value_stride, max_value_size, std::move(vbw));
  };

  // Find all `keys` in `tab`; writes the hit count to `cnt_out` and, when `expected` is non-null,
  // asserts the returned payloads match it byte-for-byte. Void return so ASSERT_* is allowed.
  auto find_check = [&](host_table_ptr_t& tab, context_ptr_t& ctx, const std::vector<char>* expected,
                        int64_t& cnt_out) {
    std::vector<max_bitmask_repr_t> hm(static_cast<uint64_t>(max_bitmask_t::mask_size(n)), 0);
    std::vector<char> out(static_cast<size_t>(n * value_stride), 0);
    auto kbw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(),
                                                           static_cast<size_t>(n) * sizeof(key_type));
    auto hbw = std::make_shared<BufferWrapper<max_bitmask_repr_t>>(
        ctx, "hit_mask", hm.data(),
        static_cast<size_t>(max_bitmask_t::mask_size(n)) * sizeof(max_bitmask_repr_t));
    auto vbw = std::make_shared<BufferWrapper<void>>(ctx, "values", out.data(), out.size());
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, std::move(kbw), std::move(hbw), value_stride,
              expected ? vbw : std::shared_ptr<BufferWrapper<void>>{}, nullptr);
    tab->get_lookup_counter(ctx, &cnt_out);
    if (expected) {
      for (size_t b{}; b != out.size(); ++b) {
        ASSERT_EQ(out[b], (*expected)[b]) << "payload mismatch at byte " << b;
      }
    }
  };

  // Insert disjoint data into each table.
  do_insert(tab_a, ctx_a, values_a);
  do_insert(tab_b, ctx_b, values_b);

  // `size()` is prefix-scoped: each table sees only its own n keys, not the combined 2*n.
  ASSERT_EQ(tab_a->size(ctx_a, true), n);
  ASSERT_EQ(tab_b->size(ctx_b, true), n);

  int64_t cnt{};
  // Each table reads back its own payloads despite the keys being identical.
  find_check(tab_a, ctx_a, &values_a, cnt);
  ASSERT_EQ(cnt, n);
  find_check(tab_b, ctx_b, &values_b, cnt);
  ASSERT_EQ(cnt, n);

  // Clearing A (SCAN+DEL by prefix 111) must not touch B (prefix 222).
  tab_a->clear(ctx_a);
  ASSERT_EQ(tab_a->size(ctx_a, true), 0);
  ASSERT_EQ(tab_b->size(ctx_b, true), n);
  find_check(tab_a, ctx_a, nullptr, cnt);
  ASSERT_EQ(cnt, 0);
  find_check(tab_b, ctx_b, &values_b, cnt);
  ASSERT_EQ(cnt, n);

  // Clean up the remaining table.
  tab_b->clear(ctx_b);
  ASSERT_EQ(tab_b->size(ctx_b, true), 0);
}

// String mode with `num_partitions > 1` fans the client-side work across thread-pool tasks. Results
// must be identical to the single-threaded path: every key found with its exact payload, prefix-scoped
// size/clear. Uses a key prefix so the parallel prefixed `find` path (per-task scratch buffers) is
// exercised too. Skips gracefully when no Redis server is reachable.
TEST(redis_plugin, single_node_string_parallel_crud) {
  using key_type = int64_t;

  RedisClusterTableFactoryConfig fac_conf;
  fac_conf.address = "localhost:6379";
  fac_conf.single_node = true;
  RedisClusterTableFactory fac{fac_conf};

  const int64_t max_value_size{24};
  const int64_t value_stride{max_value_size};
  // num_partitions == 4 -> 4 parallel work partitions over plain Redis strings.
  nlohmann::json table_conf{{"mask_size", sizeof(max_bitmask_repr_t)},
                            {"key_size", sizeof(key_type)},
                            {"max_value_size", max_value_size},
                            {"value_dtype", "e4m3"},
                            {"num_partitions", 4},
                            {"string_namespace_id", 555}};
  host_table_ptr_t tab{fac.produce(11, table_conf)};
  auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);

  try {
    tab->clear(ctx);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "No standalone Redis reachable at localhost:6379: " << e.what();
  }

  // Span many mask words so the work genuinely splits across the 4 partitions.
  const int64_t n{8000};
  std::vector<key_type> keys(static_cast<size_t>(n));
  std::iota(keys.begin(), keys.end(), key_type{1});

  std::vector<char> values(static_cast<size_t>(n * value_stride));
  for (int64_t i{}; i != n; ++i) {
    for (int64_t j{}; j != max_value_size; ++j) {
      values[static_cast<size_t>(i * value_stride + j)] = static_cast<char>((i * 7 + j) & 0xff);
    }
  }

  std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)));
  std::vector<char> out_values(static_cast<size_t>(n * value_stride));

  auto keys_bw = [&]() {
    return std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(),
                                                       static_cast<size_t>(n) * sizeof(key_type));
  };
  auto hit_mask_bw = [&]() {
    return std::make_shared<BufferWrapper<max_bitmask_repr_t>>(
        ctx, "hit_mask", hit_mask.data(),
        static_cast<size_t>(max_bitmask_t::mask_size(n)) * sizeof(max_bitmask_repr_t));
  };

  int64_t cnt{};

  // Insert (parallel) then find (parallel) — every key hits with its exact payload.
  tab->insert(ctx, n, keys_bw(), value_stride, max_value_size,
              std::make_shared<BufferWrapper<const void>>(ctx, "values_in", values.data(),
                                                          values.size()));
  ASSERT_EQ(tab->size(ctx, true), n);

  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  std::fill(out_values.begin(), out_values.end(), 0);
  tab->reset_lookup_counter(ctx);
  tab->find(ctx, n, keys_bw(), hit_mask_bw(), value_stride,
            std::make_shared<BufferWrapper<void>>(ctx, "values_out", out_values.data(),
                                                  out_values.size()),
            nullptr);
  tab->get_lookup_counter(ctx, &cnt);
  ASSERT_EQ(cnt, n);
  ASSERT_EQ(out_values, values) << "parallel find returned wrong payloads";
  // Every key's hit bit must be set.
  for (int64_t i{}; i != n; ++i) {
    const bool hit{((hit_mask[static_cast<size_t>(i) / 64] >> (static_cast<size_t>(i) % 64)) &
                    0x1u) != 0};
    ASSERT_TRUE(hit) << "missing hit for key " << i;
  }

  // Erase a sub-range across partitions, then confirm only the rest remains.
  const int64_t erase_n{n / 2};
  tab->erase(ctx, erase_n, keys_bw());
  ASSERT_EQ(tab->size(ctx, true), n - erase_n);

  std::fill(hit_mask.begin(), hit_mask.end(), 0);
  tab->reset_lookup_counter(ctx);
  tab->find(ctx, n, keys_bw(), hit_mask_bw(), value_stride, nullptr, nullptr);
  tab->get_lookup_counter(ctx, &cnt);
  ASSERT_EQ(cnt, n - erase_n);

  tab->clear(ctx);
  ASSERT_EQ(tab->size(ctx, true), 0);
}
