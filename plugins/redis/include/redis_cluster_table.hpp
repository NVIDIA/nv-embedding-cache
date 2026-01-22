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

#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wimplicit-int-conversion"
#endif
#pragma GCC diagnostic ignored "-Wconversion"
#include <sw/redis++/redis++.h>
#pragma GCC diagnostic pop

#include <bit_ops.hpp>
#include <host_table.hpp>
#include <string>
#include <string_view>
#include <thread_pool.hpp>

namespace nve {
namespace plugin {

struct RedisClusterTableConfig final : public HostTableConfig {
  using base_type = HostTableConfig;

  int64_t max_batch_size{16'384};  // Maximum batch size to use for queries into the column family.
                                   // Must be a multiple of `mask_size`.

  int64_t num_partitions{
      1};  // Either 0 or a power of 2. To denote the number of parallel partitions. If this is
           // zero, then we will not create partitions, and use key/value assignment. If >=1,
           // determines the maximum degree of parallelization. For achieving the best performance,
           // this should be significantly higher than the number of cluster nodes! We use modulo-N
           // to assign partitions. Hence, you must not change this value after writing the first
           // data to a table.
  Partitioner_t partitioner{default_partitioner};  // Partitioner to use.
  std::vector<int64_t> workgroups{0};  // Workgroup to use per partition (thread pool feature). Will wrap around.
  std::string hash_key;

  OverflowPolicyConfig overflow_policy;  // Overflow detection / handling parameters.

  void check() const;
};

void from_json(const nlohmann::json& json, RedisClusterTableConfig& conf);

void to_json(nlohmann::json& json, const RedisClusterTableConfig& conf);

using redis_cluster_ptr_t = std::shared_ptr<sw::redis::RedisCluster>;

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
class RedisClusterTable final : public HostTable<RedisClusterTableConfig> {
 public:
  using base_type = HostTable<RedisClusterTableConfig>;
  using mask_type = MaskType;
  using mask_repr_type = typename mask_type::repr_type;
  using key_type = KeyType;
  using meta_type = MetaType;
  static constexpr PartitionerType partitioner{};

  NVE_PREVENT_COPY_AND_MOVE_(RedisClusterTable);

  RedisClusterTable() = delete;

  RedisClusterTable(table_id_t table_id, const config_type& config, redis_cluster_ptr_t& cluster);

  ~RedisClusterTable() override = default;

  void clear(context_ptr_t& ctx) override;

  void erase(context_ptr_t& ctx, int64_t n, const void* keys) override;

  void find(context_ptr_t& ctx, int64_t n, const void* keys, max_bitmask_repr_t* hit_mask,
            int64_t value_stride, void* values, int64_t* value_sizes) const override;

  void insert(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  int64_t size(context_ptr_t& ctx, bool exact) const override;

  void update(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  void update_accumulate(context_ptr_t& ctx, int64_t n, const void* keys, int64_t update_stride,
                         int64_t update_size, const void* updates,
                         DataType_t update_dtype) override;

 private:
  template <bool WithValues, bool WithValueSizes>
  int64_t find_(context_ptr_t& ctx, int64_t n, const key_type* keys, char* hit_mask,
                int64_t value_stride, char* values, int64_t* value_sizes) const;

 private:
  redis_cluster_ptr_t cluster_;
};

struct RedisClusterTableFactoryConfig final : public HostTableFactoryConfig {
  using base_type = HostTableFactoryConfig;

  std::string address{"localhost:6379"};  // The destination address any Redis node in the clsuter.
  std::string user_name{"default"};       // Redis username.
  std::string password{};                 // Plaintext password of the user.

  bool keep_alive{true};  // Send keep alive messages to prevent TCP channel from collapsing.
  int64_t connections_per_node{
      5};  // Maximum number of parallel connections with the same redis node.

  bool use_tls{false};  // If true, encrypt connections with SSL/TLS.
  std::string ca_certificate{
      "cacertbundle.crt"};  // Path to a file or directory containing certificates of trusted CAs.
  std::string client_certificate{"client_cert.pem"};  // Certificate to use for this client.
  std::string client_key{"client_key.pem"};           // Private key to use for this client.
  std::string server_name_identification{
      "redis.localhost"};  // SNI to request (can deviate from connection address).

  void check() const;
};

void from_json(const nlohmann::json& json, RedisClusterTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const RedisClusterTableFactoryConfig& conf);

class RedisClusterTableFactory final
    : public HostTableFactory<RedisClusterTableFactoryConfig, RedisClusterTableConfig> {
 public:
  using base_type = HostTableFactory<RedisClusterTableFactoryConfig, RedisClusterTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(RedisClusterTableFactory);

  RedisClusterTableFactory() = delete;

  RedisClusterTableFactory(const config_type& config);

  ~RedisClusterTableFactory() override = default;

  host_table_ptr_t produce(table_id_t id, const table_config_type& config) override;

 private:
  redis_cluster_ptr_t cluster_;
};

}  // namespace plugin
}  // namespace nve
