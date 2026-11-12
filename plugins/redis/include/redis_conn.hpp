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

#include <common.hpp>
#include <memory>
#include <utility>

namespace nve {
namespace plugin {

/**
 * Thin connection wrapper that hides the difference between a Redis Cluster connection
 * (`sw::redis::RedisCluster`, sharded, hash-based storage) and a standalone Redis connection
 * (`sw::redis::Redis`, single-node, string-based storage).
 *
 * Both classes expose an identical command surface for everything the table needs except
 * `pipeline()` (the cluster variant requires a hash-tag for slot routing) and the keyspace-wide
 * `dbsize()`/`flushdb()`/`scan()` commands, which only make sense against a standalone node.
 */
class RedisConn final {
 public:
  RedisConn() = delete;

  explicit RedisConn(std::shared_ptr<sw::redis::RedisCluster> cluster)
      : cluster_{std::move(cluster)} {}
  explicit RedisConn(std::shared_ptr<sw::redis::Redis> standalone)
      : standalone_{std::move(standalone)} {}

  // True when backed by a standalone (single-node) connection.
  bool is_single_node() const noexcept { return static_cast<bool>(standalone_); }

// Commands shared by both connection types are forwarded to whichever connection is set.
#define NVE_REDIS_FWD_(_func_name_, ...)                                     \
  template <typename... Args>                                                \
  auto _func_name_(Args&&... args) __VA_ARGS__ {                             \
    if (cluster_) {                                                          \
      return cluster_->_func_name_(std::forward<Args>(args)...);             \
    } else {                                                                 \
      return standalone_->_func_name_(std::forward<Args>(args)...);          \
    }                                                                        \
  }
  NVE_REDIS_FWD_(del)
  NVE_REDIS_FWD_(mget, const)
  NVE_REDIS_FWD_(mset)
  NVE_REDIS_FWD_(hset)
  NVE_REDIS_FWD_(hsetnx)
  NVE_REDIS_FWD_(hmget, const)
  NVE_REDIS_FWD_(hdel)
  NVE_REDIS_FWD_(hlen, const)
  NVE_REDIS_FWD_(hkeys, const)
  NVE_REDIS_FWD_(hscan, const)
  NVE_REDIS_FWD_(hgetall, const)
  NVE_REDIS_FWD_(hincrby)
#undef NVE_REDIS_FWD_

  // Cluster pipelines require a hash-tag for slot routing; standalone pipelines ignore it.
  sw::redis::Pipeline pipeline(const sw::redis::StringView& hash_tag) {
    if (cluster_) {
      return cluster_->pipeline(hash_tag, false);
    } else {
      return standalone_->pipeline(false);
    }
  }

  // Keyspace-wide commands are only meaningful against a standalone node (string mode).
  long long dbsize() const {
    NVE_CHECK_(is_single_node(), "`DBSIZE` is only supported in single-node mode.");
    return standalone_->dbsize();
  }

  void flushdb() {
    NVE_CHECK_(is_single_node(), "`FLUSHDB` is only supported in single-node mode.");
    standalone_->flushdb();
  }

  template <typename... Args>
  sw::redis::Cursor scan(Args&&... args) const {
    NVE_CHECK_(is_single_node(), "`SCAN` is only supported in single-node mode.");
    return standalone_->scan(std::forward<Args>(args)...);
  }

 private:
  std::shared_ptr<sw::redis::RedisCluster> cluster_;
  std::shared_ptr<sw::redis::Redis> standalone_;
};

}  // namespace plugin
}  // namespace nve
