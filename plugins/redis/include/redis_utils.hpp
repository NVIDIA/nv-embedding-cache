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

#include <sw/redis++/redis++.h>

namespace sw {
namespace redis {
namespace reply {

/**
 * A Redis++ parser avoids materializing a string.
 *
 * WARNING: The pointer becomes invalid once reply has been consumed. Use only if you can
 * immediately consume/parse values like the iterators below.
 */
inline StringView parse(ParseTag<StringView>, redisReply& reply) {
  if (!reply::is_string(reply) && !reply::is_status(reply)) {
    throw ProtoError("Expect STRING reply.");
  }

  if (reply.str == nullptr) {
    throw ProtoError("A null string reply.");
  }

  return {reply.str, reply.len};
}

}  // namespace reply
}  // namespace redis
}  // namespace sw

namespace nve {
namespace plugin {

template <typename T>
constexpr sw::redis::StringView view_as_string(const T& value) noexcept {
  return {reinterpret_cast<const char*>(&value), sizeof(T)};
}

template <typename T>
constexpr const T& view_as_value(const sw::redis::StringView& view) noexcept {
  return *reinterpret_cast<const T*>(view.data());
}

template <typename ViewType>
class ReplyParser {
 public:
  // See `sw::redis::reply::parse` extension at the top of this file.
  using container_type = std::vector<ViewType>;
  using view_type = typename container_type::value_type;
  using callback_fn_type = std::function<void(int64_t, const view_type&&)>;

  ReplyParser(callback_fn_type callback) : callback_{std::move(callback)} {}

  inline ReplyParser& operator*() noexcept { return *this; }

  inline ReplyParser& operator=(const view_type& view) {
    callback_(idx_, std::move(view));
    return *this;
  }

  inline ReplyParser& operator++() noexcept {
    ++idx_;
    return *this;
  }

 private:
  int64_t idx_{};
  const callback_fn_type callback_;
};

}  // namespace plugin
}  // namespace nve
