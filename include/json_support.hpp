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
#pragma GCC diagnostic ignored "-Wunneeded-internal-declaration"
#endif
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop

#ifdef NVE_DEFINE_JSON_ENUM_CONVERSION_
#error NVE_DEFINE_JSON_ENUM_CONVERSION_ was already defined.
#endif
#define NVE_DEFINE_JSON_ENUM_CONVERSION_(_enum_type_, ...)                                        \
  static const auto _enum_type_##_enum_json_pairs{make_enum_json_pairs(__VA_ARGS__)};             \
                                                                                                  \
  void to_json(nlohmann::json& json, const _enum_type_ e) {                                       \
    const auto& pairs{_enum_type_##_enum_json_pairs};                                             \
    const auto it{                                                                                \
        std::find_if(pairs.begin(), pairs.end(), [e](const auto& kv) { return e == kv.first; })}; \
    NVE_CHECK_(it != pairs.end(), "Not a valid enumeration value.");                              \
    json = it->second;                                                                            \
  }                                                                                               \
                                                                                                  \
  void from_json(const nlohmann::json& j, _enum_type_& e) {                                       \
    const auto& pairs{_enum_type_##_enum_json_pairs};                                             \
    const auto it{std::find_if(pairs.begin(), pairs.end(),                                        \
                               [&j](const auto& kv) { return j == kv.second; })};                 \
    NVE_CHECK_(it != pairs.end(), "Not a valid enumeration value.");                              \
    e = it->first;                                                                                \
  }

#ifdef NVE_READ_JSON_FIELD_
#error NVE_READ_JSON_FIELD_ was already defined.
#endif
#define NVE_READ_JSON_FIELD_(_field_name_)   \
  do {                                       \
    const auto it{json.find(#_field_name_)}; \
    if (it != json.end()) {                  \
      it->get_to(conf._field_name_);         \
    }                                        \
  } while (false)

#ifdef NVE_WRITE_JSON_FIELD_
#error NVE_WRITE_JSON_FIELD_ was already defined.
#endif
#define NVE_WRITE_JSON_FIELD_(_field_name_)  \
  do {                                       \
    json[#_field_name_] = conf._field_name_; \
  } while (false)

namespace nve {

template <typename T>
using enum_json_pair_t = const std::pair<const T, const nlohmann::json>;

template <typename T>
constexpr enum_json_pair_t<T> make_enum_json_pair(const T value) {
  static_assert(std::is_enum_v<T>, "Type T must be an enum type!");
  return {value, to_string(value)};
}

template <typename Arg0, typename... Args>
constexpr std::array<enum_json_pair_t<Arg0>, 1 + sizeof...(Args)> make_enum_json_pairs(
    const Arg0&& arg0, Args&&... args) {
  return {make_enum_json_pair(arg0), make_enum_json_pair(args)...};
}

}  // namespace nve
