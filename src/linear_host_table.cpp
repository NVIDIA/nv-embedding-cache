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

#include <linear_host_table.hpp>

namespace nve {

void LinearHostTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(emb_table != nullptr, "UVM table pointer was not initialized in config");
  NVE_CHECK_(max_threads > 0, "Max threads need to be greater than 0.");
}

void from_json(const nlohmann::json& json, LinearHostTableConfig& conf) {
  using base_type = LinearHostTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  // NVE_READ_JSON_FIELD_(emb_table); Not supported right now.
  NVE_READ_JSON_FIELD_(value_dtype);
  NVE_READ_JSON_FIELD_(max_threads);
  
  NVE_THROW_("LinearHostTable always requires a dynamically allocated pointer and cannot be configured from JSON");
}

void to_json(nlohmann::json& json, const LinearHostTableConfig& conf) {
  using base_type = LinearHostTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  // NVE_WRITE_JSON_FIELD_(emb_table); Not supported right now.
  NVE_WRITE_JSON_FIELD_(value_dtype);
  NVE_WRITE_JSON_FIELD_(max_threads);

  NVE_THROW_("LinearHostTable always requires a dynamically allocated pointer and cannot be configured from JSON");
}

} // namespace nve