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

#ifndef NVE_PLUGIN_COMMON_H
#define NVE_PLUGIN_COMMON_H

/*
 * Contract shared by internal and external plugins.
 *
 * Every plugin shared object exports the five metadata symbols declared here,
 * followed by the creation symbol for the mode it reports.
 */

#include <stdint.h>

#include <nve_c_api.h> /* NVE_C_API */

#ifdef __cplusplus
extern "C" {
#endif

#define NVE_PLUGIN_ABI_VERSION_MAJOR 2u
#define NVE_PLUGIN_ABI_VERSION_MINOR 0u

typedef int32_t nve_plugin_mode_t;
enum {
  NVE_PLUGIN_MODE_INTERNAL = 0, /* C++ ABI, same toolchain       */
  NVE_PLUGIN_MODE_EXTERNAL = 1, /* pure-C ABI, unknown toolchain */
  /* reserved for future modes                                   */
};

NVE_C_API uint32_t nve_plugin_abi_version_major(void);
NVE_C_API uint32_t nve_plugin_abi_version_minor(void);
NVE_C_API const char* nve_plugin_name(void);
NVE_C_API const char* nve_plugin_author(void);
NVE_C_API nve_plugin_mode_t nve_plugin_mode(void);

#ifdef __cplusplus
} /* extern "C" */

static_assert(sizeof(nve_plugin_mode_t) == 4, "nve_plugin_mode_t must be 32-bit");
#else
_Static_assert(sizeof(nve_plugin_mode_t) == 4, "nve_plugin_mode_t must be 32-bit");
#endif

#endif /* NVE_PLUGIN_COMMON_H */
