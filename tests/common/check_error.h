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

#include "gtest/gtest.h"
#include "../../include/ecache/embed_cache.h" // for ECERROR_SUCCESS
#include <cuda_runtime.h> // for cudaSuccess

#ifdef __COVERITY__
#define CHECK_EC(ecerror)           \
do {                                \
  if ((ecerror) != ECERROR_SUCCESS) { \
    ADD_FAILURE();                  \
  }                                 \
} while (false)
#define CHECK_CUDA_ERROR(cudares)   \
do {                                \
  if ((cudares) != cudaSuccess) {     \
    ADD_FAILURE();                  \
  }                                 \
} while (false)
#else // __COVERITY__
#define CHECK_EC(ecerror) EXPECT_EQ((ecerror), ECERROR_SUCCESS)
#define CHECK_CUDA_ERROR(cudares) EXPECT_EQ((cudares), cudaSuccess)
#endif // __COVERITY__
