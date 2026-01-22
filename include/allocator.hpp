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
#include <cuda_runtime_api.h>
#include <cstdlib>

namespace nve {

/**
 * Allocator interface for large memory allocation.
 */
class Allocator {
public:
  // Todo: consider adding context to device functions
  virtual ~Allocator() = default;

  /**
   * Allocate buffer on a device.
   * @param ptr Pointer to update with the allocated buffer's address.
   * @param sz Size in bytes
   * @param device_id Device index to be used. Negative value implies use the current device.
   */
  virtual cudaError_t deviceAllocate(void** ptr, size_t sz, int device_id = -1) noexcept = 0;

  /**
   * Free a buffer on a device.
   * @param ptr Buffer address.
   * @param device_id Device index to be used. Negative value implies use the current device.
   * @note device_id must be the same as the value used during deviceAllocate
   */
  virtual cudaError_t deviceFree(void* ptr, int device_id = -1) noexcept = 0;

  /**
   * Allocate buffer on a device using a stream.
   * @param ptr Pointer to update with the allocated buffer's address.
   * @param sz Size in bytes
   * @param stream CUDA stream to use for allocation
   * @param device_id Device index to be used. Negative value implies use the current device.
   */
  virtual cudaError_t deviceAllocateAsync(void** ptr, size_t sz, cudaStream_t, int device_id = -1) noexcept {
    // Defaulting to synchronized version
    return deviceAllocate(ptr, sz, device_id);
  }

  /**
   * Free a buffer on a device using a stream.
   * @param ptr Buffer address.
   * @param stream CUDA stream to use for freeing
   * @param device_id Device index to be used. Negative value implies use the current device.
   * @note device_id must be the same as the value used during deviceAllocateAsync
   */
  virtual cudaError_t deviceFreeAsync(void* ptr, cudaStream_t, int device_id = -1) noexcept {
    // Defaulting to synchronized version
    return deviceFree(ptr, device_id);
  }

  /**
   * Allocate a buffer in host memory.
   * @param ptr Buffer address.
   * @param sz Size in bytes
   */
  virtual cudaError_t hostAllocate(void** ptr, size_t sz) noexcept = 0;

  /**
   * Free a buffer in host memory.
   * @param ptr Buffer address.
   */
  virtual cudaError_t hostFree(void* ptr) noexcept = 0;
};

}  // namespace nve
