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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cuda_support.hpp>

#ifndef EC_CHECK
#define EC_CHECK(ans) { ECAssert_((ans), __FILE__, __LINE__); }
template<typename ErrType>
inline void ECAssert_(ErrType code, const char *file, int line, bool abort=true) {
    if (code != ErrType(0)) {
        fprintf(stderr, "EC_CHECK: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}
#endif

// helper class to store and recover device when use multi gpu
// this class assumes single context and always restore to the default context
class ScopedDevice
{
public:
    ScopedDevice(int device_id) : m_device_id(device_id), m_curr_device(0), m_swap_device(false)
    {
        NVE_CHECK_(cudaGetDevice(&m_curr_device));
        m_swap_device = (m_device_id >= 0) && m_curr_device != m_device_id;
        if (m_swap_device) {
            NVE_CHECK_(cudaSetDevice(m_device_id));
        }
        
    }
    ~ScopedDevice()
    {
        if (m_swap_device) {
            NVE_CHECK_(cudaSetDevice(m_curr_device));
        }
    }
private:
    int m_device_id;
    int m_curr_device;
    bool m_swap_device;
};
