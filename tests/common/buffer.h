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
#include <cuda_runtime.h>
#include "../common/check_error.h"

template<typename T>
struct Buffer
{
    Buffer(size_t size) : m_size(size)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&pd, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&ph, size));
    }

    ~Buffer()
    {
        cudaFree(pd);
        cudaFreeHost(ph);
    }

    void DtoH(cudaStream_t stream)
    {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(ph, pd, m_size, cudaMemcpyDefault, stream));
    }

    void HtoD(cudaStream_t stream)
    {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(pd, ph, m_size, cudaMemcpyDefault, stream));
    }
    
    T* pd = nullptr;
    T* ph = nullptr;
    size_t m_size;
};
