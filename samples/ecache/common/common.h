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
#include <string>
#include <chrono>
#include <iostream>
#include <cstdio>
#include <tuple>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "include/ecache/embed_cache.h"
#include <algorithm>
#include <random>
#include <type_traits>
#include <cuda_fp16.h>

#ifndef cpuErrChk
#define cpuErrChk(cond, msg)            \
do {                                    \
    if (!(cond)) {                      \
        fprintf(stderr, "Failed assertion (" #cond "): %s %d %s\n", __FILE__, __LINE__, msg); \
        exit(-1);                       \
    }                                   \
} while (0)
#endif //cpuErrChk

#ifndef gpuErrChk
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif //gpuErrChk

#ifndef drvErrChk
#define drvErrChk(ans) { drvAssert((ans), __FILE__, __LINE__); }
inline void drvAssert(CUresult code, const char *file, int line, bool abort=true) {
    if (code != CUDA_SUCCESS) {
        const char* pErrName;
        const char* pErrDesc;
        if (cuGetErrorName(code, &pErrName) != CUDA_SUCCESS) {
            pErrName = nullptr;
        }
        if (cuGetErrorString(code, &pErrDesc) != CUDA_SUCCESS) {
            pErrDesc = nullptr;
        }
        fprintf(stderr, "DRVassert: (%d) %s %s %s %d\n", code, pErrName, pErrDesc, file, line);
        if (abort) exit(code);
    }
}
#endif //drvErrChk

#ifndef ecErrChk
#define ecErrChk(ans) { ecAssert((ans), __FILE__, __LINE__); }
inline void ecAssert(nve::ECError code, const char *file, int line, bool abort=true) {
    if (code != ECERROR_SUCCESS) {
        fprintf(stderr, "Embedding Cache assert: %d %s %d\n", code, file, line);
        if (abort) exit(static_cast<int>(code));
    }
}
#endif //ecErrChk

void ReportMemUsage(std::string msg={});
class SimpleTimer {
public:
    SimpleTimer(const std::string& name): m_name(name), m_start(std::chrono::high_resolution_clock::now()) {}
    void Report() {
        auto stop = std::chrono::high_resolution_clock::now();
        auto usec = std::chrono::duration_cast<std::chrono::microseconds>(stop - m_start).count();
        constexpr int64_t sec_ratio = 1000000l;
        constexpr int64_t ms_ratio = 1000l;
        if (usec > sec_ratio) {
            std::cout << m_name << " took: " << usec / sec_ratio << " seconds." << std::endl;
        } else {
            std::cout << m_name << " took: " << usec / ms_ratio << " milliseconds." << std::endl;
        }
    }
    const std::string& Name() const { return m_name; }
private:
    const std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

struct BandwidthFormatter {
    BandwidthFormatter(double _bytes, double _seconds) : bytes(_bytes), seconds(_seconds) {}
    double bytes;
    double seconds;
};

std::ostream& operator<<(std::ostream& os, const BandwidthFormatter& bf);

template <typename T>
inline void fillBuffer(void* buffer, int64_t volume, T min, T max) {
    T* typedBuffer = static_cast<T*>(buffer);
    std::default_random_engine engine;
    if (std::is_integral<T>::value){
        std::uniform_int_distribution<int> distribution(min,max);
    auto generator = [&engine,&distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
    }
    else {
        std::uniform_real_distribution<float> distribution(min, max);
        auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
        std::generate(typedBuffer, typedBuffer + volume, generator);
    }
}

// Specialization needed for custom type __half
template <typename H>
inline void fillBufferHalf(void* buffer, int64_t volume, H min, H max)
{
    H* typedBuffer = static_cast<H*>(buffer);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<H>(distribution(engine)); };
    std::generate(typedBuffer, typedBuffer + volume, generator);
}
template <>
inline void fillBuffer<__half>(void* buffer, int64_t volume, __half min, __half max)
{
    fillBufferHalf(buffer, volume, min, max);
}

