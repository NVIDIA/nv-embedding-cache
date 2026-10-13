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
#include "sph/error.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <utility>

// ============================================================================
// SPH_CHECK — the engine's single check macro. INTERNAL: this header lives in
// support/, which is private to the build. The exception and result types it
// throws are public and live in <sph/error.h>.
//
//   SPH_CHECK(<expr>[, <message parts>...])
//
// <expr> may be a bool, a cudaError_t, a CUresult, or an InsertResult; the
// overload set in detail:: picks the right success test and the right
// exception. This replaces the former CUDA_CHECK / CU_CHECK pair and the bare
// asserts, so there is one spelling for "this must hold" regardless of what
// produced the value:
//
//   SPH_CHECK(cudaMalloc(&p, n));                          // cudaError_t
//   SPH_CHECK(cuMemCreate(&h, sz, &prop, 0), "shard ", s);  // CUresult
//   SPH_CHECK(num_keys >= 0, "num_keys must be >= 0");      // bool
//   SPH_CHECK(result);                                      // InsertResult
//
// Unconditional failure is spelled SPH_CHECK(false, "..."). There is no second
// macro on purpose.
//
// Everything is checked in release builds too — the asserts this replaces were
// compiled out under NDEBUG, which turned a VMM commit failure into a silent
// no-op followed by a kernel faulting on unmapped memory.
//
// See <sph/error.h> for which status maps to which exception.
//
// NOTE: SPH_CHECK throws, so it must not be used in a destructor. Teardown
// paths (ShardedVmmBuffer::unmap_shard, ResizeableBuffer::~ResizeableBuffer,
// SphDeduper::~SphDeduper) deliberately call the raw driver/runtime APIs.
// ============================================================================

namespace sph {

namespace detail {

// ---- success tests: one overload per status type SPH_CHECK accepts ---------
inline bool sph_is_success(bool v)        { return v; }
inline bool sph_is_success(cudaError_t v) { return v == cudaSuccess; }
inline bool sph_is_success(CUresult v)    { return v == CUDA_SUCCESS; }
inline bool sph_is_success(const InsertResult& v) { return v.ok(); }

// ---- variadic message join ------------------------------------------------
inline void sph_append(std::ostringstream&) {}

template <typename T, typename... Rest>
void sph_append(std::ostringstream& os, const T& value, const Rest&... rest) {
    os << value;
    sph_append(os, rest...);
}

template <typename... Args>
std::string sph_join(const Args&... args) {
    std::ostringstream os;
    sph_append(os, args...);
    return os.str();
}

inline std::string sph_compose(const char* expr, const char* status_text,
                               const std::string& message, const char* file, int line) {
    std::ostringstream os;
    os << "SPH_CHECK(" << expr << ") failed";
    if (status_text && *status_text) os << ": " << status_text;
    if (!message.empty())            os << ": " << message;
    os << " [" << file << ":" << line << "]";
    return os.str();
}

// ---- failure handlers: one overload per status type ------------------------
// Each exception carries the full failure detail so API boundaries can retain
// the status text, failed expression, and source location.
[[noreturn]] inline void sph_fail(bool, const char* file, int line, const char* expr,
                                  const std::string& message) {
    std::string what = sph_compose(expr, nullptr, message, file, line);
    throw SphInvalidArgument(std::move(what), file, line, expr);
}

[[noreturn]] inline void sph_fail(cudaError_t code, const char* file, int line, const char* expr,
                                  const std::string& message) {
    std::string what = sph_compose(expr, cudaGetErrorString(code), message, file, line);
    if (code == cudaErrorMemoryAllocation) {
        throw SphOutOfMemory(std::move(what), file, line, expr);
    }
    throw SphCudaError(std::move(what), file, line, expr, static_cast<int>(code));
}

[[noreturn]] inline void sph_fail(CUresult code, const char* file, int line, const char* expr,
                                  const std::string& message) {
    const char* status_text = nullptr;
    cuGetErrorString(code, &status_text);
    std::string what = sph_compose(expr, status_text ? status_text : "(unknown driver error)",
                                   message, file, line);
    if (code == CUDA_ERROR_OUT_OF_MEMORY) {
        throw SphOutOfMemory(std::move(what), file, line, expr);
    }
    throw SphCudaError(std::move(what), file, line, expr, static_cast<int>(code));
}


[[noreturn]] inline void sph_fail(const InsertResult& r, const char* file, int line,
                                  const char* expr, const std::string& message) {
    std::ostringstream status;
    status << "dropped " << r.keys_dropped << " of " << r.keys_submitted << " keys across "
           << r.failures.total() << " bucket(s)"
           << " (pilot_failure=" << r.failures.pilot_failure
           << " max_bucket_reached=" << r.failures.max_bucket_reached
           << " bucket_offset_overflow=" << r.failures.bucket_offset_overflow
           << " page_index_overflow=" << r.failures.page_index_overflow << ")";
    std::string what = sph_compose(expr, status.str().c_str(), message, file, line);
    throw SphInsertFailed(std::move(what), file, line, expr, r);
}

}  // namespace detail
}  // namespace sph

#ifndef SPH_CHECK
#define SPH_CHECK(_expr_, ...)                                                        \
    do {                                                                              \
        const auto sph_check_result_ = (_expr_);                                      \
        if (!::sph::detail::sph_is_success(sph_check_result_)) {                 \
            ::sph::detail::sph_fail(sph_check_result_, __FILE__, __LINE__,       \
                                         #_expr_,                                     \
                                         ::sph::detail::sph_join(__VA_ARGS__));  \
        }                                                                             \
    } while (false)
#endif
