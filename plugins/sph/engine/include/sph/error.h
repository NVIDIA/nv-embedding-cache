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
// ============================================================================
// sph public error and result vocabulary.
//
// Everything a caller needs to handle a failure from the engine: the exception
// hierarchy and the InsertResult that insert()/build() report. Split out of the
// internal support/sph_check.h so the public headers depend only on include/,
// and so an embedder can catch SPH failures without pulling in the SPH_CHECK
// machinery or any CUDA header — this file is plain C++.
//
// Exception mapping is chosen so the nve C-API boundary (NVE_C_CATCH in
// src/nve_c_api_internal.hpp) classifies SPH failures with no changes there:
//
//   invalid argument / precondition -> SphInvalidArgument : std::invalid_argument
//                                                        -> NVE_ERROR_INVALID_ARGUMENT
//   allocation failure (incl. VMM)  -> SphOutOfMemory     : std::bad_alloc
//                                                        -> NVE_ERROR_OUT_OF_MEMORY
//   any other CUDA/driver status    -> SphCudaError       : std::runtime_error
//                                                        -> NVE_ERROR_RUNTIME
//   insert/build dropped keys       -> SphInsertFailed    : SphError
//                                                        -> NVE_ERROR_RUNTIME
//
// The engine stays standalone: these are its own types, so it never needs to
// depend on nve:: exception classes. Every one of them also derives from the
// SphException marker base, so an embedder can catch all SPH failures with a
// single `catch (const sph::SphException&)`.
// ============================================================================
#include <atomic>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace sph {

// ---- Teardown warnings -----------------------------------------------------
// Destructors and release paths cannot throw, so when a CUDA/driver release
// call fails there the engine reports a composed message through this hook
// instead. No handler is installed by default and the engine prints nothing;
// an embedder forwards the message to its own logger. The handler must not
// throw. Installing is a single atomic store, so it may happen at any time.
using WarningHandler = void (*)(const char* message);

namespace detail {
inline std::atomic<WarningHandler>& warning_handler() noexcept {
    static std::atomic<WarningHandler> handler{nullptr};
    return handler;
}
}  // namespace detail

inline void set_warning_handler(WarningHandler handler) noexcept {
    detail::warning_handler().store(handler, std::memory_order_relaxed);
}

inline void report_warning(const char* message) noexcept {
    if (const WarningHandler handler = detail::warning_handler().load(std::memory_order_relaxed)) {
        handler(message);
    }
}

// Marker base shared by every SPH exception. Deliberately NOT derived from
// std::exception: each concrete type picks the std base that gives it the right
// classification at the C-API boundary, and a common std base would make the
// `what()` lookup ambiguous.
class SphException {
 public:
    virtual ~SphException() = default;

    const std::string& details() const noexcept { return what_; }
    const char*        file()    const noexcept { return file_; }
    int                line()    const noexcept { return line_; }
    const char*        expr()    const noexcept { return expr_; }

 protected:
    SphException(std::string what, const char* file, int line, const char* expr)
        : what_(std::move(what)), file_(file), line_(line), expr_(expr) {}

    std::string what_;
    const char* file_;
    int         line_;
    const char* expr_;
};

// Generic SPH runtime failure.
class SphError : public SphException, public std::runtime_error {
 public:
    SphError(std::string what, const char* file, int line, const char* expr)
        : SphException(std::move(what), file, line, expr), std::runtime_error(what_) {}
};

// A CUDA runtime or driver call failed. code() is the raw cudaError_t/CUresult,
// kept as int so this header stays free of CUDA includes.
class SphCudaError : public SphError {
 public:
    SphCudaError(std::string what, const char* file, int line, const char* expr, int code)
        : SphError(std::move(what), file, line, expr), code_(code) {}

    int code() const noexcept { return code_; }

 private:
    int code_;
};

// A caller-supplied argument or a precondition failed.
class SphInvalidArgument : public SphException, public std::invalid_argument {
 public:
    SphInvalidArgument(std::string what, const char* file, int line, const char* expr)
        : SphException(std::move(what), file, line, expr), std::invalid_argument(what_) {}
};

// Device or host allocation failed (including VMM commit).
class SphOutOfMemory : public SphException, public std::bad_alloc {
 public:
    SphOutOfMemory(std::string what, const char* file, int line, const char* expr)
        : SphException(std::move(what), file, line, expr) {}

    const char* what() const noexcept override { return what_.c_str(); }
};

// ----------------------------------------------------------------------------
// insert() / build() outcome — the payload of SphInsertFailed.
// ----------------------------------------------------------------------------

// Per-failure-mode counts from one insert()/build(). One bump per bucket that
// hit each mode; a non-zero total() means some keys did not land.
struct InsertFailures {
    int64_t pilot_failure          = 0;
    int64_t max_bucket_reached     = 0;
    int64_t bucket_offset_overflow = 0;
    int64_t page_index_overflow    = 0;
    int64_t total() const {
        return pilot_failure + max_bucket_reached + bucket_offset_overflow + page_index_overflow;
    }
};

// Outcome of a failed insert()/build(). Note neither is atomic: when keys are
// dropped the rest of the batch has already been applied, so this reports how
// much landed as well as which limits were hit.
struct InsertResult {
    InsertFailures failures;              // which limits were hit, per bucket
    int64_t        keys_submitted = 0;    // num_keys passed in
    int64_t        keys_dropped   = 0;    // batch keys that did not land
    // Deliberately filtered out, not lost. Both are zero under ON_EXISTING_FAIL,
    // and under build(), which does no filtering.
    int64_t        keys_existing  = 0;    // already resident (value overwritten iff UPDATE)
    int64_t        keys_batch_duplicates = 0;  // repeats within this batch

    // Only the keys that grew the table. keys_existing / keys_batch_duplicates
    // land on a slot that already existed, so counting them would overstate
    // get_size() and skew get_fragmentation() and the defrag threshold.
    int64_t keys_inserted() const {
        return keys_submitted - keys_dropped - keys_existing - keys_batch_duplicates;
    }
    bool    ok()           const { return keys_dropped == 0 && failures.total() == 0; }
};

// Thrown by insert()/build() when any key was dropped. Carries the full per-mode
// breakdown so a caller can decide what to do (grow, defrag and retry, give up)
// instead of only learning that something went wrong.
class SphInsertFailed : public SphError {
 public:
    SphInsertFailed(std::string what, const char* file, int line, const char* expr,
                    const InsertResult& result)
        : SphError(std::move(what), file, line, expr), result_(result) {}

    const InsertResult& result() const noexcept { return result_; }

 private:
    InsertResult result_;
};

}  // namespace sph
