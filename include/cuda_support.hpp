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
#include <nvtx3/nvtx3.hpp>
#include <common.hpp>
#include <functional>
#include <memory>
#include <vector>

// NVTX macros
#ifdef NVE_NVTX_MARK_
#error NVE_NVTX_MARK_ was already defined.
#endif
#ifdef NVE_ENABLE_NVTX
#define NVE_NVTX_MARK_(msg) nvtx3::mark(msg)
#else
#define NVE_NVTX_MARK_(msg)
#endif

#ifdef NVE_NVTX_SCOPED_RANGE_
#error NVE_NVTX_SCOPED_RANGE_ was already defined.
#endif
#ifdef NVE_ENABLE_NVTX
#define NVE_NVTX_SCOPED_RANGE_(...) nvtx3::scoped_range _range(__VA_ARGS__)
#else
#define NVE_NVTX_SCOPED_RANGE_(...)
#endif

#define NVE_NVTX_SCOPED_FUNCTION_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__)
#define NVE_NVTX_SCOPED_FUNCTION_COL1_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__, nvtx3::rgb{32, 192, 224}) // Cyan
#define NVE_NVTX_SCOPED_FUNCTION_COL2_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__, nvtx3::rgb{224, 192, 32}) // Yellow
#define NVE_NVTX_SCOPED_FUNCTION_COL3_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__, nvtx3::rgb{192, 32, 224}) // Purple
#define NVE_NVTX_SCOPED_FUNCTION_COL4_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__, nvtx3::rgb{32, 224, 192}) // Green
#define NVE_NVTX_SCOPED_FUNCTION_COL5_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__, nvtx3::rgb{224, 64, 64}) // Red
#define NVE_NVTX_SCOPED_FUNCTION_COL6_() NVE_NVTX_SCOPED_RANGE_(__PRETTY_FUNCTION__, nvtx3::rgb{32, 32, 224}) // Blue

namespace nve {

template <>
constexpr bool is_success(const cudaError_t& result) noexcept {
  return result == cudaSuccess;
}

template <>
constexpr bool is_success(const CUresult& result) noexcept {
  return result == CUDA_SUCCESS;
}

/**
 * Thrown if a CUDA runtime API call fails. Don't use this directly. Use the `NVE_THROW_` and
 * `NVE_CHECK_` macros instead.
 */
template <>
class RuntimeError<cudaError_t> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[],
                      const cudaError_t& error, const std::string& hint) noexcept
      : base_type(file, line, expr, hint), error_{error} {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that), error_{that.error_} {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    error_ = that.error_;
    return *this;
  }

  inline cudaError_t error() const noexcept { return error_; }

  inline const char* errorName() const noexcept { return cudaGetErrorName(error_); }

  inline const char* errorString() const noexcept { return cudaGetErrorString(error_); }

  virtual const char* what() const noexcept override {
    return hint().empty() ? errorString() : hint().c_str();
  }

  virtual std::string to_string() const override {
    std::ostringstream o;

    const char* const what{this->what()};
    o << "CUDA runtime error " << errorName() << '[' << error() << "] = '" << what << "' @ " << file()
      << ':' << line();
    const std::string& thread{thread_name()};
    if (!thread.empty()) {
      o << " in thread: '" << thread << '\'';
    }
    const char* const expr{expression()};
    if (what != expr) {
      o << "', expression: '" << expr << '\'';
    }
    const char* const estr{errorString()};
    if (what != estr) {
      o << "', description: '" << estr << '\'';
    }
    o << '.';

    return o.str();
  }

 private:
  cudaError_t error_;
};

/**
 * Thrown if a CUDA driver API call fails. Don't use this directly. Use the `NVE_THROW_` and
 * `NVE_CHECK_` macros instead.
 */
template <>
class RuntimeError<CUresult> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[], const CUresult& result,
                      const std::string& hint) noexcept
      : base_type(file, line, expr, hint), result_{result} {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that), result_{that.result_} {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    result_ = that.result_;
    return *this;
  }

  inline CUresult result() const noexcept { return result_; }

  inline const char* errorName() const noexcept {
    const char* name;
    if (cuGetErrorName(result_, &name) != CUDA_SUCCESS) {
      name = "Call to `cuGetErrorName` failed!";
    }
    return name;
  }

  inline const char* errorString() const noexcept {
    const char* str;
    if (cuGetErrorString(result_, &str) != CUDA_SUCCESS) {
      str = "Call to `cuGetErrorString` failed!";
    }
    return str;
  }

  virtual const char* what() const noexcept override {
    return hint().empty() ? errorString() : hint().c_str();
  }

  virtual std::string to_string() const override {
    std::ostringstream o;

    const char* const what{this->what()};
    o << "CUDA driver error " << errorName() << '[' << result() << "] = '" << what << "' @ " << file()
      << ':' << line();
    const std::string& thread{thread_name()};
    if (!thread.empty()) {
      o << " in thread: '" << thread << '\'';
    }
    const char* const expr{expression()};
    if (what != expr) {
      o << "', expression: '" << expr << '\'';
    }
    const char* const estr{errorString()};
    if (what != estr) {
      o << "', description: '" << estr << '\'';
    }
    o << '.';

    return o.str();
  }

 private:
  CUresult result_;
};

}  // namespace nve
