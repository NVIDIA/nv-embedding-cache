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

#include <cstdlib>
#include <exception>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <memory>
#include <logging.hpp>

// --- Debugging system --- //

#ifdef NVE_ASSERT_
#error NVE_ASSERT_ was already defined.
#endif
#ifdef _DEBUG
#define NVE_ASSERT_(_expr_) \
  do {                      \
    if (!(_expr_)) {        \
      std::abort();         \
    }                       \
  } while (false)
#else
#define NVE_ASSERT_(_expr_) \
  do {                      \
  } while (false)
#endif

#ifdef NVE_PREVENT_COPY_
#error NVE_PREVENT_COPY_ was already defined.
#endif
#define NVE_PREVENT_COPY_(_classname_)      \
  _classname_(const _classname_&) = delete; \
  _classname_& operator=(const _classname_&) = delete

#ifdef NVE_PREVENT_MOVE_
#error NVE_PREVENT_MOVE_ was already defined.
#endif
#define NVE_PREVENT_MOVE_(_classname_) \
  _classname_(_classname_&&) = delete; \
  _classname_& operator=(_classname_&&) = delete

#ifdef NVE_PREVENT_COPY_AND_MOVE_
#error NVE_PREVENT_COPY_AND_MOVE_ was already defined.
#endif
#define NVE_PREVENT_COPY_AND_MOVE_(_classname_) \
  NVE_PREVENT_COPY_(_classname_);               \
  NVE_PREVENT_MOVE_(_classname_)

#ifdef NVE_CHECK_
#error NVE_CHECK_ was already defined.
#endif
#ifdef __COVERITY__
#define NVE_CHECK_(_expr_, ...)                               \
  do {                                                        \
    const auto _res_{_expr_};                                 \
    if (_res_ != decltype(_res_)())                           \
    {                                                         \
      NVE_LOG_CRITICAL_(nve::to_string(__VA_ARGS__));         \
    }                                                         \
  } while (false)
#else // __COVERITY__
#define NVE_CHECK_(_expr_, ...)                                                           \
  do {                                                                                    \
    const auto _res_{_expr_};                                                             \
    using res_type = std::remove_const_t<decltype(_res_)>;                                \
    if (!nve::is_success<res_type>(_res_)) {                                              \
      if constexpr (__func__[0] == '~') {                                                 \
        NVE_LOG_CRITICAL_(nve::RuntimeError<res_type>(__FILE__, __LINE__, #_expr_, _res_, \
                                                      nve::to_string(__VA_ARGS__)));      \
        NVE_IF_DEBUG_(std::exit(1));                                                      \
      } else {                                                                            \
        throw nve::RuntimeError<res_type>(__FILE__, __LINE__, #_expr_, _res_,             \
                                          nve::to_string(__VA_ARGS__));                   \
      }                                                                                   \
    }                                                                                     \
  } while (false)
#endif  // __COVERITY__

#ifdef NVE_THROW_
#error NVE_THROW_ was already defined.
#endif
#define NVE_THROW_(...) \
  throw nve::RuntimeError<bool>(__FILE__, __LINE__, "throw", false, nve::to_string(__VA_ARGS__))

#ifdef NVE_THROW_NOT_IMPLEMENTED_
#error NVE_THROW_NOT_IMPLEMENTED_ was already defined.
#endif
#define NVE_THROW_NOT_IMPLEMENTED_() NVE_THROW_("Not implemented yet!")

namespace nve {

/**
 * Dependent false.
 * https://en.cppreference.com/w/cpp/language/if
 */
template <typename>
inline constexpr bool dependent_false_v = false;

/**
 * Make std::to_array (C++20) available in C++17. This will only work for primitive types.
 */
template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> to_array(Args&&... args) {
  return {static_cast<T>(args)...};
}

inline std::string to_string() noexcept { return {}; }

template <typename T>
inline std::string to_string(const T& arg) {
  return (std::ostringstream{} << arg).str();
}

template <>
inline std::string to_string<const char*>(const char* const& arg) {
  return arg;
}

template <>
inline std::string to_string<std::string>(const std::string& arg) {
  return arg;
}

template <>
inline std::string to_string<std::string_view>(const std::string_view& arg) {
  return static_cast<std::string>(arg);
}

template <typename TArg0, typename TArg1>
inline std::string to_string(const TArg0& arg0, const TArg1& arg1) {
  return (std::ostringstream{} << arg0 << arg1).str();
}

template <typename TArg0, typename TArg1, typename... TArgs>
inline std::string to_string(const TArg0& arg0, const TArg1& arg1, TArgs&&... args) {
  std::ostringstream o;
  o << arg0;
  o << arg1;
  (o << ... << args);
  return o.str();
}

/**
 * @brief Returns a reference that can be used to identify the current thread.
 */
std::string& this_thread_name();

/**
 * Baseclass for all kinds of exceptions that we throw. Don't use this directly. Use the
 * `NVE_THROW_` and `NVE_CHECK_` macros instead.
 */
class Exception : public std::exception {
 public:
  Exception() = delete;

  inline Exception(const char file[], const int64_t line, const char expr[],
                   const std::string& hint = {}) noexcept
      : file_{file}, line_{line}, expr_{expr}, hint_{hint}, thread_{this_thread_name()} {
    NVE_ASSERT_(file_);
    NVE_ASSERT_(expr_);
    NVE_ASSERT_(hint_);
  }

  inline Exception(const Exception& that) noexcept
      : file_{that.file_},
        line_{that.line_},
        expr_{that.expr_},
        hint_{that.hint_},
        thread_{that.thread_} {}

  inline Exception& operator=(const Exception& that) noexcept {
    file_ = that.file_;
    line_ = that.line_;
    expr_ = that.expr_;
    hint_ = that.hint_;
    thread_ = that.thread_;
    return *this;
  }

  inline const char* file() const noexcept { return file_; }

  inline int64_t line() const noexcept { return line_; }

  inline const char* expression() const noexcept { return expr_; }

  inline const std::string& hint() const noexcept { return hint_; }

  virtual const char* what() const noexcept override {
    return hint_.empty() ? expr_ : hint_.c_str();
  }

  inline const std::string& thread_name() const noexcept { return thread_; }

  /**
   * Virtual to avoid callers needing to have type information upfront.
   */
  virtual std::string to_string() const;

 private:
  const char* file_;
  int64_t line_;
  const char* expr_;
  std::string hint_;
  std::string thread_;
};

inline std::ostream& operator<<(std::ostream& o, const Exception& e) { return o << e.to_string(); }

/**
 * Evaluates if a result indicates an error.
 *
 * @tparam ResultType The type that contains the result of an operation.
 * @returns `true` if the result indicates an errorneous condition.
 */
template <typename ResultType>
constexpr bool is_success(const ResultType&) noexcept;

/**
 * Runtime exceptions should be typed.
 *
 * @tparam ResultType The type that contains the result of an operation.
 */
template <typename ResultType>
class RuntimeError : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;
};

template <>
constexpr bool is_success(const bool& result) noexcept {
  return result;
}

template <>
class RuntimeError<bool> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[], const bool&,
                      const std::string& hint) noexcept
      : base_type(file, line, expr, hint) {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that) {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    return *this;
  }

  virtual std::string to_string() const override;
};

static std::random_device random_device;

using table_id_t = int64_t;

Logger* GetGlobalLogger();

}  // namespace nve

#ifdef NVE_LOG_
#error NVE_LOG_ was already defined.
#endif
#define NVE_LOG_(_level_, ...)                                            \
  do {                                                                    \
    std::string _msg_;                                                    \
    if constexpr (static_cast<int64_t>(_level_) <=                        \
                  static_cast<int64_t>(nve::LogLevel_t::Error)) {         \
      _msg_ = to_string('(', __FILE__, ':', __LINE__, ") ", __VA_ARGS__); \
    } else {                                                              \
      _msg_ = to_string(__VA_ARGS__);                                     \
    }                                                                     \
    nve::GetGlobalLogger()->log((_level_), (_msg_));                      \
  } while (0)

// Logging macros
#ifdef NVE_LOG_CRITICAL_
#error NVE_LOG_CRITICAL_ was already defined.
#endif
#define NVE_LOG_CRITICAL_(...) NVE_LOG_(nve::LogLevel_t::Critical, ##__VA_ARGS__)

#ifdef NVE_LOG_ERROR_
#error NVE_LOG_ERROR_ was already defined.
#endif
#define NVE_LOG_ERROR_(...) NVE_LOG_(nve::LogLevel_t::Error, ##__VA_ARGS__)

#ifdef NVE_LOG_WARNING_
#error NVE_LOG_WARNING_ was already defined.
#endif
#define NVE_LOG_WARNING_(...) NVE_LOG_(nve::LogLevel_t::Warning, ##__VA_ARGS__)

#ifdef NVE_LOG_PERF_
#error NVE_LOG_PERF_ was already defined.
#endif
#define NVE_LOG_PERF_(...) NVE_LOG_(nve::LogLevel_t::Perf, ##__VA_ARGS__)

#ifdef NVE_LOG_INFO_
#error NVE_LOG_INFO_ was already defined.
#endif
#define NVE_LOG_INFO_(...) NVE_LOG_(nve::LogLevel_t::Info, ##__VA_ARGS__)

#ifdef NVE_LOG_VERBOSE_
#error NVE_LOG_VERBOSE_ was already defined.
#endif
#define NVE_LOG_VERBOSE_(...) NVE_LOG_(nve::LogLevel_t::Verbose, ##__VA_ARGS__)

// Debug mode only
#ifdef NVE_IF_DEBUG_
#error NVE_IF_DEBUG_ was already defined.
#endif
#ifndef NDEBUG
#define NVE_IF_DEBUG_(...) do { __VA_ARGS__; } while (false)
#else
#define NVE_IF_DEBUG_(...) do {} while (false)
#endif

#ifdef NVE_DEBUG_PRINT_
#error NVE_DEBUG_PRINT_ was already defined.
#else
#define NVE_DEBUG_PRINT_(...) NVE_IF_DEBUG_(std::cout << __VA_ARGS__)
#endif

#ifdef NVE_DEBUG_PRINTF_
#error NVE_DEBUG_PRINTF_ was already defined.
#else
#define NVE_DEBUG_PRINTF_(...) NVE_IF_DEBUG_(printf(__VA_ARGS__))
#endif

#define NVE_LIKELY_(_expr_) (__builtin_expect((_expr_), 1))
#define NVE_UNLIKELY_(_expr_) (__builtin_expect((_expr_), 0))
