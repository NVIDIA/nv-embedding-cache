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

#include <bit>
#include <common.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace nve {

#if __cplusplus >= 202002L

using std::countl_zero;
using std::countr_zero;
using std::popcount;
using std::has_single_bit;
using std::rotl;
using std::rotr;
using std::bit_ceil;

#else // __cplusplus >= 202002L

/**
 * Reimplementations of C++20 std::countl_zero.
 */
template <typename T>
constexpr int countl_zero(T x) noexcept {
  static_assert(std::is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

#if defined(__has_builtin) && __has_builtin(__builtin_clzg)
  return __builtin_clzg(x, n);
#else
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    constexpr int n32{sizeof(uint32_t) * 8};
    return x ? __builtin_clz(x) - (n32 - n) : n;
  } else {
    return x ? __builtin_clzll(x) : n;
  }
#endif
}

/**
 * Reimplementations of C++20 std::countr_zero.
 */
template <typename T>
constexpr int countr_zero(T x) noexcept {
  static_assert(std::is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

#if defined(__has_builtin) && __has_builtin(__builtin_ctzg)
  return __builtin_ctzg(x, n);
#else
  if constexpr (sizeof(T) <= sizeof(uint16_t)) {
    return __builtin_ctz(x | (UINT32_C(1) << n));
  } else if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return x ? __builtin_ctz(x) : n;
  } else {
    return x ? __builtin_ctzll(x) : n;
  }
#endif
}

/**
 * Reimplementations of C++20 std::popcount.
 */
template <typename T>
constexpr int popcount(T x) noexcept {
  static_assert(std::is_unsigned_v<T>);

#if defined(__has_builtin) && __has_builtin(__builtin_popcountg)
  return __builtin_popcountg(x);
#else
  if constexpr (sizeof(T) <= sizeof(uint32_t)) {
    return __builtin_popcount(x);
  } else {
    return __builtin_popcountll(x);
  }
#endif
}

/**
 * Reimplementation of C++20 std::has_single_bit.
 */
template <typename T>
constexpr bool has_single_bit(T x) noexcept {
  static_assert(std::is_unsigned_v<T>);

#if defined(__x86_64__) && defined(__POPCNT__)
  return popcount(x) == 1;
#else
  return (x != 0) && !(x & (x - 1));
#endif
}

/**
 * Reimplementations of C++20 std::rotl.
 */
template <typename T>
constexpr T rotl(T x, int s) noexcept {
  static_assert(std::is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

  const auto y{(x << (s & (n - 1))) | (x >> (-s & (n - 1)))};
  return static_cast<T>(y);
}

/**
 * Reimplementations of C++20 std::rotr.
 */
template <typename T>
constexpr T rotr(T x, int s) noexcept {
  static_assert(std::is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

  const auto y{(x >> (s & (n - 1))) | (x << (-s & (n - 1)))};
  return static_cast<T>(y);
}

/**
 * Reimplementations of C++20 std::bit_ceil.
 */
template <typename T>
constexpr T bit_ceil(T x) noexcept {
  static_assert(std::is_unsigned_v<T>);
  constexpr int n{sizeof(T) * 8};

  T y{1};
  if (x > 1) {
    y = static_cast<T>(y << (n - countl_zero(--x)));
  }
  return y;
}

#endif // __cplusplus >= 202002L

/**
 * Convert a signed integer to an unsigned integer of the same width.
 * Will perform a range check in debug builds.
 */
template <typename T>
constexpr std::make_unsigned_t<T> to_uint(T x) noexcept {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T>);
  NVE_ASSERT_(x >= 0);
  return static_cast<std::make_unsigned_t<T>>(x);
}

// Multiply sizes, failing instead of silently wrapping on overflow.
template <typename T>
T checked_mul(T a, T b, const char* what) {
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool>);
  NVE_CHECK_(b == 0 || a <= std::numeric_limits<T>::max() / b,
             what, " computation overflows result type");
  return static_cast<T>(a * b);
}

/**
 * Ceiling division.
 */
template<typename T>
constexpr T ceil_div(T x, T n) noexcept {
  static_assert(std::is_integral_v<T>);
  NVE_ASSERT_(n > 0);
  return (x + n - 1) / n;
}

/**
 * Round up a value using a given base.
 */
template <typename T>
constexpr T round_up(T x, T n) noexcept { return ceil_div(x, n) * n; }

using bitmask64_t = uint64_t;
static_assert(std::is_unsigned_v<bitmask64_t> && sizeof(bitmask64_t) == 8);

/**
 * Iterable bitmask from the nvHashMap library.
 */
struct bitmask64 final {
  static constexpr int64_t size{sizeof(bitmask64_t)};
  static_assert(has_single_bit(to_uint(size)));
  static_assert(size == alignof(bitmask64_t));
  
  static constexpr int64_t num_bits{size * 8};
  static constexpr int64_t num_bits_mask{num_bits - 1};

  static constexpr bitmask64_t empty{};
  static constexpr bitmask64_t full{~empty};
  static constexpr bitmask64_t single(int64_t i) noexcept {
    NVE_ASSERT_(i >= 0 && i < num_bits);
    return bitmask64_t{1} << i;
  }

  static constexpr int count(bitmask64_t m) noexcept { return popcount(m); }
  static constexpr int next(bitmask64_t m) noexcept { return countr_zero(m); }

  static constexpr bool get(bitmask64_t m, int64_t i) noexcept {
    NVE_ASSERT_(i >= 0 && i < num_bits);
    return (m & single(i)) != 0;
  }

  static constexpr bitmask64_t skip(bitmask64_t m) noexcept { return m & (m - 1); }

  static constexpr bitmask64_t until(int64_t i) noexcept {
    NVE_ASSERT_(i >= 0);
    if (i < num_bits) {
      return ~(full << i);
    } else {
      return full;
    }
  }

  static constexpr bitmask64_t clip(bitmask64_t m, int64_t n) noexcept { return m & until(n); }

  static inline bitmask64_t atomic_load(bitmask64_t* mem) noexcept {
    NVE_ASSERT_(reinterpret_cast<uintptr_t>(mem) % size == 0);
    return __atomic_load_n(mem, __ATOMIC_RELAXED);
  }

  static inline bitmask64_t atomic_merge(bitmask64_t* mem, bitmask64_t h) noexcept {
    NVE_ASSERT_(reinterpret_cast<uintptr_t>(mem) % size == 0);
    return __atomic_or_fetch(mem, h, __ATOMIC_RELAXED);
  }
};

static constexpr int64_t cpu_cache_line_size{NVE_CACHE_LINE_SIZE};
static_assert(cpu_cache_line_size >= bitmask64::size && has_single_bit(to_uint(cpu_cache_line_size)));

template<typename Byte>
inline void l1_read_prefetch(const Byte* const __restrict r, const int64_t n) noexcept {
  static_assert(sizeof(Byte) == 1);

  for (int64_t i{}; i < n; i += cpu_cache_line_size) {
    __builtin_prefetch(&r[i], 0, 3);
  }
}

template<typename Byte>
inline void l1_write_prefetch(Byte* const __restrict w, const int64_t n) noexcept {
  static_assert(sizeof(Byte) == 1);

  for (int64_t i{}; i < n; i += cpu_cache_line_size) {
    __builtin_prefetch(&w[i], 1, 3);
  }
}

template<typename Byte>
inline void l1_prefetch(const Byte* const __restrict r, Byte* const __restrict w, const int64_t n) noexcept {
  static_assert(sizeof(Byte) == 1);

  for (int64_t i{}; i < n; i += cpu_cache_line_size) {
    __builtin_prefetch(&r[i], 0, 3);
    __builtin_prefetch(&w[i], 1, 3);
  }
}

}  // namespace nve
