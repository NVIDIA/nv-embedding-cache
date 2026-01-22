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
#include <type_traits>

namespace nve {

/**
 * Ceiling division.
 */
template<typename T>
static constexpr T ceil_div(const T x, const T n) noexcept {
  static_assert(std::is_integral_v<T>);
  return (x + n - 1) / n;
}

#if __cplusplus >= 202002L

using std::has_single_bit;
using std::countr_zero;
using std::popcount;
using std::rotl;
using std::rotr;
using std::bit_ceil;

#else // __cplusplus >= 202002L

template<typename T>
static constexpr int64_t num_bits{sizeof(T) * 8};

/**
 * Reimplementation of C++20 std::has_single_bit that can handle signed types.
 *
 * See also:
 * https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
 *
 * Note:
 * Alternative (potentially faster) could use `__builtin_popcount`.
 */
template <typename T>
static constexpr bool has_single_bit(const T x) noexcept {
  static_assert(std::is_integral_v<T>);
  return x > 0 && !(x & (x - 1));
}

/**
 * Reimplementations of C++20 std::countr_zero.
 *
 * Note: These builtin's are available in GCC and Clang.
 */
static constexpr int countr_zero(const unsigned char x) noexcept { return __builtin_ctz(x); }
static constexpr int countr_zero(const unsigned short x) noexcept { return __builtin_ctz(x); }
static constexpr int countr_zero(const unsigned int x) noexcept { return __builtin_ctz(x); }
static constexpr int countr_zero(const unsigned long x) noexcept { return __builtin_ctzl(x); }
static constexpr int countr_zero(const unsigned long long x) noexcept { return __builtin_ctzll(x); }

/**
 * Reimplementations of C++20 std::popcount.
 *
 * Note: These builtin's are available in GCC and Clang.
 */
static constexpr int popcount(const unsigned char x) noexcept { return __builtin_popcount(x); }
static constexpr int popcount(const unsigned short x) noexcept { return __builtin_popcount(x); }
static constexpr int popcount(const unsigned int x) noexcept { return __builtin_popcount(x); }
static constexpr int popcount(const unsigned long x) noexcept { return __builtin_popcountl(x); }
static constexpr int popcount(const unsigned long long x) noexcept { return __builtin_popcountll(x); }

/**
 * Reimplementations of C++20 std::rotl.
 */
template <typename T>
static constexpr T rotl(const T x, int64_t n) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) >= sizeof(uint32_t));
  NVE_ASSERT_(n >= 0);
  n &= num_bits<T> - 1;
  return (x << n) | (x >> (num_bits<T> - n));
}

/**
 * Reimplementations of C++20 std::rotr.
 */
template <typename T>
static constexpr T rotr(const T x, int64_t n) noexcept {
  static_assert(std::is_unsigned_v<T> && sizeof(T) >= sizeof(uint32_t));
  NVE_ASSERT_(n >= 0);
  n &= num_bits<T> - 1;
  return (x >> n) | (x << (num_bits<T> - n));
}

/**
 * Returns the next power of two that is sufficient to contain the provided value.
 */
template <typename T>
static constexpr T bit_ceil(T x) noexcept {
  static_assert(std::is_unsigned_v<T>);
  // Based on https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2.
  --x;
  for (size_t n{1}; n <= sizeof(T) * 4; n *= 2) {
    x |= x >> n;  // Replicate highest bit repeatedly.
  }
  ++x;
  return std::max<T>(x, 1);
}

#endif // __cplusplus >= 202002L

/**
 * Returns the next value that is aligned at n.
 */
template <typename T>
static constexpr T next_aligned(const T x, const T n) noexcept {
  NVE_ASSERT_(n > 0);
  return ceil_div(x, n) * n;
}

/**
 * Returns the next value that is aligned at N.
 */
template <int64_t N, typename T>
static constexpr T next_aligned(const T x) noexcept {
  NVE_ASSERT_(N >= 0);
  if constexpr (has_single_bit(N)) {
    static_assert(std::is_integral_v<T>);
    return (x + N - 1) & ~(N - 1);
  } else {
    return next_aligned(x, N);
  }
}

/**
 * Iterable bitmask from the nvHashMap library.
 */
template <typename Repr>
struct generic_bitmask final {
  using repr_type = Repr;
  static_assert(std::is_unsigned_v<repr_type>);
  static constexpr int64_t size{sizeof(repr_type)};
  static_assert(has_single_bit(size));
  static_assert(size == alignof(repr_type));
  static constexpr int64_t size_mask{size - 1};
  
  static constexpr int64_t num_bits{size * 8};
  static_assert(num_bits >= 0 && num_bits <= INT32_MAX);
  static constexpr int64_t num_bits_mask{num_bits - 1};

  static constexpr int64_t mask_size(const int64_t n) noexcept {
    NVE_ASSERT_(n >= 0);
    return ceil_div(n, num_bits);
  }

  static constexpr repr_type load(const void* const __restrict mem) noexcept {
    NVE_ASSERT_(mem % size == 0);
    return *reinterpret_cast<const repr_type*>(mem);
  }

  static constexpr repr_type load(const char* const __restrict mem, const int64_t ij) noexcept {
    NVE_ASSERT_(mem % size == 0);
    NVE_ASSERT_(ij >= 0);
    return load(&mem[ij / num_bits * size]);
  }

  static constexpr void store(void* const __restrict mem, const repr_type repr) noexcept {
    NVE_ASSERT_(mem % size == 0);
    *reinterpret_cast<repr_type*>(mem) = repr;
  }

  static constexpr void store(char* const __restrict mem, const int64_t ij,
                              const repr_type repr) noexcept {
    NVE_ASSERT_(mem % size == 0);
    NVE_ASSERT_(ij >= 0);
    store(&mem[ij / num_bits * size], repr);
  }

  static constexpr int count(const repr_type repr) noexcept { return popcount(repr); }

  static constexpr bool has_next(const repr_type repr) noexcept { return repr != 0; }

  static constexpr int next(const repr_type repr) noexcept { return countr_zero(repr); }

  static constexpr repr_type skip(const repr_type repr) noexcept { return repr & (repr - 1); }

  static constexpr repr_type invert(const repr_type repr) noexcept { return ~repr; }

  static constexpr repr_type join(const repr_type repr0, const repr_type repr1) noexcept {
    return repr0 | repr1;
  }

  static constexpr repr_type inters(const repr_type repr0, const repr_type repr1) noexcept {
    return repr0 & repr1;
  }

  static constexpr repr_type diff(const repr_type repr0, const repr_type repr1) noexcept {
    return repr0 ^ repr1;
  }

  static constexpr bool equals(const repr_type repr0, const repr_type repr1) noexcept {
    return repr0 == repr1;
  }

  static constexpr repr_type empty() noexcept { return {}; }

  static constexpr repr_type full() noexcept { return invert(empty()); }

  static constexpr bool is_empty(const repr_type repr) noexcept { return equals(repr, empty()); }

  static constexpr bool is_full(const repr_type repr) noexcept { return equals(repr, full()); }

  static constexpr repr_type single(const int64_t n) noexcept {
    NVE_ASSERT_(n >= 0 && n < num_bits);
    return repr_type{1} << n;
  }

  static constexpr repr_type partial(const int64_t n) noexcept {
    NVE_ASSERT_(n >= 0 && n < num_bits);
    return invert(full() << n);
  }
  
  static constexpr repr_type set(const repr_type repr, const int64_t n) noexcept {
    return join(repr, single(n));
  }

  static constexpr repr_type insert(const repr_type repr, const int64_t n) noexcept {
    return join(repr, single(n));
  }

  static constexpr repr_type insert(const repr_type repr, const int64_t n, const bool bit) noexcept {
    NVE_ASSERT_(n >= 0 && n < num_bits);
    return join(repr, static_cast<repr_type>(bit) << n);
  }

  static constexpr repr_type clip(const repr_type repr, const int64_t n) noexcept {
    return (n < num_bits) ? inters(repr, partial(n)) : repr;
  }

  static constexpr void atomic_join(void* const __restrict mem, const repr_type repr) noexcept {
    NVE_ASSERT_(mem % size == 0);
    repr_type* const dst{reinterpret_cast<repr_type*>(mem)};
    __atomic_or_fetch(dst, repr, __ATOMIC_RELAXED);
  }

  static constexpr void atomic_join(char* const __restrict mem, const int64_t ij,
                                    const repr_type repr) noexcept {
    NVE_ASSERT_(mem % size == 0);
    NVE_ASSERT_(ij >= 0);
    atomic_join(&mem[ij / num_bits * size], repr);
  }
};

using bitmask8_t = generic_bitmask<uint8_t>;
using bitmask16_t = generic_bitmask<uint16_t>;
using bitmask32_t = generic_bitmask<uint32_t>;
using bitmask64_t = generic_bitmask<uint64_t>;

using max_bitmask_t = bitmask64_t;  // TODO: Should have a bitmask128_t for SSE, NEON and SVE support.
using max_bitmask_repr_t = typename bitmask64_t::repr_type;

static constexpr int64_t cpu_cache_line_size{NVE_CACHE_LINE_SIZE};
static_assert(cpu_cache_line_size >= max_bitmask_t::size && has_single_bit(static_cast<uint64_t>(cpu_cache_line_size)));

static inline void l1_read_prefetch(const char* const __restrict r, const int64_t n) noexcept {
  for (int64_t i{}; i < n; i += cpu_cache_line_size) {
    __builtin_prefetch(&r[i], 0, 3);
  }
}

static inline void l1_write_prefetch(char* const __restrict w, const int64_t n) noexcept {
  for (int64_t i{}; i < n; i += cpu_cache_line_size) {
    __builtin_prefetch(&w[i], 1, 3);
  }
}

static inline void l1_prefetch(const char* const __restrict r, char* const __restrict w, const int64_t n) noexcept {
  for (int64_t i{}; i < n; i += cpu_cache_line_size) {
    __builtin_prefetch(&r[i], 0, 3);
    __builtin_prefetch(&w[i], 1, 3);
  }
}

}  // namespace nve
