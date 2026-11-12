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

#include <gtest/gtest.h>

#include <bit_ops.hpp>

namespace nve {

namespace {

template <typename T>
void test_checked_mul() {
  constexpr T max{std::numeric_limits<T>::max()};
  static_assert(std::is_same_v<decltype(checked_mul(T{}, T{}, "test")), T>);
  EXPECT_EQ(checked_mul(T{6}, T{7}, "test"), T{42});
  EXPECT_EQ(checked_mul(T{0}, max, "test"), T{0});
  EXPECT_EQ(checked_mul(max, T{0}, "test"), T{0});
  EXPECT_EQ(checked_mul(max, T{1}, "test"), max);
  EXPECT_EQ(checked_mul(static_cast<T>(max / 2), T{2}, "test"), max - 1);
  EXPECT_THROW(checked_mul(max, T{2}, "test"), Exception);
  EXPECT_THROW(checked_mul(T{2}, max, "test"), Exception);
}

}  // namespace

TEST(bit_ops_test, checked_mul) {
  test_checked_mul<uint8_t>();
  test_checked_mul<uint16_t>();
  test_checked_mul<uint32_t>();
  test_checked_mul<uint64_t>();
  test_checked_mul<size_t>();
}

TEST(bit_ops_test, bitmask64_count) {
  EXPECT_EQ(bitmask64::count(bitmask64::empty), 0);
  EXPECT_EQ(bitmask64::count(bitmask64::full), bitmask64::num_bits);
}

TEST(bit_ops_test, bitmask64_set_get_next) {
  for (int64_t i{}; i < bitmask64::num_bits; ++i) {
    bitmask64_t m{bitmask64::single(i)};

    EXPECT_EQ(bitmask64::count(m), 1);
    for (int64_t j{}; j < bitmask64::num_bits; ++j) {
      EXPECT_EQ(bitmask64::get(m, j), j == i);
    }
    EXPECT_EQ(bitmask64::next(m), i);
  }
}

TEST(bit_ops_test, bitmask64_skip) {
  bitmask64_t m{bitmask64::full};
  for (int64_t i{}; i < bitmask64::num_bits; ++i) {
    EXPECT_TRUE(bitmask64::get(m, i));
    m = bitmask64::skip(m);
    EXPECT_FALSE(bitmask64::get(m, i));
  }
  EXPECT_EQ(m, bitmask64::empty);
}

TEST(bit_ops_test, bitmask64_until) {
  for (int64_t n{}; n < 1024; ++n) {
    bitmask64_t m{bitmask64::until(n)};
    if (n < bitmask64::num_bits) {
      EXPECT_EQ(bitmask64::count(m), n);
    } else {
      EXPECT_EQ(bitmask64::count(m), bitmask64::num_bits);
    }
  }
}

}  // namespace nve
