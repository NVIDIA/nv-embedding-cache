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

// Thread pool tests for the C API

#include <gtest/gtest.h>
#include <nve_c_api.h>

#include "c_api_ptr.hpp"
#include "test_utils.hpp"

// ---------------------------------------------------------------------------------
// Positive lifecycle: create with valid JSON -> num_workers reports the configured
// count -> destroy succeeds.
// ---------------------------------------------------------------------------------
TEST(ThreadPool, CreateNumWorkersDestroyLifecycle) {
  nve_thread_pool_t pool = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_thread_pool_create(&pool, "{\"name\": \"simple\", \"num_workers\": 3}"));
  ASSERT_NE(nullptr, pool);

  int64_t num_workers = -1;
  EXPECT_EQ(NVE_SUCCESS, nve_thread_pool_num_workers(pool, &num_workers));
  EXPECT_EQ(3, num_workers);

  EXPECT_EQ(NVE_SUCCESS, nve_thread_pool_destroy(pool));
}

// ---------------------------------------------------------------------------------
// nve_thread_pool_create with malformed JSON: the parse exception must be caught
// and converted to an error status.
// ---------------------------------------------------------------------------------
TEST(ThreadPool, CreateWithMalformedJsonDoesNotThrow) {
  nve_thread_pool_t pool = nullptr;
  nve_status_t status = NVE_SUCCESS;
  ASSERT_NO_THROW({ status = nve_thread_pool_create(&pool, "{not valid json"); });
  EXPECT_NE(NVE_SUCCESS, status);
}

// ---------------------------------------------------------------------------------
// nve_thread_pool_num_workers with pool == NULL, or a valid pool but out == NULL
// ---------------------------------------------------------------------------------
TEST(ThreadPool, NumWorkersNullPool) {
  int64_t num_workers = -1;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_thread_pool_num_workers(nullptr, &num_workers));
}
TEST(ThreadPool, NumWorkersNullOut) {
  nve_thread_pool_t pool = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_thread_pool_create(&pool, "{\"name\": \"simple\", \"num_workers\": 1}"));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_thread_pool_num_workers(pool, nullptr));
  nve_thread_pool_destroy(pool);
}

// Note: nve_configure_default_thread_pool() only stores the JSON for later use
// (src/thread_pool.cpp); it does not eagerly validate or construct a pool, so
// there is no observable-from-here "malformed config rejected immediately" behavior
// to test independently of actually materializing the pool. Testing that here with
// an invalid config would also permanently poison the process-wide default-pool
// config for every other test in this binary that lazily uses the default pool
// (thread_pool=NULL) afterward -- so it's deliberately not attempted.

// ---------------------------------------------------------------------------------
// nve_configure_default_thread_pool is documented as "must be called before the
// default pool is first used". the underlying implementation
// enforces this by throwing once the default pool has actually been instantiated.
// Rather than relying on process-wide test-order luck to determine
// whether the default pool is already in use by the time this test runs, this test
// forces it into the "already used" state itself (creating any table/context with
// thread_pool=NULL lazily materializes the default pool), so the subsequent
// configure call is deterministically expected to fail regardless of what ran
// before it in this binary.
// ---------------------------------------------------------------------------------
TEST(ThreadPool, ConfigureDefaultPoolAfterUseIsRejected) {
  nve_table_factory_t factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));
  nve::TableFactoryPtr factory_owner(factory);
  nve_table_t table = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_table_factory_produce(factory, 0, "{}", &table));
  nve::TablePtr table_owner(table);
  nve_context_t ctx = nullptr;
  // thread_pool=NULL materializes (and thereafter locks) the process-wide default pool.
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_create_execution_context(table, &ctx, nullptr, nullptr, nullptr, nullptr));
  nve::ContextPtr ctx_owner(ctx);

  EXPECT_NE(NVE_SUCCESS,
           nve_configure_default_thread_pool("{\"name\": \"simple\", \"num_workers\": 2}"));
}
