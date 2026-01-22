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
#include <common.hpp>
#include <cuda_ops/pipeline_gather.cuh>
#include <include/ecache/ec_set_associative.cuh>
#include <include/execution_context.hpp>
#include <default_allocator.hpp>
#include <vector>
#include <random>
#include "include/gpu_table.hpp"

namespace nve {

struct GatherFlowPipelineTestParams {
    uint64_t num_keys;
    uint64_t num_keys_to_insert;
    size_t row_size_in_bytes;
    uint64_t task_size;
    uint64_t num_streams;
    uint64_t kernel_mode_type;
    uint64_t threshold;
};

template <typename IndexT>
class GatherFlowPipelineTest : public testing::TestWithParam<GatherFlowPipelineTestParams> {
public:
    GatherFlowPipelineTest() : 
        d_keys_(nullptr), 
        d_values_(nullptr), 
        uvm_table_ptr_(nullptr)
    {
        Init();
    }

    ~GatherFlowPipelineTest() {
        Cleanup();
    }

    void test_gather_flow_pipeline() {
        const auto& params = GetParam();
        
        // Create test data
        std::vector<IndexT> h_keys(params.num_keys);
        std::vector<int8_t> h_values(params.num_keys * params.row_size_in_bytes);
        
        // Initialize keys with random values
        std::random_device rd;
        std::mt19937 gen((params.num_keys * params.num_keys_to_insert) % params.row_size_in_bytes); // Fixed seed for reproducible tests
        std::uniform_int_distribution<IndexT> key_dist(0, 9999);
        for (uint64_t i = 0; i < params.num_keys; i++) {
            h_keys[i] = key_dist(gen);
        }
        
        insert_data(h_keys);

        // Copy data to device
        NVE_CHECK_(cudaMemcpy(d_keys_, h_keys.data(), params.num_keys * sizeof(IndexT), cudaMemcpyHostToDevice));
        
        // Call the function under test
        try {
            gpu_table_->find(ctx_, params.num_keys, d_keys_, nullptr, params.row_size_in_bytes, d_values_, nullptr);
            // Synchronize to ensure completion
            NVE_CHECK_(cudaStreamSynchronize(main_stream_));

            NVE_CHECK_(cudaMemcpy(h_values.data(), d_values_, params.num_keys * params.row_size_in_bytes, cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < params.num_keys; i++) {
                for (size_t j = 0; j < params.row_size_in_bytes; j++) {
                    EXPECT_EQ(h_values[i * params.row_size_in_bytes + j], uvm_table_ptr_[h_keys[i] * params.row_size_in_bytes + j]);
                }
            }
        } catch (const std::exception& e) {
            FAIL() << "gather_flow_pipeline threw exception: " << e.what();
        }
    }

private:
    void insert_data(std::vector<IndexT>& lookup_keys) {
        const auto& params = GetParam();
        if (params.num_keys_to_insert == 0) {
            return;
        }
        std::vector<IndexT> h_keys(params.num_keys_to_insert);
        std::vector<int8_t> h_values(params.num_keys_to_insert * params.row_size_in_bytes);
        for (uint64_t i = 0; i < params.num_keys_to_insert; i++) {
            h_keys[i] = lookup_keys[i];
            for (size_t j = 0; j < params.row_size_in_bytes; j++) {
                h_values[i * params.row_size_in_bytes + j] = uvm_table_ptr_[h_keys[i] * params.row_size_in_bytes + j];
            }
        }
        NVE_CHECK_(cudaMemcpy(d_values_for_insert_, h_values.data(), params.num_keys_to_insert * params.row_size_in_bytes, cudaMemcpyHostToDevice));
        gpu_table_->insert(ctx_, params.num_keys_to_insert, h_keys.data(), params.row_size_in_bytes, params.row_size_in_bytes, d_values_for_insert_);
        NVE_CHECK_(cudaDeviceSynchronize());
    }

    void Init() {
        const auto& params = GetParam();
        params_.task_size = params.task_size;
        params_.num_aux_streams = params.num_streams;
        constexpr uint64_t num_embeddings = 1000000;

        NVE_CHECK_(cudaMalloc(&d_keys_, params.num_keys * sizeof(IndexT)));
        NVE_CHECK_(cudaMalloc(&d_values_, params.num_keys * params.row_size_in_bytes));
        NVE_CHECK_(cudaMalloc(&d_values_for_insert_, params.num_keys * params.row_size_in_bytes));
        
        // Allocate UVM table
        NVE_CHECK_(cudaMallocHost(&uvm_table_ptr_, num_embeddings * params.row_size_in_bytes));
        
        // Create main stream
        NVE_CHECK_(cudaStreamCreate(&main_stream_));

        // Initialize UVM table with some data
        for (uint64_t i = 0; i < num_embeddings; i++) {
            for (size_t j = 0; j < params.row_size_in_bytes; j++) {
                uvm_table_ptr_[i * params.row_size_in_bytes + j] = static_cast<int8_t>((i + j) % 256);
            }
        }
        
        nve::GPUTableConfig cfg;
        cfg.device_id = 0;
        cfg.cache_size = static_cast<int64_t>(16*1024);
        cfg.max_modify_size = (1l << 22);
        cfg.row_size_in_bytes = params.row_size_in_bytes;
        cfg.count_misses = true;
        
        cfg.uvm_table = uvm_table_ptr_;
        cfg.kernel_mode_type = params.kernel_mode_type;
        if (params.kernel_mode_type == 3) {
            cfg.kernel_mode_value = reinterpret_cast<uintptr_t>(&params_);
        } else if (params.kernel_mode_type == 0) {
            cfg.kernel_mode_value = params.threshold;
        } else {
            cfg.kernel_mode_value = 0;
        }
        cfg.modify_on_gpu = false;
        
        gpu_table_ = std::make_shared<nve::GpuTable<IndexT>>(cfg, nullptr /* using default allocator for device 0*/);
        ctx_ = gpu_table_->create_execution_context(main_stream_, main_stream_, nullptr, nullptr);
    }
    
    void Cleanup() {
        cudaDeviceSynchronize();
        if (d_keys_) {
            NVE_CHECK_(cudaFree(d_keys_));
            d_keys_ = nullptr;
        }
        if (d_values_) {
            NVE_CHECK_(cudaFree(d_values_));
            d_values_ = nullptr;
        }
        if (uvm_table_ptr_) {
            NVE_CHECK_(cudaFreeHost(uvm_table_ptr_));
            uvm_table_ptr_ = nullptr;
        }
        ctx_.reset();
    }

    IndexT* d_keys_;
    int8_t* d_values_;
    int8_t* d_values_for_insert_;
    int8_t* uvm_table_ptr_;
    context_ptr_t ctx_;
    std::shared_ptr<nve::GpuTable<IndexT>> gpu_table_;
    GatherKernelPipelineParams params_;
    cudaStream_t main_stream_;
};

static std::vector<int64_t> cases_n = {1, 16384, 1000};
static std::vector<size_t> cases_row_size_in_bytes = {4, 18, 32, 512};
static std::vector<uint64_t> cases_task_size = {4096};
static std::vector<uint64_t> cases_num_streams = {0, 1, 16};
static std::vector<uint32_t> cases_kernel_mode_type = {3, 0, 1, 2};
static std::vector<uint64_t> cases_num_keys_to_insert = {0, 17, 1024};
static std::vector<uint64_t> cases_threshold = {1024};

static std::vector<GatherFlowPipelineTestParams> gen_cases(const std::vector<int64_t>& _n, const std::vector<size_t>& _row_size_in_bytes, const std::vector<uint64_t>& _task_size, const std::vector<uint64_t>& _num_streams, const std::vector<uint32_t>& _kernel_mode_type, const std::vector<uint64_t>& _num_keys_to_insert, const std::vector<uint64_t>& _threshold) {
    std::vector<GatherFlowPipelineTestParams> ret;
    
    for (uint32_t n = 0; n < _n.size(); n++)
    {
        for (uint32_t row_size_in_bytes = 0; row_size_in_bytes < _row_size_in_bytes.size(); row_size_in_bytes++)
        {
            for (uint32_t task_size = 0; task_size < _task_size.size(); task_size++)
            {
                for (uint32_t num_streams = 0; num_streams < _num_streams.size(); num_streams++)
                {
                    for (uint32_t kernel_mode_type = 0; kernel_mode_type < _kernel_mode_type.size(); kernel_mode_type++)
                    {
                        for (uint32_t num_keys_to_insert = 0; num_keys_to_insert < _num_keys_to_insert.size(); num_keys_to_insert++)
                        {
                            for (uint32_t threshold = 0; threshold < _threshold.size(); threshold++)
                            {
                                GatherFlowPipelineTestParams params;
                                params.num_keys = _n[n];
                                params.row_size_in_bytes = _row_size_in_bytes[row_size_in_bytes];
                                params.task_size = _task_size[task_size];
                                params.num_streams = _num_streams[num_streams];
                                params.kernel_mode_type = _kernel_mode_type[kernel_mode_type];
                                params.num_keys_to_insert = std::min(_num_keys_to_insert[num_keys_to_insert], params.num_keys);
                                params.threshold = _threshold[threshold];
                                ret.push_back(params);
                            }
                        }
                    }
                }
            }
        }
    }
    return ret;
}

using GatherFlowPipelineTest_INT32 = GatherFlowPipelineTest<int32_t>;
using GatherFlowPipelineTest_INT64 = GatherFlowPipelineTest<int64_t>;

#define TEST_FORMAT(test_name, test_func) \
TEST_P(GatherFlowPipelineTest_INT32, test_name) \
{ \
    test_func(); \
} \
TEST_P(GatherFlowPipelineTest_INT64, test_name) \
{ \
    test_func(); \
}

TEST_FORMAT(gather_flow_pipeline, test_gather_flow_pipeline);
#undef TEST_FORMAT

// Instantiate the tests with both type and value parameters
INSTANTIATE_TEST_SUITE_P(
    GatherFlowPipelineTestInt32,
    GatherFlowPipelineTest_INT32,
    testing::ValuesIn(gen_cases(cases_n, cases_row_size_in_bytes, cases_task_size, cases_num_streams, cases_kernel_mode_type, cases_num_keys_to_insert, cases_threshold)));

INSTANTIATE_TEST_SUITE_P(
    GatherFlowPipelineTestInt64,
    GatherFlowPipelineTest_INT64,
    testing::ValuesIn(gen_cases(cases_n, cases_row_size_in_bytes, cases_task_size, cases_num_streams, cases_kernel_mode_type, cases_num_keys_to_insert, cases_threshold)));

} // namespace nve /*
