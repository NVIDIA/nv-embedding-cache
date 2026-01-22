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
#include <memory>
#include <unordered_set>
#include "cache_wrapper.h"
#include "cuda_ops/cuda_common.h"
#include "cache_taskqueue.h"
#include "cache_metrics.h"
#include "mock_cache.hpp"
#include "communicator.h"

#include <samples/common/datagen.h>
using namespace nve;

template<typename IndexT>
void GenerateInput(
    uint64_t num_rows,
    uint64_t hotness,
    uint64_t batch_size,
    float alpha,
    int device_id,
    std::vector<IndexT>& input,
    std::vector<IndexT>& input_unique)
{
    ScopedDevice scope_device(device_id);
    assert((input.size() == 0) && (input_unique.size() ==0)); // If input/input_unique aren't empty, need to cudaHostUnregister before clearing.
    static std::atomic<size_t> seed = 1337;
    auto inputGenerator = getSampleGenerator<IndexT>(alpha, static_cast<IndexT>(num_rows), static_cast<uint32_t>(hotness), seed++);
    for (size_t b=0 ; b<batch_size ; b++)
    {
        auto sample = inputGenerator->getCategoryIndices();
        input.insert(input.end(), sample.begin(), sample.end());
    }
    NVE_CHECK_(cudaHostRegister(input.data(), input.size() * sizeof(IndexT),cudaHostRegisterDefault));

    std::unordered_set<IndexT> unique_set;
    for (auto i : input)
    {
        unique_set.insert(i);
    }
    input_unique.assign(unique_set.begin(), unique_set.end());
    NVE_CHECK_(cudaHostRegister(input_unique.data(), input_unique.size() * sizeof(IndexT),cudaHostRegisterDefault));
}


template<typename IndexT>
class TrainingWave {
public:
    TrainingWave(
        std::vector<std::shared_ptr<CacheTaskQueue<IndexT>>>& task_queues,
        uint64_t num_inputs,
        uint64_t num_rows,
        uint64_t hotness,
        uint64_t batch_size,
        float alpha,
        int device_id,
        bool host_inputs,
        bool dedup,
        uint64_t num_iterations,
        std::shared_ptr<MockGPUWork> gpu_fwd_bwd,
        std::shared_ptr<Communicator> comm) 
        : m_task_queues(task_queues), m_num_inputs(num_inputs), m_device_id(device_id), m_host_inputs(host_inputs), m_dedup(dedup), m_gpu_fwd_bwd(gpu_fwd_bwd), m_communicator(comm)
    {
        
        for (auto& t : m_task_queues) {
            auto c = t->GetCache();
            m_data.emplace_back(std::make_shared<TableData>(c, num_inputs, num_rows, hotness, batch_size, alpha, m_device_id, num_iterations));
        }

        // Create CUDA events needed to sync GPU FWD+BWD work
        const auto num_ctq = m_task_queues.size();
        for (size_t i=0 ; i<num_ctq ; i++) {
            cudaEvent_t e;
            NVE_CHECK_(cudaEventCreate(&e));
            m_ctq_work_done.push_back(e);
        }
        NVE_CHECK_(cudaEventCreate(&m_gpu_work_done));
        NVE_CHECK_(cudaStreamCreate(&m_gpu_work_stream));

        // Collect task queues streams for synchronizing with GPU BWD+FWD
        for (auto& t : m_task_queues) {
            auto c = t->GetCache();
            m_lookup_streams.push_back(c->GetLookupStreams()[0]);
        }
    }
    ~TrainingWave()
    {
        for (auto& e : m_ctq_work_done) {
            NVE_CHECK_(cudaEventDestroy(e));
        }
        NVE_CHECK_(cudaEventDestroy(m_gpu_work_done));
        NVE_CHECK_(cudaStreamDestroy(m_gpu_work_stream));
    }

    void ProcessWave(uint64_t iterations, bool do_pooling, bool collect_metrics)
    {
        ScopedDevice scope_device(m_device_id);
        for (uint64_t i=0 ; i<iterations ; i++)
        {
            auto curr_input = i % m_num_inputs;

            // Launch lookups for all embeddings
            std::vector<std::future<void>> lookup_futures;
            for (size_t j=0 ; j<m_task_queues.size() ; j++)
            {
                auto& ctq = m_task_queues.at(j);
                auto& data = m_data.at(j);
                auto& h_input = data->m_host_inputs.at(curr_input);
                auto& d_input = data->m_d_inputs.at(curr_input);
                if (!m_dedup)
                {
                    lookup_futures.emplace_back(ctq->Lookup(
                    d_input,
                    h_input.data(),
                    data->m_batch_size,
                    data->m_hotness,
                    data->m_d_output,
                    data->m_h_output,
                    data->m_d_hitmask,
                    data->m_h_hitmask,
                    do_pooling ? data->m_d_pooling_output : nullptr,
                    m_host_inputs,
                    collect_metrics ? &(data->m_metrics.at(i)): nullptr));
                }
                else
                {
                    lookup_futures.emplace_back(ctq->LookupDeduped(
                                d_input,
                                h_input.data(),
                                data->m_batch_size,
                                data->m_hotness,
                                data->m_d_unique_keys,
                                data->m_h_unique_keys,
                                data->m_d_counts,
                                data->m_h_counts,
                                data->m_d_inverse_buffer,
                                data->m_d_offsets,
                                data->m_h_num_runs_out,
                                data->m_d_output,
                                data->m_h_output,
                                data->m_d_hitmask,
                                data->m_h_hitmask,
                                do_pooling ? data->m_d_pooling_output : nullptr,
                                m_host_inputs,
                                collect_metrics ? &(data->m_metrics.at(i)): nullptr));
                }
            }
            // Wait for all lookups to finish
            for (auto& f : lookup_futures)
            {
                f.wait();  // when this returns all needed kernels were launched
                           // so we can launch rest of workload kernels (with appropriate stream depenecies)
            }

            
            // perform all 2 all data communiation across ranks
            if (m_communicator)
            {
                for (size_t j=0 ; j<m_task_queues.size() ; j++)
                {
                    auto& ctq = m_task_queues.at(j);
                    auto& data = m_data.at(j);
                    uint64_t buffer_size = (data->m_batch_size*ctq->GetCache()->GetConfig().embedWidth * (do_pooling ? 1 : data->m_hotness))/m_communicator->GetNumRanks();
                    if (collect_metrics) {
                        auto& metrics =  data->m_metrics.at(i); // Recording this on the metrics for the first ctq
                        NVE_CHECK_(cudaEventRecord(metrics.all_to_all.d_start, m_lookup_streams[j]));
                        
                    }
                    m_communicator->AllToAll(do_pooling ? data->m_d_pooling_output : data->m_d_output, 
                                             data->m_d_all_buffer, 
                                             buffer_size, 
                                             m_lookup_streams[j]);
                    if (collect_metrics) {
                        auto& metrics =  data->m_metrics.at(i); // Recording this on the metrics for the first ctq
                        NVE_CHECK_(cudaEventRecord(metrics.all_to_all.d_end, m_lookup_streams[j]));
                    }
                }
            }
            

            // Run the rest of the workload forward pass
            // Run backward pass and calculate gradients for embeddings
            if (m_gpu_fwd_bwd->m_latency_ms > 0)
            {
                // Set depenedencies on all lookup streams (need to wait for scatter/pooling to end)
                const size_t num_queues = m_task_queues.size();
                for (size_t q=0 ; q<num_queues ; q++) {
                    NVE_CHECK_(cudaEventRecord(m_ctq_work_done[q], m_lookup_streams[q]));
                    NVE_CHECK_(cudaStreamWaitEvent(m_gpu_work_stream, m_ctq_work_done[q]));
                }

                // Run FWD+BWD gpu work
                if (collect_metrics) {
                    auto& metrics =  m_data.at(0)->m_metrics.at(i); // Recording this on the metrics for the first ctq
                    NVE_CHECK_(cudaEventRecord(metrics.gpu_fwd_bwd.d_start, m_gpu_work_stream));
                }
                m_gpu_fwd_bwd->Run(m_gpu_work_stream);
                if (collect_metrics) {
                    auto& metrics =  m_data.at(0)->m_metrics.at(i); // Recording this on the metrics for the first ctq
                    NVE_CHECK_(cudaEventRecord(metrics.gpu_fwd_bwd.d_end, m_gpu_work_stream));
                    metrics.gpu_work_recorded = true;
                }

                // Set dependencies before continuing with dedup/accumulate
                NVE_CHECK_(cudaEventRecord(m_gpu_work_done, m_gpu_work_stream));
                for (size_t q=0 ; q<num_queues ; q++) {
                    NVE_CHECK_(cudaStreamWaitEvent(m_lookup_streams[q], m_gpu_work_done));
                }
                // Relying on the fact that all gpu caches are working in single stream mode
                // Otherwise, the post gpu work event may need to sync on the modify streams (depending on m_dedup)
            }

            // Run gradients dedup
            if (m_dedup)
            {
                std::vector<std::future<void>> gradient_dedup_futures;
                for (size_t j=0 ; j<m_task_queues.size() ; j++)
                {
                    auto& ctq = m_task_queues.at(j);
                    auto& data = m_data.at(j);
                    gradient_dedup_futures.emplace_back(ctq->GradientsDedup(
                        static_cast<uint64_t>(data->m_h_num_runs_out[0]),
                        data->m_d_output,
                        data->m_d_accumulate,
                        data->m_d_unique_keys,
                        data->m_d_counts,
                        data->m_d_offsets,
                        data->m_d_inverse_buffer,
                        collect_metrics ? &(data->m_metrics.at(i)): nullptr));
                }
                // Wait for all lookups to finish
                for (auto& f : gradient_dedup_futures)
                {
                    f.wait();  // when this returns all needed kernels were launched
                }
            }

            // Launch Accumulate for all embeddings
            std::vector<std::future<void>> accumulate_futures;
            for (size_t j=0 ; j<m_task_queues.size() ; j++)
            {
                auto& ctq = m_task_queues.at(j);
                auto& data = m_data.at(j);
                auto& acc_input = data->m_host_inputs_unique[curr_input];
                auto lookup_stream = ctq->GetCache()->GetLookupStreams().at(0);
                const auto row_size = ctq->GetCache()->GetConfig().embedWidth;

                // Copying gradients from device to host
                NVE_CHECK_(cudaMemcpyAsync(data->m_h_accumulate, data->m_d_accumulate, acc_input.size() * row_size, cudaMemcpyDefault, lookup_stream));

                // We assume gradients are already on gpu (using m_d_accumulate as storage), and therefore no copy from host is needed here.
                // Copying the unique keys set to sync between gpu and host
                if (m_dedup)
                {
                    acc_input.resize(static_cast<size_t>(data->m_h_num_runs_out[0]));
                    NVE_CHECK_(cudaMemcpyAsync(acc_input.data(), data->m_d_unique_keys, acc_input.size() * sizeof(IndexT), cudaMemcpyDefault, lookup_stream));   
                } else {
                    NVE_CHECK_(cudaMemcpyAsync(data->m_d_unique_keys, acc_input.data(), acc_input.size() * sizeof(IndexT), cudaMemcpyDefault, lookup_stream));
                }

                NVE_CHECK_(cudaStreamSynchronize(lookup_stream));

                accumulate_futures.emplace_back(ctq->Accumulate(
                    data->m_d_unique_keys, acc_input.size(), data->m_d_accumulate, nve::DATATYPE_FP16,
                    acc_input.data(), data->m_h_accumulate,
                    collect_metrics ? &(data->m_metrics.at(i)): nullptr
                ));
            }
            
            // Wait for all accumulates to finish launching (not technically mandatory in single stream mode)
            for (auto& f : accumulate_futures)
            {
                f.wait();
            }
        }
    }

    // Container class for all the input/output resources needed by the wave for a single cache/table
    class TableData {
    public:
        TableData(
            std::shared_ptr<ECacheWrapper<IndexT>> cache,
            uint64_t num_inputs,
            uint64_t num_rows,
            uint64_t hotness,
            uint64_t batch_size,
            float alpha,
            int device_id,
            uint64_t num_iterations) : m_device_id(device_id), m_cache(cache), m_batch_size(batch_size), m_hotness(hotness)
        {
            ScopedDevice scope_device(m_device_id);
            assert(cache);
            m_host_inputs.resize(num_inputs);
            m_host_inputs_unique.resize(num_inputs);
            std::vector<std::shared_ptr<std::thread>> input_gen_threads;

            for (uint64_t s=0 ; s<num_inputs ; s++) {
                input_gen_threads.push_back(std::make_shared<std::thread>(GenerateInput<IndexT>, num_rows, hotness, batch_size, alpha, m_device_id, std::ref(m_host_inputs.at(s)), std::ref(m_host_inputs_unique.at(s))));
            }
            for (auto& t : input_gen_threads) {
                t->join();
            }

            // Initialize input/output buffers
            const uint64_t num_keys = batch_size * hotness;
            const uint64_t input_size = num_keys * sizeof(IndexT);
            const uint64_t hitmask_size = ((num_keys + 63) / 64) * 64 * sizeof(uint64_t);
            const uint64_t output_size = num_keys * cache->GetConfig().embedWidth;
            const uint64_t pooling_output_size = batch_size * cache->GetConfig().embedWidth;
            auto allocator = m_cache->GetAllocator();

            for (uint64_t s=0 ; s<num_inputs ; s++) {
                IndexT* buff = nullptr;
                NVE_CHECK_(allocator->deviceAllocate((void**)(&buff), input_size));
                m_d_inputs.push_back(buff);
                
                auto& h_input = m_host_inputs.at(s);
                NVE_CHECK_(cudaMemcpy(buff, h_input.data(), h_input.size() * sizeof(IndexT), cudaMemcpyDefault));
            }
            NVE_CHECK_(allocator->hostAllocate((void**)(&m_h_hitmask), hitmask_size));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_hitmask), hitmask_size));
            NVE_CHECK_(allocator->hostAllocate((void**)(&m_h_output), output_size));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_output), output_size));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_pooling_output), pooling_output_size));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_accumulate), output_size));

            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_unique_keys), input_size));
            NVE_CHECK_(allocator->hostAllocate((void**)(&m_h_unique_keys), input_size));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_counts), input_size));
            NVE_CHECK_(allocator->hostAllocate((void**)(&m_h_counts), input_size));
            NVE_CHECK_(allocator->hostAllocate((void**)(&m_h_num_runs_out), sizeof(IndexT)));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_inverse_buffer), input_size));
            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_offsets), input_size));
        
            NVE_CHECK_(cudaMemset(m_d_accumulate, 0, output_size));
            NVE_CHECK_(allocator->hostAllocate((void**)(&m_h_accumulate), output_size));

            NVE_CHECK_(allocator->deviceAllocate((void**)(&m_d_all_buffer), output_size ));

            m_metrics.resize(num_iterations);
        }
        ~TableData() {
            ScopedDevice scope_device(m_device_id);
            for (auto& input : m_host_inputs)
            {
                NVE_CHECK_(cudaHostUnregister(input.data()));
            }
            for (auto& input : m_host_inputs_unique)
            {
                NVE_CHECK_(cudaHostUnregister(input.data()));
            }

            auto allocator = m_cache->GetAllocator();
            for (auto& input : m_d_inputs)
            {
                NVE_CHECK_(allocator->deviceFree(input));
            }
            NVE_CHECK_(allocator->hostFree(m_h_hitmask));
            NVE_CHECK_(allocator->deviceFree(m_d_hitmask));
            NVE_CHECK_(allocator->hostFree(m_h_output));
            NVE_CHECK_(allocator->deviceFree(m_d_output));
            NVE_CHECK_(allocator->deviceFree(m_d_pooling_output));
            NVE_CHECK_(allocator->deviceFree(m_d_accumulate));
            NVE_CHECK_(allocator->hostFree(m_h_accumulate));

            NVE_CHECK_(allocator->deviceFree(m_d_unique_keys));
            NVE_CHECK_(allocator->hostFree(m_h_unique_keys));
            NVE_CHECK_(allocator->deviceFree(m_d_counts));
            NVE_CHECK_(allocator->hostFree(m_h_counts));
            NVE_CHECK_(allocator->hostFree(m_h_num_runs_out));
            NVE_CHECK_(allocator->deviceFree(m_d_inverse_buffer));
            NVE_CHECK_(allocator->deviceFree(m_d_offsets));

            NVE_CHECK_(allocator->deviceFree(m_d_all_buffer));
        }

        std::vector<std::vector<IndexT>> m_host_inputs;
        std::vector<std::vector<IndexT>> m_host_inputs_unique; // Unique indices for update_accumulate
        std::vector<IndexT*> m_d_inputs;
        uint64_t* m_h_hitmask{nullptr};
        uint64_t* m_d_hitmask{nullptr};
        int8_t* m_h_output{nullptr};
        int8_t* m_d_output{nullptr};
        int8_t* m_d_pooling_output{nullptr};
        int8_t* m_d_accumulate{nullptr};
        int8_t* m_h_accumulate{nullptr};

        int m_device_id;
        std::shared_ptr<ECacheWrapper<IndexT>> m_cache{nullptr};
        std::vector<CacheIterationMetric> m_metrics;

        IndexT* m_d_unique_keys {nullptr}; 
        IndexT* m_h_unique_keys {nullptr}; 
        IndexT* m_d_counts {nullptr};
        IndexT* m_h_counts {nullptr};
        IndexT* m_h_num_runs_out {nullptr};
        IndexT* m_d_inverse_buffer {nullptr};
        IndexT* m_d_offsets {nullptr};

        int8_t* m_d_all_buffer {nullptr}; // buffer for concatnating when k-sharding

        uint64_t m_batch_size;
        uint64_t m_hotness;
    };
    std::vector<std::shared_ptr<TableData>>& GetData() { return m_data; }
    int GetDevice() const { return m_device_id; }
private:
    std::vector<std::shared_ptr<CacheTaskQueue<IndexT>>> m_task_queues;
    uint64_t m_num_inputs;
    int m_device_id;
    std::vector<std::shared_ptr<TableData>> m_data;
    bool m_host_inputs{false};
    bool m_dedup{false};
    
    // Mock GPU work for FWD+BWD and associated CUDA events/streams
    std::shared_ptr<MockGPUWork> m_gpu_fwd_bwd;
    std::vector<cudaEvent_t> m_ctq_work_done;
    std::vector<cudaStream_t> m_lookup_streams;
    cudaEvent_t m_gpu_work_done;
    cudaStream_t m_gpu_work_stream;

    // communicator
    std::shared_ptr<Communicator> m_communicator;
};
