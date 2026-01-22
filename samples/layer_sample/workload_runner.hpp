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
#include <engine_harness.h>

#include <embedding_layer.hpp>
#include <execution_context.hpp>
#include <input_gen.cuh>
#include <memory>
#include <thread>
#include <vector>

// todo: template classes on IndexT
using IndexT = int64_t;
using layer_type = nve::EmbeddingLayerBase;

class ColocatedBuffer {
 public:
  ColocatedBuffer(uint64_t size_, int device_id = 0) : size(size_) {
    ScopedDevice sc(device_id);
    NVE_CHECK_(cudaMalloc(&m_d_ptr, size));
    NVE_CHECK_(cudaMallocHost(&m_h_ptr, size));
  }
  ~ColocatedBuffer() {
    NVE_CHECK_(cudaFree(m_d_ptr));
    NVE_CHECK_(cudaFreeHost(m_h_ptr));
  }
  const uint64_t size;
  int8_t* h_ptr() const { return m_h_ptr; }
  int8_t* d_ptr() const { return m_d_ptr; }

 private:
  int8_t* m_h_ptr;
  int8_t* m_d_ptr;
};

class LayerData {
 public:
  LayerData(uint64_t num_inputs, uint64_t num_warmups, uint64_t num_rows, uint64_t hotness,
            uint64_t batch_size, float alpha, uint64_t row_size, int device_id)
      : m_num_inputs(num_inputs),
        m_num_warmups(num_warmups),
        m_num_rows(num_rows),
        m_hotness(hotness),
        m_batch_size(batch_size),
        m_alpha(alpha),
        m_row_size(row_size),
        m_device_id(device_id) {
    ScopedDevice scope_device(m_device_id);
    {  // Generate inference inputs
      std::vector<std::shared_ptr<std::thread>> input_gen_threads;
      m_host_inputs.resize(num_inputs);
      for (uint64_t s = 0; s < m_num_inputs; s++) {
        input_gen_threads.push_back(
            std::make_shared<std::thread>(GenerateInput<IndexT>, num_rows, hotness, batch_size,
                                          alpha, m_device_id, std::ref(m_host_inputs.at(s))));
      }
      for (auto& t : input_gen_threads) {
        t->join();
      }
    }
    {  // Generate warmup inputs
      std::vector<std::shared_ptr<std::thread>> warmup_gen_threads;
      m_host_warmups.resize(m_num_warmups);
      for (uint64_t s = 0; s < m_num_warmups; s++) {
        warmup_gen_threads.push_back(
            std::make_shared<std::thread>(GenerateInput<IndexT>, num_rows, hotness, batch_size,
                                          alpha, m_device_id, std::ref(m_host_warmups.at(s))));
      }
      for (auto& t : warmup_gen_threads) {
        t->join();
      }
    }

    const auto num_keys = m_hotness * m_batch_size;
    const uint64_t input_size = num_keys * sizeof(IndexT);
    m_input = std::make_shared<ColocatedBuffer>(input_size);
  }

  ~LayerData() {
    for (auto& input : m_host_inputs) {
      NVE_CHECK_(cudaHostUnregister(input.data()));
    }
  }

  const uint64_t m_num_inputs;
  const uint64_t m_num_warmups;
  const uint64_t m_num_rows;
  const uint64_t m_hotness;
  const uint64_t m_batch_size;
  const float m_alpha;
  const uint64_t m_row_size;
  const int m_device_id;
  std::vector<std::vector<IndexT>> m_host_inputs;
  std::vector<std::vector<IndexT>> m_host_warmups;
  std::shared_ptr<ColocatedBuffer> m_input;
};

class WorkloadRunner {
 public:
  WorkloadRunner(uint64_t num_inputs, uint64_t num_warmups, uint64_t num_rows, uint64_t hotness,
                 uint64_t batch_size, float alpha, uint64_t row_size,
                 uint64_t /*max_iterations*/,  // todo used for allocating metrics data
                 std::vector<std::shared_ptr<layer_type>>& layers,
                 std::shared_ptr<EngineHarness> trt_engine, uint64_t engine_ctx_id)
      : m_layers(layers), m_trt_engine(trt_engine), m_engine_ctx_id(engine_ctx_id) {
    // generate input
    const int device_id(0);  // todo multi device
    const auto num_layers = layers.size();
    for (uint64_t i = 0; i < num_layers; i++) {
      m_data.emplace_back(std::make_shared<LayerData>(num_inputs, num_warmups, num_rows, hotness,
                                                      batch_size, alpha, row_size, device_id));
    }

    // generate contexts
    for (uint64_t i = 0; i < num_layers; i++) {
      cudaStream_t lookup_stream;
      cudaStream_t modify_stream;
      NVE_CHECK_(cudaStreamCreate(&lookup_stream));
      m_lookup_streams.push_back(lookup_stream);
      NVE_CHECK_(cudaStreamCreate(&modify_stream));
      m_modify_streams.push_back(modify_stream);
      m_contexts.emplace_back(m_layers.at(i)->create_execution_context(lookup_stream, modify_stream, nullptr, nullptr));
    }

    // generate cuda events to signal each layer completion
    for (auto& l : m_layers) {
      cudaEvent_t e;
      NVE_CHECK_(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      m_layer_events.push_back(e);
    }

    // allocate output buffer for the embeddding layers (assuming all are concatenated)
    uint64_t emb_output_size = 0;
    for (auto& data : m_data) {
      emb_output_size += data->m_batch_size * data->m_hotness * data->m_row_size;
    }
    m_emb_output = std::make_shared<ColocatedBuffer>(emb_output_size);
    if (trt_engine) {
      auto bindings = trt_engine->GetIOBindings();
      if ((bindings.size() != 2) ||
          (bindings.begin()->isInput == (bindings.begin() + 1)->isInput)) {
        throw std::runtime_error("Invalid TRT engine - must have single input and output");
      }
      for (auto& b : bindings) {
        if (b.isInput) {
          // Use embedding output as engine input
          if (b.sizeInBytes > m_emb_output->size) {
            throw std::runtime_error(
                "Invalid TRT engine - engine input is larger than embedding output");
          } else if (b.sizeInBytes < m_emb_output->size) {
            std::cout << "[W] "
                      << "TRT engine input is smaller than the embedding output." << std::endl;
          }
          m_engine_buffers.push_back(m_emb_output->d_ptr());
        } else {
          // Create output buffer for the engine
          m_engine_output = std::make_shared<ColocatedBuffer>(b.sizeInBytes);
          m_engine_buffers.push_back(m_engine_output->d_ptr());
        }
      }
    }

    // todo: allocate metrics data
    // max_iterations
  }
  ~WorkloadRunner() {
    for (auto& c : m_contexts) {
      c->wait();
    }
    m_contexts.clear();
    for (auto& e : m_layer_events) {
      NVE_CHECK_(cudaEventDestroy(e));
    }
    for (auto& s : m_lookup_streams) {
      NVE_CHECK_(cudaStreamDestroy(s));
    }
    for (auto& s : m_modify_streams) {
      NVE_CHECK_(cudaStreamDestroy(s));
    }
  }

  void WarmupLayerCache(uint64_t num_iterations) {
    // todo: don't store warmups in layer data, instead create inputs on the fly to support large
    // amount of warmup iterations
    const auto num_layers = m_layers.size();
    const uint64_t num_warmups =
        m_data.at(0)
            ->m_host_warmups.size();  // assuming all layers generated the same amount of inputs
    for (uint64_t i = 0; i < num_iterations; i++) {
      for (uint64_t j = 0; j < num_layers; j++) {
        auto layer = m_layers.at(j);
        auto data = m_data.at(j);
        auto ctx = m_contexts.at(j);
        auto h_input = data->m_host_warmups.at(i % num_warmups);
        // using m_emb_output for insert sincwe it's large enough and we don't care about the values
        layer->insert(ctx, h_input.size(), h_input.data(), data->m_row_size, data->m_row_size,
                      m_emb_output->d_ptr(), 0);
        layer->insert(ctx, h_input.size(), h_input.data(), data->m_row_size, data->m_row_size,
                      m_emb_output->h_ptr(), 1);
      }
      NVE_CHECK_(cudaDeviceSynchronize());  // Just in case
    }
  }

  struct Metrics {
    std::vector<float> hitrates_gpu;
    std::vector<float> hitrates_host;
    std::vector<float> hitrates_remote;
    
    void resize(uint64_t num_iterations, uint64_t num_layers) {
      const uint64_t total_samples = num_iterations * num_layers;
      hitrates_gpu.resize(total_samples);
      hitrates_host.resize(total_samples);
      hitrates_remote.resize(total_samples);
    }
  };

  void Run(
    uint64_t num_iterations,
    Metrics* metrics,
    bool disable_copy_out = false
  ) {
    const uint64_t num_layers = m_layers.size();
    const cudaStream_t first_stream = m_contexts.at(0)->get_lookup_stream();  // Using the first layer's lookup stream for the dense
    const uint64_t num_inputs =
        m_data.at(0)
            ->m_host_inputs.size();  // assuming all layers generated the same amount of inputs
    float hitrates[3] = {0.f};
    for (uint64_t i = 0; i < num_iterations; i++) {
      // start with calling all lookups
      uint64_t output_offset = 0;
      for (uint64_t j = 0; j < num_layers; j++) {
        // todo: multithread here?
        auto& data = m_data.at(j);
        auto ctx = m_contexts.at(j);
        auto h_input = data->m_host_inputs.at(i % num_inputs);
        m_layers.at(j)->lookup(ctx, h_input.size(), h_input.data(),
                               m_emb_output->d_ptr() + output_offset, data->m_row_size,
                               nullptr, /* hitmask */
                               nullptr, /* pool_params*/
                               hitrates);
        // Collect hitrate metrics using direct indexing
        if (metrics != nullptr) {
        const uint64_t idx = i * num_layers + j;
        metrics->hitrates_gpu[idx] = hitrates[0];
          metrics->hitrates_host[idx] = hitrates[1];
          metrics->hitrates_remote[idx] = hitrates[2];
        }
        // update output offset
        const auto output_size = h_input.size() * data->m_row_size;
        output_offset += output_size;
      }
      // create dependency from all lookup streams to the first stream (which we'll reuse for the
      // dense part)
      for (uint64_t j = 1; j < num_layers; j++) {
        auto lookup_stream = m_contexts.at(j)->get_lookup_stream();
        auto event = m_layer_events.at(j);
        NVE_CHECK_(cudaEventRecord(event, lookup_stream));
        NVE_CHECK_(cudaStreamWaitEvent(first_stream, event));
      }

      if (m_trt_engine) {
        // Call trt engine for dense
        m_trt_engine->Enqueue(first_stream, static_cast<unsigned>(m_engine_ctx_id), m_engine_buffers);
        // Copy engine output to host
        if (!disable_copy_out) {
          NVE_CHECK_(cudaMemcpyAsync(m_engine_output->h_ptr(), m_engine_output->d_ptr(),
                                     m_engine_output->size, cudaMemcpyDefault, first_stream));
        }
      } else {
        // No TRT engine so copying embedding output to host
        if (!disable_copy_out) {
          NVE_CHECK_(cudaMemcpyAsync(m_emb_output->h_ptr(), m_emb_output->d_ptr(), m_emb_output->size,
                                     cudaMemcpyDefault, first_stream));
        }
      }
    }
  }

 private:
  std::vector<std::shared_ptr<layer_type>> m_layers;
  std::vector<std::shared_ptr<LayerData>> m_data;
  std::vector<std::shared_ptr<nve::ExecutionContext>> m_contexts;
  std::vector<cudaStream_t> m_lookup_streams;
  std::vector<cudaStream_t> m_modify_streams;
  std::vector<cudaEvent_t> m_layer_events;
  std::shared_ptr<ColocatedBuffer> m_emb_output;
  std::shared_ptr<ColocatedBuffer> m_engine_output;
  std::shared_ptr<EngineHarness> m_trt_engine;
  const uint64_t m_engine_ctx_id;
  std::vector<void*> m_engine_buffers;
};
