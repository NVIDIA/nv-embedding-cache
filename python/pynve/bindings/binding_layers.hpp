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

#include "include/embedding_layer.hpp"
#include "include/hierarchical_embedding_layer.hpp"
#include "include/linear_embedding_layer.hpp"
#include "include/gpu_embedding_layer.hpp"
#include "include/execution_context.hpp"
#include "include/gpu_table.hpp"
#include "include/host_table.hpp"
#include "include/insert_heuristic.hpp"
#include <iostream>
#include <fstream>
#include "third_party/dlpack/include/dlpack/dlpack.h"
#include "include/resizeable_buffer.hpp"
#include "include/allocator.hpp"
#include "cuda_ops/cuda_utils.cuh"
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/dedup_grads_kernel.cuh"
#include "cuda_ops/gradient_calculator.cuh"
#include "include/memblock.hpp"
#include "binding_tables.hpp"
#include "include/serialization.hpp"
#include "cuda_ops/pipeline_gather.cuh"

#include <sys/mman.h>

namespace nve {

struct EmbedLayerConfig {
    int64_t logging_interval = -1;
    int64_t kernel_mode = 0;
    int64_t kernel_mode_value_1 = 0;
    int64_t kernel_mode_value_2 = 0;
};

template<typename IndexT>
class NVEmbedBinding
{
public:
    // Accessors used by torch custom ops for shape inference and output allocation.
    int64_t get_embedding_dim() const {
        switch (data_type_) {
            case nve::DataType_t::Float32:
                return static_cast<int64_t>(row_size_in_bytes_ / sizeof(float));
            case nve::DataType_t::Float16:
                return static_cast<int64_t>(row_size_in_bytes_ / sizeof(__half));
            default:
                NVE_CHECK_(false, "Invalid data type");
                return 0;
        }
    }

    nve::DataType_t get_data_type() const { return data_type_; }

    void lookup(size_t num_keys, uintptr_t keys, uintptr_t output, uint64_t stream_)
    {
        float hitrates[3] = {0};
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        
        emb_layer_ptr_->lookup(ctx,
                            num_keys,
                            reinterpret_cast<const void*>(keys),
                            reinterpret_cast<void*>(output),
                            row_size_in_bytes_,
                            nullptr, /* hitmask */
                            nullptr, /* pool_params*/
                            hitrates);

        log_hit_rates(hitrates, sizeof(hitrates) / sizeof(hitrates[0]));
    }

    void lookup_with_pooling(size_t num_keys, uintptr_t keys, uintptr_t output,
                             uint32_t pooling_type, size_t num_offsets, uintptr_t offsets,
                             uint64_t weight_data_type, uintptr_t weights, uint64_t stream_)
    {
        float hitrates[3] = {0};
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        
        EmbeddingLayerBase::PoolingParams pool_params;
        pool_params.pooling_type = PoolingType_t(pooling_type);
        pool_params.sparse_type = SparseType_t::CSR;
        pool_params.key_indices = reinterpret_cast<const int64_t*>(offsets);
        pool_params.num_key_indices = num_offsets;
        pool_params.sparse_weights = reinterpret_cast<const void*>(weights);
        pool_params.weight_type = DataType_t(weight_data_type);

        emb_layer_ptr_->lookup(ctx,
                            num_keys,
                            reinterpret_cast<const void*>(keys),
                            reinterpret_cast<void*>(output),
                            row_size_in_bytes_,
                            nullptr, /* hitmask */
                            &pool_params, /* pool_params*/
                            hitrates);

        log_hit_rates(hitrates, sizeof(hitrates) / sizeof(hitrates[0]));
    }

    size_t concat_backprop(size_t num_keys, uintptr_t keys, uintptr_t grads,
                           uint64_t unique_keys, uintptr_t output_grads, uint64_t stream_)
    { 
        auto stream = reinterpret_cast<cudaStream_t>(stream_);

        if (!backprop_runner_) {
            backprop_runner_ = std::make_shared<GradientCalculator<IndexT, MAX_RUN_SIZE>>();
        }
        //TRTREC-55 handle increases in num_keys
        if (num_keys > max_num_keys_) {
            size_t tmp_mem_size_device, tmp_mem_size_host;
            size_t data_element_size = (data_type_ == nve::DataType_t::Float32) ? sizeof(float) : sizeof(__half);
            backprop_runner_->GetAllocRequirements(num_keys, data_element_size, tmp_mem_size_device, tmp_mem_size_host);
            char* tmp_device_mem = reinterpret_cast<char*>(d_tmp_device_buf_.get_ptr(tmp_mem_size_device));
            char* tmp_host_mem = reinterpret_cast<char*>(d_tmp_host_buf_.get_ptr(tmp_mem_size_host));
            backprop_runner_->SetAndInitBuffers(num_keys, data_element_size, tmp_device_mem, tmp_host_mem);
            max_num_keys_ = num_keys;
        }

        // CSR 
        constexpr bool is_fixed_hotness = false;
        uint32_t element_size = data_type_ == nve::DataType_t::Float32 ? sizeof(float) : sizeof(__half);
        uint32_t embed_width = static_cast<uint32_t>(row_size_in_bytes_  / element_size);

        if (data_type_ == nve::DataType_t::Float32) {
            uint64_t num_unique_keys = backprop_runner_->template ComputeGradients<float, is_fixed_hotness>(
                reinterpret_cast<const IndexT*>(keys),
                nullptr,
                reinterpret_cast<const float*>(grads),
                nullptr,
                reinterpret_cast<float*>(output_grads),
                reinterpret_cast<IndexT*>(unique_keys),
                1, // batch
                num_keys,
                0, // hotness
                embed_width,
                PoolingType_t::Concatenate,
                stream);
            return num_unique_keys;
        } else {
            uint64_t num_unique_keys = backprop_runner_->template ComputeGradients<__half, is_fixed_hotness>(
                reinterpret_cast<const IndexT*>(keys),
                nullptr,
                reinterpret_cast<const __half*>(grads),
                nullptr,
                reinterpret_cast<__half*>(output_grads),
                reinterpret_cast<IndexT*>(unique_keys),
                1, // batch
                num_keys,
                0, // hotness
                embed_width,
                PoolingType_t::Concatenate,
                stream);
            return num_unique_keys;
        }
    }

    size_t pooling_backprop(size_t num_keys, uintptr_t keys, uintptr_t grads,
                            uintptr_t unique_keys, uintptr_t output_grads,
                            uint32_t pooling_type, uint32_t num_offsets, uintptr_t offsets,
                            uint64_t /*data_type*/, uintptr_t weights, uint64_t stream_)
    {
        auto stream = reinterpret_cast<cudaStream_t>(stream_);

        if (!backprop_runner_) {
            backprop_runner_ = std::make_shared<GradientCalculator<IndexT, MAX_RUN_SIZE>>();
        }
        // handle increases in num_keys
        if (num_keys > max_num_keys_) {
            size_t tmp_mem_size_device, tmp_mem_size_host;
            size_t data_element_size = (data_type_ == nve::DataType_t::Float32) ? sizeof(float) : sizeof(__half);
            backprop_runner_->GetAllocRequirements(num_keys, data_element_size, tmp_mem_size_device, tmp_mem_size_host);
            char* tmp_device_mem = reinterpret_cast<char*>(d_tmp_device_buf_.get_ptr(tmp_mem_size_device));
            char* tmp_host_mem = reinterpret_cast<char*>(d_tmp_host_buf_.get_ptr(tmp_mem_size_host));
            backprop_runner_->SetAndInitBuffers(num_keys, data_element_size, tmp_device_mem, tmp_host_mem);
            max_num_keys_ = num_keys;
        }

        // CSR 
        constexpr bool is_fixed_hotness = false;
        uint32_t element_size = data_type_ == nve::DataType_t::Float32 ? sizeof(float) : sizeof(__half);
        uint32_t embed_width = static_cast<uint32_t>(row_size_in_bytes_  / element_size);

        if (data_type_ == nve::DataType_t::Float32) {
            uint64_t num_unique_keys = backprop_runner_->template ComputeGradients<float, is_fixed_hotness>(
                reinterpret_cast<const IndexT*>(keys),
                reinterpret_cast<const IndexT*>(offsets),
                reinterpret_cast<const float*>(grads),
                reinterpret_cast<const float*>(weights),
                reinterpret_cast<float*>(output_grads),
                reinterpret_cast<IndexT*>(unique_keys),
                num_offsets, // batch
                num_keys,
                0, // hotness
                embed_width,
                PoolingType_t(pooling_type),
                stream);
            return num_unique_keys;
        } else {
            uint64_t num_unique_keys = backprop_runner_->template ComputeGradients<__half, is_fixed_hotness>(
                reinterpret_cast<const IndexT*>(keys),
                reinterpret_cast<const IndexT*>(offsets),
                reinterpret_cast<const __half*>(grads),
                reinterpret_cast<const __half*>(weights),
                reinterpret_cast<__half*>(output_grads),
                reinterpret_cast<IndexT*>(unique_keys),
                num_offsets, // batch
                num_keys,
                0, // hotness
                embed_width,
                PoolingType_t(pooling_type),
                stream);
            return num_unique_keys;
        }
    }

    void accumulate(size_t num_keys, uintptr_t keys, uintptr_t updates, uint64_t stream_)
    {
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        emb_layer_ptr_->accumulate(ctx, num_keys, 
                            reinterpret_cast<const void*>(keys),
                            row_size_in_bytes_,
                            row_size_in_bytes_,
                            reinterpret_cast<const void*>(updates),
                            data_type_);
    }

    void update(size_t num_keys, uintptr_t keys, uintptr_t updates, uint64_t stream_)
    {
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        emb_layer_ptr_->update(ctx, num_keys, 
                            reinterpret_cast<const void*>(keys),
                            row_size_in_bytes_,
                            row_size_in_bytes_,
                            reinterpret_cast<const void*>(updates));
    }

    void insert(size_t num_keys, uintptr_t keys, uintptr_t values, int64_t table_id, uint64_t stream_)
    {
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        emb_layer_ptr_->insert( ctx,
                                num_keys,
                                reinterpret_cast<const void*>(keys),
                                row_size_in_bytes_,
                                row_size_in_bytes_,
                                reinterpret_cast<const void*>(values),
                                table_id);
    }

    void clear(uint64_t stream_)
    {
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        emb_layer_ptr_->clear(ctx);
    }

    void erase(size_t num_keys, uintptr_t keys, int64_t table_id, uint64_t stream_)
    {
        auto ctx = get_exec_context(reinterpret_cast<cudaStream_t>(stream_));
        emb_layer_ptr_->erase(ctx, num_keys, reinterpret_cast<const void*>(keys), table_id);
    }

    NVEmbedBinding(int device_id, size_t row_size, nve::DataType_t dtype, EmbedLayerConfig config) : allocator_(GetDefaultAllocator()), 
                        d_tmp_device_buf_(allocator_, false),
                        d_tmp_host_buf_(allocator_, true),
                        device_id_(device_id),
                        data_type_(dtype)
    {
        // allocate two entries - one for number of unique runs, and one for
        // the number of chunks after splitting the runs to avoid exceeding
        // max run size limit
        auto element_size = (dtype == nve::DataType_t::Float32) ? sizeof(float) : sizeof(__half);
        row_size_in_bytes_ = (row_size * element_size);
        
        ScopedDevice scope_device(device_id_);
        allocator_->host_allocate((void**)&h_num_runs_out_, sizeof(IndexT) * 2);
        NVE_CHECK_(cudaStreamCreate(&modify_stream_)); 

        logging_interval_ = config.logging_interval;
    }

    virtual ~NVEmbedBinding() 
    {
        ScopedDevice scope_device(device_id_);
        // children responsible for releasing contexts
        allocator_->host_free(h_num_runs_out_);
        if (modify_stream_)
        {
            ScopedDevice scope_device(device_id_);
            NVE_CHECK_(cudaStreamDestroy(modify_stream_));
        }
    }    

protected:
    void release_contexts()
    {
        // Wait for all pending work on the contexts to finish before destroying
        for (auto& stream_ctx_pair : stream_ctx_map_)
        {
            stream_ctx_pair.second->wait();
            stream_ctx_pair.second.reset();
        }
    }

    std::shared_ptr<nve::ExecutionContext> get_exec_context(cudaStream_t stream)
    {
        // we will be using one modify stream across the layer ctx, I assume for inference this is enough.
        // for training we probably want to use private stream to optimize accumulates
        if (stream_ctx_map_.count(stream) != 1)
        {
            cudaStream_t lookup_stream = stream;
            cudaStream_t modify_stream = modify_stream_;
            
            stream_ctx_map_[stream] = emb_layer_ptr_->create_execution_context(lookup_stream, modify_stream, nullptr, nullptr);
        }

        return stream_ctx_map_[stream];
    }
    
private:
    void log_hit_rates(const float* hit_rates, size_t n)
    {
        if (n == 0) {
            return;
        }

        if (logging_interval_ > 0) {
            if (logging_counter_ % logging_interval_ == 0) {
                std::string s = nve::to_string("Hit rates: ", hit_rates[0]);
                for (size_t i = 1; i < n; i++)
                {
                    s += nve::to_string(", ", hit_rates[i]);
                }
                NVE_LOG_PERF_(s);
            }
            logging_counter_++;   
        }
    }

protected:
    static constexpr int32_t MAX_RUN_SIZE = 1024;

    std::shared_ptr<nve::EmbeddingLayerBase> emb_layer_ptr_;
    uint64_t row_size_in_bytes_ = 0;
    std::shared_ptr<GradientCalculator<IndexT, MAX_RUN_SIZE>> backprop_runner_;
    allocator_ptr_t allocator_;
    IndexT* h_num_runs_out_ = nullptr;
    ResizeableBuffer d_tmp_device_buf_;
    ResizeableBuffer d_tmp_host_buf_;
    nve::DataType_t data_type_ = nve::DataType_t::Unknown;
    size_t max_num_keys_ = 0;
    std::map<cudaStream_t, std::shared_ptr<nve::ExecutionContext>> stream_ctx_map_;
    cudaStream_t modify_stream_ = 0;
    int device_id_ = -1;
    int64_t logging_counter_ = 0;
    int64_t logging_interval_ = -1;
};

template<typename IndexT>
class HierarchicalEmbedding : public NVEmbedBinding<IndexT>
{
private:
    using layer_type = nve::HierarchicalEmbeddingLayer<IndexT>;

public:
    HierarchicalEmbedding(size_t row_size, nve::DataType_t dtype, 
                          uint64_t gpu_cache_size, 
                          uint64_t host_cache_size,
                          table_ptr_t remote, 
                          bool use_private_stream, 
                          int device_id, 
                          EmbedLayerConfig config) : 
                          NVEmbedBinding<IndexT>(device_id, row_size, dtype, config), 
                          use_private_stream_(use_private_stream), 
                          ps_table_(nullptr)
    {
        ScopedDevice scope_device(this->device_id_);
        std::vector<std::shared_ptr<nve::Table>> tables;

        nve::GPUTableConfig cfg;
        cfg.device_id = this->device_id_;
        cfg.cache_size = static_cast<int64_t>(gpu_cache_size);
        cfg.max_modify_size = (1l << 20);
        cfg.row_size_in_bytes = this->row_size_in_bytes_;
        cfg.uvm_table = nullptr;
        cfg.value_dtype = dtype;
        if (use_private_stream)
        {
            NVE_CHECK_(cudaStreamCreate(&private_stream_));
            cfg.private_stream = private_stream_;
        }

        std::vector<float> insert_heuristic_thresholds;
        auto gpu_table = std::make_shared<nve::GpuTable<IndexT>>(cfg, nullptr /* using default allocator for device 0*/);
        tables.push_back(gpu_table);
        insert_heuristic_thresholds.push_back(DefaultInsertHeuristic::DEFAULT_THRESHOLD);

        if (host_cache_size > 0) {
            host_table_ptr_t nvhm_table = create_nvhm_table(host_cache_size, this->row_size_in_bytes_, dtype);
            tables.push_back(nvhm_table);
            if (remote) {
                // Host cache is L2 with remote being L3, set target hitrate to be proportional by size
                // Unless num_rows is 0, which means increase host cache can increase indefinitely
                auto num_rows = std::dynamic_pointer_cast<ParameterServerTable>(remote)->get_num_rows();
                float target_hitrate = num_rows > 0 ? float(host_cache_size) / float(num_rows * this->row_size_in_bytes_) : 1.0f;
                if (target_hitrate > 1.f) {
                    NVE_LOG_WARNING_("Cache is initialized with maximal size larger than the embedding table");
                    target_hitrate = std::min(target_hitrate, 1.0f); // Clamping hitrate to 1.0
                }
                insert_heuristic_thresholds.push_back(target_hitrate);
            } else {
                // No remote so host cache is the last level, therefore there's no way to auto-insert and no misses are expected at this level
                insert_heuristic_thresholds.push_back(0.0f);
            }
        }
        if (remote) {
            // we are using a custom table wrapper to be able to do late binding of the parameter server
            ps_table_ = std::make_shared<PyNVETable>(remote);
            tables.push_back(ps_table_);
            insert_heuristic_thresholds.push_back(0.f); // remote PS is typically updated externally instead of by the layer
        }
        typename layer_type::Config layer_cfg = {"ps_layer", std::make_shared<DefaultInsertHeuristic>(insert_heuristic_thresholds)};
        this->emb_layer_ptr_ = std::make_shared<layer_type>(layer_cfg, tables, nullptr /* using default allocator for device 0*/);
    }

    void set_ps_table(std::shared_ptr<nve::Table>& table)
    {
        ps_table_->set_table(table);
    }

    host_table_ptr_t create_nvhm_table(uint64_t table_size, uint64_t row_size, nve::DataType_t data_type)
    {
        load_host_table_plugin("nvhm");

        constexpr int64_t num_partitions = 1; // Single partition is better for inference, increase if lock contention is an issue during insert
        const int64_t keys_per_partition = table_size / row_size / num_partitions;
        NVE_CHECK_(keys_per_partition > 0, "Host table is too small");

        nlohmann::json nvhm_conf = {
          {"key_size", sizeof(IndexT)},
          {"max_value_size", row_size},
          {"num_partitions", num_partitions},
          {"value_dtype", to_string(data_type)},
          {"value_alignment", 32},
          {"overflow_policy",
            {
              {"handler", "evict_random"}, // Using random eviction, assuming gpu cache handles all hot keys.
                                           // Otherwise, replace with "evict_lru"
              {"overflow_margin", keys_per_partition},
              {"resolution_margin", 0.9}
            }
          }
        };
        nve::host_table_factory_ptr_t nvhm_fac{
          nve::create_host_table_factory(R"({"implementation": "nvhm_map"})"_json)};
        return nvhm_fac->produce(0, nvhm_conf);
    }

    ~HierarchicalEmbedding()
    {
        ScopedDevice scope_device(this->device_id_);
        this->release_contexts();
        if (use_private_stream_)
        {
            NVE_CHECK_(cudaStreamSynchronize(private_stream_));
            NVE_CHECK_(cudaStreamDestroy(private_stream_));
        }
    }
private:
    bool use_private_stream_{false};
    cudaStream_t private_stream_{0};
    std::shared_ptr<PyNVETable> ps_table_;
};

template<typename IndexT>
class LinearUVMEmbedding : public NVEmbedBinding<IndexT>
{
private:
    using layer_type = nve::LinearUVMEmbeddingLayer<IndexT>;
    public:
    LinearUVMEmbedding( size_t row_size, 
                        size_t num_embeddings, 
                        nve::DataType_t dtype, 
                        std::shared_ptr<MemBlock> mem_block, 
                        size_t gpu_cache_size, 
                        bool use_private_stream, 
                        int device_id,
                        EmbedLayerConfig config) : 
                        NVEmbedBinding<IndexT>(device_id, row_size, dtype, config), 
                        use_private_stream_(use_private_stream),
                        mem_block_(mem_block)
    {
        ScopedDevice scope_device(this->device_id_);
        auto element_size = (dtype == nve::DataType_t::Float32) ? sizeof(float) : sizeof(__half);
        
        nve::GPUTableConfig cfg;
        cfg.device_id = this->device_id_;
        cfg.cache_size = static_cast<int64_t>(gpu_cache_size);
        cfg.max_modify_size = (16l << 20);
        cfg.row_size_in_bytes = this->row_size_in_bytes_;
        cfg.value_dtype = dtype;
        cfg.count_misses = true;
        
        if (use_private_stream)
        {
            NVE_CHECK_(cudaStreamCreate(&private_stream_));
            cfg.private_stream = private_stream_;
        }

        cfg.uvm_table = mem_block_->get_ptr();

        // handle kernel mode
        cfg.kernel_mode_type = config.kernel_mode;
        
        if (cfg.kernel_mode_type == static_cast<uint64_t>(nve::KernelType::SortGather)) {
            cfg.kernel_mode_value = config.kernel_mode_value_1 > 0 ? config.kernel_mode_value_1 : 1024;
        }

        else if (cfg.kernel_mode_type == static_cast<uint64_t>(nve::KernelType::PipelineGather)) {
            gather_pipeline_params_.task_size = config.kernel_mode_value_1 > 0 ? config.kernel_mode_value_1 : 1024;
            gather_pipeline_params_.num_aux_streams = config.kernel_mode_value_2 > 0 ? config.kernel_mode_value_2 : 16;
            cfg.kernel_mode_value = reinterpret_cast<uintptr_t>(&gather_pipeline_params_);
        }
        
        auto gpu_table = std::make_shared<nve::GpuTable<IndexT>>(cfg, nullptr /* using default allocator for device 0*/);
        
        typename layer_type::Config layer_cfg = {"uvm_layer", std::make_shared<DefaultInsertHeuristic>(std::vector<float>{DefaultInsertHeuristic::DEFAULT_THRESHOLD})};
        this->emb_layer_ptr_ = std::make_shared<layer_type>(layer_cfg, gpu_table, nullptr /* using default allocator for device 0*/);

        uvm_table_.col = row_size;
        uvm_table_.row = num_embeddings;
        uvm_table_.data = cfg.uvm_table;
        uvm_table_.element_size_in_bytes = element_size;
    }

    // TODO: this is called for hierarchical, need to change
    DLManagedTensor* create_dlpack_tensor(nve::DataType_t dtype)
    {
        // Allocate and initialize your tensor here
        // For example, create a 1D tensor with 10 elements
        int64_t* shape  = new int64_t[2];
        shape[0] = uvm_table_.row;
        shape[1] = uvm_table_.col;
        unsigned char value_size = dtype == nve::DataType_t::Float32 ? 32 : 16;
        DLManagedTensor* dlm_tensor = new DLManagedTensor();
        dlm_tensor->dl_tensor.data = uvm_table_.data;
        dlm_tensor->dl_tensor.device = {kDLCUDA, this->device_id_};
        dlm_tensor->dl_tensor.ndim = 2;
        dlm_tensor->dl_tensor.dtype = {kDLFloat, value_size, 1};
        dlm_tensor->dl_tensor.shape = shape;
        dlm_tensor->dl_tensor.strides = NULL;
        dlm_tensor->dl_tensor.byte_offset = 0;
       
        // Set the deleter function
        dlm_tensor->deleter =  [](DLManagedTensor* dl) {
           
            if (!dl) {
                return;
            }
            if (dl->dl_tensor.shape) {
                delete[] dl->dl_tensor.shape;
            }
            delete dl;
        };

        return dlm_tensor;
    }

    void write_tensor_to_stream(nve::StreamWrapperBase& stream, uint64_t name)
    {
        nve::TensorFileFormat writer;
        writer.write_tensor_to_stream(stream, name, uvm_table_);
    }

    void load_tensor_from_stream(nve::StreamWrapperBase& stream, uint64_t name)
    {
        nve::TensorFileFormat reader;
        reader.load_tensor_from_stream(stream, name, uvm_table_);
    }

    ~LinearUVMEmbedding()
    {
        ScopedDevice scope_device(this->device_id_);
        // Wait for all pending work on the contexts to finish before destroying
        this->release_contexts();
        if (use_private_stream_)
        {
            NVE_CHECK_(cudaStreamSynchronize(private_stream_));
            NVE_CHECK_(cudaStreamDestroy(private_stream_));
        }
    }
    
private:
    bool use_private_stream_{false};
    cudaStream_t private_stream_{0};
    TensorWrapper uvm_table_;
    uint64_t num_embeddings_{0};
    std::shared_ptr<MemBlock> mem_block_;
    GatherKernelPipelineParams gather_pipeline_params_;
};

template<typename IndexT>
class GPUEmbedding : public NVEmbedBinding<IndexT>
{
private:
    using layer_type = nve::GPUEmbeddingLayer<IndexT>;
    public:
    GPUEmbedding(size_t row_size, 
                size_t num_embeddings, 
                nve::DataType_t dtype, 
                uintptr_t embedding_table, 
                int device_id, 
                EmbedLayerConfig config) : 
                NVEmbedBinding<IndexT>(device_id, row_size, dtype, config) 
    {
        ScopedDevice scope_device(this->device_id_);

        nve::GPUEmbeddingLayerConfig layer_cfg;
        layer_cfg.device_id = this->device_id_;
        layer_cfg.num_embeddings = num_embeddings;
        layer_cfg.embedding_width_in_bytes = this->row_size_in_bytes_;
        layer_cfg.embedding_table = reinterpret_cast<void*>(embedding_table);
        layer_cfg.value_dtype = this->data_type_;
        layer_cfg.layer_name = "gpu_layer";
        this->emb_layer_ptr_ = std::make_shared<layer_type>(layer_cfg, nullptr /* using default allocator for device 0*/);
    }

    ~GPUEmbedding()
    {
        ScopedDevice scope_device(this->device_id_);
        this->release_contexts();
    }
};

} // namespace nve
