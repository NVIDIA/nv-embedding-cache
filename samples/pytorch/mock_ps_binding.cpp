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

#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "include/table.hpp"
#include <iostream>

namespace py = pybind11;
using IndexT = int64_t;

// example class inherting from the remote interface 
// the will mock a PS by having a storage on the CPU managed by this class

class RemoteTableWrapper final : public nve::Table
{
public:
    RemoteTableWrapper(size_t num_embeddings, size_t embedding_size, nve::DataType_t data_type) : m_num_embeddings(num_embeddings), m_embedding_size(embedding_size), m_data_type(data_type) 
    {
        const size_t data_type_size_in_bytes = (data_type == nve::DataType_t::Float32) ? sizeof(float) : sizeof(__half);
        m_embedding_size_in_bytes = embedding_size * data_type_size_in_bytes;
        m_data = new int8_t[num_embeddings * m_embedding_size_in_bytes];
    }

    ~RemoteTableWrapper()
    {
        delete[] m_data;
    }

    // function to load data from a pytorch tensor
    void load(uint64_t data)
    {
        cudaMemcpy(m_data, reinterpret_cast<int8_t*>(data), m_num_embeddings * m_embedding_size_in_bytes, cudaMemcpyDefault);
    }

    void clear(nve::context_ptr_t&) override
    {
        //no need for the sample
    }

    void erase(nve::context_ptr_t&, int64_t, const void*) override
    {
       //no need for the sample
    }    

    void find(nve::context_ptr_t& /*ctx*/, 
                 int64_t n, 
                 const void* keys, 
                 nve::max_bitmask_repr_t* hit_mask,
                 int64_t value_stride, 
                 void* values, 
                 int64_t* /*value_sizes*/) const override
    {
       constexpr auto mask_elements = static_cast<int64_t>(sizeof(int64_t) * 8);
        const IndexT* typed_keys = reinterpret_cast<const IndexT*>(keys);
        int64_t total_hits = 0;
        for (int64_t i = 0; i < n; i++) {
            const uint64_t bit = 1ul << (static_cast<uint64_t>(i) % mask_elements);
            if (hit_mask[i / mask_elements] & bit) {
                continue;  // datavector was already hit before
            }
            auto* dst = reinterpret_cast<uint8_t*>(values) + (i * value_stride);
            auto* src = m_data + typed_keys[i] * static_cast<IndexT>(m_embedding_size_in_bytes);
            std::memcpy(dst, src,  m_embedding_size_in_bytes);
            total_hits++;
        }
        m_key_counter += total_hits;
    }

    void insert(nve::context_ptr_t&, int64_t, const void*, int64_t, int64_t, const void*) override
    {
       //no need for the sample
    }

    void update(nve::context_ptr_t& /*ctx*/,
                int64_t /*n*/,
                const void* /*keys*/,
                int64_t /*value_stride*/,
                int64_t /*value_size*/,
                const void* /*values*/) override
    {
        //no need for the sample
        return;
    }

    template<typename DataType>
    void compute_update(DataType* typed_data,
                           int64_t n, 
                           const IndexT* typed_keys,
                           int64_t update_stride,
                           const void* updates) {
        for (int64_t i = 0; i < n; i++)
        {
            auto key = typed_keys[i];
            DataType* dst_ptr = typed_data + key * static_cast<IndexT>(m_embedding_size);
            const DataType* src_ptr = reinterpret_cast<const DataType*>((int8_t*)updates + i * update_stride);
            for (size_t j = 0; j < m_embedding_size; j++)
            {
                dst_ptr[j] += src_ptr[j];
            }
        }
    }

    void update_accumulate( nve::context_ptr_t& /*ctx*/, 
                            int64_t n, 
                            const void* keys,
                            int64_t update_stride, 
                            int64_t /*update_size*/, 
                            const void* updates,
                            nve::DataType_t /*update_dtype*/) override
    {
        if (m_data_type == nve::DataType_t::Float32) {
            compute_update<float>(reinterpret_cast<float*>(m_data), n, reinterpret_cast<const IndexT*>(keys), update_stride, updates);
        } else {
            compute_update<__half>(reinterpret_cast<__half*>(m_data), n, reinterpret_cast<const IndexT*>(keys), update_stride, updates);
        }
    }

    int32_t get_device_id() const override { return -1; }
    int64_t get_max_row_size() const override { return static_cast<int64_t>(m_embedding_size_in_bytes); }

    void reset_lookup_counter(nve::context_ptr_t& /*ctx*/) override { m_key_counter = 0; }
    void get_lookup_counter(nve::context_ptr_t& /*ctx*/, int64_t* counter) override { *counter = m_key_counter; }
    bool lookup_counter_hits() override { return true; }

private:
    size_t m_num_embeddings;
    size_t m_embedding_size;
    size_t m_embedding_size_in_bytes;
    int8_t* m_data;
    nve::DataType_t m_data_type;
    mutable int64_t m_key_counter = 0;
};

PYBIND11_MODULE(sample_remote, m) {
    py::class_<RemoteTableWrapper, nve::Table, std::shared_ptr<RemoteTableWrapper> /* <- holder type */>(m, "RemoteTable")
        .def(py::init<size_t, size_t, nve::DataType_t>())
        .def("load", &RemoteTableWrapper::load);
}
