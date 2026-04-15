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
#include "third_party/pybind11/include/pybind11/functional.h"
#include "third_party/dlpack/include/dlpack/dlpack.h"
#include "binding_layers.hpp"
#include "binding_tables.hpp"
#include "binding_serialization.hpp"
#include "include/memblock.hpp"
#ifdef NVE_WITH_TORCH_BINDINGS
#include "python/pynve/torch_bindings/nve_registry.hpp"
#endif
#include "third_party/pybind11/include/pybind11/stl.h"
#include "include/distributed.hpp"

namespace py = pybind11;
using IndexT = int64_t;

namespace nve {

    // functions to separate the pybind11 objects from the rest of c++ code, so we can include it in other .so without needing pybind11
    void insert_keys_from_binary_file(std::shared_ptr<ParameterServerTable> table, py::object keys_stream, py::object values_stream, uint64_t batch_size);
    void insert_keys_from_numpy_file(std::shared_ptr<ParameterServerTable> table, py::object keys_stream, py::object values_stream, uint64_t batch_size);

    py::capsule get_dl_tensor(std::shared_ptr<LinearUVMEmbedding<IndexT>> layer, nve::DataType_t dtype)
    {
        DLManagedTensor* p = layer->create_dlpack_tensor(dtype);
        return py::capsule(p, "dltensor", [](PyObject* capsule) {
            const char *name = PyCapsule_GetName(capsule);
            if (!name || std::strcmp(name, "dltensor") != 0) {
                return;
            }
            auto *managed = static_cast<DLManagedTensor *>(
                PyCapsule_GetPointer(capsule, "dltensor")
            );
            if (!managed) return;
            if (managed->deleter) {
                managed->deleter(managed);
            }
        });
    }

    void write_tensor_to_stream(std::shared_ptr<LinearUVMEmbedding<IndexT>> layer, py::object stream, uint64_t name)
    {
        nve::PyStreamWrapper sw(stream);
        layer->write_tensor_to_stream(sw, name);
    }

    void load_tensor_from_stream(std::shared_ptr<LinearUVMEmbedding<IndexT>> layer, py::object stream, uint64_t name)
    {
        nve::PyStreamWrapper sw(stream);
        layer->load_tensor_from_stream(sw, name);
    }

    class PyDistributedEnv : public DistributedEnv, public py::trampoline_self_life_support {
    public:
        /* Inherit the constructors */
        using DistributedEnv::DistributedEnv;

        /* Trampoline (need one for each virtual function) */
        size_t rank() const override { PYBIND11_OVERRIDE_PURE(size_t, DistributedEnv, rank); }
        size_t world_size() const override { PYBIND11_OVERRIDE_PURE(size_t, DistributedEnv, world_size); }
        size_t device_count() const override { PYBIND11_OVERRIDE_PURE(size_t, DistributedEnv, device_count); }
        int local_device() const override { PYBIND11_OVERRIDE_PURE(int, DistributedEnv, local_device); }
        bool single_host() const override { PYBIND11_OVERRIDE_PURE(bool, DistributedEnv, single_host); }
        void barrier() override { PYBIND11_OVERRIDE_PURE(void, DistributedEnv, barrier); }
        void broadcast(uintptr_t buffer, size_t size, int root) override {
            PYBIND11_OVERRIDE_PURE(void, DistributedEnv, broadcast, buffer, size, root);
        }
        void all_gather(uintptr_t send_buffer, uintptr_t recv_buffer, size_t size) override {
            PYBIND11_OVERRIDE_PURE(void, DistributedEnv, all_gather, send_buffer, recv_buffer, size);
        }
    };

PYBIND11_MODULE(nve, m) {
    py::class_<NVEmbedBinding<IndexT>, std::shared_ptr<NVEmbedBinding<IndexT>>> (m, "NVEmbedBinding")
        .def(py::init<int, size_t, nve::DataType_t, EmbedLayerConfig>())
        .def("lookup", &NVEmbedBinding<IndexT>::lookup, py::call_guard<py::gil_scoped_release>())
        .def("lookup_with_pooling", &NVEmbedBinding<IndexT>::lookup_with_pooling, py::call_guard<py::gil_scoped_release>())
        .def("accumulate", &NVEmbedBinding<IndexT>::accumulate, py::call_guard<py::gil_scoped_release>())
        .def("concat_backprop", &NVEmbedBinding<IndexT>::concat_backprop, py::call_guard<py::gil_scoped_release>())
        .def("pooling_backprop", &NVEmbedBinding<IndexT>::pooling_backprop, py::call_guard<py::gil_scoped_release>())
        .def("update", &NVEmbedBinding<IndexT>::update, py::call_guard<py::gil_scoped_release>())
        .def("insert", &NVEmbedBinding<IndexT>::insert, py::call_guard<py::gil_scoped_release>())
        .def("clear", &NVEmbedBinding<IndexT>::clear, py::call_guard<py::gil_scoped_release>())
        .def("erase", &NVEmbedBinding<IndexT>::erase, py::call_guard<py::gil_scoped_release>());

    py::class_<HierarchicalEmbedding<IndexT>, NVEmbedBinding<IndexT>, std::shared_ptr<HierarchicalEmbedding<IndexT>>> (m, "HierarchicalEmbedding")
        .def(py::init<size_t, nve::DataType_t, uint64_t, uint64_t, table_ptr_t, bool, int, EmbedLayerConfig>())
        .def("set_ps_table", &HierarchicalEmbedding<IndexT>::set_ps_table);

    py::class_<LinearUVMEmbedding<IndexT>, NVEmbedBinding<IndexT>, std::shared_ptr<LinearUVMEmbedding<IndexT>>> (m, "LinearUVMEmbedding")
        .def(py::init<size_t, size_t, nve::DataType_t, std::shared_ptr<MemBlock>, size_t, bool, int, EmbedLayerConfig>())
        .def("load_tensor_from_stream", &LinearUVMEmbedding<IndexT>::load_tensor_from_stream)
        .def("write_tensor_to_stream", &LinearUVMEmbedding<IndexT>::write_tensor_to_stream);

    py::class_<GPUEmbedding<IndexT>, NVEmbedBinding<IndexT>, std::shared_ptr<GPUEmbedding<IndexT>>> (m, "GPUEmbedding")
        .def(py::init<size_t, size_t, nve::DataType_t, uintptr_t, int, EmbedLayerConfig>());

    py::class_<EmbedLayerConfig>(m, "EmbedLayerConfig")
        .def(py::init<>())
        .def_readwrite("logging_interval", &EmbedLayerConfig::logging_interval)
        .def_readwrite("kernel_mode", &EmbedLayerConfig::kernel_mode)
        .def_readwrite("kernel_mode_value_1", &EmbedLayerConfig::kernel_mode_value_1)
        .def_readwrite("kernel_mode_value_2", &EmbedLayerConfig::kernel_mode_value_2);

    py::class_<nve::Table, std::shared_ptr<nve::Table>>(m, "Table");

    py::class_<TensorFileFormat>(m, "TensorFileFormat")
        .def(py::init<>())
        .def("write_table_file_header",
             [](TensorFileFormat& self, py::object stream) {
                 PyStreamWrapper sw(stream);
                 self.write_table_file_header(sw);
             });

    py::enum_<PoolingType_t>(m, "PoolingType_t")
        .value("Concatenate", PoolingType_t::Concatenate)
        .value("Sum", PoolingType_t::Sum)
        .value("Mean", PoolingType_t::Mean)
        .value("WeightedSum", PoolingType_t::WeightedSum)
        .value("WeightedMean", PoolingType_t::WeightedMean)
        .export_values();

    py::enum_<DataType_t>(m, "DataType_t")
        .value("Unknown", DataType_t::Unknown)
        .value("Float32", DataType_t::Float32)
        .value("BFloat", DataType_t::BFloat)
        .value("Float16", DataType_t::Float16)
        .value("E4M3", DataType_t::E4M3)
        .value("E5M2", DataType_t::E5M2)
        .value("Float64", DataType_t::Float64)
        .export_values();

    py::enum_<MemBlockType>(m, "MemBlockType")
        .value("Linear", MemBlockType::LINEAR)
        .value("NVL", MemBlockType::NVL)
        .value("MPI", MemBlockType::MPI)
        .value("Managed", MemBlockType::MANAGED)
        .value("User", MemBlockType::USER)
        .export_values();

    py::class_<MemBlock, std::shared_ptr<MemBlock>>(m, "MemBlock")
        .def("get_handle", &MemBlock::get_handle)
        .def("get_type", &MemBlock::get_type);

    py::class_<LinearMemBlock, MemBlock, std::shared_ptr<LinearMemBlock>>(m, "LinearMemBlock")
        .def(py::init<size_t, size_t, nve::DataType_t>());

    py::class_<NVLMemBlock, MemBlock, std::shared_ptr<NVLMemBlock>>(m, "NVLMemBlock")
        .def(py::init<size_t, size_t, nve::DataType_t, std::vector<int>>());

    py::class_<MPIMemBlock, MemBlock, std::shared_ptr<MPIMemBlock>>(m, "MPIMemBlock")
        .def(py::init<size_t, size_t, nve::DataType_t, std::vector<size_t>, std::vector<int>>(),
            py::arg("row_size"),
            py::arg("num_embeddings"),
            py::arg("dtype"),
            py::arg("ranks") = std::vector<size_t>{},
            py::arg("devices") = std::vector<int>{}
        );

    py::class_<DistMemBlock, MemBlock, std::shared_ptr<DistMemBlock>>(m, "DistMemBlock")
        .def(py::init<std::shared_ptr<DistributedEnv>, size_t, size_t, nve::DataType_t>());

    py::class_<DistHostMemBlock, MemBlock, std::shared_ptr<DistHostMemBlock>>(m, "DistHostMemBlock")
        .def(py::init<std::shared_ptr<DistributedEnv>, size_t, size_t, nve::DataType_t>());

    py::class_<UserMemBlock, MemBlock, std::shared_ptr<UserMemBlock>>(m, "UserMemBlock")
        .def(py::init<uint64_t>());

    py::class_<ManagedMemBlock, MemBlock, std::shared_ptr<ManagedMemBlock>>(m, "ManagedMemBlock")
        .def(py::init<size_t, size_t, nve::DataType_t, std::vector<int>>());

    py::enum_<ParameterServerTable::PSType_t>(m, "PSType_t")
        .value("NVHashMap", ParameterServerTable::PSType_t::NVHashMap)
        .value("Abseil", ParameterServerTable::PSType_t::Abseil)
        .value("ParallelHash", ParameterServerTable::PSType_t::ParallelHash)
        .value("Redis", ParameterServerTable::PSType_t::Redis)
        .export_values();

    py::class_<ParameterServerTable, nve::Table, std::shared_ptr<ParameterServerTable> /* <- holder type */>(m, "ParameterServerTable")
        .def(py::init<uint64_t, uint64_t, nve::DataType_t, uint64_t, ParameterServerTable::PSType_t, std::string>(),
            py::arg("num_rows"),
            py::arg("row_elements"),
            py::arg("data_type"),
            py::arg("initial_size") = 0,
            py::arg("ps_type") = ParameterServerTable::PSType_t::NVHashMap,
            py::arg("extra_params") = std::string())
        .def("insert_keys", &ParameterServerTable::insert_keys)
        .def("erase_keys", &ParameterServerTable::erase_keys)
        .def("clear_keys", &ParameterServerTable::clear_keys)
        .def("insert_keys_from_filepath", &ParameterServerTable::insert_keys_from_filepath);

    py::class_<DistributedEnv, PyDistributedEnv /* <--- trampoline */, py::smart_holder>(m, "DistributedEnv")
        .def(py::init<>())
        .def("rank", &DistributedEnv::rank)
        .def("world_size", &DistributedEnv::world_size)
        .def("device_count", &DistributedEnv::device_count)
        .def("local_device", &DistributedEnv::local_device)
        .def("single_host", &DistributedEnv::single_host)
        .def("barrier", &DistributedEnv::barrier)
        .def("broadcast", &DistributedEnv::broadcast)
        .def("all_gather", &DistributedEnv::all_gather);

    m.def("raw_copy",
        [](uintptr_t dst, uintptr_t src, uint64_t size) {
            std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size);
        }, "An auxiliary function to copy from raw pointers");

    m.def("get_dl_tensor", &get_dl_tensor);
    m.def("write_tensor_to_stream", &write_tensor_to_stream);
    m.def("load_tensor_from_stream", &load_tensor_from_stream);
    m.def("insert_keys_from_numpy_file", &insert_keys_from_numpy_file);
    m.def("insert_keys_from_binary_file", &insert_keys_from_binary_file);

#ifdef NVE_WITH_TORCH_BINDINGS
    m.def("register_for_torch",
        [](int64_t layer_id, std::shared_ptr<NVEmbedBinding<int64_t>> sptr) {
            NVELayerRegistry::instance().register_binding(layer_id, std::move(sptr));
        }, py::arg("layer_id"), py::arg("binding"));
    m.def("unregister_from_torch",
        [](int64_t layer_id) {
            NVELayerRegistry::instance().unregister_binding(layer_id);
        }, py::arg("layer_id"));
#endif
}

}  // namespace nve
