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

#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "serialization.hpp"
#include <cstdint>

namespace py = pybind11;

namespace nve {

using MagicNumberType = uint32_t;

class PyStreamWrapper : public StreamWrapperBase {
public:
    PyStreamWrapper(py::object stream);
    
    void write(const void* data, size_t size) override;
    uint64_t seek(uint64_t offset) override;
    uint64_t tell() override;
    void flush() override;
    uint64_t read(void* data, size_t size) override;
    uint64_t size() override;

private:
    py::object stream_;
    py::function tell_func_;
    py::function write_func_;
    py::function seek_func_;
    py::function read_func_;
    py::function flush_func_;
};

struct TensorWrapper {
    void* data{nullptr};
    uint64_t row{0};
    uint64_t col{0};
    uint64_t element_size_in_bytes{0};
};

// class for writing and reading nve table format
class TensorFileFormat {
public:
    void write_table_file_header(py::object stream);
    bool verify_table_header(py::object stream);
    void write_tensor_to_stream(py::object stream, uint64_t name, const TensorWrapper& tensor);
    void load_tensor_from_stream(py::object stream, uint64_t name, TensorWrapper& tensor);

    static constexpr MagicNumberType MAGIC_NUMBER = 0x4e564500;

private:
    static constexpr uint32_t VERSION_MAJOR = 0;
    static constexpr uint32_t VERSION_MINOR = 1;

    // Not using bitfileds because of weird gcc errors with Wconversion
    struct TableKey {
        static constexpr uint64_t ROW_SIZE_BITS = 16;
        static constexpr uint64_t DATA_TYPE_BITS = 4;
        static constexpr uint64_t ROWS_BITS = 64 - (ROW_SIZE_BITS + DATA_TYPE_BITS);
        static constexpr uint64_t AUX_BITS = 32;
        static constexpr uint64_t RESERVED_BITS = 64 - (AUX_BITS);

        uint64_t key1{0};
        uint64_t key2{0};
    };

    struct TableEntry {
        TableKey key;
        uint64_t offset{0};
        uint64_t length{0};
    };

    TableKey get_key(uint64_t name, const TensorWrapper& tensor) const;
    uint64_t get_name_from_key(const TableKey& key) const;
};

} // namespace nve
