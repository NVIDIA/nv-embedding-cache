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

#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <cstdint>

namespace nve {

class StreamWrapperBase {
public:
    virtual ~StreamWrapperBase() = default;
    virtual void write(const void* data, size_t size) = 0;
    virtual uint64_t seek(uint64_t offset) = 0;
    virtual uint64_t tell() = 0;
    virtual void flush() = 0;
    virtual uint64_t read(void* data, size_t size) = 0;
    virtual uint64_t size() = 0;
};

class InputFileStreamWrapper : public StreamWrapperBase {
public:
    InputFileStreamWrapper(const std::string& filename);
    ~InputFileStreamWrapper() override;
    void write(const void* data, size_t size) override;
    uint64_t seek(uint64_t offset) override;
    uint64_t tell() override;
    void flush() override;
    uint64_t read(void* data, size_t size) override;
    uint64_t size() override;
private:
    std::string filename_;
    std::ifstream file_;
};

class TensorFileFormatBase {
public:
    virtual void load_batch(uint64_t batch, void* data) = 0;
    virtual uint64_t get_num_rows() const = 0; // return number of rows in the file (considering it represents a 2D table)
    virtual void reset() = 0;
    virtual ~TensorFileFormatBase() = default;
};

class BinaryTensorFileFormat : public TensorFileFormatBase {
public:
    BinaryTensorFileFormat(std::shared_ptr<StreamWrapperBase> stream, uint64_t row_size_in_bytes);
    void load_batch(uint64_t batch, void* data) override;
    uint64_t get_num_rows() const override;
    void reset() override;
    ~BinaryTensorFileFormat() override = default;
private:
    std::shared_ptr<StreamWrapperBase> stream_;
    uint64_t row_size_;
};

class NumpyTensorFileFormat : public TensorFileFormatBase {
public:
    NumpyTensorFileFormat(std::shared_ptr<StreamWrapperBase> stream);
    void load_batch(uint64_t batch, void* data) override;
    uint64_t get_num_rows() const override;
    uint64_t get_row_size_in_bytes() const; 
    std::vector<uint64_t> get_shape() const;
    void reset() override;
    ~NumpyTensorFileFormat() override = default;

private:
    void load_header();
    static const std::string NPY_MAGIC_NUMBER;

    struct NumpyHeader {
        uint64_t dtype_size_in_bytes;
        std::vector<uint64_t> shape;
        bool fortran_order;
    };

    std::shared_ptr<StreamWrapperBase> stream_;
    NumpyHeader header_;
    uint64_t vec_size_;
    uint64_t row_size_in_bytes_;
    uint64_t offset_;
};

struct TensorWrapper {
    void* data{nullptr};
    uint64_t row{0};
    uint64_t col{0};
    uint64_t element_size_in_bytes{0};
};

using MagicNumberType = uint32_t;

class TensorFileFormat {
public:
    void write_table_file_header(StreamWrapperBase& stream);
    bool verify_table_header(StreamWrapperBase& stream);
    void write_tensor_to_stream(StreamWrapperBase& stream, uint64_t name, const TensorWrapper& tensor);
    void load_tensor_from_stream(StreamWrapperBase& stream, uint64_t name, TensorWrapper& tensor);

    static constexpr MagicNumberType MAGIC_NUMBER = 0x4e564500;

private:
    static constexpr uint32_t VERSION_MAJOR = 0;
    static constexpr uint32_t VERSION_MINOR = 1;

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