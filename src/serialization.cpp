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

#include "serialization.hpp"
#include <regex>
#include <string>
#include <stdexcept>
#include "common.hpp"

namespace nve {

const std::string NumpyTensorFileFormat::NPY_MAGIC_NUMBER("\x93NUMPY");

InputFileStreamWrapper::InputFileStreamWrapper(const std::string& filename) : filename_(filename) {
    file_.open(filename_, std::ios::binary | std::ios::in);
    NVE_CHECK_(file_.is_open(), "Failed to open file " + filename_);
}

InputFileStreamWrapper::~InputFileStreamWrapper() {
    file_.close();
}

void InputFileStreamWrapper::write(const void* /*data*/, size_t /*size*/) {
    NVE_CHECK_(false, "write is not supported for input file stream wrapper");
}

uint64_t InputFileStreamWrapper::seek(uint64_t offset) {
    file_.seekg(static_cast<std::streamoff>(offset));
    NVE_CHECK_(file_.good(), "Failed to seek in file " + filename_);
    return offset;
}

uint64_t InputFileStreamWrapper::tell() {
    return static_cast<uint64_t>(file_.tellg());
}

uint64_t InputFileStreamWrapper::size() {
    auto streampos = file_.tellg(); // capture current position
    auto endpos = file_.seekg(0, std::ios::end).tellg();
    file_.seekg(streampos); // Restore position
    return static_cast<uint64_t>(endpos);
}

uint64_t InputFileStreamWrapper::read(void* data, size_t size) {  
    file_.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    NVE_CHECK_(file_.good(), "Failed to read from file " + filename_);
    return static_cast<uint64_t>(file_.gcount());
}

void InputFileStreamWrapper::flush() {
    // do nothing 
}

void NumpyTensorFileFormat::load_header() {
    stream_->seek(0);
    std::string magic_number(NPY_MAGIC_NUMBER);
    auto bytes_read = stream_->read(magic_number.data(), NPY_MAGIC_NUMBER.size()*sizeof(char));
    NVE_CHECK_(bytes_read == NPY_MAGIC_NUMBER.size()*sizeof(char), "Problem reading numpy magic number");
    NVE_CHECK_(std::string(magic_number) == NPY_MAGIC_NUMBER, "Incorrect file format");
    uint8_t major, minor;
    bytes_read = stream_->read(&major, sizeof(major));
    NVE_CHECK_(bytes_read == sizeof(major), "Problem reading numpy major version");
    bytes_read = stream_->read(&minor, sizeof(minor));
    NVE_CHECK_(bytes_read == sizeof(minor), "Problem reading numpy minor version");
    uint32_t header_len_size_in_bytes = 0; // numpy 1.0 has 2 bytes woth of header length, numpy 2.0 has 4 bytes
    uint32_t header_len = 0;
    if (major == 1 && minor == 0) {
        header_len_size_in_bytes = 2;
    } else if (major == 2 && minor == 0) {
        header_len_size_in_bytes = 4;
    } else {
        NVE_CHECK_(false, "Unsupported numpy version");
    }

    bytes_read = stream_->read(&header_len, header_len_size_in_bytes);
    NVE_CHECK_(bytes_read == header_len_size_in_bytes, "Problem reading numpy header length");
    char* buffer = new char[header_len];
    bytes_read = stream_->read(buffer, header_len);
    NVE_CHECK_(bytes_read == header_len, "Problem reading numpy header");
    std::string header_str(buffer);
    delete[] buffer;
    {
        // data format string
        std::regex descr_regex("'descr': '([^']+)'");
        std::smatch match;
        NVE_CHECK_(std::regex_search(header_str, match, descr_regex), "Incorrect file format");
        std::string dtype_str = match[1].str();
        if (dtype_str == "<f4") {
            header_.dtype_size_in_bytes = 4;
        } else if (dtype_str == "<f8") {
            header_.dtype_size_in_bytes = 8;
        } else if (dtype_str == "<i4") {
            header_.dtype_size_in_bytes = 4;
        } else if (dtype_str == "<i8") {
            header_.dtype_size_in_bytes = 8;
        } else {
            NVE_CHECK_(false, "Unsupported data type");
        }
    }

    {
        // shape
        std::regex shape_regex("shape': \\(([^)]*)\\)");
        std::smatch match;
        NVE_CHECK_(std::regex_search(header_str, match, shape_regex), "Incorrect file format");
        std::string shape_str = match[1].str();
        std::stringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) 
        {
            try 
            {
                header_.shape.push_back(std::stoul(token));
            } catch (std::exception& e) {
                NVE_LOG_ERROR_("Failed to load shape: ", e.what());
            }
        }
    }

    {
        // fortran order
        std::regex fortran_order_regex("fortran_order': (False|True)");
        std::smatch match;
        NVE_CHECK_(std::regex_search(header_str, match, fortran_order_regex), "Incorrect file format");
        header_.fortran_order = match[1].str() == "True";
    }

    offset_ = stream_->tell();
    NVE_CHECK_(offset_ == header_len + NPY_MAGIC_NUMBER.size() + sizeof(uint8_t) + sizeof(uint8_t) + header_len_size_in_bytes, "Problem reading numpy header");
}

NumpyTensorFileFormat::NumpyTensorFileFormat(std::shared_ptr<StreamWrapperBase> stream) : stream_(stream) {
    load_header();
    vec_size_ = 1;
    for (uint64_t i = 1; i < header_.shape.size(); ++i)
    {
        vec_size_ *= header_.shape[i];
    }
    row_size_in_bytes_ = vec_size_ * header_.dtype_size_in_bytes;
}

void NumpyTensorFileFormat::reset() {
    // reset read location to the first element
    stream_->seek(offset_);
}

void NumpyTensorFileFormat::load_batch(uint64_t batch, void* data) {
    for (uint64_t i = 0; i < batch; ++i) {
        uint64_t read_bytes_ = 0;
        while (read_bytes_ < row_size_in_bytes_) {
            read_bytes_ += stream_->read(static_cast<char*>(data) + i * row_size_in_bytes_ + read_bytes_, row_size_in_bytes_ - read_bytes_);
        }
    }
}

uint64_t NumpyTensorFileFormat::get_num_rows() const {
    return header_.shape[0];
}

std::vector<uint64_t> NumpyTensorFileFormat::get_shape() const {
    return header_.shape;
}

uint64_t NumpyTensorFileFormat::get_row_size_in_bytes() const {
    return row_size_in_bytes_;
}

BinaryTensorFileFormat::BinaryTensorFileFormat(std::shared_ptr<StreamWrapperBase> stream, uint64_t row_size_in_bytes) : stream_(stream), row_size_(row_size_in_bytes) {
}

void BinaryTensorFileFormat::load_batch(uint64_t batch, void* data) {
    uint64_t read_bytes = 0;
    uint64_t total_bytes_to_read = batch * row_size_;
    while (read_bytes < total_bytes_to_read) {
        read_bytes += stream_->read(static_cast<char*>(data) + read_bytes, total_bytes_to_read - read_bytes);
    }
}

uint64_t BinaryTensorFileFormat::get_num_rows() const {
    return stream_->size() / row_size_;
}

void BinaryTensorFileFormat::reset() {
    stream_->seek(0);
}
}