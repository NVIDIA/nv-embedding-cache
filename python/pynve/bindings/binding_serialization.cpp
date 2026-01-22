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
 
#include "binding_serialization.hpp"
#include "include/common.hpp"


namespace nve {

PyStreamWrapper::PyStreamWrapper(py::object stream) : stream_(stream) 
{
    tell_func_ = stream_.attr("tell");
    write_func_ = stream_.attr("write");
    seek_func_ = stream_.attr("seek");
    read_func_ = stream_.attr("readinto");
    flush_func_ = stream_.attr("flush");
}

void PyStreamWrapper::write(const void* data, size_t size) 
{
    for (size_t data_written = 0; data_written < size; data_written += write_func_(py::memoryview::from_memory(static_cast<const char*>(data)+data_written, static_cast<ssize_t>(size-data_written))).cast<size_t>());
}

uint64_t PyStreamWrapper::seek(uint64_t offset) 
{
    return seek_func_(offset).cast<uint64_t>();
}

uint64_t PyStreamWrapper::tell() 
{
    return tell_func_().cast<uint64_t>();
}

uint64_t PyStreamWrapper::size()
{
    py::object current_pos = tell_func_();
    seek_func_(0, 2);  // SEEK_END
    uint64_t size = tell_func_().cast<uint64_t>();
    seek_func_(current_pos);  // Restore position
    return size;
}

void PyStreamWrapper::flush() 
{
    flush_func_();
}

size_t PyStreamWrapper::read(void* data, size_t size)
{ 
    size_t total_bytes_read = 0;
    size_t bytes_read = read_func_(py::memoryview::from_memory(static_cast<char*>(data), static_cast<ssize_t>(size))).cast<size_t>();
    while (bytes_read != 0)
    {
        total_bytes_read += bytes_read;
        bytes_read = read_func_(py::memoryview::from_memory(static_cast<char*>(data)+total_bytes_read, static_cast<ssize_t>(size-total_bytes_read))).cast<size_t>();
    }
    return total_bytes_read;
}

void TensorFileFormat::write_table_file_header(py::object stream)
{
    PyStreamWrapper stream_wrapper(stream);
    stream_wrapper.seek(0);
    stream_wrapper.write(&MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
    stream_wrapper.write(&VERSION_MAJOR, sizeof(VERSION_MAJOR));
    stream_wrapper.write(&VERSION_MINOR, sizeof(VERSION_MINOR));
    stream_wrapper.flush();
}

bool TensorFileFormat::verify_table_header(py::object stream)
{
    PyStreamWrapper stream_wrapper(stream);
    stream_wrapper.seek(0);
    uint32_t version_major, version_minor;
    MagicNumberType magic_number;
    stream_wrapper.read(&magic_number, sizeof(magic_number));
    stream_wrapper.read(&version_major, sizeof(version_major));
    stream_wrapper.read(&version_minor, sizeof(version_minor));
    return magic_number == MAGIC_NUMBER && version_major == VERSION_MAJOR && version_minor == VERSION_MINOR;
}

void TensorFileFormat::write_tensor_to_stream(py::object stream, uint64_t name, const TensorWrapper& tensor)
{
    PyStreamWrapper stream_wrapper(stream);
    TableEntry entry;
    size_t sz = tensor.row * tensor.col * tensor.element_size_in_bytes;
    entry.key = get_key(name, tensor);    
    // offset of next entry is relative to the start of the file
    entry.offset = sz + sizeof(TableEntry) + stream_wrapper.tell();
    entry.length = sz;
    stream_wrapper.write(&entry, sizeof(TableEntry));
    stream_wrapper.write(tensor.data, sz);
    stream_wrapper.flush();
}

void TensorFileFormat::load_tensor_from_stream(py::object stream, uint64_t name, TensorWrapper& tensor)
{
    PyStreamWrapper stream_wrapper(stream);
    // this will read table header and advance the stream position. so next should be an entry
    NVE_CHECK_(verify_table_header(stream), "Incorrect file version");
    uint64_t sz = tensor.row * tensor.col * tensor.element_size_in_bytes;
    TableEntry entry;
    size_t bytes_read = stream_wrapper.read(&entry, sizeof(TableEntry));
    // bytes_read is 0 if the stream is empty
    while (bytes_read != 0)
    {
        NVE_CHECK_(bytes_read == sizeof(TableEntry), "Problem reading table entry");
        if (get_name_from_key(entry.key) == name)
        {
            NVE_CHECK_(entry.length == sz, "Incorrect tensor size");
            size_t bytes_read = stream_wrapper.read(tensor.data, entry.length);
            NVE_CHECK_(bytes_read == entry.length, "Problem reading tensor");
            break;
        }
        else {
            stream_wrapper.seek(entry.offset);
            bytes_read = stream_wrapper.read(&entry, sizeof(TableEntry));
        }
    }
}

TensorFileFormat::TableKey TensorFileFormat::get_key(uint64_t name, const TensorWrapper& tensor) const 
{
    NVE_CHECK_(tensor.col < (1llu << TableKey::ROW_SIZE_BITS), "nve support max 16 bit row size");
    NVE_CHECK_(tensor.row < (1llu << TableKey::ROWS_BITS), "nve support max 44 bit rows");
    NVE_CHECK_(name < (1llu << TableKey::AUX_BITS), "nve support max 32 bit unique name");
    NVE_CHECK_(tensor.element_size_in_bytes < (1llu << TableKey::DATA_TYPE_BITS), "nve support max 2 bit data type");
    TableKey key;
    key.key1 = tensor.col | tensor.element_size_in_bytes << TableKey::ROW_SIZE_BITS | tensor.row << (TableKey::ROW_SIZE_BITS + TableKey::DATA_TYPE_BITS);
    key.key2 = name;
    return key;
}

uint64_t TensorFileFormat::get_name_from_key(const TableKey& key) const 
{
    return key.key2 & ((1llu << TableKey::AUX_BITS) - 1);
}


} // namespace nve
