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

} // namespace nve
