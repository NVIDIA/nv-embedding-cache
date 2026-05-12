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

#include "serialization.hpp"
#include <cstdint>
#include "third_party/pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;

namespace nve {

class __attribute__((visibility("hidden"))) PyStreamWrapper : public StreamWrapperBase {
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

} // namespace nve
