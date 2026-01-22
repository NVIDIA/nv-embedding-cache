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

#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"

#ifndef gpuErrChk
#define gpuErrChk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#endif  // gpuErrChk

struct IOBinding {
  std::string name;
  bool isInput;
  size_t sizeInBytes;
  nvinfer1::DataType dataType;
  nvinfer1::Dims dims;
};

class EngineHarness {
 public:
  EngineHarness(const std::string& filename, unsigned numExecutionContexts = 1);
  ~EngineHarness() = default;
  std::vector<IOBinding> GetIOBindings() const;
  bool Enqueue(cudaStream_t stream, unsigned contextIndex, const std::vector<void*>& ioBuffers);

 private:
  std::vector<IOBinding> m_ioBindings;
  std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> m_executionContexts;
};
