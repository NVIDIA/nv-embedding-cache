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

#include <assert.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "NvInfer.h"
#include "engine_harness.h"

class ErrorLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        std::cerr << msg << std::endl;
        break;
      default:
        break;  // do nothing
    }
  }
} gLogger;

inline int64_t volume(const nvinfer1::Dims& dims, const nvinfer1::Dims& strides, int vecDim,
                      int comps, int batch) {
  int64_t maxNbElems = 1;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    // Get effective length of axis.
    int64_t d = dims.d[i];
    // Any dimension is 0, it is an empty tensor.
    if (d == 0) {
      return 0;
    }
    if (i == vecDim) {
      d = (d + comps - 1) / comps;
    }
    maxNbElems = std::max(maxNbElems, d * strides.d[i]);
  }
  return static_cast<int64_t>(maxNbElems) * batch * (vecDim < 0 ? 1 : comps);
}

EngineHarness::EngineHarness(const std::string& filename, unsigned numExecutionContexts) {
  // Load engine
  std::ifstream engineFile(filename, std::ios::binary);
  assert(engineFile);
  engineFile.seekg(0, std::ifstream::end);
  auto fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);
  std::vector<char> engineData(static_cast<size_t>(fsize));
  engineFile.read(engineData.data(), fsize);
  assert(engineFile);

  auto runtime = nvinfer1::createInferRuntime(gLogger);
  auto engine = runtime->deserializeCudaEngine(engineData.data(), static_cast<size_t>(fsize));

  // Create contexts
  for (unsigned i = 0; i < numExecutionContexts; i++) {
    m_executionContexts.emplace_back(engine->createExecutionContext());
  }

  // Collect bindings
  const int32_t nOptProfiles = engine->getNbOptimizationProfiles();
  const int32_t nBindings = engine->getNbIOTensors();
  const int32_t bindingsInProfile = nOptProfiles > 0 ? nBindings / nOptProfiles : 0;
  const int32_t endBindingIndex = bindingsInProfile ? bindingsInProfile : engine->getNbIOTensors();

  assert(nOptProfiles <= 1);
  auto context = m_executionContexts.at(0);

  for (int b = 0; b < endBindingIndex; ++b) {
    const auto name = engine->getIOTensorName(b);
    if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      [[maybe_unused]] auto dims = engine->getTensorShape(name);
      [[maybe_unused]] const bool isDynamicInput =
          std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; }) ||
          engine->isShapeInferenceIO(name);
      assert(!isDynamicInput);
    }

    const auto dims = engine->getTensorShape(name);
    const auto vecDim = engine->getTensorVectorizedDim(name);
    const auto comps = engine->getTensorComponentsPerElement(name);
    const auto dataType = engine->getTensorDataType(name);
    const auto strides = context->getTensorStrides(name);
    const int32_t batch = 1;
    const auto vol = volume(dims, strides, vecDim, comps, batch);
    const auto isInput = (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);

    int dataTypeSize = 0;
    switch (dataType) {
      case nvinfer1::DataType::kINT8:
        dataTypeSize = 1;
        break;
      case nvinfer1::DataType::kHALF:
        dataTypeSize = 2;
        break;
      case nvinfer1::DataType::kFLOAT:
      case nvinfer1::DataType::kINT32:
        dataTypeSize = 4;
        break;
      default:
        assert(0);
    }

    m_ioBindings.push_back({name, isInput,
                            static_cast<size_t>(vol) * static_cast<size_t>(dataTypeSize), dataType,
                            dims});
  }
}

std::vector<IOBinding> EngineHarness::GetIOBindings() const { return m_ioBindings; }

bool EngineHarness::Enqueue(cudaStream_t stream, unsigned contextIndex,
                            const std::vector<void*>& ioBuffers) {
  assert(ioBuffers.size() == m_ioBindings.size());
  auto ctx = m_executionContexts.at(contextIndex);
  for (size_t i = 0; i < ioBuffers.size(); i++) {
    bool result = ctx->setTensorAddress(m_ioBindings.at(i).name.c_str(), ioBuffers[i]);
    if (!result) {
      return false;
    }
  }
  return ctx->enqueueV3(stream);
}
