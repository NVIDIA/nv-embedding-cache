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

/*
 * C++ inference using AOTInductor-compiled NVEmbedding model.
 *
 * Links against libnve-torch-ops.so + libnve-common.so.
 *
 * Flow:
 *   1. LayerDirectory reads metadata.json, creates embedding layers,
 *      loads weights from .nve files, registers in NVELayerRegistry.
 *   2. AOTIModelPackageLoader loads the AOT-compiled model.
 *   3. Run inference.
 *   4. LayerDirectory destructor unregisters layers.
 */

#include <iostream>
#include <string>

#include <third_party/argparse/include/argparse/argparse.hpp>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include "python/pynve/torch_bindings/nve_loader.hpp"

int main(int argc, char* argv[]) {
    argparse::ArgumentParser args("nve_inference", "1.0",
                                  argparse::default_arguments::help);
    args.add_argument("save_dir")
        .help("Path to the exported model directory (metadata.json + weights/ + model.pt2)")
        .default_value(std::string("samples/cpp_inference/output"));
    args.add_argument("-d", "--device")
        .help("CUDA device index")
        .scan<'i', int>()
        .default_value(0);

    try {
        args.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << args;
        return 1;
    }

    std::string save_dir = args.get<std::string>("save_dir");
    int device_index = args.get<int>("--device");

    // ---- Step 1: Load NVE layers (RAII — unregisters on destruction) ----
    std::cout << "Loading NVE layers from " << save_dir << std::endl;
    nve::LayerDirectory dir(save_dir, device_index);
    std::cout << "  Loaded " << dir.size() << " layer(s)" << std::endl;

    // ---- Step 2: Load AOT-compiled model ----
    std::string model_path = save_dir + "/model.pt2";
    std::cout << "Loading AOT model: " << model_path << std::endl;
    torch::inductor::AOTIModelPackageLoader loader(model_path);

    // ---- Step 3: Run inference ----
    auto keys = torch::tensor({0L, 1L, 5L, 10L},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_index));

    std::cout << "Running inference..." << std::endl;
    c10::InferenceMode mode;
    auto outputs = loader.run({keys});

    std::cout << "Output shape: [" << outputs[0].size(0)
              << ", " << outputs[0].size(1) << "]" << std::endl;
    auto out_cpu = outputs[0].to(torch::kCPU);
    auto keys_cpu = keys.to(torch::kCPU);
    for (int64_t i = 0; i < out_cpu.size(0); ++i) {
        std::cout << "  key=" << keys_cpu[i].item<int64_t>()
                  << " -> [" << out_cpu[i][0].item<float>();
        for (int64_t j = 1; j < std::min(out_cpu.size(1), (int64_t)4); ++j)
            std::cout << ", " << out_cpu[i][j].item<float>();
        std::cout << ", ...]" << std::endl;
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
