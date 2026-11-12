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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tests/inference_test_utils.hpp"
#include "tests/test_utils.hpp"

namespace {

constexpr char kExportScript[] = R"PY(
import os
import sys

import numpy as np
import torch

import pynve.nve as nve
import pynve.torch.nve_layers as nve_layers
import pynve.torch.nve_ps as nve_ps
from pynve.torch.nve_dtypes import quantize_rowwise
from pynve.torch.nve_export import export_aot

save_dir, layer_kind, custom_remote_plugin = sys.argv[1:4]
num_embeddings = 1024
embedding_size = 8
gpu_cache_size = 4 * 1024 * 1024
device = torch.device("cuda")
keys = torch.tensor([0, 1, 5, 10], dtype=torch.int64, device=device)


class Model(torch.nn.Module):
    def __init__(self, layer_type, storage=None, weight_init=None, data_type=torch.float32):
        super().__init__()
        kwargs = {
            "num_embeddings": num_embeddings,
            "embedding_size": embedding_size,
            "data_type": data_type,
            "layer_type": layer_type,
            "optimize_for_training": False,
            "device": device,
        }
        if layer_type in (nve_layers.LayerType.LinearUVM, nve_layers.LayerType.Hierarchical):
            kwargs["gpu_cache_size"] = gpu_cache_size
        if storage is not None:
            kwargs["storage"] = storage
        if weight_init is not None:
            kwargs["weight_init"] = weight_init
        self.emb = nve_layers.NVEmbedding(**kwargs)

    def forward(self, input_keys):
        return self.emb(input_keys)


export_options = {}
if layer_kind == "linear_uvm":
    model = Model(nve_layers.LayerType.LinearUVM)
elif layer_kind == "gpu":
    model = Model(nve_layers.LayerType.GPULayer)
elif layer_kind == "host":
    # HostLayer bound to a CUDA device: host-resident table, CUDA keys/outputs.
    host_weights = torch.arange(num_embeddings, dtype=torch.float32) \
        .unsqueeze(1).expand(num_embeddings, embedding_size).contiguous()
    model = Model(nve_layers.LayerType.HostLayer, weight_init=host_weights)
elif layer_kind == "host_table":
    # HostLayer over an NVHashMap table: rows restored from ps_data_paths by the C++ loader.
    all_keys = torch.arange(num_embeddings, dtype=torch.int64)
    all_values = torch.arange(num_embeddings, dtype=torch.float32) \
        .unsqueeze(1).expand(num_embeddings, embedding_size).contiguous()
    parameter_server = nve_ps.NVEParameterServer(0, embedding_size, torch.float32)
    parameter_server.insert(all_keys, all_values)
    model = Model(nve_layers.LayerType.HostLayer, parameter_server)
    keys_path = os.path.join(save_dir, "keys.npy")
    values_path = os.path.join(save_dir, "values.npy")
    np.save(keys_path, all_keys.numpy())
    np.save(values_path, all_values.numpy())
    export_options["ps_data_paths"] = {"emb": (keys_path, values_path)}
elif layer_kind == "custom_remote":
    parameter_server = nve_ps.NVEParameterServer(
        num_embeddings=num_embeddings,
        embedding_size=embedding_size,
        data_type=torch.float32,
        plugin_name=custom_remote_plugin,
        factory_config={"implementation": "custom_remote"},
        table_config={"key_size": 8, "max_value_size": embedding_size * 4},
    )
    all_keys = torch.arange(num_embeddings, dtype=torch.int64)
    all_values = torch.arange(num_embeddings, dtype=torch.float32) \
        .unsqueeze(1).expand(num_embeddings, embedding_size).contiguous()
    parameter_server.insert(all_keys, all_values)
    model = Model(nve_layers.LayerType.Hierarchical, parameter_server)
    keys_path = os.path.join(save_dir, "keys.npy")
    values_path = os.path.join(save_dir, "values.npy")
    np.save(keys_path, all_keys.numpy())
    np.save(values_path, all_values.numpy())
    export_options["ps_data_paths"] = {"emb": (keys_path, values_path)}
elif layer_kind in ("quant_gpu", "quant_hier"):
    # Constant rows quantize exactly (affine offset = value, scale 1), so row i still decodes to i.
    qtype = nve.DataType_t.QUint8RowwiseF32
    all_values = quantize_rowwise(torch.arange(num_embeddings, dtype=torch.float32)
                                  .unsqueeze(1).expand(num_embeddings, embedding_size), qtype)
    if layer_kind == "quant_gpu":
        model = Model(nve_layers.LayerType.GPULayer, weight_init=all_values, data_type=qtype)
    else:
        all_keys = torch.arange(num_embeddings, dtype=torch.int64)
        parameter_server = nve_ps.NVEParameterServer(0, embedding_size, qtype)
        parameter_server.insert(all_keys, all_values)
        model = Model(nve_layers.LayerType.Hierarchical, parameter_server, data_type=qtype)
        keys_path = os.path.join(save_dir, "keys.npy")
        values_path = os.path.join(save_dir, "values.npy")
        np.save(keys_path, all_keys.numpy())
        np.save(values_path, all_values.numpy())
        export_options["ps_data_paths"] = {"emb": (keys_path, values_path)}
else:
    raise ValueError(f"Unknown layer kind: {layer_kind}")

if layer_kind in ("linear_uvm", "gpu"):
    weights = torch.arange(num_embeddings, dtype=torch.float32, device=device) \
        .unsqueeze(1).expand(num_embeddings, embedding_size)
    model.emb.weight.data.copy_(weights)

export_aot(model, (keys,), save_dir, **export_options)
)PY";

// Every layer kind seeds row i with the value i, so the output is kind-independent.
const std::vector<const char*> kExpectedOutput{
    "Output shape: [4, 8]", "key=0 -> [0, 0, 0, 0", "key=1 -> [1, 1, 1, 1",
    "key=5 -> [5, 5, 5, 5", "key=10 -> [10, 10, 10, 10"};

class NVEExportInferenceTest : public ::testing::TestWithParam<const char*> {};

TEST_P(NVEExportInferenceTest, ExportAndRunCppInference) {
  // These kinds build a default NVEParameterServer, which uses the NVHM plugin.
  const std::string layer_kind = GetParam();
  if (layer_kind == "host_table" || layer_kind == "quant_hier") {
    SKIP_IF_NVHM_UNAVAILABLE();
  }

  const nve_test::TemporaryDirectory export_directory("nve-export-inference-test");
  const std::string save_dir = export_directory.path().string();

  const nve_test::ProcessResult export_result = nve_test::run_process(
      {NVE_TEST_PYTHON_EXECUTABLE, "-c", kExportScript, save_dir, GetParam(),
       NVE_TEST_CUSTOM_REMOTE_PLUGIN});
  ASSERT_EQ(export_result.exit_code, 0)
      << "Python export failed:\n" << export_result.output;

  const nve_test::ProcessResult inference_result =
      nve_test::run_process({NVE_TEST_INFERENCE_EXECUTABLE, save_dir});
  ASSERT_EQ(inference_result.exit_code, 0)
      << "C++ inference failed:\n" << inference_result.output;
  nve_test::expect_output_contains(inference_result.output, kExpectedOutput);
}

INSTANTIATE_TEST_SUITE_P(LayerTypes, NVEExportInferenceTest,
                         ::testing::Values("linear_uvm", "gpu", "host", "host_table",
                                           "custom_remote", "quant_gpu", "quant_hier"));

}  // namespace
