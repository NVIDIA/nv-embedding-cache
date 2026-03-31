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

#include <third_party/argparse/include/argparse/argparse.hpp>
#include <embedding_layer.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <insert_heuristic.hpp>
#include <memory>
#include <string>
#include <vector>
#include <serialization.hpp>
#include <filesystem>
#include <stdexcept>
#include <cuda.h>

bool ParseCommandline(argparse::ArgumentParser& args, int argc, const char* const argv[]) {
  args.add_argument("--verbose").help("Verbose mode").default_value(false).implicit_value(true);
  args.add_argument("-kf", "--keys_file")
      .help("NPY file containing unique int64 keys exported from Dynamic Embedding")
      .default_value(std::string("/tmp/keys.npy"));
  args.add_argument("-vf", "--values_file")
      .help("NPY file containing values (data vectors) exported from Dynamic Embedding")
      .default_value(std::string("/tmp/values.npy"));
  args.add_argument("-bs", "--batch_size")
      .help("Number of keys in a single insert batch")
      .scan<'u', unsigned>()
      .default_value(1024u);
  args.add_argument("-gcs", "--gpu_cache_size")
      .help("GPU cache size in GB")
      .scan<'g', float>()
      .default_value(1.f);
  try {
    args.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    return false;
  }
  return true;
}

void LogVerbose(bool verbose, std::string str) {
  if (verbose) {
    std::cout << str << std::endl;
  }
}

int main(int argc, char* argv[]) {
  try {
    // Parse commandline arguments
    argparse::ArgumentParser args("Import Sample");
    if (!ParseCommandline(args, argc, argv)) {
      std::cerr << "Failed parsing commandline arguments!" << std::endl;
      return 1;
    }

    using IndexT = int64_t;
    constexpr uint64_t GB = 1ul << 30;

    const bool verbose = args.get<bool>("--verbose");
    const std::string keys_filename = args.get<std::string>("--keys_file");
    const std::string values_filename = args.get<std::string>("--values_file");
    const uint64_t batch_size = static_cast<uint64_t>(args.get<unsigned>("--batch_size"));
    const uint64_t gpu_cache_size = static_cast<uint64_t>(args.get<float>("--gpu_cache_size") * GB);

    // Load keys/values
    LogVerbose(verbose, std::string("Importing files (" + keys_filename + ", " + values_filename + ")"));
    if (!std::filesystem::is_regular_file(keys_filename)) {
        throw std::invalid_argument("Invalid keys file");
    }
    if (!std::filesystem::is_regular_file(values_filename)) {
        throw std::invalid_argument("Invalid values file");
    }
    std::shared_ptr<nve::StreamWrapperBase> keys_sw = std::make_shared<nve::InputFileStreamWrapper>(keys_filename);
    std::shared_ptr<nve::StreamWrapperBase> values_sw = std::make_shared<nve::InputFileStreamWrapper>(values_filename);
    nve::NumpyTensorFileFormat keys_file_reader(keys_sw);
    nve::NumpyTensorFileFormat values_file_reader(values_sw);
    auto keys_shape = keys_file_reader.get_shape();
    auto values_shape = values_file_reader.get_shape();
    if ((keys_shape.size() != 1) ||
        (values_shape.size() != 2) ||
        (keys_shape[0] != values_shape[0])) {
        throw std::runtime_error("Keys/values shape mismatch");
    }
    const auto num_keys = keys_shape[0];
    const auto key_size = keys_file_reader.get_row_size_in_bytes();
    const auto row_size = values_file_reader.get_row_size_in_bytes();
    if (key_size != 8) {
        throw std::runtime_error("Invalid keys size (expected 8 bytes per key)");
    }
    LogVerbose(verbose, std::string("Found ") + std::to_string(num_keys) + " keys");

    // Create GPU table (GPU cache)
    LogVerbose(verbose, std::string("Creating GPU table"));
    constexpr int device_id(0);
    nve::GPUTableConfig gpu_table_cfg;
    gpu_table_cfg.device_id = device_id;
    gpu_table_cfg.cache_size = gpu_cache_size;
    gpu_table_cfg.max_modify_size = static_cast<int64_t>(batch_size);
    gpu_table_cfg.row_size_in_bytes = static_cast<int64_t>(row_size);
    gpu_table_cfg.uvm_table = nullptr;
    auto gpu_tab = std::make_shared<nve::GpuTable<IndexT>>(gpu_table_cfg);

    // Create host table
    LogVerbose(verbose, std::string("Creating Host table"));
    std::vector<std::string> plugin_names{"nvhm"};
    nve::load_host_table_plugins(plugin_names.begin(), plugin_names.end());
    // We don't limit the size of the host table (to do that, use the overflow policy arg)
    nlohmann::json nvhm_conf = {{"mask_size", 8},
                                {"key_size", sizeof(IndexT)},
                                {"max_value_size", row_size},
                                {"value_dtype", "float32"},
                                {"num_partitions", 1},
                                {"initial_capacity", 1024},
                                {"value_alignment", 32},
                            };
    nve::load_host_table_plugins(plugin_names.begin(), plugin_names.end());
    nve::host_table_factory_ptr_t nvhm_fac{
        nve::create_host_table_factory(R"({"implementation": "nvhm_map"})"_json)};
    auto host_tab = nvhm_fac->produce(0, nvhm_conf);

    // Create hierarchical embedding layer
    LogVerbose(verbose, std::string("Creating Hierarchical embedding layer"));
    nve::HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
    // We use NeverInsertHeuristic to disable GPU cache from auto-inserting, so we can verify we only hit in the host table
    // To allow the gpu table to auto-insert, provide some insert heuristic (or leave as nullptr for the default).
    // For example this would use the default insert heuristic
    // layer_cfg.insert_heuristic = std::make_shared<nve::DefaultInsertHeuristic>(std::vector<float>{0.75f, 0.75f});
    layer_cfg.insert_heuristic = std::make_shared<nve::NeverInsertHeuristic>();
    std::vector<std::shared_ptr<nve::Table>> tables{gpu_tab, host_tab};
    auto emb_layer = std::make_shared<nve::HierarchicalEmbeddingLayer<IndexT>>(layer_cfg, tables);

    // Create an execution context
    auto ctx = emb_layer->create_execution_context(
        0 /*lookup stream*/,
        0 /*modify stream*/,
        nullptr /*thread pool*/,
        nullptr /*allocator*/
    );

    // Import key-value pairs from files to host table
    LogVerbose(verbose, std::string("Importing keys/values"));
    std::vector<int8_t> keys_buffer(key_size * batch_size);
    const uint64_t output_buffer_size = row_size * batch_size;
    std::vector<int8_t> values_buffer(output_buffer_size);
    uint64_t num_batches = num_keys / batch_size;
    uint64_t rem = num_keys % batch_size;
    for (size_t i = 0; i < num_batches; i++) {
        keys_file_reader.load_batch(batch_size, keys_buffer.data());
        values_file_reader.load_batch(batch_size, values_buffer.data());
        emb_layer->insert(ctx, static_cast<int64_t>(batch_size), keys_buffer.data(), static_cast<int64_t>(row_size), static_cast<int64_t>(row_size), values_buffer.data(), 1/* table id 1 is the host cache*/);
        ctx->wait();
    }
    if (rem > 0) {
        keys_file_reader.load_batch(rem, keys_buffer.data());
        values_file_reader.load_batch(rem, values_buffer.data());
        emb_layer->insert(ctx, static_cast<int64_t>(rem), keys_buffer.data(), static_cast<int64_t>(row_size), static_cast<int64_t>(row_size), values_buffer.data(), 1/* table id 1 is the host cache*/);
        ctx->wait();
    }

    // Check all imports succeeded
    LogVerbose(verbose, std::string("Checking import"));
    // Reset readers to first element
    keys_file_reader.reset();
    values_file_reader.reset();
    std::vector<int8_t> output_buffer(output_buffer_size);
    NVE_CHECK_(cudaHostRegister(output_buffer.data(), output_buffer_size, cudaHostRegisterDefault));
    std::vector<float> hitrates(2);
    for (size_t i = 0; i < num_batches; i++) {
        keys_file_reader.load_batch(batch_size, keys_buffer.data());
        values_file_reader.load_batch(batch_size, values_buffer.data());
        emb_layer->lookup(ctx, static_cast<int64_t>(batch_size), keys_buffer.data(), output_buffer.data(), static_cast<int64_t>(row_size), nullptr /*hitmask*/, nullptr/*pool_params*/, hitrates.data());
        ctx->wait();
        if (hitrates[1] != 1.0f) {
            throw std::runtime_error("Lookup failed to find keys!");
        }
        if (values_buffer != output_buffer) {
          throw std::runtime_error("Lookup returned wrong values!");
        }
    }
    if (rem > 0) {
        keys_file_reader.load_batch(rem, keys_buffer.data());
        values_file_reader.load_batch(rem, values_buffer.data());
        emb_layer->lookup(ctx, static_cast<int64_t>(rem), keys_buffer.data(), output_buffer.data(), static_cast<int64_t>(row_size), nullptr /*hitmask*/, nullptr/*pool_params*/, hitrates.data());
        ctx->wait();
        if (hitrates[1] != 1.0f) {
            throw std::runtime_error("Lookup failed to find keys!");
        }
        if (!std::equal(values_buffer.begin(), values_buffer.begin() + static_cast<long>(rem), output_buffer.begin())) {
          throw std::runtime_error("Lookup returned wrong values!");
        }
    }

    LogVerbose(verbose, std::string("Done!"));
    NVE_CHECK_(cudaHostUnregister(output_buffer.data()));
  } catch (const std::exception& e) {
    std::cerr << "Exception Caught! : ";
    std::cerr << e.what() << std::endl;
    return -1;
  }
  return 0;
}
