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

#include <engine_harness.h>

#include <embedding_layer.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <linear_host_table.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <insert_heuristic.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <third_party/argparse/include/argparse/argparse.hpp>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>

#include "../tests/embedding_layer/mock_host_table.hpp"
#include "workload_runner.hpp"

bool ParseCommandline(argparse::ArgumentParser& args, int argc, const char* const argv[]) {
  // Generic params
  args.add_argument("--verbose").help("Verbose mode").default_value(false).implicit_value(true);
  args.add_argument("-cm", "--collect_metrics")
      .help("Collect metrics during run.")
      .default_value(false)
      .implicit_value(true);
  args.add_argument("-csv", "--csv_filename")
      .help("Append metrics to CSV file.")
      .default_value(std::string(""));

  // Cache configurations
  args.add_argument("-gcs", "--gpu_cache_size")
      .help("GPU cache size in GB (<=0 implies no cache)")
      .scan<'g', float>()
      .default_value(1.f);
  args.add_argument("-hcs", "--host_cache_size")
      .help("Host cache size in GB (<=0 implies no cache)")
      .scan<'g', float>()
      .default_value(10.f);
  args.add_argument("-nl", "--num_layers")
      .help("Number of layers")
      .scan<'u', unsigned>()
      .default_value(1u);
  args.add_argument("-lt", "--layer_type")
      .help("Layer type to use: 0=Hiearchical, 1=Linear, 2=LinearCPU")
      .scan<'u', unsigned>()
      .default_value(0u);
  args.add_argument("-cht", "--cache_heuristic_target")
      .help("Target hitrate to use for cache insert heuristic (0 means disabled)")
      .scan<'g', float>()
      .default_value(0.f);

  // todo: linear table in sysmem (implies no host cache)
  // todo: json config (different config per layer)
  // todo: multi gpu?
  // todo: index type differetn from int64_t

  // Input generation params
  args.add_argument("-is", "--input_sets")
      .help("Number of input sets to generate")
      .scan<'u', unsigned>()
      .default_value(100u);
  args.add_argument("-r", "--rows")
      .help("Number of rows in every table")
      .scan<'u', unsigned>()
      .default_value(1u << 26);
  args.add_argument("-rs", "--row_size")
      .help("Row size in bytes")
      .scan<'u', unsigned>()
      .default_value(256u);
  args.add_argument("-h", "--hotness")
      .help("Number of indices in every batch")
      .scan<'u', unsigned>()
      .default_value(512u);
  args.add_argument("-bs", "--batch_size")
      .help("Number of batches in every lookup")
      .scan<'u', unsigned>()
      .default_value(512u);
  args.add_argument("-a", "--alpha")
      .help("Alpha to use for index generation")
      .scan<'g', float>()
      .default_value(1.05f);

  // Runtime params
  args.add_argument("-wi", "--warmup_iterations")
      .help("Number of warmup iterations to run")
      .scan<'u', unsigned>()
      .default_value(250u);
  args.add_argument("-ws", "--warmup_sets")
      .help("Number of input sets to generate for warmup")
      .scan<'u', unsigned>()
      .default_value(250u);
  args.add_argument("-i", "--iterations")
      .help("Number of iterations to run")
      .scan<'u', unsigned>()
      .default_value(1000u);
  args.add_argument("-nr", "--num_runners")
      .help("Number of workload runners (parallel inferences)")
      .scan<'u', unsigned>()
      .default_value(2u);
  args.add_argument("-te", "--trt_engine")
      .help("TensorRT engine file.")
      .default_value(std::string(""));
  args.add_argument("-pl", "--ps_latency")
      .help("Amount of nanoseconds to wait for evey key looked up on the mock PS")
      .scan<'u', unsigned>()
      .default_value(0u);
  args.add_argument("-gm", "--gpu_modify")
      .help("Execute modify operations (insert/update/accumulate) on gpu")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("-li", "--logging_interval")
      .help("Print hitrates every N iterations (0 = disabled).")
      .default_value(static_cast<unsigned>(0u))
      .scan<'u', unsigned>();
  args.add_argument("-dc", "--disable_copy_output")
      .help("Disable copying the outputs back to the host.")
      .default_value(false)
      .implicit_value(true);
  args.add_argument("-km", "--kernel_mode")
      .scan<'u', unsigned>()
      .default_value(0u);
  // Assumptions (can add params for later)
  // int64_t indices
  // fp16 value elements (don't care without pooling/accumulation

  // Parse and handle errors
  try {
    args.parse_args(argc, argv);
    // TODO: validate args
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
    argparse::ArgumentParser args("Cache Wrapper Sample");
    if (!ParseCommandline(args, argc, argv)) {
      std::cerr << "Failed parsing commandline arguments!" << std::endl;
      return 1;
    }

    using IndexT = int64_t;
    constexpr uint64_t GB = 1ul << 30;

    const bool verbose = args.get<bool>("--verbose");
    const bool collect_metrics = args.get<bool>("--collect_metrics");
    const std::string csv_filename = args.get<std::string>("--csv_filename");
    const uint64_t gpu_cache_size = static_cast<uint64_t>(args.get<float>("--gpu_cache_size") * GB);
    const uint64_t host_cache_size = static_cast<uint64_t>(args.get<float>("--host_cache_size") * GB);
    const float insert_target_hitrate = args.get<float>("--cache_heuristic_target");

    const uint64_t num_inputs = args.get<unsigned>("--input_sets");
    const uint64_t num_rows = args.get<unsigned>("--rows");
    const uint64_t row_size = args.get<unsigned>("--row_size");
    const uint64_t hotness = args.get<unsigned>("--hotness");
    const uint64_t batch_size = args.get<unsigned>("--batch_size");
    const float alpha = args.get<float>("--alpha");
    const uint64_t num_layers = args.get<unsigned>("--num_layers");

    const uint64_t num_warmup_iterations = args.get<unsigned>("--warmup_iterations");
    const uint64_t num_warmup_inputs = args.get<unsigned>("--warmup_sets");
    const uint64_t num_iterations = args.get<unsigned>("--iterations");
    const uint64_t num_runners = args.get<unsigned>("--num_runners");
    const std::string trt_engine_filename = args.get<std::string>("--trt_engine");
    const uint64_t ps_latency = args.get<unsigned>("--ps_latency");

    const uint64_t logging_interval = args.get<unsigned>("--logging_interval");
    const bool disable_copy_output = args.get<bool>("--disable_copy_output");
    const bool modify_on_gpu = args.get<bool>("--gpu_modify");
    const uint64_t kernel_mode = args.get<unsigned>("--kernel_mode");

    // create layers
    const int device_id(0);  // todo multi device
    std::vector<std::shared_ptr<nve::EmbeddingLayerBase>> layers;
    std::vector<void*> cuda_host_allocations;
    std::vector<void*> malloc_host_allocations;

    switch(args.get<unsigned>("--layer_type")) {
      case 0: // Hierarchical
      {
        nve::GPUTableConfig gpu_cfg;
        gpu_cfg.device_id = device_id;
        gpu_cfg.cache_size = static_cast<int64_t>(gpu_cache_size);
        gpu_cfg.max_modify_size = (1l << 20);
        gpu_cfg.row_size_in_bytes = row_size;
        gpu_cfg.uvm_table = nullptr;
        
        const int64_t num_partitions = 1; // Single partition is better for inference
        const int64_t keys_per_partition = host_cache_size / row_size / num_partitions;

        std::vector<std::string> plugin_names{"nvhm"};
        nlohmann::json nvhm_conf = {
          {"key_size", sizeof(IndexT)},
          {"max_value_size", row_size},
          {"num_partitions", num_partitions},
          {"base_queue_index", 0},
          {"value_alignment", 16},
          {"overflow_policy",
            {
              {"handler", "evict_random"},
              {"overflow_margin", keys_per_partition},
              {"resolution_margin", 0.8}
            }
          }
        };
        nve::load_host_table_plugins(plugin_names.begin(), plugin_names.end());
        nve::host_table_factory_ptr_t nvhm_fac{
          nve::create_host_table_factory(R"({"implementation": "nvhm_map"})"_json)};
          
        using layer_type = nve::HierarchicalEmbeddingLayer<IndexT>;
        for (uint64_t i = 0; i < num_layers; i++) {
          LogVerbose(verbose, std::string("Creating layer (" + std::to_string(i) + ")"));
    
          // Create gpu cache
          auto gpu_tab = std::make_shared<nve::GpuTable<IndexT>>(gpu_cfg);
    
          // Create host cache
          auto nvhm_tab = nvhm_fac->produce(4711, nvhm_conf);
    
          // Create mock ps
          nve::HostTableConfig remote_cfg;
          remote_cfg.max_value_size = row_size;
          auto mock_remote = std::make_shared<nve::MockHostTable<IndexT>>(
            remote_cfg, false /*functional_ref*/, nullptr, ps_latency);
    
          // Create embedding layer
          layer_type::Config layer_cfg;
          if (insert_target_hitrate > 0.f) {
            layer_cfg.insert_heuristic = std::make_shared<nve::DefaultInsertHeuristic>(std::vector<float>{insert_target_hitrate, insert_target_hitrate, insert_target_hitrate});
          } else {
            layer_cfg.insert_heuristic = std::make_shared<nve::NeverInsertHeuristic>();
          }
          std::vector<std::shared_ptr<nve::Table>> tables{gpu_tab, nvhm_tab, mock_remote};
          auto emb_layer = std::make_shared<layer_type>(layer_cfg, tables);
          layers.push_back(emb_layer);
        }
      }
      break;
      case 1: // Linear
      {
        for (uint64_t i = 0; i < num_layers; i++) {
          LogVerbose(verbose, std::string("Creating layer (" + std::to_string(i) + ")"));

          // Allocate the linear table in SysMem
          const auto table_size = num_rows * row_size;
          LogVerbose(verbose, std::string("Allocating sysmem table (" + std::to_string(float(table_size)/GB) + " GB)"));
          void* ptr(nullptr);
          nve::GetDefaultAllocator()->host_allocate(&ptr, table_size);
          if (!ptr) {
            throw std::runtime_error("Failed to allocated sysmem table!");
          } else {
            cuda_host_allocations.push_back(ptr);
          }
          
          nve::GPUTableConfig gpu_tab_cfg;
          gpu_tab_cfg.device_id = device_id;
          gpu_tab_cfg.cache_size = static_cast<int64_t>(gpu_cache_size);
          gpu_tab_cfg.max_modify_size = (1l << 20);
          gpu_tab_cfg.row_size_in_bytes = row_size;
          gpu_tab_cfg.uvm_table = ptr;
          gpu_tab_cfg.count_misses = true;
          gpu_tab_cfg.modify_on_gpu = modify_on_gpu;
          gpu_tab_cfg.kernel_mode_type = kernel_mode;
          
          // Create gpu cache
          auto gpu_tab = std::make_shared<nve::GpuTable<IndexT>>(gpu_tab_cfg);

          // Create embedding layer
          typename nve::LinearUVMEmbeddingLayer<IndexT>::Config layer_cfg;
          if (insert_target_hitrate > 0.f) {
            layer_cfg.insert_heuristic = std::make_shared<nve::DefaultInsertHeuristic>(std::vector<float>({insert_target_hitrate, insert_target_hitrate, insert_target_hitrate}));
          } else {
            layer_cfg.insert_heuristic = std::make_shared<nve::NeverInsertHeuristic>();
          }
          auto emb_layer = std::make_shared<nve::LinearUVMEmbeddingLayer<IndexT>>(layer_cfg, gpu_tab);
          layers.push_back(emb_layer);
        }
      }
      break;
      case 2: // LinearCPU
      {
        for (uint64_t i = 0; i < num_layers; i++) {
          LogVerbose(verbose, std::string("Creating layer (" + std::to_string(i) + ")"));

          // Allocate the linear table in SysMem
          const auto table_size = num_rows * row_size;
          LogVerbose(verbose, std::string("Allocating sysmem table (" + std::to_string(float(table_size)/GB) + " GB)"));
          void* ptr(nullptr);
          nve::GetDefaultAllocator()->host_allocate(&ptr, table_size);
          if (!ptr) {
            throw std::runtime_error("Failed to allocated sysmem table!");
          } else {
            std::memset(ptr, 1, table_size);
            cuda_host_allocations.push_back(ptr);
          }

          // Create the linear host table
          nve::LinearHostTableConfig table_cfg;
          table_cfg.max_threads = std::numeric_limits<int64_t>::max(); // will use all available threads in the thread pool
          table_cfg.max_value_size = row_size;
          table_cfg.value_dtype = nve::DataType_t::Float32;
          table_cfg.emb_table = ptr;
          auto host_tab = std::make_shared<nve::LinearHostTable<IndexT>>(table_cfg);

          // Create an embedding layer
          using layer_type = nve::HierarchicalEmbeddingLayer<IndexT>;
          layer_type::Config layer_cfg;
          layer_cfg.insert_heuristic = std::make_shared<nve::NeverInsertHeuristic>(); // No insert for the linear host table
          std::vector<std::shared_ptr<nve::Table>> tables{host_tab};
          auto emb_layer = std::make_shared<layer_type>(layer_cfg, tables);
          layers.push_back(emb_layer);
        }
      }
      break;
      default:
        throw std::runtime_error("Invalid Layer Type!");
    }

    // Load TRT engine
    std::shared_ptr<EngineHarness> trt_engine;
    if (trt_engine_filename.size()) {
      trt_engine = std::make_shared<EngineHarness>(trt_engine_filename, num_runners);
    }

    // Create workload runners
    std::vector<std::shared_ptr<WorkloadRunner>> workers;
    for (uint64_t i = 0; i < num_runners; i++) {
      workers.emplace_back(std::make_shared<WorkloadRunner>(num_inputs, num_warmup_inputs, num_rows,
                                                            hotness, batch_size, alpha, row_size,
                                                            num_iterations, layers, trt_engine, i));
    }

    // Cache warmup (sequential)
    LogVerbose(verbose, std::string("Cache warmup (" + std::to_string(num_warmup_iterations) + ")"));
    auto warmup_start = std::chrono::high_resolution_clock::now();
    for (auto& w : workers) {
      w->WarmupLayerCache(num_warmup_iterations);
    }
    auto warmup_stop = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(warmup_stop - warmup_start).count();
    LogVerbose(verbose, std::string("Warmup done (" + std::to_string(usec/1000) + "." + std::to_string(usec%1000) + " ms)"));

    // inference
    LogVerbose(verbose, std::string("Inference (" + std::to_string(num_iterations) + ")"));
    auto inference_start = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<std::thread>> workerThreads;
    std::vector<WorkloadRunner::Metrics> worker_metrics(workers.size());
    // Pre-allocate metrics arrays to avoid dynamic growth during measurement
    for (uint64_t i = 0; i < workers.size(); i++) {
      worker_metrics[i].resize(num_iterations, num_layers);
    }
    for (uint64_t i = 0; i < workers.size(); i++) {
      workerThreads.emplace_back(std::make_shared<std::thread>([&workers, &worker_metrics, i, num_iterations, disable_copy_output] { 
        workers[i]->Run(num_iterations, &worker_metrics[i], disable_copy_output); 
      }));
    }
    for (auto& t : workerThreads) {
      t->join();
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "Failed to synchronize device!" << std::endl;
    }
    auto inference_stop = std::chrono::high_resolution_clock::now();
    usec = std::chrono::duration_cast<std::chrono::microseconds>(inference_stop - inference_start).count();
    LogVerbose(verbose, std::string("Inference done (total E2E: " + std::to_string(usec/1000) + "." + std::to_string(usec%1000) + " ms)"));

    // Print hitrate metrics at specified intervals
    if (logging_interval > 0) {
      for (uint64_t worker_id = 0; worker_id < worker_metrics.size(); worker_id++) {
        std::cout << "Worker " << worker_id << ":" << std::endl;
        const auto& metrics = worker_metrics[worker_id];
        for (uint64_t iter = 0; iter < num_iterations; iter += logging_interval) {
          for (uint64_t layer = 0; layer < num_layers; layer++) {
            uint64_t idx = iter * num_layers + layer;
            if (idx < metrics.hitrates_gpu.size()) {
              std::cout << "  Iteration " << iter << ", Layer " << layer 
                        << ": Hitrates: " << metrics.hitrates_gpu[idx] 
                        << " " << metrics.hitrates_host[idx] 
                        << " " << metrics.hitrates_remote[idx] << std::endl;
            }
          }
        }
        std::cout << std::endl;  // Blank line between workers
      }
    }

    // Calculate timing metrics
    auto warmup_usec = std::chrono::duration_cast<std::chrono::microseconds>(warmup_stop - warmup_start).count();
    auto inference_usec = std::chrono::duration_cast<std::chrono::microseconds>(inference_stop - inference_start).count();
    double warmup_time_ms = static_cast<double>(warmup_usec) / 1000.0;
    double inference_time_ms = static_cast<double>(inference_usec) / 1000.0;
    double iterations_per_second = static_cast<double>(num_iterations * num_runners) / (static_cast<double>(inference_usec) / 1000000.0);
    
    // Hitrate metrics (calculate on the fly without storing all values)
    struct Stats { double sum = 0.0; float min_val = 0.0; float max_val = 0.0; size_t count = 0; };
    Stats gpu_stats, host_stats, remote_stats;
    
    for (const auto& m : worker_metrics) {
      // GPU hitrates
      for (float v : m.hitrates_gpu) {
        if (gpu_stats.count == 0) { gpu_stats.min_val = gpu_stats.max_val = v; }
        gpu_stats.sum += v;
        gpu_stats.min_val = std::min(gpu_stats.min_val, v);
        gpu_stats.max_val = std::max(gpu_stats.max_val, v);
        gpu_stats.count++;
      }
      // Host hitrates
      for (float v : m.hitrates_host) {
        if (host_stats.count == 0) { host_stats.min_val = host_stats.max_val = v; }
        host_stats.sum += v;
        host_stats.min_val = std::min(host_stats.min_val, v);
        host_stats.max_val = std::max(host_stats.max_val, v);
        host_stats.count++;
      }
      // Remote hitrates
      for (float v : m.hitrates_remote) {
        if (remote_stats.count == 0) { remote_stats.min_val = remote_stats.max_val = v; }
        remote_stats.sum += v;
        remote_stats.min_val = std::min(remote_stats.min_val, v);
        remote_stats.max_val = std::max(remote_stats.max_val, v);
        remote_stats.count++;
      }
    }
    
    double gpu_hitrate_mean = gpu_stats.count > 0 ? gpu_stats.sum / static_cast<double>(gpu_stats.count) : 0.0;
    double host_hitrate_mean = host_stats.count > 0 ? host_stats.sum / static_cast<double>(host_stats.count) : 0.0;
    double remote_hitrate_mean = remote_stats.count > 0 ? remote_stats.sum / static_cast<double>(remote_stats.count) : 0.0;
    
    // Write metrics to CSV file if requested
    if (!csv_filename.empty()) {
      // Create parent directory if it doesn't exist
      std::filesystem::path csv_path(csv_filename);
      if (csv_path.has_parent_path()) {
        std::filesystem::create_directories(csv_path.parent_path());
      }
      
      // Check if file exists to determine if we need to write header
      bool file_exists = std::filesystem::exists(csv_filename);
      bool write_header = !file_exists || (file_exists && std::filesystem::file_size(csv_filename) == 0);
      
      std::ofstream csv_file(csv_filename, std::ios::app);
      if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file " << csv_filename << " for writing!" << std::endl;
      } else {
        // Write header if file is new or empty
        // Parameters use '_' prefix to distinguish from metrics
        if (write_header) {
          csv_file << "_rows,_row_size,_batch_size,_hotness,_alpha,"
                   << "_gpu_cache_size,_host_cache_size,_ps_latency,"
                   << "_num_layers,_layer_type,_cache_heuristic_target,"
                   << "_input_sets,_warmup_iterations,_warmup_sets,"
                   << "_iterations,_num_runners,_gpu_modify,_kernel_mode,"
                   << "warmup_time_ms,inference_time_ms,iterations_per_second,"
                   << "gpu_hitrate_mean,gpu_hitrate_min,gpu_hitrate_max,gpu_hitrate_samples,"
                   << "host_hitrate_mean,host_hitrate_min,host_hitrate_max,host_hitrate_samples,"
                   << "remote_hitrate_mean,remote_hitrate_min,remote_hitrate_max,remote_hitrate_samples"
                   << std::endl;
        }
        
        // Write data row
        csv_file << num_rows << "," << row_size << "," << batch_size << "," 
                 << hotness << "," << alpha << ","
                 << args.get<float>("--gpu_cache_size") << "," 
                 << args.get<float>("--host_cache_size") << "," 
                 << ps_latency << ","
                 << num_layers << "," << args.get<unsigned>("--layer_type") << "," 
                 << insert_target_hitrate << ","
                 << num_inputs << "," << num_warmup_iterations << "," 
                 << num_warmup_inputs << ","
                 << num_iterations << "," << num_runners << "," 
                 << modify_on_gpu << "," << kernel_mode << ","
                 << warmup_time_ms << "," << inference_time_ms << "," 
                 << iterations_per_second << ","
                 << gpu_hitrate_mean << "," << gpu_stats.min_val << "," 
                 << gpu_stats.max_val << "," << gpu_stats.count << ","
                 << host_hitrate_mean << "," << host_stats.min_val << "," 
                 << host_stats.max_val << "," << host_stats.count << ","
                 << remote_hitrate_mean << "," << remote_stats.min_val << "," 
                 << remote_stats.max_val << "," << remote_stats.count
                 << std::endl;
        
        csv_file.close();
        LogVerbose(verbose, std::string("Metrics appended to " + csv_filename));
      }
    }

    // Free resources
    for (auto ptr : cuda_host_allocations) {
      nve::GetDefaultAllocator()->host_free(ptr);
    }
    for (auto ptr : malloc_host_allocations) {
      free(ptr);
    }

  } catch (const std::exception& e) {
    std::cerr << "Exception Caught! : ";
    std::cerr << e.what() << std::endl;
    return -1;
  }
  return 0;
}

