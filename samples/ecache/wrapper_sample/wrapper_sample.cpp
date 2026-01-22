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

#include <chrono>
#include <memory>
#include <thread>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include "cuda_ops/cuda_common.h"
#include "cache_wrapper.h"
#include "cache_taskqueue.h"
#include "training_wave.h"
#include <argparse/include/argparse/argparse.hpp>
#include "communicator.h"
#include <default_allocator.hpp>

bool ParseCommandline(argparse::ArgumentParser& args, int argc, const char * const argv[])
{
    args.add_argument("--verbose")
        .help("Verbose mode")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("-w", "--waves")
        .help("Number of parallel training waves")
        .scan<'u', unsigned>()
        .default_value(3u);
    args.add_argument("-c", "--caches")
        .help("Number of embedding caches")
        .scan<'u', unsigned>()
        .default_value(2u);
    args.add_argument("-is", "--inputs")
        .help("Number of input sets to generate")
        .scan<'u', unsigned>()
        .default_value(100u);
    args.add_argument("-r", "--rows")
        .help("Number of rows in every cache")
        .scan<'u', unsigned>()
        .default_value(1u<<26);
    args.add_argument("-rs", "--row_size")
        .help("Number of elements in every row")
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
    args.add_argument("-i", "--iterations")
        .help("Number of iterations to run")
        .scan<'u', unsigned>()
        .default_value(100u);
    args.add_argument("-wu", "--warmup")
        .help("Number of warmup iterations to run")
        .scan<'u', unsigned>()
        .default_value(10u);
    args.add_argument("-a", "--alpha")
        .help("Alpha to use for index generation")
        .scan<'g', float>()
        .default_value(1.05f);
    args.add_argument("-cs", "--cache_size")
        .help("Cache size in GB")
        .scan<'g', float>()
        .default_value(1.f);
    args.add_argument("-p", "--pooling")
        .help("Perform fixed hotness sum.")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("-cm", "--collect_metrics")
        .help("Collect metrics during run.")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("-csv", "--csv_filename")
        .help("Append metrics to CSV file.")
        .default_value(std::string(""));
    args.add_argument("-dc", "--device_count")
        .help("Number of devices to use")
        .scan<'u', unsigned>()
        .default_value(1u);
    args.add_argument("-hi", "--host_inputs")
        .help("Assume inputs are on host and need copy to the GPU in every iteration.")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("-di", "--dedup_inputs")
        .help("Preforms a de duplication on keys prior to lookup.")
        .default_value(false)
        .implicit_value(true);
    args.add_argument("-hcs", "--host_cache_size")
        .help("Host Cache size in GB")
        .scan<'g', float>()
        .default_value(1.f);
    args.add_argument("-hch", "--host_cache_hit_rate")
        .help("Host Cache hit rate")
        .scan<'g', float>()
        .default_value(.9f);
    args.add_argument("-hct", "--host_cache_threads")
        .help("Number of threads to use for every mock host cache")
        .scan<'u', unsigned>()
        .default_value(16u);
    args.add_argument("-pst", "--param_server_threads")
        .help("Number of threads to use for every mock parameter server")
        .scan<'u', unsigned>()
        .default_value(16u);
    args.add_argument("-psl", "--param_server_latency")
        .help("Minimal latency for parameter server lookup (ms)")
        .scan<'u', unsigned>()
        .default_value(100u);
    args.add_argument("-gl", "--gpu_latency")
        .help("Minimal latency for gpu work during rest of FWD+BWD (ms)")
        .scan<'u', unsigned>()
        .default_value(300u);
    args.add_argument("--num_ranks")
        .help("Number of ranks in K-Sharding algorithm")
        .scan<'u', unsigned>()
        .default_value(1u);
    args.add_argument("--rank")
        .help("The current of the process in a K sharding")
        .scan<'u', unsigned>()
        .default_value(0u);
    args.add_argument("--shared_file")
        .help("Path to a shared file location for all ranks in the group to initialize communication")
        .default_value(std::string(""));
    args.add_argument("--device_id")
        .help("The start ID of the current process when performing K-Sharding assuming other <device_count> devices are contiguous")
        .scan<'i', int>()
        .default_value(-1);
    // Parse and handle errors
    try {
        args.parse_args(argc, argv);
        const float cache_size = args.get<float>("--cache_size");
        if (cache_size <= 0.f)
        {
            std::cerr << "Invalid cache size (must be >0): " << cache_size << std::endl;
            return false;
        }
        const uint64_t device_count = args.get<unsigned>("--device_count");
        {
            int dc = 0;
            NVE_CHECK_(cudaGetDeviceCount(&dc));
            if (static_cast<uint64_t>(dc) < device_count)
            {
                std::cerr << "User requested " << device_count << " devices but found only " << dc << " available devices." << std::endl;
                return false;
            }
        }

        // arguments check for multi rank
        const uint64_t num_ranks = args.get<unsigned>("--num_ranks");
        const uint64_t rank = args.get<unsigned>("--rank");
        const std::string path_to_shared_file = args.get<std::string>("--shared_file");

        if (num_ranks > 1)
        {
            if (path_to_shared_file.empty())
            {
                std::cerr << "No shared file supplied" << std::endl;
                return false;
            }
            if (rank > num_ranks)
            {
                std::cerr << "Current Ranks " << rank << " Greater than num ranks " << num_ranks << std::endl;
                return false;
            }
        }

    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << args;
        return false;
    }
    return true;
}

class DeviceAllocator : public nve::DefaultAllocator
{
public:
    DeviceAllocator(int device_id) : nve::DefaultAllocator(nve::DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD), m_device_id(device_id) {} // 1GB is threshold size for allocating host memory using mmap
    cudaError_t deviceAllocate(void** ptr, size_t sz, int) noexcept override
    {
        return DefaultAllocator::deviceAllocate(ptr, sz, m_device_id);
    }

    cudaError_t deviceFree(void* ptr, int) noexcept override
    {
        return DefaultAllocator::deviceFree(ptr, m_device_id);
    }

private:
    int m_device_id;
};

int main(int argc, char *argv[])
{
    // Parse commandline arguments
    argparse::ArgumentParser args("Cache Wrapper Sample");
    if (!ParseCommandline(args, argc, argv)) {
        std::cerr << "Failed parsing commandline arguments!" << std::endl;
        return 1;
    }

    using IndexT = int64_t;
    using CacheType =ECacheWrapper<IndexT>;
    using CachePtr = std::shared_ptr<CacheType>;
    constexpr uint64_t GB = 1ul<<30;
    [[maybe_unused]] const bool verbose = args.get<bool>("--verbose"); // Unused, leaving getter for completeness
    const uint64_t num_waves = args.get<unsigned>("--waves");
    const uint64_t num_caches = args.get<unsigned>("--caches");
    const uint64_t num_inputs = args.get<unsigned>("--inputs");
    const uint64_t num_rows = args.get<unsigned>("--rows");
    const uint64_t row_size = args.get<unsigned>("--row_size");
    const uint64_t hotness = args.get<unsigned>("--hotness");
    const float alpha = args.get<float>("--alpha");
    const uint64_t batch_size = args.get<unsigned>("--batch_size");
    const uint64_t num_warmup_iterations = args.get<unsigned>("--warmup");
    const uint64_t num_iterations = args.get<unsigned>("--iterations");
    const uint64_t cache_size = static_cast<uint64_t>(args.get<float>("--cache_size") * GB);
    const bool collect_metrics = args.get<bool>("--collect_metrics");
    const bool do_pooling = args.get<bool>("--pooling");
    const std::string csv_filename = args.get<std::string>("--csv_filename");
    const uint64_t device_count = args.get<unsigned>("--device_count");
    const bool host_inputs = args.get<bool>("--host_inputs");
    const bool dedup = args.get<bool>("--dedup_inputs");

    const float host_cache_size = args.get<float>("--host_cache_size");
    const float host_cache_hitrate = args.get<float>("--host_cache_hit_rate");
    const uint64_t host_cache_threads = args.get<unsigned>("--host_cache_threads");
    const uint64_t ps_threads = args.get<unsigned>("--param_server_threads");
    const uint64_t ps_latency = args.get<unsigned>("--param_server_latency");
    const uint64_t gpu_latency = args.get<unsigned>("--gpu_latency");

    const uint64_t num_ranks = args.get<unsigned>("--num_ranks");
    const uint64_t rank = args.get<unsigned>("--rank");
    const int start_device_id = args.get<int>("--device_id");
    const std::string path_to_shared_file = args.get<std::string>("--shared_file");

    const int device_offset = start_device_id > -1 ? start_device_id : static_cast<int>(rank);

    // init cache wrappers
    std::vector<std::vector<CachePtr>> cache_wrappers(device_count);
    std::vector<std::vector<cudaStream_t>> cache_streams(device_count);
    std::vector<std::vector<std::shared_ptr<CacheTaskQueue<IndexT>>>> task_queues(device_count);
    std::vector<nve::allocator_ptr_t> allocators;
    std::vector<std::shared_ptr<Communicator>> communicators;
    const uint64_t key_buffer_size = batch_size * hotness * sizeof(IndexT);
    try
    {
        for (uint64_t dev = 0; dev < device_count; dev++)
        {
            int device_id = device_offset + static_cast<int>(dev);
            ScopedDevice scope_device(device_id);
            // create an allocator for the specific device id
            allocators.push_back(std::make_shared<DeviceAllocator>(device_id));
            communicators.push_back(num_ranks > 1 ? std::make_shared<Communicator>(
                num_ranks,
                rank,
                path_to_shared_file,
                device_id) : nullptr);
            for (uint64_t i=0 ; i<num_caches ; i++)
            {
                cudaStream_t stream;
                NVE_CHECK_(cudaStreamCreate(&stream));
                cache_streams[dev].push_back(stream); // create single stream per cache
                auto cw = std::make_shared<CacheType>(allocators.at(dev));
                std::vector<cudaStream_t> lookup_streams{stream};
                cw->Init(
                    cache_size,
                    row_size,
                    nve::DATATYPE_FP16,
                    lookup_streams,
                    stream,
                    1 << 20);
                cache_wrappers[dev].push_back(cw);

                // Init host cache
                uint64_t host_cache_rows = static_cast<uint64_t>(host_cache_size) * (1ul<<30) / row_size;
                static auto host_cache = std::make_shared<MockHostCache<IndexT>>(
                    host_cache_rows,
                    row_size,
                    host_cache_threads,
                    host_cache_hitrate);
                assert(host_cache);
                static auto param_server = std::make_shared<MockParameterServer<IndexT>>(row_size, ps_threads, ps_latency);
                task_queues[dev].emplace_back(std::make_shared<CacheTaskQueue<IndexT>>(
                    cw,
                    device_id,
                    key_buffer_size,
                    batch_size * hotness * cw->GetConfig().embedWidth,
                    batch_size * hotness,
                    host_cache,
                    param_server));
            }
        }
        
    } catch (std::exception& e) {
        std::cout << "Failed cache wrapper init: " << e.what() << std::endl;
        return 1;
    }

    // create multiple training waves
    std::vector<std::shared_ptr<TrainingWave<IndexT>>> waves;
    for (uint64_t i=0 ; i<num_waves ; i++ )
    {
        int dev = static_cast<int>(i % device_count);
        int device_id = dev + device_offset;
        ScopedDevice scope_device(device_id);
        auto gpu_fwd_bwd = std::make_shared<MockGPUWork>(gpu_latency);
        waves.push_back(std::make_shared<TrainingWave<IndexT>>(
            task_queues[static_cast<size_t>(dev)],
            num_inputs,
            num_rows,
            hotness,
            batch_size,
            alpha,
            device_id,
            host_inputs,
            dedup,
            num_iterations,
            gpu_fwd_bwd,
            communicators[static_cast<size_t>(dev)]));
    }

    // warmup
    std::vector<std::shared_ptr<std::thread>> wave_threads;
    for (auto& w : waves)
    {
        wave_threads.push_back(std::make_shared<std::thread>([&] {w->ProcessWave(num_warmup_iterations, do_pooling, false /*collect metrics*/);}));
    }
    for (auto& t : wave_threads)
    {
        t->join();
    }
    wave_threads.clear();

    for (int dev = 0; dev < static_cast<int>(device_count); dev++)
    {
        ScopedDevice scope_device(dev + device_offset);
        NVE_CHECK_(cudaDeviceSynchronize());
    }
    
    // run
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& w : waves)
    {
        wave_threads.push_back(std::make_shared<std::thread>([&] {w->ProcessWave(num_iterations, do_pooling, collect_metrics);}));
    }
    for (auto& t : wave_threads)
    {
        t->join();
    }
    for (int dev = 0; dev < static_cast<int>(device_count); dev++)
    {
        for (auto ctq : task_queues[static_cast<size_t>(dev)]) {
            ctq->Drain();
        }
        ScopedDevice scope_device(dev + device_offset);
        NVE_CHECK_(cudaDeviceSynchronize());
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto ips = static_cast<double>(num_iterations * hotness * num_waves * num_caches * batch_size) / static_cast<double>(usec) * 1e6;
    std::cout << "Indices Per Sec = " << ips << std::endl;

    if (collect_metrics) {
        std::vector<float> lookup_durations;
        std::vector<float> lookup_e2e_durations;
        std::vector<float> insert_durations;
        std::vector<float> insert_host_durations;
        std::vector<float> scatter_durations;
        std::vector<float> pooling_durations;
        std::vector<float> grads_dedup_durations;
        std::vector<float> accumulate_durations;
        std::vector<float> hitrate;
        std::vector<float> host_cache_lookup_durations;
        std::vector<float> host_cache_accumulate_durations;
        std::vector<float> ps_lookup_durations;
        std::vector<float> ps_accumulate_durations;
        std::vector<float> gpu_fwd_bwd_durations;
        std::vector<float> all_to_all_durations;
        float duration;

        for (auto& w : waves) {
            ScopedDevice scope_device(w->GetDevice());
            auto wave_data = w->GetData();
            for (auto& d : wave_data) {
                for (auto& m : d->m_metrics) {
                    NVE_CHECK_(cudaEventElapsedTime(&duration, m.lookup.d_start, m.lookup.d_end));
                    lookup_durations.push_back(duration);
                    lookup_e2e_durations.push_back(std::chrono::duration<float, std::milli>(m.lookup.h_end - m.lookup.h_start).count());
                    if (m.insert_done) {
                        NVE_CHECK_(cudaEventElapsedTime(&duration, m.insert.d_start, m.insert.d_end));
                        insert_durations.push_back(duration);
                        insert_host_durations.push_back(
                            std::chrono::duration<float, std::milli>(m.insert.h_end - m.insert.h_start).count()
                        );
                    }

                    NVE_CHECK_(cudaEventElapsedTime(&duration, m.scatter.d_start, m.scatter.d_end));
                    scatter_durations.push_back(duration);

                    if (do_pooling) {
                        NVE_CHECK_(cudaEventElapsedTime(&duration, m.pooling.d_start, m.pooling.d_end));
                        pooling_durations.push_back(duration);
                    }

                    if (dedup) {
                        NVE_CHECK_(cudaEventElapsedTime(&duration, m.dedup_gradients.d_start, m.dedup_gradients.d_end));
                        grads_dedup_durations.push_back(duration);
                    }

                    if (num_ranks > 1) {
                        NVE_CHECK_(cudaEventElapsedTime(&duration, m.all_to_all.d_start, m.all_to_all.d_end));
                        all_to_all_durations.push_back(duration);
                    }
                    NVE_CHECK_(cudaEventElapsedTime(&duration, m.accumulate.d_start, m.accumulate.d_end));
                    accumulate_durations.push_back(duration);

                    hitrate.push_back(static_cast<float>(m.hitrate));

                    host_cache_lookup_durations.push_back(std::chrono::duration<float, std::milli>(m.host_cache_lookup.h_end - m.host_cache_lookup.h_start).count());
                    host_cache_accumulate_durations.push_back(std::chrono::duration<float, std::milli>(m.host_cache_accumulate.h_end - m.host_cache_accumulate.h_start).count());
                    ps_lookup_durations.push_back(std::chrono::duration<float, std::milli>(m.ps_lookup.h_end - m.ps_lookup.h_start).count());
                    ps_accumulate_durations.push_back(std::chrono::duration<float, std::milli>(m.ps_accumulate.h_end - m.ps_accumulate.h_start).count());

                    if (m.gpu_work_recorded) {
                        NVE_CHECK_(cudaEventElapsedTime(&duration, m.gpu_fwd_bwd.d_start, m.gpu_fwd_bwd.d_end));
                        gpu_fwd_bwd_durations.push_back(duration);
                    }
                }
            }
        }

        std::vector<DistributionParser<float>> dps {
            {"Lookup", "Cache lookup durations (ms)", lookup_durations},
            {"Lookup E2E", "Complete Lookup durations (ms)", lookup_e2e_durations},
            {"Insert", "Cache insert durations (ms)", insert_durations},
            {"Insert_Host", "Cache insert host durations (ms)", insert_host_durations},
            {"Scatter", "Cache scatter durations (ms)", scatter_durations},
            {"Pooling", "Embedding pooling durations (ms)", pooling_durations},
            {"Gradients dedup", "Gradients accumulation durations (ms)", grads_dedup_durations},
            {"Accumulate", "Cache accumulate durations (ms)", accumulate_durations},
            {"Hitrate", "Cache hit rates", hitrate},
            {"HostCacheLookup", "Host cache lookup durations (ms)", host_cache_lookup_durations},
            {"HostCacheAccumulate", "Host cache accumulate durations (ms)", host_cache_accumulate_durations},
            {"PSLookup", "Parameter Server lookup durations (ms)", ps_lookup_durations},
            {"PSAccumulate", "Parameter Server accumulate durations (ms)", ps_accumulate_durations},
            {"GPUFwdBwd", "GPU Fwd+Bwd durations (ms)", gpu_fwd_bwd_durations},
            {"AllToAll", "All-to-all communication durations (ms)", all_to_all_durations},
        };

        // Print metrics
        size_t max_name_length(0);
        for (auto d : dps) {
            max_name_length = std::max(max_name_length, d.LongName().length());
        }
        max_name_length += 2; // Add a couple of spaces for visual separation
        for (auto d : dps) {
            std::cout 
                << std::left << std::setw(static_cast<int>(max_name_length)) << d.LongName() 
                << std::right << std::setw(10) << (" [" + std::to_string(d.NumValues()) + "]: ");
            std::cout 
                << d.OutputString() << std::endl;
        }

        // Output CSV
        if(csv_filename.length())
        {
            std::ofstream csv;
            csv.open(csv_filename, std::ios_base::app);
            if (!csv.is_open())
            {
                std::cerr << "Failed to open CSV " << csv_filename << std::endl;
            }
            else
            {
                if (csv.tellp() == csv.beg) {
                    // Output the CSV header
                    csv << "Waves,Caches,Inputs,Rows,RowSize,Hotness,Alpha,BatchSize,Warmups,Iterations,CacheSize,Devices,IPS,";
                    for (auto d : dps) {
                        csv << d.CSVHeader();
                    }
                    csv << std::endl;
                }
                // CSV line
                csv
                    << num_waves << ","
                    << num_caches << ","
                    << num_inputs << ","
                    << num_rows << ","
                    << row_size << ","
                    << hotness << ","
                    << alpha << ","
                    << batch_size << ","
                    << num_warmup_iterations << ","
                    << num_iterations << ","
                    << cache_size << ","
                    << device_count << ","
                    << ips << ", ";
                for (auto d : dps) {
                    csv << d.CSVString();
                }
                csv << std::endl;
            }
        }
    }

    // Cleanup
    waves.clear();          // Must clear waves before streams
    for (uint64_t dev = 0; dev < device_count; dev++)
    {
        task_queues[dev].clear();
        cache_wrappers[dev].clear(); // Must clear wrappers before streams
        for (auto s : cache_streams[dev]) {
            ScopedDevice scope_device(static_cast<int>(dev));
            NVE_CHECK_(cudaStreamDestroy(s));
        }
    }

    return 0;
}
