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
 * Single-layer benchmark over the C++ API.
 *
 * Builds one embedding layer of the requested type (or --num_layers independent copies of it, each with
 * its own tables, files and RocksDB directories), drives power-law lookups from --threads inference threads
 * per layer (each with its own execution context and CUDA streams) and reports throughput, per-tier hit
 * rates and per-step latency. Layer types:
 *   gpu          GPUEmbeddingLayer, the whole table in HBM.
 *   linear_uvm   LinearUVMEmbeddingLayer, GPU cache over a table in a ManagedMemBlock (host memory).
 *   hierarchical HierarchicalEmbeddingLayer, GPU cache + host tier (nvhm|rocksdb|linear_file|redis)
 *                [+ remote tier (rocksdb|linear_file|redis)]. With a remote tier the host tier is a
 *                cache sized by --host_load_factor, otherwise it holds the whole table.
 *                --load_factor 0 drops the GPU cache: the layer then runs on the host tiers alone.
 * The latency phase draws fresh keys by default (--latency_keys): replaying the throughput phase's keys
 * would find them already auto-inserted into every cache tier.
 * Keys stay in [0, num_table_rows) for every configuration. Table files and RocksDB directories
 * use default names under --file_dir; layers past the first add an _l<idx> suffix.
 * Storage created by this run is removed at teardown unless --keep_temp_files is set.
 * --dump_hier / --load_hier save and restore the contents of a hierarchical layer's cache tables (every
 * table but the last) to <filename>.<table_idx> through the table API alone, so no layer bookkeeping or
 * auto-insert is triggered. A dump happens right after warmup; a load replaces the warmup phase.
 */

#include <cuda_runtime.h>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include <bit_ops.hpp>
#include <buffer_wrapper.hpp>
#include <common.hpp>
#include <embedding_layer.hpp>
#include <gpu_embedding_layer.hpp>
#include <gpu_table.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <insert_heuristic.hpp>
#include <linear_embedding_layer.hpp>
#include <logging.hpp>
#include <memblock.hpp>
#include <plugin/plugin_loader.hpp>
#include <thread_pool.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using IndexT = int64_t;
using clock_type = std::chrono::steady_clock;

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int64_t kPrefillBatch = 1 << 16;   // rows per insert() while populating a store
constexpr int64_t kWarmupChunk = 100;        // warmup steps generated at a time
constexpr int64_t kDefaultMinInsertSize = 1 << 16;  // HierarchicalEmbeddingLayer::Config::min_insert_size_gpu
constexpr size_t kMaxTiers = 3;              // device, host, remote
constexpr int64_t kDumpBatchBytes = 256 << 20;  // row bytes per find() / insert() while dumping / loading tables
constexpr int64_t kDumpMaxBatch = 1 << 20;      // keys per batch cap (GPUTableConfig::max_modify_size above)
constexpr uint64_t kDumpMagic = 0x3150'4D55'4445'564EULL;  // "NVEDUMP1"

// ---------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------
struct RunConfig {
  std::string layer;         // gpu | linear_uvm | hierarchical
  std::string host_table;    // hierarchical tier 2: nvhm | rocksdb | linear_file | redis
  std::string remote_table;  // hierarchical tier 3: none | rocksdb | linear_file | redis
  std::string data_type;     // float32 | float16
  int64_t embedding_dim;
  int64_t num_rows;
  int64_t batch;
  int64_t hotness;
  double alpha;
  double load_factor;        // GPU cache rows / table rows
  double host_load_factor;   // host cache rows / table rows (hierarchical with a remote tier)
  double insert_threshold;   // GPU auto-insert target hit rate
  int64_t kernel;            // LinearUVM kernel mode (GPUTableConfig::kernel_mode_type)
  int64_t kernel_mode_value_1;
  int64_t num_layers;
  int64_t layer_idx{0};      // layer being built: selects its plugin table ids and file / RocksDB paths
  int64_t threads;           // per layer
  int64_t num_steps;
  int64_t num_warmup_steps;
  std::string warmup_mode;   // lookup: auto-insert warms the caches | insert: insert the warmup keys directly
  int64_t warmup_insert_size;  // insert mode: keys collected before each GPU cache insert
  int64_t latency_steps;     // -1 = num_steps, 0 = skip the latency phase
  std::string latency_keys;  // fresh: new keys for the latency phase | replay: the throughput phase's keys
  double max_latency_ms;
  int64_t hitrate_interval;  // collect hit rates every N steps, 0 = never
  bool keys_on_gpu;
  bool prefill;
  bool clear;
  int device;
  int64_t threadpool_workers;
  uint64_t seed;
  std::string log_level;
  std::string tag;
  std::string csv_filename;
  bool verbose;
  int64_t nvhm_partitions;   // nvhm host tier partitions (power of 2)
  int64_t nvhm_insert_queue; // nvhm insert prefetch queue length (0 | 1 | 2 | 4 | 8)
  std::string dump_hier;     // hierarchical: dump the cache tables to <dump_hier>.<table_idx> after warmup
  std::string load_hier;     // hierarchical: load the cache tables from <load_hier>.<table_idx> instead of warming up
  // storage
  std::string file_dir;
  std::string file_path;
  std::string rocksdb_path;
  bool keep_temp_files;
  int64_t rocksdb_threads;
  bool file_direct_io;
  int64_t file_staging_mb;
  int64_t file_threads;
  std::string file_io_engine;
  bool file_coalesce;
  std::string redis_address;
  bool redis_single_node;
  int64_t redis_num_partitions;
  int64_t redis_connections_per_node;

  nve::DataType_t dtype() const {
    if (data_type == "float32") return nve::DataType_t::Float32;
    if (data_type == "float16") return nve::DataType_t::Float16;
    NVE_THROW_("Unsupported --data_type '", data_type, "' (float32|float16)");
  }
  int64_t row_size() const { return embedding_dim * nve::dtype_size(dtype()); }
  int64_t keys_per_step() const { return batch * hotness; }
  size_t table_bytes() const { return static_cast<size_t>(num_rows) * static_cast<size_t>(row_size()); }
  size_t gpu_cache_bytes() const { return static_cast<size_t>(static_cast<double>(table_bytes()) * load_factor); }
  int64_t host_cache_rows() const { return static_cast<int64_t>(static_cast<double>(num_rows) * host_load_factor); }
  bool has_remote() const { return remote_table != "none"; }
  bool uses_linear_file() const { return layer == "hierarchical" && (host_table == "linear_file" || remote_table == "linear_file"); }
  // A hierarchical layer without a GPU cache runs entirely on host tables.
  bool has_gpu_tier() const { return layer != "hierarchical" || load_factor > 0.0; }
  bool dumps_tables() const { return layer == "hierarchical" && !dump_hier.empty(); }
  bool loads_tables() const { return layer == "hierarchical" && !load_hier.empty(); }
  std::string table_file() const {
    if (!file_path.empty()) return per_layer_path(file_path);
    return per_layer_path(file_dir + "/nve_table_" + std::to_string(num_rows) + "x" + std::to_string(embedding_dim) +
                          "_" + data_type + ".bin");
  }
  std::string rocksdb_dir() const {
    if (!rocksdb_path.empty()) return per_layer_path(rocksdb_path);
    return per_layer_path(file_dir + "/nve_rocksdb_" + std::to_string(num_rows) + "x" +
                          std::to_string(embedding_dim) + "_" + data_type);
  }
  // Layers past the first get their own file / directory: <stem>_l<idx><ext>.
  std::string per_layer_path(const std::string& path) const { return per_layer_path(path, layer_idx); }
  static std::string per_layer_path(const std::string& path, int64_t layer_idx) {
    if (layer_idx == 0) return path;
    std::filesystem::path p(path);
    if (!p.has_filename()) p = p.parent_path();
    return (p.parent_path() / (p.stem().string() + "_l" + std::to_string(layer_idx) + p.extension().string())).string();
  }
  // Distinct plugin table ids per layer (Redis namespaces its keys by table id).
  nve::table_id_t layer_table_id(nve::table_id_t base) const { return base + 1000 * layer_idx; }
  std::string layer_name(const std::string& base) const {
    return num_layers > 1 ? base + "_" + std::to_string(layer_idx) : base;
  }
};

// ---------------------------------------------------------------------------------------------
// Workload generation: power-law ranks scattered over [0, num_rows) by an affine bijection.
// ---------------------------------------------------------------------------------------------
class PowerLaw final {
 public:
  // alpha <= 0 samples uniformly; alpha == 1 is a pole of the inverse CDF so it is nudged.
  PowerLaw(double alpha, int64_t n) : n_(n), uniform_(alpha <= 0.0) {
    if (std::fabs(alpha - 1.0) < 1e-6) alpha = 1.0 + 1e-6;
    const double gamma = 1.0 - alpha;
    x0_ = 1.0;  // min rank 1 (shifted to 0 below)
    x1_sub_x0_ = std::pow(static_cast<double>(n) + 1.0, gamma) - x0_;
    exp_ = 1.0 / gamma;
  }
  template <typename G>
  int64_t operator()(G& gen) {
    if (uniform_) return static_cast<int64_t>(gen() % static_cast<uint64_t>(n_));
    const double x = dist_(gen);
    const double y = std::pow(x * x1_sub_x0_ + x0_, exp_);
    return std::clamp<int64_t>(static_cast<int64_t>(y) - 1, 0, n_ - 1);
  }

 private:
  int64_t n_;
  bool uniform_;
  double x0_{1.0}, x1_sub_x0_{0.0}, exp_{1.0};
  std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

// perm(x) = (a*x + b) mod n with gcd(a, n) = 1; memory free, shared by every thread.
class AffinePermutation final {
 public:
  AffinePermutation(int64_t n, uint64_t seed) : n_(static_cast<uint64_t>(std::max<int64_t>(n, 1))) {
    std::mt19937_64 g(seed ^ 0x9e3779b97f4a7c15ULL);
    b_ = g() % n_;
    a_ = (g() % n_) | 1ULL;
    while (std::gcd(a_, n_) != 1) {
      a_ += 2;
      if (a_ >= n_) a_ = 1;
    }
  }
  int64_t operator()(int64_t x) const {
    const unsigned __int128 t = static_cast<unsigned __int128>(a_) * static_cast<uint64_t>(x) + b_;
    return static_cast<int64_t>(static_cast<uint64_t>(t % n_));
  }

 private:
  uint64_t n_, a_{1}, b_{0};
};

class KeyGenerator final {
 public:
  KeyGenerator(const RunConfig& cfg, uint64_t seed)
      : rng_(seed), law_(cfg.alpha, cfg.num_rows), perm_(cfg.num_rows, cfg.seed) {}
  void fill(IndexT* keys, int64_t n) {
    for (int64_t i = 0; i < n; ++i) keys[i] = perm_(law_(rng_));
  }

 private:
  std::mt19937_64 rng_;
  PowerLaw law_;
  AffinePermutation perm_;
};

// ---------------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------------
class Barrier final {
 public:
  explicit Barrier(int64_t count) : threshold_(count), count_(count) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int64_t gen = generation_;
    if (--count_ == 0) {
      ++generation_;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  const int64_t threshold_;
  int64_t count_;
  int64_t generation_{0};
};

// Owning pinned host / device buffers.
struct PinnedBuffer {
  explicit PinnedBuffer(size_t bytes) { NVE_CHECK_(cudaMallocHost(&ptr, bytes)); }
  ~PinnedBuffer() { cudaFreeHost(ptr); }
  NVE_PREVENT_COPY_AND_MOVE_(PinnedBuffer);
  void* ptr{nullptr};
};
struct DeviceBuffer {
  explicit DeviceBuffer(size_t bytes) { NVE_CHECK_(cudaMalloc(&ptr, bytes)); }
  ~DeviceBuffer() { cudaFree(ptr); }
  NVE_PREVENT_COPY_AND_MOVE_(DeviceBuffer);
  void* ptr{nullptr};
};

double seconds_between(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return std::nan("");
  const auto rank = static_cast<int64_t>(std::ceil(q * static_cast<double>(sorted.size())));
  return sorted[static_cast<size_t>(std::clamp<int64_t>(rank, 1, static_cast<int64_t>(sorted.size())) - 1)];
}

double mean_of(const std::vector<double>& v) {
  return v.empty() ? std::nan("") : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double mem_available_gb() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  double kb;
  while (f >> key >> kb) {
    std::string unit;
    std::getline(f, unit);
    if (key == "MemAvailable:") return kb / (1024.0 * 1024.0);
  }
  return std::nan("");
}

std::string gpu_name(int device) {
  cudaDeviceProp prop{};
  NVE_CHECK_(cudaGetDeviceProperties(&prop, device));
  return prop.name;
}

std::string host_name() {
  char buf[256] = {0};
  return gethostname(buf, sizeof(buf) - 1) == 0 ? std::string(buf) : "unknown";
}

std::string now_string() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream os;
  os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return os.str();
}

std::string fmt(double v, int precision = 3) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << v;
  return os.str();
}

// Outlives the layers so plugin files are closed before removing newly created storage.
class TempStorage final {
 public:
  explicit TempStorage(bool keep) : keep_(keep) {}
  NVE_PREVENT_COPY_AND_MOVE_(TempStorage);

  void track(const std::string& path, bool directory) {
    for (const auto& entry : entries_) {
      if (entry.path == path) return;
    }
    const bool existed = std::filesystem::exists(std::filesystem::symlink_status(path));
    entries_.push_back({path, directory, existed});
    std::cout << (existed ? "Using existing " : "Creating temp ")
              << (directory ? "RocksDB directory " : "file ") << path << std::endl;
  }

  ~TempStorage() {
    for (const auto& entry : entries_) {
      if (entry.existed) continue;
      std::error_code ec;
      if (keep_) {
        if (std::filesystem::exists(entry.path, ec)) {
          std::cout << "Keeping temp " << (entry.directory ? "RocksDB directory " : "file ")
                    << entry.path << std::endl;
        }
        continue;
      }
      const auto removed = entry.directory ? std::filesystem::remove_all(entry.path, ec)
                                           : std::filesystem::remove(entry.path, ec);
      if (ec) {
        using nve::to_string;
        NVE_LOG_ERROR_("Cannot remove temp storage ", entry.path, ": ", ec.message());
      } else if (removed) {
        std::cout << "Removed temp " << (entry.directory ? "RocksDB directory " : "file ")
                  << entry.path << std::endl;
      }
    }
  }

 private:
  struct Entry {
    std::string path;
    bool directory;
    bool existed;
  };
  bool keep_;
  std::vector<Entry> entries_;
};

// Writes num_rows rows for real (a hole-only sparse file reads as zeros without touching storage).
// Rows start with their int64 index; the rest is pseudo-random.
double ensure_table_file(const std::string& path, int64_t num_rows, int64_t row_size) {
  const auto needed = static_cast<uintmax_t>(num_rows) * static_cast<uintmax_t>(row_size);
  std::error_code ec;
  if (std::filesystem::exists(path, ec) && std::filesystem::file_size(path, ec) >= needed) return 0.0;
  const auto t0 = clock_type::now();
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  NVE_CHECK_(out.good(), "Cannot create table file ", path);
  const int64_t rows_per_chunk = std::max<int64_t>(1, (64 << 20) / row_size);
  std::vector<uint64_t> chunk(static_cast<size_t>(rows_per_chunk * row_size) / sizeof(uint64_t) + 1);
  std::mt19937_64 rng(0xC0FFEE);
  for (int64_t row = 0; row < num_rows; row += rows_per_chunk) {
    const int64_t rows = std::min(rows_per_chunk, num_rows - row);
    const size_t bytes = static_cast<size_t>(rows * row_size);
    for (auto& w : chunk) w = rng();
    auto* bytes_ptr = reinterpret_cast<uint8_t*>(chunk.data());
    if (row_size >= static_cast<int64_t>(sizeof(int64_t))) {
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t idx = row + r;
        std::memcpy(bytes_ptr + r * row_size, &idx, sizeof(idx));
      }
    }
    out.write(reinterpret_cast<const char*>(bytes_ptr), static_cast<std::streamsize>(bytes));
    NVE_CHECK_(out.good(), "Failed writing table file ", path);
  }
  out.close();
  return seconds_between(t0, clock_type::now());
}

// ---------------------------------------------------------------------------------------------
// Layer construction
// ---------------------------------------------------------------------------------------------
struct LayerBundle {
  std::shared_ptr<nve::EmbeddingLayerBase> layer;
  std::shared_ptr<nve::MemBlock> table_block;  // gpu / linear_uvm: the dense table
  std::vector<nve::table_ptr_t> tables;        // hierarchical: the layer's tables, for direct table access
  std::vector<std::string> tier_names;         // one per layer table, e.g. {"gpu", "nvhm", "rocksdb"}
  std::vector<int64_t> prefill_tables;         // table ids that hold the whole table and need rows
  // Cache tiers that auto-insert would warm up; --warmup_mode insert feeds them directly.
  struct CacheTable {
    int64_t table_id;
    int64_t min_insert_size;  // keys collected before an insert, as the layer's auto-insert does
    bool on_gpu;
  };
  std::vector<CacheTable> cache_tables;
  std::string description;
  double file_create_s{0.0};
};

nve::table_ptr_t make_plugin_table(const std::string& so_name, const nlohmann::json& factory_conf,
                                   const nlohmann::json& table_conf, nve::table_id_t id, bool verbose) {
  if (verbose) std::cout << so_name << " factory " << factory_conf.dump() << " table " << table_conf.dump() << std::endl;
  nve::Plugin plugin(so_name);
  return plugin.create_table_factory(factory_conf)->produce(id, table_conf);
}

nlohmann::json base_table_conf(const RunConfig& cfg) {
  return {{"key_size", sizeof(IndexT)}, {"max_value_size", cfg.row_size()}, {"value_dtype", cfg.data_type}};
}

// nvhm holds `capacity_rows` rows: as a cache it evicts at that margin, as the full store it just grows.
nve::table_ptr_t make_nvhm_table(const RunConfig& cfg, int64_t capacity_rows, bool is_cache) {
  auto conf = base_table_conf(cfg);
  conf["num_partitions"] = cfg.nvhm_partitions;  // a single partition is usually best for inference
  conf["value_alignment"] = 32;
  conf["insert_fetch_queue_length"] = cfg.nvhm_insert_queue;
  // Capacity and overflow margin are per partition.
  const int64_t partition_rows = nve::ceil_div(capacity_rows, cfg.nvhm_partitions);
  conf["initial_capacity"] = std::max<int64_t>(partition_rows, 1024);
  if (is_cache) {
    NVE_CHECK_(capacity_rows > 0, "--host_load_factor must be positive for an nvhm cache");
    NVE_CHECK_(partition_rows >= 8, "nvhm partitions must contain at least 8 rows");
    const int64_t overflow_margin = partition_rows / 8 * 7;  // overflow at 7/8 capacity to avoid growing
    conf["overflow_policy"] = {{"handler", "evict_random"}, {"overflow_margin", overflow_margin}, {"resolution_margin", 0.9}};
  }
  return make_plugin_table("libnve-plugin-nvhm.so", nlohmann::json::object(), conf, cfg.layer_table_id(100), cfg.verbose);
}

nve::table_ptr_t make_rocksdb_table(const RunConfig& cfg, nve::table_id_t id) {
  const nlohmann::json factory = {{"path", cfg.rocksdb_dir()}, {"num_threads", cfg.rocksdb_threads}};
  return make_plugin_table("libnve-plugin-rocksdb.so", factory, base_table_conf(cfg), id, cfg.verbose);
}

nve::table_ptr_t make_linear_file_table(const RunConfig& cfg, nve::table_id_t id) {
  auto conf = base_table_conf(cfg);
  conf["path"] = cfg.table_file();
  conf["num_rows"] = cfg.num_rows;
  conf["direct_io"] = cfg.file_direct_io;
  conf["staging_bytes_per_thread"] = cfg.file_staging_mb << 20;
  conf["io_engine"] = cfg.file_io_engine;
  conf["coalesce_adjacent"] = cfg.file_coalesce;
  if (cfg.file_threads > 0) conf["io_threads"] = cfg.file_threads;
  return make_plugin_table("libnve-plugin-linear-file.so", nlohmann::json::object(), conf, id, cfg.verbose);
}

nve::table_ptr_t make_redis_table(const RunConfig& cfg, nve::table_id_t id) {
  const nlohmann::json factory = {{"address", cfg.redis_address},
                                  {"single_node", cfg.redis_single_node},
                                  {"connections_per_node", cfg.redis_connections_per_node}};
  auto conf = base_table_conf(cfg);
  conf["num_partitions"] = cfg.redis_num_partitions;
  return make_plugin_table("libnve-plugin-redis.so", factory, conf, id, cfg.verbose);
}

// A tier that holds every row of the table (host tier without a remote, or the remote tier).
nve::table_ptr_t make_store_table(const RunConfig& cfg, const std::string& kind, nve::table_id_t id,
                                  LayerBundle& bundle, bool& needs_prefill, TempStorage& storage) {
  needs_prefill = cfg.prefill;
  if (kind == "nvhm") return make_nvhm_table(cfg, cfg.num_rows, false /*is_cache*/);
  if (kind == "rocksdb") {
    std::error_code ec;
    const auto dir = cfg.rocksdb_dir();
    storage.track(dir, true /*directory*/);
    const bool populated = std::filesystem::is_directory(dir, ec) && !std::filesystem::is_empty(dir, ec);
    if (populated && !cfg.clear) {
      needs_prefill = false;  // reuse the rows already in the database
      if (cfg.verbose) std::cout << "Reusing populated RocksDB at " << dir << std::endl;
    }
    return make_rocksdb_table(cfg, id);
  }
  if (kind == "linear_file") {
    storage.track(cfg.table_file(), false /*directory*/);
    bundle.file_create_s += ensure_table_file(cfg.table_file(), cfg.num_rows, cfg.row_size());
    needs_prefill = false;  // the file is dense by construction
    return make_linear_file_table(cfg, id);
  }
  if (kind == "redis") return make_redis_table(cfg, id);
  NVE_THROW_("Unsupported table type '", kind, "' (nvhm|rocksdb|linear_file|redis)");
}

nve::GPUTableConfig gpu_cache_config(const RunConfig& cfg) {
  nve::GPUTableConfig gpu_cfg;
  gpu_cfg.device_id = cfg.device;
  gpu_cfg.cache_size = cfg.gpu_cache_bytes();
  gpu_cfg.row_size_in_bytes = cfg.row_size();
  gpu_cfg.value_dtype = cfg.dtype();
  gpu_cfg.count_misses = true;
  gpu_cfg.max_modify_size = int64_t(1) << 20;
  return gpu_cfg;
}

LayerBundle build_gpu_layer(const RunConfig& cfg) {
  LayerBundle b;
  b.table_block = std::make_shared<nve::LinearMemBlock>(static_cast<size_t>(cfg.embedding_dim),
                                                        static_cast<size_t>(cfg.num_rows), cfg.dtype(), cfg.device);
  nve::GPUEmbeddingLayerConfig layer_cfg;
  layer_cfg.layer_name = cfg.layer_name("bench_gpu");
  layer_cfg.device_id = cfg.device;
  layer_cfg.embedding_table = b.table_block->get_ptr();
  layer_cfg.num_embeddings = cfg.num_rows;
  layer_cfg.embedding_width_in_bytes = cfg.row_size();
  layer_cfg.value_dtype = cfg.dtype();
  b.layer = std::make_shared<nve::GPUEmbeddingLayer<IndexT>>(layer_cfg);
  b.tier_names = {"gpu"};
  b.description = "gpu layer, table in HBM";
  return b;
}

LayerBundle build_linear_uvm_layer(const RunConfig& cfg) {
  LayerBundle b;
  auto gpu_cfg = gpu_cache_config(cfg);
  // ManagedMemBlock advises a host preferred location, so the table stays in system memory.
  b.table_block = std::make_shared<nve::ManagedMemBlock>(static_cast<size_t>(cfg.embedding_dim),
                                                         static_cast<size_t>(cfg.num_rows), cfg.dtype(),
                                                         std::vector<int>{cfg.device});
  b.description = "linear_uvm layer, GPU cache + table in a ManagedMemBlock (host memory)";
  gpu_cfg.uvm_table = b.table_block->get_ptr();
  gpu_cfg.uvm_num_rows = cfg.num_rows;
  gpu_cfg.kernel_mode_type = static_cast<uint64_t>(cfg.kernel);
  if (cfg.kernel == static_cast<int64_t>(nve::KernelType::SortGather)) {
    gpu_cfg.kernel_mode_value = static_cast<uint64_t>(cfg.kernel_mode_value_1 > 0 ? cfg.kernel_mode_value_1 : 1024);
  }
  auto gpu_table = std::make_shared<nve::GpuTable<IndexT>>(gpu_cfg);

  typename nve::LinearUVMEmbeddingLayer<IndexT>::Config layer_cfg;
  layer_cfg.layer_name = cfg.layer_name("bench_linear_uvm");
  layer_cfg.insert_heuristic =
      std::make_shared<nve::DefaultInsertHeuristic>(std::vector<float>{static_cast<float>(cfg.insert_threshold)});
  b.layer = std::make_shared<nve::LinearUVMEmbeddingLayer<IndexT>>(layer_cfg, gpu_table);
  b.tier_names = {"gpu"};
  b.cache_tables.push_back({0, cfg.warmup_insert_size, true});
  return b;
}

LayerBundle build_hierarchical_layer(const RunConfig& cfg, TempStorage& storage) {
  LayerBundle b;
  std::vector<nve::table_ptr_t> tables;
  std::vector<float> thresholds;
  if (cfg.has_gpu_tier()) {
    tables.push_back(std::make_shared<nve::GpuTable<IndexT>>(gpu_cache_config(cfg)));
    thresholds.push_back(static_cast<float>(cfg.insert_threshold));
    b.tier_names.push_back("gpu");
    b.cache_tables.push_back({0, cfg.warmup_insert_size, true});
  }

  bool needs_prefill = false;
  if (cfg.has_remote()) {
    // Host tier is a cache in front of the remote store; target hit rate follows its share of the table.
    NVE_CHECK_(cfg.host_table != "linear_file", "linear_file is dense and cannot serve as a cache in front of a remote tier");
    if (cfg.host_table == "nvhm") {
      tables.push_back(make_nvhm_table(cfg, cfg.host_cache_rows(), true /*is_cache*/));
    } else {
      // rocksdb / redis have no eviction here: they grow with every auto-insert.
      tables.push_back(make_store_table(cfg, cfg.host_table, cfg.layer_table_id(200), b, needs_prefill, storage));
    }
    thresholds.push_back(static_cast<float>(std::min(cfg.host_load_factor, 1.0)));
    b.tier_names.push_back(cfg.host_table);
    // The layer's min_insert_size_host default is 0: the host cache is inserted into on every lookup.
    b.cache_tables.push_back({static_cast<int64_t>(tables.size()) - 1, 0, false});

    tables.push_back(make_store_table(cfg, cfg.remote_table, cfg.layer_table_id(300), b, needs_prefill, storage));
    if (needs_prefill) b.prefill_tables.push_back(static_cast<int64_t>(tables.size()) - 1);
    thresholds.push_back(0.f);  // last tier is never auto-inserted into
    b.tier_names.push_back(cfg.remote_table);
  } else {
    tables.push_back(make_store_table(cfg, cfg.host_table, cfg.layer_table_id(200), b, needs_prefill, storage));
    if (needs_prefill) b.prefill_tables.push_back(static_cast<int64_t>(tables.size()) - 1);
    thresholds.push_back(0.f);
    b.tier_names.push_back(cfg.host_table);
  }

  typename nve::HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
  layer_cfg.layer_name = cfg.layer_name("bench_hierarchical");
  layer_cfg.insert_heuristic = std::make_shared<nve::DefaultInsertHeuristic>(thresholds);
  layer_cfg.default_embedding.assign(static_cast<size_t>(cfg.row_size()), 0);  // misses read as zeros
  b.layer = std::make_shared<nve::HierarchicalEmbeddingLayer<IndexT>>(layer_cfg, tables);
  b.tables = tables;
  b.description = std::string("hierarchical layer, ") + (cfg.has_gpu_tier() ? "GPU cache + " : "") + cfg.host_table +
                  (cfg.has_remote() ? " cache + " + cfg.remote_table : "");
  return b;
}

// Mirrors AutoInsertHandler::collect_keys_and_data: a collection holds max(num_keys, min_insert_size)
// keys, later lookups only top it up to min_insert_size, and the insert fires once that is reached.
class WarmupInserter final {
 public:
  WarmupInserter(const RunConfig& cfg, const LayerBundle::CacheTable& table, nve::EmbeddingLayerBase* layer)
      : layer_(layer), table_id_(table.table_id), min_insert_size_(table.min_insert_size), row_size_(cfg.row_size()),
        capacity_(std::max(cfg.keys_per_step(), min_insert_size_)),
        keys_(static_cast<size_t>(capacity_) * sizeof(IndexT)) {
    // Garbage rows; placed where the table reads them from without an extra copy.
    const size_t value_bytes = static_cast<size_t>(capacity_ * row_size_);
    if (table.on_gpu) d_values_ = std::make_unique<DeviceBuffer>(value_bytes); else h_values_ = std::make_unique<PinnedBuffer>(value_bytes);
  }

  void collect(nve::context_ptr_t& ctx, const IndexT* keys, int64_t num_keys) {
    const int64_t total = (collected_ == 0) ? std::max(num_keys, min_insert_size_) : min_insert_size_;
    const int64_t part = std::min(num_keys, total - collected_);
    std::memcpy(static_cast<IndexT*>(keys_.ptr) + collected_, keys, static_cast<size_t>(part) * sizeof(IndexT));
    collected_ += part;
    if (collected_ >= min_insert_size_) flush(ctx);
  }

  // Auto-insert would drop a partial collection; flush it so short warmups still populate the cache.
  void flush(nve::context_ptr_t& ctx) {
    if (collected_ == 0) return;
    layer_->insert(ctx, collected_, keys_.ptr, row_size_, row_size_, d_values_ ? d_values_->ptr : h_values_->ptr, table_id_);
    collected_ = 0;
  }

 private:
  nve::EmbeddingLayerBase* layer_;
  int64_t table_id_, min_insert_size_, row_size_, capacity_;
  PinnedBuffer keys_;
  std::unique_ptr<DeviceBuffer> d_values_;
  std::unique_ptr<PinnedBuffer> h_values_;
  int64_t collected_{0};
};

LayerBundle build_layer(const RunConfig& cfg, TempStorage& storage) {
  if (cfg.layer == "gpu") return build_gpu_layer(cfg);
  if (cfg.layer == "linear_uvm") return build_linear_uvm_layer(cfg);
  if (cfg.layer == "hierarchical") return build_hierarchical_layer(cfg, storage);
  NVE_THROW_("Unsupported --layer '", cfg.layer, "' (gpu|linear_uvm|hierarchical)");
}

// Populate the dense table (gpu / linear_uvm) and every store tier with rows [0, num_rows).
double prefill(const RunConfig& cfg, LayerBundle& b, nve::context_ptr_t& ctx) {
  const auto t0 = clock_type::now();
  const int64_t row_size = cfg.row_size();
  if (b.table_block) {
    // Touching the table from the host leaves managed pages resident in system memory.
    if (b.table_block->get_type() == nve::MemBlockType::MANAGED) {
      std::memset(b.table_block->get_ptr(), 0, b.table_block->get_size_in_bytes());
    } else {
      NVE_CHECK_(cudaMemset(b.table_block->get_ptr(), 0, b.table_block->get_size_in_bytes()));
      NVE_CHECK_(cudaDeviceSynchronize());
    }
  }
  if (!b.prefill_tables.empty()) {
    PinnedBuffer keys(static_cast<size_t>(kPrefillBatch) * sizeof(IndexT));
    PinnedBuffer values(static_cast<size_t>(kPrefillBatch * row_size));
    // One random value block reused for every batch; the values do not affect lookup cost.
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    const int64_t num_elements = kPrefillBatch * cfg.embedding_dim;
    for (int64_t i = 0; i < num_elements; ++i) {
      if (cfg.dtype() == nve::DataType_t::Float32) {
        static_cast<float*>(values.ptr)[i] = dist(rng);
      } else {
        static_cast<half*>(values.ptr)[i] = __float2half(dist(rng));
      }
    }
    auto* key_ptr = static_cast<IndexT*>(keys.ptr);
    for (int64_t start = 0; start < cfg.num_rows; start += kPrefillBatch) {
      const int64_t n = std::min(kPrefillBatch, cfg.num_rows - start);
      std::iota(key_ptr, key_ptr + n, start);
      for (const auto table_id : b.prefill_tables) {
        b.layer->insert(ctx, n, key_ptr, row_size, row_size, values.ptr, table_id);
      }
      if (cfg.verbose && ((start / kPrefillBatch) % 64 == 0)) {
        std::cout << "Prefill " << start + n << "/" << cfg.num_rows << " rows" << std::endl;
      }
    }
    ctx->wait();
  }
  return seconds_between(t0, clock_type::now());
}

// ---------------------------------------------------------------------------------------------
// Table dump / load (hierarchical layer): table API only, so no layer bookkeeping or auto-insert runs.
// File layout: DumpHeader, then num_rows records of (IndexT key, row_size bytes).
// ---------------------------------------------------------------------------------------------
struct DumpHeader {
  uint64_t magic{kDumpMagic};
  int64_t key_size{sizeof(IndexT)};
  int64_t row_size{0};
  int64_t num_rows{0};  // patched once the dump is complete
};

int64_t dump_batch_rows(const RunConfig& cfg) {
  return std::clamp<int64_t>(kDumpBatchBytes / cfg.row_size(), 1, kDumpMaxBatch);
}

// <dump_hier>.<table_idx>; layers past the first use <dump_hier>_l<layer_idx>.<table_idx>.
std::string dump_file_name(const std::string& base, int64_t layer_idx, size_t table_idx) {
  return RunConfig::per_layer_path(base, layer_idx) + "." + std::to_string(table_idx);
}

// Wraps a host buffer so a table of any location can consume it; GPU tables copy through the context.
template <typename T>
std::shared_ptr<nve::BufferWrapper<T>> wrap(nve::context_ptr_t& ctx, const char* name, void* ptr, size_t bytes) {
  return std::make_shared<nve::BufferWrapper<T>>(ctx, name, static_cast<T*>(ptr), bytes);
}

// find() over every valid key in batches; the hit mask selects the (key, row) records written to `path`.
int64_t dump_table(const RunConfig& cfg, nve::Table& table, const std::string& path) {
  const int64_t row_size = cfg.row_size();
  const int64_t batch = dump_batch_rows(cfg);
  const size_t mask_words = static_cast<size_t>(nve::ceil_div(batch, nve::bitmask64::num_bits));
  const size_t record_size = sizeof(IndexT) + static_cast<size_t>(row_size);
  PinnedBuffer keys(static_cast<size_t>(batch) * sizeof(IndexT));
  PinnedBuffer mask(mask_words * sizeof(nve::bitmask64_t));
  PinnedBuffer values(static_cast<size_t>(batch * row_size));
  std::vector<uint8_t> records(record_size * static_cast<size_t>(batch));

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  NVE_CHECK_(out.good(), "Cannot create dump file ", path);
  DumpHeader header;
  header.row_size = row_size;
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));

  auto ctx = table.create_execution_context(0, 0, nullptr, nullptr);
  const cudaStream_t stream = ctx->get_lookup_stream();
  auto* key_ptr = static_cast<IndexT*>(keys.ptr);
  int64_t total = 0;
  for (int64_t start = 0; start < cfg.num_rows; start += batch) {
    const int64_t n = std::min(batch, cfg.num_rows - start);
    const size_t n_words = static_cast<size_t>(nve::ceil_div(n, nve::bitmask64::num_bits));
    std::iota(key_ptr, key_ptr + n, start);
    std::memset(mask.ptr, 0, n_words * sizeof(nve::bitmask64_t));  // set bits are skipped as already resolved
    auto keys_bw = wrap<const void>(ctx, "keys", keys.ptr, static_cast<size_t>(n) * sizeof(IndexT));
    auto mask_bw = wrap<nve::bitmask64_t>(ctx, "hitmask", mask.ptr, n_words * sizeof(nve::bitmask64_t));
    auto values_bw = wrap<void>(ctx, "values", values.ptr, static_cast<size_t>(n * row_size));
    table.find(ctx, n, keys_bw, mask_bw, row_size, values_bw, nullptr);
    // A GPU table resolves into device copies; this pulls the results back into the pinned buffers.
    const auto* h_mask = mask_bw->access_buffer(cudaMemoryTypeHost, true /*copy_content*/, stream);
    const auto* h_values = static_cast<const uint8_t*>(values_bw->access_buffer(cudaMemoryTypeHost, true /*copy_content*/, stream));
    ctx->wait();

    size_t packed = 0;
    for (size_t w = 0; w < n_words; ++w) {
      const int64_t base = static_cast<int64_t>(w) * nve::bitmask64::num_bits;
      for (auto m = nve::bitmask64::clip(h_mask[w], n - base); m != nve::bitmask64::empty; m = nve::bitmask64::skip(m)) {
        const int64_t i = base + nve::bitmask64::next(m);
        const IndexT key = start + i;
        std::memcpy(records.data() + packed, &key, sizeof(key));
        std::memcpy(records.data() + packed + sizeof(key), h_values + i * row_size, static_cast<size_t>(row_size));
        packed += record_size;
      }
    }
    out.write(reinterpret_cast<const char*>(records.data()), static_cast<std::streamsize>(packed));
    NVE_CHECK_(out.good(), "Failed writing dump file ", path);
    total += static_cast<int64_t>(packed / record_size);
  }
  header.num_rows = total;
  out.seekp(0);
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  NVE_CHECK_(out.good(), "Failed writing dump file ", path);
  return total;
}

// insert() the records of `path` in batches; `insert_s` receives the time spent in insert() alone.
int64_t load_table(const RunConfig& cfg, nve::Table& table, const std::string& path, double& insert_s) {
  const int64_t row_size = cfg.row_size();
  std::ifstream in(path, std::ios::binary);
  NVE_CHECK_(in.good(), "Cannot open dump file ", path);
  DumpHeader header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  NVE_CHECK_(in.good() && header.magic == kDumpMagic, "Not a table dump: ", path);
  NVE_CHECK_(header.key_size == static_cast<int64_t>(sizeof(IndexT)) && header.row_size == row_size,
             "Dump ", path, " holds ", header.row_size, " byte rows, the layer uses ", row_size);

  const int64_t batch = dump_batch_rows(cfg);
  const size_t record_size = sizeof(IndexT) + static_cast<size_t>(row_size);
  PinnedBuffer keys(static_cast<size_t>(batch) * sizeof(IndexT));
  PinnedBuffer values(static_cast<size_t>(batch * row_size));
  std::vector<uint8_t> records(record_size * static_cast<size_t>(batch));
  auto* key_ptr = static_cast<IndexT*>(keys.ptr);
  auto* value_ptr = static_cast<uint8_t*>(values.ptr);

  auto ctx = table.create_execution_context(0, 0, nullptr, nullptr);
  for (int64_t done = 0; done < header.num_rows; done += batch) {
    const int64_t n = std::min(batch, header.num_rows - done);
    in.read(reinterpret_cast<char*>(records.data()), static_cast<std::streamsize>(record_size * static_cast<size_t>(n)));
    NVE_CHECK_(in.good(), "Truncated dump file ", path);
    for (int64_t i = 0; i < n; ++i) {
      const uint8_t* rec = records.data() + static_cast<size_t>(i) * record_size;
      std::memcpy(key_ptr + i, rec, sizeof(IndexT));
      std::memcpy(value_ptr + i * row_size, rec + sizeof(IndexT), static_cast<size_t>(row_size));
    }
    auto keys_bw = wrap<const void>(ctx, "keys", keys.ptr, static_cast<size_t>(n) * sizeof(IndexT));
    auto values_bw = wrap<const void>(ctx, "values", values.ptr, static_cast<size_t>(n * row_size));
    const auto t_insert = clock_type::now();
    table.insert(ctx, n, keys_bw, row_size, row_size, values_bw);
    ctx->wait();  // a GPU table copies from the pinned buffers asynchronously
    insert_s += seconds_between(t_insert, clock_type::now());
  }
  return header.num_rows;
}

// Every table but the last is a cache tier worth saving; the last one holds the whole table.
void dump_hier_tables(const RunConfig& cfg, const LayerBundle& bundle, int64_t layer_idx) {
  const std::string prefix = cfg.num_layers > 1 ? "Layer " + std::to_string(layer_idx) + ": " : "";
  const auto t0 = clock_type::now();
  for (size_t t = 0; t + 1 < bundle.tables.size(); ++t) {
    const auto path = dump_file_name(cfg.dump_hier, layer_idx, t);
    const auto t_table = clock_type::now();
    const int64_t rows = dump_table(cfg, *bundle.tables[t], path);
    std::cout << prefix << "Dumped " << rows << " rows of table " << t << " (" << bundle.tier_names[t] << ") to " << path
              << " in " << fmt(seconds_between(t_table, clock_type::now()), 2) << " s" << std::endl;
  }
  std::cout << prefix << "Dump: " << fmt(seconds_between(t0, clock_type::now()), 2) << " s" << std::endl;
}

void load_hier_tables(const RunConfig& cfg, const LayerBundle& bundle, int64_t layer_idx) {
  const std::string prefix = cfg.num_layers > 1 ? "Layer " + std::to_string(layer_idx) + ": " : "";
  std::cout << prefix << "Loading tables from " << RunConfig::per_layer_path(cfg.load_hier, layer_idx)
            << ".* instead of warming up" << std::endl;
  const auto t0 = clock_type::now();
  for (size_t t = 0; t + 1 < bundle.tables.size(); ++t) {
    const auto path = dump_file_name(cfg.load_hier, layer_idx, t);
    const auto t_table = clock_type::now();
    double insert_s = 0.0;
    const int64_t rows = load_table(cfg, *bundle.tables[t], path, insert_s);
    std::cout << prefix << "Loaded " << rows << " rows into table " << t << " (" << bundle.tier_names[t] << ") from " << path
              << " in " << fmt(seconds_between(t_table, clock_type::now()), 2) << " s (insert " << fmt(insert_s, 2) << " s)" << std::endl;
  }
  std::cout << prefix << "Load: " << fmt(seconds_between(t0, clock_type::now()), 2) << " s" << std::endl;
}

// ---------------------------------------------------------------------------------------------
// Inference threads
// ---------------------------------------------------------------------------------------------
struct ThreadResult {
  clock_type::time_point tput_start, tput_end;
  std::vector<std::vector<double>> hitrates;      // throughput phase, [tier][sample]
  std::vector<std::vector<double>> lat_hitrates;  // latency phase, [tier][sample]
  std::vector<double> latency_ms;
};

void worker_main(const RunConfig& cfg, const LayerBundle& bundle, nve::context_ptr_t ctx, int64_t thread_idx,
                 Barrier& barrier, ThreadResult& result) {
  nve::EmbeddingLayerBase* const layer = bundle.layer.get();
  NVE_CHECK_(cudaSetDevice(cfg.device));
  const int64_t row_size = cfg.row_size();
  const int64_t keys_per_step = cfg.keys_per_step();
  const size_t step_key_bytes = static_cast<size_t>(keys_per_step) * sizeof(IndexT);
  const size_t num_tables = static_cast<size_t>(layer->get_num_tables());
  const cudaStream_t lookup_stream = ctx->get_lookup_stream();
  KeyGenerator keygen(cfg, cfg.seed + 1000 * static_cast<uint64_t>(thread_idx + 1));

  // Output lands where the last GPU tier writes it; a host-only layer gathers straight into pinned memory.
  const size_t out_bytes = static_cast<size_t>(keys_per_step * row_size);
  std::unique_ptr<DeviceBuffer> d_out;
  std::unique_ptr<PinnedBuffer> h_out;
  if (cfg.has_gpu_tier()) d_out = std::make_unique<DeviceBuffer>(out_bytes); else h_out = std::make_unique<PinnedBuffer>(out_bytes);
  void* const out_ptr = cfg.has_gpu_tier() ? d_out->ptr : h_out->ptr;
  // Measured key sets, generated before each phase and optionally moved to the GPU.
  const bool fresh_latency_keys = cfg.latency_keys == "fresh";
  const auto key_steps = static_cast<size_t>(fresh_latency_keys ? std::max(cfg.num_steps, cfg.latency_steps) : cfg.num_steps);
  PinnedBuffer h_keys(step_key_bytes * key_steps);
  std::unique_ptr<DeviceBuffer> d_keys;
  if (cfg.keys_on_gpu) d_keys = std::make_unique<DeviceBuffer>(step_key_bytes * key_steps);
  auto generate_keys = [&](int64_t steps) {
    keygen.fill(static_cast<IndexT*>(h_keys.ptr), keys_per_step * steps);
    if (cfg.keys_on_gpu) {
      NVE_CHECK_(cudaMemcpy(d_keys->ptr, h_keys.ptr, step_key_bytes * static_cast<size_t>(steps), cudaMemcpyHostToDevice));
    }
  };
  auto step_keys = [&](int64_t step) -> const IndexT* {
    const auto* base = static_cast<const IndexT*>(cfg.keys_on_gpu ? d_keys->ptr : h_keys.ptr);
    return base + step * keys_per_step;
  };
  generate_keys(cfg.num_steps);
  std::vector<float> hitrates(num_tables);
  // Table dumps / loads run on one thread per layer while the others wait at the barrier.
  const int64_t layer_idx = thread_idx / cfg.threads;
  const bool first_in_layer = thread_idx % cfg.threads == 0;

  // Warmup on fresh keys from the same distribution, generated in chunks. In lookup mode the layer's
  // auto-insert fills the caches; in insert mode the warmup keys go straight into the cache tiers.
  if (cfg.loads_tables()) {
    if (first_in_layer) load_hier_tables(cfg, bundle, layer_idx);
  } else {
    const bool insert_mode = cfg.warmup_mode == "insert";
    std::vector<std::unique_ptr<WarmupInserter>> inserters;
    if (insert_mode) {
      for (const auto& t : bundle.cache_tables) inserters.push_back(std::make_unique<WarmupInserter>(cfg, t, layer));
    }
    const int64_t chunk_steps = std::min(kWarmupChunk, cfg.num_warmup_steps);
    PinnedBuffer h_chunk(step_key_bytes * static_cast<size_t>(std::max<int64_t>(chunk_steps, 1)));
    std::unique_ptr<DeviceBuffer> d_chunk;
    const bool chunk_on_gpu = cfg.keys_on_gpu && !insert_mode;  // insert() takes host keys
    if (chunk_on_gpu) d_chunk = std::make_unique<DeviceBuffer>(step_key_bytes * static_cast<size_t>(std::max<int64_t>(chunk_steps, 1)));
    for (int64_t done = 0; done < cfg.num_warmup_steps; done += chunk_steps) {
      const int64_t steps = std::min(chunk_steps, cfg.num_warmup_steps - done);
      keygen.fill(static_cast<IndexT*>(h_chunk.ptr), keys_per_step * steps);
      if (chunk_on_gpu) {
        NVE_CHECK_(cudaMemcpy(d_chunk->ptr, h_chunk.ptr, step_key_bytes * static_cast<size_t>(steps), cudaMemcpyHostToDevice));
      }
      const auto* base = static_cast<const IndexT*>(chunk_on_gpu ? d_chunk->ptr : h_chunk.ptr);
      for (int64_t s = 0; s < steps; ++s) {
        if (insert_mode) {
          for (auto& ins : inserters) ins->collect(ctx, base + s * keys_per_step, keys_per_step);
        } else {
          layer->lookup(ctx, keys_per_step, base + s * keys_per_step, out_ptr, row_size, nullptr, nullptr, nullptr);
        }
      }
      NVE_CHECK_(cudaStreamSynchronize(lookup_stream));
    }
    for (auto& ins : inserters) ins->flush(ctx);
    ctx->wait();  // let pending (auto-)inserts settle before measuring
  }
  barrier.wait();
  if (cfg.dumps_tables()) {
    if (first_in_layer) dump_hier_tables(cfg, bundle, layer_idx);
    barrier.wait();
  }

  // Throughput phase: back-to-back lookups, one sync at the end (as single_gpu_bench.py).
  result.hitrates.assign(num_tables, {});
  result.tput_start = clock_type::now();
  for (int64_t s = 0; s < cfg.num_steps; ++s) {
    const bool collect = cfg.hitrate_interval > 0 && (s % cfg.hitrate_interval) == 0;
    layer->lookup(ctx, keys_per_step, step_keys(s), out_ptr, row_size, nullptr, nullptr,
                  collect ? hitrates.data() : nullptr);
    if (collect) {
      for (size_t t = 0; t < num_tables; ++t) result.hitrates[t].push_back(hitrates[t]);
    }
  }
  NVE_CHECK_(cudaStreamSynchronize(lookup_stream));
  result.tput_end = clock_type::now();
  // Replayed keys were auto-inserted into the caches during the throughput phase, which flatters the
  // latency of every cache tier; fresh keys are new draws from the same distribution.
  if (fresh_latency_keys && cfg.latency_steps > 0) generate_keys(cfg.latency_steps);
  barrier.wait();

  // Latency phase: one lookup at a time, synchronized after each.
  result.lat_hitrates.assign(num_tables, {});
  result.latency_ms.reserve(static_cast<size_t>(cfg.latency_steps));
  for (int64_t s = 0; s < cfg.latency_steps; ++s) {
    const bool collect = cfg.hitrate_interval > 0 && (s % cfg.hitrate_interval) == 0;
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream));
    const auto t0 = clock_type::now();
    layer->lookup(ctx, keys_per_step, step_keys(fresh_latency_keys ? s : s % cfg.num_steps), out_ptr, row_size, nullptr,
                  nullptr, collect ? hitrates.data() : nullptr);
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream));
    result.latency_ms.push_back(seconds_between(t0, clock_type::now()) * 1e3);
    if (collect) {
      for (size_t t = 0; t < num_tables; ++t) result.lat_hitrates[t].push_back(hitrates[t]);
    }
  }
  barrier.wait();  // no thread frees buffers while another still looks up
  ctx->wait();
}

// ---------------------------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------------------------
using MetricList = std::vector<std::pair<std::string, std::string>>;

void write_csv(const std::string& path, const MetricList& metrics) {
  std::error_code ec;
  const bool write_header = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
  std::ofstream out(path, std::ios::app);
  NVE_CHECK_(out.good(), "Cannot open CSV file ", path);
  auto quote = [](const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string q = "\"";
    for (const char c : s) q += (c == '"') ? "\"\"" : std::string(1, c);
    return q + "\"";
  };
  for (int pass = write_header ? 0 : 1; pass < 2; ++pass) {
    for (size_t i = 0; i < metrics.size(); ++i) {
      out << (i ? "," : "") << quote(pass == 0 ? metrics[i].first : metrics[i].second);
    }
    out << "\n";
  }
}

RunConfig parse_args(int argc, char** argv) {
  argparse::ArgumentParser args("single_layer_bench");
  args.add_description("Throughput, hit-rate and latency benchmark of a single embedding layer through the C++ API");
  args.add_argument("--layer").default_value(std::string("linear_uvm")).help("layer type: gpu | linear_uvm | hierarchical");
  args.add_argument("--host_table").default_value(std::string("nvhm")).help("hierarchical host tier: nvhm | rocksdb | linear_file | redis");
  args.add_argument("--remote_table").default_value(std::string("none")).help("hierarchical remote tier: none | rocksdb | linear_file | redis (turns the host tier into a cache)");
  args.add_argument("-dt", "--data_type").default_value(std::string("float32")).help("float32 | float16");
  args.add_argument("-ed", "--embedding_dim").default_value(int64_t(128)).scan<'i', int64_t>().help("embedding dimension");
  args.add_argument("-nr", "--num_table_rows").default_value(int64_t(1'000'000)).scan<'i', int64_t>().help("rows in the embedding table");
  args.add_argument("-b", "--batch").default_value(int64_t(1024)).scan<'i', int64_t>().help("batch size");
  args.add_argument("--hotness").default_value(int64_t(1024)).scan<'i', int64_t>().help("keys per sample");
  args.add_argument("-a", "--alpha").default_value(1.05).scan<'g', double>().help("power-law exponent of the key distribution, <= 0 for uniform");
  args.add_argument("-lf", "--load_factor").default_value(0.01).scan<'g', double>().help("GPU cache size as a fraction of the table; 0 runs a hierarchical layer on its host tiers alone");
  args.add_argument("-hlf", "--host_load_factor").default_value(0.1).scan<'g', double>().help("host cache size as a fraction of the table (hierarchical with a remote tier)");
  args.add_argument("--insert_threshold").default_value(static_cast<double>(nve::DefaultInsertHeuristic::DEFAULT_THRESHOLD)).scan<'g', double>().help("GPU cache auto-insert target hit rate");
  args.add_argument("-k", "--kernel").default_value(int64_t(1)).scan<'i', int64_t>().help("linear_uvm kernel mode: 0 dynamic, 1 lookup_uvm, 2 sort_gather");
  args.add_argument("-k1", "--kernel_mode_value_1").default_value(int64_t(0)).scan<'i', int64_t>().help("kernel mode value (sort_gather task size)");
  args.add_argument("-nl", "--num_layers").default_value(int64_t(1)).scan<'i', int64_t>().help("independent layers built from the same spec, each with its own tables, files and RocksDB directories");
  args.add_argument("-t", "--threads").default_value(int64_t(1)).scan<'i', int64_t>().help("parallel inference threads per layer, each with its own execution context");
  args.add_argument("-ns", "--num_steps").default_value(int64_t(100)).scan<'i', int64_t>().help("measured inference steps per thread");
  args.add_argument("-nw", "--num_warmup_steps").default_value(int64_t(100)).scan<'i', int64_t>().help("warmup steps per thread");
  args.add_argument("--warmup_mode").default_value(std::string("lookup")).help("lookup: warm the caches through lookups and auto-insert | insert: insert the warmup keys (garbage values) straight into the cache tiers, collecting --warmup_insert_size keys per GPU insert as auto-insert does");
  args.add_argument("--warmup_insert_size").default_value(kDefaultMinInsertSize).scan<'i', int64_t>().help("insert mode: warmup keys collected before each GPU cache insert (layer default 65536)");
  args.add_argument("--latency_steps").default_value(int64_t(-1)).scan<'i', int64_t>().help("steps of the per-step latency phase per thread; -1 = num_steps, 0 = skip");
  args.add_argument("--latency_keys").default_value(std::string("fresh")).help("latency phase keys: fresh (new draws from the distribution) | replay (the throughput phase's keys, already auto-inserted into the caches)");
  args.add_argument("--max_latency_ms").default_value(0.0).scan<'g', double>().help("latency budget; steps above it are counted (0 = none)");
  args.add_argument("--hitrate_interval").default_value(int64_t(1)).scan<'i', int64_t>().help("collect hit rates every N steps (0 = never; collecting may sync the GPU)");
  args.add_argument("--keys_on").default_value(std::string("gpu")).help("where lookup keys reside: gpu | host (pinned)");
  args.add_argument("--skip_prefill").default_value(false).implicit_value(true).help("do not populate the table / store tiers (reuse persistent content)");
  args.add_argument("-c", "--clear").default_value(false).implicit_value(true).help("clear all tables before prefill (persistent stores)");
  args.add_argument("--device").default_value(0).scan<'i', int>().help("CUDA device id");
  args.add_argument("--threadpool_workers").default_value(int64_t(0)).scan<'i', int64_t>().help("workers of the default thread pool used by host tiers, 0 = library default");
  args.add_argument("--seed").default_value(uint64_t(42)).scan<'i', uint64_t>().help("RNG seed");
  args.add_argument("--log_level").default_value(std::string("warning")).help("NVE log level: critical|error|warning|perf|info|verbose");
  args.add_argument("--tag").default_value(std::string("")).help("free-form label written to the CSV");
  args.add_argument("-csv", "--csv_filename").default_value(std::string("")).help("append the metrics as one CSV row");
  args.add_argument("-v", "--verbose").default_value(false).implicit_value(true).help("print configuration and progress");
  args.add_argument("--nvhm_partitions").default_value(int64_t(1)).scan<'i', int64_t>().help("nvhm host tier partitions (power of 2); inserts parallelize across partitions, lookups fan out per partition");
  args.add_argument("--nvhm_insert_queue").default_value(int64_t(0)).scan<'i', int64_t>().help("nvhm insert prefetch queue length: buckets are prefetched this many keys ahead (0 = off | 1 | 2 | 4 | 8)");
  args.add_argument("--dump_hier").default_value(std::string("")).help("hierarchical: after warmup, dump every table but the last to <filename>.<table_idx> through the table API (layers past the first use <filename>_l<idx>.<table_idx>)");
  args.add_argument("--load_hier").default_value(std::string("")).help("hierarchical: insert the rows of <filename>.<table_idx> into every table but the last through the table API instead of warming up");
  // storage
  const char* env_dir = std::getenv("NVE_BENCH_FILE_DIR");
  args.add_argument("--file_dir").default_value(std::string(env_dir ? env_dir : "/tmp")).help("directory of the default table file / RocksDB (default: $NVE_BENCH_FILE_DIR or /tmp)");
  args.add_argument("--file_path").default_value(std::string("")).help("table file for linear_file (default: <file_dir>/nve_table_<rows>x<dim>_<dtype>.bin)");
  args.add_argument("--rocksdb_path").default_value(std::string("")).help("RocksDB directory (default: <file_dir>/nve_rocksdb_<rows>x<dim>_<dtype>); a populated one is reused unless --clear");
  args.add_argument("--keep_temp_files").default_value(false).implicit_value(true).help("keep linear_file files and RocksDB directories created by this run (existing paths are always kept)");
  args.add_argument("--rocksdb_threads").default_value(int64_t(16)).scan<'i', int64_t>().help("RocksDB background threads");
  args.add_argument("--no_file_direct_io").default_value(false).implicit_value(true).help("linear_file: buffered reads instead of O_DIRECT");
  args.add_argument("--file_staging_mb").default_value(int64_t(1)).scan<'i', int64_t>().help("linear_file: staging buffer per worker thread in MiB");
  args.add_argument("--file_threads").default_value(int64_t(0)).scan<'i', int64_t>().help("linear_file: I/O threads per lookup incl. the caller, 0 = plugin default");
  args.add_argument("--file_io_engine").default_value(std::string("auto")).help("linear_file: auto | io_uring | aio | pread");
  args.add_argument("--file_coalesce").default_value(false).implicit_value(true).help("linear_file: sort missed keys and merge reads of touching blocks");
  args.add_argument("-ra", "--redis_address").default_value(std::string("localhost:7000")).help("Redis node address host:port");
  args.add_argument("-rsn", "--redis_single_node").default_value(false).implicit_value(true).help("standalone Redis server instead of a cluster");
  args.add_argument("-rnp", "--redis_num_partitions").default_value(int64_t(1)).scan<'i', int64_t>().help("Redis table partitions (0 or a power of 2)");
  args.add_argument("-rcn", "--redis_connections_per_node").default_value(int64_t(5)).scan<'i', int64_t>().help("max parallel connections per Redis node");

  try {
    args.parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n" << args;
    std::exit(1);
  }

  RunConfig cfg;
  cfg.layer = args.get<std::string>("--layer");
  cfg.host_table = args.get<std::string>("--host_table");
  cfg.remote_table = args.get<std::string>("--remote_table");
  cfg.data_type = args.get<std::string>("--data_type");
  cfg.embedding_dim = args.get<int64_t>("--embedding_dim");
  cfg.num_rows = args.get<int64_t>("--num_table_rows");
  cfg.batch = args.get<int64_t>("--batch");
  cfg.hotness = args.get<int64_t>("--hotness");
  cfg.alpha = args.get<double>("--alpha");
  cfg.load_factor = args.get<double>("--load_factor");
  cfg.host_load_factor = args.get<double>("--host_load_factor");
  cfg.insert_threshold = args.get<double>("--insert_threshold");
  cfg.kernel = args.get<int64_t>("--kernel");
  cfg.kernel_mode_value_1 = args.get<int64_t>("--kernel_mode_value_1");
  cfg.num_layers = args.get<int64_t>("--num_layers");
  cfg.threads = args.get<int64_t>("--threads");
  cfg.num_steps = args.get<int64_t>("--num_steps");
  cfg.num_warmup_steps = args.get<int64_t>("--num_warmup_steps");
  cfg.warmup_mode = args.get<std::string>("--warmup_mode");
  cfg.warmup_insert_size = args.get<int64_t>("--warmup_insert_size");
  cfg.latency_steps = args.get<int64_t>("--latency_steps");
  if (cfg.latency_steps < 0) cfg.latency_steps = cfg.num_steps;
  cfg.latency_keys = args.get<std::string>("--latency_keys");
  cfg.max_latency_ms = args.get<double>("--max_latency_ms");
  cfg.hitrate_interval = args.get<int64_t>("--hitrate_interval");
  cfg.keys_on_gpu = args.get<std::string>("--keys_on") != "host" && cfg.has_gpu_tier();  // host-only layers read host keys
  cfg.prefill = !args.get<bool>("--skip_prefill");
  cfg.clear = args.get<bool>("--clear");
  cfg.device = args.get<int>("--device");
  cfg.threadpool_workers = args.get<int64_t>("--threadpool_workers");
  cfg.seed = args.get<uint64_t>("--seed");
  cfg.log_level = args.get<std::string>("--log_level");
  cfg.tag = args.get<std::string>("--tag");
  cfg.csv_filename = args.get<std::string>("--csv_filename");
  cfg.verbose = args.get<bool>("--verbose");
  cfg.nvhm_partitions = args.get<int64_t>("--nvhm_partitions");
  cfg.nvhm_insert_queue = args.get<int64_t>("--nvhm_insert_queue");
  cfg.dump_hier = args.get<std::string>("--dump_hier");
  cfg.load_hier = args.get<std::string>("--load_hier");
  cfg.file_dir = args.get<std::string>("--file_dir");
  cfg.file_path = args.get<std::string>("--file_path");
  cfg.rocksdb_path = args.get<std::string>("--rocksdb_path");
  cfg.keep_temp_files = args.get<bool>("--keep_temp_files");
  cfg.rocksdb_threads = args.get<int64_t>("--rocksdb_threads");
  cfg.file_direct_io = !args.get<bool>("--no_file_direct_io");
  cfg.file_staging_mb = args.get<int64_t>("--file_staging_mb");
  cfg.file_threads = args.get<int64_t>("--file_threads");
  cfg.file_io_engine = args.get<std::string>("--file_io_engine");
  cfg.file_coalesce = args.get<bool>("--file_coalesce");
  cfg.redis_address = args.get<std::string>("--redis_address");
  cfg.redis_single_node = args.get<bool>("--redis_single_node");
  cfg.redis_num_partitions = args.get<int64_t>("--redis_num_partitions");
  cfg.redis_connections_per_node = args.get<int64_t>("--redis_connections_per_node");

  NVE_CHECK_ARG_(cfg.num_rows > 0 && cfg.embedding_dim > 0 && cfg.batch > 0 && cfg.hotness > 0, "sizes must be positive");
  NVE_CHECK_ARG_(cfg.num_layers > 0 && cfg.threads > 0 && cfg.num_steps > 0, "--num_layers, --threads and --num_steps must be positive");
  NVE_CHECK_ARG_(cfg.layer != "linear_uvm" || cfg.load_factor > 0.0, "--load_factor must be positive for the linear_uvm layer");
  NVE_CHECK_ARG_(cfg.kernel >= 0 && cfg.kernel <= 2, "--kernel must be 0, 1 or 2");
  NVE_CHECK_ARG_(cfg.warmup_mode == "lookup" || cfg.warmup_mode == "insert", "--warmup_mode must be lookup or insert");
  NVE_CHECK_ARG_(cfg.warmup_insert_size > 0, "--warmup_insert_size must be positive");
  NVE_CHECK_ARG_(cfg.nvhm_partitions > 0 && (cfg.nvhm_partitions & (cfg.nvhm_partitions - 1)) == 0, "--nvhm_partitions must be a power of 2");
  NVE_CHECK_ARG_(cfg.latency_keys == "fresh" || cfg.latency_keys == "replay", "--latency_keys must be fresh or replay");
  const auto keys_on = args.get<std::string>("--keys_on");
  NVE_CHECK_ARG_(keys_on == "gpu" || keys_on == "host", "--keys_on must be gpu or host");
  (void)cfg.dtype();  // validates --data_type
  return cfg;
}

nve::LogLevel_t parse_log_level(const std::string& s) {
  if (s == "critical") return nve::LogLevel_t::Critical;
  if (s == "error") return nve::LogLevel_t::Error;
  if (s == "warning") return nve::LogLevel_t::Warning;
  if (s == "perf") return nve::LogLevel_t::Perf;
  if (s == "info") return nve::LogLevel_t::Info;
  if (s == "verbose") return nve::LogLevel_t::Verbose;
  NVE_THROW_ARG_("invalid --log_level '", s, "' (critical|error|warning|perf|info|verbose)");
}

int run(const RunConfig& cfg) {
  nve::GetGlobalLogger()->set_verbosity_level(parse_log_level(cfg.log_level));
  NVE_CHECK_(cudaSetDevice(cfg.device));
  if (cfg.threadpool_workers > 0) {
    nve::configure_default_thread_pool({{"num_workers", cfg.threadpool_workers}});
  }
  const int64_t row_size = cfg.row_size();
  const int64_t keys_per_step = cfg.keys_per_step();

  // --- Setup: build the layers, populate them ------------------------------------------------
  const double mem_before_gb = mem_available_gb();
  const auto setup_start = clock_type::now();
  TempStorage storage(cfg.keep_temp_files);
  std::vector<LayerBundle> bundles;
  std::vector<nve::context_ptr_t> main_ctxs;
  double prefill_s = 0.0;
  double file_create_s = 0.0;
  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    RunConfig layer_cfg = cfg;
    layer_cfg.layer_idx = l;
    auto& bundle = bundles.emplace_back(build_layer(layer_cfg, storage));
    auto& main_ctx = main_ctxs.emplace_back(bundle.layer->create_execution_context(0, 0, nullptr, nullptr));
    if (cfg.clear) bundle.layer->clear(main_ctx);
    if (cfg.prefill) prefill_s += prefill(layer_cfg, bundle, main_ctx);
    file_create_s += bundle.file_create_s;
  }
  const LayerBundle& bundle = bundles.front();  // every layer shares the spec
  const double setup_s = seconds_between(setup_start, clock_type::now());
  if (!cfg.dump_hier.empty() && !cfg.dumps_tables()) std::cout << "--dump_hier ignored: layer is not hierarchical" << std::endl;
  if (!cfg.load_hier.empty() && !cfg.loads_tables()) std::cout << "--load_hier ignored: layer is not hierarchical" << std::endl;
  const double host_mem_delta_gb = mem_before_gb - mem_available_gb();

  std::cout << "Layer: " << bundle.description << "; ";
  if (cfg.num_layers > 1) std::cout << cfg.num_layers << " layers of ";
  std::cout << cfg.data_type << " x " << cfg.embedding_dim << ", "
            << cfg.num_rows << " rows (" << fmt(static_cast<double>(cfg.table_bytes()) / kGiB, 2) << " GB)";
  if (cfg.layer != "gpu" && cfg.has_gpu_tier()) {
    std::cout << ", GPU cache " << static_cast<int64_t>(cfg.gpu_cache_bytes()) / row_size << " rows ("
              << fmt(static_cast<double>(cfg.gpu_cache_bytes()) / kGiB, 3) << " GB)";
  }
  if (cfg.layer == "hierarchical" && cfg.has_remote()) std::cout << ", host cache " << cfg.host_cache_rows() << " rows";
  std::cout << ", keys on " << (cfg.keys_on_gpu ? "gpu" : "host");
  std::cout << "\nSetup: " << fmt(setup_s, 2) << " s (prefill " << fmt(prefill_s, 2) << " s, file create "
            << fmt(file_create_s, 2) << " s), host mem delta " << fmt(host_mem_delta_gb, 2) << " GB" << std::endl;

  // --- Inference threads ----------------------------------------------------------------------
  // Thread i drives layer i / threads; all threads of all layers share the phase barriers.
  struct ThreadContext {
    cudaStream_t lookup{nullptr}, modify{nullptr};
    nve::context_ptr_t ctx;
  };
  const int64_t total_threads = cfg.num_layers * cfg.threads;
  std::vector<ThreadContext> thread_ctx(static_cast<size_t>(total_threads));
  for (int64_t i = 0; i < total_threads; ++i) {
    auto& tc = thread_ctx[static_cast<size_t>(i)];
    NVE_CHECK_(cudaStreamCreateWithFlags(&tc.lookup, cudaStreamNonBlocking));
    NVE_CHECK_(cudaStreamCreateWithFlags(&tc.modify, cudaStreamNonBlocking));
    tc.ctx = bundles[static_cast<size_t>(i / cfg.threads)].layer->create_execution_context(tc.lookup, tc.modify, nullptr, nullptr);
  }
  std::vector<ThreadResult> results(static_cast<size_t>(total_threads));
  Barrier barrier(total_threads);
  std::vector<std::thread> workers;
  std::mutex error_mutex;
  for (int64_t i = 0; i < total_threads; ++i) {
    workers.emplace_back([&, i] {
      const auto idx = static_cast<size_t>(i);
      try {
        worker_main(cfg, bundles[idx / static_cast<size_t>(cfg.threads)], thread_ctx[idx].ctx, i, barrier, results[idx]);
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(error_mutex);
        std::cerr << "ERROR: layer " << i / cfg.threads << " thread " << i % cfg.threads << ": " << e.what() << std::endl;
        std::exit(2);  // other threads are blocked on the barrier, so there is no clean unwind
      }
    });
  }
  for (auto& w : workers) w.join();

  // --- Aggregate ------------------------------------------------------------------------------
  // Wall time of the throughput phase over results [first, last).
  const auto tput_time = [&](size_t first, size_t last) {
    auto start = results[first].tput_start;
    auto end = results[first].tput_end;
    for (size_t i = first; i < last; ++i) {
      start = std::min(start, results[i].tput_start);
      end = std::max(end, results[i].tput_end);
    }
    return seconds_between(start, end);
  };
  const double infer_time = tput_time(0, results.size());
  const double total_keys = static_cast<double>(total_threads * cfg.num_steps * keys_per_step);
  const double kps = total_keys / infer_time;
  const double gbps = kps * static_cast<double>(row_size) / kGiB;
  std::cout << "Time: " << fmt(infer_time, 2) << ", MKPS: " << fmt(kps / 1e6, 2) << ", Algo BW: " << fmt(gbps, 2) << " GB/s (";
  if (cfg.num_layers > 1) std::cout << cfg.num_layers << " layers x ";
  std::cout << cfg.threads << " thread" << (cfg.threads > 1 ? "s" : "") << " x " << cfg.num_steps << " steps of "
            << cfg.batch << "x" << cfg.hotness << " keys)" << std::endl;
  if (cfg.num_layers > 1) {
    const auto threads = static_cast<size_t>(cfg.threads);
    const double layer_keys = static_cast<double>(cfg.threads * cfg.num_steps * keys_per_step);
    std::cout << "Per-layer MKPS:";
    for (size_t l = 0; l < bundles.size(); ++l) {
      std::cout << (l ? "," : "") << " " << fmt(layer_keys / tput_time(l * threads, (l + 1) * threads) / 1e6, 2);
    }
    std::cout << std::endl;
  }

  const size_t num_tables = static_cast<size_t>(bundle.layer->get_num_tables());
  const std::array<const char*, kMaxTiers> tier_labels{"gpu", "host", "remote"};
  struct TierStats { double mean{std::nan("")}, min{std::nan("")}, max{std::nan("")}; size_t samples{0}; };
  const size_t tier_offset = cfg.has_gpu_tier() ? 0 : 1;  // without a GPU cache table 0 is the host tier
  // Per-tier hit rates of one phase over every thread's samples, indexed by tier label, not by table.
  const auto collect_tier_stats = [&](std::vector<std::vector<double>> ThreadResult::*samples, const char* label) {
    std::vector<TierStats> stats(kMaxTiers);
    if (cfg.hitrate_interval <= 0) return stats;
    std::cout << label << ":";
    for (size_t t = 0; t < num_tables && t + tier_offset < kMaxTiers; ++t) {
      std::vector<double> all;
      for (const auto& r : results) all.insert(all.end(), (r.*samples)[t].begin(), (r.*samples)[t].end());
      auto& s = stats[t + tier_offset];
      s.samples = all.size();
      if (!all.empty()) {
        s.mean = mean_of(all);
        s.min = *std::min_element(all.begin(), all.end());
        s.max = *std::max_element(all.begin(), all.end());
      }
      std::cout << (t ? "," : "") << " " << tier_labels[t + tier_offset] << " (" << bundle.tier_names[t] << ") "
                << fmt(s.mean, 4) << " [" << fmt(s.min, 4) << ", " << fmt(s.max, 4) << "]";
    }
    std::cout << " over " << stats[tier_offset].samples << " samples" << std::endl;
    return stats;
  };
  const std::vector<TierStats> tier_stats = collect_tier_stats(&ThreadResult::hitrates, "Hitrates");
  const std::vector<TierStats> lat_tier_stats =
      cfg.latency_steps > 0 ? collect_tier_stats(&ThreadResult::lat_hitrates, ("Latency-phase hitrates (" + cfg.latency_keys + " keys)").c_str())
                            : std::vector<TierStats>(kMaxTiers);

  std::vector<double> latency;
  for (const auto& r : results) latency.insert(latency.end(), r.latency_ms.begin(), r.latency_ms.end());
  std::sort(latency.begin(), latency.end());
  int64_t over_budget = 0;
  if (!latency.empty()) {
    if (cfg.max_latency_ms > 0) {
      over_budget = static_cast<int64_t>(std::count_if(latency.begin(), latency.end(), [&](double v) { return v > cfg.max_latency_ms; }));
    }
    std::cout << "Latency (" << latency.size() << " steps of " << cfg.batch << "x" << cfg.hotness << " keys): mean "
              << fmt(mean_of(latency)) << " ms, p50 " << fmt(percentile(latency, 0.5)) << ", p95 "
              << fmt(percentile(latency, 0.95)) << ", p99 " << fmt(percentile(latency, 0.99)) << ", max "
              << fmt(latency.back()) << " ms";
    if (cfg.max_latency_ms > 0) std::cout << ", over " << cfg.max_latency_ms << " ms: " << over_budget << "/" << latency.size();
    std::cout << std::endl;
  }

  // --- CSV ------------------------------------------------------------------------------------
  if (!cfg.csv_filename.empty()) {
    MetricList m = {
        {"timestamp", now_string()}, {"hostname", host_name()}, {"gpu", gpu_name(cfg.device)}, {"tag", cfg.tag},
        {"layer", cfg.layer}, {"host_table", cfg.layer == "hierarchical" ? cfg.host_table : ""},
        {"remote_table", cfg.layer == "hierarchical" ? cfg.remote_table : ""},
        {"data_type", cfg.data_type}, {"embedding_dim", std::to_string(cfg.embedding_dim)},
        {"num_table_rows", std::to_string(cfg.num_rows)}, {"batch", std::to_string(cfg.batch)},
        {"hotness", std::to_string(cfg.hotness)}, {"alpha", fmt(cfg.alpha, 4)},
        {"load_factor", fmt(cfg.load_factor, 6)},  // 0 = hierarchical layer without a GPU cache
        {"host_load_factor", fmt(cfg.host_load_factor, 6)},
        {"kernel", std::to_string(cfg.kernel)}, {"num_layers", std::to_string(cfg.num_layers)},
        {"threads", std::to_string(cfg.threads)},  // per layer
        {"num_steps", std::to_string(cfg.num_steps)}, {"num_warmup_steps", std::to_string(cfg.num_warmup_steps)},
        {"warmup_mode", cfg.warmup_mode}, {"latency_keys", cfg.latency_keys},
        {"keys_on", cfg.keys_on_gpu ? "gpu" : "host"},
        {"file_io_engine", cfg.uses_linear_file() ? cfg.file_io_engine : ""},
        {"file_threads", cfg.uses_linear_file() ? std::to_string(cfg.file_threads) : ""},
        {"file_coalesce", cfg.uses_linear_file() ? std::to_string(cfg.file_coalesce) : ""},
        {"inference_time_s", fmt(infer_time, 4)}, {"mkps", fmt(kps / 1e6, 2)}, {"gbps", fmt(gbps, 2)},
    };
    for (size_t t = 0; t < kMaxTiers; ++t) {
      const std::string p = std::string(tier_labels[t]) + "_hitrate_";
      const auto& s = tier_stats[t];
      const bool present = s.samples > 0;
      m.emplace_back(p + "mean", present ? fmt(s.mean, 6) : "");
      m.emplace_back(p + "min", present ? fmt(s.min, 6) : "");
      m.emplace_back(p + "max", present ? fmt(s.max, 6) : "");
      m.emplace_back(p + "samples", std::to_string(present ? s.samples : 0));
    }
    for (size_t t = 0; t < kMaxTiers; ++t) {
      const auto& s = lat_tier_stats[t];
      m.emplace_back("lat_" + std::string(tier_labels[t]) + "_hitrate_mean", s.samples > 0 ? fmt(s.mean, 6) : "");
    }
    const bool has_lat = !latency.empty();
    m.emplace_back("lat_steps", std::to_string(latency.size()));
    m.emplace_back("lat_mean_ms", has_lat ? fmt(mean_of(latency), 4) : "");
    m.emplace_back("lat_p50_ms", has_lat ? fmt(percentile(latency, 0.5), 4) : "");
    m.emplace_back("lat_p95_ms", has_lat ? fmt(percentile(latency, 0.95), 4) : "");
    m.emplace_back("lat_p99_ms", has_lat ? fmt(percentile(latency, 0.99), 4) : "");
    m.emplace_back("lat_max_ms", has_lat ? fmt(latency.back(), 4) : "");
    m.emplace_back("lat_over_budget", cfg.max_latency_ms > 0 ? std::to_string(over_budget) : "");
    m.emplace_back("setup_s", fmt(setup_s));
    m.emplace_back("prefill_s", fmt(prefill_s));
    m.emplace_back("file_create_s", fmt(file_create_s));
    m.emplace_back("host_mem_delta_gb", fmt(host_mem_delta_gb));
    write_csv(cfg.csv_filename, m);
    if (cfg.verbose) std::cout << "Appended metrics to " << cfg.csv_filename << std::endl;
  }

  // --- Teardown: contexts must be released before the layers ----------------------------------
  for (auto& main_ctx : main_ctxs) {
    main_ctx->wait();
    main_ctx.reset();
  }
  for (auto& tc : thread_ctx) {
    tc.ctx->wait();
    tc.ctx.reset();
    NVE_CHECK_(cudaStreamDestroy(tc.lookup));
    NVE_CHECK_(cudaStreamDestroy(tc.modify));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const RunConfig cfg = parse_args(argc, argv);
    return run(cfg);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
