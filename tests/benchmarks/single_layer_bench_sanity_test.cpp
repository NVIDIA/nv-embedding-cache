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

#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <common.hpp>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TempDir {
  std::string path;

  TempDir() {
    std::string pattern = (std::filesystem::temp_directory_path() / "nve_single_layer_bench_XXXXXX").string();
    char* created = ::mkdtemp(pattern.data());
    NVE_CHECK_(created != nullptr, "mkdtemp failed");
    path = created;
  }

  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

struct ProcessResult {
  int status;
  std::string output;
};

ProcessResult run_benchmark(const std::vector<std::string>& arguments) {
  std::vector<char*> argv{const_cast<char*>(NVE_SINGLE_LAYER_BENCH_PATH)};
  for (const auto& arg : arguments) argv.push_back(const_cast<char*>(arg.c_str()));
  argv.push_back(nullptr);

  int pipe_fds[2];
  NVE_CHECK_(::pipe(pipe_fds) == 0, "pipe failed");

  const pid_t child = ::fork();
  if (child < 0) {
    ::close(pipe_fds[0]);
    ::close(pipe_fds[1]);
    NVE_THROW_("fork failed");
  }
  if (child == 0) {
    ::close(pipe_fds[0]);
    ::dup2(pipe_fds[1], STDOUT_FILENO);
    ::dup2(pipe_fds[1], STDERR_FILENO);
    ::close(pipe_fds[1]);
    ::alarm(30);
    ::execv(argv[0], argv.data());
    ::_exit(127);
  }

  ::close(pipe_fds[1]);
  std::string output;
  char buffer[4096];
  bool read_failed = false;
  while (true) {
    const ssize_t bytes = ::read(pipe_fds[0], buffer, sizeof(buffer));
    if (bytes > 0) {
      output.append(buffer, static_cast<size_t>(bytes));
    } else if (bytes == 0) {
      break;
    } else if (errno != EINTR) {
      read_failed = true;
      break;
    }
  }
  ::close(pipe_fds[0]);
  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    NVE_CHECK_(errno == EINTR, "waitpid failed");
  }
  NVE_CHECK_(!read_failed, "read failed");
  return {status, std::move(output)};
}

TEST(SingleLayerBenchSanity, Help) {
  const auto result = run_benchmark({"--help"});
  ASSERT_TRUE(WIFEXITED(result.status)) << result.output;
  EXPECT_EQ(WEXITSTATUS(result.status), 0) << result.output;
  EXPECT_NE(result.output.find("--host_table"), std::string::npos);
  EXPECT_NE(result.output.find("--keep_temp_files"), std::string::npos);
}

struct Scenario {
  const char* name;
  const char* layer;
  const char* host_table;
  const char* remote_table;
  const char* description;
};

std::vector<Scenario> scenarios() {
  std::vector<Scenario> result = {
      {"Gpu", "gpu", "nvhm", "none", "Layer: gpu layer, table in HBM"},
      {"LinearUvm", "linear_uvm", "nvhm", "none", "Layer: linear_uvm layer, GPU cache"},
      {"HierarchicalLinearFile", "hierarchical", "linear_file", "none",
       "Layer: hierarchical layer, GPU cache + linear_file"},
  };
#ifdef NVE_HAVE_NVHM
  result.push_back({"HierarchicalNvhm", "hierarchical", "nvhm", "none",
                    "Layer: hierarchical layer, GPU cache + nvhm"});
  result.push_back({"HierarchicalNvhmRemoteLinearFile", "hierarchical", "nvhm", "linear_file",
                    "Layer: hierarchical layer, GPU cache + nvhm cache + linear_file"});
#endif
#ifdef NVE_HAVE_ROCKSDB
  result.push_back({"HierarchicalRocksdb", "hierarchical", "rocksdb", "none",
                    "Layer: hierarchical layer, GPU cache + rocksdb"});
#endif
  return result;
}

class SingleLayerBenchRun : public ::testing::TestWithParam<Scenario> {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device unavailable";
    }
  }
};

std::vector<std::string> small_run_args(const Scenario& scenario, const TempDir& dir) {
  return {"--layer", scenario.layer, "--host_table", scenario.host_table,
          "--remote_table", scenario.remote_table,
          "--embedding_dim", "5", "--num_table_rows", "8192",
          "--batch", "1", "--hotness", "256", "--alpha", "0.95",
          "--load_factor", "0.2", "--num_steps", "3", "--num_warmup_steps", "2",
          "--latency_steps", "2", "--file_threads", "8", "--rocksdb_threads", "2",
          "--file_dir", dir.path};
}

TEST_P(SingleLayerBenchRun, SmallRun) {
  const auto& scenario = GetParam();
  TempDir dir;
  const auto result = run_benchmark(small_run_args(scenario, dir));
  ASSERT_TRUE(WIFEXITED(result.status)) << result.output;
  ASSERT_EQ(WEXITSTATUS(result.status), 0) << result.output;
  EXPECT_NE(result.output.find(scenario.description), std::string::npos) << result.output;
  EXPECT_NE(result.output.find("Time:"), std::string::npos) << result.output;
  EXPECT_NE(result.output.find("Hitrates:"), std::string::npos) << result.output;
  EXPECT_NE(result.output.find("Latency"), std::string::npos) << result.output;
  EXPECT_TRUE(std::filesystem::is_empty(dir.path)) << result.output;
}

// Each layer gets its own tables, so stores backed by files / RocksDB directories must not collide.
TEST_P(SingleLayerBenchRun, MultiLayer) {
  const auto& scenario = GetParam();
  TempDir dir;
  auto args = small_run_args(scenario, dir);
  args.insert(args.end(), {"--num_layers", "2", "--threads", "2"});
  const auto result = run_benchmark(args);
  ASSERT_TRUE(WIFEXITED(result.status)) << result.output;
  ASSERT_EQ(WEXITSTATUS(result.status), 0) << result.output;
  EXPECT_NE(result.output.find(scenario.description), std::string::npos) << result.output;
  EXPECT_NE(result.output.find("2 layers x 2 threads"), std::string::npos) << result.output;
  EXPECT_NE(result.output.find("Per-layer MKPS:"), std::string::npos) << result.output;
  EXPECT_TRUE(std::filesystem::is_empty(dir.path)) << result.output;
}

TEST_P(SingleLayerBenchRun, KeepAndReuseStorage) {
  const auto& scenario = GetParam();
  const bool linear_file = std::string(scenario.host_table) == "linear_file" ||
                           std::string(scenario.remote_table) == "linear_file";
  const bool rocksdb = std::string(scenario.host_table) == "rocksdb" ||
                       std::string(scenario.remote_table) == "rocksdb";
  if (!linear_file && !rocksdb) GTEST_SKIP() << "Scenario has no file storage";

  TempDir dir;
  auto args = small_run_args(scenario, dir);
  args.insert(args.end(), {"--file_path", dir.path + "/table.bin", "--rocksdb_path", dir.path + "/rocksdb"});
  auto keep_args = args;
  keep_args.insert(keep_args.end(), {"--num_layers", "2", "--keep_temp_files"});
  const auto kept = run_benchmark(keep_args);
  ASSERT_TRUE(WIFEXITED(kept.status)) << kept.output;
  ASSERT_EQ(WEXITSTATUS(kept.status), 0) << kept.output;
  EXPECT_NE(kept.output.find("Creating temp "), std::string::npos) << kept.output;
  EXPECT_NE(kept.output.find("Keeping temp "), std::string::npos) << kept.output;
  EXPECT_EQ(kept.output.find("Removed temp "), std::string::npos) << kept.output;
  std::vector<std::filesystem::path> existing;
  for (const auto& entry : std::filesystem::directory_iterator(dir.path)) existing.push_back(entry.path());
  ASSERT_EQ(existing.size(), 2) << kept.output;

  // Reuse two existing layers and create a third; only the third layer's storage is removed.
  args.insert(args.end(), {"--num_layers", "3"});
  const auto reused = run_benchmark(args);
  ASSERT_TRUE(WIFEXITED(reused.status)) << reused.output;
  ASSERT_EQ(WEXITSTATUS(reused.status), 0) << reused.output;
  EXPECT_NE(reused.output.find("Using existing "), std::string::npos) << reused.output;
  EXPECT_NE(reused.output.find("Removed temp "), std::string::npos) << reused.output;
  for (const auto& path : existing) EXPECT_TRUE(std::filesystem::exists(path)) << reused.output;
  EXPECT_FALSE(std::filesystem::exists(dir.path + (linear_file ? "/table_l2.bin" : "/rocksdb_l2")))
      << reused.output;
}

INSTANTIATE_TEST_SUITE_P(LayersAndTables, SingleLayerBenchRun, ::testing::ValuesIn(scenarios()),
                         [](const ::testing::TestParamInfo<Scenario>& info) { return info.param.name; });

}  // namespace
