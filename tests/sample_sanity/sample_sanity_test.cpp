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
#include <cstdlib>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <common.hpp>

class SampleSanity : public ::testing::TestWithParam<std::vector<std::string>> {
public:    
    void TestSample(const std::vector<std::string> commands) {
        auto root_path = std::filesystem::canonical("/proc/self/exe").parent_path();
        root_path += std::filesystem::path("/");
        for (auto& cmd : commands) {
            auto cmd_with_path = root_path;
            cmd_with_path += std::filesystem::path(cmd);
            if (cmd[0] == '!' ) {
                // Commands starting with '!' are to be executed as-is with no path manipulation
                cmd_with_path = cmd.substr(1);
            }
            std::cout << "Running " << cmd_with_path << std::endl;
            std::string full_cmd = cmd_with_path.string() + std::string(" > /dev/null 2>&1");
            NVE_IF_DEBUG_(full_cmd = cmd_with_path.string()); // In debug leave output visible
            EXPECT_EQ(std::system(full_cmd.c_str()), 0); // Check command ran and returned 0
        }
    }
};

TEST_P(SampleSanity, NoError) {
    TestSample(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Sanity,
    SampleSanity,
    ::testing::Values(
        std::vector<std::string>{
            "../../samples/import_sample/gen_np_files.py",  // Generate temporary files needed by the sample
            "import_sample",                                // Run the sample
            "!rm /tmp/keys.npy /tmp/values.npy"},           // Cleanup temporary files
        std::vector<std::string>{"simple_cpp"},
        std::vector<std::string>{"simple_sample"},
        std::vector<std::string>{"layer_sample"},
        std::vector<std::string>{"c_api_sample"},
        std::vector<std::string>{"c_api_custom_remote_sample"}
    )
);
