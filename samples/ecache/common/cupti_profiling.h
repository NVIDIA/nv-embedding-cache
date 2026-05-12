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

#include <bits/stdint-uintn.h>
#include <string>
#include <vector>
#include <map>


/**
 * Example usage:
 * 
 * {
 *      CuptiProfiler profiler;
 *      profiler.start_session();
 *      for (int i=0 ; i < NUM_PASSES ; i++) {
 *          profiler.start_pass();
 *
 *          profiler.push_range("Range1");
 *              // Profiling for Range1
 *          profiler.pop_range();
 *
 *          profiler.push_range("Range2");
 *              // Profiling for Range2
 *          profiler.pop_range();
 *
 *          profiler.end_pass();
 *      }
 *      profiler.stop_session();
 *      auto metrics = profiler.get_metrics();
 * }
*/

class CuptiProfiler {
public:
    using ProfilerMetrics = std::map<std::string,std::map<std::string,double>>; // < RangeName, < MetricName, MetricValue > >
    CuptiProfiler(int device_id = 0);
    ~CuptiProfiler();

    // Start a session to setup metrics
    bool start_session(
        const std::vector<std::string>& metrics = {
            "dram__bytes_write.sum",
            "dram__bytes_read.sum",
            },
        uint32_t maxRanges = 1000
    );
    void stop_session();

    // Passes should be started within a session
    // Use multiple passes to average values
    void start_pass();
    void end_pass();

    // Ranges should be used within a pass
    // Use ranges within passes to separate measurements on meaningful intervals
    void push_range(const std::string& name);
    void pop_range();

    void print_metrics(bool legacy_print=false) const;
    ProfilerMetrics get_metrics() const;

    // Can only be used before a profiling session has been started
    std::vector<std::string> get_all_metric_names() const;

private:
    std::vector<std::string> metric_names_;
    std::vector<uint8_t> counter_data_image_;
    void* host_object_ = nullptr; // CUpti_Profiler_Host_Object*, used on CUDA >= 12.5
    std::vector<uint8_t> counter_data_scratch_buffer_;
    std::vector<uint8_t> counter_data_image_prefix_;
    std::vector<uint8_t> config_image_;
    std::string chip_name_;

    bool create_counter_data_image(
        std::vector<uint8_t>& counterDataImage,
        std::vector<uint8_t>& counterDataScratchBuffer,
        std::vector<uint8_t>& counterDataImagePrefix,
        uint32_t maxRanges);
};
