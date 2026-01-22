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
 *      profiler.StartSession();
 *      for (int i=0 ; i < NUM_PASSES ; i++) {
 *          profiler.StartPass();
 *
 *          profiler.PushRange("Range1");
 *              // Profiling for Range1
 *          profiler.PopRange();
 *
 *          profiler.PushRange("Range2");
 *              // Profiling for Range2
 *          profiler.PopRange();
 *
 *          profiler.StopPass();
 *      }
 *      profiler.StopSession();
 *      auto metrics = profiler.GetMetrics();
 * }
*/

class CuptiProfiler {
public:
    using ProfilerMetrics = std::map<std::string,std::map<std::string,double>>; // < RangeName, < MetricName, MetricValue > >
    CuptiProfiler(int device_id = 0);
    ~CuptiProfiler();

    // Start a session to setup metrics
    bool StartSession(
        const std::vector<std::string>& metrics = {
            "dram__bytes_write.sum",
            "dram__bytes_read.sum",
            },
        uint32_t maxRanges = 1000
    );
    void StopSession();

    // Passes should be started within a session
    // Use multiple passes to average values
    void StartPass();
    void EndPass();

    // Ranges should be used within a pass
    // Use ranges within passes to separate measurements on meaningful intervals
    void PushRange(const std::string& name);
    void PopRange();

    void PrintMetrics(bool legacy_print=false) const;
    ProfilerMetrics GetMetrics() const;

    // Can only be used before a profiling session has been started
    std::vector<std::string> GetAllMetricNames() const;

private:
    std::vector<std::string> m_metricNames;
    std::vector<uint8_t> m_counterDataImage;
    std::vector<uint8_t> m_counterDataScratchBuffer;
    std::vector<uint8_t> m_counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::string m_chipName;

    bool CreateCounterDataImage(
        std::vector<uint8_t>& counterDataImage,
        std::vector<uint8_t>& counterDataScratchBuffer,
        std::vector<uint8_t>& counterDataImagePrefix,
        uint32_t maxRanges);
};
