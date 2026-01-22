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
#include <vector>
#include <chrono>
#include <sstream>
#include <algorithm>
#include "cuda_ops/cuda_common.h"

// Performance metrics container
class CacheIterationMetric {
public:
    class BasicMetric {
    public:
        //BasicMetric(const BasicMetric&) = delete;
        BasicMetric() {
            NVE_CHECK_(cudaEventCreate(&d_start));
            NVE_CHECK_(cudaEventCreate(&d_end));
        }
        ~BasicMetric() {
            NVE_CHECK_(cudaEventDestroy(d_start));
            NVE_CHECK_(cudaEventDestroy(d_end));
        
        }
        cudaEvent_t d_start;
        cudaEvent_t d_end;
        using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
        TimePoint h_start;
        TimePoint h_end;
    };
    BasicMetric lookup;
    BasicMetric insert;
    BasicMetric scatter;
    BasicMetric pooling;
    BasicMetric dedup_gradients;
    BasicMetric accumulate;
    BasicMetric host_cache_lookup;
    BasicMetric host_cache_accumulate;
    BasicMetric ps_lookup;
    BasicMetric ps_accumulate;
    BasicMetric gpu_fwd_bwd;
    BasicMetric all_to_all; 
    double hitrate{0};
    bool insert_done{false};
    bool gpu_work_recorded{false};

    static inline BasicMetric::TimePoint Now() { return std::chrono::high_resolution_clock::now(); }
};

template <typename T>
class DistributionParser {
public:
    DistributionParser(
        std::string short_name,
        std::string long_name,
        std::vector<T>& values,
        std::vector<float> percentiles = {0.0f, 0.05f, 0.5f, 0.95f, 1.0f}
    ) : m_short_name(short_name), m_long_name(long_name), m_percentiles(percentiles), m_num_values(values.size())
    {
        std::sort(values.begin(), values.end());
        for (auto p : m_percentiles)
        {
            if (m_num_values == 0) {
                m_dist.push_back(T(0));
            } else {
                size_t pos(static_cast<size_t>(p * static_cast<double>(m_num_values)));
                pos = std::min(pos, m_num_values - 1);
                m_dist.push_back(values[pos]);
            }
        }
        m_avg = std::accumulate(values.begin(), values.end(), 0.f) / float(m_num_values ? m_num_values : size_t(1));
    }
    std::string OutputString() const {
        std::stringstream ss;
        for (size_t i=0 ; i<m_percentiles.size() ; i++) {
            ss << std::to_string(static_cast<int>(m_percentiles[i] * 100)) << "% = "
               << m_dist.at(i) << ", ";
        }
        ss << "Avg = " << m_avg;
        return ss.str();
    }
    std::string CSVString() const {
        std::stringstream ss;
        for (auto d : m_dist) {
            ss << d << ",";
        }
        ss << m_avg << ",";
        return ss.str();
    }
    std::string CSVHeader() const {
        std::stringstream ss;
        for (auto p : m_percentiles) {
            ss << m_short_name << "_" << std::to_string(static_cast<int>(p * 100)) << "%,";
        }
        ss << "Avg,";
        return ss.str();
    }
    size_t NumValues() const { return m_num_values; }
    const std::string& ShortName() const { return m_short_name; }
    const std::string& LongName() const { return m_long_name; }
private:
    std::string m_short_name;
    std::string m_long_name;
    std::vector<float> m_percentiles;
    std::vector<float> m_dist;
    float m_avg;
    const size_t m_num_values;
};
