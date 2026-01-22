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

#include "sys/types.h"
#include "sys/sysinfo.h"
#include <iostream>
#include <iomanip>
#include "common.h"

void ReportMemUsage(std::string msg) {
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    int64_t page_size = memInfo.mem_unit;
    constexpr int64_t MB(1024*1024);
    constexpr int64_t GB(1024*MB);
    const double page_gb = static_cast<double>(page_size) / static_cast<double>(GB);
    
    double cpu_totalGB = double(memInfo.totalram) * page_gb;
    double cpu_freeGB = double(memInfo.freeram) * page_gb;
    double cpu_usedGB = cpu_totalGB - cpu_freeGB;

    size_t gpu_free = 0;
    size_t gpu_total = 0;
    gpuErrChk(cudaMemGetInfo(&gpu_free, &gpu_total));

    double gpu_totalGB = double(gpu_total) / GB;
    double gpu_freeGB = double(gpu_free) / GB;
    double gpu_usedGB = gpu_totalGB - gpu_freeGB;

    std::cout << "[MEM_USAGE]" << std::fixed << std::setprecision(2)
    << "[ " << msg << " ]" << std::endl;
    std::cout << "\t[ HOST ] Used: " << cpu_usedGB << " GB"
    << ",  Free: " << cpu_freeGB << " GB"
    << ",  Total: " << cpu_totalGB << " GB"
    << std::endl;
    std::cout <<  "\t[ GPU ] Used: " << gpu_usedGB << " GB"
    << ",  Free: " << gpu_freeGB << " GB"
    << ",  Total: " << gpu_totalGB << " GB"
    << std::endl;
}

std::ostream& operator<<(std::ostream& os, const BandwidthFormatter& bf)
{
    auto bps = bf.bytes / bf.seconds;
    std::string units("B/s");

    // sorted bandwidth unit scales
    static const std::vector<std::pair<double,std::string>> Scales = {
        {1e12, "TB/s"},
        {1e9, "GB/s"},
        {1e6, "MB/s"},
        {1e3, "KB/s"},
    };

    for (auto& scale : Scales) {
        if (bps > scale.first) {
            bps /= scale.first;
            units = scale.second;
            break;
        }
    }
    os << bps << " " << units;
    return os;
}
