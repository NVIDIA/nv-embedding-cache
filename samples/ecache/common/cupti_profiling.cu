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

#include "cupti_profiling.h"

#include <iostream>
#include <iomanip>
#include <cuda.h>

#include <cupti.h>
#include <helper_cupti.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#include <nvperf_cuda_host.h>

#include <Metric.h>
#include <Eval.h>
#include <Utils.h>
#include <Parser.h>
#include <List.h>


#ifdef __COVERITY__
// For Coverity scans replace CUPTI DRIVER_API_CALL with a simplified version that doesn't 
// trigger error_interface for the unchecked cuGetErrorString call.
#undef DRIVER_API_CALL
#define DRIVER_API_CALL(apiFunctionCall)                                            \
do                                                                                  \
{                                                                                   \
    CUresult _status = apiFunctionCall;                                             \
    if (_status != CUDA_SUCCESS)                                                    \
    {                                                                               \
        std::cerr << "\n\nError: " << __FILE__ << ":" << __LINE__ << ": Function "  \
        << #apiFunctionCall << " failed with error(" << _status << ").\n\n";        \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)
#endif // __COVERITY__

// Disabling warnings of code mostly copied from CUPTI samples (keeping as close as possible to original)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

CuptiProfiler::CuptiProfiler(int device_id) {
    DRIVER_API_CALL(cuInit(0));
    CUdevice device;
    DRIVER_API_CALL(cuDeviceGet(&device, device_id));

    // Initialize profiler API and test device compatibility
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };

    params.cuDevice = device_id;

    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        std::cerr << "Unable to profile on device " << device_id << std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice architecture is not supported" << std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice sli configuration is not supported" << std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice vgpu configuration is not supported" << std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            std::cerr << "\tdevice vgpu configuration disabled profiling support" << std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tdevice confidential compute configuration is not supported" << std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << std::endl;
        }
        return;
    }

    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = static_cast<size_t>(device_id);
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    m_chipName = (getChipNameParams.pChipName);

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
}

CuptiProfiler::~CuptiProfiler(){
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

bool CuptiProfiler::StartSession(const std::vector<std::string>& metrics, uint32_t maxRanges) {
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.ctx = cuContext;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    std::vector<uint8_t> counterAvailabilityImage;
    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    m_metricNames = metrics;
    if (m_metricNames.size())
    {
        if(!NV::Metric::Config::GetConfigImage(m_chipName, m_metricNames, configImage, counterAvailabilityImage.data()))
        {
            std::cout << "Failed to create configImage" << std::endl;
            return false;
        }
        if(!NV::Metric::Config::GetCounterDataPrefixImage(m_chipName, m_metricNames, m_counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        return false;
    }

    if(!CreateCounterDataImage(m_counterDataImage, m_counterDataScratchBuffer, m_counterDataImagePrefix, maxRanges))
    {
        std::cout << "Failed to create counterDataImage" << std::endl;
        return false;
    }

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = m_counterDataImage.size();
    beginSessionParams.pCounterDataImage = &m_counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = m_counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &m_counterDataScratchBuffer[0];
    beginSessionParams.range = CUPTI_UserRange;
    beginSessionParams.replayMode = CUPTI_UserReplay;
    beginSessionParams.maxRangesPerPass = maxRanges;
    beginSessionParams.maxLaunchesPerPass = maxRanges;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

    return true;
}

void CuptiProfiler::StopSession() {
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
}

void CuptiProfiler::StartPass() {
    CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
}

void CuptiProfiler::EndPass() {
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
}

void CuptiProfiler::PushRange(const std::string& name) {
    CUpti_Profiler_PushRange_Params pushRangeParamsA = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    pushRangeParamsA.pRangeName = name.c_str();
    CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParamsA));
}

void CuptiProfiler::PopRange() {
    CUpti_Profiler_PopRange_Params popRangeParamsA = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParamsA));
}

void CuptiProfiler::PrintMetrics(bool legacy_print) const {
    if (legacy_print) {
        NV::Metric::Eval::PrintMetricValues(m_chipName, m_counterDataImage, m_metricNames);
        return;
    }
    auto cupti_metrics = GetMetrics();
    auto saved_flags = std::cout.flags(); // save stream flags
    for (auto& region_metrics : cupti_metrics) {
        std::cout << region_metrics.first << ":" << std::endl;
        for (auto& metric : region_metrics.second) {
            std::cout << " - " << std::setw(100) << std::left << metric.first << metric.second << std::endl;
        }
    }
    std::cout.flags(saved_flags); // restore stream flags
}

CuptiProfiler::ProfilerMetrics CuptiProfiler::GetMetrics() const {
    CuptiProfiler::ProfilerMetrics all_metrics;

    if (!m_counterDataImage.size())
    {
        std::cerr << "Counter Data Image is empty!" << std::endl;
        return all_metrics;
    }
    const uint8_t* pCounterAvailabilityImage = nullptr;
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calculateScratchBufferSizeParam.pChipName = m_chipName.c_str();
    calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
    RETURN_IF_NVPW_ERROR({}, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

    std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
    metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
    metricEvaluatorInitializeParams.pChipName = m_chipName.c_str();
    metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    metricEvaluatorInitializeParams.pCounterDataImage = m_counterDataImage.data();
    metricEvaluatorInitializeParams.counterDataImageSize = m_counterDataImage.size();
    RETURN_IF_NVPW_ERROR({}, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
    NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = m_counterDataImage.data();
    RETURN_IF_NVPW_ERROR({}, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

    std::string reqName;
    bool isolated = true;
    bool keepInstances = true;
    for (std::string metricName : m_metricNames)
    {
        NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
        NVPW_MetricEvalRequest metricEvalRequest;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
        convertMetricToEvalRequest.pMetricName = reqName.c_str();
        convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
        convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        RETURN_IF_NVPW_ERROR({}, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

        for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
        {
            NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
            getRangeDescParams.pCounterDataImage = m_counterDataImage.data();
            getRangeDescParams.rangeIndex = rangeIndex;
            RETURN_IF_NVPW_ERROR({}, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
            std::vector<const char*> descriptionPtrs(getRangeDescParams.numDescriptions);
            getRangeDescParams.ppDescriptions = descriptionPtrs.data();
            RETURN_IF_NVPW_ERROR({}, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

            std::string rangeName;
            for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
            {
                if (descriptionIndex)
                {
                    rangeName += "/";
                }
                rangeName += descriptionPtrs[descriptionIndex];
            }

            NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = { NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE };
            setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
            setDeviceAttribParams.pCounterDataImage = m_counterDataImage.data();
            setDeviceAttribParams.counterDataImageSize = m_counterDataImage.size();
            RETURN_IF_NVPW_ERROR({}, NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

            double metricValue;
            NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = { NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
            evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
            evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
            evaluateToGpuValuesParams.numMetricEvalRequests = 1;
            evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
            evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
            evaluateToGpuValuesParams.pCounterDataImage = m_counterDataImage.data();
            evaluateToGpuValuesParams.counterDataImageSize = m_counterDataImage.size();
            evaluateToGpuValuesParams.rangeIndex = rangeIndex;
            evaluateToGpuValuesParams.isolated = true;
            evaluateToGpuValuesParams.pMetricValues = &metricValue;
            RETURN_IF_NVPW_ERROR({}, NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams));

            all_metrics[rangeName][metricName] = metricValue;
        }
    }
    
    NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
    metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
    RETURN_IF_NVPW_ERROR({}, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
    return all_metrics;
}

bool CuptiProfiler::CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix,
    uint32_t maxRanges)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = maxRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = maxRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

std::vector<std::string> CuptiProfiler::GetAllMetricNames() const {
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    getCounterAvailabilityParams.ctx = cuContext;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    std::vector<uint8_t> counterAvailabilityImage;
    counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
    getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    std::vector<std::string> metric_list;
    NV::Metric::Enum::ExportSupportedMetrics(m_chipName.c_str(), true /* listSubMetrics */, counterAvailabilityImage.data(), metric_list);
    return metric_list;
}

#pragma GCC diagnostic pop // -Wmissing-field-initializers
