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
#include "common.h"
#include <cuda.h>
#include <stddef.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <numaif.h>

template<typename T>
inline T divUp(T m, T n)
{
  return (m + (n  - 1)) / n;
}

template<typename T>
inline T alignTo(T m, T n)
{
  return divUp(m, n) * n;
}

class MemBlock
{
public:
  MemBlock() : m_ptr(nullptr), m_totalBytes(0) {}
  virtual ~MemBlock() {}
  int8_t* getPtr() const { return m_ptr; }
  size_t getSize() const { return m_totalBytes; }
protected:
  
  int8_t* m_ptr;
  size_t m_totalBytes;
};

class MemBlockMultiNode : public MemBlock
{
public:
  static constexpr size_t kLargePageSizeBytes = size_t(2) * 1024 * 1024;
  MemBlockMultiNode(size_t szToAllocate) 
  {
    int currDevice = 0, nDevices = 0;
    gpuErrChk(cudaGetDevice(&currDevice));
    assert(currDevice == 0);
    gpuErrChk(cudaGetDeviceCount(&nDevices));
    std::vector<int> gpu_ids;
    std::vector<size_t> bytes_per_gpu;
    size_t memToAllocate = szToAllocate;
    for (int peerDevice = 1; peerDevice < nDevices && memToAllocate > 0; peerDevice++)
    {
        gpuErrChk(cudaSetDevice(peerDevice));
        size_t avail_phy_vidmem = 0, total_phy_vidmem = 0;
        gpuErrChk(cudaMemGetInfo(&avail_phy_vidmem, &total_phy_vidmem));
        size_t limited_phy_mem = alignTo((size_t)(0.99f * static_cast<float>(avail_phy_vidmem)) , kLargePageSizeBytes); // memory we can allocate
        size_t gpuBytes = memToAllocate > limited_phy_mem ? limited_phy_mem : memToAllocate;

        assert(gpuBytes % kLargePageSizeBytes == 0);
        gpu_ids.push_back(peerDevice);
        bytes_per_gpu.push_back(gpuBytes);
        memToAllocate -= gpuBytes;
    }
    m_handles.resize(gpu_ids.size());
    for (size_t i = 0; i < gpu_ids.size(); ++i) 
    {
        CUmemAllocationProp prop;
        memset(&prop, 0, sizeof(CUmemAllocationProp));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = gpu_ids[i];
    
        // Create context on each GPU
        gpuErrChk(cudaSetDevice(gpu_ids[i]));
        gpuErrChk(cudaFree(0));
        gpuErrChk(cudaSetDevice(0));
    
        if (bytes_per_gpu[i] != 0) 
        {
          assert(bytes_per_gpu[i] % kLargePageSizeBytes == 0);
          drvErrChk(cuMemCreate(&m_handles[i], bytes_per_gpu[i], &prop, 0));
        }
    }
  
    // Reserve VA space for the total size of allocation.
    m_totalBytes = std::accumulate(bytes_per_gpu.begin(), bytes_per_gpu.end(), size_t(0));
    drvErrChk(cuMemAddressReserve(&ptr_, m_totalBytes, 0, 0, 0));
  
    // Map each GPU's physical allocation to corresponding portion of VA space.
    size_t offset = 0;
    for (size_t i = 0; i < bytes_per_gpu.size(); ++i) 
    {
        if (bytes_per_gpu[i] != 0) 
        {
            drvErrChk(cuMemMap(ptr_ + offset, bytes_per_gpu[i], 0, m_handles[i], 0));
            offset = offset + bytes_per_gpu[i];
        }
    }

    // Make allocation visible to all local GPUs.
    int num_local_gpus = 0;
    gpuErrChk(cudaGetDeviceCount(&num_local_gpus));
    std::vector<CUmemAccessDesc> access_descriptors(static_cast<size_t>(num_local_gpus));
    for (size_t i = 0; i < access_descriptors.size(); ++i) 
    {
        access_descriptors[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_descriptors[i].location.id = static_cast<int>(i);  // ID of the GPU which will get access enabled by the call below.
        access_descriptors[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    drvErrChk(cuMemSetAccess(ptr_, m_totalBytes, access_descriptors.data(), access_descriptors.size()));
  
    gpuErrChk(cudaDeviceSynchronize());
    gpuErrChk(cudaSetDevice(currDevice));

    m_ptr = reinterpret_cast<int8_t*>(ptr_);
    assert(m_ptr);
  }

  ~MemBlockMultiNode()
  {
    drvErrChk(cuMemUnmap(ptr_, m_totalBytes));
    for (const auto& handle : m_handles) 
    {
      drvErrChk(cuMemRelease(handle));
    }
    drvErrChk(cuMemAddressFree(ptr_, m_totalBytes));
    m_ptr = nullptr;
  
  }

private:
  std::vector<CUmemGenericAllocationHandle> m_handles;
  CUdeviceptr ptr_;
};

class MemBlockManaged : public MemBlock
{
public:
  MemBlockManaged(size_t szToAllocate)
  {
    m_totalBytes = szToAllocate;
    gpuErrChk(cudaMallocManaged(&m_ptr, szToAllocate));
    gpuErrChk(cudaMemAdvise(m_ptr, szToAllocate, cudaMemAdviseSetAccessedBy, 0));
    gpuErrChk(cudaMemAdvise(m_ptr, szToAllocate, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  }

  ~MemBlockManaged()
  {
    gpuErrChk(cudaFree(m_ptr));
  }
};

class MemBlockDevice : public MemBlock
{
public:
  MemBlockDevice(size_t szToAllocate)
  {
    m_totalBytes = szToAllocate;
    gpuErrChk(cudaMalloc(&m_ptr, szToAllocate));
  }

  ~MemBlockDevice()
  {
    gpuErrChk(cudaFree(m_ptr));
  }
};

class MemBlockHost : public MemBlock
{
public:
  MemBlockHost(size_t szToAllocate)
  {
      m_totalBytes = szToAllocate;
      gpuErrChk(cudaMallocHost(&m_ptr, szToAllocate));
  }

  ~MemBlockHost()
  {
    gpuErrChk(cudaFreeHost(m_ptr));
  }
};

class MemBlockSingleNode : public MemBlock
{
public:
  MemBlockSingleNode(size_t szToAllocate)
  {
      m_totalBytes = szToAllocate;
      int nDevices = 0;
      int currDevice = 0;
      int peerDevice = 1;
      gpuErrChk(cudaGetDeviceCount(&nDevices));
      assert(nDevices > 1);
      gpuErrChk(cudaGetDevice(&currDevice));
      assert(currDevice == 0);
      int ret = 0;
      // since this function run in a loop we can get values different than ret
      ret = cudaDeviceEnablePeerAccess(peerDevice, 0);
      if (ret != cudaErrorPeerAccessAlreadyEnabled && ret != cudaSuccess)
      {
          printf("Cant access peer device\n");
          assert(0);
      }
      gpuErrChk(cudaDeviceCanAccessPeer(&ret, currDevice, peerDevice));
      assert(ret == 1);
      gpuErrChk(cudaSetDevice(peerDevice));
      gpuErrChk(cudaMalloc(&m_ptr, szToAllocate));
      gpuErrChk(cudaSetDevice(currDevice));
  }

  ~MemBlockSingleNode()
  {
    gpuErrChk(cudaFree(m_ptr));
  }
};

#define PAGESIZE_BITS_2MB   (21)
#define PAGESIZE_BITS_512MB (29)
#define PAGESIZE_BITS_16GB  (34)

template<int PAGESIZE_BITS, int NUMA_NODE=-1>
class MemBlockMMAP : public MemBlock
{
public:
  static constexpr size_t kLargePageSizeBytes = size_t(1) << PAGESIZE_BITS;
  MemBlockMMAP(size_t szToAllocate)
  {
#if 0
    std::cout << "Allocating " << szToAllocate << " bytes on numa node " << NUMA_NODE << std::endl;
#endif
    m_totalBytes = alignTo(szToAllocate, kLargePageSizeBytes);
    void* ptr = mmap(NULL, m_totalBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_HUGETLB | (PAGESIZE_BITS << MAP_HUGE_SHIFT), -1, 0);
    cpuErrChk(ptr != MAP_FAILED, "Failed to allocate using MMAP()!");
    if (ptr == MAP_FAILED) {
      std::cerr << "mmap failed: " << std::strerror(errno) << std::endl;
      throw std::runtime_error("mmap failed!");
    }
    else {
      m_ptr = static_cast<int8_t*>(ptr);
    }

    if (NUMA_NODE > -1) {
        unsigned long nodemask = 0;
        nodemask |= 1 << NUMA_NODE;
        if (mbind(m_ptr, m_totalBytes, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0) < 0) {
            std::cerr << "Failed on MBIND with error: " << std::strerror(errno) << std::endl;
            throw std::runtime_error("mbind failed!");
        }
    }
  }

  ~MemBlockMMAP()
  {
    int res = munmap(m_ptr, m_totalBytes); // munmap must get size in multiples of page_size (https://man7.org/linux/man-pages/man2/mmap.2.html)
    if (res) {
      std::cerr << "munmap failed: " << std::strerror(errno) << std::endl;
    }
  }
};
