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

#include "communicator.h"
#include "cuda_ops/cuda_common.h"
#include <fstream>
#include <iostream>
#include <string>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(res);                                      \
  }                                                 \
} while(0)


class IPCMessage
{
public:
    IPCMessage(const std::string& file_path) : m_file_path(file_path) {}
    void Send(char* buf, uint32_t sz_buf)
    {
        std::fstream f(m_file_path.c_str() , std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        if (f.fail())
        {
            throw std::runtime_error("Failed to open communication file");
        }
        f.write(buf, sz_buf);  
        f.close();
    }
    void Recieve(char* buf, uint32_t sz_buf)
    {
        bool bo_do = true;
        while (bo_do)
        {
            std::ifstream iif(m_file_path.c_str(), std::ios_base::in | std::ios_base::binary);
            if (iif.is_open())
            {
                bo_do = false;
            }
        }
        
        std::ifstream f(m_file_path.c_str(), std::ios_base::in | std::ios_base::binary);
        auto byte_read = f.readsome(buf, sz_buf);
        while(byte_read < sz_buf)
        {
            byte_read += f.readsome(buf + byte_read, (sz_buf-byte_read));
        }
    }
    ~IPCMessage()
    {
        //clear the file for later usages
        std::fstream f(m_file_path.c_str() , std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
        f.close();
    }
private:
    std::string m_file_path;
};

Communicator::Communicator(uint32_t nrank, uint32_t rank, const std::string& file_path, int device_id) : 
                m_nranks(nrank), 
                m_rank(rank), 
                m_device_id(device_id)
{
    ScopedDevice scope(device_id);
    m_ipc = new IPCMessage(file_path);
	NCCLCHECK(ncclGetUniqueId(&m_id));
    if (rank == 0)
    {
        m_ipc->Send((char*)&m_id, sizeof(m_id));
    }
    else
    {
        m_ipc->Recieve((char*)&m_id, sizeof(m_id));
    }
	NCCLCHECK(ncclCommInitRank(&m_comm, m_nranks, m_id, m_rank));
}

void Communicator::AllToAll(const int8_t* src, int8_t* dst, size_t buffer_size, cudaStream_t stream)
{
    ScopedDevice scope(m_device_id);
    NCCLCHECK(ncclGroupStart());
    for (uint32_t i = 0; i < m_nranks; i++)
    {
        
        NCCLCHECK(ncclSend(src + i*buffer_size, buffer_size, ncclInt8, i, m_comm, stream));
        NCCLCHECK(ncclRecv(dst + i*buffer_size, buffer_size, ncclInt8, i, m_comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());   
}

Communicator::~Communicator()
{
    ScopedDevice scope(m_device_id);
    ncclCommDestroy(m_comm);
    delete m_ipc;
}
