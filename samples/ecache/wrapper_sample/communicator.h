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
#include <nccl.h>
#include <string>
#include <cuda.h>

class IPCMessage;

class Communicator
{
public:
    Communicator(uint32_t nrank, uint32_t rank, const std::string& file_path, int device_id);

    // send a buffer size worth of data to all peers from src
    // recieve a buffer size worth of data to peer i to dst + i * buffer_size
    // require a syhc after call on stream
    void AllToAll(const int8_t* src, int8_t* dst, size_t buffer_size, cudaStream_t stream);
    uint32_t GetNumRanks() const { return m_nranks; }
    ~Communicator();
private:
    ncclComm_t m_comm;
    uint32_t m_nranks;
    uint32_t m_rank;
    int m_device_id;
    ncclUniqueId m_id;
    IPCMessage* m_ipc; // helper class for sending the unique id across ipcs via shared file
};
