# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import pynve.nve as nve
from typing import Optional
from collections.abc import Iterator
import warnings

class SimpleInitializer (Iterator):
    def __init__(self,
                 num_rows: int,
                 row_size: int,
                 data_type: torch.dtype,
                 init_tensor: torch.tensor = None,
                 init_batch: Optional[int] = 2**16):
        self.num_rows = num_rows
        self.row_size = row_size
        self.data_type = data_type
        self.init_tensor = init_tensor.flatten() if init_tensor is not None else None
        if (init_tensor is not None) and (torch.numel(self.init_tensor) < (self.num_rows * self.row_size)):
            raise ValueError("Init tensor is too small")
        self.init_batch = init_batch
        self.current_row = 0

    def __next__(self):
        if self.current_row == self.num_rows:
            raise StopIteration
        end_row = min(self.num_rows, self.current_row + self.init_batch)
        keys = torch.tensor(range(self.current_row, end_row), dtype=torch.int64) 
        values = (
            torch.rand(self.row_size * (end_row - self.current_row), dtype=self.data_type)
            if self.init_tensor is None
            else self.init_tensor[self.current_row * self.row_size : end_row * self.row_size]
        )
        self.current_row = end_row
        return keys, values

class NVLocalParameterServer ():
    """Class for a local parameter server in system memory.

    This class is meant to be used by NVEmbedding of type CacheType.Hierarchical.
    It can be used as a secondary cache, between the GPU cache and a remote parameter server, or as a
    local key-vector storage to back the GPU cache.

    Args:
        num_embeddings (int): Size of the embedding dictionary, when set to 0, the local storage will increase until OOM
                              Use erase or clear to manually remove data.
        embedding_size (int): Size of each embedding vector (in elements)
        data_type (torch.dtype): Data type of embedding vectors (float32 or float16)
        initializer (Optional[Iterator]): Initializer for the local data.
                                          Iterator must return a tuple of tensors, keys(int64) and values(vector with embedding_size elements of data_type per key)
        initial_size (Optional[int]): initial amount of embeddings to allocate
        ps_type (Optional[nve.PSType_t]): type of parameter server backend to use
    """
    def __init__(self, 
                 num_embeddings: int,
                 embedding_size: int,
                 data_type: torch.dtype,
                 initializer: Optional[Iterator] = None,
                 initial_size: Optional[int] = 1024,
                 ps_type: Optional[nve.PSType_t] = nve.NVHashMap):
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        if data_type == torch.float32:
            self.layer_data_type = nve.DataType_t.Float32
        elif data_type == torch.float16:
            self.layer_data_type = nve.DataType_t.Float16
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        self.local_parameter_server = nve.LocalParameterServer(self.num_embeddings, self.embedding_size, self.layer_data_type, initial_size, ps_type)

        if initializer:
            while True:
                try:
                    keys,values = next(initializer)
                    self.insert(keys, values)
                except StopIteration:
                    break

    def insert(self, keys: torch.Tensor, values: torch.Tensor):
        """Method to insert key-value pairs to the parameter server.

        Args:
            keys (torch.Tensor): Tensor of int64 keys to insert
            values (torch.Tensor): Tensor of value vectors (containing embedding_size elements each), matching the given keys
        """
        if keys.dtype is not torch.int64:
            warnings.warn(f"Keys provided to NVLocalParameterServer were of type {keys.dtype} instead of int64 (converting on the fly)")
            keys = keys.long()
        values_size = torch.numel(values)
        values_needed = torch.numel(keys) * self.embedding_size
        if values_size < values_needed:
            raise ValueError(f"Values tensor has too few elements, expected {values_needed} - got {values_size}")
        self.local_parameter_server.insert_keys(torch.numel(keys), keys.data_ptr(), values.data_ptr())

    def load_from_file(self, keys_path: str, values_path: str, batch_size: Optional[int] = 1024*1024):
        """Method to load key-value pairs from a numpy file.

        Args:
            keys_path (str): Path to the keys file
            values_path (str): Path to the values file
            batch_size (int): Number of keys to load from the file at a time
        """
        
        self.local_parameter_server.insert_keys_from_filepath(keys_path, values_path, batch_size)

    def erase(self, keys: torch.Tensor):
        """Method to erase key-value pairs from the parameter server.

        Args:
            keys (torch.Tensor): Tensor of int64 keys to erase
        """
        if keys.dtype is not torch.int64:
            warnings.warn(f"Keys provided to NVLocalParameterServer were of type {keys.dtype} instead of int64 (converting on the fly)")
            keys = keys.long()
        self.local_parameter_server.erase_keys(torch.numel(keys), keys.data_ptr())

    def clear(self):
        """Method to clear all keys from the parameter server.
        """
        self.local_parameter_server.clear_keys()
