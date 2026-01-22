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

import pynve.torch.nve_ops as nve_ops
from pynve.torch.nve_tensors import CachedTable
import pynve.nve as nve
import pynve.torch.nve_ps as nve_ps
import torch
from enum import Enum
from typing import Optional

def nve_type_to_torch_type(nve_type: nve.DataType_t):
    if nve_type == nve.DataType_t.Float32:
        return torch.float32
    elif nve_type == nve.DataType_t.Float16:
        return torch.float16
    else:
        raise ValueError(f"Invalid data type: {nve_type}")

def torch_type_to_nve_type(torch_type: torch.dtype):
    if torch_type == torch.float32:
        return nve.DataType_t.Float32
    elif torch_type == torch.float16:
        return nve.DataType_t.Float16
    else:
        raise ValueError(f"Invalid data type: {torch_type}")

def config_to_nve_config(config: dict):
    embed_config = nve.EmbedLayerConfig()
    if config is None:
        return embed_config
    if "kernel_mode" in config:
        embed_config.kernel_mode = config["kernel_mode"]
    if "logging_interval" in config:
        embed_config.logging_interval = config["logging_interval"]
    return embed_config

class CacheType(Enum):
    """Enum class defining the different types of caching strategies available.

    Attributes:
        None: No caching is used - embeddings are stored directly in GPU memory
        LinearUVM: Linear UVM caching - embeddings are stored in CPU memory with a GPU cache
        Hierarchical: Hierarchical caching - embeddings are stored remotely with a GPU cache
    """
    NoCache = 1
    LinearUVM = 2  
    Hierarchical = 3

# class to manage ids for NVEmbedding and NVEmbeddingBag
# the class will try to assign a unique id to each instance if not provided
# if provided the id will be used as is
class _NVENameManager:
    _id_dict = []
    _instance_counter = 0

    @staticmethod
    def get_id(id : Optional[int] = None) -> int:
        if id is not None:
            if id not in _NVENameManager._id_dict:
                _NVENameManager._id_dict.append(id)
            return id
        else:
            ret = _NVENameManager._instance_counter
            _NVENameManager._instance_counter += 1
            while ret in _NVENameManager._id_dict:
                ret = _NVENameManager._instance_counter
                _NVENameManager._instance_counter += 1
            _NVENameManager._id_dict.append(ret)
            return ret
        
class NVEmbeddingBase(torch.nn.Module):
    """Base class for NVEmbedding layers.

    This class serves as the foundation for embedding layers that support different caching strategies.
    It provides a wrapper from the NVEmbedding python api to the torch.nn.Embedding class.

    Note:
        1. Users are not expected to create a NVEmbeddingBase directly, 
        instead they should use one of the subclasses see NVEmbedding and NVEmbeddingBag.

        2. the attribute weight has a life time tied to the NVEmbeddingBase i.e any user that holds a reference to the tensor held in 
        NVEmbeddingBase.weight cannot use after the destruction of the NVEmbeddingBase.
        
    Args:
        num_embeddings (int): Size of the embedding dictionary
        embedding_size (int): Size of each embedding vector
        data_type (torch.dtype): Data type of embedding vectors (float32 or float16)
        cache_type (CacheType): Type of caching strategy to use, see CacheType enum for more details
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Required for LinearUVM and Hierarchical. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        remote_interface (Optional[nve.Table | nve_ps.NVLocalParameterServer], optional): Interface for remote storage. Required for Hierarchical. Defaults to None.
        weight_init (Optional[torch.Tensor], optional): Initial values for embeddings. Not supported with Hierarchical. Defaults to None.
        memblock (Optional[nve.MemBlock], optional): Memblock for LinearUVM. Defaults to None.
        optimize_for_training (bool, optional): Whether to optimize caching for training vs inference. Defaults to True.
        device (torch.device, optional): Device to use for the embedding layer. Defaults to 'cuda', note gpu cache currently unable to move between devices
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_size: int, 
                 data_type: torch.dtype, 
                 cache_type: CacheType, * ,  
                 gpu_cache_size: int = 0, 
                 host_cache_size: int = 0,
                 remote_interface: Optional[nve.Table | nve_ps.NVLocalParameterServer] = None, 
                 weight_init: Optional[torch.Tensor] = None, 
                 memblock: Optional[nve.MemBlock] = None,
                 optimize_for_training: bool = True, 
                 device : Optional[torch.device] = None, 
                 id : Optional[int] = None,
                 config: Optional[dict] = None):
        
        super().__init__()

        self.config = config_to_nve_config(config)
        self.cache_type = cache_type
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.data_type = data_type
        self.layer_data_type = torch_type_to_nve_type(data_type)
        self.save_stream = None
        self.gpu_cache_size = gpu_cache_size
        self.optimize_for_training = optimize_for_training
        # set up device shananigans
        self.device = device if device is not None else torch.device("cuda")
        if self.device.type != 'cuda':
            raise ValueError("NV Embedding only supports cuda devices")
        self.device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self.id = _NVENameManager.get_id(id)
        
        if cache_type == CacheType.NoCache:
            if (weight_init != None):
                tensor_storage = weight_init.detach().clone().to(self.device)
            else:
                tensor_storage = torch.empty(num_embeddings, embedding_size, dtype=data_type, device=self.device)
            self.emb_layer = nve.GPUEmbedding(embedding_size, num_embeddings, self.layer_data_type, tensor_storage.data_ptr(), self.device_index, self.config)
        elif cache_type == CacheType.LinearUVM:
            if gpu_cache_size == 0:
                raise ValueError("GPU cache size > 0 is required for UVM embedding")
            if memblock is None:
                memblock = nve.ManagedMemBlock(embedding_size, num_embeddings, self.layer_data_type, [self.device_index])
            self.emb_layer = nve.LinearUVMEmbedding(embedding_size, num_embeddings, self.layer_data_type, memblock, gpu_cache_size, optimize_for_training, self.device_index, self.config)
            tensor_storage = torch.utils.dlpack.from_dlpack(self.emb_layer.get_dl_tensor(self.layer_data_type))
            if (weight_init != None):
                tensor_storage.copy_(weight_init)
        elif cache_type == CacheType.Hierarchical:
            if remote_interface == None:
                raise ValueError("Remote interface is required for hierarchical embedding")
            if gpu_cache_size == 0:
                raise ValueError("GPU cache size > 0 is required for hierarchical embedding")
            if weight_init != None:
                raise ValueError("Weight init is not supported for hierarchical embedding")
            if isinstance(remote_interface, nve_ps.NVLocalParameterServer):
                remote_interface = remote_interface.local_parameter_server
            self.emb_layer = nve.HierarchicalEmbedding(embedding_size, self.layer_data_type, gpu_cache_size, host_cache_size, remote_interface, optimize_for_training, self.device_index, self.config)
            tensor_storage = torch.sparse_coo_tensor(size=(num_embeddings, embedding_size), dtype=data_type, device=self.device)
        self.weight = CachedTable(tensor_storage, cache=self.emb_layer)
    
    def to(self, *args, **kwargs):
        # since the weights need to remain on cpu in case of non dummy weight need to hijack this function 
        # and keep weights on the cpu
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        # Move all parameters except weight to the specified device
        for name, param in self.named_parameters():                
            if name != 'weight':
                param.data = param.data.to(device, dtype if dtype else param.dtype, non_blocking)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device, dtype if dtype else param.dtype, non_blocking)

        # Move all buffers to the specified device
        for name, buf in self.named_buffers():
            buf.data = buf.data.to(device, dtype if dtype else buf.dtype, non_blocking)
    

    def update(self, keys : torch.Tensor, updates : torch.Tensor):
        """Update the embedding table with new values.

        This method updates the embedding table with new values for specified keys.
        It uses the current CUDA stream to ensure proper synchronization.
        The update is asynchronous and might not be visible to the host immediately.
        Subsequent calls to forward() might not reflect the updates immediately.

        Args:
            keys (torch.Tensor): Tensor of integer keys identifying the embedding vectors to update
            updates (torch.Tensor): Tensor containing the new embedding vector values
        """
        self.emb_layer.update(torch.numel(keys), keys.data_ptr(), updates.data_ptr(), torch.cuda.current_stream(self.device).cuda_stream)

    def insert(self, keys : torch.Tensor, values : torch.Tensor, table_id : int):
        """Insert values to a table in the embedding layer.

        This method inserts new keys to a table in the the embedding layer.
        Keys are not guaranteed to be inserted (best effort).
        It uses the current CUDA stream to ensure proper synchronization.
        The insert is asynchronous and might not be visible to the host immediately.

        Args:
            keys (torch.Tensor): Tensor of integer keys identifying the embedding vectors to insert
            values (torch.Tensor): Tensor containing the new embedding vector values
            table_id (int) : Index of the table in the layer to insert to
        """
        self.emb_layer.insert(torch.numel(keys), keys.data_ptr(), values.data_ptr(), table_id, torch.cuda.current_stream(self.device).cuda_stream)

    def clear(self):
        self.emb_layer.clear(torch.cuda.current_stream(self.device).cuda_stream)

    def load_from_stream(self, stream):
        self.emb_layer.load_tensor_from_stream(stream, self.id)

    def set_save_stream(self, stream):
        self.save_stream = stream

class NVEmbedding(NVEmbeddingBase):
    """A Wrapper similar to torch.nn.Embedding that supports different caching strategies.

    This layer performs embedding lookups with caching support. It inherits from NVEmbeddingBase
    which provides the core caching functionality.
    See torch.nn.Embedding for more information on the operation.

    Args:
        num_embeddings (int): Size of the embedding dictionary
        embedding_size (int): Size of each embedding vector
        data_type (torch.dtype): Data type of the embedding weights
        cache_type (CacheType): Type of caching strategy to use
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        memblock (Optional[nve.MemBlock]): Memblock for LinearUVM. Defaults to None.
        remote_interface (Optional[nve.Table | nve_ps.NVLocalParameterServer]): Interface for remote storage. Required for hierarchical embedding. Defaults to None.
        weight_init (Optional[torch.Tensor]): Initial values for embedding weights. Not supported for hierarchical embedding. Defaults to None.
        optimize_for_training (bool): Whether to optimize caching for training vs inference. Defaults to True.
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """

    def __init__(self, 
                    num_embeddings: int, 
                    embedding_size: int, 
                    data_type: torch.dtype, 
                    cache_type: CacheType, 
                    * ,  
                    gpu_cache_size: int = 0, 
                    host_cache_size: int = 0,
                    memblock: Optional[nve.MemBlock] = None, 
                    remote_interface: Optional[nve.Table | nve_ps.NVLocalParameterServer] = None, 
                    weight_init: Optional[torch.Tensor] = None, 
                    optimize_for_training: bool = True, 
                    device : Optional[torch.device] = None, 
                    id : Optional[int] = None,
                    config: Optional[dict] = None):
        super().__init__(num_embeddings, embedding_size, data_type, cache_type, gpu_cache_size=gpu_cache_size, host_cache_size=host_cache_size, memblock=memblock, remote_interface=remote_interface, weight_init=weight_init, optimize_for_training=optimize_for_training, device=device, id=id, config=config)

    def forward(self, keys : torch.Tensor):
        """Performs embedding lookup for the input keys.

        Args:
            keys (torch.Tensor): Input indices to lookup in the embedding table

        Returns:
            torch.Tensor: Embedding vectors for the input keys
        """
        return nve_ops.CacheEmbeddingOp.apply(keys, self.weight)

    def __reduce_ex__(self, protocol):
        if self.save_stream is not None and self.cache_type == CacheType.LinearUVM:
            self.emb_layer.write_tensor_to_stream(self.save_stream, self.id)
            return (NVEmbedding._custom_builder, (self.num_embeddings, self.embedding_size, self.data_type, self.cache_type, {"gpu_cache_size": self.gpu_cache_size, "optimize_for_training": self.optimize_for_training, "device": self.device, "id": self.id}), None )
        else:
            return (NVEmbedding._custom_builder, (self.num_embeddings, self.embedding_size, self.data_type, self.cache_type, {"gpu_cache_size": self.gpu_cache_size, "optimize_for_training": self.optimize_for_training, "device": self.device, "weight_init": self.weight, "id": self.id}), None )

    def _custom_builder(num_embeddings, embedding_size, data_type, cache_type, kwargs):
            return NVEmbedding(num_embeddings, embedding_size, data_type, cache_type, **kwargs)

class NVEmbeddingBag(NVEmbeddingBase):
    """A Wrapper similar to torch.nn.EmbeddingBag that supports different caching strategies.

    This layer performs embedding bag operations with caching support. It inherits from NVEmbeddingBase
    which provides the core caching functionality. Embedding bag operations combine multiple embeddings
    into a single output vector using operations like sum, mean or concatenation.
    See torch.nn.EmbeddingBag for more information on the operation.

    Args:
        num_embeddings (int): Size of the embedding dictionary
        embedding_size (int): Size of each embedding vector
        data_type (torch.dtype): Data type of the embedding weights
        cache_type (CacheType): Type of caching strategy to use
        mode (str): The operation to use for combining embeddings ('sum', 'mean', 'max', or 'concat')
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        memblock (Optional[nve.MemBlock]): Memblock for LinearUVM. Defaults to None.
        remote_interface (Optional[nve.Table | nve_ps.NVLocalParameterServer]): Interface for remote storage. Required for hierarchical embedding. Defaults to None.
        weight_init (Optional[torch.Tensor]): Initial values for embedding weights. Not supported for hierarchical embedding. Defaults to None.
        optimize_for_training (bool): Whether to optimize caching for training vs inference. Defaults to True.
        device (torch.device, optional): Device to use for the embedding layer. Defaults to 'cuda', note gpu cache currently unable to move between devices
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_size: int, 
                 data_type: torch.dtype, 
                 cache_type: CacheType, 
                 mode : str, * ,  
                 gpu_cache_size: int = 0, 
                 host_cache_size: int = 0, 
                 remote_interface: Optional[nve.Table | nve_ps.NVLocalParameterServer] = None, 
                 memblock: Optional[nve.MemBlock] = None, 
                 weight_init: Optional[torch.Tensor] = None, 
                 optimize_for_training: bool = True, 
                 device : Optional[torch.device] = None, 
                 id : Optional[int] = None):
        super().__init__(num_embeddings, embedding_size, data_type, cache_type, gpu_cache_size=gpu_cache_size, host_cache_size=host_cache_size, remote_interface=remote_interface, memblock=memblock, weight_init=weight_init, optimize_for_training=optimize_for_training, device=device, id=id)
        self.mode = mode

    def forward(self, input: torch.Tensor, offsets: torch.Tensor, per_sample_weights: Optional[torch.Tensor] = None):
        """Performs embedding bag lookup and reduction for the input indices.

        Args:
            input (torch.Tensor): Input indices to lookup in the embedding table
            offsets (torch.Tensor): Offsets into input tensor defining the boundaries of sequences
            per_sample_weights (Optional[torch.Tensor], optional): Weights for each embedding entry. Defaults to None.

        Returns:
            torch.Tensor: Reduced embedding vectors for each sequence. For 'concat' mode, returns
                         concatenated embeddings. For other modes ('sum', 'mean', 'max'), returns
                         the reduced embeddings according to the specified mode.
        """
        if (self.mode == "concat"):
            return nve_ops.CacheEmbeddingOp.apply(input, self.weight)
        else:
            return nve_ops.CacheEmbeddingBagOp.apply(input, self.weight, offsets, self.mode, per_sample_weights)

    def __reduce_ex__(self, protocol):
        if self.save_stream is not None and self.cache_type == CacheType.LinearUVM:
            self.emb_layer.write_tensor_to_stream(self.save_stream, self.id)
            return (NVEmbeddingBag._custom_builder, (self.num_embeddings, self.embedding_size, self.data_type, self.cache_type, self.mode, {"gpu_cache_size": self.gpu_cache_size, "optimize_for_training": self.optimize_for_training, "device": self.device, "id": self.id}), None )
        else:
            return (NVEmbeddingBag._custom_builder, (self.num_embeddings, self.embedding_size, self.data_type, self.cache_type, self.mode, {"gpu_cache_size": self.gpu_cache_size, "optimize_for_training": self.optimize_for_training, "device": self.device, "weight_init": self.weight, "id": self.id}), None )

    def _custom_builder(num_embeddings, embedding_size, data_type, cache_type, mode, kwargs):
            return NVEmbeddingBag(num_embeddings, embedding_size, data_type, cache_type, mode, **kwargs)
