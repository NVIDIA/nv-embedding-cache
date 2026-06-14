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
from pynve.torch import HAS_TORCH_OPS
from pynve.torch.nve_tensors import CachedTable
import pynve.nve as nve
import pynve.torch.nve_ps as nve_ps
import torch
from enum import Enum
from typing import Optional
import builtins

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

def _current_stream_handle(device: torch.device) -> int:
    """Return the CUDA stream handle to pass into the C++ binding, or 0 on CPU.

    NVE's binding identifies execution contexts by stream pointer; on the host-only
    path (device.type == 'cpu') we use 0 as the sentinel — the underlying layer
    never touches the CUDA runtime in that case.
    """
    if device.type == 'cpu':
        return 0
    return torch.cuda.current_stream(device).cuda_stream

def config_to_nve_config(config: dict):
    embed_config = nve.EmbedLayerConfig()
    if config is None:
        return embed_config
    if "kernel_mode" in config:
        embed_config.kernel_mode = config["kernel_mode"]
    if "logging_interval" in config:
        embed_config.logging_interval = config["logging_interval"]
    if "kernel_mode_value_1" in config:
        embed_config.kernel_mode_value_1 = config["kernel_mode_value_1"]
    if "kernel_mode_value_2" in config:
        embed_config.kernel_mode_value_2 = config["kernel_mode_value_2"]
    if "max_modify_size" in config:
        embed_config.max_modify_size = config["max_modify_size"]
    return embed_config

class LayerType(Enum):
    """Enum class defining the different layer implementations available.

    Attributes:
        GPULayer: Embeddings stored directly in GPU memory (no cache).
        LinearUVM: Embeddings stored in UVM/host memory with a GPU cache.
        Hierarchical: GPU cache + optional host cache + remote parameter server.
        HostLayer: Embeddings stored on the CPU host via LinearHostTable (no GPU cache).
    """
    GPULayer = 1
    LinearUVM = 2
    Hierarchical = 3
    HostLayer = 4

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
        layer_type (LayerType): Layer implementation to use, see LayerType enum for details.
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Required for LinearUVM and Hierarchical. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        storage (Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer]):
            Backing storage for the layer.
            - GPULayer / LinearUVM / HostLayer: a ``nve.MemBlock`` (optional; one
              is allocated if omitted — HostLayer auto-allocates a pinned host
              tensor, GPULayer a CUDA tensor, LinearUVM a ManagedMemBlock).
            - Hierarchical: required ``nve.Table`` or ``nve_ps.NVEParameterServer``.
        weight_init (Optional[torch.Tensor], optional): Initial values for embeddings. Not supported with Hierarchical. Defaults to None.
        optimize_for_training (bool, optional): Whether to optimize caching for training vs inference. Defaults to True.
        device (torch.device, optional): Device to use for the embedding layer. Defaults to 'cuda', note gpu cache currently unable to move between devices
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_size: int,
                 data_type: torch.dtype,
                 layer_type: LayerType, * ,
                 gpu_cache_size: int = 0,
                 host_cache_size: int = 0,
                 storage: Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer] = None,
                 weight_init: Optional[torch.Tensor] = None,
                 optimize_for_training: bool = True,
                 device : Optional[torch.device] = None,
                 id : Optional[int] = None,
                 config: Optional[dict] = None):
        super().__init__()

        self.config = config_to_nve_config(config)
        self.layer_type = layer_type
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.data_type = data_type
        self.layer_data_type = torch_type_to_nve_type(data_type)
        self.gpu_cache_size = gpu_cache_size
        self.host_cache_size = host_cache_size
        self.optimize_for_training = optimize_for_training
        # set up device shananigans
        self.device = device if device is not None else torch.device("cuda")
        if self.device.type == 'cpu':
            if layer_type != LayerType.HostLayer:
                raise ValueError(
                    f"device='cpu' is only supported with LayerType.HostLayer, got layer_type={layer_type}")
            # -1 is the binding-level sentinel for "no GPU".
            self.device_index = -1
        elif self.device.type == 'cuda':
            self.device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        else:
            raise ValueError(f"NV Embedding only supports cuda or cpu devices, got {self.device.type}")
        self.id = id if id is not None else builtins.id(self)
        self.table_ids = {}

        if layer_type == LayerType.GPULayer:
            if storage is not None and not isinstance(storage, nve.MemBlock):
                raise ValueError("GPULayer requires storage to be a MemBlock or None")
            if (weight_init != None):
                tensor_storage = weight_init.detach().clone().to(self.device)
            else:
                tensor_storage = torch.empty(num_embeddings, embedding_size, dtype=data_type, device=self.device)
            memblock = storage if storage is not None else nve.UserMemBlock(tensor_storage.data_ptr())
            self.memblock_type = memblock.get_type()
            self.storage = memblock
            self.emb_layer = nve.GPUEmbedding(embedding_size, num_embeddings, self.layer_data_type, memblock, self.device_index, self.config)
            self.table_ids = {0: 'gpu'}
        elif layer_type == LayerType.LinearUVM:
            if gpu_cache_size == 0:
                raise ValueError("GPU cache size > 0 is required for UVM embedding")
            if storage is not None and not isinstance(storage, nve.MemBlock):
                raise ValueError("LinearUVM requires storage to be a MemBlock or None")
            memblock = storage if storage is not None else nve.ManagedMemBlock(embedding_size, num_embeddings, self.layer_data_type, [self.device_index])
            self.memblock_type = memblock.get_type()
            self.storage = memblock
            self.emb_layer = nve.LinearUVMEmbedding(embedding_size, num_embeddings, self.layer_data_type, memblock, gpu_cache_size, optimize_for_training, self.device_index, self.config)
            tensor_storage = torch.utils.dlpack.from_dlpack(nve.get_dl_tensor(self.emb_layer, self.layer_data_type))
            if (weight_init != None):
                tensor_storage.copy_(weight_init)
            self.table_ids = {0: 'gpu_cache'}
        elif layer_type == LayerType.Hierarchical:
            if storage is None:
                raise ValueError("Hierarchical requires storage to be a Table or NVEParameterServer")
            if not isinstance(storage, (nve.Table, nve_ps.NVEParameterServer)):
                raise ValueError("Hierarchical requires storage to be a Table or NVEParameterServer")
            if gpu_cache_size == 0:
                raise ValueError("GPU cache size > 0 is required for hierarchical embedding")
            if weight_init != None:
                raise ValueError("Weight init is not supported for hierarchical embedding")
            self.storage = storage
            if isinstance(storage, nve_ps.NVEParameterServer):
                if storage.data_type != data_type:
                    raise ValueError(
                        f"NVEmbedding(Hierarchical): storage.data_type="
                        f"{storage.data_type} doesn't match layer "
                        f"data_type={data_type} — they must match")
                remote_interface = storage.parameter_server
            else:
                remote_interface = storage
            self.emb_layer = nve.HierarchicalEmbedding(embedding_size, self.layer_data_type, gpu_cache_size, host_cache_size, remote_interface, num_embeddings, optimize_for_training, self.device_index, self.config)
            tensor_storage = torch.sparse_coo_tensor(size=(num_embeddings, embedding_size), dtype=data_type, device=self.device)
            self.table_ids = {0: 'gpu_cache'}
            self.table_ids |= {1: 'host_cache', 2: 'remote_ps'} if host_cache_size > 0 else {1: 'remote_ps'}
        elif layer_type == LayerType.HostLayer:
            # HostLayer is inference-only: gradient computation requires a CUDA device
            # and the C++ binding rejects backprop on a host layer. Fail fast here rather
            # than at backward-pass time (forward() would route through the training op).
            if optimize_for_training:
                raise ValueError(
                    "HostLayer is inference-only and does not support training. "
                    "Pass optimize_for_training=False when using LayerType.HostLayer.")
            if storage is not None and not isinstance(storage, nve.MemBlock):
                raise ValueError("HostLayer requires storage to be a MemBlock or None")
            if storage is not None and storage.get_type() in (nve.MemBlockType.NVL, nve.MemBlockType.MPI):
                # LinearHostTable's CPU gather/update require host accessible memblock
                raise ValueError(
                    f"HostLayer requires a host-accessible memblock; "
                    f"got memblock_type={storage.get_type()}. Use HostMemBlock, "
                    f"ManagedMemBlock, UserMemBlock around a pinned tensor, or a "
                    f"host-side LinearMemBlock (device_id=-1).")
            if storage is None:
                # pin_memory() requires a CUDA driver — skip it for cpu-device HostLayer.
                pin = self.device.type != 'cpu'
                if weight_init is not None:
                    tensor_storage = weight_init.detach().to(device="cpu", dtype=data_type).contiguous()
                    if pin:
                        tensor_storage = tensor_storage.pin_memory()
                else:
                    tensor_storage = torch.empty(num_embeddings, embedding_size, dtype=data_type)
                    if pin:
                        tensor_storage = tensor_storage.pin_memory()
                memblock = nve.UserMemBlock(tensor_storage.data_ptr())
            else:
                #user gave both weight_init and memblock (we should consider removing this path)
                memblock = storage
                if weight_init is not None:
                    src = weight_init.detach().to(device="cpu", dtype=data_type).contiguous()
                    size_bytes = num_embeddings * embedding_size * src.element_size()
                    nve.raw_copy(memblock.get_handle(), src.data_ptr(), size_bytes)
                tensor_storage = torch.sparse_coo_tensor(size=(num_embeddings, embedding_size), dtype=data_type, device=self.device)
            self.memblock_type = memblock.get_type()
            self.storage = memblock
            self.emb_layer = nve.HostEmbedding(embedding_size, num_embeddings, self.layer_data_type, memblock, self.device_index, self.config)
            self.table_ids = {0: 'host'}
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")
        self.weight = CachedTable(tensor_storage, cache=self.emb_layer)
        # Per-layer identity marker. Its data_ptr() is the registry key the
        # custom op dispatches on at runtime (globally unique across host + all
        # CUDA devices under UVA), so multiple models/instances coexist in one
        # process. persistent=False keeps it out of state_dict; torch.export
        # still captures it as a graph constant threaded into every op call.
        # Value = self.id for per-layer distinctness (avoids AOTI constant
        # dedup) and debuggability.
        self.register_buffer(
            "marker_tensor",
            torch.tensor([self.id], dtype=torch.int64, device=self.device),
            persistent=False)
        # dtype op-arg (int value of the nve DataType_t enum) — baked into the
        # op so the meta/fake impl can infer the output dtype without a registry
        # lookup.
        self.dtype_tag = int(self.layer_data_type)
        if HAS_TORCH_OPS:
            nve.register_for_torch(self.marker_tensor.data_ptr(), self.emb_layer)

    def __del__(self):
        # marker_tensor may not exist if __init__ raised before assigning it
        # (e.g. an invalid device/layer_type/storage combination). Guard so
        # __del__ stays quiet.
        if HAS_TORCH_OPS and hasattr(self, "marker_tensor"):
            nve.unregister_from_torch(self.marker_tensor.data_ptr())

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

        # Move all buffers to the specified device, except the identity marker:
        # its data_ptr() is the live registry key, so it must not be relocated.
        for name, buf in self.named_buffers():
            if name == 'marker_tensor':
                continue
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
        self.emb_layer.update(torch.numel(keys), keys.data_ptr(), updates.data_ptr(), _current_stream_handle(self.device))

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
        self.emb_layer.insert(torch.numel(keys), keys.data_ptr(), values.data_ptr(), table_id, _current_stream_handle(self.device))

    def clear(self):
        self.emb_layer.clear(_current_stream_handle(self.device))

    def erase(self, keys : torch.Tensor, table_id : int):
        """Erase keys from a table in the embedding layer.

        Keys not resident in the table will be ignored.
        It uses the current CUDA stream to ensure proper synchronization.
        The erase is asynchronous and might not be visible to the host immediately.

        Args:
            keys (torch.Tensor): Tensor of integer keys to erase
            table_id (int): Index of the table in the layer to erase from
        """
        self.emb_layer.erase(torch.numel(keys), keys.data_ptr(), table_id, _current_stream_handle(self.device))

    def load_from_stream(self, stream):
        nve.load_tensor_from_stream(self.emb_layer, stream, self.id)

class NVEmbedding(NVEmbeddingBase):
    """A Wrapper similar to torch.nn.Embedding that supports different layer implementations.

    This layer performs embedding lookups with optional caching. It inherits from NVEmbeddingBase
    which provides the core functionality.
    See torch.nn.Embedding for more information on the operation.

    Args:
        num_embeddings (int): Size of the embedding dictionary
        embedding_size (int): Size of each embedding vector
        data_type (torch.dtype): Data type of the embedding weights
        layer_type (LayerType): Layer implementation to use
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        storage (Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer]):
            Backing storage. MemBlock for GPULayer/LinearUVM/HostLayer; Table or
            NVEParameterServer for Hierarchical. Defaults to None.
        weight_init (Optional[torch.Tensor]): Initial values for embedding weights. Not supported for Hierarchical. Defaults to None.
        optimize_for_training (bool): Whether to optimize caching for training vs inference. Defaults to True.
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """

    def __init__(self,
                    num_embeddings: int,
                    embedding_size: int,
                    data_type: torch.dtype,
                    layer_type: LayerType,
                    * ,
                    gpu_cache_size: int = 0,
                    host_cache_size: int = 0,
                    storage: Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer] = None,
                    weight_init: Optional[torch.Tensor] = None,
                    optimize_for_training: bool = True,
                    device : Optional[torch.device] = None,
                    id : Optional[int] = None,
                    config: Optional[dict] = None):
        super().__init__(num_embeddings, embedding_size, data_type, layer_type, gpu_cache_size=gpu_cache_size, host_cache_size=host_cache_size, storage=storage, weight_init=weight_init, optimize_for_training=optimize_for_training, device=device, id=id, config=config)

    def forward(self, keys : torch.Tensor):
        """Performs embedding lookup for the input keys.

        Args:
            keys (torch.Tensor): Input indices to lookup in the embedding table

        Returns:
            torch.Tensor: Embedding vectors for the input keys
        """
        if HAS_TORCH_OPS:
            if self.optimize_for_training:
                return nve_ops.NVEmbeddingOpTraining.apply(
                    self.marker_tensor, keys, self.weight,
                    self.embedding_size, self.dtype_tag)
            else:
                return nve_ops.NVEmbeddingOp.apply(
                    self.marker_tensor, keys,
                    self.embedding_size, self.dtype_tag)
        else:
            return nve_ops.CacheEmbeddingOp.apply(keys, self.weight)

class NVEmbeddingBag(NVEmbeddingBase):
    """A Wrapper similar to torch.nn.EmbeddingBag that supports different layer implementations.

    This layer performs embedding bag operations with optional caching. It inherits from NVEmbeddingBase
    which provides the core functionality. Embedding bag operations combine multiple embeddings
    into a single output vector using operations like sum, mean or concatenation.
    See torch.nn.EmbeddingBag for more information on the operation.

    Args:
        num_embeddings (int): Size of the embedding dictionary
        embedding_size (int): Size of each embedding vector
        data_type (torch.dtype): Data type of the embedding weights
        layer_type (LayerType): Layer implementation to use
        mode (str): The operation to use for combining embeddings ('sum', 'mean', 'max', or 'concat')
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        storage (Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer]):
            Backing storage. MemBlock for GPULayer/LinearUVM; Table or
            NVEParameterServer for Hierarchical. Defaults to None.
        weight_init (Optional[torch.Tensor]): Initial values for embedding weights. Not supported for Hierarchical. Defaults to None.
        optimize_for_training (bool): Whether to optimize caching for training vs inference. Defaults to True.
        device (torch.device, optional): Device to use for the embedding layer. Defaults to 'cuda', note gpu cache currently unable to move between devices
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.

    Note:
        LayerType.HostLayer is not supported by NVEmbeddingBag — pooled lookups
        are not implemented for the host layer. Use NVEmbedding for HostLayer.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_size: int,
                 data_type: torch.dtype,
                 layer_type: LayerType,
                 mode : str, * ,
                 gpu_cache_size: int = 0,
                 host_cache_size: int = 0,
                 storage: Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer] = None,
                 weight_init: Optional[torch.Tensor] = None,
                 optimize_for_training: bool = True,
                 device : Optional[torch.device] = None,
                 id : Optional[int] = None,
                 config: Optional[dict] = None):
        # HostLayer has no pooled-lookup implementation (HostEmbeddingLayer::lookup
        # rejects pool_params), and the bag forward always issues a pooled lookup.
        # Fail fast at construction rather than at the first forward().
        if layer_type == LayerType.HostLayer:
            raise ValueError(
                "NVEmbeddingBag does not support LayerType.HostLayer (pooled lookups "
                "are not implemented for the host layer). Use NVEmbedding for HostLayer.")
        super().__init__(num_embeddings, embedding_size, data_type, layer_type, gpu_cache_size=gpu_cache_size, host_cache_size=host_cache_size, storage=storage, weight_init=weight_init, optimize_for_training=optimize_for_training, device=device, id=id, config=config)
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
        if not HAS_TORCH_OPS:
            if self.mode == "concat":
                return nve_ops.CacheEmbeddingOp.apply(input, self.weight)
            else:
                return nve_ops.CacheEmbeddingBagOp.apply(
                    input, self.weight, offsets, self.mode, per_sample_weights)
        elif self.mode == "concat":
            if self.optimize_for_training:
                return nve_ops.NVEmbeddingOpTraining.apply(
                    self.marker_tensor, input, self.weight,
                    self.embedding_size, self.dtype_tag)
            else:
                return nve_ops.NVEmbeddingOp.apply(
                    self.marker_tensor, input,
                    self.embedding_size, self.dtype_tag)
        else:
            if per_sample_weights is not None:
                pooling_type = int(nve.PoolingType_t.WeightedSum) if self.mode == "sum" \
                               else int(nve.PoolingType_t.WeightedMean)
            else:
                pooling_type = int(nve.PoolingType_t.Sum) if self.mode == "sum" \
                               else int(nve.PoolingType_t.Mean)
            if self.optimize_for_training:
                return nve_ops.NVEmbeddingBagOpTraining.apply(
                    self.marker_tensor, input, self.weight, offsets,
                    per_sample_weights, pooling_type,
                    self.embedding_size, self.dtype_tag)
            else:
                return nve_ops.NVEmbeddingBagOp.apply(
                    self.marker_tensor, input, offsets, per_sample_weights,
                    pooling_type, self.embedding_size, self.dtype_tag)
