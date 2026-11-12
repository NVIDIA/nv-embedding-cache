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
from pynve.torch.nve_tensors import CachedTable, _current_stream_handle
import pynve.nve as nve
import pynve.torch.nve_ps as nve_ps
import pynve.torch.nve_dtypes as nve_dtypes
# Re-exported for existing importers (nve_export, torch fakes, tests).
from pynve.torch.nve_dtypes import nve_type_to_torch_type, torch_type_to_nve_type
import torch
from enum import Enum
from typing import Optional
import builtins
import warnings

def _check_forward_inputs(keys, offsets=None, per_sample_weights=None):
    """Validate dtypes/contiguity before the raw data_ptr() handoff to the binding.

    The binding is int64-keyed and receives untyped pointers, so a wrong dtype
    or a non-contiguous view would be silently misread as garbage.
    """
    if keys.dtype != torch.int64:
        raise TypeError(f"keys must be an int64 tensor, got {keys.dtype}")
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    if offsets is not None:
        if offsets.dtype != torch.int64:
            raise TypeError(f"offsets must be an int64 tensor, got {offsets.dtype}")
        if not offsets.is_contiguous():
            raise ValueError("offsets must be contiguous")
        # num_bags + 1 entries (include_last_offset=True convention). The values
        # (start at 0, non-decreasing, trailing sentinel == len(keys)) are checked
        # at lookup time in C++ — reading them here would break export tracing.
        if offsets.numel() < 2:
            raise ValueError("offsets must have at least two entries (num_bags + 1, "
                             "include_last_offset=True convention)")
    if per_sample_weights is not None:
        if per_sample_weights.dtype not in (torch.float32, torch.float16):
            raise TypeError(f"per_sample_weights must be a float32 or float16 tensor, "
                            f"got {per_sample_weights.dtype}")
        if not per_sample_weights.is_contiguous():
            raise ValueError("per_sample_weights must be contiguous")
        if per_sample_weights.numel() != keys.numel():
            raise ValueError(f"per_sample_weights must have one element per key: "
                             f"got {per_sample_weights.numel()} weights for {keys.numel()} keys")

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
    if "default_row_index" in config:
        embed_config.default_row_index = config["default_row_index"]
    if "default_embedding" in config:
        value = config["default_embedding"]
        if isinstance(value, torch.Tensor):
            # Raw row bytes in the tensor's own dtype; the layer validates the
            # byte length against its row size at construction.
            value = value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
        # else: an iterable of byte values (e.g. the export/load json round-trip).
        embed_config.default_embedding = list(value)
    if "disable_threadpool" in config:
        embed_config.disable_threadpool = bool(config["disable_threadpool"])
    return embed_config

class LayerType(Enum):
    """Enum class defining the different layer implementations available.

    Attributes:
        GPULayer: Embeddings stored directly in GPU memory (no cache).
        LinearUVM: Embeddings stored in UVM/host memory with a GPU cache.
        Hierarchical: GPU cache + optional host cache + remote parameter server.
        HostLayer: Embeddings stored on the CPU host (no GPU cache), either in a LinearHostTable
            over a MemBlock or in a host table (e.g. NVHashMap) given as storage.
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
        data_type (torch.dtype | nve.DataType_t): Storage type of the embedding table: torch.float32,
            torch.float16, or a rowwise-quantized nve.DataType_t (QInt8RowwiseF32/F16,
            QUint8RowwiseF32/F16). Quantized layers are inference-only, return float32/float16
            (per the type's F32/F16 suffix) from forward(), and take raw stored rows
            (uint8 [rows, row_bytes]) for weight_init, update, insert and default_embedding.
            After construction, ``data_type`` is the torch dtype forward() returns and
            ``storage_data_type`` is the storage nve.DataType_t.
        layer_type (LayerType): Layer implementation to use, see LayerType enum for details.
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Required for LinearUVM and Hierarchical. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        storage (Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer]):
            Backing storage for the layer.
            - GPULayer / LinearUVM / HostLayer: a ``nve.MemBlock`` (optional; one
              is allocated if omitted — HostLayer auto-allocates a pinned host
              tensor, GPULayer a CUDA tensor, LinearUVM a ManagedMemBlock).
            - HostLayer: alternatively a host ``nve.Table`` or ``nve_ps.NVEParameterServer``
              (e.g. ps_type=nve.NVHashMap) for non-linear keys. Keys missing from the table
              return default_embedding. A bounded parameter server (its own num_embeddings > 0,
              not this layer's) may evict rows when full; create the parameter server with
              num_embeddings=0 for an unbounded table.
            - Hierarchical: required ``nve.Table`` or ``nve_ps.NVEParameterServer``.
        weight_init (Optional[torch.Tensor], optional): Initial values for embeddings ([num_embeddings, row_bytes] raw rows for quantized types). Not supported with Hierarchical or a table-backed HostLayer. Defaults to None.
        optimize_for_training (bool, optional): Whether to optimize caching for training vs inference. Defaults to True.
        device (torch.device, optional): Device to use for the embedding layer. Defaults to 'cuda', note gpu cache currently unable to move between devices
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_size: int,
                 data_type: nve_dtypes.StorageType,
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

        # storage_data_type is the source of truth; data_type is the dtype forward() returns.
        self.storage_data_type = nve_dtypes.resolve_data_type(data_type)
        self.data_type = nve_dtypes.output_torch_type(self.storage_data_type)
        self.is_quantized = nve.is_quant_rowwise(self.storage_data_type)
        self.row_bytes = nve.row_size_in_bytes(self.storage_data_type, embedding_size)
        if self.is_quantized:
            if optimize_for_training:
                raise ValueError(
                    f"Rowwise-quantized layers ({self.storage_data_type.name}) are inference-only. "
                    "Pass optimize_for_training=False.")

        default_embedding = config.get("default_embedding") if config is not None else None
        if isinstance(default_embedding, torch.Tensor):
            # Checked before config_to_nve_config flattens the tensor to raw
            # bytes, where a wrong dtype could still pass the byte-length check.
            if self.is_quantized:
                nve_dtypes.check_raw_rows(default_embedding, self.row_bytes,
                                          "config['default_embedding']", num_rows=1, exact=True)
            elif default_embedding.dtype != self.data_type:
                raise ValueError(
                    f"config['default_embedding'] dtype "
                    f"{default_embedding.dtype} does not match layer "
                    f"data_type {self.data_type}")
        self.config = config_to_nve_config(config)
        self.layer_type = layer_type
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        if len(self.config.default_embedding) not in (0, self.row_bytes):
            raise ValueError(
                f"config['default_embedding'] must hold one row ({embedding_size} "
                f"values of {self.storage_data_type.name} = {self.row_bytes} bytes), got "
                f"{len(self.config.default_embedding)} bytes")
        if weight_init is not None:
            if self.is_quantized:
                nve_dtypes.check_raw_rows(weight_init, self.row_bytes, "weight_init",
                                          num_rows=num_embeddings, exact=True)
                weight_init = weight_init.detach().view(torch.uint8).reshape(num_embeddings, self.row_bytes)
            else:
                # Float init is cast to the storage dtype (memblocks are filled bytewise).
                weight_init = weight_init.detach().to(dtype=self.data_type).contiguous()
                nve_dtypes.check_num_rows(weight_init, embedding_size, "weight_init",
                                          num_embeddings, exact=True)
        # Quantized storage is held as raw bytes; float storage as values.
        storage_shape = (num_embeddings, self.row_bytes) if self.is_quantized else (num_embeddings, embedding_size)
        storage_dtype = torch.uint8 if self.is_quantized else self.data_type
        if layer_type in (LayerType.HostLayer, LayerType.Hierarchical):
            if config is None or "default_embedding" not in config:
                # Misses return zeros unless the user picks a value; pass an explicit
                # empty list to disable the default fill (misses undefined).
                self.config.default_embedding = [0] * self.row_bytes
        elif len(self.config.default_embedding) > 0:
            raise ValueError(
                f"config['default_embedding'] is only supported for HostLayer and "
                f"Hierarchical layers, got layer_type={layer_type}")
        if self.config.disable_threadpool and layer_type != LayerType.HostLayer:
            raise ValueError(
                f"config['disable_threadpool'] is only supported for HostLayer, "
                f"got layer_type={layer_type}")
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
                tensor_storage = torch.empty(storage_shape, dtype=storage_dtype, device=self.device)
            memblock = storage if storage is not None else nve.UserMemBlock(tensor_storage.data_ptr(), tensor_storage.nbytes)
            self.memblock_type = memblock.get_type()
            self.storage = memblock
            self.emb_layer = nve.GPUEmbedding(embedding_size, num_embeddings, self.storage_data_type, memblock, self.device_index, self.config)
            self.table_ids = {0: 'gpu'}
        elif layer_type == LayerType.LinearUVM:
            if gpu_cache_size == 0:
                raise ValueError("GPU cache size > 0 is required for UVM embedding")
            if storage is not None and not isinstance(storage, nve.MemBlock):
                raise ValueError("LinearUVM requires storage to be a MemBlock or None")
            memblock = storage if storage is not None else nve.ManagedMemBlock(embedding_size, num_embeddings, self.storage_data_type, [self.device_index])
            self.memblock_type = memblock.get_type()
            self.storage = memblock
            self.emb_layer = nve.LinearUVMEmbedding(embedding_size, num_embeddings, self.storage_data_type, memblock, gpu_cache_size, optimize_for_training, self.device_index, self.config)
            tensor_storage = torch.utils.dlpack.from_dlpack(nve.get_dl_tensor(self.emb_layer, self.storage_data_type))
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
            remote_interface = self._resolve_table_storage(storage)
            self.emb_layer = nve.HierarchicalEmbedding(embedding_size, self.storage_data_type, gpu_cache_size, host_cache_size, remote_interface, num_embeddings, optimize_for_training, self.device_index, self.config)
            tensor_storage = torch.sparse_coo_tensor(size=storage_shape, dtype=storage_dtype, device=self.device)
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
            if storage is not None and not isinstance(storage, (nve.MemBlock, nve.Table, nve_ps.NVEParameterServer)):
                raise ValueError("HostLayer requires storage to be a MemBlock, Table, NVEParameterServer or None")
            if isinstance(storage, (nve.Table, nve_ps.NVEParameterServer)):
                # weight_init holds rows for linear keys [0, num_embeddings); hash tables hold arbitrary keys.
                if weight_init is not None:
                    raise ValueError(
                        "weight_init is not supported for a table-backed HostLayer; "
                        "populate the table instead (e.g. NVEParameterServer(initializer=...))")
                self.storage = storage
                self.emb_layer = nve.HostEmbedding(embedding_size, self.storage_data_type,
                                                   self._resolve_table_storage(storage),
                                                   self.device_index, self.config)
                tensor_storage = torch.sparse_coo_tensor(size=storage_shape, dtype=storage_dtype, device=self.device)
                self.table_ids = {0: 'host'}
            else:
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
                        tensor_storage = weight_init.detach().to(device="cpu", dtype=storage_dtype).contiguous()
                        if pin:
                            tensor_storage = tensor_storage.pin_memory()
                    else:
                        tensor_storage = torch.empty(storage_shape, dtype=storage_dtype)
                        if pin:
                            tensor_storage = tensor_storage.pin_memory()
                    memblock = nve.UserMemBlock(tensor_storage.data_ptr(), tensor_storage.nbytes)
                else:
                    #user gave both weight_init and memblock (we should consider removing this path)
                    memblock = storage
                    if weight_init is not None:
                        src = weight_init.detach().to(device="cpu", dtype=storage_dtype).contiguous()
                        # view as uint8: numpy has no bfloat16, and copy_from is a byte copy
                        memblock.copy_from(src.view(torch.uint8).numpy())
                    tensor_storage = torch.sparse_coo_tensor(size=storage_shape, dtype=storage_dtype, device=self.device)
                self.memblock_type = memblock.get_type()
                self.storage = memblock
                self.emb_layer = nve.HostEmbedding(embedding_size, num_embeddings, self.storage_data_type, memblock, self.device_index, self.config)
                self.table_ids = {0: 'host'}
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")
        # Packed quantized bytes cannot carry gradients (and integer Parameters must not require them).
        self.weight = CachedTable(tensor_storage, requires_grad=not self.is_quantized, cache=self.emb_layer)
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
        # dtype op-arg (int value of the output nve DataType_t enum) — baked into
        # the op so the meta/fake impl can infer the output dtype without a
        # registry lookup.
        self.dtype_tag = int(nve_dtypes.output_nve_type(self.storage_data_type))
        if HAS_TORCH_OPS:
            nve.register_for_torch(self.marker_tensor.data_ptr(), self.emb_layer)

    @property
    def layer_data_type(self) -> nve.DataType_t:
        """Deprecated alias of storage_data_type."""
        warnings.warn("layer_data_type is deprecated, use storage_data_type",
                      DeprecationWarning, stacklevel=2)
        return self.storage_data_type

    def __del__(self):
        # marker_tensor may not exist if __init__ raised before assigning it
        # (e.g. an invalid device/layer_type/storage combination). Guard so
        # __del__ stays quiet.
        if HAS_TORCH_OPS and hasattr(self, "marker_tensor"):
            nve.unregister_from_torch(self.marker_tensor.data_ptr())

    def _resolve_table_storage(self, storage) -> nve.Table:
        """Return the nve.Table behind table storage, checking an NVEParameterServer's data type."""
        # Raw tables are checked by the binding (value type and row bytes).
        if not isinstance(storage, nve_ps.NVEParameterServer):
            return storage
        if storage.storage_data_type != self.storage_data_type:
            raise ValueError(
                f"NVEmbedding({self.layer_type.name}): storage data type {storage.storage_data_type.name} "
                f"doesn't match layer data type {self.storage_data_type.name} — they must match")
        return storage.parameter_server

    def _check_rows(self, num_keys: int, values: torch.Tensor, what: str):
        """Validate row data before its raw data_ptr() is handed to the binding."""
        nve_dtypes.check_rows(values, self.storage_data_type, self.embedding_size, num_keys, what)

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
    

    def update(self, keys : torch.Tensor, updates : torch.Tensor, table_id : int = -1):
        """Update the embedding table with new values.

        This method updates the embedding table with new values for specified keys.
        It uses the current CUDA stream to ensure proper synchronization.
        The update is asynchronous and might not be visible to the host immediately.
        Subsequent calls to forward() might not reflect the updates immediately.

        Args:
            keys (torch.Tensor): Tensor of integer keys identifying the embedding vectors to update
            updates (torch.Tensor): Tensor containing the new embedding vector values, in data_type.
                For rowwise-quantized layers: raw stored rows, a uint8/int8 tensor
                [num_keys, row_bytes] (see nve_dtypes.quantize_rowwise).
            table_id (int, optional): Index of the table in the layer to update, negative implies all. Defaults to -1.
        """
        self._check_rows(torch.numel(keys), updates, "updates")
        self.emb_layer.update(torch.numel(keys), keys.data_ptr(), updates.data_ptr(), table_id, _current_stream_handle(self.device))

    def insert(self, keys : torch.Tensor, values : torch.Tensor, table_id : int):
        """Insert values to a table in the embedding layer.

        This method inserts new keys to a table in the the embedding layer.
        Keys are not guaranteed to be inserted (best effort).
        It uses the current CUDA stream to ensure proper synchronization.
        The insert is asynchronous and might not be visible to the host immediately.

        Args:
            keys (torch.Tensor): Tensor of integer keys identifying the embedding vectors to insert
            values (torch.Tensor): Tensor containing the new embedding vector values, in data_type.
                For rowwise-quantized layers: raw stored rows, a uint8/int8 tensor
                [num_keys, row_bytes] (see nve_dtypes.quantize_rowwise).
            table_id (int) : Index of the table in the layer to insert to
        """
        self._check_rows(torch.numel(keys), values, "values")
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
        data_type (torch.dtype | nve.DataType_t): Storage type of the embedding table, see NVEmbeddingBase.
        layer_type (LayerType): Layer implementation to use
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        storage (Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer]):
            Backing storage. MemBlock for GPULayer/LinearUVM/HostLayer; Table or
            NVEParameterServer for Hierarchical or a table-backed HostLayer. Defaults to None.
        weight_init (Optional[torch.Tensor]): Initial values for embedding weights. Not supported for Hierarchical or a table-backed HostLayer. Defaults to None.
        optimize_for_training (bool): Whether to optimize caching for training vs inference. Defaults to True.
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.
    """

    def __init__(self,
                    num_embeddings: int,
                    embedding_size: int,
                    data_type: nve_dtypes.StorageType,
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
        _check_forward_inputs(keys)
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
        data_type (torch.dtype | nve.DataType_t): Storage type of the embedding table, see NVEmbeddingBase.
        layer_type (LayerType): Layer implementation to use
        mode (str): The operation to use for combining embeddings ('sum', 'mean', or 'concat'). 'max' is not supported.
        gpu_cache_size (int, optional): Size of GPU cache in bytes. Defaults to 0.
        host_cache_size (int, optional): Size of host cache. Defaults to 0.
        storage (Optional[nve.MemBlock | nve.Table | nve_ps.NVEParameterServer]):
            Backing storage. MemBlock for GPULayer/LinearUVM/HostLayer; Table or
            NVEParameterServer for Hierarchical or a table-backed HostLayer. Defaults to None.
        weight_init (Optional[torch.Tensor]): Initial values for embedding weights. Not supported for Hierarchical or a table-backed HostLayer. Defaults to None.
        optimize_for_training (bool): Whether to optimize caching for training vs inference. Defaults to True.
        device (torch.device, optional): Device to use for the embedding layer. Defaults to 'cuda', note gpu cache currently unable to move between devices
        id (int, optional): Identifier for the embedding layer. Defaults to None, if None the layer will be assigned a Id automatically. Ids must be unique inside a nested model, in order for serialization to work.

    Note:
        'max' pooling (unlike torch.nn.EmbeddingBag) is not implemented.

        offsets always follow torch's include_last_offset=True convention:
        num_bags + 1 entries, starting at 0 and ending with a trailing sentinel
        equal to len(input). torch-default offsets (no sentinel) are rejected.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_size: int,
                 data_type: nve_dtypes.StorageType,
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
        # Fail fast on unsupported or mistyped modes.
        if mode != "concat":
            nve_ops.pooling_type_from_mode(mode, weighted=False)
        super().__init__(num_embeddings, embedding_size, data_type, layer_type, gpu_cache_size=gpu_cache_size, host_cache_size=host_cache_size, storage=storage, weight_init=weight_init, optimize_for_training=optimize_for_training, device=device, id=id, config=config)
        self.mode = mode

    def forward(self, input: torch.Tensor, offsets: torch.Tensor, per_sample_weights: Optional[torch.Tensor] = None):
        """Performs embedding bag lookup and reduction for the input indices.

        Args:
            input (torch.Tensor): Input indices to lookup in the embedding table
            offsets (torch.Tensor): Offsets into input tensor defining the boundaries of sequences.
                Uses torch's include_last_offset=True convention: num_bags + 1 entries, with
                offsets[0] == 0 and a trailing sentinel offsets[-1] == input.numel().
            per_sample_weights (Optional[torch.Tensor], optional): Weights for each embedding entry. Defaults to None.

        Returns:
            torch.Tensor: Reduced embedding vectors for each sequence. For 'concat' mode, returns
                         concatenated embeddings. For other modes ('sum', 'mean'), returns
                         the reduced embeddings according to the specified mode.
        """
        _check_forward_inputs(input, offsets, per_sample_weights)
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
            pooling_type = int(nve_ops.pooling_type_from_mode(
                self.mode, weighted=per_sample_weights is not None))
            if self.optimize_for_training:
                return nve_ops.NVEmbeddingBagOpTraining.apply(
                    self.marker_tensor, input, self.weight, offsets,
                    per_sample_weights, pooling_type,
                    self.embedding_size, self.dtype_tag)
            else:
                return nve_ops.NVEmbeddingBagOp.apply(
                    self.marker_tensor, input, offsets, per_sample_weights,
                    pooling_type, self.embedding_size, self.dtype_tag)
