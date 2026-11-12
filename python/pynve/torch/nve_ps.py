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
import pynve.torch.nve_dtypes as nve_dtypes
from typing import Optional
from collections.abc import Iterator
import warnings
from json import dumps, loads

class SimpleInitializer (Iterator):
    """Yields (keys, values) batches for NVEParameterServer.

    Values are random rows of data_type, or slices of init_tensor. For a rowwise-quantized
    data_type, init_tensor is required and holds raw stored rows (uint8 [num_rows, row_bytes]).
    """
    def __init__(self,
                 num_rows: int,
                 row_size: int,
                 data_type: nve_dtypes.StorageType,
                 init_tensor: torch.tensor = None,
                 init_batch: Optional[int] = 2**16):
        self.num_rows = num_rows
        storage_data_type = nve_dtypes.resolve_data_type(data_type)
        self.data_type = nve_dtypes.output_torch_type(storage_data_type)
        self.raw_rows = nve.is_quant_rowwise(storage_data_type)
        # Per-row width of init_tensor: stored bytes for quantized types, values otherwise.
        if self.raw_rows:
            if init_tensor is None:
                raise ValueError(
                    f"SimpleInitializer: {storage_data_type.name} requires init_tensor with raw stored "
                    "rows; random initialization is only supported for float types")
            self.row_size = nve.row_size_in_bytes(storage_data_type, row_size)
            nve_dtypes.check_raw_rows(init_tensor, self.row_size, "init_tensor")
            init_tensor = init_tensor.view(torch.uint8)
        else:
            self.row_size = row_size
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
        if self.raw_rows:
            values = values.view(-1, self.row_size)
        self.current_row = end_row
        return keys, values

class NVEParameterServer ():
    """Wrapper class for a parameter server (can be local or remote).

    This class is meant to be used by NVEmbedding of type LayerType.Hierarchical, or as the
    host table of a LayerType.HostLayer.
    It can be used as a secondary cache, between the GPU cache and a remote parameter server, or as a
    local key-vector storage to back the GPU cache.

    Two construction modes are supported:

    * **Enum-based:** pass ``ps_type`` (one of ``nve.NVHashMap``,
      ``nve.Abseil``, ``nve.ParallelHash``, ``nve.Redis``) and an optional
      ``extra_params`` dict. The wrapper translates this to the plugin
      mechanism internally, so exported metadata looks identical to the
      explicit-plugin path.

    * **Plugin-based:** pass ``plugin_name`` as a plugin shared object name or
      path (e.g. ``"libnve-plugin-custom_remote.so"`` or
      ``"/tmp/my_plugin.so"``), plus ``factory_config`` and ``table_config``
      dicts that configure the plugin's factory and produced table respectively.

    Args:
        num_embeddings (int): Size of the embedding dictionary, when set to 0, storage will increase until OOM
                              Use erase or clear to manually remove data.
        embedding_size (int): Size of each embedding vector (in elements)
        data_type (torch.dtype | nve.DataType_t): Storage type: torch.float32, torch.float16 or a
            rowwise-quantized nve.DataType_t. After construction ``data_type`` is the decoded torch
            dtype and ``storage_data_type`` the storage nve.DataType_t.
        initializer (Optional[Iterator]): Initializer for the data.
                                          Iterator must return a tuple of tensors, keys(int64) and values(vector with embedding_size elements of data_type per key)
        initial_size (Optional[int]): Initial amount of embeddings to allocate
        ps_type (Optional[nve.PSType_t]): Type of parameter server backend to use
        extra_params: Optional[dict]: Additional parameters in dict format. plugin params under "plugin" node, table params under "table" node.
                                      Valid parameters will depend on ps_type.
                                      E.g. when using a Redis PS, use the following to set the server address {"plugin": {"address": "localhost:12345"}}
        plugin_name (Optional[str]): Plugin shared object name/path. Mutually exclusive with ``ps_type``.
        factory_config (Optional[dict]): JSON-serializable, plugin-specific factory options; plugin selection is by ``plugin_name`` (the ``.so``).
        table_config (Optional[dict]): JSON-serializable table config passed to the plugin's ``produce()``.
        table_id (Optional[int]): Id of the produced table, plugin-based mode only (default 1000). Some plugins derive
                                  where the data is stored from it (e.g. the Redis hash names), so give tables that
                                  share a store different ids. Exports record the id and loading reuses it.
                                  The enum-based mode uses a fixed id per ``ps_type``.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_size: int,
                 data_type: nve_dtypes.StorageType,
                 initializer: Optional[Iterator] = None,
                 initial_size: Optional[int] = 1024,
                 ps_type: Optional[nve.PSType_t] = None,
                 extra_params: Optional[dict] = None,
                 plugin_name: Optional[str] = None,
                 factory_config: Optional[dict] = None,
                 table_config: Optional[dict] = None,
                 table_id: Optional[int] = None):
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        # storage_data_type is the source of truth; data_type is the decoded torch dtype.
        self.storage_data_type = nve_dtypes.resolve_data_type(data_type)
        self.data_type = nve_dtypes.output_torch_type(self.storage_data_type)
        self.row_bytes = nve.row_size_in_bytes(self.storage_data_type, embedding_size)

        if plugin_name is not None:
            if ps_type is not None or extra_params is not None:
                raise ValueError(
                    "NVEParameterServer: pass either (ps_type, extra_params) "
                    "or (plugin_name, factory_config, table_config), not both")
            factory_config = factory_config or {}
            table_config = table_config or {}
            # Leave table_id out when not given, so the binding's default applies.
            table_id_arg = {} if table_id is None else {"table_id": table_id}
            self.parameter_server = nve.ParameterServerTable(
                row_elements=embedding_size,
                data_type=self.storage_data_type,
                plugin_name=plugin_name,
                factory_config_json=dumps(factory_config),
                table_config_json=dumps(table_config),
                num_rows=num_embeddings,
                **table_id_arg,
            )
        else:
            # Enum-based path
            if factory_config is not None or table_config is not None or table_id is not None:
                raise ValueError(
                    "NVEParameterServer: factory_config / table_config / table_id require "
                    "plugin_name; pass plugin_name to use the plugin-based ctor, "
                    "or use ps_type/extra_params for the enum-based ctor")
            ps_type = ps_type if ps_type is not None else nve.NVHashMap
            extra_params = extra_params if extra_params is not None else {}
            extra_params_str = dumps(extra_params)
            self.parameter_server = nve.ParameterServerTable(
                num_rows=num_embeddings,
                row_elements=embedding_size,
                data_type=self.storage_data_type,
                initial_size=initial_size,
                ps_type=ps_type,
                extra_params=extra_params_str,
            )

        if initializer:
            while True:
                try:
                    keys,values = next(initializer)
                    self.insert(keys, values)
                except StopIteration:
                    break

    @property
    def layer_data_type(self) -> nve.DataType_t:
        """Deprecated alias of storage_data_type."""
        warnings.warn("layer_data_type is deprecated, use storage_data_type",
                      DeprecationWarning, stacklevel=2)
        return self.storage_data_type

    def insert(self, keys: torch.Tensor, values: torch.Tensor):
        """Method to insert key-value pairs to the parameter server.

        Args:
            keys (torch.Tensor): Tensor of int64 keys to insert
            values (torch.Tensor): Tensor of value vectors (containing embedding_size elements each), matching the given keys.
                For rowwise-quantized types: raw stored rows, a uint8/int8 tensor [num_keys, row_bytes].
        """
        if keys.dtype is not torch.int64:
            warnings.warn(f"Keys provided to NVEParameterServer were of type {keys.dtype} instead of int64 (converting on the fly)")
            keys = keys.long()
        nve_dtypes.check_rows(values, self.storage_data_type, self.embedding_size, torch.numel(keys), "values")
        self.parameter_server.insert_keys(torch.numel(keys), keys.data_ptr(), values.data_ptr())

    def load_from_file(self, keys_path: str, values_path: str, batch_size: Optional[int] = 1024*1024):
        """Method to load key-value pairs from a numpy file.

        Args:
            keys_path (str): Path to the keys file
            values_path (str): Path to the values file
            batch_size (int): Number of keys to load from the file at a time
        """
        
        self.parameter_server.insert_keys_from_filepath(keys_path, values_path, batch_size)

    def erase(self, keys: torch.Tensor):
        """Method to erase key-value pairs from the parameter server.

        Args:
            keys (torch.Tensor): Tensor of int64 keys to erase
        """
        if keys.dtype is not torch.int64:
            warnings.warn(f"Keys provided to NVEParameterServer were of type {keys.dtype} instead of int64 (converting on the fly)")
            keys = keys.long()
        self.parameter_server.erase_keys(torch.numel(keys), keys.data_ptr())

    def clear(self):
        """Method to clear all keys from the parameter server.
        """
        self.parameter_server.clear_keys()

    def num_keys(self) -> int:
        """Number of keys currently stored in the parameter server."""
        return self.parameter_server.num_keys()

    def keys(self) -> torch.Tensor:
        """Return the keys currently stored in the parameter server as an int64 CPU tensor (unordered).

        Keys inserted or erased concurrently may or may not be included.
        Sizing the result calls num_keys() first, so backends whose size is itself a scan
        (RocksDB, prefixed-string Redis) traverse their keyspace twice.
        """
        # get_keys only writes to host memory, so allocate on the CPU regardless of torch's default device.
        keys = torch.empty(self.num_keys(), dtype=torch.int64, device="cpu")
        written = self.parameter_server.export_keys(torch.numel(keys), keys.data_ptr())
        return keys[:written]

    def export_config(self) -> dict:
        """Return a JSON-serializable dict describing how to recreate this PS.

        Delegates to the underlying ``ParameterServerTable.export_config_json``

        Returns:
            dict with keys: ``remote_ps_type``, ``plugin_name``,
            ``factory_config``, ``table_config``, ``row_elements``,
            ``num_rows``, ``data_type``, ``table_id``.
        """
        return loads(self.parameter_server.export_config_json())
