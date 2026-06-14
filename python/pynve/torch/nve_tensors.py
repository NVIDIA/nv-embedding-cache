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

# Tensor wrapper class for a Parameter Tensor with caches
class CachedTable(torch.nn.parameter.Parameter):
    """A Parameter tensor wrapper that supports caching functionality.

    This class extends torch.nn.parameter.Parameter to add caching capabilities for embedding tables.
    It requires a cache object to be passed during initialization which handles the actual caching logic.

    Args:
        *args: Variable length argument list passed to Parameter constructor
        **kwargs: Arbitrary keyword arguments. Must include 'cache' which is the caching implementation
            to use for this tensor.

    Raises:
        Exception: If no 'cache' argument is provided in kwargs

    Notes:
        Users are not expected to create a CachedTable directly.
        Instead, they should use one of the layer classes in nve_layers
    """
    def __new__(cls, *args, **kwargs):
        if not ("cache" in kwargs):
            raise Exception("building a CachedTable Tensor require a viable cache argument")
        cache = kwargs["cache"]
        kwargs.pop("cache")
       
        instance = super().__new__(cls, *args, **kwargs)
        instance.emb_layer = cache
        return instance

    def __getstate__(self):
        # `emb_layer` is a C++ cache handle (nve.LinearUVMEmbedding / GPUEmbedding)
        # that cannot be pickled. torch.export.save() serializes the module's
        # state_dict (this Parameter included) via torch.save, so the handle must
        # be excluded from the pickled state. torch's tensor reduction calls
        # __getstate__ (via torch._utils._get_obj_state), so we return a copy of
        # __dict__ with the handle dropped — leaving the live `self.emb_layer`
        # intact so the layer stays usable after export (mutating self here would
        # break it). The exported graph dispatches on the per-layer marker tensor
        # + C++ registry and reloads weights from the .nve file, so the cache
        # handle is not needed in the artifact.
        state = dict(self.__dict__)
        state["emb_layer"] = None
        return state

    # sgd optimizer will call this function to modify the storage
    # we will overload it here so we can tell the caches to update
    def add_(self, other : torch.Tensor, *, alpha : float = 1):
        """In-place addition of a tensor to the current tensor.

        This method performs an in-place addition of another tensor to the current tensor.
        The operation is performed element-wise, and the result is stored in the current tensor.

        Args:
            other (torch.Tensor): The tensor to add to the current tensor
            alpha (float, optional): The scaling factor for the addition. Defaults to 1.
        """
        factored = other * alpha
        #update caches and data
        self.emb_layer.accumulate(factored._nnz(), factored._indices().data_ptr(), factored._values().data_ptr(), torch.cuda.current_stream(self.device).cuda_stream)
