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
from pynve.torch.nve_tensors import CachedTable
import pynve.nve as nve

# ---------------------------------------------------------------------------
# torch.export-compatible ops
#
# These wrap torch.ops.nve_ops.embedding_lookup / embedding_lookup_with_pooling
# (registered in nve.so via STABLE_TORCH_LIBRARY) so that:
#   - Forward: the custom op is traced by torch.export / torch.jit.trace.
#   - Backward: two variants per op:
#       *Training  — takes weight (CachedTable), returns sparse gradient
#                    for use with any PyTorch optimizer.
#       *Inference — no weight input, raises on backward.
# ---------------------------------------------------------------------------


# ===== Embedding =====

class NVEmbeddingOp(torch.autograd.Function):
    """Inference-only embedding lookup via torch.ops.nve_ops.embedding_lookup.

    Forward is traceable by torch.export / torch.jit.trace.
    Backward raises RuntimeError — use NVEmbeddingOpTraining for training.
    """

    @staticmethod
    def forward(keys: torch.Tensor, layer_id: int):
        return torch.ops.nve_ops.embedding_lookup(keys, layer_id)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "NVEmbeddingOp does not support backward. "
            "Use optimize_for_training=True to enable gradient computation.")


class NVEmbeddingOpTraining(torch.autograd.Function):
    """Training embedding lookup via torch.ops.nve_ops.embedding_lookup.

    Forward is traceable by torch.export / torch.jit.trace.
    weight (CachedTable) is an unused tensor input in the forward graph but
    receives a sparse gradient in backward, compatible with any optimizer.
    """

    @staticmethod
    def forward(keys: torch.Tensor, weight: CachedTable, layer_id: int):
        return torch.ops.nve_ops.embedding_lookup(keys, layer_id)

    @staticmethod
    def setup_context(ctx, inputs, output):
        keys, weight, layer_id = inputs
        ctx.save_for_backward(keys, weight)

    @staticmethod
    def backward(ctx, grad_output):
        keys, weight = ctx.saved_tensors
        keys_flat = keys.reshape((1, -1))
        num_keys = keys_flat.numel()
        embed_dim = grad_output.shape[-1]
        unique_keys = torch.empty(num_keys, dtype=torch.int64, device=keys.device)
        result = torch.empty(num_keys, embed_dim, dtype=grad_output.dtype, device=keys.device)
        stream = torch.cuda.current_stream(keys.device).cuda_stream
        nve_op = weight.emb_layer
        num_unique = nve_op.concat_backprop(
            num_keys, keys_flat.data_ptr(), grad_output.data_ptr(),
            unique_keys.data_ptr(), result.data_ptr(), stream)
        unique_keys = unique_keys[:num_unique].reshape((1, -1))
        result = result[:num_unique]
        grad_weight = torch.sparse_coo_tensor(
            unique_keys, result, weight.shape, device=weight.device)
        return None, grad_weight, None


# ===== EmbeddingBag =====

class NVEmbeddingBagOp(torch.autograd.Function):
    """Inference-only embedding bag lookup via torch.ops.nve_ops.embedding_lookup_with_pooling.

    Forward is traceable by torch.export / torch.jit.trace.
    Backward raises RuntimeError — use NVEmbeddingBagOpTraining for training.
    """

    @staticmethod
    def forward(keys: torch.Tensor, offsets: torch.Tensor,
                per_sample_weights, pooling_type: int, layer_id: int):
        return torch.ops.nve_ops.embedding_lookup_with_pooling(
            keys, offsets, per_sample_weights, pooling_type, layer_id)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "NVEmbeddingBagOp does not support backward. "
            "Use optimize_for_training=True to enable gradient computation.")


class NVEmbeddingBagOpTraining(torch.autograd.Function):
    """Training embedding bag lookup via torch.ops.nve_ops.embedding_lookup_with_pooling.

    Forward is traceable by torch.export / torch.jit.trace.
    weight (CachedTable) is an unused tensor input in the forward graph but
    receives a sparse gradient in backward, compatible with any optimizer.
    """

    @staticmethod
    def forward(keys: torch.Tensor, weight: CachedTable, offsets: torch.Tensor,
                per_sample_weights, pooling_type: int, layer_id: int):
        return torch.ops.nve_ops.embedding_lookup_with_pooling(
            keys, offsets, per_sample_weights, pooling_type, layer_id)

    @staticmethod
    def setup_context(ctx, inputs, output):
        keys, weight, offsets, per_sample_weights, pooling_type, layer_id = inputs
        ctx.pooling_type = pooling_type
        if per_sample_weights is not None:
            ctx.save_for_backward(keys, weight, offsets, per_sample_weights)
            ctx.has_per_sample_weights = True
        else:
            ctx.save_for_backward(keys, weight, offsets)
            ctx.has_per_sample_weights = False

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_per_sample_weights:
            keys, weight, offsets, per_sample_weights = ctx.saved_tensors
        else:
            keys, weight, offsets = ctx.saved_tensors
            per_sample_weights = None

        num_keys = keys.numel()
        embed_dim = grad_output.shape[-1]
        unique_keys = torch.empty(num_keys, dtype=torch.int64, device=keys.device)
        result = torch.empty(num_keys, embed_dim, dtype=grad_output.dtype, device=keys.device)
        stream = torch.cuda.current_stream(keys.device).cuda_stream
        nve_op = weight.emb_layer

        weight_dtype = nve.DataType_t.Unknown
        weight_ptr = 0
        if per_sample_weights is not None:
            weight_dtype = nve.DataType_t.Float32 if per_sample_weights.dtype == torch.float32 \
                           else nve.DataType_t.Float16
            weight_ptr = per_sample_weights.data_ptr()

        num_unique = nve_op.pooling_backprop(
            num_keys, keys.data_ptr(), grad_output.data_ptr(),
            unique_keys.data_ptr(), result.data_ptr(),
            ctx.pooling_type, offsets.numel() - 1, offsets.data_ptr(),
            weight_dtype, weight_ptr, stream)

        unique_keys = unique_keys[:num_unique].reshape((1, -1))
        result = result[:num_unique]
        grad_weight = torch.sparse_coo_tensor(
            unique_keys, result, weight.shape, device=weight.device)
        # grad_keys=None, grad_weight=sparse, grad_offsets=None,
        # grad_per_sample_weights=None, grad_pooling_type=None, grad_layer_id=None
        return None, grad_weight, None, None, None, None


# ---------------------------------------------------------------------------
# Legacy ops (pybind11 direct calls, no torch custom ops)
#
# Used as fallback when HAS_TORCH_OPS is False (torch bindings unavailable).
# Not traceable by torch.export / torch.jit.trace.
# ---------------------------------------------------------------------------

class CacheEmbeddingOp(torch.autograd.Function):
    @staticmethod
    def forward(keys: torch.Tensor, weight: CachedTable):
        nve_op = weight.emb_layer
        result = torch.empty(keys.numel(), weight.shape[-1], dtype=weight.dtype, device=keys.device)
        nve_op.lookup(keys.numel(), keys.data_ptr(), result.data_ptr(),
                      torch.cuda.current_stream(keys.device).cuda_stream)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        keys, weight = inputs
        ctx.save_for_backward(keys, weight)

    @staticmethod
    def backward(ctx, grad_output):
        keys, weight = ctx.saved_tensors
        keys_flat = keys.reshape((1, -1))
        num_keys = keys_flat.numel()
        embed_dim = grad_output.shape[-1]
        unique_keys = torch.empty(num_keys, dtype=torch.int64, device=keys.device)
        result = torch.empty(num_keys, embed_dim, dtype=grad_output.dtype, device=keys.device)
        nve_op = weight.emb_layer
        num_unique = nve_op.concat_backprop(
            num_keys, keys_flat.data_ptr(), grad_output.data_ptr(),
            unique_keys.data_ptr(), result.data_ptr(),
            torch.cuda.current_stream(keys.device).cuda_stream)
        unique_keys = unique_keys[:num_unique].reshape((1, -1))
        result = result[:num_unique]
        return None, torch.sparse_coo_tensor(unique_keys, result, weight.shape, device=weight.device)


class CacheEmbeddingBagOp(torch.autograd.Function):
    @staticmethod
    def forward(keys: torch.Tensor, weight: CachedTable, offsets: torch.Tensor,
                mode: str, per_sample_weights: torch.Tensor):
        nve_op = weight.emb_layer
        device = keys.device
        result = torch.empty(offsets.numel() - 1, weight.shape[-1], dtype=weight.dtype, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        if per_sample_weights is not None:
            w_type = nve.DataType_t.Float32 if per_sample_weights.dtype == torch.float32 \
                     else nve.DataType_t.Float16
            pool = nve.PoolingType_t.WeightedSum if mode == 'sum' else nve.PoolingType_t.WeightedMean
            nve_op.lookup_with_pooling(keys.numel(), keys.data_ptr(), result.data_ptr(),
                                       pool, offsets.numel() - 1, offsets.data_ptr(),
                                       w_type, per_sample_weights.data_ptr(), stream)
        else:
            pool = nve.PoolingType_t.Sum if mode == 'sum' else nve.PoolingType_t.Mean
            nve_op.lookup_with_pooling(keys.numel(), keys.data_ptr(), result.data_ptr(),
                                       pool, offsets.numel() - 1, offsets.data_ptr(),
                                       nve.DataType_t.Float32, 0, stream)
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        keys, weight, offsets, mode, per_sample_weights = inputs
        ctx.mode = mode
        ctx.save_for_backward(keys, weight, offsets, per_sample_weights)

    @staticmethod
    def backward(ctx, grad_output):
        keys, weight, offsets, per_sample_weights = ctx.saved_tensors
        num_keys = keys.numel()
        embed_dim = grad_output.shape[-1]
        unique_keys = torch.empty(num_keys, dtype=torch.int64, device=weight.device)
        result = torch.empty(num_keys, embed_dim, dtype=grad_output.dtype, device=weight.device)
        nve_op = weight.emb_layer
        stream = torch.cuda.current_stream(weight.device).cuda_stream
        if per_sample_weights is not None:
            w_type = nve.DataType_t.Float32 if per_sample_weights.dtype == torch.float32 \
                     else nve.DataType_t.Float16
            pool = nve.PoolingType_t.WeightedSum if ctx.mode == 'sum' else nve.PoolingType_t.WeightedMean
            num_unique = nve_op.pooling_backprop(
                num_keys, keys.data_ptr(), grad_output.data_ptr(),
                unique_keys.data_ptr(), result.data_ptr(),
                pool, offsets.numel() - 1, offsets.data_ptr(),
                w_type, per_sample_weights.data_ptr(), stream)
        else:
            pool = nve.PoolingType_t.Sum if ctx.mode == 'sum' else nve.PoolingType_t.Mean
            num_unique = nve_op.pooling_backprop(
                num_keys, keys.data_ptr(), grad_output.data_ptr(),
                unique_keys.data_ptr(), result.data_ptr(),
                pool, offsets.numel() - 1, offsets.data_ptr(),
                nve.DataType_t.Float32, 0, stream)
        unique_keys = unique_keys[:num_unique].reshape((1, -1))
        result = result[:num_unique]
        return None, torch.sparse_coo_tensor(unique_keys, result, weight.shape, device=weight.device), None, None, None
