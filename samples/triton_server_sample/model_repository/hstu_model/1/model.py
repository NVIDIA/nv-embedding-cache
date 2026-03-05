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

"""
Triton Python Backend Model for NVE Recommendation
"""

import sys
import os
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from typing import List, Optional, Tuple, Dict
import gc

# Add parent directory to path to import hstu_model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, parent_dir)

import triton_python_backend_utils as pb_utils
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec import EmbeddingConfig
from hstu_model import NVERecommendationModel


def get_dist_params(args: dict, model_config: dict, logger) -> Tuple[int, int, int]:
    """
    Derive distributed parameters from Triton args and model config.

    Rank formula (per group, with offset for preceding groups):
        rank = group_offset + gpu_index_in_group * count + instance_index

    Where instance_index (0..count-1) comes from the instance name suffix,
    and group_offset accumulates count*len(gpus) for all preceding groups.

    If no instance_group is specified, we assume a single group with all GPUs

    Args:
        args: Triton args dictionary (model_instance_name, model_instance_device_id)
        model_config: Parsed model config dict from config.pbtxt

    Returns:
        (rank, world_size, device_id)
    """
    device_id = int(args['model_instance_device_id'])
    # Instance index within the group (0..count-1), same across GPUs in the group
    instance_index = int(args['model_instance_name'].split('_')[-1])

    if 'instance_group' in model_config and len(model_config['instance_group']) > 0:
        groups = model_config['instance_group']        
        world_size = sum(g.get('count', 1) * len(g.get('gpus', [0])) for g in groups)

        group_offset = 0
        rank = None
        for g in groups:
            gpus = g.get('gpus', [0])
            count = g.get('count', 1)
            if device_id in gpus:
                gpu_index = gpus.index(device_id)
                rank = group_offset + gpu_index * count + instance_index
                break
            group_offset += len(gpus) * count

        if rank is None:
            raise ValueError(f"device_id {device_id} not found in any instance_group")
    else:
        world_size = torch.cuda.device_count()
        rank = device_id
    return rank, world_size, device_id


def setup_distributed(rank: int, world_size: int, device_id: int) -> torch.device:
    """
    Setup distributed environment for inference.

    Args:
        rank: Process rank
        world_size: Total number of processes
        device_id: GPU device ID

    Returns:
        device: CUDA device to run inference on
    """
    os.environ.setdefault('RANK', str(rank))
    os.environ.setdefault('LOCAL_RANK', str(rank))
    os.environ.setdefault('WORLD_SIZE', str(world_size))
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12357')

    device = torch.device(f"cuda:{device_id}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    
    torch.cuda.set_device(device)

    return device


def get_checkpoint_dir(args: dict, model_config: dict) -> str:
    """
    Resolve the checkpoint directory from config.pbtxt parameters.

    Args:
        args: Triton args dictionary (model_repository, model_name)
        model_config: Parsed model config dict from config.pbtxt

    Returns:
        Absolute path to the checkpoint directory

    Raises:
        ValueError: If no checkpoint_path parameter is specified
    """
    checkpoint_dir: Optional[str] = None
    if 'parameters' in model_config:
        params = model_config['parameters']
        if 'checkpoint_path' in params:
            checkpoint_dir = params['checkpoint_path']['string_value']
            # Resolve relative paths relative to model directory
            if not os.path.isabs(checkpoint_dir):
                model_dir = args['model_repository']
                checkpoint_dir = os.path.abspath(
                    os.path.join(model_dir, checkpoint_dir)
                )

    if checkpoint_dir is None:
        raise ValueError("No checkpoint_path specified in config.pbtxt parameters")

    return checkpoint_dir


class TritonPythonModel:
    """Triton Python backend model wrapper for NVE recommendations."""

    def initialize(self, args):
        """
        Initialize the model. Called once when the model is loaded.

        Args:
            args: Dictionary containing model configuration
        """
        # Parse model config
        model_config = json.loads(args['model_config'])

        self.logger = pb_utils.Logger
        # Derive rank, world_size, device_id from Triton args
        rank, world_size, device_id = get_dist_params(args, model_config, self.logger)

        # Setup distributed environment
        self.device = setup_distributed(rank, world_size, device_id)
        self.logger.log_info(f"Distributed initialized: instance={args['model_instance_name']}, rank={rank}, world_size={world_size}, device={self.device}")

        # Resolve checkpoint directory from config.pbtxt
        checkpoint_dir = get_checkpoint_dir(args, model_config)
        self.logger.log_info(f"Loading model from: {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Load metadata
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            self.logger.log_error(f"Metadata file not found: {metadata_path}")
            raise ValueError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if rank == 0:
            self.logger.log_info(f"Loaded metadata: {metadata}")

        # Reconstruct EmbeddingConfig tables from metadata
        tables = [
            EmbeddingConfig(
                name=t['name'],
                embedding_dim=t['embedding_dim'],
                num_embeddings=t['num_embeddings'],
                feature_names=t['feature_names'],
            )
            for t in metadata['tables']
        ]

        self.logger.log_info("Creating model in inference mode...")
        # Store table feature names for input processing
        self.feature_names = sorted(
            feature for table in tables for feature in table.feature_names
        )
        # Create model in inference mode
        try:
            self.model = NVERecommendationModel(
                tables=tables,
                hidden_dim=metadata['hidden_dim'],
                num_hstu_layers=metadata['num_layers'],
                mode="inference",
                device=self.device
            )

            # Build mapping: saved state dict key -> NVE weight tensor
            # table_name_to_weight_key maps e.g. "user_embeddings" -> "embedding_collection.embedding_collection.embeddings.user_embeddings.weight"
            # The NVE weight tensor lives at model.embedding_collection.embeddings[table_name].weight
            embedding_weight_names: Dict[str, torch.Tensor] = {
                weight_key: self.model.embedding_collection.embeddings[table_name].weight
                for table_name, weight_key in metadata['table_name_to_weight_key'].items()
            }

            # first we load the sparse model
            # since NVE shares the same weight tensor across all ranks, we need only to load the weight tensor for rank 0
            if rank == 0:
                partial_state_dict = {"model": embedding_weight_names}
            else:
                partial_state_dict = {"model": {}}
            dist_cp.load(state_dict=partial_state_dict, checkpoint_id=checkpoint_dir)
            dist.barrier()
            # loading the dense model is straightforward, we just need to remove the already loaded sparse model weight
            # NOTE: embedding_weight_names uses training/checkpoint keys (might be different implementation for the embeddings), but inference model
            # state_dict uses different keys (no DMP wrapper). Build the exclusion set from the inference model directly.
            inference_emb_keys = {
                k for k in self.model.state_dict().keys()
                if any(table_name in k for table_name in metadata['table_name_to_weight_key'].keys())
            }
            dense_state = {k: v for k, v in self.model.state_dict().items() if k not in inference_emb_keys}
            dist_cp.load(state_dict={"model": dense_state}, checkpoint_id=checkpoint_dir)
            
            self.model.eval()

            self.logger.log_info("Model loaded successfully!")

        except Exception as e:
            self.logger.log_error(f"Failed to load model: {e}")
            self.logger.log_info("Creating dummy model for testing (NVE not implemented yet)")
            self.model = None

        

        self.logger.log_info(f"Expected features: {self.feature_names}")

    def execute(self, requests):
        """
        Execute inference on a batch of requests.

        Input tensor "indices":
            - dtype : INT64
            - shape : [batch_size, num_features]
            - Each row corresponds to one sample in the batch.
            - Each column corresponds to one embedding feature, in sorted order
              of feature names (e.g. ['item_id', 'user_id'] for a 2-table model).
            - Each value is a lookup index into the corresponding embedding table,
              in the range [0, num_embeddings) for that table.
            - Example for batch_size=2, features=[item_id, user_id]:
                [[42,  7],   # sample 0: item_id=42, user_id=7
                 [13, 99]]   # sample 1: item_id=13, user_id=99

        Output tensor "predictions":
            - dtype : FP32
            - shape : [batch_size]
            - Raw logits (pre-sigmoid) per sample. Apply sigmoid to get [0, 1] scores.

        Args:
            requests: List of pb_utils.InferenceRequest

        Returns:
            List of pb_utils.InferenceResponse
        """
        responses = []

        for request in requests:
            try:
                # Get input tensor (indices)
                indices_tensor = pb_utils.get_input_tensor_by_name(request, "indices")
                indices = indices_tensor.as_numpy()

                # Convert to torch tensor on correct device
                indices_torch = torch.from_numpy(indices).to(self.device)

                # Create KeyedJaggedTensor
                # Assuming indices come in as [batch_size, num_features]
                batch_size = indices_torch.shape[0]
                num_features = len(self.feature_names)

                # Flatten and create KJT
                values = indices_torch.flatten()
                lengths = torch.ones(batch_size * num_features, dtype=torch.int32, device=self.device)

                kjt = KeyedJaggedTensor(
                    keys=self.feature_names,
                    values=values,
                    lengths=lengths
                )

                # Run inference
                if self.model is not None:
                    with torch.no_grad():
                        predictions = self.model(kjt)

                    # Convert to numpy
                    predictions_np = predictions.cpu().numpy()
                else:
                    # Dummy predictions for testing
                    predictions_np = np.random.rand(batch_size).astype(np.float32)

                # Create output tensor
                output_tensor = pb_utils.Tensor("predictions", predictions_np)

                # Create response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                self.logger.log_error(f"Error during inference: {e}")
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Inference failed: {str(e)}")
                )
                responses.append(error_response)

        return responses

    def finalize(self):
        """
        Cleanup resources. Called when the model is unloaded.
        """
        self.logger.log_info("Cleaning up model resources...")

        del self.model
        gc.collect()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        self.logger.log_info("Model finalized.")
