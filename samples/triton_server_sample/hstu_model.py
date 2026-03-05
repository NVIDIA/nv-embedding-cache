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
NVE Recommendation Model Definition

Model with switchable embedding collection (TorchRec for training, NVE for inference)
and HSTU block for sequence modeling.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Optional, List, Union
from torchrec import EmbeddingCollection, EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollectionInterface
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor, JaggedTensor
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingType
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
import pynve.nve as nve            
import pynve.torch.nve_layers as nve_layers
from pynve.torch.nve_distributed import TorchDistEnv


# Default embedding table configurations
DEFAULT_TABLES = [
    EmbeddingConfig(
        name="user_embeddings",
        embedding_dim=128,
        num_embeddings=100000,
        feature_names=["user_id"]
    ),
    EmbeddingConfig(
        name="item_embeddings",
        embedding_dim=128,
        num_embeddings=100000,
        feature_names=["item_id"]
    )
]


class HSTUBlock(nn.Module):
    """
    Hierarchical Sequential Transduction Unit

    A sequence modeling block for recommendation systems.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize HSTU block.

        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Projection layer to match hidden dimension
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.dropout_value = dropout
        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through HSTU block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Project to hidden dimension
        x = self.input_projection(x)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Project back to embedding dimension
        x = self.output_projection(x)

        return x

class NVEEmbeddingCollection(EmbeddingCollectionInterface):
    def __init__(self, tables: List[EmbeddingConfig], device: torch.device, gpu_cache_sizes: List[int], memblocks: List[nve.MemBlock] ):
        
        super().__init__()
        self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._embedding_configs = tables
        self._lengths_per_embedding: List[int] = []
        self.memblocks = dict()
        table_names = set()
        for embedding_config, memblock, gpu_cache_size in zip(tables, memblocks, gpu_cache_sizes):
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            self.memblocks[embedding_config.name] = memblock
            self.embeddings[embedding_config.name] = nve_layers.NVEmbedding(
                embedding_config.num_embeddings, 
                embedding_config.embedding_dim, 
                torch.float32, 
                nve_layers.CacheType.LinearUVM, 
                memblock=self.memblocks[embedding_config.name], 
                gpu_cache_size=gpu_cache_size, 
                device=device, 
                optimize_for_training=False
            )

        self._device: torch.device = device
        self._feature_names: Dict[str, List[str]] = {table.name: table.feature_names for table in tables}

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Run the EmbeddingCollection forward pass. This method takes in a `KeyedJaggedTensor`
        and returns a `Dict` of `JaggedTensor`s, which is the result of getting the embeddings for each feature.

        Args:
            features (KeyedJaggedTensor): Input KJT
        Returns:
            Dict[str, JaggedTensor]
        """

        feature_embeddings: Dict[str, JaggedTensor] = {}
        for embedding_name, embedding in self.embeddings.items():
            for feature_name in self._feature_names[embedding_name]:
                f = features[feature_name]
                lookup = embedding(f.values())
            feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup,
                    lengths=f.lengths(),
                    weights=None,
            )
        
        return feature_embeddings

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_configs

    def need_indices(self) -> bool:
        return True

    def embedding_dim(self) -> int:
        return self._embedding_configs[0].embedding_dim

    def embedding_names_by_table(self) -> List[List[str]]:
        return list(self._feature_names.values())


class NVERecommendationModel(nn.Module):
    """
    Recommendation model with switchable embedding backend.

    Uses TorchRec EmbeddingBagCollection for training and NVE embeddings for inference.
    Includes HSTU block for sequence modeling.
    """

    def __init__(
        self,
        tables: Optional[List[EmbeddingConfig]] = None,
        hidden_dim: int = 256,
        num_hstu_layers: int = 2,
        mode: str = "training",
        device: Union[str, torch.device] = "cuda:0",
        optimizer_type: EmbOptimType = EmbOptimType.EXACT_SGD,
        learning_rate: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize the recommendation model.

        Args:
            tables: List of EmbeddingConfig for the embedding tables (default: DEFAULT_TABLES)
            hidden_dim: Hidden dimension for HSTU block
            num_hstu_layers: Number of HSTU transformer layers
            mode: "training" or "inference"
            device: Device to run on
            optimizer_type: Optimizer type for embeddings (default: EXACT_SGD)
            learning_rate: Learning rate for optimizer (default: 0.01)
            eps: Epsilon for optimizer (default: 1e-8)
            weight_decay: Weight decay for optimizer (default: 0.0)
        """
        super().__init__()

        self.tables = tables if tables is not None else DEFAULT_TABLES
        self.embedding_dim = self.tables[0].embedding_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        # Normalize device to string format
        self.device = str(device) if isinstance(device, torch.device) else device

        # Optimizer configuration
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize embedding collection based on mode
        if mode == "training":
            # Create and shard TorchRec EmbeddingCollection
            self.embedding_collection = self._create_and_shard_embeddings()
        elif mode == "inference":
            # Create NVE EmbeddingCollection
            self.embedding_collection = self._create_nve_embeddings()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'training' or 'inference'")

        # HSTU block for sequence modeling
        self.hstu_block = HSTUBlock(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_hstu_layers
        ).to(device)

        # Output layer for prediction
        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)
        ).to(device)

    def _create_nve_embeddings(self) -> nve_layers.NVEmbedding:
        """
        Create NVE Embedding on device.
        """
        env = TorchDistEnv()
        memblocks = []
        for table in self.tables:
            if dist.is_initialized() and dist.get_world_size() > 1:
                memblocks.append(nve.DistMemBlock(env, table.num_embeddings, table.embedding_dim, nve.DataType_t.Float32))
            else:
                memblocks.append(nve.DistHostMemBlock(env, table.num_embeddings, table.embedding_dim, nve.DataType_t.Float32))
        return NVEEmbeddingCollection(self.tables, torch.device(self.device), [1024**3] * len(memblocks), memblocks)

    def _create_torchrec_embeddings(self) -> EmbeddingCollection:
        """
        Create TorchRec EmbeddingCollection on meta device (for sharding).

        Returns:
            EmbeddingCollection
        """
        return EmbeddingCollection(
            tables=self.tables,
            device=torch.device("meta")  # Use meta device for sharding
        )

    def _create_and_shard_embeddings(self):
        """
        Create and shard TorchRec EmbeddingCollection for distributed training.

        Returns:
            Sharded EmbeddingCollection wrapped in DistributedModelParallel
        """
        # Create embedding collection on meta device
        ec = self._create_torchrec_embeddings()

        # Setup sharder with optimizer parameters
        optimizer_kwargs = {
            "optimizer": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }
        fused_params = {"output_dtype": SparseType.FP32}
        fused_params.update(optimizer_kwargs)

        sharder = EmbeddingCollectionSharder(
            fused_params=fused_params,
            use_index_dedup=True,
        )

        # Setup topology (simplified - uses current system info)
        topology = Topology(
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            local_world_size=dist.get_world_size() if dist.is_initialized() else 1,
            compute_device="cuda",
        )

        # Setup sharding constraints (default to row-wise)
        constraints = {}
        for table in self.tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value]
            )

        # Create sharding plan
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
        )
        plan = planner.collective_plan(
            ec, [sharder], dist.GroupMember.WORLD if dist.is_initialized() else None
        )

        # Wrap in DistributedModelParallel
        from torchrec.distributed import DistributedModelParallel

        sharded_model = DistributedModelParallel(
            module=ec,
            device=torch.device(self.device),
            sharders=[sharder],
            plan=plan,
        )

        return sharded_model

    def get_optimizer_config(self) -> Dict:
        """
        Get optimizer configuration for non-embedding parameters.

        Returns:
            Dictionary with optimizer configuration
        """
        return {
            "type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def forward(self, kjt: KeyedJaggedTensor, sequence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            kjt: KeyedJaggedTensor with input features
                (works with both TorchRec and NVE implementations)
            sequence_mask: Optional mask for sequence padding

        Returns:
            Predictions tensor of shape (batch_size,)
        """
        # Lookup embeddings - unified interface for TorchRec and NVE
        embeddings = self.embedding_collection(kjt)

        # Get batch size from KeyedJaggedTensor
        # batch_size = total_lengths / number_of_keys
        batch_size = len(kjt.lengths()) // len(kjt.keys())

        # Concatenate all embedding features
        embedding_list = [embeddings[key].values() for key in sorted(embeddings.keys())]
        concatenated_embeddings = torch.cat(embedding_list, dim=1)

        # Reshape to (batch_size, seq_len, embedding_dim)
        concatenated_embeddings = concatenated_embeddings.view(batch_size, -1, self.embedding_dim)

        # Apply HSTU block
        sequence_output = self.hstu_block(concatenated_embeddings, mask=sequence_mask)

        # Pool sequence output (take mean over sequence dimension)
        pooled_output = sequence_output.mean(dim=1)

        # Final prediction
        output = self.output_layer(pooled_output)

        return output.squeeze(-1)
