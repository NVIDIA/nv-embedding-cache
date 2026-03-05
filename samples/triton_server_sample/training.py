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
Training script for NVE Recommendation Model
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import os
import argparse
import json
from typing import Tuple, Optional, List, Dict, Any
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec import EmbeddingConfig
from hstu_model import NVERecommendationModel, DEFAULT_TABLES


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NVE Recommendation Model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for HSTU")
    parser.add_argument("--num_hstu_layers", type=int, default=2, help="Number of HSTU layers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_path", type=str, default="trained_models", help="Path to save model")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")
    return parser.parse_args()


def setup_distributed(backend: str = "nccl") -> Tuple[int, int, torch.device]:
    """Initialize distributed training."""
    # Set required environment variables if not already set
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    return rank, world_size, device


def generate_synthetic_batch(
    batch_size: int,
    device: torch.device,
    tables: Optional[List[EmbeddingConfig]] = None
) -> KeyedJaggedTensor:
    """
    Generate synthetic training batch as KeyedJaggedTensor.

    Args:
        batch_size: Number of samples in the batch
        device: Device to create tensors on
        tables: List of EmbeddingConfig (default: DEFAULT_TABLES)

    Returns:
        KeyedJaggedTensor with user and item features
    """
    if tables is None:
        tables = DEFAULT_TABLES

    # Build a mapping from feature_name to num_embeddings
    feature_to_num_embeddings = {}
    for table in tables:
        for feature_name in table.feature_names:
            feature_to_num_embeddings[feature_name] = table.num_embeddings

    # Generate random IDs for each feature
    keys = []
    values_list = []

    for feature_name in sorted(feature_to_num_embeddings.keys()):
        num_embeddings = feature_to_num_embeddings[feature_name]
        ids = torch.randint(0, num_embeddings, (batch_size,), device=device)

        keys.append(feature_name)
        values_list.append(ids)

    # Create KeyedJaggedTensor
    # For batch_size=8 with 2 features, each sample has 1 value per feature
    # lengths should be [1, 1, ..., 1] with length = batch_size * num_keys
    num_keys = len(keys)
    lengths = torch.ones(batch_size * num_keys, dtype=torch.int32, device=device)

    kjt = KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(values_list),
        lengths=lengths
    )

    return kjt


def generate_synthetic_labels(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Generate synthetic labels for training.

    Args:
        batch_size: Number of labels
        device: Device to create tensors on

    Returns:
        Binary labels (0 or 1)
    """
    return torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)


def train(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> None:
    """
    Main training loop.

    Args:
        args: Command line arguments
        rank: Process rank
        world_size: Total number of processes
        device: Device to train on
    """
    if rank == 0:
        print("=" * 80)
        print("Training NVE Recommendation Model")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Training steps: {args.num_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  HSTU layers: {args.num_hstu_layers}")
        print(f"  World size: {world_size}")
        print("=" * 80)

    # Print device info for each rank
    print(f"[Rank {rank}] Using device: {device}")

    # Create model
    if rank == 0:
        print("\nCreating model...")

    model = NVERecommendationModel(
        tables=None,  # Use DEFAULT_TABLES
        hidden_dim=args.hidden_dim,
        num_hstu_layers=args.num_hstu_layers,
        mode="training",
        device=device,
        learning_rate=args.learning_rate
    )

    if rank == 0:
        print("Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Get optimizer config from model to ensure consistency
    optimizer_config = model.get_optimizer_config()

    if rank == 0:
        print(f"Optimizer config: {optimizer_config}")

    # Create optimizer for non-embedding parameters
    # Embedding parameters are managed by TorchRec's fused optimizer
    non_embedding_params = [
        p for name, p in model.named_parameters()
        if 'embedding_collection' not in name and p.requires_grad
    ]

    # Use SGD to match TorchRec's EXACT_SGD
    optimizer = torch.optim.SGD(
        non_embedding_params,
        lr=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay']
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    if rank == 0:
        print("\nStarting training...")

    model.train()
    total_loss = 0.0

    for step in range(args.num_steps):
        # Generate synthetic batch
        kjt = generate_synthetic_batch(args.batch_size, device, model.tables)
        labels = generate_synthetic_labels(args.batch_size, device)

        # Forward pass
        predictions = model(kjt)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Log progress
        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            if rank == 0:
                print(f"Step [{step + 1}/{args.num_steps}] Loss: {avg_loss:.4f}")
            total_loss = 0.0

    if rank == 0:
        print("\nTraining completed!")

    # Save checkpoint using distributed checkpoint (all ranks participate)
    if rank == 0:
        print(f"\nSaving checkpoint to {args.save_path}...")

    # Create checkpoint directory
    checkpoint_dir = args.save_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    # All ranks save tensors using distributed checkpoint
    state_dict = {
        "model": model.state_dict(),
    }

    dist_cp.save(
        state_dict=state_dict,
        checkpoint_id=checkpoint_dir,
    )

    # Wait for all ranks to finish saving
    dist.barrier()

    # Rank 0 saves metadata
    if rank == 0:
        # Build mapping from table name to its weight tensor key in the state dict
        full_state_dict = model.state_dict()
        table_name_to_weight_key: Dict[str, str] = {}
        for table in model.tables:
            for key in full_state_dict.keys():
                if table.name in key:
                    table_name_to_weight_key[table.name] = key
                    break

        # Dropout is stored as a float on TransformerEncoderLayer
        dropout = model.hstu_block.transformer.layers[0].dropout

        metadata: Dict[str, Any] = {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_hstu_layers,
            "dropout": model.hstu_block.dropout_value,
            "table_name_to_weight_key": table_name_to_weight_key,
            "tables": [
                {
                    "name": table.name,
                    "embedding_dim": table.embedding_dim,
                    "num_embeddings": table.num_embeddings,
                    "feature_names": table.feature_names,
                }
                for table in model.tables
            ],
        }

        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed(args.backend)

    try:
        # Train model
        train(args, rank, world_size, device)
        dist.barrier()

    finally:
        # Cleanup
        if rank == 0:
            print("\nCleaning up...")
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
