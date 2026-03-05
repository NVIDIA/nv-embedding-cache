#!/usr/bin/python
#
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

import csv
import os
from argparse import Namespace, ArgumentParser

def benchmark_arg_parser():
    """
    Create ArgumentParser with shared args across benchmarks.
    """
    parser = ArgumentParser()
    parser.add_argument("--batch", default=1024, help="Batch size (default: 1024)", type=int)
    parser.add_argument("--hotness", default=1024 ,help="Hotness (default: 1024)", type=int)
    parser.add_argument("--data_type", default='float32', choices=['float32', 'float16'] ,help="Data type (default: float32)")
    parser.add_argument("--num_table_rows", "-nr", default=1_000_000 ,help="Number of rows in the embedding table (default: 1000000)", type=int)
    parser.add_argument("--load_factor", "-lf", default=0.01 ,help="Load factor of the GPU cache(default: 0.01)", type=float)
    parser.add_argument("--num_steps", "-ns", default=100, help="Number of inference steps (default: 100)", type=int)
    parser.add_argument("--num_warmup_steps", "-nw", default=100, help="Number of warmup steps (default: 100)", type=int)
    parser.add_argument("--alpha", "-a", default=1.05 ,help="Alpha power-law coefficient used to generate input distribution (default: 1.05)", type=float)
    parser.add_argument("--embedding_dim", "-ed", default=128 ,help="Embedding dimension (default: 128)", type=int)
    parser.add_argument("--kernel", "-k", default=1 ,help="Kernel mode override (default: 1)", type=int)
    parser.add_argument("--kernel_mode_value_1", "-k1", default=0 ,help="Kernel value 1 override (default: -1)", type=int)
    parser.add_argument("--logging_interval", "-li", default=-1 ,help="Hitrate logging interval (default: -1)", type=int)
    parser.add_argument("--csv_filename", "-csv", default=None, help="CSV output file for metrics", type=str)
    parser.add_argument("--verbose", "-v", help="Increase output verbosity", action="store_true")
    return parser

def write_benchmark_csv(filename: str, args: Namespace, **metrics):
    """Write benchmark results to CSV with automatic arg extraction and flexible metrics.
    
    This function automatically extracts all arguments from the argparse Namespace,
    prefixes them with '_', and combines them with any metrics you provide as kwargs.
    
    Args:
        filename: Path to the CSV file to write/append to
        args: Argparse Namespace containing benchmark arguments (written with '_' prefix)
        **metrics: Any metrics to write to CSV (e.g., mkps=10.5, gbps=42.3).
                  These are written without prefix.
    
    Example:
        write_benchmark_csv('results.csv', args, 
                          mkps=10.5, gbps=42.3, inference_time_s=1.5)
        # CSV: _batch,_mode,mkps,gbps,inference_time_s
    """
    # Extract all args
    arg_fields = list(vars(args).keys())
    
    # Build column names: args with '_' prefix, then metrics without prefix
    arg_columns = [f'_{field}' for field in arg_fields]
    metric_columns = list(metrics.keys())
    all_columns = arg_columns + metric_columns
    
    # Create directory if needed
    csv_dir = os.path.dirname(filename)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    
    # Check if file exists and has content
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
    
    # Build row data
    row_data = {}
    
    # Add args with '_' prefix
    for field in arg_fields:
        arg_value = getattr(args, field, '')
        row_data[f'_{field}'] = arg_value
    
    # Add metrics without prefix
    row_data.update(metrics)
    
    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        
        # Write header if file is new or empty
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(row_data)
    
    print(f"Metrics appended to {filename}")

