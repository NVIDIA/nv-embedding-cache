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
import numpy as np
from scipy.optimize import brentq
from argparse import Namespace, ArgumentParser, Action

class _MutuallyExclusiveAlphaPareto(Action):
    """Raises an error if both --alpha and --pareto are explicitly provided."""
    def __call__(self, parser, namespace, values, _=None):
        setattr(namespace, self.dest, values)
        if not hasattr(namespace, '_explicit_args'):
            namespace._explicit_args = set()
        namespace._explicit_args.add(self.dest)
        if {'alpha', 'pareto'}.issubset(namespace._explicit_args):
            parser.error("--alpha and --pareto are mutually exclusive; use one or the other")

def benchmark_arg_parser():
    """
    Create ArgumentParser with shared args across benchmarks.
    """
    parser = ArgumentParser()
    parser.add_argument("--batch", "-b", default=1024, help="Batch size (default: 1024)", type=int)
    parser.add_argument("--hotness", default=1024 ,help="Hotness (default: 1024)", type=int)
    parser.add_argument("--data_type", "-dt", default='float32', choices=['float32', 'float16'] ,help="Data type (default: float32)")
    parser.add_argument("--num_table_rows", "-nr", default=1_000_000 ,help="Number of rows in the embedding table (default: 1000000)", type=int)
    parser.add_argument("--load_factor", "-lf", default=0.01 ,help="Load factor of the GPU cache(default: 0.01)", type=float)
    parser.add_argument("--num_steps", "-ns", default=100, help="Number of inference steps (default: 100)", type=int)
    parser.add_argument("--num_warmup_steps", "-nw", default=100, help="Number of warmup steps (default: 100)", type=int)
    parser.add_argument("--alpha", "-a", default=1.05 ,help="Alpha power-law coefficient used to generate input distribution (default: 1.05)", type=float, action=_MutuallyExclusiveAlphaPareto)
    parser.add_argument("--pareto", "-p", default=0. ,help="Pareto ratio to derive power-law Alpha from e.g.use 0.2 for 80/20 rule, 0.1 for 90/10 etc. This overwrites --alpha (default: 0.)", type=float, action=_MutuallyExclusiveAlphaPareto)
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
    # Extract all args, skipping internal fields prefixed with '_'
    arg_fields = [k for k in vars(args).keys() if not k.startswith('_')]

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

def convert_pareto_to_alpha(x: float, N: int):
    """
    Solves the equation 1-x = ((x*N)^(1-a) - 1) / (N^(1-a) - 1) for a.
    Parameters:
    x (float): The fraction of items (e.g., 0.2 for the 80/20 rule).
    N (int): The total number of items.
    Returns:
    float: The value of a (G(x; N)).
    """
    def objective(a):
        s = 1 - a
        # Handle the singularity at a = 1 using L'Hopital's Rule
        # The limit as s -> 0 of (B^s - 1) / (A^s - 1) is ln(B) / ln(A)
        if abs(s) < 1e-12:
            val = np.log(x * N) / np.log(N)
        else:
            try:
                val = (np.power(x * N, s) - 1) / (np.power(N, s) - 1)
            except OverflowError:
                # Handle large s to prevent numerical instability
                val = np.power(x, s) if s > 0 else 1.0
        return val - (1 - x)
    # Search for the root in a reasonable range for power-law exponents
    # a typically falls between 0 and 5 for most real-world datasets.
    try:
        a_solution = brentq(objective, 0.001, 10.0)
        return a_solution
    except ValueError:
        return None
