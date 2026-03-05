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
Triton Inference Client for NVE Recommendation Model

Sends inference requests to a running Triton server and displays predictions.
Reads num_embeddings and feature layout from the model's metadata.json,
located via checkpoint_path in config.pbtxt.

Usage:
    python client_inference.py
    python client_inference.py --url localhost:8000 --batch-size 32 --num-requests 5
"""

import argparse
import json
import os
import re
import time
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


MODEL_NAME = "hstu_model"

# Default model repo path relative to this script: ../model_repository
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_REPO = os.path.join(_SCRIPT_DIR, "..", "model_repository")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test NVE model inference via Triton")
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton server URL (host:port)")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of samples per request")
    parser.add_argument("--num-requests", type=int, default=3, help="Number of requests to send")
    parser.add_argument("--model-repo", type=str, default=_DEFAULT_MODEL_REPO,
                        help="Path to Triton model repository (to locate config.pbtxt)")
    return parser.parse_args()


def read_checkpoint_path_from_pbtxt(model_repo: str, model_name: str) -> str:
    """Parse checkpoint_path parameter from config.pbtxt and resolve to absolute path."""
    config_path = os.path.join(model_repo, model_name, "config.pbtxt")
    with open(config_path) as f:
        content = f.read()

    match = re.search(r'key:\s*"checkpoint_path".*?string_value:\s*"([^"]+)"', content, re.DOTALL)
    if not match:
        raise ValueError(f"checkpoint_path not found in {config_path}")

    checkpoint_path = match.group(1)
    if not os.path.isabs(checkpoint_path):
        model_dir = os.path.join(model_repo, model_name)
        checkpoint_path = os.path.abspath(os.path.join(model_dir, checkpoint_path))
        
    return checkpoint_path


def read_model_metadata(checkpoint_path: str) -> dict:
    """Load metadata.json from the checkpoint directory."""
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    with open(metadata_path) as f:
        return json.load(f)


def build_feature_index_map(metadata: dict) -> list:
    """
    Returns a list of (feature_name, num_embeddings) sorted by feature_name,
    matching the order used by the model (sorted feature names).
    """
    feature_map = {}
    for table in metadata["tables"]:
        for feature_name in table["feature_names"]:
            feature_map[feature_name] = table["num_embeddings"]
    return [(name, feature_map[name]) for name in sorted(feature_map.keys())]


def create_client(url: str) -> httpclient.InferenceServerClient:
    client = httpclient.InferenceServerClient(url=url)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live")
    if not client.is_server_ready():
        raise RuntimeError(f"Triton server at {url} is not ready")
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on server")
    return client


def build_request(batch_size: int, feature_index_map: list) -> np.ndarray:
    """
    Generate random indices shaped [batch_size, num_features].
    Each column uses the correct num_embeddings for that feature's table.
    """
    num_features = len(feature_index_map)
    indices = np.zeros((batch_size, num_features), dtype=np.int64)
    for col, (_, num_embeddings) in enumerate(feature_index_map):
        indices[:, col] = np.random.randint(0, num_embeddings, size=batch_size)
    return indices


def send_request(
    client: httpclient.InferenceServerClient,
    indices: np.ndarray,
) -> np.ndarray:
    """Send a single inference request and return predictions."""
    infer_input = httpclient.InferInput("indices", indices.shape, "INT64")
    infer_input.set_data_from_numpy(indices)

    infer_output = httpclient.InferRequestedOutput("predictions")

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )

    return response.as_numpy("predictions")


def main() -> None:
    args = parse_args()

    # Load model config from pbtxt and metadata.json
    checkpoint_path = read_checkpoint_path_from_pbtxt(args.model_repo, MODEL_NAME)
    metadata = read_model_metadata(checkpoint_path)
    feature_index_map = build_feature_index_map(metadata)

    print(f"Model features: {[name for name, _ in feature_index_map]}")
    print(f"Num embeddings per feature: {[(name, n) for name, n in feature_index_map]}")

    print(f"\nConnecting to Triton at {args.url}...")
    try:
        client = create_client(args.url)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print(f"Server ready. Sending {args.num_requests} request(s) with batch_size={args.batch_size}\n")

    latencies = []

    for i in range(args.num_requests):
        indices = build_request(args.batch_size, feature_index_map)

        t0 = time.perf_counter()
        try:
            predictions = send_request(client, indices)
        except InferenceServerException as e:
            print(f"Request {i+1} FAILED: {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        print(f"Request {i+1}/{args.num_requests}  latency={latency_ms:.1f}ms")
        print(f"  predictions: {predictions}")

    if latencies:
        print(f"\nSummary over {len(latencies)} request(s):")
        print(f"  mean latency : {np.mean(latencies):.1f} ms")
        print(f"  p50          : {np.percentile(latencies, 50):.1f} ms")
        print(f"  p99          : {np.percentile(latencies, 99):.1f} ms")


if __name__ == "__main__":
    main()
