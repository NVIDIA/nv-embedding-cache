# PyNVE Triton Server Sample

This sample demonstrates how to deploy a PyTorch model using NVE (NVIDIA Embedding Cache) with
NVIDIA Triton Inference Server using the Python backend.

## Overview

This example shows how to:
- Define a PyTorch model with NVE-backed embeddings (HSTU architecture)
- Train and save the model
- Deploy the model on Triton Server using the Python backend

## Directory Structure

```
triton_server_sample/
├── README.md                           # This file
├── Dockerfile                          # Triton server image with PyNVE installed
├── requirements.txt                    # Python dependencies for the client
├── hstu_model.py                       # PyTorch model definition
├── training.py                         # Training script
├── trained_models/                     # Saved model weights (created after training)
├── model_repository/
│   └── hstu_model/
│       ├── config.pbtxt                # Triton model configuration
│       └── 1/
│           └── model.py               # Triton Python backend inference script
└── client/
    └── client_inference.py               # HTTP client — latency measurements
```
## Step 1 — Train the Model

Run the training script to produce weights and metadata in `trained_models/`:

```bash
python training.py
```

This saves a distributed checkpoint and a `metadata.json` describing the model
architecture (table names, embedding dims, hidden dim, etc.) that the server reads at startup.

---

## Step 2 — Build the Docker Image

Build from the **repository root** so the full source tree is available as the build context:

```bash
docker build \
  -f samples/triton_server_sample/Dockerfile \
  -t hstu_triton \
  .
```

---

## Step 3 — Start the Triton Server

Run the container, mounting the model repository and the trained model weights:

```bash
docker run --gpus all --rm \
  --net=host \
  --cap-add=SYS_PTRACE \
  -v $(pwd)/samples/triton_server_sample/:/triton_server_sample \
  hstu_triton
```

**Check server health** (once the server is up):

```bash
curl -s localhost:8000/v2/health/ready
```

---

## Step 4 — Send Inference Requests

Install client dependencies (outside Docker, on the host):

```bash
pip install -r samples/triton_server_sample/requirements.txt
```

### `client/client_inference.py` — HTTP client with latency reporting

Sends random requests to the HTTP endpoint and prints per-request latency and a summary:

```bash
python client/client_inference.py
```

| Option | Default | Description |
|---|---|---|
| `--url` | `localhost:8000` | Triton HTTP endpoint (`host:port`) |
| `--batch-size` | `8` | Number of samples per request |
| `--num-requests` | `3` | Number of requests to send |
| `--model-repo` | `../model_repository` | Path used to locate `config.pbtxt` |

Example output:
```
Model features: ['item_id', 'user_id']
Num embeddings per feature: [('item_id', 100000), ('user_id', 50000)]

Connecting to Triton at localhost:8000...
Server ready. Sending 3 request(s) with batch_size=8

Request 1/3  latency=12.4ms
  predictions: [0.512 0.348 ...]
...
Summary over 3 request(s):
  mean latency : 11.8 ms
  p50          : 11.5 ms
  p99          : 14.2 ms
```

## Model Configuration (`config.pbtxt`)

- **Backend**: Python
- **Max batch size**: 128
- **Input**: `indices` — INT64, shape `[batch_size, num_features]`
- **Output**: `predictions` — FP32, shape `[batch_size]`
- **Dynamic batching**: enabled, preferred sizes 32 / 64 / 128

The `checkpoint_path` parameter tells the backend where to find the saved weights
and `metadata.json`:

```protobuf
parameters: {
  key: "checkpoint_path"
  value: { string_value: "../../trained_models" }
}
```

The path is resolved relative to the model directory
(`model_repository/hstu_model/`), so `../../trained_models` points to
`trained_models/` at the sample root.  When using the Docker volume mounts from
Step 3, set it to the absolute path inside the container:

```protobuf
parameters: {
  key: "checkpoint_path"
  value: { string_value: "/trained_models" }
}
```

---

## Key Components

### `hstu_model.py`
Defines the model architecture: `EmbeddingCollection` (NVE/TorchRec), HSTU transformer
layers, and a linear output head.

### `training.py`
Creates the model, runs synthetic training iterations, and saves a distributed
checkpoint plus `metadata.json` to `trained_models/`.

### `model_repository/hstu_model/1/model.py`
Triton Python backend:
- `initialize()` — reads `checkpoint_path` from parameters, sets up the distributed
  process group, loads weights from the checkpoint.
- `execute()` — receives `indices` (INT64, `[batch, num_features]`), builds a
  `KeyedJaggedTensor`, runs a forward pass, returns `predictions` (FP32, `[batch]`).
- `finalize()` — destroys the process group.

---

## References

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Triton Python Backend](https://github.com/triton-inference-server/python_backend)
- [PyNVE Documentation](../../README.md)
