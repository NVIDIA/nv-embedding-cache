# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Triton Server image with PyNVE installed.
#
# Build (run from repo root):
#   docker build -f samples/triton_server_sample/Dockerfile -t hstu_triton .
#
# Run:
#  docker run --gpus all --rm \
#  --net=host \
#  --cap-add=SYS_PTRACE \
#  -v $(pwd)/samples/triton_server_sample/:/triton_server_sample \
#  hstu_triton

FROM nvcr.io/nvidia/tritonserver:26.01-py3

# Install build dependencies for pynve
RUN apt-get update && apt-get install -y \
    libssl-dev

# Copy repo source and build pynve
WORKDIR /workspace/trt-recsys
COPY . .
RUN pip install .

# Install Python dependencies for the model backend
RUN pip install -r samples/triton_server_sample/requirements.txt


ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/triton_server_sample/model_repository"]
