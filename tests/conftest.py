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

import pytest


def is_nvhm_plugin_available():
    """Check if the nvhm plugin is available by attempting to load it."""
    try:
        import torch
        from pynve.torch import nve_ps
        # Try to create a minimal NVLocalParameterServer - this loads the nvhm plugin
        ps = nve_ps.NVLocalParameterServer(0, 1, torch.float32)
        del ps
        return True
    except Exception:
        return False


# Cache the result to avoid repeated checks
_nvhm_available = None


def nvhm_available():
    global _nvhm_available
    if _nvhm_available is None:
        _nvhm_available = is_nvhm_plugin_available()
    return _nvhm_available


# Pytest marker for tests requiring NVHM
requires_nvhm = pytest.mark.skipif(
    not nvhm_available(),
    reason="NVHM plugin is not available"
)
