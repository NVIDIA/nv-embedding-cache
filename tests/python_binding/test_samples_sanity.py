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

"""Smoke tests for the scripts under samples/pytorch/export_sample."""

import sys
import os
import ctypes
import pytest

# Workaround to import the python samples
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{dir_path}/../../samples/pytorch")
sys.path.append(f"{dir_path}/../../samples/pytorch/export_sample")

from conftest import requires_nvhm


def _custom_remote_plugin_available():
    """True iff libnve-plugin-custom_remote.so can be loaded by the dynamic linker."""
    try:
        ctypes.CDLL("libnve-plugin-custom_remote.so")
        return True
    except OSError:
        pass
    # Fall back to the pynve package directory (where pip install drops the .so)
    try:
        import pynve
        plugin_path = os.path.join(
            os.path.dirname(pynve.__file__), "libnve-plugin-custom_remote.so")
        if os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path)
            return True
    except OSError:
        pass
    return False


requires_custom_remote = pytest.mark.skipif(
    not _custom_remote_plugin_available(),
    reason="libnve-plugin-custom_remote.so not on LD_LIBRARY_PATH "
           "(build samples/common/custom_remote_plugin)"
)


@requires_nvhm
def test_pytorch_simple_remote_ps_export():
    import simple_remote_ps_export
    simple_remote_ps_export.main()


@requires_custom_remote
def test_pytorch_custom_remote_ps_export():
    import custom_remote_ps_export
    custom_remote_ps_export.main()
