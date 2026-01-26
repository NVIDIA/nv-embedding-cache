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

try:
    from pynve._version import __version__, __version_info__
except ImportError:
    # Fallback: read from version.txt if _version.py doesn't exist
    import os
    version_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'version.txt')
    try:
        with open(version_file) as f:
            __version__ = f.read().strip()
            __version_info__ = tuple(int(x) for x in __version__.split('.'))
    except (FileNotFoundError, ValueError):
        __version__ = "unknown"
        __version_info__ = (0, 0, 0)

__all__ = ['__version__', '__version_info__']
