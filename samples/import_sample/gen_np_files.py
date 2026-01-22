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

import numpy as np

keys_path = "/tmp/keys.npy"
values_path = "/tmp/values.npy"

num_keys = 2**20
row_dim = 128
elem_type = np.float32


rng = np.random.default_rng()
keys = rng.integers(0, 2**60, size=num_keys * 2, dtype=np.int64)
keys = np.unique(keys)[:num_keys]
if len(keys) < num_keys:
    raise Exception("Failed to generate enough unique keys")

values = np.random.rand(num_keys, row_dim).astype(elem_type)

np.save(keys_path, keys)
np.save(values_path, values)
