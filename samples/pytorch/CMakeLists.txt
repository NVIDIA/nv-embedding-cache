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

# Pytorch samples are pure-Python; the host-table custom_remote plugin used by
# pytorch_samples_common.get_remote_interface and the export_sample/ scripts is
# built once in samples/common/custom_remote_plugin (libnve-plugin-custom_remote.so)
# and resolved at runtime via load_host_table_plugin("libnve-plugin-custom_remote.so").
