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


"""test the license of source files."""

from pathlib import Path
import subprocess

# pylint:disable=missing-docstring, no-self-use

class TestLicense():

    def test_license(self):
        root = Path(__file__).parent.parent.parent.absolute()
        root_len = len(str(root))

        # Collect files ending with relevant extensions
        file_list = []
        file_types = ['*.py', '*.h', '*.c', '*.hpp', '*.cpp', '*.cuh', '*.cu', '*.sh', 'CMakeLists.txt', '*.toml']
        for ft in file_types:
            file_list += list(root.rglob(ft))

        # Trim files not tracked by git
        trim_files = []
        for f in file_list:
            res = subprocess.run(['git', 'ls-files', '--error-unmatch', f], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0:
                trim_files.append(f)

        for f in trim_files:
            file_list.remove(f)

        file_list = sorted(file_list)
        print (f"Found {len(file_list)} source files")
 
        cmake_header = (root / 'tests' / 'license' / 'license_test_header_cmake.txt').open().readlines()
        cxx_header = (root / 'tests' / 'license' / 'license_test_header_cxx.txt').open().readlines()
        py_header = (root / 'tests' / 'license' / 'license_test_header_py.txt').open().readlines()
        sh_header = (root / 'tests' / 'license' / 'license_test_header_sh.txt').open().readlines()
        ignore_lines = (root / 'tests' / 'license' / 'ignore_lines.txt').open().readlines()

        invalid_files = []
        for f in file_list:
            with open(f) as src_file:
                src_lines = src_file.readlines()

            # Skip empty files
            if len(src_lines) == 0:
                continue

            # Pick correct ref header
            if f.suffix == '.py':
                header = py_header
            elif f.suffix == '.toml':
                header = py_header
            elif f.suffix == '.sh':
                header = sh_header
            elif (f.suffix == '.txt') and (f.stem == 'CMakeLists'):
                header = cmake_header
            else:
                header = cxx_header

            # Trim ignored lines from start
            while  src_lines[0] in ignore_lines:
                del src_lines[0]

            # Trivial reject
            num_lines = len(header)
            if len(src_lines) < num_lines:
                invalid_files.append(f)
                continue

            # Now verify header conforms with license
            for i in range(num_lines):
                if src_lines[i] != header[i]:
                    invalid_files.append(f)
                    break

        if len(invalid_files) > 0:
            for f in invalid_files:
                print(f"The file {f} has an invalid header!")
            raise AssertionError("%d files have invalid headers!" % (len(invalid_files)))
