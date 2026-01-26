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

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
from glob import glob
import distutils.command.build

# Environment variables:
# - PYNVE_BUILD_DIR: Build directory path (default: 'build')
# - PYNVE_BUILD_SAMPLES: Set to '1' to build samples/tests (default: disabled)
# - PYNVE_DISABLE_AVX512: Set to disable AVX512 in CI builds
_build_dir = os.environ.get('PYNVE_BUILD_DIR', 'build')


def get_version():
    """Read version from version.txt file."""
    version_file = "version.txt"
    try:
        with open(version_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0"  # Fallback version
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = _build_dir

class CMakeBuild(build_ext):
    def run(self):
        # Run CMake build
        try:
            # Cmake configure and build
            cmake_args = ['cmake',
                          '-B', _build_dir,
                          '-DCMAKE_BUILD_TYPE=Release',
                          '-DNVE_ORIGIN_RPATH=1',
                          '--fresh']

            # Disable tests/samples by default, unless PYNVE_BUILD_SAMPLES=1
            if os.environ.get('PYNVE_BUILD_SAMPLES') != '1':
                cmake_args.append('-DNVE_DISABLE_TESTS_AND_SAMPLES=1')

            # In CI disable AVX512
            if (os.environ.get('PYNVE_DISABLE_AVX512')):
                cmake_args.append('-DNVE_DISABLE_AVX512=1')

            subprocess.check_call(cmake_args)
            build_threads = min(os.cpu_count(), 32)
            subprocess.check_call(['cmake', '--build', _build_dir, f'-j{build_threads}'])
        except subprocess.CalledProcessError:
            raise Exception("Failed building pynve") 
        
        super().run()
    
        # Copy binaries
        self.copy_binaries()
        
        # Manually remove unneeded .so
        for filename in glob(f'{_build_dir}/lib.linux-*/*.so', recursive=True):
            subprocess.check_call(['rm', filename])

    def copy_binaries(self):
        # Specify the source and destination directories
        src_dir = f"{_build_dir}/lib"
        dest_dir = f"{self.build_lib}/pynve"

        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Copy binary files
        for filename in os.listdir(src_dir):
            if filename.endswith('.so') and not ("sample" in filename):
                src_file = os.path.join(src_dir, filename)
                dest_file = os.path.join(dest_dir, filename)
                self.copy_file(src_file, dest_file)

        # Generate _version.py from version.txt
        version = get_version()
        version_dest = os.path.join(dest_dir, "_version.py")
        version_content = f"""# Generated from version.txt
__version__ = "{version}"
__version_info__ = tuple(int(x) for x in "{version}".split('.'))
"""
        with open(version_dest, 'w') as f:
            f.write(version_content)

setup(
        name='pynve',
        version=get_version(),
        package_dir={"": "python"},
        packages=['pynve', 'pynve.torch'],
        ext_modules=[Extension('pynve', sources=[])],
        cmdclass={
            'build_ext': CMakeBuild,
            'build': BuildCommand,
        },
        zip_safe=False,
        )

