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

"""Generate explicit-instantiation sources for the find_and_combine kernel.

The find_and_combine kernel (cuda_ops/find_and_combine_kernel.cuh) is instantiated across a large
combination space. Compiling the whole matrix in the single translation unit that uses it
(src/gpu_table.cu) dominates the build time of nve-common.

This script enumerates every combination the dispatch logic can launch and emits:

  * find_and_combine_kernel_extern.inc   - `extern template` declaration of nve_fac_kernel per
                                           combination, included by find_and_combine_dispatch.cu so
                                           it calls them instead of instantiating the kernels.
  * find_and_combine_kernel_inst_NNN.cu  - explicit instantiation definitions, split across NUM_TUS
                                           files so the matrix compiles in parallel instead of in
                                           one TU.
  * find_and_combine_dispatch_extern.inc - `extern template` declaration of call_find_and_combine_kernel
                                           per IDX_CACHE entry, included at the end of the header.
  * find_and_combine_dispatch.cu         - the matching explicit instantiations; the only TU that
                                           calls the (hidden) nve_fac_kernel functions.
  * find_and_combine_generated.cmake     - sets FAC_GENERATED_SOURCES to the list of .cu files.

The enumeration mirrors call_find_and_combine_kernel_types_resolved / NVE_FAC_DISPATCH /
NVE_FAC_SELECT_MASK; keep it in sync if those change.
"""

import argparse
import os

# --- tunables -----------------------------------------------------------------------------------
# Number of generated .cu translation units the instantiation matrix is split across. Every TU pays
# a fixed ~8 s to parse the headers, so a few large TUs build fastest, as long as each one finishes
# before the slowest hand-written nve-common TUs: 8 for the 2,880 instantiations of the default
# matrix, 32 for the 11,520 of --specialize-flags. Override with --num-tus (or -DFAC_NUM_TUS).
NUM_TUS = 8
NUM_TUS_SPECIALIZED = 32

# --- combination space (mirrors the dispatch logic) ---------------------------------------------
ELEMENT_TYPES = [
    "Float32", "Float16",
    "QUint8RowwiseF32", "QInt8RowwiseF32",
    "QUint8RowwiseF16", "QInt8RowwiseF16",
]

# (ACC_TYPE, WEIGHT_TYPE) dispatched in call_find_and_combine_kernel. Only Float32 accumulation is
# supported; the accumulator pairs with either a float or a __half weight.
ACC_WEIGHT = [("float", "float"), ("float", "__half")]

OUTPUT_TYPES = ["float", "__half"]

# (INDEX_TYPE, CacheDataT) combinations nve-common dispatches; also the exported
# call_find_and_combine_kernel instantiations (find_and_combine_dispatch.cu).
IDX_CACHE = [
    ("int32_t", "typename nve::EmbedCacheSA<int32_t, int32_t>::CacheData"),
    ("int64_t", "typename nve::EmbedCacheSA<int64_t, int64_t>::CacheData"),
    ("int32_t", "typename nve::ECNoCache<int32_t>::CacheData"),
    ("int64_t", "typename nve::ECNoCache<int64_t>::CacheData"),
]

# SZ_ACCUM values instantiated per vector width (Vec1 / Vec2 / Vec4, selected by rowSizeInElements
# alignment), dispatched by call_find_and_combine_kernel_types_resolved.
#
# The kernel loops over the row in chunks of SUBWARP_WIDTH * SZ_ACCUM vectors, so any row length is
# handled regardless of SZ_ACCUM: SZ_ACCUM only trades register pressure against the number of
# passes. The dispatch picks the smallest SZ_ACCUM that covers the row in a single pass, clamped per
# width to min(4, 8 // width) so that vec_width * SZ_ACCUM <= 8 (bounded register pressure) and
# SZ_ACCUM <= 4. We instantiate exactly that clamped range per width; keep this in sync with the
# per-width clamp in the kernel header (call_find_and_combine_kernel_types_resolved).
MAX_SZ_ACCUM = 4
WIDTH_ACCUM = [(width, sz)
               for width in (4, 2, 1)
               for sz in range(1, min(MAX_SZ_ACCUM, 8 // width) + 1)]

# Packed MASK values. The kernel unpacks the bits as bit0 = SUM_POOLING, bit1 = IS_WEIGHTED,
# bit2 = FIXED_HOTNESS, bit3 = LOAD_INDICES (see NVE_FAC_MASK_* in
# cuda_ops/find_and_combine_kernel.cuh). Bits 2-3 are only used with --specialize-flags
# (NVE_FAC_SPECIALIZE_FLAGS); otherwise they are kernel arguments.
MASKS = list(range(4))
MASKS_SPECIALIZED = list(range(16))
MASK_IS_WEIGHTED = 1 << 1

# Unweighted kernels never read the weights, so the dispatch launches them with this weight type
# whatever the caller's weight type is (nve_fac_weight_t in the kernel header).
UNWEIGHTED_WEIGHT_TYPE = "float"

LICENSE_HEADER = """\
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// GENERATED FILE - DO NOT EDIT.
// Produced by cmake/gen_find_and_combine_instantiations.py.
"""


def combinations(masks):
    """Yield (element, idx, acc, wgt, out, cache, width, sz, mask) for every dispatched kernel."""
    seen = set()
    for element in ELEMENT_TYPES:
        for acc, wgt in ACC_WEIGHT:
            for out in OUTPUT_TYPES:
                for idx, cache in IDX_CACHE:
                    for width, sz in WIDTH_ACCUM:
                        for mask in masks:
                            # Same mapping as nve_fac_weight_t: unweighted kernels use one weight type.
                            mapped_wgt = wgt if mask & MASK_IS_WEIGHTED else UNWEIGHTED_WEIGHT_TYPE
                            combo = (element, idx, acc, mapped_wgt, out, cache, width, sz, mask)
                            if combo not in seen:
                                seen.add(combo)
                                yield combo


def template_args(combo):
    """Return the angle-bracket template argument list for nve_fac_kernel<...>."""
    element, idx, acc, wgt, out, cache, width, sz, mask = combo
    elem_id = f"DataType_t::{element}"
    input_vec = f"typename QuantizationHelper<{elem_id}>::Vec{width}"
    out_vec = f"typename VecWidthHelper<{out}>::Vec{width}"
    elem_vec = f"typename VecWidthHelper<typename QuantizationHelper<{elem_id}>::ParamType>::Vec{width}"
    acc_vec = f"typename VecWidthHelper<{acc}>::Vec{width}"
    return (f"{elem_id}, {idx}, {acc}, {wgt}, {out}, {input_vec}, {out_vec}, {elem_vec}, {acc_vec}, "
            f"{cache}, {sz}u, {mask}u")


def instantiation(combo, extern):
    """Return one explicit-instantiation (or extern declaration) statement for a combination.

    Instantiates the nve_fac_kernel host function (not the __global__ kernel directly): a plain host
    function links across TUs without relocatable device code, and taking the kernel's address in it
    co-locates the kernel's device-code instantiation in the same TU.
    """
    _element, idx, _acc, wgt, out, cache, _width, _sz, _mask = combo
    prefix = "extern template" if extern else "template"
    return (f"{prefix} nve_fac_kernel_t<{idx}, {wgt}, {out}, {cache}>\n"
            f"    nve_fac_kernel<{template_args(combo)}>();")


def dispatch_instantiation(idx, cache, extern):
    """Return the explicit-instantiation (or extern declaration) of call_find_and_combine_kernel."""
    prefix = "extern template" if extern else "template"
    return (
        f"{prefix} void call_find_and_combine_kernel<{idx}, {cache}>(\n"
        f"    const uint32_t, const uint32_t, const int8_t*, const {idx}*, const {idx}*, const void*,\n"
        f"    int32_t, {cache}, int32_t, SparseType_t, PoolingType_t, bool, DataType_t, DataType_t,\n"
        f"    DataType_t, DataType_t, void*, cudaStream_t);"
    )


def write_if_different(path, content):
    """Write content only when it differs, to avoid triggering needless recompiles."""
    if os.path.exists(path):
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Directory for generated files.")
    parser.add_argument("--num-tus", type=int, default=None,
                        help=f"Number of generated .cu files to split the matrix across (default "
                             f"{NUM_TUS}, or {NUM_TUS_SPECIALIZED} with --specialize-flags).")
    parser.add_argument("--specialize-flags", action="store_true",
                        help="Also instantiate FIXED_HOTNESS / LOAD_INDICES as MASK bits "
                             "(must match NVE_FAC_SPECIALIZE_FLAGS, i.e. "
                             "NVE_OPTIMIZE_BIN_SIZE=OFF).")
    args = parser.parse_args()
    if args.num_tus is None:
        args.num_tus = NUM_TUS_SPECIALIZED if args.specialize_flags else NUM_TUS

    if args.num_tus < 1:
        parser.error("--num-tus must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    combos = list(combinations(MASKS_SPECIALIZED if args.specialize_flags else MASKS))

    # Split the matrix across (at most) num_tus files as evenly as possible. per_file is the ceiling
    # so no file exceeds it; num_files is then recomputed from per_file so we never emit empty
    # trailing TUs (e.g. when num_tus exceeds the combination count).
    num_files = min(args.num_tus, len(combos))
    per_file = (len(combos) + num_files - 1) // num_files
    num_files = (len(combos) + per_file - 1) // per_file

    # extern template declaration header.
    extern_lines = [LICENSE_HEADER,
                    "#pragma once",
                    "",
                    "// extern template declarations of the nve_fac_kernel host function for every",
                    "// dispatched find_and_combine combination.",
                    "",
                    f"static_assert(NVE_FAC_SPECIALIZE_FLAGS == {int(args.specialize_flags)},",
                    '              "NVE_FAC_SPECIALIZE_FLAGS does not match --specialize-flags");',
                    ""]
    extern_lines += [instantiation(c, extern=True) for c in combos]
    extern_lines.append("")
    write_if_different(os.path.join(args.out_dir, "find_and_combine_kernel_extern.inc"),
                       "\n".join(extern_lines))

    # Explicit instantiation .cu files, up to per_file instantiations each.
    width = max(3, len(str(num_files - 1)))
    sources = []
    for fidx in range(num_files):
        chunk = combos[fidx * per_file:(fidx + 1) * per_file]
        name = f"find_and_combine_kernel_inst_{fidx:0{width}d}.cu"
        sources.append(name)
        body = [LICENSE_HEADER,
                '#include "cuda_ops/find_and_combine_kernel.cuh"',
                "",
                "using namespace nve;",
                ""]
        body += [instantiation(c, extern=False) for c in chunk]
        body.append("")
        write_if_different(os.path.join(args.out_dir, name), "\n".join(body))

    # Dispatcher instantiations: extern declarations for every user of the header, definitions in one
    # TU that includes the nve_fac_kernel extern declarations above.
    dispatch_extern = [LICENSE_HEADER,
                       "#pragma once",
                       "",
                       "// extern template declarations of call_find_and_combine_kernel for every",
                       "// index/cache combination instantiated in nve-common.",
                       ""]
    dispatch_extern += [dispatch_instantiation(idx, cache, extern=True) for idx, cache in IDX_CACHE]
    dispatch_extern.append("")
    write_if_different(os.path.join(args.out_dir, "find_and_combine_dispatch_extern.inc"),
                       "\n".join(dispatch_extern))
    dispatch_body = [LICENSE_HEADER,
                     '#include "cuda_ops/find_and_combine_kernel.cuh"',
                     '#include "find_and_combine_kernel_extern.inc"',
                     "",
                     "using namespace nve;",
                     ""]
    dispatch_body += [dispatch_instantiation(idx, cache, extern=False) for idx, cache in IDX_CACHE]
    dispatch_body.append("")
    write_if_different(os.path.join(args.out_dir, "find_and_combine_dispatch.cu"), "\n".join(dispatch_body))
    sources.append("find_and_combine_dispatch.cu")

    # Remove any stale .cu files from a previous run with a different TU count.
    for existing in os.listdir(args.out_dir):
        if existing.startswith("find_and_combine_kernel_inst_") and existing.endswith(".cu") \
                and existing not in sources:
            os.remove(os.path.join(args.out_dir, existing))

    # CMake fragment listing the generated sources.
    abs_sources = [os.path.join(args.out_dir, s).replace("\\", "/") for s in sources]
    cmake_lines = ["# GENERATED FILE - DO NOT EDIT.",
                   "set(FAC_GENERATED_SOURCES"]
    cmake_lines += [f"  {s}" for s in abs_sources]
    cmake_lines += [")", ""]
    write_if_different(os.path.join(args.out_dir, "find_and_combine_generated.cmake"),
                       "\n".join(cmake_lines))

    print(f"Generated {len(combos)} find_and_combine instantiations across {num_files} TUs "
          f"(<= {per_file}/file) in {args.out_dir}")


if __name__ == "__main__":
    main()
