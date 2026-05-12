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

#pragma once
#include <stdint.h>
#include <type_traits>

// Marker so these helpers are usable from both nvcc-compiled (.cu/.cuh) and
// host-only (.cpp) translation units. Host C++ compilers don't recognise
// __host__/__device__, so gate them behind __CUDACC__.
#ifdef __CUDACC__
#define NVE_CUDA_CALLABLE __host__ __device__
#else
#define NVE_CUDA_CALLABLE
#endif

namespace nve {

// Centralized cache-set hashing. Cast through the unsigned counterpart so a
// signed negative key doesn't yield a negative set index (which would OOB
// when used as an array offset).
template<typename KeyT, typename SizeT>
NVE_CUDA_CALLABLE inline uint32_t embed_cache_hash_set_idx(KeyT key, SizeT num_sets)
{
    using UKeyT = typename std::make_unsigned<KeyT>::type;
    return static_cast<uint32_t>(static_cast<UKeyT>(key) % num_sets);
}

// Inverse of the hash for the tag bits. Casting through unsigned keeps it in
// lockstep with embed_cache_hash_set_idx so construct_key/construct_tag/hash
// round-trip for any in-range key.
template<typename TagT, typename KeyT, typename SizeT>
NVE_CUDA_CALLABLE inline TagT embed_cache_construct_tag(KeyT key, SizeT num_sets)
{
    using UKeyT = typename std::make_unsigned<KeyT>::type;
    return static_cast<TagT>(static_cast<UKeyT>(key) / num_sets);
}

// Reconstructs a key from a stored tag and a set index. KeyT must be specified
// explicitly because it can't be deduced from the argument types.
template<typename KeyT, typename TagT, typename SetT, typename SizeT>
NVE_CUDA_CALLABLE inline KeyT embed_cache_construct_key(TagT tag, SetT set_idx, SizeT num_sets)
{
    using UKeyT = typename std::make_unsigned<KeyT>::type;
    return static_cast<KeyT>(static_cast<UKeyT>(tag) * num_sets + set_idx);
}

} // namespace nve
