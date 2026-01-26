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

#include "gtest/gtest.h"
#include <datagen.h>
#include <memory>
#include <algorithm>
#include <thread>
#include <numeric>
#include "embedding_cache_combined.h"
#include "embedding_cache_combined.cuh"
#include "cuda_ops/cuda_utils.cuh"
#include "../common/buffer.h"
#include <default_allocator.hpp>

using namespace nve;

template <typename T>
class SortAndInsertTest : public ::testing::Test {
  public:
    typedef typename T::KeyType KeyType;
    typedef typename T::TagType TagType;

    using CounterType = float;
    using CacheType = EmbedCacheSA<KeyType, TagType>;
    using ModifyEntry = typename CacheType::ModifyEntry;
    using ModifyList = typename CacheType::ModifyList;
    static const uint32_t NUM_WAYS = CacheType::NUM_WAYS;

    struct key_data {
        key_data(float pr, uint32_t loc, bool h) : priority(pr), location(loc), hit(h) {}
        float priority;
        uint32_t location;
        bool hit;
    };

    uint32_t SetupAndRunTest(std::shared_ptr<Buffer<TagType>>& tags,
                             std::shared_ptr<Buffer<CounterType>>& counters,
                             std::shared_ptr<Buffer<KeyType>>& unique_keys,
                             std::shared_ptr<Buffer<CounterType>>& priorities,
                             std::shared_ptr<Buffer<uint64_t>>& data_ptrs,
                             std::shared_ptr<Buffer<ModifyList>>& replace_list,
                             size_t extra_mem_size, int8_t* extra_mem_buf,
                             uint32_t num_sets, size_t embedding_size, float decay_rate,
                             uint32_t tags_per_set, uint32_t tags_offset,
                             CounterType priority, CounterType priority_step = 0,
                             uint32_t set_step = 1) {
        uint32_t actual_num_keys = 0;
        for (uint32_t i=0 ; i < num_sets; i += set_step) {
            CounterType curr_priority = priority;
            for (uint32_t j=0 ; j < tags_per_set; j++) {
                KeyType index = static_cast<KeyType>(j + tags_offset + 1) * num_sets + i;
                unique_keys->ph[actual_num_keys] = index;
                priorities->ph[actual_num_keys++] = curr_priority;
                curr_priority += priority_step;
            }
        }

        unique_keys->HtoD(0);
        priorities->HtoD(0);
        replace_list->ph[0].num_entries = 0;
        replace_list->HtoD(0);

        CHECK_CUDA_ERROR((ComputeSetReplaceData<KeyType, TagType, CounterType, NUM_WAYS, true>(
            reinterpret_cast<const int8_t* const*>(data_ptrs->pd),
            unique_keys->pd,
            extra_mem_buf,
            extra_mem_size,
            actual_num_keys,
            priorities->pd,
            tags->pd,
            counters->pd,
            nullptr, // cache ptr participates in dst address compute only
            embedding_size,
            decay_rate,
            num_sets,
            NUM_WAYS * num_sets,
            replace_list->pd,
            0)));

        replace_list->DtoH(0);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return replace_list->ph[0].num_entries;
    }

    void AllocateExtraMem(uint32_t num_sets, uint32_t num_keys, size_t embedding_size, size_t& extra_mem_size, int8_t** extra_mem_buf) {
        CHECK_CUDA_ERROR((ComputeSetReplaceData<KeyType, TagType, CounterType, NUM_WAYS, true>(
            nullptr,
            nullptr,
            nullptr,
            extra_mem_size,
            num_keys,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            embedding_size,
            0,
            num_sets,
            num_keys,
            nullptr,
            0)));

        CHECK_CUDA_ERROR(cudaMalloc(extra_mem_buf, extra_mem_size));
    }

    void InitRandomTestInputs(uint32_t num_keys,
                             std::shared_ptr<Buffer<KeyType>>& unique_keys,
                             std::shared_ptr<Buffer<CounterType>>& priorities,
                             std::shared_ptr<Buffer<uint64_t>>& data_ptrs,
                             std::mt19937& genr) {
        std::uniform_real_distribution<float> dist_float(0.1f, 17.0f);
        std::uniform_int_distribution<uint32_t> dist_keys(1, num_keys * 20);
        std::uniform_int_distribution<uint64_t> dist_data_ptrs(1, 0xfedc0000);

        std::set<KeyType> keys;

        for (uint32_t i=0 ; i < num_keys; i++) {
            priorities->ph[i] = dist_float(genr);
            data_ptrs->ph[i] = dist_data_ptrs(genr);
        }

        while (keys.size() < num_keys) {
            KeyType new_key = dist_keys(genr);
            if (keys.find(new_key) == keys.end()) {
                unique_keys->ph[keys.size()] = new_key;
                keys.insert(new_key);
            }
        }
        unique_keys->HtoD(0);
        priorities->HtoD(0);
        data_ptrs->HtoD(0);
    }

    void InitRandomTestCountersAndTags(uint32_t num_sets,
                                std::shared_ptr<Buffer<CounterType>>& counters,
                                std::shared_ptr<Buffer<TagType>>& tags,
                                std::vector<uint32_t>& num_hits,
                                std::mt19937& genr) {
        std::uniform_real_distribution<float> dist_float(0.1f, 17.0f);

        // reset counters, so that new values reflect insert history
        // all free entries should have low, but different counters
        // and tags that are valid, but not causing hits
        memset(counters->ph, 0, num_sets * NUM_WAYS * sizeof(CounterType));

        // unique counters in 'miss' entries for well defined replacement order
        for (uint32_t i = 0 ; i < num_sets; i++) {
            CounterType val = 0.7f;
            for (uint32_t j = 0; j < num_hits[i]; j++) {
                counters->ph[i * NUM_WAYS + j] = dist_float(genr) + 0.8f;
            }
            for (uint32_t j = num_hits[i]; j < NUM_WAYS; j++) {
                counters->ph[i * NUM_WAYS + j] = val;
                tags->ph[i * NUM_WAYS + j] = -2;
                val -= 0.1f;
            }
        }
        counters->HtoD(0);
        tags->HtoD(0);
    }

    void LaunchSortTest(uint32_t num_sets) {
        auto counters = std::make_shared<Buffer<CounterType>>(num_sets * NUM_WAYS * sizeof(CounterType));
        auto sorted_sets = std::make_shared<Buffer<KeyType>>(num_sets * NUM_WAYS * sizeof(KeyType));
        auto sorted_sets_ref = std::make_shared<Buffer<KeyType>>(num_sets * NUM_WAYS * sizeof(KeyType));

        std::vector<CounterType> counters_tmp(num_sets * NUM_WAYS);
        // this test is called twice for each KeyType, with dofferent TagType
        // incorporating TagType into the seed to make the tests different
        std::mt19937 genr(0X753812 + 11 * sizeof(TagType));
        std::uniform_real_distribution<float> dist_float(0.1f, 17.0f);
        for (uint32_t i=0 ; i < num_sets * NUM_WAYS; i++) {
            counters->ph[i] = dist_float(genr);
            counters_tmp[i] = counters->ph[i];
        }
        // sort ref
        for (uint32_t i = 0 ; i < num_sets; i++) {
            for (uint32_t j = 0 ; j < NUM_WAYS; j++) {
                // compute j-th element of the res
                CounterType res_el = counters_tmp[i * NUM_WAYS];
                KeyType pos = 0;
                for (uint32_t k = 1 ; k < NUM_WAYS; k++) {
                    if (counters_tmp[i * NUM_WAYS + k] > res_el) {
                        res_el = counters_tmp[i * NUM_WAYS + k];
                        pos = k;
                    }
                }
                sorted_sets_ref->ph[i * NUM_WAYS + j] = pos;
                counters_tmp[i * NUM_WAYS + pos] = -1;
            }
        }

        counters->HtoD(0);
        CallSortKernel<KeyType, CounterType, NUM_WAYS>(counters->pd, sorted_sets->pd, num_sets);
        sorted_sets->DtoH(0);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // compare
        EXPECT_TRUE(std::equal(sorted_sets->ph, sorted_sets->ph + num_sets * NUM_WAYS, sorted_sets_ref->ph));
    }

    void LaunchSyntheticInsertTests(uint32_t num_sets, size_t embedding_size, float decay_rate) {
        auto tags = std::make_shared<Buffer<TagType>>(num_sets * NUM_WAYS * sizeof(TagType));
        auto counters = std::make_shared<Buffer<CounterType>>(num_sets * NUM_WAYS * sizeof(CounterType));

        auto priorities = std::make_shared<Buffer<CounterType>>(num_sets * NUM_WAYS * sizeof(CounterType));
        auto unique_keys = std::make_shared<Buffer<KeyType>>(num_sets * NUM_WAYS * sizeof(KeyType));

        // the type of data_ptrs doesn't really matter, generating as uint64_t
        auto data_ptrs = std::make_shared<Buffer<uint64_t>>(num_sets * NUM_WAYS * sizeof(uint64_t));

        auto replace_entries = std::make_shared<Buffer<ModifyEntry>>(num_sets * NUM_WAYS * sizeof(ModifyEntry));
        auto replace_list = std::make_shared<Buffer<ModifyList>>(sizeof(ModifyList));

        // allocate extra mem according to max of NUM_WAYS * num_sets keys
        size_t extra_mem_size;
        int8_t* extra_mem_buf;
        AllocateExtraMem(num_sets, num_sets * NUM_WAYS, embedding_size, extra_mem_size, &extra_mem_buf);

        CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(counters->pd, 0, num_sets * sizeof(TagType) * NUM_WAYS, 0));
        
        // these tests do not change tags, just compute replacements
        // if tags are set to invalid they remain invalid and impact test results
        // initializing tags to a value that will not be used in the tests
        CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(tags->pd, 0x40, num_sets * sizeof(TagType) * NUM_WAYS, 0));

        replace_list->ph[0].entries = replace_entries->pd;

        uint32_t tags_per_set = 3;
        uint32_t tags_offset = 0;
        do {
            uint32_t num_replacements_device = 
                SetupAndRunTest(tags, counters, unique_keys, priorities, data_ptrs, replace_list,
                                extra_mem_size, extra_mem_buf,
                                num_sets, embedding_size, decay_rate, tags_per_set, tags_offset, 50);
            EXPECT_TRUE(num_replacements_device == (num_sets * tags_per_set));

            tags_offset += tags_per_set;
            tags_per_set = std::min(tags_per_set, NUM_WAYS - tags_offset);
        } while (tags_offset < NUM_WAYS);

        // try to insert lower priority keys
        uint32_t num_replacements_device = 
            SetupAndRunTest(tags, counters, unique_keys, priorities, data_ptrs, replace_list,
                            extra_mem_size, extra_mem_buf,
                            num_sets, embedding_size, decay_rate, NUM_WAYS, 10, 10);
        EXPECT_TRUE(num_replacements_device == 0);

        // insert higher priority keys
        num_replacements_device = 
            SetupAndRunTest(tags, counters, unique_keys, priorities, data_ptrs, replace_list,
                            extra_mem_size, extra_mem_buf,
                            num_sets, embedding_size, decay_rate, NUM_WAYS, 10, 1000);
        EXPECT_TRUE(num_replacements_device == (NUM_WAYS * num_sets));
               
        // clear cache and insert low priority again
        CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(tags->pd, INVALID_IDX, num_sets * sizeof(TagType) * NUM_WAYS, 0));

        num_replacements_device = 
            SetupAndRunTest(tags, counters, unique_keys, priorities, data_ptrs, replace_list,
                            extra_mem_size, extra_mem_buf,
                            num_sets, embedding_size, decay_rate, NUM_WAYS, 20, 10);
        EXPECT_TRUE(num_replacements_device == (NUM_WAYS * num_sets));

        CACHE_CUDA_ERR_CHK_AND_THROW(cudaMemsetAsync(tags->pd, 0x40, num_sets * sizeof(TagType) * NUM_WAYS, 0));

        // Insert with priorities 6,7,...13, whem current counters are 10 * 0.95
        num_replacements_device = 
            SetupAndRunTest(tags, counters, unique_keys, priorities, data_ptrs, replace_list,
                            extra_mem_size, extra_mem_buf,
                            num_sets, embedding_size, decay_rate, NUM_WAYS, 40, 6, 1);
        EXPECT_TRUE(num_replacements_device == (4 * num_sets));

        // Insert NUM_WAYS * 2 elements to every 4th set
        num_replacements_device = 
            SetupAndRunTest(tags, counters, unique_keys, priorities, data_ptrs, replace_list,
                            extra_mem_size, extra_mem_buf,
                            num_sets, embedding_size, decay_rate, NUM_WAYS * 2, 60, 15, 0, 4);
        EXPECT_TRUE(num_replacements_device == (NUM_WAYS * ((num_sets + 3) / 4)));

        CHECK_CUDA_ERROR(cudaFree(extra_mem_buf));
    }

    void LaunchRandomInsertTest(uint32_t num_sets, size_t embedding_size, float decay_rate) {
        const uint32_t num_keys = num_sets * (NUM_WAYS - 1);

        auto tags = std::make_shared<Buffer<TagType>>(num_sets * NUM_WAYS * sizeof(TagType));
        auto counters = std::make_shared<Buffer<CounterType>>(num_sets * NUM_WAYS * sizeof(CounterType));

        auto priorities = std::make_shared<Buffer<CounterType>>(num_keys * sizeof(CounterType));
        auto unique_keys = std::make_shared<Buffer<KeyType>>(num_keys * sizeof(KeyType));

        // the type of data_ptrs doesn't really matter, generating as uint64_t
        auto data_ptrs = std::make_shared<Buffer<uint64_t>>(num_sets * NUM_WAYS * sizeof(uint64_t));

        auto replace_entries = std::make_shared<Buffer<ModifyEntry>>(num_sets * NUM_WAYS * sizeof(ModifyEntry));
        auto replace_list = std::make_shared<Buffer<ModifyList>>(sizeof(ModifyList));

        std::mt19937 genr(0X753812);
        InitRandomTestInputs(num_keys, unique_keys, priorities, data_ptrs, genr);

        // reference mimics kernel logic for ease of comparison

        // decide on hit/miss keys, set tags in hit positions
        std::vector<std::vector<key_data>> set_keys(num_sets);
        std::vector<uint32_t> num_hits(num_sets, 0);
        std::bernoulli_distribution distrib(0.5); // for hit decision

        for (uint32_t i = 0 ; i < num_keys; i++) {
            KeyType index = unique_keys->ph[i];
            uint32_t set = uint32_t(index % num_sets);
            TagType tag = TagType(index / num_sets);
            if (distrib(genr) && (num_hits[set] < NUM_WAYS)) {
                // hit
                uint32_t pos = num_hits[set]++;
                tags->ph[set * NUM_WAYS + pos] = tag;
                set_keys[set].emplace_back(priorities->ph[i], i, true);
            } else {
                set_keys[set].emplace_back(priorities->ph[i], i, false);
            }
        }

        // complete tags and counters for the test
        InitRandomTestCountersAndTags(num_sets, counters, tags, num_hits, genr);

        // counters already copied to device, so we can compute in place 
        for (uint32_t i=0 ; i < num_sets * NUM_WAYS; i++) {
            counters->ph[i] *= decay_rate;
        }

        for (uint32_t i = 0 ; i < num_sets; i++) {
            uint32_t pos = 0;
            for (uint32_t j = 0 ; j < set_keys[i].size(); j++) {
                if ((set_keys[i][j].hit == true) && (pos < NUM_WAYS)) {
                    // hit handling
                    counters->ph[i * NUM_WAYS + pos] += set_keys[i][j].priority;
                    ++pos;
                }
            }
        }

        std::vector<ModifyEntry> replace_entries_ref;
        for (uint32_t i = 0 ; i < num_sets; i++) {
            std::vector<CounterType> set_counters;
            std::vector<uint32_t> set_positions(NUM_WAYS);
            for (uint32_t j = 0 ; j < NUM_WAYS; j++) {
                set_positions[j] = j;
            }
            // sort set counters, but need to keep original ways
            for (uint32_t j = 0 ; j < NUM_WAYS; j++) {
                set_counters.push_back(counters->ph[i * NUM_WAYS + j]);
                // bubble sort
                for (size_t k = set_counters.size() - 1; k > 0; k--) {
                    if (set_counters[k] > set_counters[k-1]) {
                        std::swap(set_counters[k], set_counters[k-1]);
                        std::swap(set_positions[k], set_positions[k-1]);
                    }
                }
            }

            // sort keys by priority, keep track of locations
            // give misses priority over hits
            std::sort(set_keys[i].begin(), set_keys[i].end(),
                [](key_data const & a, key_data const & b) -> bool
                { return (a.hit != b.hit) ? (a.hit == false) : (a.priority > b.priority); } );

            // find replacements
            std::vector<TagType> replace_tags(NUM_WAYS, -1);
            std::vector<uint32_t> replace_locations(NUM_WAYS);

            uint32_t rep_loc = NUM_WAYS-1;
            for (uint32_t j = 0 ; j < set_keys[i].size(); j++) {
                uint32_t key_loc = set_keys[i][j].location;
                KeyType index = unique_keys->ph[key_loc];

                if (set_keys[i][j].priority > set_counters[rep_loc]) {
                    uint32_t way = set_positions[rep_loc];
                    replace_tags[way] = TagType(index / num_sets);
                    replace_locations[way] = key_loc;

                    if (rep_loc == 0) break;
                    rep_loc--;
                }
            }

            for (uint32_t j = 0 ; j < NUM_WAYS; j++) {
                if (replace_tags[j] != -1) {
                    for (uint32_t k = 0 ; k < NUM_WAYS; k++) {
                        if (tags->ph[i * NUM_WAYS + k] == replace_tags[j]) {
                            // hit, no need to write, swap loc
                            replace_tags[j] = replace_tags[k];
                            replace_tags[k] = -1;
                            break;
                        }
                    }
                }
            }

            for (uint32_t j = 0 ; j < NUM_WAYS; j++) {
                if (replace_tags[j] != -1) {
                    uint32_t key_loc = replace_locations[j];
                    replace_entries_ref.push_back(ModifyEntry{
                            .src = reinterpret_cast<const int8_t*>(data_ptrs->ph[key_loc]),
                            .dst = reinterpret_cast<int8_t*>((i * NUM_WAYS + j) * embedding_size),
                            .set = i,
                            .way = j,
                            .tag = replace_tags[j]
                        });
                }
            }
        }

        // run kernnel
        size_t extra_mem_size;
        int8_t* extra_mem_buf;
        AllocateExtraMem(num_sets, num_keys, embedding_size, extra_mem_size, &extra_mem_buf);

        CHECK_CUDA_ERROR(cudaMalloc(&extra_mem_buf, extra_mem_size));

        replace_list->ph[0].entries = replace_entries->pd;
        replace_list->ph[0].num_entries = 0; // for atomic
        replace_list->HtoD(0);
        
        CHECK_CUDA_ERROR((ComputeSetReplaceData<KeyType, TagType, CounterType, NUM_WAYS, true>(
            reinterpret_cast<const int8_t* const*>(data_ptrs->pd),
            unique_keys->pd,
            extra_mem_buf,
            extra_mem_size,
            num_keys,
            priorities->pd,
            tags->pd,
            counters->pd,
            nullptr, // cache ptr participates in dst address compute only
            embedding_size,
            decay_rate,
            num_sets,
            num_keys,
            replace_list->pd,
            0)));

        replace_entries->DtoH(0);
        replace_list->DtoH(0);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaFree(extra_mem_buf));

        // compare vs ref
        // num of replacements should match
        // replacements themselves should match, but not necessarily in the same order

        uint32_t num_replacements_device = replace_list->ph[0].num_entries;
        EXPECT_TRUE(num_replacements_device == replace_entries_ref.size());

        std::sort(replace_entries->ph, replace_entries->ph + num_replacements_device,
                [](ModifyEntry const & a, ModifyEntry const & b) -> bool
                { return (a.set > b.set) || ((a.set == b.set) && (a.way > b.way)); } );

        std::sort(replace_entries_ref.begin(), replace_entries_ref.end(),
                [](ModifyEntry const & a, ModifyEntry const & b) -> bool
                { return (a.set > b.set) || ((a.set == b.set) && (a.way > b.way)); } );

        for (uint32_t i = 0; i < num_replacements_device; i++) {
            ModifyEntry& rep_ref = replace_entries_ref[i];
            ModifyEntry& rep_dev = replace_entries->ph[i];

            EXPECT_TRUE (rep_ref.set == rep_dev.set);
            EXPECT_TRUE (rep_ref.way == rep_dev.way);
            EXPECT_TRUE (rep_ref.tag == rep_dev.tag);
            EXPECT_TRUE (rep_ref.src == rep_dev.src);
            EXPECT_TRUE (rep_ref.dst == rep_dev.dst);
        }
    }
};

template <typename KeyT, typename TagT>
struct GpuInsertType {
  typedef KeyT KeyType;
  typedef TagT TagType;
};

typedef ::testing::Types<
                         GpuInsertType<int32_t, int32_t>,
                         GpuInsertType<int32_t, int16_t>,
                         GpuInsertType<int64_t, int32_t>,
                         GpuInsertType<int64_t, int16_t>>
    GpuInsertTestTypes;

TYPED_TEST_SUITE_P(SortAndInsertTest);

TYPED_TEST_P(SortAndInsertTest, TestSortAgainstRef) {
    const uint32_t num_sets = 511;
    this->LaunchSortTest(num_sets);
}

TYPED_TEST_P(SortAndInsertTest, TestInsertCounters) {
    const uint32_t num_sets = 511;
    const size_t embedding_size = 512;
    float decay_rate = 0.95f;
    this->LaunchSyntheticInsertTests(num_sets, embedding_size, decay_rate);
}

TYPED_TEST_P(SortAndInsertTest, TestInsertRandomAgainstRef) {
    const uint32_t num_sets = 511;
    const size_t embedding_size = 507;
    float decay_rate = 0.95f;
    this->LaunchRandomInsertTest(num_sets, embedding_size, decay_rate);
}

REGISTER_TYPED_TEST_SUITE_P(SortAndInsertTest, TestSortAgainstRef, TestInsertCounters, TestInsertRandomAgainstRef);

INSTANTIATE_TYPED_TEST_SUITE_P(EmbeddingCacheGpuInsert,
                               SortAndInsertTest,
                               GpuInsertTestTypes);
