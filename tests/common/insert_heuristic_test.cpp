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

#include <gtest/gtest.h>
#include <insert_heuristic.hpp>
#include <random>
#include <vector>

static constexpr uint32_t SEED = 42;
static constexpr uint32_t NUM_EXAMPLES = 10;

// this function generates a random hitrate vector with a normal distribution with num_warmup_samples increasing linearly from 0 to maximal_hit_rate with normal noise
// and last steady_state_samples are with maximal_hit_rate + normal noise
std::vector<float> generate_random_hitrate(uint32_t num_warmup_samples, float maximal_hit_rate, uint32_t steady_state_samples, float std_dev, uint32_t seed) {
    std::vector<float> hitrates;
    std::random_device rd;
    std::mt19937 gen(seed);
    float mean = 0.0f;
    std::normal_distribution<float> dis(mean, std_dev);
    if(num_warmup_samples > 0) {
        float delta = maximal_hit_rate / static_cast<float>(num_warmup_samples);
        for (uint32_t i = 0; i <= num_warmup_samples; ++i) {
            hitrates.push_back(std::max(0.0f, static_cast<float>(i)*delta));
        }
    }
    if(steady_state_samples > 0) {
        for (uint32_t i = 0; i < steady_state_samples; ++i) {
            hitrates.push_back(maximal_hit_rate + dis(gen));
        }
    }
    return hitrates;
}


void TestHeuristic(nve::InsertHeuristic& heuristic, uint32_t num_samples, uint32_t steady_state_samples, uint32_t num_examples, double thershold, float std_dev, std::function<double(std::vector<float>)> expected_func, uint32_t seed) {
    auto hitrates = generate_random_hitrate(num_samples, 0.8f, steady_state_samples, std_dev, seed);
    double avg_count = 0;
    for (uint32_t i = 0; i < num_examples; i++) 
    {
        int64_t count = 0;
        for (auto hitrate : hitrates) {
            count += heuristic.InsertNeeded(hitrate, 0) ? 1 : 0;
        }
        avg_count += static_cast<double>(count);
    }
    avg_count /= num_examples;
    double double_expected = expected_func(hitrates);
    EXPECT_NEAR(double_expected, avg_count, thershold*static_cast<double>(hitrates.size()));
}

TEST(DefaultInsertHeuristic, DefaultInsert) {
    auto func = [](std::vector<float> hitrates) {
        double expected = 0;
        for (auto hitrate : hitrates) {
            auto chance = 1-hitrate;
            expected += hitrate < 0.8f ? chance : chance*chance*chance;
        }
        return expected;
    };
    for (uint32_t i = 0; i < NUM_EXAMPLES; i++) {
        nve::DefaultInsertHeuristic heuristic(std::vector<float>{0.8f, 0.8f, 0.8f});
        TestHeuristic(heuristic, 100, 100, 1, 0.02, 0.005f, func, SEED + i);
    }
}

TEST(FSMInsertHeuristic, FSMInsertOnlyWarmup) {
    uint32_t num_warmup = 35;
    uint32_t num_steady = 0;
    float std_dev = 0.005f; // such that we will be in 0.01 for 2 sigma
    
    auto func = [&](std::vector<float> /*hitrates*/) { 
        // 0.05 is the chance to be 2 sigma above the maximal hitrate
        return static_cast<double>(num_warmup) + 0.05 * num_steady;
    };
    for (uint32_t i = 0; i < NUM_EXAMPLES; i++) {
        nve::FSMInsertHeuristic heuristic({2 * std_dev, 2 * std_dev, 2 * std_dev});
        TestHeuristic(heuristic, num_warmup, num_steady, 1, 0.1, std_dev, func, SEED + i);
    }
}

TEST(FSMInsertHeuristic, FSMInsertSteady) {
    uint32_t num_warmup = 40;
    uint32_t num_steady = 100;
    float std_dev = 0.005f; // such that we will be in 0.01 for 2 sigma

    auto func = [&](std::vector<float> /*hitrates*/) { 
        // if we had the normal noise on warmup:
        // for warmup we need to calucluate the change that delta of each step will be in the range of [-threshodl, thershold]
        // delta = max_hitrate / num_steps + normal_noise(0, std_dev)
        // this is equivelent the probability that normal_noise(0, std_dev) will be in the range of [-threshodl + max_hitrate / num_steps, thershold + max_hitrate / num_steps]
        // which is hard to calcualte so im going to assume delta is big enough

        // for steady state we have the normal noise on the hitrate
        // 0.05 is the chance to be 2 sigma above the maximal hitrate
        // bit since each anomaly will probably inccur two inserts for the anamoly and to get back
        // we multiply by 2
        return static_cast<double>(num_warmup) + 2 * 0.05 * num_steady;
    };
    for (uint32_t i = 0; i < NUM_EXAMPLES; i++) {
        nve::FSMInsertHeuristic heuristic({2 * std_dev, 2 * std_dev, 2 * std_dev});
        TestHeuristic(heuristic, num_warmup, num_steady, 1, 0.1, std_dev, func, SEED + i);
    }
}

TEST(StatisticalInsertHeuristic, StatisticalInsertSimpleTest) {
    // Test that the heuristic starts properly and then detects statistical anomalies and triggers insertions
    const size_t lookup_keys = 0; // Not a true cache lookup, variance calculated from hitrates
    const float sigma_threshold = 2.4f;
    const size_t sample_window = 10;
    const size_t num_insertions_needed = 5;
    const size_t num_unsteady_samples = 3;
    const size_t steady_state_iterations = 20;
    const float baseline_hitrate = 0.5f;
    
    nve::StatisticalInsertHeuristic heuristic(lookup_keys, {sigma_threshold}, sample_window, 
                                            num_insertions_needed, num_unsteady_samples);
    
    // Start & Insert states: collect window and perform insertions
    for (size_t i = 0; i < sample_window; ++i) {
        EXPECT_FALSE(heuristic.InsertNeeded(baseline_hitrate, 0));
    }
    for (size_t i = 0; i < num_insertions_needed; ++i) {
        EXPECT_TRUE(heuristic.InsertNeeded(baseline_hitrate, 0));
    }

    // Now in steady state with stable hitrate (0.5f)
    // Should not insert for stable hitrates within the statistical bounds
    for (size_t i = 0; i < steady_state_iterations; ++i) {
        EXPECT_FALSE(heuristic.InsertNeeded(baseline_hitrate, 0));
    }
    
    // Now introduce anomalies
    // For the above hitrate, the mean is baseline_hitrate, and the std_dev is 0. 
    // Any hitrate above baseline_hitrate is an anomaly
    const float anomalous_hitrate = baseline_hitrate + 0.1f;
    for (size_t i = 0; i < num_unsteady_samples; ++i) {
        EXPECT_FALSE(heuristic.InsertNeeded(anomalous_hitrate, 0));
    }
    
    // After num_unsteady_samples consecutive anomalies, should trigger num_insertions_needed insertions
    for (size_t i = 0; i < num_insertions_needed; ++i) {
        EXPECT_TRUE(heuristic.InsertNeeded(baseline_hitrate, 0));
    }
}

TEST(StatisticalInsertHeuristic, StatisticalInsertUnstableRandomHitrates) {
    // Test the statistical heuristic with random hitrates with unstable insertions.
    // With those parameters, we'll expect to have a lot of noise, meaning mainly be in unstable state and to spiral into a loop of insertions.
    const size_t lookup_keys = 0;
    const float sigma_threshold = 2.0;
    const size_t sample_window = 10;
    const size_t num_insertions_needed = 1;
    const size_t num_unsteady_samples = 1;
    const float std_dev = 0.002f;
    const size_t warmup_samples = 0;
    const size_t steady_samples = 150;
   
    nve::StatisticalInsertHeuristic heuristic(lookup_keys, {sigma_threshold}, sample_window, 
                                            num_insertions_needed, num_unsteady_samples);
    auto func = [&](std::vector<float> hitrates) {
        // With one abnormal sample to move from steady to unsteady, we are very unstable. 
        // We'll expect to enter a loop of insertions, meaning looping constantly between 
        // unsteady->collect_window->insert->unsteady -> ..., each loop will have num_insertions_needed inserts.
        // Also, we'll have num_insertions_needed inserts from the warmup.
        return num_insertions_needed + static_cast<double>(hitrates.size()) * (num_insertions_needed) / (num_insertions_needed + sample_window + 1);
    };
    // Possibly we'll have additional inserts from steady state, 95% of the data is within 2 sigma, 
    // so we expect to have 5% of the data outside 2 sigma, and perform num_insertions_needed inserts.
    double steady_state_insert_prob = pow(1.0f-0.95, num_unsteady_samples);
    for (uint32_t i = 0; i < NUM_EXAMPLES; i++) {
        TestHeuristic(heuristic, warmup_samples, steady_samples, 1, steady_state_insert_prob * num_insertions_needed, std_dev, func, SEED + i);
    }
}

TEST(StatisticalInsertHeuristic, StatisticalInsertStableRandomHitrates) {
    // Test the statistical heuristic with random hitrates with stable insertions
    // With those parameters, we'll expect to be mainly in stable state and not to have a lot of noise.
    const size_t lookup_keys = 0;
    const float sigma_threshold = 3.0;
    const size_t sample_window = 10;
    const size_t num_insertions_needed = 1;
    const size_t num_unsteady_samples = 2;
    const float std_dev = 0.005f;
    const size_t warmup_samples = 0;
    const size_t steady_samples = 150;
   
    nve::StatisticalInsertHeuristic heuristic(lookup_keys, {sigma_threshold}, sample_window, 
                                            num_insertions_needed, num_unsteady_samples);
    auto func = [&](std::vector<float> /*hitrates*/) { 
        // With k = 3, we expect to have 99.7% of the data in 3 sigma, and in order to perform insert from a stable state, 
        // we need to see 2 consecutive outliers. Meaning- the probability of 2 consecutive outliers is 0.003^2 = 0.000009, very low.
        // Therefore, we expect to have 1 insert during heuristic warmup.
        return num_insertions_needed;
    };
    // We'll allow at most 1 insert during steady state.
    for (uint32_t i = 0; i < NUM_EXAMPLES; i++) {
        TestHeuristic(heuristic, warmup_samples, steady_samples, 1, 1.0f/static_cast<double>(warmup_samples + steady_samples), std_dev, func, SEED + i);
    }
}

