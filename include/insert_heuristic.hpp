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

#include <random>
#include <nve_types.hpp>
#include <vector>
#include <deque>

namespace nve {


class InsertHeuristic {
  public:
  virtual ~InsertHeuristic() = default;
  virtual bool insert_needed(const float hitrate, const size_t table_id) = 0;
};

// TODO: TRTREC-88
class DefaultInsertHeuristic : public InsertHeuristic {
  public:
  DefaultInsertHeuristic(const std::vector<float>& thresholds); 
  ~DefaultInsertHeuristic() override = default;
  
  bool insert_needed(const float hitrate, const size_t table_id) override;

  private:
  const std::vector<float> thresholds_;
  std::mt19937 gen_;  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis_;
  static constexpr uint64_t seed_ = 98437598437ULL;
};

// TODO: TRTREC-88
class FSMInsertHeuristic :  public InsertHeuristic
{
public:
    FSMInsertHeuristic(const std::vector<float>& thresholds);
    ~FSMInsertHeuristic() override = default;
    bool insert_needed(const float hitrate, const size_t table_id) override;
private:
    enum class State : uint64_t
    {
        Start,
        Steady,
        
        NumStates, // must be last
    };
    std::vector<float> prev_hitrate_;
    std::vector<float> threshold_;
    std::vector<State> state_;
};  

// This heuristic decides whether to insert based on the probability of the hitrate being outside [mean - k * std, mean + k * std] interval.
class StatisticalInsertHeuristic : public InsertHeuristic
{
public:
    // parameters:
    // num_keys: the number of keys in a search batch.
    // k_factor: k_factor[i] is the factor of the standard deviation distance for table i. Empirically, 2.6 is a good value.
    // window_size: the size of the window of the hitrates to calculate the mean and variance from.
    // num_inserts_needed: the number of insertions to make before the state is considered stable.
    // max_unsteady_samples: the maximum number of unsteady samples in a row to allow before the state is considered unstable.
    StatisticalInsertHeuristic(const size_t num_keys, const std::vector<float>& k_factor, const size_t window_size = 14, const size_t num_inserts_needed = 50, const size_t max_unsteady_samples = 3);
    ~StatisticalInsertHeuristic() override = default;
    bool insert_needed(const float hitrate, const size_t table_id) override;

private:
    enum class State : uint64_t
    {
        Start,
        Insert,
        Steady,
        Unstable,
        NumStates, // must be last
    };
    
    // Helper methods
    void updateDistributionParams(const float hitrate, const size_t table_id, const size_t num_keys);
    float getChebyshevProb(const size_t table_id);
    
    // Configuration parameters (immutable)
    const size_t num_keys_;
    const std::vector<float> k_factor_;
    const size_t window_size_;
    const size_t num_inserts_needed_;
    const size_t max_unsteady_samples_;
    
    // Statistical tracking per table
    std::vector<std::deque<float>> hitrate_window_;
    std::vector<float> hitrate_mean_;
    std::vector<float> hitrate_var_;
    std::vector<float> chebyshev_probability_;
    
    // State machine per table
    std::vector<State> state_;
    std::vector<size_t> window_samples_collected_;
    std::vector<size_t> insertions_performed_;
    std::vector<size_t> consecutive_unsteady_samples_;
    
    // Random number generation
    std::mt19937 gen_;  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis_;
    
    // Constants
    static constexpr uint64_t seed_ = 98437598437ULL;
};

}  // namespace nve
