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

#include "insert_heuristic.hpp"

namespace nve {

DefaultInsertHeuristic::DefaultInsertHeuristic(const std::vector<float>& thresholds)
    : thresholds_(std::move(thresholds)),
    gen_(seed_),
    dis_(0.0, 1.0) {}

bool DefaultInsertHeuristic::insert_needed(const float hitrate, const size_t table_id) {
    auto warmup_chance = (1 - hitrate);
    auto steadystate_chance = (warmup_chance * warmup_chance * warmup_chance);

    NVE_CHECK_(table_id < thresholds_.size(), "Invalid table index");
    return (hitrate < thresholds_.at(static_cast<size_t>(table_id))) ? 
            (dis_(gen_) < warmup_chance) :
            (dis_(gen_) < steadystate_chance);
}

FSMInsertHeuristic::FSMInsertHeuristic(const std::vector<float>& threshold) :
    threshold_(std::move(threshold))
{
    const auto num_tables = threshold.size();
    for(size_t i=0 ; i<num_tables ; i++) {
        prev_hitrate_.push_back(0);
        state_.push_back(State::Start);
    }
}

bool FSMInsertHeuristic::insert_needed(const float hitrate, const size_t table_id)
{
    NVE_CHECK_(table_id < threshold_.size(), "Invalid table index");
    if (threshold_[table_id] <= 0.f) {
        return false; // threshold <= 0, disables auto insert (useful for parameter server)
    }
    switch (state_[table_id])
    {
    case State::Start:
        state_[table_id] = State::Steady;
        prev_hitrate_[table_id] = hitrate;
        return true;
    case State::Steady:
    {
        if (std::abs(hitrate - prev_hitrate_[table_id]) < threshold_[table_id])
        {
            return false;
        }
        else
        {
            prev_hitrate_[table_id] = hitrate;
            return true;
        }
    }
    default:
        NVE_THROW_("Invalid heuristic state!"); // This should not be possible
        return false;
    }
}


float StatisticalInsertHeuristic::getChebyshevProb(const size_t table_id)
{
    // calculation for finite samples
    float term1 = 1.0f / static_cast<float>(window_size_ + 1);
    float term2 = static_cast<float>(window_size_ + 1) / static_cast<float>(window_size_);
    float term3 = (static_cast<float>(window_size_ - 1) / (k_factor_[table_id] * k_factor_[table_id])) + 1;
    return term1 * std::floor(term2 * term3);
}

void StatisticalInsertHeuristic::updateDistributionParams(const float hitrate, const size_t table_id, const size_t num_keys)
{
    // update hitrate window
    float old_hitrate = hitrate_window_[table_id].front();
    hitrate_window_[table_id].pop_front();
    hitrate_window_[table_id].push_back(hitrate);

    // update mean
    float old_mean = hitrate_mean_[table_id];
    float curr_mean = old_mean + (hitrate - old_hitrate) / static_cast<float>(window_size_);
    hitrate_mean_[table_id] = curr_mean;

    // update variance
    // the hitrate is a sample of a binomial distribution, so we can use the formula for variance
    // if we didn't receive the number of keys, we'll use the collected sample's variance formula
    if (num_keys == 0) 
    {
        float curr_var = 0;
        for (size_t i = 0; i < window_size_; i++)
        {
            curr_var += (hitrate_window_[table_id][i] - curr_mean) * (hitrate_window_[table_id][i] - curr_mean);
        }
        hitrate_var_[table_id] = curr_var / static_cast<float>(window_size_ - 1); // calculate sample variance
    }
    else 
    {
        float p = curr_mean;
        hitrate_var_[table_id] = (p * (1 - p)) / static_cast<float>(num_keys);
    }
}

StatisticalInsertHeuristic::StatisticalInsertHeuristic(const size_t num_keys, const std::vector<float>& k_factor, const size_t window_size, const size_t num_inserts_needed, const size_t max_unsteady_samples) :
    num_keys_(num_keys),
    k_factor_(k_factor),
    window_size_(window_size),
    num_inserts_needed_(num_inserts_needed),
    max_unsteady_samples_(max_unsteady_samples),
    gen_(seed_),
    dis_(0.0, 1.0)
{
    // Validate constructor parameters
    NVE_CHECK_(!k_factor_.empty(), "k_factor vector cannot be empty");
    NVE_CHECK_(window_size_ > 1, "window_size must be greater than 1");
    NVE_CHECK_(num_inserts_needed_ > 0, "num_inserts_needed must be greater than 0");
    NVE_CHECK_(max_unsteady_samples_ > 0, "max_unsteady_samples must be greater than 0");
    
    const auto num_tables = k_factor_.size();
    for (size_t i = 0; i < num_tables; i++) 
    {
        NVE_CHECK_(k_factor_[i] > 0.0f, "k_factor values must be greater than 0");
        hitrate_window_.push_back(std::deque<float>(window_size_, 0));
        hitrate_mean_.push_back(0);
        hitrate_var_.push_back(0);
        chebyshev_probability_.push_back(getChebyshevProb(i));
        state_.push_back(State::Start);
        window_samples_collected_.push_back(0);
        insertions_performed_.push_back(0);
        consecutive_unsteady_samples_.push_back(0);
    }
}

bool StatisticalInsertHeuristic::insert_needed(const float hitrate, const size_t table_id)
{
    NVE_CHECK_(table_id < k_factor_.size(), "Invalid table index");
    switch (state_[table_id])
    {
        case State::Start: // collect a new window of samples
        {
            updateDistributionParams(hitrate, table_id, num_keys_);
            window_samples_collected_[table_id]++;
            if (window_samples_collected_[table_id] == window_size_)
            {
                state_[table_id] = State::Insert;
            }
            return false;
        }
        case State::Insert: // perform num_inserts_needed_ insertions
        {
            insertions_performed_[table_id]++; 
            if (insertions_performed_[table_id] % num_inserts_needed_ == 0)
            {
                state_[table_id] = State::Unstable;
            }
            return true;
        }
        case State::Steady:
        {
            float k_std = k_factor_[table_id] * std::sqrt(hitrate_var_[table_id]);
            if (std::abs(hitrate - hitrate_mean_[table_id]) >= k_std)
            {
                // this is a bad sample, if we see a few in a row, means distribution might be changing and we need to insert.
                consecutive_unsteady_samples_[table_id]++;
                if (consecutive_unsteady_samples_[table_id] == max_unsteady_samples_)
                {
                    state_[table_id] = State::Insert;
                    consecutive_unsteady_samples_[table_id] = 0;
                }
            }
            else
            {
                consecutive_unsteady_samples_[table_id] = 0;
                updateDistributionParams(hitrate, table_id, num_keys_);
            }
            return false;
        }
        case State::Unstable:
        {
            // we'll reach here after inserting. if the insertion made a change, we should collect a new window of samples and insert again. Otherwise- we are in a steady state.
            state_[table_id] = State::Steady;
            float k_std = k_factor_[table_id] * std::sqrt(hitrate_var_[table_id]);
            if (std::abs(hitrate - hitrate_mean_[table_id]) >= k_std)
            {
                float steadystate_chance = chebyshev_probability_[table_id];
                if (dis_(gen_) > steadystate_chance)
                {
                    state_[table_id] = State::Start;
                    window_samples_collected_[table_id] = 1;
                }
            }       
            updateDistributionParams(hitrate, table_id, num_keys_);
            return false;
        }
        default:
            NVE_THROW_("Invalid heuristic state!"); // This should not be possible
            return false;
    }
}

} // namespace nve
