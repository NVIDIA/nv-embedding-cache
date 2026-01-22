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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace nve {

// Abstract base class for generators of feature categories. Derived classes will
// be specialised to draw random values from particular distributions.
// Any generator requires two parameters for the features: number of categories and
// the number of indices per sample (aka hotness).
// Each call to the getCategoryIndices() method return a C++ vector with the randomly
// generated category indices. There will be no index repetitions in this returned
// vector.
template <typename IndexType>
class FeatureGenerator {
 public:
  FeatureGenerator(
      const IndexType num_categories,
      const size_t num_hot,
      const bool shuffle = false,
      const bool permute = false) : num_categories_(num_categories), num_hot_(num_hot), shuffle_(shuffle), permute_(permute) {
    assert(this->num_categories_ > 1);
    if (permute) {
      this->permutation_.resize(static_cast<size_t>(num_categories) + 1);
      this->inverse_permutation_.resize(static_cast<size_t>(num_categories) + 1);
      std::iota(this->permutation_.begin(), this->permutation_.end(), 0);

      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(this->permutation_.begin(), this->permutation_.end(), g);

      for (IndexType i = 0; i < num_categories + 1; ++i) {
        this->inverse_permutation_[static_cast<size_t>(this->permutation_[static_cast<size_t>(i)])] = i;
      }
    }
  }

  virtual ~FeatureGenerator() = default;

  // Returns a vector of random category indices.
  virtual std::vector<IndexType> getCategoryIndices() = 0;

  // Returns the number of categories for the feature.
  IndexType getNumCategories() const {
    return num_categories_;
  }

  // Returns the hotness for teh feature (number of looks ups per sampel).
  size_t getNumHot() const {
    return num_hot_;
  }

  IndexType getPermutedIndex(IndexType index) const {
    if (this->permute_) {
      return this->permutation_[static_cast<size_t>(index)];
    } else {
      return index;
    }
  }

  const std::vector<IndexType>& getInversePermutation() const {
    return this->inverse_permutation_;
  }

 protected:
  IndexType num_categories_ = 0;
  size_t    num_hot_ = 0;
  bool      shuffle_ = false;
  bool      permute_ = false;
  std::vector<IndexType> permutation_;
  std::vector<IndexType> inverse_permutation_;
};

// A class for generating category indices in sequence starting from 0. Usefule for 
// measuring achieved memory throughput from a random uniform distribution.
// Category index 0 is not generated as it is assumed to be reserved for a "missing"
// category. Thus, given num_categories, returned indices are drawn from [1, num_categories]
// range. Each returned set of indices contains exactly num_categories indices, with no repetitions.
template <typename IndexType = int>
class SequentialFeatureGenerator : public FeatureGenerator<IndexType> {
 public:
  SequentialFeatureGenerator() = delete;

  SequentialFeatureGenerator(
      const IndexType num_categories,
      const size_t num_hot) : FeatureGenerator<IndexType>(num_categories, num_hot) {
    next_idx_ = 0;
  }

  std::vector<IndexType> getCategoryIndices() override {
    std::vector<IndexType> indices;
    size_t hotness = this->getNumHot();
    for (size_t i = 0; i < hotness; ++i) {
      indices.push_back(next_idx_);
      next_idx_++;
      if (next_idx_ >= this->num_categories_) {
        next_idx_ = 0;
      }
    }

    return indices;
  }

 protected:
  std::default_random_engine generator_;
  std::unique_ptr<std::uniform_int_distribution<IndexType>> distribution_;
  IndexType next_idx_ = 0;
};

// A class for generating category indices drawn from a random uniform distribution.
// Category index 0 is not generated as it is assumed to be reserved for a "missing"
// category. Thus, given num_categories, returned indices are drawn from [1, num_categories]
// range. Each returned set of indices contains exactly num_categories indices, with no repetitions.
template <typename IndexType = int>
class UniformFeatureGenerator : public FeatureGenerator<IndexType> {
 public:
  UniformFeatureGenerator() = delete;

  UniformFeatureGenerator(
      const IndexType num_categories,
      const size_t num_hot,
      const size_t seed,
      const bool shuffle = false,
      const bool permute = false) : FeatureGenerator<IndexType>(num_categories, num_hot, shuffle, permute), generator_(seed) {
    distribution_.reset(new std::uniform_int_distribution<IndexType>(0, this->getNumCategories() - 1));
  }

  std::vector<IndexType> getCategoryIndices() override {
    std::set<IndexType> used_indices; // A set created to track already used indices.
    while (used_indices.size() < this->getNumHot()) {
      used_indices.insert(this->getPermutedIndex((*distribution_)(generator_)));
    }

    std::vector<IndexType> indices;
    for (const auto& x : used_indices) {
      indices.push_back(x);
    }

    if (this->shuffle_) {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(indices.begin(), indices.end(), g);
    }

    return indices;
  }

 protected:
  std::default_random_engine generator_;
  std::unique_ptr<std::uniform_int_distribution<IndexType>> distribution_;
};


// A class for generating category indices drawn from a powr law distribution. Category index 0 is
// not generated as it is assumed to be reserved for a "missing" category. Thus, given num_categories,
// returned indices are drawn from [1, num_categories] range. Each returned set of indices contains
// exactly num_categories, with no repetitions. Power law distribution is specified via its exponent,
// alpha > 0. Smaller indices correspond to more frequent categories (i.e. 1 will be the most frequent
// category, 2 - the second most frequent one, etc.).
// If math_numpy is true, generate distribution that matches numpy
// https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html.
// TODO: check if the trasnslateToPowerLaw really returns values in hte [1, num_categories] range or
//  if the highest value never gets generated.
template <typename IndexType = int>
class PowerLawFeatureGenerator : public FeatureGenerator<IndexType> {
 public:
  PowerLawFeatureGenerator() = delete;

  PowerLawFeatureGenerator(
      const IndexType num_categories,
      const size_t num_hot,
      const double alpha,
      const size_t seed,
      const bool shuffle = false,
      const bool permute = false,
      const bool math_numpy = false)
      : FeatureGenerator<IndexType>(num_categories, num_hot, shuffle, permute), alpha_(alpha), math_numpy_(math_numpy), generator_(seed) {
    distribution_.reset(new std::uniform_real_distribution<double>(0., 1.));
  }

  std::vector<IndexType> getCategoryIndices() override {
    std::set<IndexType> used_indices;
    while (used_indices.size() < this->num_hot_) {
      const double x = (*distribution_)(generator_);
      IndexType y = (IndexType)-1;
      if (!this->math_numpy_) {
        y = IndexType(translateToPowerLaw(1., double(this->num_categories_), alpha_, x));
      } else {
        y = IndexType(translateToPowerLaw(0., 1., 1. - alpha_, x) * static_cast<double>(this->num_categories_));
      }
      used_indices.insert(this->getPermutedIndex(y));
    }

    std::vector<IndexType> indices;
    for (const auto& x : used_indices) {
      indices.push_back(x);
    }

    if (this->shuffle_) {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(indices.begin(), indices.end(), g);
    }

    return indices;
  }

 protected:
  double alpha_ = 0.; // Exponent for the power-law distribution.
  bool math_numpy_ = false;  // If true, match numpy random.powerlaw
  std::default_random_engine generator_;
  std::unique_ptr<std::uniform_real_distribution<double>> distribution_;

  // Function that "translates" a value drawn uniformly from [0, 1] range into a
  // value drawn from a power-law distribution. Power-law distribution is characterized
  // by the range of values [min_value, max_value] and the exponent value alpha.
  // Assumptions:
  //  * k_min >= 1
  //  * alpha > 0
  template <typename Type = double>
  static double translateToPowerLaw(
      const Type min_value,
      const Type max_value,
      const Type alpha,
      const Type random_uniform_value) {
    const Type gamma = 1 - alpha;
    const Type y =
        pow(random_uniform_value * (pow(max_value, gamma) - pow(min_value, gamma)) + pow(min_value, gamma),
            1.0 / gamma);
    return y;
  }
};

template <typename IndexType>
static std::shared_ptr<FeatureGenerator<IndexType>> getSampleGenerator(float alpha, IndexType num_categories, size_t num_hot, size_t seed)
{
  std::shared_ptr<FeatureGenerator<IndexType>> sample_generator;
  if (alpha == -1.0)
  {
    sample_generator.reset(new SequentialFeatureGenerator<IndexType>(num_categories, num_hot));
  }
  else if (alpha == 0.0)
  {
    sample_generator.reset(new UniformFeatureGenerator<IndexType>(num_categories, num_hot, seed)); 
  }
  else
  {
    sample_generator.reset(new PowerLawFeatureGenerator<IndexType>(num_categories, num_hot, alpha, seed));
  }
  return sample_generator;
}

}  // namespace nve
