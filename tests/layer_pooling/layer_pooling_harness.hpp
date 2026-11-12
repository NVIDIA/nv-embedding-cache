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

// Unified, capability-driven harness for validating embedding-layer POOLING and
// DEQUANTIZATION across all layer types (GPU / LinearUVM / HierWithGPU / Host /
// HierWithoutGPU) against the MockHostTable::combine() reference.
//
// Design (see docs / plan):
//   * Each layer is backed by a linear table of `num_rows` rows indexed directly by key:
//       - GPU           : a device buffer (a copy of the host buffer)
//       - LinearUVM     : a UVM-backed GpuTable (uvm_table == the host buffer)
//       - HierWithGPU   : a GPU cache tier backed by a LinearHostTable holding all rows
//       - Host          : a LinearHostTable (emb_table == the host buffer)
//       - HierWithoutGPU: a bounded PHMap cache backed by a LinearHostTable holding all rows
//     Every key resolves from a full backing table, and the row a layer reads for key k is exactly
//     the host buffer's row k. The GPU hierarchy keeps its cache empty; the host-only hierarchy
//     warms its PHMap cache after the lookup.
//   * The reference gathers rows straight from the host buffer by key and reduces them with
//     MockHostTable::combine() (the single oracle for pooling + dequant). combine() takes no
//     ExecutionContext, so the reference path is independent of the layer's CUDA context.
//   * Quant row strides are NOT padded, so combine()'s internal value_count = stride - meta
//     stays exactly equal to value_count.
//
// The matrix is generated as the cartesian product of the dimensions and filtered by Classify():
//   Enabled    -> runs and must match the reference
//   Rejected   -> a tracked capability gap (e.g. CUDA cross-precision dequant output, which is not
//                 yet implemented): runs and must be refused with an InvalidArgumentError, so a
//                 regression that silently returns a wrong result is caught
//   Unsupported-> dropped entirely (never-valid combos)
// Adding/removing support later is a one-line change in Classify().

#pragma once

#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_support.hpp>
#include "cuda_ops/cuda_common.h"   // ScopedDevice
#include <embedding_layer.hpp>
#include <execution_context.hpp>
#include <gpu_embedding_layer.hpp>
#include <gpu_table.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <host_embedding_layer.hpp>
#include <host_table.hpp>
#include <plugin/plugin_loader.hpp>

#include "cuda_ptr.hpp"
#include "test_utils.hpp"
#include <linear_embedding_layer.hpp>
#include <linear_host_table.hpp>
#include <nve_types.hpp>

#include "emb_layer_utils.hpp"   // InitTableRows(Quant), GenerateWeights, SetupCSROffsets, load_as_float
#include "mock_host_table.hpp"   // MockHostTable<IndexT>::combine() -- the reference oracle

namespace nve {
namespace layer_pooling {

// ---------------------------------------------------------------------------
// Test matrix
// ---------------------------------------------------------------------------

// HierWithoutGPU is a HierarchicalEmbeddingLayer built from host tables only (no GPU tier): it
// exercises the layer's gpu_device_ < 0 pooling/dequant path, which reuses the host CPU kernels.
enum class LayerKind { GPU, LinearUVM, HierWithGPU, Host, HierWithoutGPU };

struct LayerPoolCase {
  LayerKind     layer;
  PoolingType_t pooling;
  SparseType_t  sparse;
  DataType_t    in_dtype;       // stored (input) value dtype
  DataType_t    out_dtype;      // output dtype: Float32/Float16
  DataType_t    weight_dtype;   // Float32/Float16 for weighted modes, else Unknown
  bool          key_is_int64;   // true -> int64 keys, false -> int32

  // Fixed-cost dimensions (correctness suite, not stress): kept modest so many cases coexist.
  int64_t       value_count;    // number of value elements per row
  int64_t       hotness;        // bag size (Fixed) / average bag (CSR)
  int64_t       num_keys;
  int64_t       num_rows;       // size of the backing table
};

enum class CaseStatus { Enabled, Rejected, Unsupported };

// ---------------------------------------------------------------------------
// Small dtype/stride helpers (no padding -- keeps combine()'s value_count exact)
// ---------------------------------------------------------------------------

// Output row stride in bytes for the (always float) output dtype.
inline int64_t output_row_bytes(DataType_t out_dtype, int64_t value_count) {
  return value_count * dtype_size(out_dtype);
}

// ---------------------------------------------------------------------------
// Capability classification -- the crux of the matrix.
// ---------------------------------------------------------------------------

inline CaseStatus Classify(const LayerPoolCase& c) {
  const bool quant = is_quant_rowwise(c.in_dtype);
  const bool weighted = is_weighted_pooling(c.pooling);

  // Output is always a float type.
  if (c.out_dtype != DataType_t::Float32 && c.out_dtype != DataType_t::Float16) {
    return CaseStatus::Unsupported;
  }
  // weight_dtype is a derived field, meaningful only for the weighted pooling modes: the layers
  // consult it solely when weights != nullptr and otherwise ignore it (they do NOT error on
  // a stray weight dtype). So only gate the case it actually matters -- weights must be a float type
  // for the weighted modes. For non-weighted modes the weight dtype is irrelevant and not gated
  // (the generator leaves it Unknown).
  if (weighted && c.weight_dtype != DataType_t::Float32 && c.weight_dtype != DataType_t::Float16) {
    return CaseStatus::Unsupported;
  }
  // Quant input is dequantized to the requested float output. Non-quant input is cast only by the
  // CPU pooling path, whose output dtype dispatch is independent of the input dtype
  // (cpu_kernel_pooling_dispatch_out); the CUDA kernels have no float->float cast, so the
  // GPU-backed layers only ever emit the input dtype and those combinations are dropped.
  if (!quant && c.out_dtype != c.in_dtype &&
      c.layer != LayerKind::Host && c.layer != LayerKind::HierWithoutGPU) {
    return CaseStatus::Unsupported;
  }

  // Per-layer capability gaps below are real but not (yet) supported. They are kept in the matrix
  // as Rejected -- they still run, asserting the layer refuses the request -- so the gap is tracked,
  // self-documenting, and cannot regress into a silently wrong result. Promoting one to Enabled is
  // a one-line change here when the underlying support lands.
  switch (c.layer) {
    case LayerKind::GPU:
      // Quantized tables gather raw rows first, then use the shared CUDA dequant/combine path.
      // That path supports both key widths and either float weight type, but emits only the
      // quantization metadata's float precision.
      if (quant) {
        if (c.out_dtype != quant_rowwise_output_dtype(c.in_dtype)) return CaseStatus::Rejected;
        return CaseStatus::Enabled;
      }
      // cuEmbed assumes the weight dtype equals the value dtype (always holds for enabled GPU cases,
      // since non-quant implies out == in and the generator sets weight == out).
      if (weighted && c.weight_dtype != c.in_dtype) return CaseStatus::Unsupported;
      return CaseStatus::Enabled;

    case LayerKind::LinearUVM:
    case LayerKind::HierWithGPU:
      // The GPU dequant/combine kernels emit the dequantized result in the scale's float precision,
      // so cross-precision output (fp32 scale -> fp16 output, or vice versa) is not supported here.
      if (quant && c.out_dtype != quant_rowwise_output_dtype(c.in_dtype)) {
        return CaseStatus::Rejected;
      }
      return CaseStatus::Enabled;

    case LayerKind::Host:
    case LayerKind::HierWithoutGPU:
      // The host CPU dequant path supports any fp32/fp16 output precision (incl. cross-precision).
      // The host-only hierarchical layer routes through the same kernels, so it has no extra gaps.
      return CaseStatus::Enabled;
  }
  return CaseStatus::Unsupported;
}

// Human-readable reason a case is Rejected (for diagnostics and the support-matrix doc). Returns
// nullptr for cases that are not Rejected.
inline const char* RejectionReason(const LayerPoolCase& c) {
  if (Classify(c) != CaseStatus::Rejected) return nullptr;
  const bool quant = is_quant_rowwise(c.in_dtype);
  if (c.layer == LayerKind::GPU) {
    if (quant) {
      return "GPU dequant kernels emit only the scale's float precision (no cross-precision out)";
    }
  }
  if (c.layer == LayerKind::LinearUVM || c.layer == LayerKind::HierWithGPU) {
    return "UVM/Hier dequant kernels emit only the scale's float precision (no cross-precision out)";
  }
  return "unsupported";
}

// ---------------------------------------------------------------------------
// Readable, gtest-legal ([A-Za-z0-9_]) case names.
// ---------------------------------------------------------------------------

inline const char* LayerName(LayerKind l) {
  switch (l) {
    case LayerKind::GPU:          return "GPU";
    case LayerKind::LinearUVM:    return "LinearUVM";
    case LayerKind::HierWithGPU:  return "HierWithGPU";
    case LayerKind::Host:         return "Host";
    case LayerKind::HierWithoutGPU: return "HierWithoutGPU";
  }
  return "Unknown";
}

inline const char* PoolName(PoolingType_t p) {
  switch (p) {
    case PoolingType_t::Concatenate:  return "Concat";
    case PoolingType_t::Sum:          return "Sum";
    case PoolingType_t::Mean:         return "Mean";
    case PoolingType_t::WeightedSum:  return "WSum";
    case PoolingType_t::WeightedMean: return "WMean";
  }
  return "Unknown";
}

inline const char* SparseName(SparseType_t s) {
  return s == SparseType_t::Fixed ? "Fixed" : "CSR";
}

inline const char* DtShort(DataType_t dt) {
  switch (dt) {
    case DataType_t::Float32:          return "F32";
    case DataType_t::Float16:          return "F16";
    case DataType_t::QInt8RowwiseF32:  return "QI8F32";
    case DataType_t::QInt8RowwiseF16:  return "QI8F16";
    case DataType_t::QUint8RowwiseF32: return "QU8F32";
    case DataType_t::QUint8RowwiseF16: return "QU8F16";
    default:                           return "X";
  }
}

inline std::string CaseLabel(const LayerPoolCase& c) {
  std::string s = std::string(LayerName(c.layer)) + "_" + PoolName(c.pooling) + "_" +
                  SparseName(c.sparse) + "_in" + DtShort(c.in_dtype) + "_out" +
                  DtShort(c.out_dtype) + "_k" + (c.key_is_int64 ? "64" : "32");
  if (is_weighted_pooling(c.pooling)) s += "_w" + std::string(DtShort(c.weight_dtype));
  return s;
}

struct CaseName {
  template <typename ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
    return CaseLabel(info.param);
  }
};

inline std::ostream& operator<<(std::ostream& o, const LayerPoolCase& c) {
  return o << CaseLabel(c);
}

// ---------------------------------------------------------------------------
// Case generation: cartesian product, then split into enabled / rejected.
// ---------------------------------------------------------------------------

struct GeneratedCases {
  std::vector<LayerPoolCase> enabled;
  std::vector<LayerPoolCase> rejected;
};

inline GeneratedCases GenerateCases() {
  static constexpr int64_t kValueCount = 64;
  static constexpr int64_t kHotness    = 8;
  static constexpr int64_t kNumKeys    = 1 << 10;
  static constexpr int64_t kNumRows    = 1 << 16;

  const LayerKind     layers[]  = {LayerKind::GPU, LayerKind::LinearUVM, LayerKind::HierWithGPU,
                                   LayerKind::Host, LayerKind::HierWithoutGPU};
  const PoolingType_t pools[]   = {PoolingType_t::Concatenate, PoolingType_t::Sum,
                                   PoolingType_t::Mean, PoolingType_t::WeightedSum,
                                   PoolingType_t::WeightedMean};
  const SparseType_t  sparses[] = {SparseType_t::Fixed, SparseType_t::CSR};
  const DataType_t    ins[]     = {DataType_t::Float32, DataType_t::Float16,
                                   DataType_t::QInt8RowwiseF32, DataType_t::QInt8RowwiseF16,
                                   DataType_t::QUint8RowwiseF32, DataType_t::QUint8RowwiseF16};
  const DataType_t    outs[]    = {DataType_t::Float32, DataType_t::Float16};
  const bool          keys64[]  = {true, false};

  const DataType_t weight_dtypes[] = {DataType_t::Float32, DataType_t::Float16};

  GeneratedCases result;
  auto emit = [&](LayerPoolCase c) {
    switch (Classify(c)) {
      case CaseStatus::Enabled:     result.enabled.push_back(c);  break;
      case CaseStatus::Rejected:    result.rejected.push_back(c); break;
      case CaseStatus::Unsupported: break;
    }
  };

  for (auto layer : layers)
    for (auto pool : pools)
      for (auto sparse : sparses)
        for (auto in : ins)
          for (auto out : outs)
            for (auto k64 : keys64) {
              LayerPoolCase c;
              c.layer        = layer;
              c.pooling      = pool;
              c.sparse       = sparse;
              c.in_dtype     = in;
              c.out_dtype    = out;
              c.key_is_int64 = k64;
              c.value_count  = kValueCount;
              c.hotness      = kHotness;
              c.num_keys     = kNumKeys;
              c.num_rows     = kNumRows;
              if (is_weighted_pooling(pool)) {
                for (auto wdt : weight_dtypes) {
                  c.weight_dtype = wdt;
                  emit(c);
                }
              } else {
                c.weight_dtype = DataType_t::Unknown;
                emit(c);
              }
            }
  return result;
}

// ---------------------------------------------------------------------------
// Tolerance: reuse the cpu_pooling convention, loosened for fp16 accumulation.
// ---------------------------------------------------------------------------

inline bool involves_fp16(DataType_t dt) {
  return dt == DataType_t::Float16 || dt == DataType_t::QInt8RowwiseF16 ||
         dt == DataType_t::QUint8RowwiseF16;
}

inline float pooling_tolerance(const LayerPoolCase& c) {
  // Base tolerance: 1e-3 if any operand is fp16-ish, else 1e-6.
  float tol = (involves_fp16(c.in_dtype) || involves_fp16(c.out_dtype)) ? 1e-3f : 1e-6f;
  // Summation accumulates ~hotness terms; fp16 ULP grows with magnitude, so scale Sum/WeightedSum.
  const bool sum_like = (c.pooling == PoolingType_t::Sum || c.pooling == PoolingType_t::WeightedSum);
  if (sum_like && (involves_fp16(c.in_dtype) || involves_fp16(c.out_dtype))) {
    tol *= static_cast<float>(std::max<int64_t>(c.hotness, 1));
  }
  return tol;
}

// ---------------------------------------------------------------------------
// The per-case harness. Builds the layer + backing store, runs lookup with pooling,
// computes the reference via combine(), and compares.
// ---------------------------------------------------------------------------

inline bool gpu_available() {
  int n = 0;
  return (cudaGetDeviceCount(&n) == cudaSuccess) && (n > 0);
}

template <typename IndexT>
class PoolingHarness {
 public:
  static constexpr int kDeviceId = 0;

  explicit PoolingHarness(const LayerPoolCase& c) : c_(c) {
    ScopedDevice dev(kDeviceId);

    in_row_bytes_  = row_size_in_bytes(c_.in_dtype, c_.value_count);
    out_row_bytes_ = output_row_bytes(c_.out_dtype, c_.value_count);

    // 1. Seed the backing host buffer (the source of truth read by both layer and reference).
    const size_t table_bytes = static_cast<size_t>(c_.num_rows * in_row_bytes_);
    NVE_CHECK_(cudaMallocHost(&h_table_, table_bytes));
    if (is_quant_rowwise(c_.in_dtype)) {
      InitTableRowsQuant(h_table_, in_row_bytes_, c_.value_count, 0,
                         static_cast<uint64_t>(c_.num_rows), c_.in_dtype, kSeed);
    } else {
      InitTableRows(h_table_, static_cast<uint64_t>(in_row_bytes_), 0,
                    static_cast<uint64_t>(c_.num_rows), c_.in_dtype, kSeed);
    }

    // 2. Reference table -- only used for combine() (which ignores the context).
    HostTableConfig mock_cfg;
    mock_cfg.value_dtype    = c_.in_dtype;
    mock_cfg.max_value_size = in_row_bytes_;
    ref_ = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/);

    // 3. Build the layer under test.
    switch (c_.layer) {
      case LayerKind::GPU:            BuildGpu(); break;
      case LayerKind::LinearUVM:      BuildUvm(); break;
      case LayerKind::HierWithGPU:    BuildHierWithGPU(); break;
      case LayerKind::Host:           BuildHost(); break;
      case LayerKind::HierWithoutGPU: BuildHierWithoutGPU(); break;
    }
    ctx_ = layer_->create_execution_context(0, 0, nullptr, nullptr);
  }

  ~PoolingHarness() {
    ScopedDevice dev(kDeviceId);
    if (ctx_) {
      ctx_->wait();
      ctx_.reset();
    }
    layer_.reset();
    gpu_tab_.reset();
    host_tab_.reset();
    if (d_table_) NVE_CHECK_(cudaFree(d_table_));
    if (h_table_) NVE_CHECK_(cudaFreeHost(h_table_));
  }

  // expect_rejected drives a Rejected case: the combination is a tracked capability gap, so the
  // layer must refuse the request instead of producing output.
  void Run(bool expect_rejected = false) {
    ScopedDevice dev(kDeviceId);
    const bool concat = (c_.pooling == PoolingType_t::Concatenate);

    // Keys (host-resident; layers copy as needed). Deterministic.
    std::vector<IndexT> keys(static_cast<size_t>(c_.num_keys));
    {
      std::mt19937 gen(kSeed + 1);
      std::uniform_int_distribution<int64_t> kd(0, c_.num_rows - 1);
      for (auto& k : keys) k = static_cast<IndexT>(kd(gen));
    }

    // Pooling params + offsets.
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = c_.pooling;
    // Every layer requires an explicit output type when PoolingParams is provided.
    pp.output_type  = c_.out_dtype;

    SetupCSROffsets<IndexT> csr(c_.sparse == SparseType_t::CSR ? static_cast<IndexT>(c_.num_keys)
                                                               : IndexT{1},
                                static_cast<IndexT>(std::max<int64_t>(c_.hotness, 1)));
    std::vector<int8_t> weights;
    double ignored_weight = 0.0;
    int64_t output_bags = c_.num_keys;
    if (concat) {
      // Deliberately poison every ignored field. A Concatenate implementation must use only the
      // keys vector and output_type; touching any of these values should make the case fail.
      pp.sparse_type = static_cast<SparseType_t>(2);
      pp.csr_offsets = nullptr;
      pp.num_csr_offsets = -1;
      pp.fixed_hotness = -1;
      pp.weights = &ignored_weight;
      pp.weight_type = DataType_t::Float64;
    } else {
      pp.sparse_type = c_.sparse;
      if (c_.sparse == SparseType_t::CSR) {
        pp.csr_offsets = csr.offsets_buffer;
        pp.num_csr_offsets = static_cast<int64_t>(csr.num_offsets);
        output_bags = static_cast<int64_t>(csr.num_offsets) - 1;
      } else {
        pp.fixed_hotness = c_.hotness;
        output_bags = c_.num_keys / c_.hotness;
      }

      if (is_weighted_pooling(c_.pooling)) {
        GenerateWeights(weights, static_cast<uint64_t>(c_.num_keys), c_.weight_dtype, kSeed + 2);
        pp.weights = weights.data();
        pp.weight_type = c_.weight_dtype;
      }
    }

    // Allocate exactly the rows promised by the pooling layout. This also verifies that layers do
    // not size a staging copy as num_keys rows when a reduction emits fewer output bags.
    int8_t* output_raw = nullptr;
    const size_t out_bytes = static_cast<size_t>(output_bags * out_row_bytes_);
    NVE_CHECK_(cudaMallocHost(&output_raw, out_bytes));
    // RAII guard: ASSERT_* macros issue an early return on failure, which would bypass a manual
    // cudaFreeHost call and leak pinned memory. output_guard will handle free in such cases.
    HostPtr<int8_t> output_guard(output_raw);
    int8_t* output = output_raw;
    std::memset(output, 0, out_bytes);

    std::vector<float> hitrates(static_cast<size_t>(layer_->get_num_tables()), 0.f);
    if (expect_rejected) {
      // The request is well-formed; only the dtype combination is unimplemented. It must be
      // refused up front rather than produce a silently wrong result, so assert the error type
      // and stop here -- there is no reference output to compare against.
      EXPECT_THROW(layer_->lookup(ctx_, c_.num_keys, keys.data(), output, out_row_bytes_,
                                  nullptr /*hitmask*/, &pp, hitrates.data()),
                   InvalidArgumentError)
          << "case " << c_ << " (" << RejectionReason(c_) << ")";
      return;
    }
    layer_->lookup(ctx_, c_.num_keys, keys.data(), output, out_row_bytes_, nullptr /*hitmask*/,
                   &pp, hitrates.data());
    if (c_.layer != LayerKind::Host && c_.layer != LayerKind::HierWithoutGPU) {
      NVE_CHECK_(cudaDeviceSynchronize());
    }
    ctx_->wait();

    // Every key is resident in a full backing table, so the comparison below is always against a
    // full gather. The reported hitrate is only a meaningful per-key metric for the GPU
    // (cuEmbed or gathered-row GPU path) and Host layers; the LinearUVM/HierWithGPU pooled+dequant
    // path counts misses differently (its output is bags / dequantized rows, not per-key cache
    // rows), so it is not
    // asserted there -- matching the existing UVMPooling / RunQuantUVMLookup tests, which also skip
    // the hitrate check. HierWithoutGPU starts with an empty PHMap cache, so every initial lookup
    // misses tier 0 and resolves from the full linear table in tier 1. Its default insert heuristic
    // then warms the cache from those gathered rows.
    if (c_.layer == LayerKind::GPU || c_.layer == LayerKind::Host) {
      EXPECT_NEAR(hitrates[0], 1.0f, 1e-6f) << "case " << c_;
    } else if (c_.layer == LayerKind::HierWithoutGPU) {
      ASSERT_EQ(hitrates.size(), 2u) << "case " << c_;
      EXPECT_NEAR(hitrates[0], 0.0f, 1e-6f) << "case " << c_;
      EXPECT_NEAR(hitrates[1], 1.0f, 1e-6f) << "case " << c_;
      const int64_t cache_rows = host_cache_tab_->size(ctx_);
      EXPECT_GT(cache_rows, 0) << "PHMap cache was not warmed for case " << c_;
      EXPECT_LE(cache_rows, kHostCacheRows)
          << "PHMap cache exceeded its configured bound for case " << c_;

      // Repeat the same request after the PHMap cache has warmed. LinearHostTable must report only
      // keys not already resolved by the cache, so the two per-tier hit rates still partition the
      // request instead of summing above one.
      std::memset(output, 0, out_bytes);
      std::fill(hitrates.begin(), hitrates.end(), 0.0f);
      layer_->lookup(ctx_, c_.num_keys, keys.data(), output, out_row_bytes_, nullptr /*hitmask*/,
                     &pp, hitrates.data());
      ctx_->wait();
      EXPECT_GT(hitrates[0], 0.0f) << "warmed cache did not resolve any keys for case " << c_;
      EXPECT_GE(hitrates[1], 0.0f) << "case " << c_;
      EXPECT_NEAR(hitrates[0] + hitrates[1], 1.0f, 1e-6f) << "case " << c_;
    }

    // Reference: gather raw rows by key from the host buffer, then combine().
    std::vector<int8_t> gathered(static_cast<size_t>(c_.num_keys * in_row_bytes_));
    for (int64_t i = 0; i < c_.num_keys; ++i) {
      const int64_t key = static_cast<int64_t>(keys[static_cast<size_t>(i)]);
      std::memcpy(gathered.data() + i * in_row_bytes_, h_table_ + key * in_row_bytes_,
                  static_cast<size_t>(in_row_bytes_));
    }
    // The GPU layer uses cuEmbed's input-typed output only for non-quantized tables. Quantized GPU
    // tables, like the other layers, dequantize to the requested output dtype.
    const DataType_t ref_out =
        (c_.layer == LayerKind::GPU && !is_quant_rowwise(c_.in_dtype)) ? c_.in_dtype : c_.out_dtype;
    std::vector<int8_t> ref_output(static_cast<size_t>(output_bags * out_row_bytes_));
    // The legacy test oracle still accepts bag metadata for Concatenate. Supply a canonical
    // hotness-one layout to the oracle; the layer under test received the poisoned fields above.
    const bool ref_is_csr = !concat && pp.sparse_type == SparseType_t::CSR;
    const int64_t ref_fixed_hotness = concat ? 1 : pp.fixed_hotness;
    ref_->combine(gathered.data(), c_.num_keys, pp.pooling_type,
                  concat ? SparseType_t::Fixed : pp.sparse_type,
                  ref_is_csr ? pp.csr_offsets : nullptr,
                  ref_is_csr ? pp.num_csr_offsets : 0,
                  ref_fixed_hotness,
                  concat ? nullptr : pp.weights,
                  concat ? DataType_t::Unknown : pp.weight_type, ref_output.data(), ref_out);

    const float tol = pooling_tolerance(c_);
    const int64_t elems = output_bags * c_.value_count;
    for (int64_t i = 0; i < elems; ++i) {
      ASSERT_NEAR(load_as_float(output, i, c_.out_dtype),
                  load_as_float(ref_output.data(), i, c_.out_dtype), tol)
          << "case " << c_ << " element " << i;
    }

  }

  // Run one Fixed-hotness Sum lookup with a padded output stride. The CUDA pool/dequant kernels
  // derive the embedding width from the output stride, so the GPU-backed layers must reject any
  // padding outright; the CPU pooling path takes the width from the stored row and must honor the
  // padded layout. Expects a Sum/Fixed case.
  void RunPaddedStride() {
    ScopedDevice dev(kDeviceId);

    std::vector<IndexT> keys(static_cast<size_t>(c_.num_keys));
    {
      std::mt19937 gen(kSeed + 1);
      std::uniform_int_distribution<int64_t> kd(0, c_.num_rows - 1);
      for (auto& k : keys) k = static_cast<IndexT>(kd(gen));
    }

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type  = c_.pooling;
    pp.output_type   = c_.out_dtype;
    pp.sparse_type   = SparseType_t::Fixed;
    pp.fixed_hotness = c_.hotness;
    const int64_t output_bags = c_.num_keys / c_.hotness;

    const int64_t padded_stride = out_row_bytes_ + 2 * dtype_size(c_.out_dtype);
    int8_t* output_raw = nullptr;
    const size_t out_bytes = static_cast<size_t>(output_bags * padded_stride);
    NVE_CHECK_(cudaMallocHost(&output_raw, out_bytes));
    HostPtr<int8_t> output_guard(output_raw);
    std::memset(output_raw, 0, out_bytes);

    const bool gpu_backed = c_.layer == LayerKind::GPU || c_.layer == LayerKind::LinearUVM ||
                            c_.layer == LayerKind::HierWithGPU;
    if (gpu_backed) {
      EXPECT_THROW(layer_->lookup(ctx_, c_.num_keys, keys.data(), output_raw, padded_stride,
                                  nullptr /*hitmask*/, &pp, nullptr /*hitrates*/),
                   Exception)
          << "padded output stride must be rejected for case " << c_;
      ctx_->wait();
      return;
    }

    layer_->lookup(ctx_, c_.num_keys, keys.data(), output_raw, padded_stride, nullptr /*hitmask*/,
                   &pp, nullptr /*hitrates*/);
    ctx_->wait();

    std::vector<int8_t> gathered(static_cast<size_t>(c_.num_keys * in_row_bytes_));
    for (int64_t i = 0; i < c_.num_keys; ++i) {
      const int64_t key = static_cast<int64_t>(keys[static_cast<size_t>(i)]);
      std::memcpy(gathered.data() + i * in_row_bytes_, h_table_ + key * in_row_bytes_,
                  static_cast<size_t>(in_row_bytes_));
    }
    std::vector<int8_t> ref_output(static_cast<size_t>(output_bags * out_row_bytes_));
    ref_->combine(gathered.data(), c_.num_keys, pp.pooling_type, SparseType_t::Fixed,
                  nullptr /*csr_offsets*/, 0 /*num_csr_offsets*/, pp.fixed_hotness,
                  nullptr /*weights*/, DataType_t::Unknown, ref_output.data(), c_.out_dtype);

    const float tol = pooling_tolerance(c_);
    for (int64_t bag = 0; bag < output_bags; ++bag) {
      const int8_t* out_row = output_raw + bag * padded_stride;
      const int8_t* ref_row = ref_output.data() + bag * out_row_bytes_;
      for (int64_t e = 0; e < c_.value_count; ++e) {
        ASSERT_NEAR(load_as_float(out_row, e, c_.out_dtype), load_as_float(ref_row, e, c_.out_dtype),
                    tol)
            << "case " << c_ << " bag " << bag << " element " << e;
      }
    }
  }

  // Run one CSR Mean/WeightedMean lookup whose offsets contain empty bags — and, for the weighted
  // variant, a bag whose weights sum to zero. Every zero-denominator bag must produce a zero output
  // row (matching the CPU kernels and torch's EmbeddingBag) rather than a division-by-zero NaN.
  // Expects a CSR case with num_keys == 12; the bag layout is fixed: sizes {3,0,3,0,0,3,3,0}, and
  // for WeightedMean the last non-empty bag's weights sum to zero.
  void RunEmptyBagCsr() {
    ScopedDevice dev(kDeviceId);
    ASSERT_EQ(c_.num_keys, 12) << "RunEmptyBagCsr expects the fixed 12-key bag layout";
    const bool weighted = is_weighted_pooling(c_.pooling);

    std::vector<IndexT> keys(static_cast<size_t>(c_.num_keys));
    {
      std::mt19937 gen(kSeed + 1);
      std::uniform_int_distribution<int64_t> kd(0, c_.num_rows - 1);
      for (auto& k : keys) k = static_cast<IndexT>(kd(gen));
    }

    const std::vector<IndexT> offsets = {0, 3, 3, 6, 6, 6, 9, 12, 12};
    const int64_t output_bags = static_cast<int64_t>(offsets.size()) - 1;

    // Unit weights except the bag spanning keys [9, 12), whose weights sum to zero.
    std::vector<float> weights(static_cast<size_t>(c_.num_keys), 1.f);
    weights[9] = 1.f;
    weights[10] = -2.f;
    weights[11] = 1.f;

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = c_.pooling;
    pp.output_type = c_.out_dtype;
    pp.sparse_type = SparseType_t::CSR;
    pp.csr_offsets = offsets.data();
    pp.num_csr_offsets = static_cast<int64_t>(offsets.size());
    if (weighted) {
      pp.weights = weights.data();
      pp.weight_type = DataType_t::Float32;
    }

    int8_t* output_raw = nullptr;
    const size_t out_bytes = static_cast<size_t>(output_bags * out_row_bytes_);
    NVE_CHECK_(cudaMallocHost(&output_raw, out_bytes));
    HostPtr<int8_t> output_guard(output_raw);
    std::memset(output_raw, 0x5A, out_bytes);  // poison so zero rows are proven written

    layer_->lookup(ctx_, c_.num_keys, keys.data(), output_raw, out_row_bytes_, nullptr /*hitmask*/,
                   &pp, nullptr /*hitrates*/);
    if (c_.layer != LayerKind::Host && c_.layer != LayerKind::HierWithoutGPU) {
      NVE_CHECK_(cudaDeviceSynchronize());
    }
    ctx_->wait();

    std::vector<int8_t> gathered(static_cast<size_t>(c_.num_keys * in_row_bytes_));
    for (int64_t i = 0; i < c_.num_keys; ++i) {
      const int64_t key = static_cast<int64_t>(keys[static_cast<size_t>(i)]);
      std::memcpy(gathered.data() + i * in_row_bytes_, h_table_ + key * in_row_bytes_,
                  static_cast<size_t>(in_row_bytes_));
    }
    std::vector<int8_t> ref_output(static_cast<size_t>(output_bags * out_row_bytes_));
    ref_->combine(gathered.data(), c_.num_keys, pp.pooling_type, SparseType_t::CSR, offsets.data(),
                  pp.num_csr_offsets, 0 /*fixed_hotness*/, weighted ? pp.weights : nullptr,
                  weighted ? pp.weight_type : DataType_t::Unknown, ref_output.data(), c_.out_dtype);

    const float tol = pooling_tolerance(c_);
    const int64_t elems = output_bags * c_.value_count;
    for (int64_t i = 0; i < elems; ++i) {
      const float got = load_as_float(output_raw, i, c_.out_dtype);
      ASSERT_FALSE(std::isnan(got)) << "case " << c_ << " element " << i << " is NaN";
      ASSERT_NEAR(got, load_as_float(ref_output.data(), i, c_.out_dtype), tol)
          << "case " << c_ << " element " << i;
    }
  }

 private:
  static constexpr size_t kSeed = 31337;
  static constexpr int64_t kHostCacheRows = int64_t{1} << 10;
  static constexpr int64_t kHostCachePartitions = 2;

  void BuildGpu() {
    const size_t table_bytes = static_cast<size_t>(c_.num_rows * in_row_bytes_);
    NVE_CHECK_(cudaMalloc(&d_table_, table_bytes));
    NVE_CHECK_(cudaMemcpy(d_table_, h_table_, table_bytes, cudaMemcpyHostToDevice));

    GPUEmbeddingLayerConfig cfg;
    cfg.layer_name               = "layer_pooling_gpu";
    cfg.device_id                = kDeviceId;
    cfg.embedding_table          = d_table_;
    cfg.num_embeddings           = c_.num_rows;
    cfg.embedding_width_in_bytes = in_row_bytes_;
    cfg.value_dtype              = c_.in_dtype;
    layer_ = std::make_shared<GPUEmbeddingLayer<IndexT>>(cfg);
  }

  void BuildUvm() {
    gpu_tab_ = MakeUvmGpuTable();
    typename LinearUVMEmbeddingLayer<IndexT>::Config cfg;
    cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
    layer_ = std::make_shared<LinearUVMEmbeddingLayer<IndexT>>(cfg, gpu_tab_);
  }

  void BuildHierWithGPU() {
    // A genuine two-tier hierarchy: a GPU cache tier (front) backed by a LinearHostTable (full
    // backing store). With NeverInsertHeuristic the GPU cache stays empty, so every key falls
    // through to the host tier -- exercising the cross-tier gather that feeds the GPU pooling/dequant
    // kernels. The GPU tier (no UVM here) still provides the device those kernels run on.
    GPUTableConfig gcfg;
    gcfg.device_id         = kDeviceId;
    gcfg.cache_size        = int64_t(1) << 20;
    gcfg.row_size_in_bytes = in_row_bytes_;
    gcfg.value_dtype       = c_.in_dtype;
    gcfg.count_misses      = true;
    gpu_tab_ = std::make_shared<GpuTable<IndexT>>(gcfg);

    LinearHostTableConfig hcfg;

    hcfg.key_size = sizeof(IndexT);
    hcfg.value_dtype    = c_.in_dtype;
    hcfg.max_threads    = 64;
    hcfg.max_value_size = in_row_bytes_;
    hcfg.num_rows       = c_.num_rows;
    hcfg.emb_table      = h_table_;       // full backing table, shared with the reference
    host_tab_ = std::make_shared<LinearHostTable<IndexT>>(hcfg);

    typename HierarchicalEmbeddingLayer<IndexT>::Config cfg;
    cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
    std::vector<table_ptr_t> tables{gpu_tab_, host_tab_};  // GPU tier must precede host tier
    layer_ = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(cfg, tables);
  }

  void BuildHost() {
    LinearHostTableConfig cfg;
    cfg.key_size = sizeof(IndexT);
    cfg.value_dtype    = c_.in_dtype;
    cfg.max_threads    = 64;
    cfg.max_value_size = in_row_bytes_;
    cfg.num_rows       = c_.num_rows;
    cfg.emb_table      = h_table_;
    host_tab_ = std::make_shared<LinearHostTable<IndexT>>(cfg);

    typename HostEmbeddingLayer<IndexT>::Config cfg_layer;
    cfg_layer.layer_name = "layer_pooling_host";
    layer_ = std::make_shared<HostEmbeddingLayer<IndexT>>(cfg_layer, host_tab_);
  }

  void BuildHierWithoutGPU() {
    // A two-tier host hierarchy: a bounded parallel hash map acts as a small cache in front of a
    // LinearHostTable containing every row. gpu_device_ stays -1, so the layer takes its CPU
    // pool/dequant path (pool_gathered_host), the same kernels the Host layer uses.
    const nlohmann::json phmap_cfg = {
        {"key_size", sizeof(IndexT)},
        {"max_value_size", in_row_bytes_},
        {"value_dtype", to_string(c_.in_dtype)},
        {"num_partitions", kHostCachePartitions},
        {"initial_capacity", kHostCacheRows / kHostCachePartitions},
        {"value_alignment", 32},
        {"allocation_rate", int64_t{1} << 16},
        {"overflow_policy",
         {{"overflow_margin", kHostCacheRows / kHostCachePartitions},
          {"handler", "evict_lru"},
          {"resolution_margin", 0.5}}}};
    table_factory_ptr_t phmap_factory{nve_test::plugin_factory("phmap")};
    host_cache_tab_ = std::dynamic_pointer_cast<HostTableLike>(phmap_factory->produce(4714, phmap_cfg));

    LinearHostTableConfig hcfg;

    hcfg.key_size = sizeof(IndexT);
    hcfg.value_dtype    = c_.in_dtype;
    hcfg.max_threads    = 64;
    hcfg.max_value_size = in_row_bytes_;
    hcfg.num_rows       = c_.num_rows;
    hcfg.emb_table      = h_table_;
    host_tab_ = std::make_shared<LinearHostTable<IndexT>>(hcfg);

    typename HierarchicalEmbeddingLayer<IndexT>::Config cfg;
    // Keep the default insert heuristic: the first lookup warms the PHMap cache from the complete
    // backing table. The final table's default threshold is zero, so it is never used as a cache.
    std::vector<table_ptr_t> tables{host_cache_tab_, host_tab_};
    layer_ = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(cfg, tables);
  }

  std::shared_ptr<GpuTable<IndexT>> MakeUvmGpuTable() {
    GPUTableConfig cfg;
    cfg.device_id        = kDeviceId;
    cfg.cache_size       = int64_t(1) << 20;
    cfg.row_size_in_bytes = in_row_bytes_;
    cfg.uvm_table        = h_table_;     // every key resolves via the UVM fallback
    cfg.uvm_num_rows     = c_.num_rows;
    cfg.count_misses     = true;         // required to report hitrates
    cfg.value_dtype      = c_.in_dtype;
    return std::make_shared<GpuTable<IndexT>>(cfg);
  }

  LayerPoolCase c_;
  int64_t in_row_bytes_{0};
  int64_t out_row_bytes_{0};
  int8_t* h_table_{nullptr};
  int8_t* d_table_{nullptr};
  std::shared_ptr<GpuTable<IndexT>> gpu_tab_;
  host_table_ptr_t host_cache_tab_;
  std::shared_ptr<LinearHostTable<IndexT>> host_tab_;
  std::shared_ptr<EmbeddingLayerBase> layer_;
  std::shared_ptr<MockHostTable<IndexT>> ref_;
  context_ptr_t ctx_;
};

// Dispatch on the key width chosen by the case.
inline void RunCase(const LayerPoolCase& c, bool expect_rejected = false) {
  if (c.key_is_int64) {
    PoolingHarness<int64_t> h(c);
    h.Run(expect_rejected);
  } else {
    PoolingHarness<int32_t> h(c);
    h.Run(expect_rejected);
  }
}

inline void RunPaddedStrideCase(const LayerPoolCase& c) {
  if (c.key_is_int64) {
    PoolingHarness<int64_t> h(c);
    h.RunPaddedStride();
  } else {
    PoolingHarness<int32_t> h(c);
    h.RunPaddedStride();
  }
}

inline void RunEmptyBagCsrCase(const LayerPoolCase& c) {
  if (c.key_is_int64) {
    PoolingHarness<int64_t> h(c);
    h.RunEmptyBagCsr();
  } else {
    PoolingHarness<int32_t> h(c);
    h.RunEmptyBagCsr();
  }
}

}  // namespace layer_pooling
}  // namespace nve
