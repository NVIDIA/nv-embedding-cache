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

#include <linear_file_table.hpp>

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace nve {
namespace plugin {

constexpr int64_t kBlockSize = 4096;  // O_DIRECT alignment unit

// One read of a batch.
struct ReadSpan {
  off_t offset;    // file offset
  size_t len;      // bytes to read
  int8_t* dst;
  int64_t first;   // caller data: the rows [first, last) this read serves
  int64_t last;
  int64_t result;  // set by the reader: bytes read, or -errno
};

// Issues batches of reads for one thread (not thread-safe).
class BatchReader {
 public:
  virtual ~BatchReader() = default;

  // Scratch for one I/O thread. The reader owns it so that reads it abandoned can only land in memory
  // that is never handed out again (the reader is leaked instead of destroyed).
  int8_t* staging() const { return staging_.get(); }  // block aligned
  ReadSpan* spans() { return spans_.data(); }          // `depth` entries

  // Reads every span and returns once all of them completed; per-read failures and short reads are
  // reported through `result`, not thrown. Throws only when the engine itself fails, after waiting
  // for the reads already issued, so none of them writes into a span buffer later.
  virtual void read(int fd, ReadSpan* spans, size_t n) = 0;

  virtual IoEngine engine() const = 0;

  // True after read() threw without being able to wait for its reads: the kernel may still write
  // into those span buffers, so they must not be reused.
  virtual bool abandoned_reads() const { return false; }

 protected:
  // `staging_bytes` must be a multiple of kBlockSize.
  BatchReader(size_t staging_bytes, size_t depth);

 private:
  struct FreeDeleter {
    void operator()(int8_t* ptr) const { std::free(ptr); }
  };
  std::unique_ptr<int8_t, FreeDeleter> staging_;
  std::vector<ReadSpan> spans_;
};

// Returns nullptr when the engine is unavailable (kernel support, seccomp, limits). At most `depth`
// reads are in flight at once; the reader owns a staging buffer of `staging_bytes`.
std::unique_ptr<BatchReader> make_batch_reader(IoEngine engine, size_t depth, size_t staging_bytes);

}  // namespace plugin
}  // namespace nve
