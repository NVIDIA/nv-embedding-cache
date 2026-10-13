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

#include "batch_reader.hpp"

#include <common.hpp>

#include <linux/aio_abi.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// io_uring needs Linux 5.6+ UAPI headers (IORING_FEAT_RW_CUR_POS came with IORING_OP_READ); older ones build without it.
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#endif
#if defined(IORING_FEAT_RW_CUR_POS) && defined(__NR_io_uring_setup)
#define NVE_LINEAR_FILE_HAS_IO_URING
#endif

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <vector>

namespace nve {
namespace plugin {

BatchReader::BatchReader(const size_t staging_bytes, const size_t depth)
    : staging_(static_cast<int8_t*>(std::aligned_alloc(kBlockSize, staging_bytes))), spans_(depth) {
  NVE_CHECK_(staging_ != nullptr, "LinearFileTable: failed to allocate ", staging_bytes, " bytes of staging");
}

namespace {

#ifdef NVE_LINEAR_FILE_HAS_IO_URING
// io_uring over raw syscalls (no liburing dependency): one ring, reads keep it full until done.
class UringReader final : public BatchReader {
 public:
  UringReader(const size_t staging_bytes, const size_t depth) : BatchReader(staging_bytes, depth) {}
  NVE_PREVENT_COPY_AND_MOVE_(UringReader);

  ~UringReader() override {
    if (sqes_ != MAP_FAILED) ::munmap(sqes_, sqes_bytes_);
    if (cq_ring_ != MAP_FAILED && cq_ring_ != sq_ring_) ::munmap(cq_ring_, cq_ring_bytes_);
    if (sq_ring_ != MAP_FAILED) ::munmap(sq_ring_, sq_ring_bytes_);
    if (ring_fd_ >= 0) ::close(ring_fd_);
  }

  bool init(size_t depth) {
    io_uring_params params{};
    ring_fd_ = static_cast<int>(::syscall(__NR_io_uring_setup, static_cast<unsigned>(depth), &params));
    // IORING_OP_READ arrived together with IORING_FEAT_RW_CUR_POS (Linux 5.6).
    if (ring_fd_ < 0 || (params.features & IORING_FEAT_RW_CUR_POS) == 0) return false;

    sq_entries_ = params.sq_entries;
    sq_ring_bytes_ = params.sq_off.array + params.sq_entries * sizeof(unsigned);
    cq_ring_bytes_ = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
    const bool single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (single_mmap) sq_ring_bytes_ = cq_ring_bytes_ = std::max(sq_ring_bytes_, cq_ring_bytes_);
    sq_ring_ = map(sq_ring_bytes_, IORING_OFF_SQ_RING);
    cq_ring_ = single_mmap ? sq_ring_ : map(cq_ring_bytes_, IORING_OFF_CQ_RING);
    sqes_bytes_ = params.sq_entries * sizeof(io_uring_sqe);
    sqes_ = map(sqes_bytes_, IORING_OFF_SQES);
    if (sq_ring_ == MAP_FAILED || cq_ring_ == MAP_FAILED || sqes_ == MAP_FAILED) return false;

    auto* sq = static_cast<char*>(sq_ring_);
    auto* cq = static_cast<char*>(cq_ring_);
    sq_head_ = reinterpret_cast<unsigned*>(sq + params.sq_off.head);
    sq_tail_ = reinterpret_cast<unsigned*>(sq + params.sq_off.tail);
    sq_mask_ = *reinterpret_cast<unsigned*>(sq + params.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<unsigned*>(sq + params.sq_off.array);
    cq_head_ = reinterpret_cast<unsigned*>(cq + params.cq_off.head);
    cq_tail_ = reinterpret_cast<unsigned*>(cq + params.cq_off.tail);
    cq_mask_ = *reinterpret_cast<unsigned*>(cq + params.cq_off.ring_mask);
    cqes_ = reinterpret_cast<io_uring_cqe*>(cq + params.cq_off.cqes);
    return true;
  }

  IoEngine engine() const override { return IoEngine::IoUring; }

  bool abandoned_reads() const override { return abandoned_reads_; }

  void read(const int fd, ReadSpan* spans, const size_t n) override {
    const unsigned first_head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);  // the ring is empty between calls
    size_t queued = 0, completed = 0;
    while (completed < n) {
      // In-flight reads never exceed the SQ size, so the CQ (twice as large) cannot overflow.
      unsigned tail = *sq_tail_;
      for (; queued < n && queued - completed < sq_entries_; ++queued, ++tail) {
        const unsigned idx = tail & sq_mask_;
        io_uring_sqe& sqe = static_cast<io_uring_sqe*>(sqes_)[idx];
        std::memset(&sqe, 0, sizeof(sqe));
        sqe.opcode = IORING_OP_READ;
        sqe.fd = fd;
        sqe.addr = reinterpret_cast<uint64_t>(spans[queued].dst);
        sqe.len = static_cast<uint32_t>(spans[queued].len);
        sqe.off = static_cast<uint64_t>(spans[queued].offset);
        sqe.user_data = queued;
        sq_array_[idx] = idx;
      }
      __atomic_store_n(sq_tail_, tail, __ATOMIC_RELEASE);

      // Once everything is queued wait for all of it, otherwise for a free slot.
      const unsigned to_submit = tail - __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
      const auto wait_for = static_cast<unsigned>(queued == n ? queued - completed : 1);
      if (!enter(to_submit, wait_for)) {
        const int err = errno;
        // Reads the kernel already took would otherwise land in the span buffers after we return.
        abandoned_reads_ = !drain(spans, first_head, completed);
        NVE_THROW_("LinearFileTable: io_uring_enter failed: ", std::strerror(err));
      }
      completed += reap(spans);
    }
  }

 private:
  // Submits and waits; false on errors other than an interrupted or busy ring (errno is kept).
  bool enter(const unsigned to_submit, const unsigned wait_for) {
    return ::syscall(__NR_io_uring_enter, ring_fd_, to_submit, wait_for, IORING_ENTER_GETEVENTS, nullptr, 0) >= 0 ||
           errno == EINTR || errno == EAGAIN || errno == EBUSY;
  }

  // Moves the completed reads' results into `spans`; returns how many there were.
  size_t reap(ReadSpan* spans) {
    unsigned head = *cq_head_;
    const unsigned cq_tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
    size_t reaped = 0;
    for (; head != cq_tail; ++head, ++reaped) {
      const io_uring_cqe& cqe = cqes_[head & cq_mask_];
      spans[cqe.user_data].result = cqe.res;
    }
    __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
    return reaped;
  }

  // Waits for every read the kernel took since `first_head`; false if the ring cannot even wait.
  bool drain(ReadSpan* spans, const unsigned first_head, size_t completed) {
    const auto submitted = static_cast<size_t>(__atomic_load_n(sq_head_, __ATOMIC_ACQUIRE) - first_head);
    for (completed += reap(spans); completed < submitted; completed += reap(spans)) {
      if (!enter(0, static_cast<unsigned>(submitted - completed))) return false;
    }
    return true;
  }

  void* map(size_t bytes, uint64_t offset) const {
    return ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd_, static_cast<off_t>(offset));
  }

  int ring_fd_{-1};
  unsigned sq_entries_{0};
  size_t sq_ring_bytes_{0}, cq_ring_bytes_{0}, sqes_bytes_{0};
  void* sq_ring_{MAP_FAILED};
  void* cq_ring_{MAP_FAILED};
  void* sqes_{MAP_FAILED};
  unsigned *sq_head_{nullptr}, *sq_tail_{nullptr}, *sq_array_{nullptr};
  unsigned *cq_head_{nullptr}, *cq_tail_{nullptr};
  unsigned sq_mask_{0}, cq_mask_{0};
  io_uring_cqe* cqes_{nullptr};
  bool abandoned_reads_{false};
};
#endif

// Linux AIO: io_submit + io_getevents. Truly asynchronous only on O_DIRECT descriptors.
class AioReader final : public BatchReader {
 public:
  AioReader(const size_t staging_bytes, const size_t depth) : BatchReader(staging_bytes, depth) {}
  NVE_PREVENT_COPY_AND_MOVE_(AioReader);

  ~AioReader() override {
    if (ctx_ != 0) ::syscall(__NR_io_destroy, ctx_);
  }

  bool init(size_t depth) {
    depth_ = depth;
    iocbs_.resize(depth);
    iocb_ptrs_.resize(depth);
    events_.resize(depth);
    return ::syscall(__NR_io_setup, static_cast<unsigned>(depth), &ctx_) == 0;
  }

  IoEngine engine() const override { return IoEngine::Aio; }

  void read(const int fd, ReadSpan* spans, const size_t n) override {
    size_t queued = 0, completed = 0;
    while (completed < n) {
      // The kernel copies the iocbs during io_submit, so the scratch array is reused every round.
      size_t batch = 0;
      for (; queued + batch < n && queued + batch - completed < depth_; ++batch) {
        const ReadSpan& span = spans[queued + batch];
        iocb& cb = iocbs_[batch];
        std::memset(&cb, 0, sizeof(cb));
        cb.aio_data = queued + batch;
        cb.aio_fildes = static_cast<uint32_t>(fd);
        cb.aio_lio_opcode = IOCB_CMD_PREAD;
        cb.aio_buf = reinterpret_cast<uint64_t>(span.dst);
        cb.aio_nbytes = span.len;
        cb.aio_offset = span.offset;
        iocb_ptrs_[batch] = &cb;
      }
      if (batch > 0) {
        const long submitted = ::syscall(__NR_io_submit, ctx_, static_cast<long>(batch), iocb_ptrs_.data());
        if (submitted > 0) {
          queued += static_cast<size_t>(submitted);
        } else if (errno != EINTR && !(errno == EAGAIN && queued > completed)) {
          // The first iocb was rejected outright (e.g. EINVAL); report it and carry on with the rest.
          spans[queued++].result = -errno;
          ++completed;
          continue;
        }
      }
      if (queued == completed) continue;

      const auto in_flight = static_cast<long>(queued - completed);
      const long min_events = queued == n ? in_flight : 1;
      const long got = ::syscall(__NR_io_getevents, ctx_, min_events, in_flight, events_.data(), nullptr);
      if (got < 0) {
        NVE_CHECK_(errno == EINTR, "LinearFileTable: io_getevents failed: ", std::strerror(errno));
        continue;
      }
      for (long e = 0; e < got; ++e, ++completed) {
        spans[events_[static_cast<size_t>(e)].data].result = events_[static_cast<size_t>(e)].res;
      }
    }
  }

 private:
  aio_context_t ctx_{0};
  size_t depth_{0};
  std::vector<iocb> iocbs_;
  std::vector<iocb*> iocb_ptrs_;
  std::vector<io_event> events_;
};

class PreadReader final : public BatchReader {
 public:
  PreadReader(const size_t staging_bytes, const size_t depth) : BatchReader(staging_bytes, depth) {}

  IoEngine engine() const override { return IoEngine::Pread; }

  void read(const int fd, ReadSpan* spans, const size_t n) override {
    for (size_t i = 0; i < n; ++i) {
      ssize_t got;
      do {
        got = ::pread(fd, spans[i].dst, spans[i].len, spans[i].offset);
      } while (got < 0 && errno == EINTR);
      spans[i].result = got < 0 ? -errno : got;
    }
  }
};

}  // namespace

std::unique_ptr<BatchReader> make_batch_reader(const IoEngine engine, const size_t depth, const size_t staging_bytes) {
  switch (engine) {
    case IoEngine::IoUring: {
#ifdef NVE_LINEAR_FILE_HAS_IO_URING
      auto reader = std::make_unique<UringReader>(staging_bytes, depth);
      return reader->init(depth) ? std::move(reader) : nullptr;
#else
      return nullptr;  // built against pre-5.6 kernel headers
#endif
    }
    case IoEngine::Aio: {
      auto reader = std::make_unique<AioReader>(staging_bytes, depth);
      return reader->init(depth) ? std::move(reader) : nullptr;
    }
    case IoEngine::Pread:
      return std::make_unique<PreadReader>(staging_bytes, depth);
  }
  return nullptr;
}

}  // namespace plugin
}  // namespace nve
