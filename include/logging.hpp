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

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace nve {

enum class LogLevel_t : uint64_t {
  Critical = 0,
  Error = 10,
  Warning = 20,
  Perf = 30,
  Info = 40,
  Verbose = 100,
};

inline std::ostream& operator<<(std::ostream& o, const LogLevel_t& ll) {
  return o << static_cast<int64_t>(ll);
}

/**
 * Logger backend interface.
 *
 * An application may derive from this interface and replace the backend implenmentation by calling
 * Logger::setBackend(). Logger backends do not need to filter logs, or guarantee thread safety. The
 * Logger class handles this for the backend.
 */
class LoggerBackend {
 public:
  LoggerBackend() = default;

  virtual ~LoggerBackend() = default;

  /**
   * Log message with a given verbosity level.
   */
  virtual void log(LogLevel_t level, const std::string_view& msg) noexcept = 0;
};

class DefaultLoggerBackend : public LoggerBackend {
 public:
  DefaultLoggerBackend(const std::string& prefix) : prefix_{prefix} {}
  ~DefaultLoggerBackend() override {}
  void log(const LogLevel_t level, const std::string_view& msg) noexcept override {
    std::ostringstream o;
    o << prefix_ << '[';
    switch (level) {
      case LogLevel_t::Critical:
        o << 'C';
        break;
      case LogLevel_t::Error:
        o << 'E';
        break;
      case LogLevel_t::Warning:
        o << 'W';
        break;
      case LogLevel_t::Info:
        o << 'I';
        break;
      case LogLevel_t::Verbose:
        o << 'V';
        break;
      case LogLevel_t::Perf:
        o << 'P';
        break;
      default:
        o << '*';
        break;
    }
    o << "] " << msg;
    if (level <= LogLevel_t::Error) {
      std::cerr << o.str() << std::endl;
    } else {
      std::cout << o.str() << std::endl;
    }
  }
 protected:
  const std::string prefix_;
};

using LogMessage = std::pair<LogLevel_t, std::string>;

class Logger final {
 public:
  Logger() :
    verbosity_level_(LogLevel_t::Warning),
    logger_backend_(std::make_shared<DefaultLoggerBackend>("[NVE]")) 
    {
      setVerbosityLevel(getEnvVerbosityLevel());
    }

  /**
   * Set the Logger verbosity level. Only messages on the same level, or a lower will be logged.
   */
  inline void setVerbosityLevel(const LogLevel_t level) {
    verbosity_level_ = level;
  }

  inline LogLevel_t getVerbosityLevel() noexcept { return verbosity_level_; }

  /**
   * Set the logger backend (override the default one).
   * Setting logger backend to nullptr will disable logging.
   */
  inline void setBackend(std::shared_ptr<LoggerBackend>& logger_backend) {
    std::lock_guard lock(mutex_);
    logger_backend_ = logger_backend;
  }

  inline std::shared_ptr<LoggerBackend> getBackend() noexcept {
    std::lock_guard lock(mutex_);
    return logger_backend_;
  }

  /**
   * Log a message with a given verbosity.
   */
  inline void log(const LogLevel_t level, const std::string_view& msg) {
    if (logger_backend_ && level <= verbosity_level_) {
      std::lock_guard lock(mutex_);
      logger_backend_->log(level, msg);
    }
  }

  inline void log(const LogMessage& lm) { log(lm.first, lm.second); }
 private:
  LogLevel_t getEnvVerbosityLevel() {
    const char* env_var = std::getenv("NVE_LOG_LEVEL");
    if (env_var) {
      if (std::strncmp(env_var, "CRITICAL", std::strlen("CRITICAL")) == 0) {
        return LogLevel_t::Critical;
      } else if (std::strncmp(env_var, "ERROR", std::strlen("ERROR")) == 0) {
        return LogLevel_t::Error;
      } else if (std::strncmp(env_var, "WARNING", std::strlen("WARNING")) == 0) {
        return LogLevel_t::Warning;
      } else if (std::strncmp(env_var, "INFO", std::strlen("INFO")) == 0) {
        return LogLevel_t::Info;
      } else if (std::strncmp(env_var, "VERBOSE", std::strlen("VERBOSE")) == 0) {
        return LogLevel_t::Verbose;
      } else if (std::strncmp(env_var, "PERF", std::strlen("PERF")) == 0) {
        return LogLevel_t::Perf;
      }
      // log should make sure backend was initialized if not nothing will be printed
      log(LogLevel_t::Warning, "Invalid log level from env var: " + std::string(env_var));
      return LogLevel_t::Warning;
    }
    return LogLevel_t::Warning;
  }
 private:
  LogLevel_t verbosity_level_ = LogLevel_t::Warning;
  std::mutex mutex_;
  std::shared_ptr<LoggerBackend> logger_backend_;
};

}  // namespace nve
