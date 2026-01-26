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

#include <common.hpp>
#include <logging.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nve {

class LogCapture {
 public:
  LogCapture() {
    original_cout_ = std::cout.rdbuf(captured_cout_.rdbuf());
    original_cerr_ = std::cerr.rdbuf(captured_cerr_.rdbuf());
  }
  ~LogCapture() {
    std::cout.rdbuf(original_cout_);
    std::cerr.rdbuf(original_cerr_);
  }
  std::string GetCoutString() const { return captured_cout_.str(); }
  std::string GetCerrString() const { return captured_cerr_.str(); }
  void DumpLog() {
    // Use this for debug
    // Intentionally using printf to circumvent cout/cerr
    printf("[*COUT START*]\n%s[*COUT END*]\n", GetCoutString().c_str());
    printf("[*CERR START*]\n%s[*CERR END*]\n", GetCerrString().c_str());
  }

 private:
  std::stringstream captured_cout_;
  std::stringstream captured_cerr_;
  std::streambuf* original_cout_;
  std::streambuf* original_cerr_;
};

struct LogTestParams {
  std::string message;
  bool appears_in_cout;
};

// Order is aligned with verbosity level enum values.
std::map<LogLevel_t, LogTestParams> test_params = {
    {LogLevel_t::Critical, {"Critical Message!", false}},
    {LogLevel_t::Error, {"Error Message!", false}},
    {LogLevel_t::Warning, {"Warning Message!", true}},
    {LogLevel_t::Perf, {"Perf Message!", true}},
    {LogLevel_t::Info, {"Info Message!", true}},
    {LogLevel_t::Verbose, {"Verbose Message!", true}},
};

static void LogMessages(std::vector<LogLevel_t> msg_levels) {
  for (auto lvl : msg_levels) {
    const auto msg = test_params.at(lvl).message;
    switch (lvl) {
      case LogLevel_t::Critical:
        NVE_LOG_CRITICAL_(msg);
        break;
      case LogLevel_t::Error:
        NVE_LOG_ERROR_(msg);
        break;
      case LogLevel_t::Warning:
        NVE_LOG_WARNING_(msg);
        break;
      case LogLevel_t::Perf:
        NVE_LOG_PERF_(msg);
        break;
      case LogLevel_t::Info:
        NVE_LOG_INFO_(msg);
        break;
      case LogLevel_t::Verbose:
        NVE_LOG_VERBOSE_(msg);
        break;

      default:
        throw std::invalid_argument("Invalid verbosity level");
        break;
    }
  }
}

static void LogAllMessages() {
  LogMessages({
      LogLevel_t::Critical,
      LogLevel_t::Error,
      LogLevel_t::Warning,
      LogLevel_t::Perf,
      LogLevel_t::Info,
      LogLevel_t::Verbose,
  });
}

static void LogAndTestSingleMessage(LogLevel_t lvl) {
  GetGlobalLogger()->set_verbosity_level(lvl);
  LogCapture lc;
  LogMessages({lvl});
  const auto msg = test_params.at(lvl).message;

  if (test_params.at(lvl).appears_in_cout) {
    EXPECT_EQ("", lc.GetCerrString());
    EXPECT_TRUE(lc.GetCoutString().find(msg) != std::string::npos);
  } else {
    EXPECT_EQ("", lc.GetCoutString());
    EXPECT_TRUE(lc.GetCerrString().find(msg) != std::string::npos);
  }
}

static void LogAndTestVerbosity(LogLevel_t test_lvl) {
  GetGlobalLogger()->set_verbosity_level(test_lvl);
  LogCapture lc;
  LogAllMessages();
  for (const auto& [lvl, param] : test_params) {
    std::string primary_output;
    std::string secondary_output;
    if (param.appears_in_cout) {
      primary_output = lc.GetCoutString();
      secondary_output = lc.GetCerrString();
    } else {
      primary_output = lc.GetCerrString();
      secondary_output = lc.GetCoutString();
    }
    const auto& msg = param.message;
    if (lvl <= test_lvl) {
      EXPECT_TRUE(primary_output.find(msg) != std::string::npos);
    } else {
      EXPECT_FALSE(primary_output.find(msg) != std::string::npos);
    }
    EXPECT_FALSE(secondary_output.find(msg) != std::string::npos);
  }
}

// Test log capture class works (intercept cout and cerr)
TEST(logger_test, check_capture) {
  LogCapture lc;
  std::string log_str = "This is a log string!";
  std::string err_str = "This is an error string!";
  std::cout << log_str;
  std::cerr << err_str;

  EXPECT_EQ(log_str, lc.GetCoutString());
  EXPECT_EQ(err_str, lc.GetCerrString());
}

// Sanity test for logging critical error messages
TEST(logger_test, log_critical_sanity) { LogAndTestSingleMessage(LogLevel_t::Critical); }

// Sanity test for logging error messages
TEST(logger_test, log_error_sanity) { LogAndTestSingleMessage(LogLevel_t::Error); }

// Sanity test for logging warning messages
TEST(logger_test, log_warning_sanity) { LogAndTestSingleMessage(LogLevel_t::Warning); }

// Sanity test for logging perf messages
TEST(logger_test, log_perf_sanity) { LogAndTestSingleMessage(LogLevel_t::Perf); }

// Sanity test for logging info messages
TEST(logger_test, log_info_sanity) { LogAndTestSingleMessage(LogLevel_t::Info); }

// Sanity test for logging verbose messages
TEST(logger_test, log_verbose_sanity) { LogAndTestSingleMessage(LogLevel_t::Verbose); }

// Test critical verbosity is obeyed
TEST(logger_test, verbosity_critical) { LogAndTestVerbosity(LogLevel_t::Critical); }

// Test error verbosity is obeyed
TEST(logger_test, verbosity_error) { LogAndTestVerbosity(LogLevel_t::Error); }

// Test warning verbosity is obeyed
TEST(logger_test, verbosity_warning) { LogAndTestVerbosity(LogLevel_t::Warning); }

// Test perf verbosity is obeyed
TEST(logger_test, verbosity_perf) { LogAndTestVerbosity(LogLevel_t::Perf); }

// Test info verbosity is obeyed
TEST(logger_test, verbosity_info) { LogAndTestVerbosity(LogLevel_t::Info); }

// Test verbose verbosity is obeyed
TEST(logger_test, verbosity_verbose) { LogAndTestVerbosity(LogLevel_t::Verbose); }

}  // namespace nve
