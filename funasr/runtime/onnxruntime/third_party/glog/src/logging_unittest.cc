// Copyright (c) 2023, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: Ray Sidney

#include "config.h"
#include "utilities.h"

#include <fcntl.h>
#ifdef HAVE_GLOB_H
# include <glob.h>
#endif
#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_WAIT_H
# include <sys/wait.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "base/commandlineflags.h"
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include "googletest.h"

DECLARE_string(log_backtrace_at);  // logging.cc

#ifdef HAVE_LIB_GFLAGS
#include <gflags/gflags.h>
using namespace GFLAGS_NAMESPACE;
#endif

#ifdef HAVE_LIB_GMOCK
#include <gmock/gmock.h>
#include "mock-log.h"
// Introduce several symbols from gmock.
using testing::_;
using testing::AnyNumber;
using testing::HasSubstr;
using testing::AllOf;
using testing::StrNe;
using testing::StrictMock;
using testing::InitGoogleMock;
using GOOGLE_NAMESPACE::glog_testing::ScopedMockLog;
#endif

using namespace std;
using namespace GOOGLE_NAMESPACE;

// Some non-advertised functions that we want to test or use.
_START_GOOGLE_NAMESPACE_
namespace base {
namespace internal {
bool GetExitOnDFatal();
void SetExitOnDFatal(bool value);
}  // namespace internal
}  // namespace base
_END_GOOGLE_NAMESPACE_

static void TestLogging(bool check_counts);
static void TestRawLogging();
static void LogWithLevels(int v, int severity, bool err, bool alsoerr);
static void TestLoggingLevels();
static void TestVLogModule();
static void TestLogString();
static void TestLogSink();
static void TestLogToString();
static void TestLogSinkWaitTillSent();
static void TestCHECK();
static void TestDCHECK();
static void TestSTREQ();
static void TestBasename();
static void TestBasenameAppendWhenNoTimestamp();
static void TestTwoProcessesWrite();
static void TestSymlink();
static void TestExtension();
static void TestWrapper();
static void TestErrno();
static void TestTruncate();
static void TestCustomLoggerDeletionOnShutdown();
static void TestLogPeriodically();

static int x = -1;
static void BM_Check1(int n) {
  while (n-- > 0) {
    CHECK_GE(n, x);
    CHECK_GE(n, x);
    CHECK_GE(n, x);
    CHECK_GE(n, x);
    CHECK_GE(n, x);
    CHECK_GE(n, x);
    CHECK_GE(n, x);
    CHECK_GE(n, x);
  }
}
BENCHMARK(BM_Check1)

static void CheckFailure(int a, int b, const char* file, int line, const char* msg);
static void BM_Check3(int n) {
  while (n-- > 0) {
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
    if (n < x) CheckFailure(n, x, __FILE__, __LINE__, "n < x");
  }
}
BENCHMARK(BM_Check3)

static void BM_Check2(int n) {
  if (n == 17) {
    x = 5;
  }
  while (n-- > 0) {
    CHECK(n >= x);
    CHECK(n >= x);
    CHECK(n >= x);
    CHECK(n >= x);
    CHECK(n >= x);
    CHECK(n >= x);
    CHECK(n >= x);
    CHECK(n >= x);
  }
}
BENCHMARK(BM_Check2)

static void CheckFailure(int, int, const char* /* file */, int /* line */,
                         const char* /* msg */) {
}

static void BM_logspeed(int n) {
  while (n-- > 0) {
    LOG(INFO) << "test message";
  }
}
BENCHMARK(BM_logspeed)

static void BM_vlog(int n) {
  while (n-- > 0) {
    VLOG(1) << "test message";
  }
}
BENCHMARK(BM_vlog)

// Dynamically generate a prefix using the default format and write it to the stream.
void PrefixAttacher(std::ostream &s, const LogMessageInfo &l, void* data) {
  // Assert that `data` contains the expected contents before producing the
  // prefix (otherwise causing the tests to fail):
  if (data == nullptr || *static_cast<string*>(data) != "good data") {
    return;
  }

  s << l.severity[0]
    << setw(4) << 1900 + l.time.year()
    << setw(2) << 1 + l.time.month()
    << setw(2) << l.time.day()
    << ' '
    << setw(2) << l.time.hour() << ':'
    << setw(2) << l.time.min()  << ':'
    << setw(2) << l.time.sec() << "."
    << setw(6) << l.time.usec()
    << ' '
    << setfill(' ') << setw(5)
    << l.thread_id << setfill('0')
    << ' '
    << l.filename << ':' << l.line_number << "]";
}

int main(int argc, char **argv) {
  FLAGS_colorlogtostderr = false;
  FLAGS_timestamp_in_logfile_name = true;

  // Make sure stderr is not buffered as stderr seems to be buffered
  // on recent windows.
  setbuf(stderr, nullptr);

  // Test some basics before InitGoogleLogging:
  CaptureTestStderr();
  LogWithLevels(FLAGS_v, FLAGS_stderrthreshold,
                FLAGS_logtostderr, FLAGS_alsologtostderr);
  LogWithLevels(0, 0, false, false);  // simulate "before global c-tors"
  const string early_stderr = GetCapturedTestStderr();

  EXPECT_FALSE(IsGoogleLoggingInitialized());

  // Setting a custom prefix generator (it will use the default format so that
  // the golden outputs can be reused):
  string prefix_attacher_data = "good data";
  InitGoogleLogging(argv[0], &PrefixAttacher, static_cast<void*>(&prefix_attacher_data));

  EXPECT_TRUE(IsGoogleLoggingInitialized());

  RunSpecifiedBenchmarks();

  FLAGS_logtostderr = true;

  InitGoogleTest(&argc, argv);
#ifdef HAVE_LIB_GMOCK
  InitGoogleMock(&argc, argv);
#endif

#ifdef HAVE_LIB_GFLAGS
  ParseCommandLineFlags(&argc, &argv, true);
#endif

  // so that death tests run before we use threads
  CHECK_EQ(RUN_ALL_TESTS(), 0);

  CaptureTestStderr();

  // re-emit early_stderr
  LogMessage("dummy", LogMessage::kNoLogPrefix, GLOG_INFO).stream() << early_stderr;

  TestLogging(true);
  TestRawLogging();
  TestLoggingLevels();
  TestVLogModule();
  TestLogString();
  TestLogSink();
  TestLogToString();
  TestLogSinkWaitTillSent();
  TestCHECK();
  TestDCHECK();
  TestSTREQ();

  // TODO: The golden test portion of this test is very flakey.
  EXPECT_TRUE(
      MungeAndDiffTestStderr(FLAGS_test_srcdir + "/src/logging_unittest.err"));

  FLAGS_logtostderr = false;

  FLAGS_logtostdout = true;
  FLAGS_stderrthreshold = NUM_SEVERITIES;
  CaptureTestStdout();
  TestRawLogging();
  TestLoggingLevels();
  TestLogString();
  TestLogSink();
  TestLogToString();
  TestLogSinkWaitTillSent();
  TestCHECK();
  TestDCHECK();
  TestSTREQ();
  EXPECT_TRUE(
      MungeAndDiffTestStdout(FLAGS_test_srcdir + "/src/logging_unittest.out"));
  FLAGS_logtostdout = false;

  TestBasename();
  TestBasenameAppendWhenNoTimestamp();
  TestTwoProcessesWrite();
  TestSymlink();
  TestExtension();
  TestWrapper();
  TestErrno();
  TestTruncate();
  TestCustomLoggerDeletionOnShutdown();
  TestLogPeriodically();

  fprintf(stdout, "PASS\n");
  return 0;
}

void TestLogging(bool check_counts) {
  int64 base_num_infos   = LogMessage::num_messages(GLOG_INFO);
  int64 base_num_warning = LogMessage::num_messages(GLOG_WARNING);
  int64 base_num_errors  = LogMessage::num_messages(GLOG_ERROR);

  LOG(INFO) << string("foo ") << "bar " << 10 << ' ' << 3.4;
  for ( int i = 0; i < 10; ++i ) {
    int old_errno = errno;
    errno = i;
    PLOG_EVERY_N(ERROR, 2) << "Plog every 2, iteration " << COUNTER;
    errno = old_errno;

    LOG_EVERY_N(ERROR, 3) << "Log every 3, iteration " << COUNTER << endl;
    LOG_EVERY_N(ERROR, 4) << "Log every 4, iteration " << COUNTER << endl;

    LOG_IF_EVERY_N(WARNING, true, 5) << "Log if every 5, iteration " << COUNTER;
    LOG_IF_EVERY_N(WARNING, false, 3)
        << "Log if every 3, iteration " << COUNTER;
    LOG_IF_EVERY_N(INFO, true, 1) << "Log if every 1, iteration " << COUNTER;
    LOG_IF_EVERY_N(ERROR, (i < 3), 2)
        << "Log if less than 3 every 2, iteration " << COUNTER;
  }
  LOG_IF(WARNING, true) << "log_if this";
  LOG_IF(WARNING, false) << "don't log_if this";

  char s[] = "array";
  LOG(INFO) << s;
  const char const_s[] = "const array";
  LOG(INFO) << const_s;
  int j = 1000;
  LOG(ERROR) << string("foo") << ' '<< j << ' ' << setw(10) << j << " "
             << setw(1) << hex << j;
  LOG(INFO) << "foo " << std::setw(10) << 1.0;

  {
    google::LogMessage outer(__FILE__, __LINE__, GLOG_ERROR);
    outer.stream() << "outer";

    LOG(ERROR) << "inner";
  }

  LogMessage("foo", LogMessage::kNoLogPrefix, GLOG_INFO).stream() << "no prefix";

  if (check_counts) {
    CHECK_EQ(base_num_infos   + 15, LogMessage::num_messages(GLOG_INFO));
    CHECK_EQ(base_num_warning + 3,  LogMessage::num_messages(GLOG_WARNING));
    CHECK_EQ(base_num_errors  + 17, LogMessage::num_messages(GLOG_ERROR));
  }
}

static void NoAllocNewHook() {
  LOG(FATAL) << "unexpected new";
}

struct NewHook {
  NewHook() {
    g_new_hook = &NoAllocNewHook;
  }
  ~NewHook() { g_new_hook = nullptr; }
};

TEST(DeathNoAllocNewHook, logging) {
  // tests that NewHook used below works
  NewHook new_hook;
  ASSERT_DEATH({
    new int;
  }, "unexpected new");
}

void TestRawLogging() {
  auto* foo = new string("foo ");
  string huge_str(50000, 'a');

  FlagSaver saver;

  // Check that RAW loggging does not use mallocs.
  NewHook new_hook;

  RAW_LOG(INFO, "%s%s%d%c%f", foo->c_str(), "bar ", 10, ' ', 3.4);
  char s[] = "array";
  RAW_LOG(WARNING, "%s", s);
  const char const_s[] = "const array";
  RAW_LOG(INFO, "%s", const_s);
  void* p = reinterpret_cast<void*>(PTR_TEST_VALUE);
  RAW_LOG(INFO, "ptr %p", p);
  p = nullptr;
  RAW_LOG(INFO, "ptr %p", p);
  int j = 1000;
  RAW_LOG(ERROR, "%s%d%c%010d%s%1x", foo->c_str(), j, ' ', j, " ", j);
  RAW_VLOG(0, "foo %d", j);

#if defined(NDEBUG)
  RAW_LOG(INFO, "foo %d", j);  // so that have same stderr to compare
#else
  RAW_DLOG(INFO, "foo %d", j);  // test RAW_DLOG in debug mode
#endif

  // test how long messages are chopped:
  RAW_LOG(WARNING, "Huge string: %s", huge_str.c_str());
  RAW_VLOG(0, "Huge string: %s", huge_str.c_str());

  FLAGS_v = 0;
  RAW_LOG(INFO, "log");
  RAW_VLOG(0, "vlog 0 on");
  RAW_VLOG(1, "vlog 1 off");
  RAW_VLOG(2, "vlog 2 off");
  RAW_VLOG(3, "vlog 3 off");
  FLAGS_v = 2;
  RAW_LOG(INFO, "log");
  RAW_VLOG(1, "vlog 1 on");
  RAW_VLOG(2, "vlog 2 on");
  RAW_VLOG(3, "vlog 3 off");

#if defined(NDEBUG)
  RAW_DCHECK(1 == 2, " RAW_DCHECK's shouldn't be compiled in normal mode");
#endif

  RAW_CHECK(1 == 1, "should be ok");
  RAW_DCHECK(true, "should be ok");

  delete foo;
}

void LogWithLevels(int v, int severity, bool err, bool alsoerr) {
  RAW_LOG(INFO,
          "Test: v=%d stderrthreshold=%d logtostderr=%d alsologtostderr=%d",
          v, severity, err, alsoerr);

  FlagSaver saver;

  FLAGS_v = v;
  FLAGS_stderrthreshold = severity;
  FLAGS_logtostderr = err;
  FLAGS_alsologtostderr = alsoerr;

  RAW_VLOG(-1, "vlog -1");
  RAW_VLOG(0, "vlog 0");
  RAW_VLOG(1, "vlog 1");
  RAW_LOG(INFO, "log info");
  RAW_LOG(WARNING, "log warning");
  RAW_LOG(ERROR, "log error");

  VLOG(-1) << "vlog -1";
  VLOG(0) << "vlog 0";
  VLOG(1) << "vlog 1";
  LOG(INFO) << "log info";
  LOG(WARNING) << "log warning";
  LOG(ERROR) << "log error";

  VLOG_IF(-1, true) << "vlog_if -1";
  VLOG_IF(-1, false) << "don't vlog_if -1";
  VLOG_IF(0, true) << "vlog_if 0";
  VLOG_IF(0, false) << "don't vlog_if 0";
  VLOG_IF(1, true) << "vlog_if 1";
  VLOG_IF(1, false) << "don't vlog_if 1";
  LOG_IF(INFO, true) << "log_if info";
  LOG_IF(INFO, false) << "don't log_if info";
  LOG_IF(WARNING, true) << "log_if warning";
  LOG_IF(WARNING, false) << "don't log_if warning";
  LOG_IF(ERROR, true) << "log_if error";
  LOG_IF(ERROR, false) << "don't log_if error";

  int c;
  c = 1; VLOG_IF(100, c -= 2) << "vlog_if 100 expr"; EXPECT_EQ(c, -1);
  c = 1; VLOG_IF(0, c -= 2) << "vlog_if 0 expr"; EXPECT_EQ(c, -1);
  c = 1; LOG_IF(INFO, c -= 2) << "log_if info expr"; EXPECT_EQ(c, -1);
  c = 1; LOG_IF(ERROR, c -= 2) << "log_if error expr"; EXPECT_EQ(c, -1);
  c = 2; VLOG_IF(0, c -= 2) << "don't vlog_if 0 expr"; EXPECT_EQ(c, 0);
  c = 2; LOG_IF(ERROR, c -= 2) << "don't log_if error expr"; EXPECT_EQ(c, 0);

  c = 3; LOG_IF_EVERY_N(INFO, c -= 4, 1) << "log_if info every 1 expr";
  EXPECT_EQ(c, -1);
  c = 3; LOG_IF_EVERY_N(ERROR, c -= 4, 1) << "log_if error every 1 expr";
  EXPECT_EQ(c, -1);
  c = 4; LOG_IF_EVERY_N(ERROR, c -= 4, 3) << "don't log_if info every 3 expr";
  EXPECT_EQ(c, 0);
  c = 4; LOG_IF_EVERY_N(ERROR, c -= 4, 3) << "don't log_if error every 3 expr";
  EXPECT_EQ(c, 0);
  c = 5; VLOG_IF_EVERY_N(0, c -= 4, 1) << "vlog_if 0 every 1 expr";
  EXPECT_EQ(c, 1);
  c = 5; VLOG_IF_EVERY_N(100, c -= 4, 3) << "vlog_if 100 every 3 expr";
  EXPECT_EQ(c, 1);
  c = 6; VLOG_IF_EVERY_N(0, c -= 6, 1) << "don't vlog_if 0 every 1 expr";
  EXPECT_EQ(c, 0);
  c = 6; VLOG_IF_EVERY_N(100, c -= 6, 3) << "don't vlog_if 100 every 1 expr";
  EXPECT_EQ(c, 0);
}

void TestLoggingLevels() {
  LogWithLevels(0, GLOG_INFO, false, false);
  LogWithLevels(1, GLOG_INFO, false, false);
  LogWithLevels(-1, GLOG_INFO, false, false);
  LogWithLevels(0, GLOG_WARNING, false, false);
  LogWithLevels(0, GLOG_ERROR, false, false);
  LogWithLevels(0, GLOG_FATAL, false, false);
  LogWithLevels(0, GLOG_FATAL, true, false);
  LogWithLevels(0, GLOG_FATAL, false, true);
  LogWithLevels(1, GLOG_WARNING, false, false);
  LogWithLevels(1, GLOG_FATAL, false, true);
}

int TestVlogHelper() {
  if (VLOG_IS_ON(1)) {
    return 1;
  }
  return 0;
}

void TestVLogModule() {
  int c = TestVlogHelper();
  EXPECT_EQ(0, c);

#if defined(__GNUC__)
  EXPECT_EQ(0, SetVLOGLevel("logging_unittest", 1));
  c = TestVlogHelper();
  EXPECT_EQ(1, c);
#endif
}

TEST(DeathRawCHECK, logging) {
  ASSERT_DEATH(RAW_CHECK(false, "failure 1"),
               "RAW: Check false failed: failure 1");
  ASSERT_DEBUG_DEATH(RAW_DCHECK(1 == 2, "failure 2"),
               "RAW: Check 1 == 2 failed: failure 2");
}

void TestLogString() {
  vector<string> errors;
  vector<string>* no_errors = nullptr;

  LOG_STRING(INFO, &errors) << "LOG_STRING: " << "collected info";
  LOG_STRING(WARNING, &errors) << "LOG_STRING: " << "collected warning";
  LOG_STRING(ERROR, &errors) << "LOG_STRING: " << "collected error";

  LOG_STRING(INFO, no_errors) << "LOG_STRING: " << "reported info";
  LOG_STRING(WARNING, no_errors) << "LOG_STRING: " << "reported warning";
  LOG_STRING(ERROR, nullptr) << "LOG_STRING: "
                             << "reported error";

  for (auto& error : errors) {
    LOG(INFO) << "Captured by LOG_STRING:  " << error;
  }
}

void TestLogToString() {
  string error;
  string* no_error = nullptr;

  LOG_TO_STRING(INFO, &error) << "LOG_TO_STRING: " << "collected info";
  LOG(INFO) << "Captured by LOG_TO_STRING:  " << error;
  LOG_TO_STRING(WARNING, &error) << "LOG_TO_STRING: " << "collected warning";
  LOG(INFO) << "Captured by LOG_TO_STRING:  " << error;
  LOG_TO_STRING(ERROR, &error) << "LOG_TO_STRING: " << "collected error";
  LOG(INFO) << "Captured by LOG_TO_STRING:  " << error;

  LOG_TO_STRING(INFO, no_error) << "LOG_TO_STRING: " << "reported info";
  LOG_TO_STRING(WARNING, no_error) << "LOG_TO_STRING: " << "reported warning";
  LOG_TO_STRING(ERROR, nullptr) << "LOG_TO_STRING: "
                                << "reported error";
}

class TestLogSinkImpl : public LogSink {
 public:
  vector<string> errors;
  void send(LogSeverity severity, const char* /* full_filename */,
            const char* base_filename, int line,
            const LogMessageTime& logmsgtime, const char* message,
            size_t message_len) override {
    errors.push_back(
      ToString(severity, base_filename, line, logmsgtime, message, message_len));
  }
};

void TestLogSink() {
  TestLogSinkImpl sink;
  LogSink* no_sink = nullptr;

  LOG_TO_SINK(&sink, INFO) << "LOG_TO_SINK: " << "collected info";
  LOG_TO_SINK(&sink, WARNING) << "LOG_TO_SINK: " << "collected warning";
  LOG_TO_SINK(&sink, ERROR) << "LOG_TO_SINK: " << "collected error";

  LOG_TO_SINK(no_sink, INFO) << "LOG_TO_SINK: " << "reported info";
  LOG_TO_SINK(no_sink, WARNING) << "LOG_TO_SINK: " << "reported warning";
  LOG_TO_SINK(nullptr, ERROR) << "LOG_TO_SINK: "
                              << "reported error";

  LOG_TO_SINK_BUT_NOT_TO_LOGFILE(&sink, INFO)
      << "LOG_TO_SINK_BUT_NOT_TO_LOGFILE: " << "collected info";
  LOG_TO_SINK_BUT_NOT_TO_LOGFILE(&sink, WARNING)
      << "LOG_TO_SINK_BUT_NOT_TO_LOGFILE: " << "collected warning";
  LOG_TO_SINK_BUT_NOT_TO_LOGFILE(&sink, ERROR)
      << "LOG_TO_SINK_BUT_NOT_TO_LOGFILE: " << "collected error";

  LOG_TO_SINK_BUT_NOT_TO_LOGFILE(no_sink, INFO)
      << "LOG_TO_SINK_BUT_NOT_TO_LOGFILE: " << "thrashed info";
  LOG_TO_SINK_BUT_NOT_TO_LOGFILE(no_sink, WARNING)
      << "LOG_TO_SINK_BUT_NOT_TO_LOGFILE: " << "thrashed warning";
  LOG_TO_SINK_BUT_NOT_TO_LOGFILE(nullptr, ERROR)
      << "LOG_TO_SINK_BUT_NOT_TO_LOGFILE: "
      << "thrashed error";

  LOG(INFO) << "Captured by LOG_TO_SINK:";
  for (auto& error : sink.errors) {
    LogMessage("foo", LogMessage::kNoLogPrefix, GLOG_INFO).stream() << error;
  }
}

// For testing using CHECK*() on anonymous enums.
enum {
  CASE_A,
  CASE_B
};

void TestCHECK() {
  // Tests using CHECK*() on int values.
  CHECK(1 == 1);
  CHECK_EQ(1, 1);
  CHECK_NE(1, 2);
  CHECK_GE(1, 1);
  CHECK_GE(2, 1);
  CHECK_LE(1, 1);
  CHECK_LE(1, 2);
  CHECK_GT(2, 1);
  CHECK_LT(1, 2);

  // Tests using CHECK*() on anonymous enums.
  // Apple's GCC doesn't like this.
#if !defined(GLOG_OS_MACOSX)
  CHECK_EQ(CASE_A, CASE_A);
  CHECK_NE(CASE_A, CASE_B);
  CHECK_GE(CASE_A, CASE_A);
  CHECK_GE(CASE_B, CASE_A);
  CHECK_LE(CASE_A, CASE_A);
  CHECK_LE(CASE_A, CASE_B);
  CHECK_GT(CASE_B, CASE_A);
  CHECK_LT(CASE_A, CASE_B);
#endif
}

void TestDCHECK() {
#if defined(NDEBUG)
  DCHECK( 1 == 2 ) << " DCHECK's shouldn't be compiled in normal mode";
#endif
  DCHECK( 1 == 1 );
  DCHECK_EQ(1, 1);
  DCHECK_NE(1, 2);
  DCHECK_GE(1, 1);
  DCHECK_GE(2, 1);
  DCHECK_LE(1, 1);
  DCHECK_LE(1, 2);
  DCHECK_GT(2, 1);
  DCHECK_LT(1, 2);

  auto* orig_ptr = new int64;
  int64* ptr = DCHECK_NOTNULL(orig_ptr);
  CHECK_EQ(ptr, orig_ptr);
  delete orig_ptr;
}

void TestSTREQ() {
  CHECK_STREQ("this", "this");
  CHECK_STREQ(nullptr, nullptr);
  CHECK_STRCASEEQ("this", "tHiS");
  CHECK_STRCASEEQ(nullptr, nullptr);
  CHECK_STRNE("this", "tHiS");
  CHECK_STRNE("this", nullptr);
  CHECK_STRCASENE("this", "that");
  CHECK_STRCASENE(nullptr, "that");
  CHECK_STREQ((string("a")+"b").c_str(), "ab");
  CHECK_STREQ(string("test").c_str(),
              (string("te") + string("st")).c_str());
}

TEST(DeathSTREQ, logging) {
  ASSERT_DEATH(CHECK_STREQ(nullptr, "this"), "");
  ASSERT_DEATH(CHECK_STREQ("this", "siht"), "");
  ASSERT_DEATH(CHECK_STRCASEEQ(nullptr, "siht"), "");
  ASSERT_DEATH(CHECK_STRCASEEQ("this", "siht"), "");
  ASSERT_DEATH(CHECK_STRNE(nullptr, nullptr), "");
  ASSERT_DEATH(CHECK_STRNE("this", "this"), "");
  ASSERT_DEATH(CHECK_STREQ((string("a")+"b").c_str(), "abc"), "");
}

TEST(CheckNOTNULL, Simple) {
  int64 t;
  void *ptr = static_cast<void *>(&t);
  void *ref = CHECK_NOTNULL(ptr);
  EXPECT_EQ(ptr, ref);
  CHECK_NOTNULL(reinterpret_cast<char *>(ptr));
  CHECK_NOTNULL(reinterpret_cast<unsigned char *>(ptr));
  CHECK_NOTNULL(reinterpret_cast<int *>(ptr));
  CHECK_NOTNULL(reinterpret_cast<int64 *>(ptr));
}

TEST(DeathCheckNN, Simple) {
  ASSERT_DEATH(CHECK_NOTNULL(static_cast<void*>(nullptr)), "");
}

// Get list of file names that match pattern
static void GetFiles(const string& pattern, vector<string>* files) {
  files->clear();
#if defined(HAVE_GLOB_H)
  glob_t g;
  const int r = glob(pattern.c_str(), 0, nullptr, &g);
  CHECK((r == 0) || (r == GLOB_NOMATCH)) << ": error matching " << pattern;
  for (size_t i = 0; i < g.gl_pathc; i++) {
    files->push_back(string(g.gl_pathv[i]));
  }
  globfree(&g);
#elif defined(GLOG_OS_WINDOWS)
  WIN32_FIND_DATAA data;
  HANDLE handle = FindFirstFileA(pattern.c_str(), &data);
  size_t index = pattern.rfind('\\');
  if (index == string::npos) {
    LOG(FATAL) << "No directory separator.";
  }
  const string dirname = pattern.substr(0, index + 1);
  if (handle == INVALID_HANDLE_VALUE) {
    // Finding no files is OK.
    return;
  }
  do {
    files->push_back(dirname + data.cFileName);
  } while (FindNextFileA(handle, &data));
  BOOL result = FindClose(handle);
  LOG_SYSRESULT(result != 0);
#else
# error There is no way to do glob.
#endif
}

// Delete files patching pattern
static void DeleteFiles(const string& pattern) {
  vector<string> files;
  GetFiles(pattern, &files);
  for (auto& file : files) {
    CHECK(unlink(file.c_str()) == 0) << ": " << strerror(errno);
  }
}

//check string is in file (or is *NOT*, depending on optional checkInFileOrNot)
static void CheckFile(const string& name, const string& expected_string, const bool checkInFileOrNot = true) {
  vector<string> files;
  GetFiles(name + "*", &files);
  CHECK_EQ(files.size(), 1UL);

  FILE* file = fopen(files[0].c_str(), "r");
  CHECK(file != nullptr) << ": could not open " << files[0];
  char buf[1000];
  while (fgets(buf, sizeof(buf), file) != nullptr) {
    char* first = strstr(buf, expected_string.c_str());
    // if first == nullptr, not found.
    // Terser than if (checkInFileOrNot && first != nullptr || !check...
    if (checkInFileOrNot != (first == nullptr)) {
      fclose(file);
      return;
    }
  }
  fclose(file);
  LOG(FATAL) << "Did " << (checkInFileOrNot? "not " : "") << "find " << expected_string << " in " << files[0];
}

static void TestBasename() {
  fprintf(stderr, "==== Test setting log file basename\n");
  const string dest = FLAGS_test_tmpdir + "/logging_test_basename";
  DeleteFiles(dest + "*");

  SetLogDestination(GLOG_INFO, dest.c_str());
  LOG(INFO) << "message to new base";
  FlushLogFiles(GLOG_INFO);

  CheckFile(dest, "message to new base");

  // Release file handle for the destination file to unlock the file in Windows.
  LogToStderr();
  DeleteFiles(dest + "*");
}

static void TestBasenameAppendWhenNoTimestamp() {
  fprintf(stderr, "==== Test setting log file basename without timestamp and appending properly\n");
  const string dest = FLAGS_test_tmpdir + "/logging_test_basename_append_when_no_timestamp";
  DeleteFiles(dest + "*");

  ofstream out(dest.c_str());
  out << "test preexisting content" << endl;
  out.close();

  CheckFile(dest, "test preexisting content");

  FLAGS_timestamp_in_logfile_name=false;
  SetLogDestination(GLOG_INFO, dest.c_str());
  LOG(INFO) << "message to new base, appending to preexisting file";
  FlushLogFiles(GLOG_INFO);
  FLAGS_timestamp_in_logfile_name=true;

  //if the logging overwrites the file instead of appending it will fail.
  CheckFile(dest, "test preexisting content");
  CheckFile(dest, "message to new base, appending to preexisting file");

  // Release file handle for the destination file to unlock the file in Windows.
  LogToStderr();
  DeleteFiles(dest + "*");
}

static void TestTwoProcessesWrite() {
// test only implemented for platforms with fork & wait; the actual implementation relies on flock
#if defined(HAVE_SYS_WAIT_H) && defined(HAVE_UNISTD_H) && defined(HAVE_FCNTL)
  fprintf(stderr, "==== Test setting log file basename and two processes writing - second should fail\n");
  const string dest = FLAGS_test_tmpdir + "/logging_test_basename_two_processes_writing";
  DeleteFiles(dest + "*");

  //make both processes write into the same file (easier test)
  FLAGS_timestamp_in_logfile_name=false;
  SetLogDestination(GLOG_INFO, dest.c_str());
  LOG(INFO) << "message to new base, parent";
  FlushLogFiles(GLOG_INFO);

  pid_t pid = fork();
  CHECK_ERR(pid);
  if (pid == 0) {
    LOG(INFO) << "message to new base, child - should only appear on STDERR not on the file";
    ShutdownGoogleLogging(); //for children proc
    exit(EXIT_SUCCESS);
  } else if (pid > 0) {
    wait(nullptr);
  }
  FLAGS_timestamp_in_logfile_name=true;

  CheckFile(dest, "message to new base, parent");
  CheckFile(dest, "message to new base, child - should only appear on STDERR not on the file", false);

  // Release
  LogToStderr();
  DeleteFiles(dest + "*");
#endif
}

static void TestSymlink() {
#ifndef GLOG_OS_WINDOWS
  fprintf(stderr, "==== Test setting log file symlink\n");
  string dest = FLAGS_test_tmpdir + "/logging_test_symlink";
  string sym = FLAGS_test_tmpdir + "/symlinkbase";
  DeleteFiles(dest + "*");
  DeleteFiles(sym + "*");

  SetLogSymlink(GLOG_INFO, "symlinkbase");
  SetLogDestination(GLOG_INFO, dest.c_str());
  LOG(INFO) << "message to new symlink";
  FlushLogFiles(GLOG_INFO);
  CheckFile(sym, "message to new symlink");

  DeleteFiles(dest + "*");
  DeleteFiles(sym + "*");
#endif
}

static void TestExtension() {
  fprintf(stderr, "==== Test setting log file extension\n");
  string dest = FLAGS_test_tmpdir + "/logging_test_extension";
  DeleteFiles(dest + "*");

  SetLogDestination(GLOG_INFO, dest.c_str());
  SetLogFilenameExtension("specialextension");
  LOG(INFO) << "message to new extension";
  FlushLogFiles(GLOG_INFO);
  CheckFile(dest, "message to new extension");

  // Check that file name ends with extension
  vector<string> filenames;
  GetFiles(dest + "*", &filenames);
  CHECK_EQ(filenames.size(), 1UL);
  CHECK(strstr(filenames[0].c_str(), "specialextension") != nullptr);

  // Release file handle for the destination file to unlock the file in Windows.
  LogToStderr();
  DeleteFiles(dest + "*");
}

struct MyLogger : public base::Logger {
  string data;

  explicit MyLogger(bool* set_on_destruction)
      : set_on_destruction_(set_on_destruction) {}

  ~MyLogger() override { *set_on_destruction_ = true; }

  void Write(bool /* should_flush */, time_t /* timestamp */,
             const char* message, size_t length) override {
    data.append(message, length);
  }

  void Flush() override {}

  uint32 LogSize() override { return data.length(); }

 private:
  bool* set_on_destruction_;
};

static void TestWrapper() {
  fprintf(stderr, "==== Test log wrapper\n");

  bool custom_logger_deleted = false;
  auto* my_logger = new MyLogger(&custom_logger_deleted);
  base::Logger* old_logger = base::GetLogger(GLOG_INFO);
  base::SetLogger(GLOG_INFO, my_logger);
  LOG(INFO) << "Send to wrapped logger";
  CHECK(strstr(my_logger->data.c_str(), "Send to wrapped logger") != nullptr);
  FlushLogFiles(GLOG_INFO);

  EXPECT_FALSE(custom_logger_deleted);
  base::SetLogger(GLOG_INFO, old_logger);
  EXPECT_TRUE(custom_logger_deleted);
}

static void TestErrno() {
  fprintf(stderr, "==== Test errno preservation\n");

  errno = ENOENT;
  TestLogging(false);
  CHECK_EQ(errno, ENOENT);
}

static void TestOneTruncate(const char *path, uint64 limit, uint64 keep,
                            size_t dsize, size_t ksize, size_t expect) {
  int fd;
  CHECK_ERR(fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600));

  const char *discardstr = "DISCARDME!", *keepstr = "KEEPME!";
  const size_t discard_size = strlen(discardstr), keep_size = strlen(keepstr);

  // Fill the file with the requested data; first discard data, then kept data
  size_t written = 0;
  while (written < dsize) {
    size_t bytes = min(dsize - written, discard_size);
    CHECK_ERR(write(fd, discardstr, bytes));
    written += bytes;
  }
  written = 0;
  while (written < ksize) {
    size_t bytes = min(ksize - written, keep_size);
    CHECK_ERR(write(fd, keepstr, bytes));
    written += bytes;
  }

  TruncateLogFile(path, limit, keep);

  // File should now be shorter
  struct stat statbuf;
  CHECK_ERR(fstat(fd, &statbuf));
  CHECK_EQ(static_cast<size_t>(statbuf.st_size), expect);
  CHECK_ERR(lseek(fd, 0, SEEK_SET));

  // File should contain the suffix of the original file
  const size_t buf_size = static_cast<size_t>(statbuf.st_size) + 1;
  char* buf = new char[buf_size];
  memset(buf, 0, buf_size);
  CHECK_ERR(read(fd, buf, buf_size));

  const char* p = buf;
  size_t checked = 0;
  while (checked < expect) {
    size_t bytes = min(expect - checked, keep_size);
    CHECK(!memcmp(p, keepstr, bytes));
    checked += bytes;
  }
  close(fd);
  delete[] buf;
}

static void TestTruncate() {
#ifdef HAVE_UNISTD_H
  fprintf(stderr, "==== Test log truncation\n");
  string path = FLAGS_test_tmpdir + "/truncatefile";

  // Test on a small file
  TestOneTruncate(path.c_str(), 10, 10, 10, 10, 10);

  // And a big file (multiple blocks to copy)
  TestOneTruncate(path.c_str(), 2U << 20U, 4U << 10U, 3U << 20U, 4U << 10U,
                  4U << 10U);

  // Check edge-case limits
  TestOneTruncate(path.c_str(), 10, 20, 0, 20, 20);
  TestOneTruncate(path.c_str(), 10, 0, 0, 0, 0);
  TestOneTruncate(path.c_str(), 10, 50, 0, 10, 10);
  TestOneTruncate(path.c_str(), 50, 100, 0, 30, 30);

  // MacOSX 10.4 doesn't fail in this case.
  // Windows doesn't have symlink.
  // Let's just ignore this test for these cases.
#if !defined(GLOG_OS_MACOSX) && !defined(GLOG_OS_WINDOWS)
  // Through a symlink should fail to truncate
  string linkname = path + ".link";
  unlink(linkname.c_str());
  CHECK_ERR(symlink(path.c_str(), linkname.c_str()));
  TestOneTruncate(linkname.c_str(), 10, 10, 0, 30, 30);
#endif

  // The /proc/self path makes sense only for linux.
#if defined(GLOG_OS_LINUX)
  // Through an open fd symlink should work
  int fd;
  CHECK_ERR(fd = open(path.c_str(), O_APPEND | O_WRONLY));
  char fdpath[64];
  snprintf(fdpath, sizeof(fdpath), "/proc/self/fd/%d", fd);
  TestOneTruncate(fdpath, 10, 10, 10, 10, 10);
#endif

#endif
}

struct RecordDeletionLogger : public base::Logger {
  RecordDeletionLogger(bool* set_on_destruction,
                       base::Logger* wrapped_logger) :
      set_on_destruction_(set_on_destruction),
      wrapped_logger_(wrapped_logger)
  {
    *set_on_destruction_ = false;
  }
  ~RecordDeletionLogger() override { *set_on_destruction_ = true; }
  void Write(bool force_flush, time_t timestamp, const char* message,
             size_t length) override {
    wrapped_logger_->Write(force_flush, timestamp, message, length);
  }
  void Flush() override { wrapped_logger_->Flush(); }
  uint32 LogSize() override { return wrapped_logger_->LogSize(); }

 private:
  bool* set_on_destruction_;
  base::Logger* wrapped_logger_;
};

static void TestCustomLoggerDeletionOnShutdown() {
  bool custom_logger_deleted = false;
  base::SetLogger(GLOG_INFO,
                  new RecordDeletionLogger(&custom_logger_deleted,
                                           base::GetLogger(GLOG_INFO)));
  EXPECT_TRUE(IsGoogleLoggingInitialized());
  ShutdownGoogleLogging();
  EXPECT_TRUE(custom_logger_deleted);
  EXPECT_FALSE(IsGoogleLoggingInitialized());
}

namespace LogTimes {
// Log a "message" every 10ms, 10 times. These numbers are nice compromise
// between total running time of 100ms and the period of 10ms. The period is
// large enough such that any CPU and OS scheduling variation shouldn't affect
// the results from the ideal case by more than 5% (500us or 0.5ms)
constexpr int64_t LOG_PERIOD_NS = 10000000;    // 10ms
constexpr int64_t LOG_PERIOD_TOL_NS = 500000;  // 500us

// Set an upper limit for the number of times the stream operator can be
// called. Make sure not to exceed this number of times the stream operator is
// called, since it is also the array size and will be indexed by the stream
// operator.
constexpr size_t MAX_CALLS = 10;
}  // namespace LogTimes

struct LogTimeRecorder {
  LogTimeRecorder() = default;
  size_t m_streamTimes{0};
  std::chrono::steady_clock::time_point m_callTimes[LogTimes::MAX_CALLS];
};
// The stream operator is called by LOG_EVERY_T every time a logging event
// occurs. Make sure to save the times for each call as they will be used later
// to verify the time delta between each call.
std::ostream& operator<<(std::ostream& stream, LogTimeRecorder& t) {
  t.m_callTimes[t.m_streamTimes++] = std::chrono::steady_clock::now();
  return stream;
}
// get elapsed time in nanoseconds
int64 elapsedTime_ns(const std::chrono::steady_clock::time_point& begin,
        const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin))
          .count();
}

static void TestLogPeriodically() {
  fprintf(stderr, "==== Test log periodically\n");

  LogTimeRecorder timeLogger;

  constexpr double LOG_PERIOD_SEC = LogTimes::LOG_PERIOD_NS * 1e-9;

  while (timeLogger.m_streamTimes < LogTimes::MAX_CALLS) {
      LOG_EVERY_T(INFO, LOG_PERIOD_SEC)
          << timeLogger << "Timed Message #" << timeLogger.m_streamTimes;
  }

  // Calculate time between each call in nanoseconds for higher resolution to
  // minimize error.
  int64 nsBetweenCalls[LogTimes::MAX_CALLS - 1];
  for (size_t i = 1; i < LogTimes::MAX_CALLS; ++i) {
    nsBetweenCalls[i - 1] = elapsedTime_ns(
            timeLogger.m_callTimes[i - 1], timeLogger.m_callTimes[i]);
  }

  for (long time_ns : nsBetweenCalls) {
    EXPECT_NEAR(time_ns, LogTimes::LOG_PERIOD_NS, LogTimes::LOG_PERIOD_TOL_NS);
  }
}

_START_GOOGLE_NAMESPACE_
namespace glog_internal_namespace_ {
extern  // in logging.cc
bool SafeFNMatch_(const char* pattern, size_t patt_len,
                  const char* str, size_t str_len);
} // namespace glog_internal_namespace_
using glog_internal_namespace_::SafeFNMatch_;
_END_GOOGLE_NAMESPACE_

static bool WrapSafeFNMatch(string pattern, string str) {
  pattern += "abc";
  str += "defgh";
  return SafeFNMatch_(pattern.data(), pattern.size() - 3,
                      str.data(), str.size() - 5);
}

TEST(SafeFNMatch, logging) {
  CHECK(WrapSafeFNMatch("foo", "foo"));
  CHECK(!WrapSafeFNMatch("foo", "bar"));
  CHECK(!WrapSafeFNMatch("foo", "fo"));
  CHECK(!WrapSafeFNMatch("foo", "foo2"));
  CHECK(WrapSafeFNMatch("bar/foo.ext", "bar/foo.ext"));
  CHECK(WrapSafeFNMatch("*ba*r/fo*o.ext*", "bar/foo.ext"));
  CHECK(!WrapSafeFNMatch("bar/foo.ext", "bar/baz.ext"));
  CHECK(!WrapSafeFNMatch("bar/foo.ext", "bar/foo"));
  CHECK(!WrapSafeFNMatch("bar/foo.ext", "bar/foo.ext.zip"));
  CHECK(WrapSafeFNMatch("ba?/*.ext", "bar/foo.ext"));
  CHECK(WrapSafeFNMatch("ba?/*.ext", "baZ/FOO.ext"));
  CHECK(!WrapSafeFNMatch("ba?/*.ext", "barr/foo.ext"));
  CHECK(!WrapSafeFNMatch("ba?/*.ext", "bar/foo.ext2"));
  CHECK(WrapSafeFNMatch("ba?/*", "bar/foo.ext2"));
  CHECK(WrapSafeFNMatch("ba?/*", "bar/"));
  CHECK(!WrapSafeFNMatch("ba?/?", "bar/"));
  CHECK(!WrapSafeFNMatch("ba?/*", "bar"));
}

// TestWaitingLogSink will save messages here
// No lock: Accessed only by TestLogSinkWriter thread
// and after its demise by its creator.
static vector<string> global_messages;

// helper for TestWaitingLogSink below.
// Thread that does the logic of TestWaitingLogSink
// It's free to use LOG() itself.
class TestLogSinkWriter : public Thread {
 public:
  TestLogSinkWriter() {
    SetJoinable(true);
    Start();
  }

  // Just buffer it (can't use LOG() here).
  void Buffer(const string& message) {
    mutex_.Lock();
    RAW_LOG(INFO, "Buffering");
    messages_.push(message);
    mutex_.Unlock();
    RAW_LOG(INFO, "Buffered");
  }

  // Wait for the buffer to clear (can't use LOG() here).
  void Wait() {
    RAW_LOG(INFO, "Waiting");
    mutex_.Lock();
    while (!NoWork()) {
      mutex_.Unlock();
      SleepForMilliseconds(1);
      mutex_.Lock();
    }
    RAW_LOG(INFO, "Waited");
    mutex_.Unlock();
  }

  // Trigger thread exit.
  void Stop() {
    MutexLock l(&mutex_);
    should_exit_ = true;
  }

 private:

  // helpers ---------------

  // For creating a "Condition".
  bool NoWork() { return messages_.empty(); }
  bool HaveWork() { return !messages_.empty() || should_exit_; }

  // Thread body; CAN use LOG() here!
  void Run() override {
    while (true) {
      mutex_.Lock();
      while (!HaveWork()) {
        mutex_.Unlock();
        SleepForMilliseconds(1);
        mutex_.Lock();
      }
      if (should_exit_ && messages_.empty()) {
        mutex_.Unlock();
        break;
      }
      // Give the main thread time to log its message,
      // so that we get a reliable log capture to compare to golden file.
      // Same for the other sleep below.
      SleepForMilliseconds(20);
      RAW_LOG(INFO, "Sink got a messages");  // only RAW_LOG under mutex_ here
      string message = messages_.front();
      messages_.pop();
      // Normally this would be some more real/involved logging logic
      // where LOG() usage can't be eliminated,
      // e.g. pushing the message over with an RPC:
      size_t messages_left = messages_.size();
      mutex_.Unlock();
      SleepForMilliseconds(20);
      // May not use LOG while holding mutex_, because Buffer()
      // acquires mutex_, and Buffer is called from LOG(),
      // which has its own internal mutex:
      // LOG()->LogToSinks()->TestWaitingLogSink::send()->Buffer()
      LOG(INFO) << "Sink is sending out a message: " << message;
      LOG(INFO) << "Have " << messages_left << " left";
      global_messages.push_back(message);
    }
  }

  // data ---------------

  Mutex mutex_;
  bool should_exit_{false};
  queue<string> messages_;  // messages to be logged
};

// A log sink that exercises WaitTillSent:
// it pushes data to a buffer and wakes up another thread to do the logging
// (that other thread can than use LOG() itself),
class TestWaitingLogSink : public LogSink {
 public:

  TestWaitingLogSink() {
    tid_ = pthread_self();  // for thread-specific behavior
    AddLogSink(this);
  }
  ~TestWaitingLogSink() override {
    RemoveLogSink(this);
    writer_.Stop();
    writer_.Join();
  }

  // (re)define LogSink interface

  void send(LogSeverity severity, const char* /* full_filename */,
            const char* base_filename, int line,
            const LogMessageTime& logmsgtime, const char* message,
            size_t message_len) override {
    // Push it to Writer thread if we are the original logging thread.
    // Note: Something like ThreadLocalLogSink is a better choice
    //       to do thread-specific LogSink logic for real.
    if (pthread_equal(tid_, pthread_self())) {
      writer_.Buffer(ToString(severity, base_filename, line,
                              logmsgtime, message, message_len));
    }
  }

  void WaitTillSent() override {
    // Wait for Writer thread if we are the original logging thread.
    if (pthread_equal(tid_, pthread_self()))  writer_.Wait();
  }

 private:

  pthread_t tid_;
  TestLogSinkWriter writer_;
};

// Check that LogSink::WaitTillSent can be used in the advertised way.
// We also do golden-stderr comparison.
static void TestLogSinkWaitTillSent() {
  // Clear global_messages here to make sure that this test case can be
  // reentered
  global_messages.clear();
  { TestWaitingLogSink sink;
    // Sleeps give the sink threads time to do all their work,
    // so that we get a reliable log capture to compare to the golden file.
    LOG(INFO) << "Message 1";
    SleepForMilliseconds(60);
    LOG(ERROR) << "Message 2";
    SleepForMilliseconds(60);
    LOG(WARNING) << "Message 3";
    SleepForMilliseconds(60);
  }
  for (auto& global_message : global_messages) {
    LOG(INFO) << "Sink capture: " << global_message;
  }
  CHECK_EQ(global_messages.size(), 3UL);
}

TEST(Strerror, logging) {
  int errcode = EINTR;
  char *msg = strdup(strerror(errcode));
  const size_t buf_size = strlen(msg) + 1;
  char *buf = new char[buf_size];
  CHECK_EQ(posix_strerror_r(errcode, nullptr, 0), -1);
  buf[0] = 'A';
  CHECK_EQ(posix_strerror_r(errcode, buf, 0), -1);
  CHECK_EQ(buf[0], 'A');
  CHECK_EQ(posix_strerror_r(errcode, nullptr, buf_size), -1);
#if defined(GLOG_OS_MACOSX) || defined(GLOG_OS_FREEBSD) || defined(GLOG_OS_OPENBSD)
  // MacOSX or FreeBSD considers this case is an error since there is
  // no enough space.
  CHECK_EQ(posix_strerror_r(errcode, buf, 1), -1);
#else
  CHECK_EQ(posix_strerror_r(errcode, buf, 1), 0);
#endif
  CHECK_STREQ(buf, "");
  CHECK_EQ(posix_strerror_r(errcode, buf, buf_size), 0);
  CHECK_STREQ(buf, msg);
  delete[] buf;
  CHECK_EQ(msg, StrError(errcode));
  free(msg);
}

// Simple routines to look at the sizes of generated code for LOG(FATAL) and
// CHECK(..) via objdump
/*
static void MyFatal() {
  LOG(FATAL) << "Failed";
}
static void MyCheck(bool a, bool b) {
  CHECK_EQ(a, b);
}
*/
#ifdef HAVE_LIB_GMOCK

TEST(DVLog, Basic) {
  ScopedMockLog log;

#if defined(NDEBUG)
  // We are expecting that nothing is logged.
  EXPECT_CALL(log, Log(_, _, _)).Times(0);
#else
  EXPECT_CALL(log, Log(GLOG_INFO, __FILE__, "debug log"));
#endif

  FLAGS_v = 1;
  DVLOG(1) << "debug log";
}

TEST(DVLog, V0) {
  ScopedMockLog log;

  // We are expecting that nothing is logged.
  EXPECT_CALL(log, Log(_, _, _)).Times(0);

  FLAGS_v = 0;
  DVLOG(1) << "debug log";
}

TEST(LogAtLevel, Basic) {
  ScopedMockLog log;

  // The function version outputs "logging.h" as a file name.
  EXPECT_CALL(log, Log(GLOG_WARNING, StrNe(__FILE__), "function version"));
  EXPECT_CALL(log, Log(GLOG_INFO, __FILE__, "macro version"));

  int severity = GLOG_WARNING;
  LogAtLevel(severity, "function version");

  severity = GLOG_INFO;
  // We can use the macro version as a C++ stream.
  LOG_AT_LEVEL(severity) << "macro" << ' ' << "version";
}

TEST(TestExitOnDFatal, ToBeOrNotToBe) {
  // Check the default setting...
  EXPECT_TRUE(base::internal::GetExitOnDFatal());

  // Turn off...
  base::internal::SetExitOnDFatal(false);
  EXPECT_FALSE(base::internal::GetExitOnDFatal());

  // We don't die.
  {
    ScopedMockLog log;
    //EXPECT_CALL(log, Log(_, _, _)).Times(AnyNumber());
    // LOG(DFATAL) has severity FATAL if debugging, but is
    // downgraded to ERROR if not debugging.
    const LogSeverity severity =
#if defined(NDEBUG)
        GLOG_ERROR;
#else
        GLOG_FATAL;
#endif
    EXPECT_CALL(log, Log(severity, __FILE__, "This should not be fatal"));
    LOG(DFATAL) << "This should not be fatal";
  }

  // Turn back on...
  base::internal::SetExitOnDFatal(true);
  EXPECT_TRUE(base::internal::GetExitOnDFatal());

#ifdef GTEST_HAS_DEATH_TEST
  // Death comes on little cats' feet.
  EXPECT_DEBUG_DEATH({
      LOG(DFATAL) << "This should be fatal in debug mode";
    }, "This should be fatal in debug mode");
#endif
}

#ifdef HAVE_STACKTRACE

static void BacktraceAtHelper() {
  LOG(INFO) << "Not me";

// The vertical spacing of the next 3 lines is significant.
  LOG(INFO) << "Backtrace me";
}
static int kBacktraceAtLine = __LINE__ - 2;  // The line of the LOG(INFO) above

TEST(LogBacktraceAt, DoesNotBacktraceWhenDisabled) {
  StrictMock<ScopedMockLog> log;

  FLAGS_log_backtrace_at = "";

  EXPECT_CALL(log, Log(_, _, "Backtrace me"));
  EXPECT_CALL(log, Log(_, _, "Not me"));

  BacktraceAtHelper();
}

TEST(LogBacktraceAt, DoesBacktraceAtRightLineWhenEnabled) {
  StrictMock<ScopedMockLog> log;

  char where[100];
  snprintf(where, 100, "%s:%d", const_basename(__FILE__), kBacktraceAtLine);
  FLAGS_log_backtrace_at = where;

  // The LOG at the specified line should include a stacktrace which includes
  // the name of the containing function, followed by the log message.
  // We use HasSubstr()s instead of ContainsRegex() for environments
  // which don't have regexp.
  EXPECT_CALL(log, Log(_, _, AllOf(HasSubstr("stacktrace:"),
                                   HasSubstr("BacktraceAtHelper"),
                                   HasSubstr("main"),
                                   HasSubstr("Backtrace me"))));
  // Other LOGs should not include a backtrace.
  EXPECT_CALL(log, Log(_, _, "Not me"));

  BacktraceAtHelper();
}

#endif // HAVE_STACKTRACE

#endif // HAVE_LIB_GMOCK

struct UserDefinedClass {
  bool operator==(const UserDefinedClass&) const { return true; }
};

inline ostream& operator<<(ostream& out, const UserDefinedClass&) {
  out << "OK";
  return out;
}

TEST(UserDefinedClass, logging) {
  UserDefinedClass u;
  vector<string> buf;
  LOG_STRING(INFO, &buf) << u;
  CHECK_EQ(1UL, buf.size());
  CHECK(buf[0].find("OK") != string::npos);

  // We must be able to compile this.
  CHECK_EQ(u, u);
}

TEST(LogMsgTime, gmtoff) {
  /*
   * Unit test for GMT offset API
   * TODO: To properly test this API, we need a platform independent way to set time-zone.
   * */
  google::LogMessage log_obj(__FILE__, __LINE__);

  long int nGmtOff = log_obj.getLogMessageTime().gmtoff();
  // GMT offset ranges from UTC-12:00 to UTC+14:00
  const long utc_min_offset = -43200;
  const long utc_max_offset = 50400;
  EXPECT_TRUE( (nGmtOff >= utc_min_offset) && (nGmtOff <= utc_max_offset) );
}
