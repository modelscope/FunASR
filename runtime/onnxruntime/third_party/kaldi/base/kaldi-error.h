// base/kaldi-error.h

// Copyright 2019 LAIX (Yi Sun)
// Copyright 2019 SmartAction LLC (kkm)
// Copyright 2016 Brno University of Technology (author: Karel Vesely)
// Copyright 2009-2011  Microsoft Corporation;  Ondrej Glembek;  Lukas Burget;
//                      Saarland University

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_BASE_KALDI_ERROR_H_
#define KALDI_BASE_KALDI_ERROR_H_ 1

#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/kaldi-types.h"
#include "base/kaldi-utils.h"
/* Important that this file does not depend on any other kaldi headers. */

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

namespace kaldi {

/// \addtogroup error_group
/// @{

/***** PROGRAM NAME AND VERBOSITY LEVEL *****/

/// Called by ParseOptions to set base name (no directory) of the executing
/// program. The name is printed in logging code along with every message,
/// because in our scripts, we often mix together the stderr of many programs.
/// This function is very thread-unsafe.
void SetProgramName(const char *basename);

/// This is set by util/parse-options.{h,cc} if you set --verbose=? option.
/// Do not use directly, prefer {Get,Set}VerboseLevel().
extern int32 g_kaldi_verbose_level;

/// Get verbosity level, usually set via command line '--verbose=' switch.
inline int32 GetVerboseLevel() { return g_kaldi_verbose_level; }

/// This should be rarely used, except by programs using Kaldi as library;
/// command-line programs set the verbose level automatically from ParseOptions.
inline void SetVerboseLevel(int32 i) { g_kaldi_verbose_level = i; }

/***** KALDI LOGGING *****/

/// Log message severity and source location info.
struct LogMessageEnvelope {
  /// Message severity. In addition to these levels, positive values (1 to 6)
  /// specify verbose logging level. Verbose messages are produced only when
  /// SetVerboseLevel() has been called to set logging level to at least the
  /// corresponding value.
  enum Severity {
    kAssertFailed = -3, //!< Assertion failure. abort() will be called.
    kError = -2,        //!< Fatal error. KaldiFatalError will be thrown.
    kWarning = -1,      //!< Indicates a recoverable but abnormal condition.
    kInfo = 0,          //!< Informational message.
  };
  int severity;     //!< A Severity value, or positive verbosity level.
  const char *func; //!< Name of the function invoking the logging.
  const char *file; //!< Source file name with up to 1 leading directory.
  int32 line;       //<! Line number in the source file.
};

/// Kaldi fatal runtime error exception. This exception is thrown from any use
/// of the KALDI_ERR logging macro after the logging function, either set by
/// SetLogHandler(), or the Kaldi's internal one, has returned.
class KaldiFatalError : public std::runtime_error {
public:
  explicit KaldiFatalError(const std::string &message)
      : std::runtime_error(message) {}
  explicit KaldiFatalError(const char *message) : std::runtime_error(message) {}

  /// Returns the exception name, "kaldi::KaldiFatalError".
  virtual const char *what() const noexcept override {
    return "kaldi::KaldiFatalError";
  }

  /// Returns the Kaldi error message logged by KALDI_ERR.
  const char *KaldiMessage() const { return std::runtime_error::what(); }
};

// Class MessageLogger is the workhorse behind the KALDI_ASSERT, KALDI_ERR,
// KALDI_WARN, KALDI_LOG and KALDI_VLOG macros. It formats the message, then
// either prints it to stderr or passes to the custom logging handler if
// provided. Then, in case of the error, throws a KaldiFatalError exception, or
// in case of failed KALDI_ASSERT, calls std::abort().
class MessageLogger {
public:
  /// The constructor stores the message's "envelope", a set of data which
  // identifies the location in source which is sending the message to log.
  // The pointers to strings are stored internally, and not owned or copied,
  // so that their storage must outlive this object.
  MessageLogger(LogMessageEnvelope::Severity severity, const char *func,
                const char *file, int32 line);

  // The stream insertion operator, used in e.g. 'KALDI_LOG << "Message"'.
  template <typename T> MessageLogger &operator<<(const T &val) {
    ss_ << val;
    return *this;
  }

  // When assigned a MessageLogger, log its contents.
  struct Log final {
    void operator=(const MessageLogger &logger) { logger.LogMessage(); }
  };

  // When assigned a MessageLogger, log its contents and then throw
  // a KaldiFatalError.
  struct LogAndThrow final {
    [[noreturn]] void operator=(const MessageLogger &logger) {
      logger.LogMessage();
      throw KaldiFatalError(logger.GetMessage());
    }
  };

private:
  std::string GetMessage() const { return ss_.str(); }
  void LogMessage() const;

  LogMessageEnvelope envelope_;
  std::ostringstream ss_;
};

// Logging macros.
#define KALDI_ERR                                                              \
  ::kaldi::MessageLogger::LogAndThrow() = ::kaldi::MessageLogger(              \
      ::kaldi::LogMessageEnvelope::kError, __func__, __FILE__, __LINE__)
#define KALDI_WARN                                                             \
  ::kaldi::MessageLogger::Log() = ::kaldi::MessageLogger(                      \
      ::kaldi::LogMessageEnvelope::kWarning, __func__, __FILE__, __LINE__)
#define KALDI_LOG                                                              \
  ::kaldi::MessageLogger::Log() = ::kaldi::MessageLogger(                      \
      ::kaldi::LogMessageEnvelope::kInfo, __func__, __FILE__, __LINE__)
#define KALDI_VLOG(v)                                                          \
  if ((v) <= ::kaldi::GetVerboseLevel())                                       \
  ::kaldi::MessageLogger::Log() =                                              \
      ::kaldi::MessageLogger((::kaldi::LogMessageEnvelope::Severity)(v),       \
                             __func__, __FILE__, __LINE__)

/***** KALDI ASSERTS *****/

[[noreturn]] void KaldiAssertFailure_(const char *func, const char *file,
                                      int32 line, const char *cond_str);

// Note on KALDI_ASSERT and KALDI_PARANOID_ASSERT:
//
// A single block {} around if /else  does not work, because it causes
// syntax error (unmatched else block) in the following code:
//
// if (condition)
//   KALDI_ASSERT(condition2);
// else
//   SomethingElse();
//
// do {} while(0) -- note there is no semicolon at the end! -- works nicely,
// and compilers will be able to optimize the loop away (as the condition
// is always false).
//
// Also see KALDI_COMPILE_TIME_ASSERT, defined in base/kaldi-utils.h, and
// KALDI_ASSERT_IS_INTEGER_TYPE and KALDI_ASSERT_IS_FLOATING_TYPE, also defined
// there.
#ifndef NDEBUG
#define KALDI_ASSERT(cond)                                                     \
  do {                                                                         \
    if (cond)                                                                  \
      (void)0;                                                                 \
    else                                                                       \
      ::kaldi::KaldiAssertFailure_(__func__, __FILE__, __LINE__, #cond);       \
  } while (0)
#else
#define KALDI_ASSERT(cond) (void)0
#endif

// Some more expensive asserts only checked if this defined.
#ifdef KALDI_PARANOID
#define KALDI_PARANOID_ASSERT(cond)                                            \
  do {                                                                         \
    if (cond)                                                                  \
      (void)0;                                                                 \
    else                                                                       \
      ::kaldi::KaldiAssertFailure_(__func__, __FILE__, __LINE__, #cond);       \
  } while (0)
#else
#define KALDI_PARANOID_ASSERT(cond) (void)0
#endif

/***** THIRD-PARTY LOG-HANDLER *****/

/// Type of third-party logging function.
typedef void (*LogHandler)(const LogMessageEnvelope &envelope,
                           const char *message);

/// Set logging handler. If called with a non-NULL function pointer, the
/// function pointed by it is called to send messages to a caller-provided log.
/// If called with a NULL pointer, restores default Kaldi error logging to
/// stderr. This function is obviously not thread safe; the log handler must be.
/// Returns a previously set logging handler pointer, or NULL.
LogHandler SetLogHandler(LogHandler);

/// @} end "addtogroup error_group"

//// Functions within internal is exported for testing only, do not use.
//namespace internal {
//bool LocateSymbolRange(const std::string &trace_name, size_t *begin,
//                       size_t *end);
//} // namespace internal
} // namespace kaldi

#endif // KALDI_BASE_KALDI_ERROR_H_
