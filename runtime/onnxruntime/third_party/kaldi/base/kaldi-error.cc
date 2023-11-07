// base/kaldi-error.cc

// Copyright 2019 LAIX (Yi Sun)
// Copyright 2019 SmartAction LLC (kkm)
// Copyright 2016 Brno University of Technology (author: Karel Vesely)
// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;  Ondrej Glembek

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

#ifdef HAVE_EXECINFO_H
#include <execinfo.h> // To get stack trace in error messages.
// If this #include fails there is an error in the Makefile, it does not
// support your platform well. Make sure HAVE_EXECINFO_H is undefined,
// and the code will compile.
#ifdef HAVE_CXXABI_H
#include <cxxabi.h> // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif // HAVE_CXXABI_H
#endif // HAVE_EXECINFO_H

#include "base/kaldi-common.h"
#include "base/kaldi-error.h"

// KALDI_GIT_HEAD is useless currently in full repo
//#if !defined(KALDI_VERSION)
//#include "base/version.h"
//#endif

namespace kaldi {

/***** GLOBAL VARIABLES FOR LOGGING *****/

int32 g_kaldi_verbose_level = 0;
static std::string program_name;
static LogHandler log_handler = NULL;

void SetProgramName(const char *basename) {
  // Using the 'static std::string' for the program name is mostly harmless,
  // because (a) Kaldi logging is undefined before main(), and (b) no stdc++
  // string implementation has been found in the wild that would not be just
  // an empty string when zero-initialized but not yet constructed.
  program_name = basename;
}

/***** HELPER FUNCTIONS *****/

// Trim filename to at most 1 trailing directory long. Given a filename like
// "/a/b/c/d/e/f.cc", return "e/f.cc". Support both '/' and '\' as the path
// separator.
static const char *GetShortFileName(const char *path) {
  if (path == nullptr)
    return "";

  const char *prev = path, *last = path;
  while ((path = std::strpbrk(path, "\\/")) != nullptr) {
    ++path;
    prev = last;
    last = path;
  }
  return prev;
}

/***** STACK TRACE *****/

namespace internal {
bool LocateSymbolRange(const std::string &trace_name, size_t *begin,
                       size_t *end) {
  // Find the first '_' with leading ' ' or '('.
  *begin = std::string::npos;
  for (size_t i = 1; i < trace_name.size(); i++) {
    if (trace_name[i] != '_') {
      continue;
    }
    if (trace_name[i - 1] == ' ' || trace_name[i - 1] == '(') {
      *begin = i;
      break;
    }
  }
  if (*begin == std::string::npos) {
    return false;
  }
  *end = trace_name.find_first_of(" +", *begin);
  return *end != std::string::npos;
}
} // namespace internal

#ifdef HAVE_EXECINFO_H
static std::string Demangle(std::string trace_name) {
#ifndef HAVE_CXXABI_H
  return trace_name;
#else  // HAVE_CXXABI_H
  // Try demangle the symbol. We are trying to support the following formats
  // produced by different platforms:
  //
  // Linux:
  //   ./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d]
  //
  // Mac:
  //   0 server 0x000000010f67614d _ZNK5kaldi13MessageLogger10LogMessageEv + 813
  //
  // We want to extract the name e.g., '_ZN5kaldi13UnitTestErrorEv' and
  // demangle it info a readable name like kaldi::UnitTextError.
  size_t begin, end;
  if (!internal::LocateSymbolRange(trace_name, &begin, &end)) {
    return trace_name;
  }
  std::string symbol = trace_name.substr(begin, end - begin);
  int status;
  char *demangled_name = abi::__cxa_demangle(symbol.c_str(), 0, 0, &status);
  if (status == 0 && demangled_name != nullptr) {
    symbol = demangled_name;
    free(demangled_name);
  }
  return trace_name.substr(0, begin) + symbol +
         trace_name.substr(end, std::string::npos);
#endif // HAVE_CXXABI_H
}
#endif // HAVE_EXECINFO_H

static std::string KaldiGetStackTrace() {
  std::string ans;
#ifdef HAVE_EXECINFO_H
  const size_t KALDI_MAX_TRACE_SIZE = 50;
  const size_t KALDI_MAX_TRACE_PRINT = 50; // Must be even.
  // Buffer for the trace.
  void *trace[KALDI_MAX_TRACE_SIZE];
  // Get the trace.
  size_t size = backtrace(trace, KALDI_MAX_TRACE_SIZE);
  // Get the trace symbols.
  char **trace_symbol = backtrace_symbols(trace, size);
  if (trace_symbol == NULL)
    return ans;

  // Compose a human-readable backtrace string.
  ans += "[ Stack-Trace: ]\n";
  if (size <= KALDI_MAX_TRACE_PRINT) {
    for (size_t i = 0; i < size; i++) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
  } else { // Print out first+last (e.g.) 5.
    for (size_t i = 0; i < KALDI_MAX_TRACE_PRINT / 2; i++) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    ans += ".\n.\n.\n";
    for (size_t i = size - KALDI_MAX_TRACE_PRINT / 2; i < size; i++) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    if (size == KALDI_MAX_TRACE_SIZE)
      ans += ".\n.\n.\n"; // Stack was too long, probably a bug.
  }

  // We must free the array of pointers allocated by backtrace_symbols(),
  // but not the strings themselves.
  free(trace_symbol);
#endif // HAVE_EXECINFO_H
  return ans;
}

/***** KALDI LOGGING *****/

MessageLogger::MessageLogger(LogMessageEnvelope::Severity severity,
                             const char *func, const char *file, int32 line) {
  // Obviously, we assume the strings survive the destruction of this object.
  envelope_.severity = severity;
  envelope_.func = func;
  envelope_.file = GetShortFileName(file); // Points inside 'file'.
  envelope_.line = line;
}

void MessageLogger::LogMessage() const {
  // Send to the logging handler if provided.
  if (log_handler != NULL) {
    log_handler(envelope_, GetMessage().c_str());
    return;
  }

  // Otherwise, use the default Kaldi logging.
  // Build the log-message header.
  std::stringstream full_message;
  if (envelope_.severity > LogMessageEnvelope::kInfo) {
    full_message << "VLOG[" << envelope_.severity << "] (";
  } else {
    switch (envelope_.severity) {
    case LogMessageEnvelope::kInfo:
      //full_message << "LOG (\n";
      break;
    case LogMessageEnvelope::kWarning:
      //full_message << "WARNING (\n";
      break;
    case LogMessageEnvelope::kAssertFailed:
      full_message << "ASSERTION_FAILED (\n";
      break;
    case LogMessageEnvelope::kError:
    default: // If not the ERROR, it still an error!
      full_message << "ERROR (\n";
      break;
    }
  }
  // Add other info from the envelope and the message text.
  //full_message << program_name.c_str() << "[" KALDI_VERSION "]" << ':'
  //             << envelope_.func << "():" << envelope_.file << ':'
  //             << envelope_.line << ") " << GetMessage().c_str();

  // Add stack trace for errors and assertion failures, if available.
  if (envelope_.severity < LogMessageEnvelope::kWarning) {
    const std::string &stack_trace = KaldiGetStackTrace();
    if (!stack_trace.empty()) {
      full_message << "\n\n" << stack_trace;
    }
  }

  // Print the complete message to stderr.
  std::cerr << full_message.str();
}

/***** KALDI ASSERTS *****/

void KaldiAssertFailure_(const char *func, const char *file, int32 line,
                         const char *cond_str) {
  MessageLogger::Log() =
      MessageLogger(LogMessageEnvelope::kAssertFailed, func, file, line)
      << "Assertion failed: (" << cond_str << ")";
  fflush(NULL); // Flush all pending buffers, abort() may not flush stderr.
  std::abort();
}

/***** THIRD-PARTY LOG-HANDLER *****/

LogHandler SetLogHandler(LogHandler handler) {
  LogHandler old_handler = log_handler;
  log_handler = handler;
  return old_handler;
}

} // namespace kaldi
