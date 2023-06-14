// base/kaldi-error.cc

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
#include <execinfo.h>  // To get stack trace in error messages.
// If this #include fails there is an error in the Makefile, it does not
// support your platform well. Make sure HAVE_EXECINFO_H is undefined,
// and the code will compile.
#ifdef HAVE_CXXABI_H
#include <cxxabi.h>  // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif  // HAVE_CXXABI_H
#endif  // HAVE_EXECINFO_H

#include "base/kaldi-common.h"
#include "base/kaldi-error.h"
#include "base/version.h"

namespace kaldi {

/***** GLOBAL VARIABLES FOR LOGGING *****/

int32 g_kaldi_verbose_level = 0;
const char *g_program_name = NULL;
static LogHandler g_log_handler = NULL;

// If the program name was set (g_program_name != ""), GetProgramName
// returns the program name (without the path), e.g. "gmm-align".
// Otherwise it returns the empty string "".
const char *GetProgramName() {
  return g_program_name == NULL ? "" : g_program_name;
}

/***** HELPER FUNCTIONS *****/

// Given a filename like "/a/b/c/d/e/f.cc",  GetShortFileName
// returns "e/f.cc".  Does not currently work if backslash is
// the filename separator.
static const char *GetShortFileName(const char *filename) {
  const char *last_slash = strrchr(filename, '/');
  if (!last_slash) {
    return filename;
  } else {
    while (last_slash > filename && last_slash[-1] != '/')
      last_slash--;
    return last_slash;
  }
}


/***** STACKTRACE *****/

static std::string Demangle(std::string trace_name) {
#if defined(HAVE_CXXABI_H) && defined(HAVE_EXECINFO_H)
  // at input the string looks like:
  //   ./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d]
  // We want to extract the name e.g. '_ZN5kaldi13UnitTestErrorEv",
  // demangle it and return it.

  // try to locate '(' and '+', take the string in between,
  size_t begin(trace_name.find("(")),
         end(trace_name.rfind("+"));
  if (begin != std::string::npos && end != std::string::npos && begin < end) {
    trace_name = trace_name.substr(begin+1,end-(begin+1));
  }
  // demangle,
  int status;
  char *demangled_name = abi::__cxa_demangle(trace_name.c_str(), 0, 0, &status);
  std::string ans;
  if (status == 0) {
    ans = demangled_name;
    free(demangled_name);
  } else {
    ans = trace_name;
  }
  // return,
  return ans;
#else
  return trace_name;
#endif
}


static std::string KaldiGetStackTrace() {
  std::string ans;
#ifdef HAVE_EXECINFO_H
#define KALDI_MAX_TRACE_SIZE 50
#define KALDI_MAX_TRACE_PRINT 20  // must be even.
  // buffer for the trace,
  void *trace[KALDI_MAX_TRACE_SIZE];
  // get the trace,
  size_t size = backtrace(trace, KALDI_MAX_TRACE_SIZE);
  // get the trace symbols,
  char **trace_symbol = backtrace_symbols(trace, size);

  // Compose the 'string',
  ans += "[ Stack-Trace: ]\n";
  if (size <= KALDI_MAX_TRACE_PRINT) {
    for (size_t i = 0; i < size; i++) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
  } else {  // print out first+last (e.g.) 5.
    for (size_t i = 0; i < KALDI_MAX_TRACE_PRINT/2; i++) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    ans += ".\n.\n.\n";
    for (size_t i = size - KALDI_MAX_TRACE_PRINT/2; i < size; i++) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    if (size == KALDI_MAX_TRACE_SIZE)
      ans += ".\n.\n.\n";  // stack was too long, probably a bug.
  }

  // cleanup,
  free(trace_symbol);  // it's okay, just the pointers, not the strings.
#endif  // HAVE_EXECINFO_H
  return ans;
}


/***** KALDI LOGGING *****/

MessageLogger::MessageLogger(LogMessageEnvelope::Severity severity,
                             const char *func, const char *file, int32 line) {
  // Obviously, we assume the strings survive the destruction of this object.
  envelope_.severity = severity;
  envelope_.func = func;
  envelope_.file = GetShortFileName(file);  // Pointer inside 'file'.
  envelope_.line = line;
}


MessageLogger::~MessageLogger() KALDI_NOEXCEPT(false) {
  // remove trailing '\n',
  std::string str = ss_.str();
  while (!str.empty() && str[str.length() - 1] == '\n')
    str.resize(str.length() - 1);

  // print the mesage (or send to logging handler),
  MessageLogger::HandleMessage(envelope_, str.c_str());
}


void MessageLogger::HandleMessage(const LogMessageEnvelope &envelope,
                                  const char *message) {
  // Send to a logging handler if provided.
  if (g_log_handler != NULL) {
    g_log_handler(envelope, message);
  } else {
    // Otherwise, we use the default Kaldi logging.
    // Build the log-message 'header',
    std::stringstream header;
    if (envelope.severity > LogMessageEnvelope::kInfo) {
      header << "VLOG[" << envelope.severity << "] (";
    } else {
      switch (envelope.severity) {
        case LogMessageEnvelope::kInfo :
          header << "LOG (";
          break;
        case LogMessageEnvelope::kWarning :
          header << "WARNING (";
          break;
        case LogMessageEnvelope::kError :
          header << "ERROR (";
          break;
        case LogMessageEnvelope::kAssertFailed :
          header << "ASSERTION_FAILED (";
          break;
        default:
          abort();  // coding error (unknown 'severity'),
      }
    }
    // fill the other info from the envelope,
    header << GetProgramName() << "[" KALDI_VERSION "]" << ':'
           << envelope.func << "():" << envelope.file << ':' << envelope.line
           << ")";

    // Printing the message,
    if (envelope.severity >= LogMessageEnvelope::kWarning) {
      // VLOG, LOG, WARNING:
      fprintf(stderr, "%s %s\n", header.str().c_str(), message);
    } else {
      // ERROR, ASSERT_FAILED (print with stack-trace):
      fprintf(stderr, "%s %s\n\n%s\n", header.str().c_str(), message,
              KaldiGetStackTrace().c_str());
    }
  }

  // Should we throw exception, or abort?
  switch (envelope.severity) {
    case LogMessageEnvelope::kAssertFailed:
      abort(); // ASSERT_FAILED,
      break;
    case LogMessageEnvelope::kError:
      if (!std::uncaught_exception()) {
        // throw exception with empty message,
        throw std::runtime_error(""); // KALDI_ERR,
      } else {
        // If we got here, this thread has already thrown exception,
        // and this exception has not yet arrived to its 'catch' clause...
        // Throwing a new exception would be unsafe!
        // (can happen during 'stack unwinding', if we have 'KALDI_ERR << msg'
        // in a destructor of some local object).
        abort();
      }
      break;
  }
}


/***** KALDI ASSERTS *****/

void KaldiAssertFailure_(const char *func, const char *file,
                         int32 line, const char *cond_str) {
  MessageLogger ml(LogMessageEnvelope::kAssertFailed, func, file, line);
  ml.stream() << ": '" << cond_str << "' ";
}


/***** THIRD-PARTY LOG-HANDLER *****/

LogHandler SetLogHandler(LogHandler new_handler) {
  LogHandler old_handler = g_log_handler;
  g_log_handler = new_handler;
  return old_handler;
}

}  // end namespace kaldi
