// base/kaldi-error-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "base/kaldi-common.h"

// testing that we get the stack trace.
namespace kaldi {

void MyFunction2() { KALDI_ERR << "Ignore this error"; }

void MyFunction1() { MyFunction2(); }

void UnitTestError() {
  {
    std::cerr << "Ignore next error:\n";
    MyFunction1();
  }
}

void VerifySymbolRange(const std::string &trace, const bool want_found,
                       const std::string &want_symbol) {
  size_t begin, end;
  const bool found = internal::LocateSymbolRange(trace, &begin, &end);
  if (found != want_found) {
    KALDI_ERR << "Found mismatch, got " << found << " want " << want_found;
  }
  if (!found) {
    return;
  }
  const std::string symbol = trace.substr(begin, end - begin);
  if (symbol != want_symbol) {
    KALDI_ERR << "Symbol mismatch, got " << symbol << " want " << want_symbol;
  }
}

void TestLocateSymbolRange() {
  VerifySymbolRange("", false, "");
  VerifySymbolRange(
      R"TRACE(./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d])TRACE",
      true, "_ZN5kaldi13UnitTestErrorEv");
  // It is ok thread_start is not found because it is a C symbol.
  VerifySymbolRange(
      R"TRACE(31  libsystem_pthread.dylib             0x00007fff6fe4e40d thread_start + 13)TRACE",
      false, "");
  VerifySymbolRange(
      R"TRACE(0 server 0x000000010f67614d _ZNK5kaldi13MessageLogger10LogMessageEv + 813)TRACE",
      true, "_ZNK5kaldi13MessageLogger10LogMessageEv");
  VerifySymbolRange(
      R"TRACE(29  libsystem_pthread.dylib             0x00007fff6fe4f2eb _pthread_body + 126)TRACE",
      true, "_pthread_body");
}

} // namespace kaldi

int main() {
  kaldi::TestLocateSymbolRange();

  kaldi::SetProgramName("/foo/bar/kaldi-error-test");
  try {
    kaldi::UnitTestError();
    KALDI_ASSERT(0); // should not happen.
    exit(1);
  } catch (kaldi::KaldiFatalError &e) {
    std::cout << "The error we generated was: '" << e.KaldiMessage() << "'\n";
  }
}
