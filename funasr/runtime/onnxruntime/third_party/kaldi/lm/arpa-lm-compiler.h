// lm/arpa-lm-compiler.h

// Copyright 2009-2011 Gilles Boulianne
// Copyright 2016 Smart Action LLC (kkm)

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

#ifndef KALDI_LM_ARPA_LM_COMPILER_H_
#define KALDI_LM_ARPA_LM_COMPILER_H_

#include <fst/fstlib.h>

#include "lm/arpa-file-parser.h"

namespace kaldi {

class ArpaLmCompilerImplInterface;

class ArpaLmCompiler : public ArpaFileParser {
 public:
  ArpaLmCompiler(const ArpaParseOptions& options, int sub_eps,
                 fst::SymbolTable* symbols)
      : ArpaFileParser(options, symbols),
        sub_eps_(sub_eps), impl_(NULL) {
  }
  ~ArpaLmCompiler();

  const fst::StdVectorFst& Fst() const { return fst_; }
  fst::StdVectorFst* MutableFst() { return &fst_; }

 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable();
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete();


 private:
  // this function removes states that only have a backoff arc coming
  // out of them.
  void RemoveRedundantStates();
  void Check() const;

  int sub_eps_;
  ArpaLmCompilerImplInterface* impl_;  // Owned.
  fst::StdVectorFst fst_;
  template <class HistKey> friend class ArpaLmCompilerImpl;
};

}  // namespace kaldi

#endif  // KALDI_LM_ARPA_LM_COMPILER_H_
