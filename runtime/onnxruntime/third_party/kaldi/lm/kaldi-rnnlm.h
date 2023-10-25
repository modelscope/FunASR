// lm/kaldi-rnnlm.h

// Copyright 2015  Guoguo Chen

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

#ifndef KALDI_LM_KALDI_RNNLM_H_
#define KALDI_LM_KALDI_RNNLM_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "lm/mikolov-rnnlm-lib.h"
#include "util/common-utils.h"

namespace kaldi {

struct KaldiRnnlmWrapperOpts {
  std::string unk_symbol;
  std::string eos_symbol;

  KaldiRnnlmWrapperOpts() : unk_symbol("<RNN_UNK>"), eos_symbol("</s>") {}

  void Register(OptionsItf *opts) {
    opts->Register("unk-symbol", &unk_symbol, "Symbol for out-of-vocabulary "
                   "words in rnnlm.");
    opts->Register("eos-symbol", &eos_symbol, "End of sentence symbol in "
                   "rnnlm.");
  }
};

class KaldiRnnlmWrapper {
 public:
  KaldiRnnlmWrapper(const KaldiRnnlmWrapperOpts &opts,
                    const std::string &unk_prob_rspecifier,
                    const std::string &word_symbol_table_rxfilename,
                    const std::string &rnnlm_rxfilename);

  int32 GetHiddenLayerSize() const { return rnnlm_.getHiddenLayerSize(); }

  int32 GetEos() const { return eos_; }

  BaseFloat GetLogProb(int32 word, const std::vector<int32> &wseq,
                       const std::vector<float> &context_in,
                       std::vector<float> *context_out);

 private:
  rnnlm::CRnnLM rnnlm_;
  std::vector<std::string> label_to_word_;
  int32 eos_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiRnnlmWrapper);
};

class RnnlmDeterministicFst
    : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  RnnlmDeterministicFst(int32 max_ngram_order, KaldiRnnlmWrapper *rnnlm);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;

  KaldiRnnlmWrapper *rnnlm_;
  int32 max_ngram_order_;
  std::vector<std::vector<float> > state_to_context_;
};

}  // namespace kaldi

#endif  // KALDI_LM_KALDI_RNNLM_H_
