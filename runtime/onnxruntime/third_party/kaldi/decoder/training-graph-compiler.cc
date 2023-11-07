// decoder/training-graph-compiler.cc

// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
//                2021  Xiaomi Corporation (Author: Junbo Zhang)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "decoder/training-graph-compiler.h"
#include "hmm/hmm-utils.h" // for GetHTransducer

namespace kaldi {


TrainingGraphCompiler::TrainingGraphCompiler(const TransitionModel &trans_model,
                                             const ContextDependency &ctx_dep,  // Does not maintain reference to this.
                                             fst::VectorFst<fst::StdArc> *lex_fst,
                                             const std::vector<int32> &disambig_syms,
                                             const TrainingGraphCompilerOptions &opts):
    trans_model_(trans_model), ctx_dep_(ctx_dep), lex_fst_(lex_fst),
    disambig_syms_(disambig_syms), opts_(opts) {
  using namespace fst;
  const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.

  KALDI_ASSERT(!phone_syms.empty());
  KALDI_ASSERT(IsSortedAndUniq(phone_syms));
  SortAndUniq(&disambig_syms_);
  for (int32 i = 0; i < disambig_syms_.size(); i++)
    if (std::binary_search(phone_syms.begin(), phone_syms.end(),
                           disambig_syms_[i]))
      KALDI_ERR << "Disambiguation symbol " << disambig_syms_[i]
                << " is also a phone.";

  subsequential_symbol_ = 1 + phone_syms.back();
  if (!disambig_syms_.empty() && subsequential_symbol_ <= disambig_syms_.back())
    subsequential_symbol_ = 1 + disambig_syms_.back();
  
  if (lex_fst == NULL) return;

  {
    int32 N = ctx_dep.ContextWidth(),
        P = ctx_dep.CentralPosition();
    if (P != N-1)
      AddSubsequentialLoop(subsequential_symbol_, lex_fst_);  // This is needed for
    // systems with right-context or we will not successfully compose
    // with C.
  }

  {  // make sure lexicon is olabel sorted.
    fst::OLabelCompare<fst::StdArc> olabel_comp;
    fst::ArcSort(lex_fst_, olabel_comp);
  }
}

bool TrainingGraphCompiler::CompileGraphFromText(
    const std::vector<int32> &transcript,
    fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;
  VectorFst<StdArc> word_fst;
  MakeLinearAcceptor(transcript, &word_fst);
  return CompileGraph(word_fst, out_fst);
}

bool TrainingGraphCompiler::CompileGraphFromLG(const fst::VectorFst<fst::StdArc> &phone2word_fst,
                                               fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;

  KALDI_ASSERT(phone2word_fst.Start() != kNoStateId);

  const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.

  // inv_cfst will be expanded on the fly, as needed.
  InverseContextFst inv_cfst(subsequential_symbol_,
                             phone_syms,
                             disambig_syms_,
                             ctx_dep_.ContextWidth(),
                             ctx_dep_.CentralPosition());


  VectorFst<StdArc> ctx2word_fst;
  ComposeDeterministicOnDemandInverse(phone2word_fst, &inv_cfst, &ctx2word_fst);
  // now ctx2word_fst is C * LG, assuming phone2word_fst is written as LG.
  KALDI_ASSERT(ctx2word_fst.Start() != kNoStateId);

  HTransducerConfig h_cfg;
  h_cfg.transition_scale = opts_.transition_scale;

  std::vector<int32> disambig_syms_h; // disambiguation symbols on
  // input side of H.
  VectorFst<StdArc> *H = GetHTransducer(inv_cfst.IlabelInfo(),
                                        ctx_dep_,
                                        trans_model_,
                                        h_cfg,
                                        &disambig_syms_h);

  VectorFst<StdArc> &trans2word_fst = *out_fst;  // transition-id to word.
  TableCompose(*H, ctx2word_fst, &trans2word_fst);

  KALDI_ASSERT(trans2word_fst.Start() != kNoStateId);

  // Epsilon-removal and determinization combined. This will fail if not determinizable.
  DeterminizeStarInLog(&trans2word_fst);

  if (!disambig_syms_h.empty()) {
    RemoveSomeInputSymbols(disambig_syms_h, &trans2word_fst);
    // we elect not to remove epsilons after this phase, as it is
    // a little slow.
    if (opts_.rm_eps)
      RemoveEpsLocal(&trans2word_fst);
  }


  // Encoded minimization.
  MinimizeEncoded(&trans2word_fst);

  std::vector<int32> disambig;
  bool check_no_self_loops = true;
  AddSelfLoops(trans_model_,
               disambig,
               opts_.self_loop_scale,
               opts_.reorder,
               check_no_self_loops,
               &trans2word_fst);

  delete H;
  return true;
}

bool TrainingGraphCompiler::CompileGraph(const fst::VectorFst<fst::StdArc> &word_fst,
                                         fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;
  KALDI_ASSERT(lex_fst_ !=NULL);
  KALDI_ASSERT(out_fst != NULL);

  VectorFst<StdArc> phone2word_fst;
  // TableCompose more efficient than compose.
  TableCompose(*lex_fst_, word_fst, &phone2word_fst, &lex_cache_);
  return CompileGraphFromLG(phone2word_fst, out_fst);
}

bool TrainingGraphCompiler::CompileGraphsFromText(
    const std::vector<std::vector<int32> > &transcripts,
    std::vector<fst::VectorFst<fst::StdArc>*> *out_fsts) {
  using namespace fst;
  std::vector<const VectorFst<StdArc>* > word_fsts(transcripts.size());
  for (size_t i = 0; i < transcripts.size(); i++) {
    VectorFst<StdArc> *word_fst = new VectorFst<StdArc>();
    MakeLinearAcceptor(transcripts[i], word_fst);
    word_fsts[i] = word_fst;
  }
  bool ans = CompileGraphs(word_fsts, out_fsts);
  for (size_t i = 0; i < transcripts.size(); i++)
    delete word_fsts[i];
  return ans;
}

bool TrainingGraphCompiler::CompileGraphs(
    const std::vector<const fst::VectorFst<fst::StdArc>* > &word_fsts,
    std::vector<fst::VectorFst<fst::StdArc>* > *out_fsts) {
  out_fsts->resize(word_fsts.size(), NULL);
  for (size_t i = 0; i < word_fsts.size(); i++) {
    fst::VectorFst<fst::StdArc> trans2word_fst;
    if (!CompileGraph(*(word_fsts[i]), &trans2word_fst)) return false;
    (*out_fsts)[i] = trans2word_fst.Copy();
  }
  return true;
}


}  // end namespace kaldi
