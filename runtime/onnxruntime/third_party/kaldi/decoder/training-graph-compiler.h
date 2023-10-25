// decoder/training-graph-compiler.h

// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)

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
#ifndef KALDI_DECODER_TRAINING_GRAPH_COMPILER_H_
#define KALDI_DECODER_TRAINING_GRAPH_COMPILER_H_

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"


namespace kaldi {

struct TrainingGraphCompilerOptions {

  BaseFloat transition_scale;
  BaseFloat self_loop_scale;
  bool rm_eps;
  bool reorder;  // (Dan-style graphs)

  explicit TrainingGraphCompilerOptions(BaseFloat transition_scale = 1.0,
                                        BaseFloat self_loop_scale = 1.0,
                                        bool b = true) :
      transition_scale(transition_scale),
      self_loop_scale(self_loop_scale),
      rm_eps(false),
      reorder(b) { }

  void Register(OptionsItf *opts) {
    opts->Register("transition-scale", &transition_scale, "Scale of transition "
                   "probabilities (excluding self-loops)");
    opts->Register("self-loop-scale", &self_loop_scale, "Scale of self-loop vs. "
                   "non-self-loop probability mass ");
    opts->Register("reorder", &reorder, "Reorder transition ids for greater decoding efficiency.");
    opts->Register("rm-eps", &rm_eps,  "Remove [most] epsilons before minimization (only applicable "
                   "if disambig symbols present)");
  }
};


class TrainingGraphCompiler {
 public:
  TrainingGraphCompiler(const TransitionModel &trans_model,  // Maintains reference to this object.
                        const ContextDependency &ctx_dep,  // And this.
                        fst::VectorFst<fst::StdArc> *lex_fst,  // Takes ownership of this object.
                        // It should not contain disambiguation symbols or subsequential symbol,
                        // but it should contain optional silence.
                        const std::vector<int32> &disambig_syms, // disambig symbols in phone symbol table.
                        const TrainingGraphCompilerOptions &opts);


  // CompileGraph compiles a single training graph its input is a
  // weighted acceptor (G) at the word level, its output is HCLG.
  // Note: G could actually be a transducer, it would also work.
  // This function is not const for technical reasons involving the cache.
  // if not for "table_compose" we could make it const.
  bool CompileGraph(const fst::VectorFst<fst::StdArc> &word_grammar,
                    fst::VectorFst<fst::StdArc> *out_fst);

  // Same as `CompileGraph`, but uses an external LG fst.
  bool CompileGraphFromLG(const fst::VectorFst<fst::StdArc> &phone2word_fst,
                                   fst::VectorFst<fst::StdArc> * out_fst);

  // CompileGraphs allows you to compile a number of graphs at the same
  // time.  This consumes more memory but is faster.
  bool CompileGraphs(
      const std::vector<const fst::VectorFst<fst::StdArc> *> &word_fsts,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

  // This version creates an FST from the text and calls CompileGraph.
  bool CompileGraphFromText(const std::vector<int32> &transcript,
                            fst::VectorFst<fst::StdArc> *out_fst);

  // This function creates FSTs from the text and calls CompileGraphs.
  bool CompileGraphsFromText(
      const std::vector<std::vector<int32> >  &word_grammar,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);


  ~TrainingGraphCompiler() { delete lex_fst_; }
 private:
  const TransitionModel &trans_model_;
  const ContextDependency &ctx_dep_;
  fst::VectorFst<fst::StdArc> *lex_fst_; // lexicon FST (an input; we take
  // ownership as we need to modify it).
  std::vector<int32> disambig_syms_; // disambig symbols (if any) in the phone
  int32 subsequential_symbol_;  // search in ../fstext/context-fst.h for more info.
  // symbol table.
  fst::TableComposeCache<fst::Fst<fst::StdArc> > lex_cache_;  // stores matcher..
  // this is one of Dan's extensions.

  TrainingGraphCompilerOptions opts_;
};



}  // end namespace kaldi.

#endif
