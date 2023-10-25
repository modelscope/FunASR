// fstext/rand-fst.h

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

#ifndef KALDI_FSTEXT_RAND_FST_H_
#define KALDI_FSTEXT_RAND_FST_H_

#include <sstream>
#include <string>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "base/kaldi-math.h"

namespace fst {

// Note: all weights are constructed from nonnegative floats.
// (so no "negative costs").
struct RandFstOptions {
  size_t n_syms;
  size_t n_states;
  size_t n_arcs;
  size_t n_final;
  bool allow_empty;
  bool acyclic;
  float weight_multiplier;
  RandFstOptions() {  // Initializes the options randomly.
    n_syms = 2 + kaldi::Rand() % 5;
    n_states = 3 + kaldi::Rand() % 10;
    n_arcs = 5 + kaldi::Rand() % 30;
    n_final = 1 + kaldi::Rand()%3;
    allow_empty = true;
    acyclic = false;
    weight_multiplier = 0.25;
  }
};


/// Returns a random FST.  Useful for randomized algorithm testing.
/// Only works if weight can be constructed from float.
template<class Arc> VectorFst<Arc>* RandFst(RandFstOptions opts = RandFstOptions() ) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();

 start:

  // Create states.
  std::vector<StateId> all_states;
  for (size_t i = 0;i < (size_t)opts.n_states;i++) {
    StateId this_state = fst->AddState();
    if (i == 0) fst->SetStart(i);
    all_states.push_back(this_state);
  }
  // Set final states.
  for (size_t j = 0;j < (size_t)opts.n_final;j++) {
    StateId id = all_states[kaldi::Rand() % opts.n_states];
    Weight weight = (Weight)(opts.weight_multiplier*(kaldi::Rand() % 5));
    fst->SetFinal(id, weight);
  }
  // Create arcs.
  for (size_t i = 0;i < (size_t)opts.n_arcs;i++) {
    Arc a;
    StateId start_state;
    if(!opts.acyclic) { // no restriction on arcs.
      start_state = all_states[kaldi::Rand() % opts.n_states];
      a.nextstate = all_states[kaldi::Rand() % opts.n_states];
    } else {
      start_state = all_states[kaldi::Rand() % (opts.n_states-1)];
      a.nextstate = start_state + 1 + (kaldi::Rand() % (opts.n_states-start_state-1));
    }
    a.ilabel = kaldi::Rand() % opts.n_syms;
    a.olabel = kaldi::Rand() % opts.n_syms;  // same input+output vocab.
    a.weight = (Weight) (opts.weight_multiplier*(kaldi::Rand() % 4));

    fst->AddArc(start_state, a);
  }

  // Trim resulting FST.
  Connect(fst);
  if (opts.acyclic)
    assert(fst->Properties(kAcyclic, true) & kAcyclic);
  if (fst->Start() == kNoStateId && !opts.allow_empty) {
    goto start;
  }
  return fst;
}


/// Returns a random FST.  Useful for randomized algorithm testing.
/// Only works if weight can be constructed from a pair of floats
template<class Arc> VectorFst<Arc>* RandPairFst(RandFstOptions opts = RandFstOptions() ) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();

 start:

  // Create states.
  std::vector<StateId> all_states;
  for (size_t i = 0;i < (size_t)opts.n_states;i++) {
    StateId this_state = fst->AddState();
    if (i == 0) fst->SetStart(i);
    all_states.push_back(this_state);
  }
  // Set final states.
  for (size_t j = 0; j < (size_t)opts.n_final;j++) {
    StateId id = all_states[kaldi::Rand() % opts.n_states];
    Weight weight (opts.weight_multiplier*(kaldi::Rand() % 5), opts.weight_multiplier*(kaldi::Rand() % 5));
    fst->SetFinal(id, weight);
  }
  // Create arcs.
  for (size_t i = 0;i < (size_t)opts.n_arcs;i++) {
    Arc a;
    StateId start_state;
    if(!opts.acyclic) { // no restriction on arcs.
      start_state = all_states[kaldi::Rand() % opts.n_states];
      a.nextstate = all_states[kaldi::Rand() % opts.n_states];
    } else {
      start_state = all_states[kaldi::Rand() % (opts.n_states-1)];
      a.nextstate = start_state + 1 + (kaldi::Rand() % (opts.n_states-start_state-1));
    }
    a.ilabel = kaldi::Rand() % opts.n_syms;
    a.olabel = kaldi::Rand() % opts.n_syms;  // same input+output vocab.
    a.weight = Weight (opts.weight_multiplier*(kaldi::Rand() % 4), opts.weight_multiplier*(kaldi::Rand() % 4));

    fst->AddArc(start_state, a);
  }

  // Trim resulting FST.
  Connect(fst);
  if (opts.acyclic)
    assert(fst->Properties(kAcyclic, true) & kAcyclic);
  if (fst->Start() == kNoStateId && !opts.allow_empty) {
    goto start;
  }
  return fst;
}


} // end namespace fst.


#endif

