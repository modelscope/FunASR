// fstext/deterministic-fst.h

// Copyright 2011-2012 Gilles Boulianne
//                2014 Telepoint Global Hosting Service, LLC. (Author: David Snyder)
//           2012-2015 Johns Hopkins University (author: Daniel Povey)

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
//
// This file includes material from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: riley@google.com (Michael Riley)

#ifndef KALDI_FSTEXT_DETERMINISTIC_FST_H_
#define KALDI_FSTEXT_DETERMINISTIC_FST_H_

/* This header defines the DeterministicOnDemand interface,
   which is an FST with a special interface that allows
   only a single arc with a non-epsilon input symbol
   out of each state.
*/

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "util/stl-utils.h"

namespace fst {

/// \addtogroup deterministic_fst_group "Classes and functions related to on-demand deterministic FST's"
/// @{


/// class DeterministicOnDemandFst is an "FST-like" base-class.  It does not
/// actually inherit from any Fst class because its interface is not exactly the
/// same; it's much smaller.  It assumes that the FST can have only one arc for
/// any given input symbol, which makes the GetArc function below possible.
/// (The FST is also assumed to be free of input epsilons).  Note: we don't use
/// "const" in this interface, because it creates problems when we do things
/// like caching.
template<class Arc>
class DeterministicOnDemandFst {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  virtual StateId Start() = 0;

  virtual Weight Final(StateId s) = 0;

  /// Note: ilabel must not be epsilon.
  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc) = 0;

  virtual ~DeterministicOnDemandFst() { }
};

/**
   This class wraps an Fst, representing a language model, using the interface
   for "BackoffDeterministicOnDemandFst".  We expect that backoff arcs in the
   language model will have the epsilon label (label 0) on the arcs, and that
   there will be no other epsilons in the language model.  We follow the epsilon
   arcs as long as a particular arc (or a final-prob) is not found at the
   current state.
 */
template<class Arc>
class BackoffDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  explicit BackoffDeterministicOnDemandFst(const Fst<Arc> &fst);

  StateId Start() { return fst_.Start(); }

  Weight Final(StateId s);

  bool GetArc(StateId s, Label ilabel, Arc *oarc);

 private:
  inline StateId GetBackoffState(StateId s, Weight *w);

  const Fst<Arc> &fst_;
};

/**
   Class ScaleDeterministicOnDemandFst takes another DeterministicOnDemandFst
   and scales the weights (like applying a language-model scale).  For instance,
   to subtract existing LM scores from a lattice you could use this with
   a negative weight; and to interpolate LMs you can also use this with
   weights less than one.

   It's specialized for StdArc because there is no generic way to scale weights.
*/
class ScaleDeterministicOnDemandFst:  public DeterministicOnDemandFst<StdArc> {
 public:
  typedef StdArc::Weight Weight;
  typedef StdArc::StateId StateId;
  typedef StdArc::Label Label;

  // Constructor does not take ownership of 'det_fst'.
  ScaleDeterministicOnDemandFst(float scale,
                                DeterministicOnDemandFst<StdArc> *det_fst):
      scale_(scale), det_fst_(*det_fst) { }

  StateId Start() { return det_fst_.Start(); }

  Weight Final(StateId s) {
    // Note: Weight is indirectly a typedef to TropicalWeight.
    Weight final = det_fst_.Final(s);
    if (final == Weight::Zero()) return Weight::Zero();
    else return TropicalWeight(final.Value() * scale_);
  }

  inline bool GetArc(StateId s, Label ilabel, StdArc *oarc) {
    if (det_fst_.GetArc(s, ilabel, oarc)) {
      oarc->weight = TropicalWeight(oarc->weight.Value() * scale_);
      return true;
    } else {
      return false;
    }
  }

 private:
  float scale_;
  DeterministicOnDemandFst<StdArc> &det_fst_;
};

/**
  The class UnweightedNgramFst is a DeterministicOnDemandFst whose states encode
  an n-gram history. Conceptually, for n-gram order n and k labels, the FST is an
  unweighted acceptor with about k^(n-1) states (ignoring end effects). However,
  the FST is created on demand and doesn't need the label vocabulary; GetArc
  matches on any input label. This class is primarily used together with
  ComposeDeterministicOnDemandFst to expand the n-gram history of lattices, ensuring
  that each arc has a sufficiently long unique word history.
 */
template<class Arc>
class UnweightedNgramFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  UnweightedNgramFst(int n);

  StateId Start() { return start_state_; };

  Weight Final(StateId s);

  bool GetArc(StateId s, Label ilabel, Arc *oarc);

 private:
  typedef unordered_map<std::vector<Label>,
    StateId, kaldi::VectorHasher<Label> > MapType;
  // The order of the n-gram.
  int n_;
  MapType state_map_;
  StateId start_state_;
  // Map from history-state to pair.
  std::vector<std::vector<Label> > state_vec_;
};

template<class Arc>
class ComposeDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  /// Note: constructor does not "take ownership" of the input fst's.  The input
  /// fst's should be treated as const, in that their contents do not change,
  /// but they are not const as the DeterministicOnDemandFst's data-access
  /// functions are not const, for reasons relating to caching.
  ComposeDeterministicOnDemandFst(DeterministicOnDemandFst<Arc> *fst1,
                                  DeterministicOnDemandFst<Arc> *fst2);

  virtual StateId Start() { return start_state_; }

  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc);

 private:
  DeterministicOnDemandFst<Arc> *fst1_;
  DeterministicOnDemandFst<Arc> *fst2_;
  typedef unordered_map<std::pair<StateId, StateId>, StateId, kaldi::PairHasher<StateId> > MapType;
  MapType state_map_;
  std::vector<std::pair<StateId, StateId> > state_vec_; // maps from
  // StateId to pair.
  StateId next_state_;
  StateId start_state_;
};

template<class Arc>
class CacheDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  /// We don't take ownership of this pointer.  The argument is "really" const.
  CacheDeterministicOnDemandFst(DeterministicOnDemandFst<Arc> *fst,
                                StateId num_cached_arcs = 100000);

  virtual StateId Start() { return fst_->Start(); }

  /// We don't bother caching the final-probs, just the arcs.
  virtual Weight Final(StateId s) { return fst_->Final(s); }

  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc);

 private:
  // Get index for cached arc.
  inline size_t GetIndex(StateId src_state, Label ilabel);

  DeterministicOnDemandFst<Arc> *fst_;
  StateId num_cached_arcs_;
  std::vector<std::pair<StateId, Arc> > cached_arcs_;
};


/// This class is for didactic purposes, it does not really do anything.
/// It shows how you would wrap a language model.  Note: you should probably
/// have <s> and </s> not be real words in your LM, but <s> correspond somehow
/// to the initial-state of the LM, and </s> be encoded in the final-probs.
template<class Arc>
class LmExampleDeterministicOnDemandFst: public DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  LmExampleDeterministicOnDemandFst(void *lm,
                                    Label bos_symbol,
                                    Label eos_symbol);


  virtual StateId Start() { return start_state_; }

  /// We don't bother caching the final-probs, just the arcs.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc);

 private:
  // Get index for cached arc.
  inline size_t GetIndex(StateId src_state, Label ilabel);

  typedef unordered_map<std::vector<Label>, StateId, kaldi::VectorHasher<Label> > MapType;
  void *lm_;
  Label bos_symbol_; // beginning of sentence symbol
  Label eos_symbol_; // end of sentence symbol.
  // This example code does not handle <UNK>; we assume the LM has the same vocab as
  // the recognizer.
  MapType state_map_;
  StateId start_state_;
  std::vector<std::vector<Label> > state_vec_; // maps from history-state to pair.

  void *lm; // wouldn't really be void.
};


// Compose an FST (which may be a lattice) with a DeterministicOnDemandFst and
// store the result in fst_composed.  This is mainly used for expanding lattice
// n-gram histories, where fst1 is a lattice and fst2 is an UnweightedNgramFst.
// This does not call Connect.
template<class Arc>
void ComposeDeterministicOnDemand(const Fst<Arc> &fst1,
                                  DeterministicOnDemandFst<Arc> *fst2,
                                  MutableFst<Arc> *fst_composed);

/**
   This function does
   '*fst_composed = Compose(Inverse(*fst2), fst1)'
   Note that the arguments are reversed; this is unfortunate but it's
   because the fst2 argument needs to be non-const and non-const arguments
   must follow const ones.
   This is the counterpart to ComposeDeterministicOnDemand, used for
   the case where the DeterministicOnDemandFst is on the left.  The
   reason why we need to make the left-hand argument to compose the
   inverse of 'fst2' (i.e. with the input and output symbols swapped),
   is that the DeterministicOnDemandFst interface only supports lookup
   by ilabel (see its function GetArc).
   This does not call Connect().
*/
template<class Arc>
void ComposeDeterministicOnDemandInverse(const Fst<Arc> &fst1,
                                         DeterministicOnDemandFst<Arc> *fst2,
                                         MutableFst<Arc> *fst_composed);




/// @}

}  // namespace fst

#include "deterministic-fst-inl.h"

#endif
