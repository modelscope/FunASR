// fstext/fstext-utils-inl.h

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//                2014  Telepoint Global Hosting Service, LLC. (Author: David Snyder)

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

#ifndef KALDI_FSTEXT_FSTEXT_UTILS_INL_H_
#define KALDI_FSTEXT_FSTEXT_UTILS_INL_H_
#include <cstring>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"
#include "util/kaldi-io.h"
#include "fstext/factor.h"
#include "fstext/pre-determinize.h"
#include "fstext/determinize-star.h"

#include <sstream>
#include <algorithm>
#include <string>

namespace fst {



template<class Arc>
typename Arc::Label HighestNumberedOutputSymbol(const Fst<Arc> &fst) {
  typename Arc::Label ans = 0;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      ans = std::max(ans, arc.olabel);
    }
  }
  return ans;
}

template<class Arc>
typename Arc::Label HighestNumberedInputSymbol(const Fst<Arc> &fst) {
  typename Arc::Label ans = 0;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      ans = std::max(ans, arc.ilabel);
    }
  }
  return ans;
}

template<class Arc>
typename Arc::StateId NumArcs(const ExpandedFst<Arc> &fst) {
  typedef typename Arc::StateId StateId;
  StateId num_arcs = 0;
  for (StateId s = 0; s < fst.NumStates(); s++)
    num_arcs += fst.NumArcs(s);
  return num_arcs;
}

template<class Arc, class I>
void GetOutputSymbols(const Fst<Arc> &fst,
                      bool include_eps,
                      std::vector<I> *symbols) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  std::set<I> all_syms;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      all_syms.insert(arc.olabel);
    }
  }

  // Remove epsilon, if instructed.
  if (!include_eps && !all_syms.empty() && *all_syms.begin() == 0)
    all_syms.erase(0);
  KALDI_ASSERT(symbols != NULL);
  kaldi::CopySetToVector(all_syms, symbols);
}

template<class Arc, class I>
void GetInputSymbols(const Fst<Arc> &fst,
                     bool include_eps,
                     std::vector<I> *symbols) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  unordered_set<I> all_syms;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done();  aiter.Next()) {
      const Arc &arc = aiter.Value();
      all_syms.insert(arc.ilabel);
    }
  }
  // Remove epsilon, if instructed.
  if (!include_eps && all_syms.count(0) != 0)
    all_syms.erase(0);
  KALDI_ASSERT(symbols != NULL);
  kaldi::CopySetToVector(all_syms, symbols);
  std::sort(symbols->begin(), symbols->end());
}

template<class Arc, class I>
class RemoveSomeInputSymbolsMapper {
public:
  Arc operator ()(const Arc &arc_in) {
    Arc ans = arc_in;
    if (to_remove_set_.count(ans.ilabel) != 0) ans.ilabel = 0;  // remove this symbol
    return ans;
  }
  MapFinalAction FinalAction() { return MAP_NO_SUPERFINAL; }
  MapSymbolsAction InputSymbolsAction() { return MAP_CLEAR_SYMBOLS; }
  MapSymbolsAction OutputSymbolsAction() { return MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const {
    // remove the following as we don't know now if any of them are true.
    uint64 to_remove = kAcceptor|kNotAcceptor|kIDeterministic|kNonIDeterministic|
        kNoEpsilons|kNoIEpsilons|kILabelSorted|kNotILabelSorted;
    return props & ~to_remove;
  }
  RemoveSomeInputSymbolsMapper(const std::vector<I> &to_remove):
      to_remove_set_(to_remove) {
    KALDI_ASSERT_IS_INTEGER_TYPE(I);
         assert(to_remove_set_.count(0) == 0);  // makes no sense to remove epsilon.
       }
private:
  kaldi::ConstIntegerSet<I> to_remove_set_;
};

template<class Arc, class I>
using LookaheadFst = ArcMapFst<Arc, Arc, RemoveSomeInputSymbolsMapper<Arc, I> >;

// Lookahead composition is used for optimized online
// composition of FSTs during decoding. See
// nnet3/nnet3-latgen-faster-lookahead.cc. For details of compose filters
// see DefaultLookAhead in fst/compose.h
template<class Arc, class I>
LookaheadFst<Arc, I> *LookaheadComposeFst(const Fst<Arc> &ifst1,
                                          const Fst<Arc> &ifst2,
                                          const std::vector<I> &to_remove) {
  fst::CacheOptions cache_opts(true, 1 << 25LL);
  fst::CacheOptions cache_opts_map(true, 0);
  fst::ArcMapFstOptions arcmap_opts(cache_opts);
  RemoveSomeInputSymbolsMapper<Arc, I> mapper(to_remove);
  return new LookaheadFst<Arc, I>(ComposeFst<Arc>(ifst1, ifst2, cache_opts), mapper, arcmap_opts);
}

template<class Arc, class I>
void RemoveSomeInputSymbols(const std::vector<I> &to_remove,
                            MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  RemoveSomeInputSymbolsMapper<Arc, I> mapper(to_remove);
  Map(fst, mapper);
}

template<class Arc, class I>
class MapInputSymbolsMapper {
 public:
  Arc operator ()(const Arc &arc_in) {
    Arc ans = arc_in;
    if (ans.ilabel > 0 &&
       ans.ilabel < static_cast<typename Arc::Label>((*symbol_mapping_).size()))
      ans.ilabel = (*symbol_mapping_)[ans.ilabel];
    return ans;
  }
  MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }
  MapSymbolsAction InputSymbolsAction() const { return MAP_CLEAR_SYMBOLS; }
  MapSymbolsAction OutputSymbolsAction() const { return MAP_COPY_SYMBOLS; }
  uint64 Properties(uint64 props) const {  // Not tested.
    bool remove_epsilons = (symbol_mapping_->size() > 0 && (*symbol_mapping_)[0] != 0);
    bool add_epsilons = (symbol_mapping_->size() > 1 &&
                         *std::min_element(symbol_mapping_->begin()+1, symbol_mapping_->end()) == 0);

    // remove the following as we don't know now if any of them are true.
    uint64 props_to_remove = kAcceptor|kNotAcceptor|kIDeterministic|kNonIDeterministic|
        kILabelSorted|kNotILabelSorted;
    if (remove_epsilons) props_to_remove |= kEpsilons|kIEpsilons;
    if (add_epsilons) props_to_remove |=  kNoEpsilons|kNoIEpsilons;
    uint64 props_to_add = 0;
    if (remove_epsilons && !add_epsilons) props_to_add |= kNoEpsilons|kNoIEpsilons;
    return (props & ~props_to_remove) | props_to_add;
  }
  // initialize with copy = false only if the "to_remove" argument will not be deleted
  // in the lifetime of this object.
  MapInputSymbolsMapper(const std::vector<I> &to_remove, bool copy) {
    KALDI_ASSERT_IS_INTEGER_TYPE(I);
    if (copy) symbol_mapping_ = new std::vector<I> (to_remove);
    else symbol_mapping_ = &to_remove;
    owned = copy;
  }
  ~MapInputSymbolsMapper() { if (owned && symbol_mapping_ != NULL) delete symbol_mapping_; }
 private:
  bool owned;
  const std::vector<I> *symbol_mapping_;
};

template<class Arc, class I>
void MapInputSymbols(const std::vector<I> &symbol_mapping,
                     MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  // false == don't copy the "symbol_mapping", retain pointer--
  // safe since short-lived object.
  MapInputSymbolsMapper<Arc, I> mapper(symbol_mapping, false);
  Map(fst, mapper);
}

template<class Arc, class I>
bool GetLinearSymbolSequence(const Fst<Arc> &fst,
                             std::vector<I> *isymbols_out,
                             std::vector<I> *osymbols_out,
                             typename Arc::Weight *tot_weight_out) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  Weight tot_weight = Weight::One();
  std::vector<I> ilabel_seq;
  std::vector<I> olabel_seq;

  StateId cur_state = fst.Start();
  if (cur_state == kNoStateId) {  // empty sequence.
    if (isymbols_out != NULL) isymbols_out->clear();
    if (osymbols_out != NULL) osymbols_out->clear();
    if (tot_weight_out != NULL) *tot_weight_out = Weight::Zero();
    return true;
  }
  while (1) {
    Weight w = fst.Final(cur_state);
    if (w != Weight::Zero()) {  // is final..
      tot_weight = Times(tot_weight, w);
      if (fst.NumArcs(cur_state) != 0) return false;
      if (isymbols_out != NULL) *isymbols_out = ilabel_seq;
      if (osymbols_out != NULL) *osymbols_out = olabel_seq;
      if (tot_weight_out != NULL) *tot_weight_out = tot_weight;
      return true;
    } else {
      if (fst.NumArcs(cur_state) != 1) return false;

      ArcIterator<Fst<Arc> > iter(fst, cur_state);  // get the only arc.
      const Arc &arc = iter.Value();
      tot_weight = Times(tot_weight, arc.weight);
      if (arc.ilabel != 0) ilabel_seq.push_back(arc.ilabel);
      if (arc.olabel != 0) olabel_seq.push_back(arc.olabel);
      cur_state = arc.nextstate;
    }
  }
}


// see fstext-utils.h for comment.
template<class Arc>
void ConvertNbestToVector(const Fst<Arc> &fst,
                          std::vector<VectorFst<Arc> > *fsts_out) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  fsts_out->clear();
  StateId start_state = fst.Start();
  if (start_state == kNoStateId) return; // No output.
  size_t n_arcs = fst.NumArcs(start_state);
  bool start_is_final = (fst.Final(start_state) != Weight::Zero());
  fsts_out->reserve(n_arcs + (start_is_final ? 1 : 0));

  if (start_is_final) {
    fsts_out->resize(fsts_out->size() + 1);
    StateId start_state_out = fsts_out->back().AddState();
    fsts_out->back().SetFinal(start_state_out, fst.Final(start_state));
  }

  for (ArcIterator<Fst<Arc> > start_aiter(fst, start_state);
       !start_aiter.Done();
       start_aiter.Next()) {
    fsts_out->resize(fsts_out->size() + 1);
    VectorFst<Arc> &ofst = fsts_out->back();
    const Arc &first_arc = start_aiter.Value();
    StateId cur_state = start_state,
        cur_ostate = ofst.AddState();
    ofst.SetStart(cur_ostate);
    StateId next_ostate = ofst.AddState();
    ofst.AddArc(cur_ostate, Arc(first_arc.ilabel, first_arc.olabel,
                                first_arc.weight, next_ostate));
    cur_state = first_arc.nextstate;
    cur_ostate = next_ostate;
    while (1) {
      size_t this_n_arcs = fst.NumArcs(cur_state);
      KALDI_ASSERT(this_n_arcs <= 1); // or it violates our assumptions
                                      // about the input.
      if (this_n_arcs == 1) {
        KALDI_ASSERT(fst.Final(cur_state) == Weight::Zero());
        // or problem with ShortestPath.
        ArcIterator<Fst<Arc> > aiter(fst, cur_state);
        const Arc &arc = aiter.Value();
        next_ostate = ofst.AddState();
        ofst.AddArc(cur_ostate, Arc(arc.ilabel, arc.olabel,
                                    arc.weight, next_ostate));
        cur_state = arc.nextstate;
        cur_ostate = next_ostate;
      } else {
        KALDI_ASSERT(fst.Final(cur_state) != Weight::Zero());
        // or problem with ShortestPath.
        ofst.SetFinal(cur_ostate, fst.Final(cur_state));
        break;
      }
    }
  }
}


// see fstext-utils.sh for comment.
template<class Arc>
void NbestAsFsts(const Fst<Arc> &fst,
                 size_t n,
                 std::vector<VectorFst<Arc> > *fsts_out) {
  KALDI_ASSERT(n > 0);
  KALDI_ASSERT(fsts_out != NULL);
  VectorFst<Arc> nbest_fst;
  ShortestPath(fst, &nbest_fst, n);
  ConvertNbestToVector(nbest_fst, fsts_out);
}

template<class Arc, class I>
void MakeLinearAcceptorWithAlternatives(const std::vector<std::vector<I> > &labels,
                                        MutableFst<Arc> *ofst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);
  for (size_t i = 0; i < labels.size(); i++) {
    KALDI_ASSERT(labels[i].size() != 0);
    StateId next_state = ofst->AddState();
    for (size_t j = 0; j < labels[i].size(); j++) {
      Arc arc(labels[i][j], labels[i][j], Weight::One(), next_state);
      ofst->AddArc(cur_state, arc);
    }
    cur_state = next_state;
  }
  ofst->SetFinal(cur_state, Weight::One());
}

template<class Arc, class I>
void MakeLinearAcceptor(const std::vector<I> &labels, MutableFst<Arc> *ofst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);
  for (size_t i = 0; i < labels.size(); i++) {
    StateId next_state = ofst->AddState();
    Arc arc(labels[i], labels[i], Weight::One(), next_state);
    ofst->AddArc(cur_state, arc);
    cur_state = next_state;
  }
  ofst->SetFinal(cur_state, Weight::One());
}


template<class I>
void GetSymbols(const SymbolTable &symtab,
                bool include_eps,
                std::vector<I> *syms_out) {
  KALDI_ASSERT(syms_out != NULL);
  syms_out->clear();
  for (SymbolTableIterator iter(symtab);
      !iter.Done();
      iter.Next()) {
    if (include_eps || iter.Value() != 0) {
      syms_out->push_back(iter.Value());
      KALDI_ASSERT(syms_out->back() == iter.Value());  // an integer-range thing.
    }
  }
}

template<class Arc>
void SafeDeterminizeWrapper(MutableFst<Arc> *ifst, MutableFst<Arc> *ofst, float delta) {
  typename Arc::Label highest_sym = HighestNumberedInputSymbol(*ifst);
  std::vector<typename Arc::Label> extra_syms;
  PreDeterminize(ifst,
                 (typename Arc::Label)(highest_sym+1),
                 &extra_syms);
  DeterminizeStar(*ifst, ofst, delta);
  RemoveSomeInputSymbols(extra_syms, ofst);  // remove the extra symbols.
}


template<class Arc>
void SafeDeterminizeMinimizeWrapper(MutableFst<Arc> *ifst, VectorFst<Arc> *ofst, float delta) {
  typename Arc::Label highest_sym = HighestNumberedInputSymbol(*ifst);
  std::vector<typename Arc::Label> extra_syms;
  PreDeterminize(ifst,
                 (typename Arc::Label)(highest_sym+1),
                 &extra_syms);
  DeterminizeStar(*ifst, ofst, delta);
  RemoveSomeInputSymbols(extra_syms, ofst);  // remove the extra symbols.
  RemoveEpsLocal(ofst);  // this is "safe" and will never hurt.
  MinimizeEncoded(ofst, delta);
}


inline
void DeterminizeStarInLog(VectorFst<StdArc> *fst, float delta, bool *debug_ptr, int max_states) {
  // DeterminizeStarInLog determinizes 'fst' in the log semiring, using
  // the DeterminizeStar algorithm (which also removes epsilons).

  ArcSort(fst, ILabelCompare<StdArc>());  // helps DeterminizeStar to be faster.
  VectorFst<LogArc> *fst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*fst, fst_log);
  VectorFst<StdArc> tmp;
  *fst = tmp;  // make fst empty to free up memory. [actually may make no difference..]
  VectorFst<LogArc> *fst_det_log = new VectorFst<LogArc>;
  DeterminizeStar(*fst_log, fst_det_log, delta, debug_ptr, max_states);
  Cast(*fst_det_log, fst);
  delete fst_log;
  delete fst_det_log;
}

inline
void DeterminizeInLog(VectorFst<StdArc> *fst) {
  // DeterminizeInLog determinizes 'fst' in the log semiring.

  ArcSort(fst, ILabelCompare<StdArc>());  // helps DeterminizeStar to be faster.
  VectorFst<LogArc> *fst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*fst, fst_log);
  VectorFst<StdArc> tmp;
  *fst = tmp;  // make fst empty to free up memory. [actually may make no difference..]
  VectorFst<LogArc> *fst_det_log = new VectorFst<LogArc>;
  Determinize(*fst_log, fst_det_log);
  Cast(*fst_det_log, fst);
  delete fst_log;
  delete fst_det_log;
}



// make it inline to avoid having to put it in a .cc file.
// destructive algorithm (changes ifst as well as ofst).
inline
void SafeDeterminizeMinimizeWrapperInLog(VectorFst<StdArc> *ifst, VectorFst<StdArc> *ofst, float delta) {
  VectorFst<LogArc> *ifst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*ifst, ifst_log);
  VectorFst<LogArc> *ofst_log = new VectorFst<LogArc>;
  SafeDeterminizeWrapper(ifst_log, ofst_log, delta);
  Cast(*ofst_log, ofst);
  delete ifst_log;
  delete ofst_log;
  RemoveEpsLocal(ofst);  // this is "safe" and will never hurt.  Do this in tropical, which is important.
  MinimizeEncoded(ofst, delta);  // Non-deterministic minimization will fail in log semiring so do it with StdARc.
}

inline
void SafeDeterminizeWrapperInLog(VectorFst<StdArc> *ifst, VectorFst<StdArc> *ofst, float delta) {
  VectorFst<LogArc> *ifst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*ifst, ifst_log);
  VectorFst<LogArc> *ofst_log = new VectorFst<LogArc>;
  SafeDeterminizeWrapper(ifst_log, ofst_log, delta);
  Cast(*ofst_log, ofst);
  delete ifst_log;
  delete ofst_log;
}



template<class Arc>
void RemoveWeights(MutableFst<Arc> *ifst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  for (StateIterator<MutableFst<Arc> > siter(*ifst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> >  aiter(ifst, s); !aiter.Done(); aiter.Next()) {
      Arc arc(aiter.Value());
      arc.weight = Weight::One();
      aiter.SetValue(arc);
    }
    if (ifst->Final(s) != Weight::Zero())
      ifst->SetFinal(s, Weight::One());
  }
  ifst->SetProperties(kUnweighted, kUnweighted);
}

// Used in PrecedingInputSymbolsAreSame (non-functor version), and
// similar routines.
template<class T> struct IdentityFunction {
  typedef T Arg;
  typedef T Result;
  T operator () (const T &t) const { return t; }
};

template<class Arc>
bool PrecedingInputSymbolsAreSame(bool start_is_epsilon, const Fst<Arc> &fst) {
  IdentityFunction<typename Arc::Label> f;
  return PrecedingInputSymbolsAreSameClass(start_is_epsilon, fst, f);
}

template<class Arc, class F> // F is functor type from labels to classes.
bool PrecedingInputSymbolsAreSameClass(bool start_is_epsilon, const Fst<Arc> &fst, const F &f) {
  typedef typename F::Result ClassType;
  typedef typename Arc::StateId StateId;
  std::vector<ClassType> classes;
  ClassType noClass = f(kNoLabel);

  if (start_is_epsilon) {
    StateId start_state = fst.Start();
    if (start_state < 0 || start_state == kNoStateId)
      return true;  // empty fst-- doesn't matter.
    classes.resize(start_state+1, noClass);
    classes[start_state] = 0;
  }

  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (classes.size() <= arc.nextstate)
        classes.resize(arc.nextstate+1, noClass);
      if (classes[arc.nextstate] == noClass)
        classes[arc.nextstate] = f(arc.ilabel);
      else
        if (classes[arc.nextstate] != f(arc.ilabel))
          return false;
    }
  }
  return true;
}

template<class Arc>
bool FollowingInputSymbolsAreSame(bool end_is_epsilon, const Fst<Arc> &fst) {
  IdentityFunction<typename Arc::Label> f;
  return FollowingInputSymbolsAreSameClass(end_is_epsilon, fst, f);
}


template<class Arc, class F>
bool FollowingInputSymbolsAreSameClass(bool end_is_epsilon, const Fst<Arc> &fst, const F &f) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename F::Result ClassType;
  const ClassType noClass = f(kNoLabel), epsClass = f(0);
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    ClassType c = noClass;
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (c == noClass)
        c = f(arc.ilabel);
      else
        if (c != f(arc.ilabel))
          return false;
    }
    if (end_is_epsilon && c != noClass &&
       c != epsClass && fst.Final(s) != Weight::Zero())
      return false;
  }
  return true;
}

template<class Arc>
void MakePrecedingInputSymbolsSame(bool start_is_epsilon, MutableFst<Arc> *fst) {
  IdentityFunction<typename Arc::Label> f;
  MakePrecedingInputSymbolsSameClass(start_is_epsilon, fst, f);
}

template<class Arc, class F>
void MakePrecedingInputSymbolsSameClass(bool start_is_epsilon, MutableFst<Arc> *fst, const F &f) {
  typedef typename F::Result ClassType;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  std::vector<ClassType> classes;
  ClassType noClass = f(kNoLabel);
  ClassType epsClass = f(0);
  if (start_is_epsilon) {  // treat having-start-state as epsilon in-transition.
    StateId start_state = fst->Start();
    if (start_state < 0 || start_state == kNoStateId) // empty FST.
      return;
    classes.resize(start_state+1, noClass);
    classes[start_state] = epsClass;
  }

  // Find bad states (states with multiple input-symbols into them).
  std::set<StateId> bad_states;  // states that we need to change.
  for (StateIterator<Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (classes.size() <= static_cast<size_t>(arc.nextstate))
        classes.resize(arc.nextstate+1, noClass);
      if (classes[arc.nextstate] == noClass)
        classes[arc.nextstate] = f(arc.ilabel);
      else
        if (classes[arc.nextstate] != f(arc.ilabel))
          bad_states.insert(arc.nextstate);
    }
  }
  if (bad_states.empty()) return;  // Nothing to do.
  kaldi::ConstIntegerSet<StateId> bad_states_ciset(bad_states);  // faster lookup.

  // Work out list of arcs we have to change as (state, arc-offset).
  // Can't do the actual changes in this pass, since we have to add new
  // states which invalidates the iterators.
  std::vector<std::pair<StateId, size_t> > arcs_to_change;
  for (StateIterator<Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (ArcIterator<Fst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0 &&
         bad_states_ciset.count(arc.nextstate) != 0)
        arcs_to_change.push_back(std::make_pair(s, aiter.Position()));
    }
  }
  KALDI_ASSERT(!arcs_to_change.empty());  // since !bad_states.empty().

  std::map<std::pair<StateId, ClassType>, StateId> state_map;
  // state_map is a map from (bad-state, input-symbol-class) to dummy-state.

  for (size_t i = 0; i < arcs_to_change.size(); i++) {
    StateId s = arcs_to_change[i].first;
    ArcIterator<MutableFst<Arc> > aiter(*fst, s);
    aiter.Seek(arcs_to_change[i].second);
    Arc arc = aiter.Value();

    // Transition is non-eps transition to "bad" state.  Introduce new state (or find
    // existing one).
    std::pair<StateId, ClassType> p(arc.nextstate, f(arc.ilabel));
    if (state_map.count(p) == 0) {
      StateId newstate = state_map[p] = fst->AddState();
      fst->AddArc(newstate, Arc(0, 0, Weight::One(), arc.nextstate));
    }
    StateId dst_state = state_map[p];
    arc.nextstate = dst_state;

    // Initialize the MutableArcIterator only now, as the call to NewState()
    // may have invalidated the first arc iterator.
    MutableArcIterator<MutableFst<Arc> > maiter(fst, s);
    maiter.Seek(arcs_to_change[i].second);
    maiter.SetValue(arc);
  }
}

template<class Arc>
void MakeFollowingInputSymbolsSame(bool end_is_epsilon, MutableFst<Arc> *fst) {
  IdentityFunction<typename Arc::Label> f;
  MakeFollowingInputSymbolsSameClass(end_is_epsilon, fst, f);
}

template<class Arc, class F>
void MakeFollowingInputSymbolsSameClass(bool end_is_epsilon, MutableFst<Arc> *fst, const F &f) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename F::Result ClassType;
  std::vector<StateId> bad_states;
  ClassType noClass = f(kNoLabel);
  ClassType epsClass = f(0);
  for (StateIterator<Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    ClassType c = noClass;
    bool bad = false;
    for (ArcIterator<Fst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (c == noClass)
        c = f(arc.ilabel);
      else
        if (c != f(arc.ilabel)) {
          bad = true;
          break;
        }
    }
    if (end_is_epsilon && c != noClass &&
       c != epsClass && fst->Final(s) != Weight::Zero())
      bad = true;
    if (bad)
      bad_states.push_back(s);
  }
  std::vector<Arc> my_arcs;
  for (size_t i = 0; i < bad_states.size(); i++) {
    StateId s = bad_states[i];
    my_arcs.clear();
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next())
      my_arcs.push_back(aiter.Value());

    for (size_t j = 0; j < my_arcs.size(); j++) {
      Arc &arc = my_arcs[j];
      if (arc.ilabel != 0) {
        StateId newstate = fst->AddState();
        // Create a new state for each non-eps arc in original FST, out of each bad state.
        // Not as optimal as it could be, but does avoid some complicated weight-pushing
        // issues in which, to maintain stochasticity, we would have to know which semiring
        // we want to maintain stochasticity in.
        fst->AddArc(newstate, Arc(arc.ilabel, 0, Weight::One(), arc.nextstate));
        MutableArcIterator<MutableFst<Arc> > maiter(fst, s);
        maiter.Seek(j);
        maiter.SetValue(Arc(0, arc.olabel, arc.weight, newstate));
      }
    }
  }
}


template<class Arc>
VectorFst<Arc>* MakeLoopFst(const std::vector<const ExpandedFst<Arc> *> &fsts) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  VectorFst<Arc> *ans = new VectorFst<Arc>;
  StateId loop_state = ans->AddState();  // = 0.
  ans->SetStart(loop_state);
  ans->SetFinal(loop_state, Weight::One());

  // "cache" is used as an optimization when some of the pointers in "fsts"
  // may have the same value.
  unordered_map<const ExpandedFst<Arc> *, Arc> cache;

  for (Label i = 0; i < static_cast<Label>(fsts.size()); i++) {
    const ExpandedFst<Arc> *fst = fsts[i];
    if (fst == NULL) continue;
    { // optimization with cache: helpful if some members of "fsts" may
      // contain the same pointer value (e.g. in GetHTransducer).
      typename unordered_map<const ExpandedFst<Arc> *, Arc>::iterator
          iter = cache.find(fst);
      if (iter != cache.end()) {
        Arc arc = iter->second;
        arc.olabel = i;
        ans->AddArc(0, arc);
        continue;
      }
    }

    KALDI_ASSERT(fst->Properties(kAcceptor, true) == kAcceptor);  // expect acceptor.

    StateId fst_num_states = fst->NumStates();
    StateId fst_start_state = fst->Start();

    if (fst_start_state == kNoStateId)
      continue;  // empty fst.

    bool share_start_state =
        fst->Properties(kInitialAcyclic, true) == kInitialAcyclic
        && fst->NumArcs(fst_start_state) == 1
        && fst->Final(fst_start_state) == Weight::Zero();

    std::vector<StateId> state_map(fst_num_states);  // fst state -> ans state
    for (StateId s = 0; s < fst_num_states; s++) {
      if (s == fst_start_state && share_start_state) state_map[s] = loop_state;
      else state_map[s] = ans->AddState();
    }
    if (!share_start_state) {
      Arc arc(0, i, Weight::One(), state_map[fst_start_state]);
      cache[fst] = arc;
      ans->AddArc(0, arc);
    }
    for (StateId s = 0; s < fst_num_states; s++) {
      // Add arcs out of state s.
      for (ArcIterator<ExpandedFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        Label olabel = (s == fst_start_state && share_start_state ? i : 0);
        Arc newarc(arc.ilabel, olabel, arc.weight, state_map[arc.nextstate]);
        ans->AddArc(state_map[s], newarc);
        if (s == fst_start_state && share_start_state)
          cache[fst] = newarc;
      }
      if (fst->Final(s) != Weight::Zero()) {
        KALDI_ASSERT(!(s == fst_start_state && share_start_state));
        ans->AddArc(state_map[s], Arc(0, 0, fst->Final(s), loop_state));
      }
    }
  }
  return ans;
}


template<class Arc>
void ClearSymbols(bool clear_input,
                  bool clear_output,
                  MutableFst<Arc> *fst) {
  for (StateIterator<MutableFst<Arc> > siter(*fst);
       !siter.Done();
       siter.Next()) {
    typename Arc::StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      bool change = false;
      if (clear_input && arc.ilabel != 0) {
        arc.ilabel = 0;
        change = true;
      }
      if (clear_output && arc.olabel != 0) {
        arc.olabel = 0;
        change = true;
      }
      if (change) {
        aiter.SetValue(arc);
      }
    }
  }
}


template<class Arc>
void ApplyProbabilityScale(float scale, MutableFst<Arc> *fst) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  for (StateIterator<MutableFst<Arc> > siter(*fst);
       !siter.Done();
       siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
        !aiter.Done();
        aiter.Next()) {
      Arc arc = aiter.Value();
      arc.weight = Weight(arc.weight.Value() * scale);
      aiter.SetValue(arc);
    }
    if (fst->Final(s) != Weight::Zero())
      fst->SetFinal(s, Weight(fst->Final(s).Value() * scale));
  }
}


// return arc-offset of self-loop with ilabel (or -1 if none exists).
// if more than one such self-loop, pick first one.
template<class Arc>
ssize_t FindSelfLoopWithILabel(const Fst<Arc> &fst, typename Arc::StateId s) {
  for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next())
    if (aiter.Value().nextstate == s
       && aiter.Value().ilabel != 0) return static_cast<ssize_t>(aiter.Position());
  return static_cast<ssize_t>(-1);
}


template<class Arc>
bool EqualAlign(const Fst<Arc> &ifst,
                typename Arc::StateId length,
                int rand_seed,
                MutableFst<Arc> *ofst,
                int num_retries) {
  srand(rand_seed);
  KALDI_ASSERT(ofst->NumStates() == 0);  // make sure ofst empty.
  // make sure all states can reach final-state (or this algorithm may enter
  // infinite loop.
  KALDI_ASSERT(ifst.Properties(kCoAccessible, true) == kCoAccessible);

  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  if (ifst.Start() == kNoStateId) {
    KALDI_WARN << "Empty input fst.";
    return false;
  }
  // First select path through ifst.
  std::vector<StateId> path;
  std::vector<size_t> arc_offsets;  // arc taken out of each state.
  std::vector<int> nof_ilabels;

  StateId num_ilabels = 0;
  int retry_no = 0;

  // Under normal circumstances, this will be one-pass-only process
  // Multiple tries might be needed in special cases, typically when
  // the number of frames is close to number of transitions from
  // the start node to the final node. It usually happens for really
  // short utterances
  do {
    num_ilabels = 0;
    arc_offsets.clear();
    path.clear();
    path.push_back(ifst.Start());

    while (1) {
      // Select either an arc or final-prob.
      StateId s = path.back();
      size_t num_arcs = ifst.NumArcs(s);
      size_t num_arcs_tot = num_arcs;
      if (ifst.Final(s) != Weight::Zero()) num_arcs_tot++;
      // kaldi::RandInt is a bit like Rand(), but gets around situations
      // where RAND_MAX is very small.
      // Change this to Rand() % num_arcs_tot if compile issues arise
      size_t arc_offset = static_cast<size_t>(kaldi::RandInt(0, num_arcs_tot-1));

      if (arc_offset < num_arcs) {  // an actual arc.
        ArcIterator<Fst<Arc> > aiter(ifst, s);
        aiter.Seek(arc_offset);
        const Arc &arc = aiter.Value();
        if (arc.nextstate == s) {
          continue;  // don't take this self-loop arc
        } else {
          arc_offsets.push_back(arc_offset);
          path.push_back(arc.nextstate);
          if (arc.ilabel != 0) num_ilabels++;
        }
      } else {
        break;  // Chose final-prob.
      }
    }

    nof_ilabels.push_back(num_ilabels);
  } while (( ++retry_no < num_retries) && (num_ilabels > length));

  if (num_ilabels > length) {
    std::stringstream ilabel_vec;
    std::copy(nof_ilabels.begin(), nof_ilabels.end(),
          std::ostream_iterator<int>(ilabel_vec, ","));
    std::string s = ilabel_vec.str();
    s.erase(s.end() - 1);
    KALDI_WARN << "EqualAlign: the randomly constructed paths lengths: " << s;
    KALDI_WARN << "EqualAlign: utterance has too few frames " << length
               << " to align.";
    return false;  // can't make it shorter by adding self-loops!.
  }

  StateId num_self_loops = 0;
  std::vector<ssize_t> self_loop_offsets(path.size());
  for (size_t i = 0; i < path.size(); i++)
    if ( (self_loop_offsets[i] = FindSelfLoopWithILabel(ifst, path[i]))
         != static_cast<ssize_t>(-1) )
      num_self_loops++;

  if (num_self_loops == 0
      && num_ilabels < length) {
    KALDI_WARN << "No self-loops on chosen path; cannot match length.";
    return false;  // no self-loops to make it longer.
  }

  StateId num_extra = length - num_ilabels;  // Number of self-loops we need.

  StateId min_num_loops = 0;
  if (num_extra != 0) min_num_loops = num_extra / num_self_loops;  // prevent div by zero.
  StateId num_with_one_more_loop = num_extra - (min_num_loops*num_self_loops);
  KALDI_ASSERT(num_with_one_more_loop < num_self_loops || num_self_loops == 0);

  ofst->AddState();
  ofst->SetStart(0);
  StateId cur_state = 0;
  StateId counter = 0;  // tell us when we should stop adding one more loop.
  for (size_t i = 0; i < path.size(); i++) {
    // First, add any self-loops that are necessary.
    StateId num_loops = 0;
    if (self_loop_offsets[i] != static_cast<ssize_t>(-1)) {
      num_loops = min_num_loops + (counter < num_with_one_more_loop ? 1 : 0);
      counter++;
    }
    for (StateId j = 0; j < num_loops; j++) {
      ArcIterator<Fst<Arc> > aiter(ifst, path[i]);
      aiter.Seek(self_loop_offsets[i]);
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.nextstate == path[i]
             && arc.ilabel != 0);  // make sure self-loop with ilabel.
      StateId next_state = ofst->AddState();
      ofst->AddArc(cur_state, Arc(arc.ilabel, arc.olabel, arc.weight, next_state));
      cur_state = next_state;
    }
    if (i+1 < path.size()) {  // add forward transition.
      ArcIterator<Fst<Arc> > aiter(ifst, path[i]);
      aiter.Seek(arc_offsets[i]);
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.nextstate == path[i+1]);
      StateId next_state = ofst->AddState();
      ofst->AddArc(cur_state, Arc(arc.ilabel, arc.olabel, arc.weight, next_state));
      cur_state = next_state;
    } else {  // add final-prob.
      Weight weight = ifst.Final(path[i]);
      KALDI_ASSERT(weight != Weight::Zero());
      ofst->SetFinal(cur_state, weight);
    }
  }
  return true;
}


// This function identifies two types of useless arcs:
// those where arc A and arc B both go from state X to
// state Y with the same input symbol (remove the one
// with smaller probability, or an arbitrary one if they
// are the same); and those where A is an arc from state X
// to state X, with epsilon input symbol [remove A].
// Only works for tropical (not log) semiring as it uses
// NaturalLess.
template<class Arc>
void RemoveUselessArcs(MutableFst<Arc> *fst) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  NaturalLess<Weight> nl;
  StateId non_coacc_state = kNoStateId;
  size_t num_arcs_removed = 0, tot_arcs = 0;
  for (StateIterator<MutableFst<Arc> > siter(*fst);
      !siter.Done();
      siter.Next()) {
    std::vector<size_t> arcs_to_delete;
    std::vector<Arc> arcs;
    // pair2arclist lets us look up the arcs
    std::map<std::pair<Label, StateId>, std::vector<size_t> > pair2arclist;
    StateId state = siter.Value();
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, state);
        !aiter.Done();
        aiter.Next()) {
      size_t pos = arcs.size();
      const Arc &arc = aiter.Value();
      arcs.push_back(arc);
      pair2arclist[std::make_pair(arc.ilabel, arc.nextstate)].push_back(pos);
    }
    typename std::map<std::pair<Label, StateId>, std::vector<size_t> >::iterator
        iter = pair2arclist.begin(), end = pair2arclist.end();
    for (; iter!= end; ++iter) {
      const std::vector<size_t> &poslist = iter->second;
      if (poslist.size() > 1) {  // >1 arc with same ilabel, dest-state
        size_t best_pos = poslist[0];
        Weight best_weight = arcs[best_pos].weight;
        for (size_t j = 1; j < poslist.size(); j++) {
          size_t pos = poslist[j];
          Weight this_weight = arcs[pos].weight;
          if (nl(this_weight, best_weight)) {  // NaturalLess seems to be somehow
            // "backwards".
            best_weight = this_weight;  // found a better one.
            best_pos = pos;
          }
        }
        for (size_t j = 0; j < poslist.size(); j++)
          if (poslist[j] != best_pos)
            arcs_to_delete.push_back(poslist[j]);
      } else {
        KALDI_ASSERT(poslist.size() == 1);
        size_t pos = poslist[0];
        Arc &arc = arcs[pos];
        if (arc.ilabel == 0 && arc.nextstate == state)
          arcs_to_delete.push_back(pos);
      }
    }
    tot_arcs += arcs.size();
    if (arcs_to_delete.size() != 0) {
      num_arcs_removed += arcs_to_delete.size();
      if (non_coacc_state == kNoStateId)
        non_coacc_state = fst->AddState();
      MutableArcIterator<MutableFst<Arc> > maiter(fst, state);
      for (size_t j = 0; j < arcs_to_delete.size(); j++) {
        size_t pos = arcs_to_delete[j];
        maiter.Seek(pos);
        arcs[pos].nextstate = non_coacc_state;
        maiter.SetValue(arcs[pos]);
      }
    }
  }
  if (non_coacc_state != kNoStateId)
    Connect(fst);
  KALDI_VLOG(1) << "removed " << num_arcs_removed << " of " << tot_arcs
                << "arcs.";
}

template<class Arc>
void PhiCompose(const Fst<Arc> &fst1,
                const Fst<Arc> &fst2,
                typename Arc::Label phi_label,
                MutableFst<Arc> *ofst) {
  KALDI_ASSERT(phi_label != kNoLabel); // just use regular compose in this case.
  typedef Fst<Arc> F;
  typedef PhiMatcher<SortedMatcher<F> > PM;
  CacheOptions base_opts;
  base_opts.gc_limit = 0; // Cache only the last state for fastest copy.
  // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
  // The matcher for fst1 doesn't matter; we'll use fst2's matcher.
  ComposeFstImplOptions<SortedMatcher<F>, PM> impl_opts(base_opts);

  // the false below is something called phi_loop which is something I don't
  // fully understand, but I don't think we want it.

  // These pointers are taken ownership of, by ComposeFst.
  PM *phi_matcher =
      new PM(fst2, MATCH_INPUT, phi_label, false);
  SortedMatcher<F> *sorted_matcher =
      new SortedMatcher<F>(fst1, MATCH_NONE); // tell it
  // not to use this matcher, as this would mean we would
  // not follow phi transitions.
  impl_opts.matcher1 = sorted_matcher;
  impl_opts.matcher2 = phi_matcher;
  *ofst = ComposeFst<Arc>(fst1, fst2, impl_opts);
  Connect(ofst);
}

template<class Arc>
void PropagateFinalInternal(
    typename Arc::Label phi_label,
    typename Arc::StateId s,
    MutableFst<Arc> *fst) {
  typedef typename Arc::Weight Weight;
  if (fst->Final(s) == Weight::Zero()) {
    // search for phi transition.  We assume there
    // is just one-- phi nondeterminism is not allowed
    // anyway.
    int num_phis = 0;
    for (ArcIterator<Fst<Arc> > aiter(*fst, s);
         !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == phi_label) {
        num_phis++;
        if (arc.nextstate == s) continue; // don't expect
        // phi loops but ignore them anyway.

        // If this recurses infinitely, it means there
        // are loops of phi transitions, which there should
        // not be in a normal backoff LM.  We could make this
        // routine work for this case, but currently there is
        // no need.
        PropagateFinalInternal(phi_label, arc.nextstate, fst);
        if (fst->Final(arc.nextstate) != Weight::Zero())
          fst->SetFinal(s, Times(fst->Final(arc.nextstate), arc.weight));
      }
      KALDI_ASSERT(num_phis <= 1 && "Phi nondeterminism found");
    }
  }
}

template<class Arc>
void PropagateFinal(typename Arc::Label phi_label,
                    MutableFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  if (fst->Properties(kIEpsilons, true)) // just warn.
    KALDI_WARN << "PropagateFinal: this may not work as desired "
        "since your FST has input epsilons.";
  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++)
    PropagateFinalInternal(phi_label, s, fst);
}

template<class Arc>
void RhoCompose(const Fst<Arc> &fst1,
                const Fst<Arc> &fst2,
                typename Arc::Label rho_label,
                MutableFst<Arc> *ofst) {
  KALDI_ASSERT(rho_label != kNoLabel); // just use regular compose in this case.
  typedef Fst<Arc> F;
  typedef RhoMatcher<SortedMatcher<F> > RM;
  CacheOptions base_opts;
  base_opts.gc_limit = 0; // Cache only the last state for fastest copy.
  // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
  // The matcher for fst1 doesn't matter; we'll use fst2's matcher.
  ComposeFstImplOptions<SortedMatcher<F>, RM> impl_opts(base_opts);

  // the false below is something called rho_loop which is something I don't
  // fully understand, but I don't think we want it.

  // These pointers are taken ownership of, by ComposeFst.
  RM *rho_matcher =
      new RM(fst2, MATCH_INPUT, rho_label);
  SortedMatcher<F> *sorted_matcher =
      new SortedMatcher<F>(fst1, MATCH_NONE); // tell it
  // not to use this matcher, as this would mean we would
  // not follow rho transitions.
  impl_opts.matcher1 = sorted_matcher;
  impl_opts.matcher2 = rho_matcher;
  *ofst = ComposeFst<Arc>(fst1, fst2, impl_opts);
  Connect(ofst);
}


// Declare an override of the template below.
template<>
inline bool IsStochasticFst(const Fst<LogArc> &fst,
                            float delta,
                            LogArc::Weight *min_sum,
                            LogArc::Weight *max_sum);

// Will override this for LogArc where NaturalLess will not work.
template<class Arc>
inline bool IsStochasticFst(const Fst<Arc> &fst,
                     float delta,
                     typename Arc::Weight *min_sum,
                     typename Arc::Weight *max_sum) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  NaturalLess<Weight> nl;
  bool first_time = true;
  bool ans = true;
  if (min_sum) *min_sum = Arc::Weight::One();
  if (max_sum) *max_sum = Arc::Weight::One();
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight sum = fst.Final(s);
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      sum = Plus(sum, arc.weight);
    }
    if (!ApproxEqual(Weight::One(), sum, delta)) ans = false;
    if (first_time) {
      first_time = false;
      if (max_sum) *max_sum = sum;
      if (min_sum) *min_sum = sum;
    } else {
      if (max_sum && nl(*max_sum, sum)) *max_sum = sum;
      if (min_sum && nl(sum, *min_sum)) *min_sum = sum;
    }
  }
  if (first_time) {  // just avoid NaNs if FST was empty.
    if (max_sum) *max_sum = Weight::One();
    if (min_sum) *min_sum = Weight::One();
  }
  return ans;
}


// Overriding template for LogArc as NaturalLess does not work there.
template<>
inline bool IsStochasticFst(const Fst<LogArc> &fst,
                     float delta,
                     LogArc::Weight *min_sum,
                     LogArc::Weight *max_sum) {
  typedef LogArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  bool first_time = true;
  bool ans = true;
  if (min_sum) *min_sum = LogArc::Weight::One();
  if (max_sum) *max_sum = LogArc::Weight::One();
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight sum = fst.Final(s);
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      sum = Plus(sum, arc.weight);
    }
    if (!ApproxEqual(Weight::One(), sum, delta)) ans = false;
    if (first_time) {
      first_time = false;
      if (max_sum) *max_sum = sum;
      if (min_sum) *min_sum = sum;
    } else {
      // note that max and min are reversed from their normal
      // meanings here (max and min w.r.t. the underlying probabilities).
      if (max_sum && sum.Value() < max_sum->Value()) *max_sum = sum;
      if (min_sum && sum.Value() > min_sum->Value()) *min_sum = sum;
    }
  }
  if (first_time) {  // just avoid NaNs if FST was empty.
    if (max_sum) *max_sum = Weight::One();
    if (min_sum) *min_sum = Weight::One();
  }
  return ans;
}

// Tests whether a tropical FST is stochastic in the log
// semiring. (casts it and does the check.)
// This function deals with the generic fst.
// This version currently supports ConstFst<StdArc> or VectorFst<StdArc>.
// Otherwise, it will be died with an error.
inline bool IsStochasticFstInLog(const Fst<StdArc> &fst,
                          float delta,
                          StdArc::Weight *min_sum,
                          StdArc::Weight *max_sum) {
  bool ans = false;
  LogArc::Weight log_min = LogArc::Weight::One(),
    log_max = LogArc::Weight::Zero();
  if (fst.Type() == "const") {
    ConstFst<LogArc> logfst;
    Cast(dynamic_cast<const ConstFst<StdArc>&>(fst), &logfst);
    ans = IsStochasticFst(logfst, delta, &log_min, &log_max);
  } else if (fst.Type() == "vector") {
    VectorFst<LogArc> logfst;
    Cast(dynamic_cast<const VectorFst<StdArc>&>(fst), &logfst);
    ans = IsStochasticFst(logfst, delta, &log_min, &log_max);
  } else {
    KALDI_ERR << "This version currently supports ConstFst<StdArc> "
              << "or VectorFst<StdArc>";
  }
  if (min_sum) *min_sum = StdArc::Weight(log_min.Value());
  if (max_sum) *max_sum = StdArc::Weight(log_max.Value());
  return ans;
}

} // namespace fst.

#endif
