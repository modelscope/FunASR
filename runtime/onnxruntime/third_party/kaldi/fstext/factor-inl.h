// fstext/factor-inl.h

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

#ifndef KALDI_FSTEXT_FACTOR_INL_H_
#define KALDI_FSTEXT_FACTOR_INL_H_

#include "util/stl-utils.h"
// Do not include this file directly.  It is included by factor.h.

namespace fst {

// GetStateProperties takes in an FST and a number "max_state" which is the
// highest numbered state in the FST (this could be fst.NumStates()-1 for an
// ExpandedFst, or derived from some kind of traversal).  It outputs a vector
// numbered from 0..max_state, of type FstStateProperties which is a bitmask
// with information about the states.

// GetStateProperties has not been tested directly (only implicitly via
// testing Factor).
template<class Arc>
void GetStateProperties(const Fst<Arc> &fst,
                        typename Arc::StateId max_state,
                        std::vector<StatePropertiesType> *props) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  assert(props != NULL);
  props->clear();
  if (fst.Start() < 0) return;  // Empty fst.
  props->resize(max_state+1, 0);
  assert(fst.Start() <= max_state);
  (*props)[fst.Start()] |= kStateInitial;
  for (StateId s = 0; s <= max_state; s++) {
    StatePropertiesType &s_info = (*props)[s];
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) s_info |= kStateIlabelsOut;
      if (arc.olabel != 0) s_info |= kStateOlabelsOut;
      StateId nexts = arc.nextstate;
      assert(nexts <= max_state);  // or input was invalid.
      StatePropertiesType &nexts_info = (*props)[nexts];
      if (s_info&kStateArcsOut) s_info |= kStateMultipleArcsOut;
      s_info |= kStateArcsOut;
      if (nexts_info&kStateArcsIn) nexts_info |= kStateMultipleArcsIn;
      nexts_info |= kStateArcsIn;
    }
    if (fst.Final(s) != Weight::Zero())  s_info |= kStateFinal;
  }
}



template<class Arc, class I>
void Factor(const Fst<Arc> &fst, MutableFst<Arc> *ofst,
               std::vector<std::vector<I> > *symbols_out) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  assert(symbols_out != NULL);
  ofst->DeleteStates();
  if (fst.Start() < 0) return;  // empty FST.
  std::vector<StateId> order;
  DfsOrderVisitor<Arc> dfs_order_visitor(&order);
  DfsVisit(fst, &dfs_order_visitor);
  assert(order.size() > 0);
  StateId max_state = *(std::max_element(order.begin(), order.end()));
  std::vector<StatePropertiesType> state_properties;
  GetStateProperties(fst, max_state, &state_properties);

  std::vector<bool> remove(max_state+1);  // if true, will remove this state.

  // Now identify states that will be removed (made the middle of a chain).
  // The basic rule is that if the FstStateProperties equals
  // (kStateArcsIn|kStateArcsOut) or (kStateArcsIn|kStateArcsOut|kStateIlabelsOut),
  // then it is in the middle of a chain.  This eliminates state with
  // multiple input or output arcs, final states, and states with arcs out
  // that have olabels [we assume these are pushed to the left, so occur on the
  // 1st arc of a chain.

  for (StateId i = 0; i <= max_state; i++)
    remove[i] = (state_properties[i] == (kStateArcsIn|kStateArcsOut)
                 || state_properties[i] == (kStateArcsIn|kStateArcsOut|kStateIlabelsOut));
  std::vector<StateId> state_mapping(max_state+1, kNoStateId);

  typedef unordered_map<std::vector<I>, Label, kaldi::VectorHasher<I> > SymbolMapType;
  SymbolMapType symbol_mapping;
  Label symbol_counter = 0;
  {
    std::vector<I> eps;
    symbol_mapping[eps] = symbol_counter++;
  }
  std::vector<I> this_sym;  // a temporary used inside the loop.
  for (size_t i = 0; i < order.size(); i++) {
    StateId state = order[i];
    if (!remove[state]) {  // Process this state...
      StateId &new_state = state_mapping[state];
      if (new_state == kNoStateId) new_state = ofst->AddState();
      for (ArcIterator<Fst<Arc> > aiter(fst, state); !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel == 0) this_sym.clear();
        else {
          this_sym.resize(1);
          this_sym[0] = arc.ilabel;
        }
        while (remove[arc.nextstate]) {
          ArcIterator<Fst<Arc> > aiter2(fst, arc.nextstate);
          assert(!aiter2.Done());
          const Arc &nextarc = aiter2.Value();
          arc.weight = Times(arc.weight, nextarc.weight);
          assert(nextarc.olabel == 0);
          if (nextarc.ilabel != 0) this_sym.push_back(nextarc.ilabel);
          assert(static_cast<Label>(static_cast<I>(nextarc.ilabel))
                 == nextarc.ilabel); // check within integer range.
          arc.nextstate = nextarc.nextstate;
        }
        StateId &new_nextstate = state_mapping[arc.nextstate];
        if (new_nextstate == kNoStateId) new_nextstate = ofst->AddState();
        arc.nextstate = new_nextstate;
        if (symbol_mapping.count(this_sym) != 0) arc.ilabel = symbol_mapping[this_sym];
        else arc.ilabel = symbol_mapping[this_sym] = symbol_counter++;
        ofst->AddArc(new_state, arc);
      }
      if (fst.Final(state) != Weight::Zero())
        ofst->SetFinal(new_state, fst.Final(state));
    }
  }
  ofst->SetStart(state_mapping[fst.Start()]);

  // Now output the symbol sequences.
  symbols_out->resize(symbol_counter);
  for (typename SymbolMapType::const_iterator iter = symbol_mapping.begin();
      iter != symbol_mapping.end(); ++iter) {
    (*symbols_out)[iter->second] = iter->first;
  }
}

template<class Arc>
void Factor(const Fst<Arc> &fst, MutableFst<Arc> *ofst1,
            MutableFst<Arc> *ofst2) {
  typedef typename Arc::Label Label;
  std::vector<std::vector<Label> > symbols;
  Factor(fst, ofst2, &symbols);
  CreateFactorFst(symbols, ofst1);
}

template<class Arc, class I>
void ExpandInputSequences(const std::vector<std::vector<I> > &sequences,
                          MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  fst->SetInputSymbols(NULL);
  size_t size = sequences.size();
  if (sequences.size() > 0) assert(sequences[0].size() == 0);  // should be eps.
  StateId num_states_at_start = fst->NumStates();
  for (StateId s = 0; s < num_states_at_start; s++) {
    StateId num_arcs = fst->NumArcs(s);
    for (StateId aidx = 0; aidx < num_arcs; aidx++) {
      ArcIterator<MutableFst<Arc> > aiter(*fst, s);
      aiter.Seek(aidx);
      Arc arc = aiter.Value();

      Label ilabel = arc.ilabel;
      Label dest_state = arc.nextstate;
      if (ilabel != 0) {  // non-eps [nothing to do if eps]...
        assert(ilabel < static_cast<Label>(size));
        size_t len = sequences[ilabel].size();
        if (len <= 1) {
          if (len == 0) arc.ilabel = 0;
          else arc.ilabel = sequences[ilabel][0];
          MutableArcIterator<MutableFst<Arc> > mut_aiter(fst, s);
          mut_aiter.Seek(aidx);
          mut_aiter.SetValue(arc);
        } else {  // len>=2.  Must create new states...
          StateId curstate = -1;  // keep compiler happy: this value never used.
          for (size_t n = 0; n < len; n++) {  // adding/modifying "len" arcs.
            StateId nextstate;
            if (n < len-1) {
              nextstate = fst->AddState();
              assert(nextstate >= num_states_at_start);
            } else nextstate = dest_state;  // going back to original arc's
            // destination.
            if (n == 0) {
              arc.ilabel = sequences[ilabel][0];
              arc.nextstate = nextstate;
              MutableArcIterator<MutableFst<Arc> > mut_aiter(fst, s);
              mut_aiter.Seek(aidx);
              mut_aiter.SetValue(arc);
            } else {
              arc.ilabel = sequences[ilabel][n];
              arc.olabel = 0;
              arc.weight = Weight::One();
              arc.nextstate = nextstate;
              fst->AddArc(curstate, arc);
            }
            curstate = nextstate;
          }
        }
      }
    }
  }
}



template<class Arc, class I>
void CreateFactorFst(const std::vector<std::vector<I> > &sequences,
                     MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  assert(fst != NULL);
  fst->DeleteStates();
  StateId loopstate = fst->AddState();
  assert(loopstate == 0);
  fst->SetStart(0);
  fst->SetFinal(0, Weight::One());
  if (sequences.size() != 0) assert(sequences[0].size() == 0);  // can't replace epsilon...

  for (Label olabel = 1; olabel < static_cast<Label>(sequences.size()); olabel++) {
    size_t len = sequences[olabel].size();
    if (len == 0) {
      Arc arc(0, olabel, Weight::One(), loopstate);
      fst->AddArc(loopstate, arc);
    } else {
      StateId curstate = loopstate;
      for (size_t i = 0; i < len; i++) {
        StateId nextstate = (i == len-1 ? loopstate : fst->AddState());
        Arc arc(sequences[olabel][i], (i == 0 ? olabel : 0), Weight::One(), nextstate);
        fst->AddArc(curstate, arc);
        curstate = nextstate;
      }
    }
  }
  fst->SetProperties(kOLabelSorted, kOLabelSorted);
}


template<class Arc, class I>
void CreateMapFst(const std::vector<I> &symbol_map,
                  MutableFst<Arc> *fst) {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  assert(fst != NULL);
  fst->DeleteStates();
  StateId loopstate = fst->AddState();
  assert(loopstate == 0);
  fst->SetStart(0);
  fst->SetFinal(0, Weight::One());
  assert(symbol_map.empty() || symbol_map[0] == 0);  // FST cannot map epsilon to something else.
  for (Label olabel = 1; olabel < static_cast<Label>(symbol_map.size()); olabel++) {
    Arc arc(symbol_map[olabel], olabel, Weight::One(), loopstate);
    fst->AddArc(loopstate, arc);
  }
}




} // end namespace fst.

#endif
