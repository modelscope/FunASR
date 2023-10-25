// fstext/deterministic-fst-inl.h

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

#ifndef KALDI_FSTEXT_DETERMINISTIC_FST_INL_H_
#define KALDI_FSTEXT_DETERMINISTIC_FST_INL_H_
#include "base/kaldi-common.h"
#include "fstext/fstext-utils.h"


namespace fst {
// Do not include this file directly.  It is included by deterministic-fst.h.

template<class Arc>
typename Arc::StateId
BackoffDeterministicOnDemandFst<Arc>::GetBackoffState(StateId s,
                                                      Weight *w) {
  ArcIterator<Fst<Arc> > aiter(fst_, s);
  if (aiter.Done()) // no arcs.
    return kNoStateId;
  const Arc &arc = aiter.Value();
  if (arc.ilabel == 0) {
    *w = arc.weight;
    return arc.nextstate;
  } else {
    return kNoStateId;
  }
}

template<class Arc>
typename Arc::Weight BackoffDeterministicOnDemandFst<Arc>::Final(StateId state) {
  Weight w = fst_.Final(state);
  if (w != Weight::Zero()) return w;
  Weight backoff_w;
  StateId backoff_state = GetBackoffState(state, &backoff_w);
  if (backoff_state == kNoStateId) return Weight::Zero();
  else return Times(backoff_w, this->Final(backoff_state));
}

template<class Arc>
BackoffDeterministicOnDemandFst<Arc>::BackoffDeterministicOnDemandFst(
    const Fst<Arc> &fst): fst_(fst) {
#ifdef KALDI_PARANOID
  KALDI_ASSERT(fst_.Properties(kILabelSorted|kIDeterministic, true) ==
               (kILabelSorted|kIDeterministic) &&
               "Input FST is not i-label sorted and deterministic.");
#endif
}

template<class Arc>
bool BackoffDeterministicOnDemandFst<Arc>::GetArc(
    StateId s, Label ilabel, Arc *oarc) {
  KALDI_ASSERT(ilabel != 0); //  We don't allow GetArc for epsilon.

  SortedMatcher<Fst<Arc> > sm(fst_, MATCH_INPUT, 1);
  sm.SetState(s);
  if (sm.Find(ilabel)) {
    const Arc &arc = sm.Value();
    *oarc = arc;
    return true;
  } else {
    Weight backoff_w;
    StateId backoff_state = GetBackoffState(s, &backoff_w);
    if (backoff_state == kNoStateId) return false;
    if (!this->GetArc(backoff_state, ilabel, oarc)) return false;
    oarc->weight = Times(oarc->weight, backoff_w);
    return true;
  }
}

template<class Arc>
UnweightedNgramFst<Arc>::UnweightedNgramFst(int n): n_(n) {
  // Starting state is an empty vector
  std::vector<Label> start_state;
  state_vec_.push_back(start_state);
  start_state_ = 0;
  state_map_[start_state] = 0;
}

template<class Arc>
bool UnweightedNgramFst<Arc>::GetArc(
  StateId s, Label ilabel, Arc *oarc) {

  // The state ids increment with each state we encounter.
  // if the assert fails, then we are trying to access
  // unseen states that are not immediately traversable.
  KALDI_ASSERT(static_cast<size_t>(s) < state_vec_.size());
  std::vector<Label> seq = state_vec_[s];
  // Update state info.
  seq.push_back(ilabel);
  if (seq.size() > n_-1) {
    // Remove oldest word in the history.
    seq.erase(seq.begin());
  }
  std::pair<const std::vector<Label>, StateId> new_state(
    seq,
    static_cast<Label>(state_vec_.size()));
  // Now get state id for destination state.
  typedef typename MapType::iterator IterType;
  std::pair<IterType, bool> result = state_map_.insert(new_state);
  if (result.second == true) {
    state_vec_.push_back(seq);
  }
  oarc->weight = Weight::One(); // Because the FST is unweightd.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second; // The next state id.
  // All arcs can be matched.
  return true;
}

template<class Arc>
typename Arc::Weight UnweightedNgramFst<Arc>::Final(StateId state) {
  KALDI_ASSERT(state < static_cast<StateId>(state_vec_.size()));
  return Weight::One();
}

template<class Arc>
ComposeDeterministicOnDemandFst<Arc>::ComposeDeterministicOnDemandFst(
    DeterministicOnDemandFst<Arc> *fst1,
    DeterministicOnDemandFst<Arc> *fst2): fst1_(fst1), fst2_(fst2) {
  KALDI_ASSERT(fst1 != NULL && fst2 != NULL);
  if (fst1_->Start() == -1 || fst2_->Start() == -1) {
    start_state_ = -1;
    next_state_ = 0; // actually we don't care about this value.
  } else {
    start_state_ = 0;
    std::pair<StateId,StateId> start_pair(fst1_->Start(), fst2_->Start());
    state_map_[start_pair] = start_state_;
    state_vec_.push_back(start_pair);
    next_state_ = 1;
  }
}

template<class Arc>
typename Arc::Weight ComposeDeterministicOnDemandFst<Arc>::Final(StateId s) {
  KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
  const std::pair<StateId, StateId> &pr (state_vec_[s]);
  return Times(fst1_->Final(pr.first), fst2_->Final(pr.second));
}

template<class Arc>
bool ComposeDeterministicOnDemandFst<Arc>::GetArc(StateId s, Label ilabel,
                                                  Arc *oarc) {
  typedef typename MapType::iterator IterType;
  KALDI_ASSERT(ilabel != 0 &&
         "This program expects epsilon-free compact lattices as input");
  KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
  const std::pair<StateId, StateId> pr (state_vec_[s]);

  Arc arc1;
  if (!fst1_->GetArc(pr.first, ilabel, &arc1)) return false;
  if (arc1.olabel == 0) { // There is no output label on the
    // arc, so only the first state changes.
    std::pair<const std::pair<StateId, StateId>, StateId> new_value(
        std::pair<StateId, StateId>(arc1.nextstate, pr.second),
        next_state_);

    std::pair<IterType, bool> result = state_map_.insert(new_value);
    oarc->ilabel = ilabel;
    oarc->olabel = 0;
    oarc->nextstate = result.first->second;
    oarc->weight = arc1.weight;
    if (result.second == true) { // was inserted
      next_state_++;
      const std::pair<StateId, StateId> &new_pair (new_value.first);
      state_vec_.push_back(new_pair);
    }
    return true;
  }
  // There is an output label, so we need to traverse an arc on the
  // second fst also.
  Arc arc2;
  if (!fst2_->GetArc(pr.second, arc1.olabel, &arc2)) return false;
  std::pair<const std::pair<StateId, StateId>, StateId> new_value(
      std::pair<StateId, StateId>(arc1.nextstate, arc2.nextstate),
      next_state_);
  std::pair<IterType, bool> result =
      state_map_.insert(new_value);
  oarc->ilabel = ilabel;
  oarc->olabel = arc2.olabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Times(arc1.weight, arc2.weight);
  if (result.second == true) { // was inserted
    next_state_++;
    const std::pair<StateId, StateId> &new_pair (new_value.first);
    state_vec_.push_back(new_pair);
  }
  return true;
}

template<class Arc>
inline size_t CacheDeterministicOnDemandFst<Arc>::GetIndex(
    StateId src_state, Label ilabel) {
  const StateId p1 = 26597, p2 = 50329; // these are two
  // values that I drew at random from a table of primes.
  // note: num_cached_arcs_ > 0.

  // We cast to size_t before the modulus, to ensure the
  // result is positive.
  return static_cast<size_t>(src_state * p1 + ilabel * p2) %
      static_cast<size_t>(num_cached_arcs_);
}

template<class Arc>
CacheDeterministicOnDemandFst<Arc>::CacheDeterministicOnDemandFst(
    DeterministicOnDemandFst<Arc> *fst,
    StateId num_cached_arcs): fst_(fst),
                              num_cached_arcs_(num_cached_arcs),
                              cached_arcs_(num_cached_arcs) {
  KALDI_ASSERT(num_cached_arcs > 0);
  for (StateId i = 0; i < num_cached_arcs; i++)
    cached_arcs_[i].first = kNoStateId; // Invalidate all elements of the cache.
}

template<class Arc>
bool CacheDeterministicOnDemandFst<Arc>::GetArc(StateId s, Label ilabel,
                                                Arc *oarc) {
  // Note: we don't cache anything in case a requested arc does not exist.
  // In the uses that we imagine this will be put to, essentially all the
  // requested arcs will exist.  This only affects efficiency.
  KALDI_ASSERT(s >= 0 && ilabel != 0);
  size_t index = this->GetIndex(s, ilabel);
  if (cached_arcs_[index].first == s &&
      cached_arcs_[index].second.ilabel == ilabel) {
    *oarc = cached_arcs_[index].second;
    return true;
  } else {
    Arc arc;
    if (fst_->GetArc(s, ilabel, &arc)) {
      cached_arcs_[index].first = s;
      cached_arcs_[index].second = arc;
      *oarc = arc;
      return true;
    } else {
      return false;
    }
  }
}

template<class Arc>
LmExampleDeterministicOnDemandFst<Arc>::LmExampleDeterministicOnDemandFst(
    void *lm, Label bos_symbol, Label eos_symbol):
    lm_(lm), bos_symbol_(bos_symbol), eos_symbol_(eos_symbol) {
  std::vector<Label> begin_state; // history state corresponding to beginning of sentence
  begin_state.push_back(bos_symbol); // Depending how your LM is set up, you might
  // want to have a history vector with more than one bos_symbol on it.

  state_vec_.push_back(begin_state);
  start_state_ = 0;
  state_map_[begin_state] = 0;
}

template<class Arc>
typename Arc::Weight LmExampleDeterministicOnDemandFst<Arc>::Final(StateId s) {
  KALDI_ASSERT(static_cast<size_t>(s) < state_vec_.size());
  // In a real version you would probably use the following variable somehow
  // (commenting it because it's generating warnings).
  // const std::vector<Label> &wseq = state_vec_[s];
  float log_prob = -0.5; // e.g. log_prob = lm->GetLogProb(wseq, eos_symbol_);
  return Weight(-log_prob); // assuming weight is FloatWeight.
}

template<class Arc>
bool LmExampleDeterministicOnDemandFst<Arc>::GetArc(
    StateId s, Label ilabel, Arc *oarc) {
  KALDI_ASSERT(static_cast<size_t>(s) < state_vec_.size());
  std::vector<Label> wseq = state_vec_[s];
  float log_prob = -0.25; // e.g. log_prob = lm->GetLogProb(wseq, ilabel);
  wseq.push_back(ilabel); // the code might be different if your histories are the
  // other way around.

  while (0) { // e.g. while !lm->HistoryStateExists(wseq)
    wseq.erase(wseq.begin(), wseq.begin() + 1); // remove most distant element of history.
    // note: if your histories are the other way round, you might just do
    // wseq.pop() here.
  }
  if (log_prob == -std::numeric_limits<float>::infinity()) { // assume this
    // is what happens if prob of the word is zero.  Some LMs will never
    // return zero.
    return false; // no arc.
  }
  std::pair<const std::vector<Label>, StateId> new_value(
      wseq,
      static_cast<Label>(state_vec_.size()));

  // Now get state id for destination state.
  typedef typename MapType::iterator IterType;
  std::pair<IterType, bool> result = state_map_.insert(new_value);
  if (result.second == true) // was inserted
    state_vec_.push_back(wseq);
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second; // the next-state id.
  oarc->weight = Weight(-log_prob);
  return true;
}


template<class Arc>
void ComposeDeterministicOnDemand(const Fst<Arc> &fst1,
                                  DeterministicOnDemandFst<Arc> *fst2,
                                  MutableFst<Arc> *fst_composed) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef std::pair<StateId, StateId> StatePair;
  typedef unordered_map<StatePair, StateId,
    kaldi::PairHasher<StateId> > MapType;
  typedef typename MapType::iterator IterType;

  fst_composed->DeleteStates();

  MapType state_map;
  std::queue<StatePair> state_queue;

  // Set start state in fst_composed.
  StateId s1 = fst1.Start(),
          s2 = fst2->Start(),
          start_state = fst_composed->AddState();
  StatePair start_pair(s1, s2);
  state_queue.push(start_pair);
  fst_composed->SetStart(start_state);
  // A mapping between pairs of states in fst1 and fst2 and the corresponding
  // state in fst_composed.
  std::pair<const StatePair, StateId> start_map(start_pair, start_state);
  std::pair<IterType, bool> result = state_map.insert(start_map);
  KALDI_ASSERT(result.second == true);

  while (!state_queue.empty()) {
    StatePair q = state_queue.front();
    StateId q1 = q.first,
            q2 = q.second;
    state_queue.pop();
    // If the product of the final weights of the two fsts is non-zero then
    // we can set a final-prob in fst_composed
    Weight final_weight = Times(fst1.Final(q1), fst2->Final(q2));
    if (final_weight != Weight::Zero()) {
      KALDI_ASSERT(state_map.find(q) != state_map.end());
      fst_composed->SetFinal(state_map[q], final_weight);
    }

    // for each pair of edges from fst1 and fst2 at q1 and q2.
    for (ArcIterator<Fst<Arc> > aiter(fst1, q1); !aiter.Done(); aiter.Next()) {
      const Arc &arc1 = aiter.Value();
      Arc arc2;
      StatePair next_pair;
      StateId next_state1 = arc1.nextstate,
              next_state2,
              next_state;
      // If there is an epsilon on the arc of fst1 we transition to the next
      // state but keep fst2 at the current state.
      if (arc1.olabel == 0) {
        next_state2 = q2;
      } else {
        bool match = fst2->GetArc(q2, arc1.olabel, &arc2);
        if (!match)  // There is no matching arc -> nothing to do.
          continue;
        next_state2 = arc2.nextstate;
      }
      next_pair = StatePair(next_state1, next_state2);
      IterType sitr = state_map.find(next_pair);
      // If sitr == state_map.end() then the state isn't in fst_composed yet.
      if (sitr == state_map.end()) {
        next_state = fst_composed->AddState();
        std::pair<const StatePair, StateId> new_state(
          next_pair, next_state);
        std::pair<IterType, bool> result = state_map.insert(new_state);
        // Since we already checked if state_map contained new_state,
        // it should always be added if we reach here.
        KALDI_ASSERT(result.second == true);
        state_queue.push(next_pair);
      // If sitr != state_map.end() then the next state is already in
      // the state_map.
      } else {
        next_state = sitr->second;
      }
      if (arc1.olabel == 0) {
        fst_composed->AddArc(state_map[q], Arc(arc1.ilabel, 0, arc1.weight,
                                               next_state));
      } else {
        fst_composed->AddArc(state_map[q], Arc(arc1.ilabel, arc2.olabel,
          Times(arc1.weight, arc2.weight), next_state));
      }
    }
  }
}


// we are doing *fst_composed = Compose(Inverse(*left), right).
template<class Arc>
void ComposeDeterministicOnDemandInverse(const Fst<Arc> &right,
                                         DeterministicOnDemandFst<Arc> *left,
                                         MutableFst<Arc> *fst_composed) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef std::pair<StateId, StateId> StatePair;
  typedef unordered_map<StatePair, StateId,
    kaldi::PairHasher<StateId> > MapType;
  typedef typename MapType::iterator IterType;

  fst_composed->DeleteStates();

  // the queue and map contain pairs (state-in-left, state-in-right)
  MapType state_map;
  std::queue<StatePair> state_queue;

  // Set start state in fst_composed.
  StateId s_left = left->Start(),
      s_right = right.Start();
  if (s_left == kNoStateId || s_right == kNoStateId)
    return;  // Empty result.
  StatePair start_pair(s_left, s_right);
  StateId start_state = fst_composed->AddState();
  state_queue.push(start_pair);
  fst_composed->SetStart(start_state);
  // A mapping between pairs of states in *left and right, and the corresponding
  // state in fst_composed.
  std::pair<const StatePair, StateId> start_map(start_pair, start_state);
  std::pair<IterType, bool> result = state_map.insert(start_map);
  KALDI_ASSERT(result.second == true);

  while (!state_queue.empty()) {
    StatePair q = state_queue.front();
    StateId q_left = q.first,
            q_right = q.second;
    state_queue.pop();
    // If the product of the final weights of the two fsts is non-zero then
    // we can set a final-prob in fst_composed
    Weight final_weight = Times(left->Final(q_left), right.Final(q_right));
    if (final_weight != Weight::Zero()) {
      KALDI_ASSERT(state_map.find(q) != state_map.end());
      fst_composed->SetFinal(state_map[q], final_weight);
    }

    for (ArcIterator<Fst<Arc> > aiter(right, q_right); !aiter.Done(); aiter.Next()) {
      const Arc &arc_right = aiter.Value();
      Arc arc_left;
      StatePair next_pair;
      StateId next_state_right = arc_right.nextstate,
              next_state_left,
              next_state;
      // If there is an epsilon on the input side of the rigth arc, we
      // transition to the next state of the output but keep 'left' at the
      // current state.
      if (arc_right.ilabel == 0) {
        next_state_left = q_left;
      } else {
        bool match = left->GetArc(q_left, arc_right.ilabel, &arc_left);
        if (!match)  // There is no matching arc -> nothing to do.
          continue;
        // the next 'swap' is because we are composing with the inverse of
        // *left.  Just removing the swap statement wouldn't let us compose
        // with non-inverted *left though, because the GetArc function call
        // above interprets the second argument as an ilabel not an olabel.
        std::swap(arc_left.ilabel, arc_left.olabel);
        next_state_left = arc_left.nextstate;
      }
      next_pair = StatePair(next_state_left, next_state_right);
      IterType sitr = state_map.find(next_pair);
      // If sitr == state_map.end() then the state isn't in fst_composed yet.
      if (sitr == state_map.end()) {
        next_state = fst_composed->AddState();
        std::pair<const StatePair, StateId> new_state(
          next_pair, next_state);
        std::pair<IterType, bool> result = state_map.insert(new_state);
        // Since we already checked if state_map contained new_state,
        // it should always be added if we reach here.
        KALDI_ASSERT(result.second == true);
        state_queue.push(next_pair);
      // If sitr != state_map.end() then the next state is already in
      // the state_map.
      } else {
        next_state = sitr->second;
      }
      if (arc_right.ilabel == 0) {
        // we didn't get an actual arc from the left FST.
        fst_composed->AddArc(state_map[q], Arc(0, arc_right.olabel,
                                               arc_right.weight,
                                               next_state));
      } else {
        fst_composed->AddArc(state_map[q],
                             Arc(arc_left.ilabel, arc_right.olabel,
                                 Times(arc_left.weight, arc_right.weight),
                                 next_state));
      }
    }
  }
}



} // end namespace fst


#endif
