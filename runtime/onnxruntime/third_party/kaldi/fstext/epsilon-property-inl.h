// fstext/epsilon-property-inl.h

// Copyright 2014    Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_FSTEXT_EPSILON_PROPERTY_INL_H_
#define KALDI_FSTEXT_EPSILON_PROPERTY_INL_H_

namespace fst {



template<class Arc>
void ComputeStateInfo(const VectorFst<Arc> &fst,
                      std::vector<char> *epsilon_info) {
  typedef typename Arc::StateId StateId;
  typedef VectorFst<Arc> Fst;
  epsilon_info->clear();
  epsilon_info->resize(fst.NumStates(), static_cast<char>(0));
  for (StateId s = 0; s < fst.NumStates(); s++) {
    for (ArcIterator<Fst> aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0 && arc.olabel == 0) {
        (*epsilon_info)[arc.nextstate] |= static_cast<char>(kStateHasEpsilonArcsEntering);
        (*epsilon_info)[s] |= static_cast<char>(kStateHasEpsilonArcsLeaving);
      } else {
        (*epsilon_info)[arc.nextstate] |= static_cast<char>(kStateHasNonEpsilonArcsEntering);
        (*epsilon_info)[s] |= static_cast<char>(kStateHasNonEpsilonArcsLeaving);
      }
    }
  }
}

template<class Arc>
void EnsureEpsilonProperty(VectorFst<Arc> *fst) {

  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef VectorFst<Arc> Fst;
  std::vector<char> epsilon_info;
  ComputeStateInfo(*fst, &epsilon_info);


  StateId num_states_old = fst->NumStates();
  StateId non_coaccessible_state = fst->AddState();

  /// new_state_vec is for those states that have both epsilon and 
  /// non-epsilon arcs entering.  For these states, we'll create a new
  /// state for the non-epsilon arcs to enter and put it in this array,
  /// and we'll put an epsilon transition from the new state to the old state.
  std::vector<StateId> new_state_vec(num_states_old, kNoStateId);
  for (StateId s = 0; s < num_states_old; s++) {
    if ((epsilon_info[s] & kStateHasEpsilonArcsEntering) != 0 &&
        (epsilon_info[s] & kStateHasNonEpsilonArcsEntering) != 0) {
      assert(s != fst->Start()); // a type of cyclic FST we can't handle
                                 // easily.
      StateId new_state = fst->AddState();
      new_state_vec[s] = new_state;
      fst->AddArc(new_state, Arc(0, 0, Weight::One(), s));
    }
  }

  /// First modify arcs to point to states in new_state_vec when
  /// necessary.
  for (StateId s = 0; s < num_states_old; s++) {
    for (MutableArcIterator<Fst> aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0 || arc.olabel != 0) { // non-epsilon arc
        StateId replacement_state;
        if (arc.nextstate >= 0 && arc.nextstate < num_states_old &&
            (replacement_state = new_state_vec[arc.nextstate]) !=
             kNoStateId) {
          arc.nextstate = replacement_state;
          aiter.SetValue(arc);
        }
      }
    }
  }

  /// Now handle the situation where states have both epsilon and non-epsilon
  /// arcs leaving.
  for (StateId s = 0; s < num_states_old; s++) {
    if ((epsilon_info[s] & kStateHasEpsilonArcsLeaving) != 0 &&
        (epsilon_info[s] & kStateHasNonEpsilonArcsLeaving) != 0) {
      // state has non-epsilon and epsilon arcs leaving.
      // create a new state and move the non-epsilon arcs to leave
      // from there instead.
      StateId new_state = fst->AddState();
      for (MutableArcIterator<Fst> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0 || arc.olabel != 0) { // non-epsilon arc.
          assert(arc.nextstate != s); // we don't handle cyclic FSTs.
          // move this arc to leave from the new state:
          fst->AddArc(new_state, arc); 
          arc.nextstate = non_coaccessible_state;
          aiter.SetValue(arc); // invalidate the arc, Connect() will remove it.
        }
      }
      // Create an epsilon arc to the new state.
      fst->AddArc(s, Arc(0, 0, Weight::One(), new_state));
    }
  }
  Connect(fst); // Removes arcs to the non-coaccessible state.
}



} // namespace fst.

#endif
