// fstext/remove-eps-local-inl.h

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey

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

#ifndef KALDI_FSTEXT_REMOVE_EPS_LOCAL_INL_H_
#define KALDI_FSTEXT_REMOVE_EPS_LOCAL_INL_H_


namespace fst {


template<class Weight>
struct ReweightPlusDefault {
  inline Weight operator () (const Weight &a, const Weight &b) {
    return Plus(a, b);
  }
};

struct ReweightPlusLogArc {
  inline TropicalWeight operator () (const TropicalWeight &a,
                                     const TropicalWeight &b) {
    LogWeight a_log(a.Value()), b_log(b.Value());
    return TropicalWeight(Plus(a_log, b_log).Value());
  }
};



template<class Arc, class ReweightPlus = ReweightPlusDefault<typename Arc::Weight> >
class RemoveEpsLocalClass {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

 public:
  RemoveEpsLocalClass(MutableFst<Arc> *fst):
      fst_(fst) {
    if (fst_->Start() == kNoStateId) return;  // empty.
    non_coacc_state_ = fst_->AddState();
    InitNumArcs();
    StateId num_states = fst_->NumStates();
    for (StateId s = 0; s < num_states; s++)
      for (size_t pos = 0; pos < fst_->NumArcs(s); pos++)
        RemoveEps(s, pos);
    assert(CheckNumArcs());
    Connect(fst);  // remove inaccessible states.
  }
 private:
  MutableFst<Arc> *fst_;
  StateId non_coacc_state_;  //  use this to delete arcs: make it nextstate
  std::vector<StateId> num_arcs_in_;   // The number of arcs into the state, plus one
                                  // if it's the start state.
  std::vector<StateId> num_arcs_out_;  // The number of arcs out of the state, plus
                                  // one if it's a final state.
  ReweightPlus reweight_plus_;

  bool CanCombineArcs(const Arc &a, const Arc &b, Arc *c) {
    if (a.ilabel != 0 && b.ilabel != 0) return false;
    if (a.olabel != 0 && b.olabel != 0) return false;
    c->weight = Times(a.weight, b.weight);
    c->ilabel = (a.ilabel != 0 ? a.ilabel : b.ilabel);
    c->olabel = (a.olabel != 0 ? a.olabel : b.olabel);
    c->nextstate = b.nextstate;
    return true;
  }

  static bool CanCombineFinal(const Arc &a, Weight final_prob, Weight *final_prob_out) {
    if (a.ilabel != 0 || a.olabel != 0) return false;
    else {
      *final_prob_out = Times(a.weight, final_prob);
      return true;
    }
  }

  void InitNumArcs() {  // init num transitions in/out of each state.
    StateId num_states = fst_->NumStates();
    num_arcs_in_.resize(num_states);
    num_arcs_out_.resize(num_states);
    num_arcs_in_[fst_->Start()]++;  // count start as trans in.
    for (StateId s = 0; s < num_states; s++) {
      if (fst_->Final(s) != Weight::Zero())
        num_arcs_out_[s]++;  // count final as transition.
      for (ArcIterator<MutableFst<Arc> > aiter(*fst_, s); !aiter.Done(); aiter.Next()) {
        num_arcs_in_[aiter.Value().nextstate]++;
        num_arcs_out_[s]++;
      }
    }
  }

  bool CheckNumArcs() {  // check num arcs in/out of each state, at end.  Debug.
    num_arcs_in_[fst_->Start()]--;  // count start as trans in.
    StateId num_states =   fst_->NumStates();
    for (StateId s = 0; s < num_states; s++) {
      if (s == non_coacc_state_) continue;
      if (fst_->Final(s) != Weight::Zero())
        num_arcs_out_[s]--;  // count final as transition.
      for (ArcIterator<MutableFst<Arc> > aiter(*fst_, s); !aiter.Done(); aiter.Next()) {
        if (aiter.Value().nextstate == non_coacc_state_) continue;
        num_arcs_in_[aiter.Value().nextstate]--;
        num_arcs_out_[s]--;
      }
    }
    for (StateId s = 0; s < num_states; s++) {
      assert(num_arcs_in_[s] == 0);
      assert(num_arcs_out_[s] == 0);
    }
    return true;  // always does this.  so we can assert it w/o warnings.
  }

  inline void GetArc(StateId s, size_t pos, Arc *arc) const {
    ArcIterator<MutableFst<Arc> > aiter(*fst_, s);
    aiter.Seek(pos);
    *arc = aiter.Value();
  }

  inline void SetArc(StateId s, size_t pos, const Arc &arc) {
    MutableArcIterator<MutableFst<Arc> > aiter(fst_, s);
    aiter.Seek(pos);
    aiter.SetValue(arc);
  }


  void Reweight(StateId s, size_t pos, Weight reweight) {
    // Reweight is called from RemoveEpsPattern1; it is a step we
    // do to preserve stochasticity.  This function multiplies the
    // arc at (s, pos) by reweight and divides all the arcs [+final-prob]
    // out of the next state by the same.  This is only valid if
    // the next state has only one arc in and is not the start state.
    assert(reweight != Weight::Zero());
    MutableArcIterator<MutableFst<Arc> > aiter(fst_, s);
    aiter.Seek(pos);
    Arc arc = aiter.Value();
    assert(num_arcs_in_[arc.nextstate] == 1);
    arc.weight = Times(arc.weight, reweight);
    aiter.SetValue(arc);

    for (MutableArcIterator<MutableFst<Arc> > aiter_next(fst_, arc.nextstate);
         !aiter_next.Done();
         aiter_next.Next()) {
      Arc nextarc = aiter_next.Value();
      if (nextarc.nextstate != non_coacc_state_) {
        nextarc.weight = Divide(nextarc.weight, reweight, DIVIDE_LEFT);
        aiter_next.SetValue(nextarc);
      }
    }
    Weight final = fst_->Final(arc.nextstate);
    if (final != Weight::Zero()) {
      fst_->SetFinal(arc.nextstate, Divide(final, reweight, DIVIDE_LEFT));
    }
  }

  // RemoveEpsPattern1 applies where this arc, which is not a
  // self-loop, enters a state which has only one input transition
  // [and is not the start state], and has multiple output
  // transitions [counting being the final-state as a final-transition].

  void RemoveEpsPattern1(StateId s, size_t pos, Arc arc) {
    const StateId nextstate = arc.nextstate;
    Weight total_removed = Weight::Zero(),
        total_kept = Weight::Zero();  // totals out of nextstate.
    std::vector<Arc> arcs_to_add;  // to add to state s.
    for (MutableArcIterator<MutableFst<Arc> > aiter_next(fst_, nextstate);
        !aiter_next.Done();
        aiter_next.Next()) {
      Arc nextarc = aiter_next.Value();
      if (nextarc.nextstate == non_coacc_state_) continue;  // deleted.
      Arc combined;
      if (CanCombineArcs(arc, nextarc, &combined)) {
        total_removed = reweight_plus_(total_removed, nextarc.weight);
        num_arcs_out_[nextstate]--;
        num_arcs_in_[nextarc.nextstate]--;
        nextarc.nextstate = non_coacc_state_;
        aiter_next.SetValue(nextarc);
        arcs_to_add.push_back(combined);
      } else {
        total_kept = reweight_plus_(total_kept, nextarc.weight);
      }
    }

    {  // now final-state.
      Weight next_final = fst_->Final(nextstate);
      if (next_final != Weight::Zero()) {
        Weight new_final;
        if (CanCombineFinal(arc, next_final, &new_final)) {
          total_removed = reweight_plus_(total_removed, next_final);
          if (fst_->Final(s) == Weight::Zero())
            num_arcs_out_[s]++;  // final is counted as arc.
          fst_->SetFinal(s, Plus(fst_->Final(s), new_final));
          num_arcs_out_[nextstate]--;
          fst_->SetFinal(nextstate, Weight::Zero());
        } else {
          total_kept = reweight_plus_(total_kept, next_final);
        }
      }
    }

    if (total_removed != Weight::Zero()) {  // did something...
      if (total_kept == Weight::Zero()) {  // removed everything: remove arc.
        num_arcs_out_[s]--;
        num_arcs_in_[arc.nextstate]--;
        arc.nextstate = non_coacc_state_;
        SetArc(s, pos, arc);
      } else {
        // Have to reweight.
        Weight total = reweight_plus_(total_removed, total_kept);
        Weight reweight = Divide(total_kept, total, DIVIDE_LEFT);  // <=1
        Reweight(s, pos, reweight);
      }
    }
    // Now add the arcs we were going to add.
    for (size_t i = 0; i < arcs_to_add.size(); i++) {
      num_arcs_out_[s]++;
      num_arcs_in_[arcs_to_add[i].nextstate]++;
      fst_->AddArc(s, arcs_to_add[i]);
    }
  }

  void RemoveEpsPattern2(StateId s, size_t pos, Arc arc) {

    // Pattern 2 is where "nextstate" has only one arc out, counting
    // being-the-final-state as an arc, but possibly multiple arcs in.
    // Also, nextstate != s.

    const StateId nextstate = arc.nextstate;
    bool can_delete_next = (num_arcs_in_[nextstate] == 1);  // if
    // we combine, can delete the corresponding out-arc/final-prob
    // of nextstate.
    bool delete_arc = false;  // set to true if this arc to be deleted.

    Weight next_final = fst_->Final(arc.nextstate);
    if (next_final != Weight::Zero()) {  // nextstate has no actual arcs out, only final-prob.
      Weight new_final;
      if (CanCombineFinal(arc, next_final, &new_final)) {
        if (fst_->Final(s) == Weight::Zero())
          num_arcs_out_[s]++;  // final is counted as arc.
        fst_->SetFinal(s, Plus(fst_->Final(s), new_final));
        delete_arc = true;  // will delete "arc".
        if (can_delete_next) {
          num_arcs_out_[nextstate]--;
          fst_->SetFinal(nextstate, Weight::Zero());
        }
      }
    } else {  // has an arc but no final prob.
      MutableArcIterator<MutableFst<Arc> > aiter_next(fst_, nextstate);
      assert(!aiter_next.Done());
      while (aiter_next.Value().nextstate == non_coacc_state_) {
        aiter_next.Next();
        assert(!aiter_next.Done());
      }
      // now aiter_next points to a real arc out of nextstate.
      Arc nextarc = aiter_next.Value();
      Arc combined;
      if (CanCombineArcs(arc, nextarc, &combined)) {
        delete_arc = true;
        if (can_delete_next) {  // do it before we invalidate iterators
          num_arcs_out_[nextstate]--;
          num_arcs_in_[nextarc.nextstate]--;
          nextarc.nextstate = non_coacc_state_;
          aiter_next.SetValue(nextarc);
        }
        num_arcs_out_[s]++;
        num_arcs_in_[combined.nextstate]++;
        fst_->AddArc(s, combined);
      }
    }
    if (delete_arc) {
      num_arcs_out_[s]--;
      num_arcs_in_[nextstate]--;
      arc.nextstate = non_coacc_state_;
      SetArc(s, pos, arc);
    }
  }

  void RemoveEps(StateId s, size_t pos) {
    // Tries to do local epsilon-removal for arc sequences starting with this arc
    Arc arc;
    GetArc(s, pos, &arc);
    StateId nextstate = arc.nextstate;
    if (nextstate == non_coacc_state_) return;  // deleted arc.
    if (nextstate == s) return;  // don't handle self-loops: too complex.

    if (num_arcs_in_[nextstate] == 1 && num_arcs_out_[nextstate] > 1) {
      RemoveEpsPattern1(s, pos, arc);
    } else if (num_arcs_out_[nextstate] == 1) {
      RemoveEpsPattern2(s, pos, arc);
    }
  }

};


template<class Arc>
void RemoveEpsLocal(MutableFst<Arc> *fst) {
  RemoveEpsLocalClass<Arc> c(fst);  // work gets done in initializer.
}


void RemoveEpsLocalSpecial(MutableFst<StdArc> *fst) {
  // work gets done in initializer.
  RemoveEpsLocalClass<StdArc, ReweightPlusLogArc> c(fst);
}

} // end namespace fst.

#endif
