// fstext/prune-special-inl.h

// Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
//                 Guoguo Chen

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

#ifndef KALDI_FSTEXT_PRUNE_SPECIAL_INL_H_
#define KALDI_FSTEXT_PRUNE_SPECIAL_INL_H_
// Do not include this file directly.  It is included by prune-special.h

#include "fstext/prune-special.h"
#include "base/kaldi-error.h"

namespace fst {


/// This class is used to implement the function PruneSpecial.
template<class Arc> class PruneSpecialClass {
 public:
  typedef typename Arc::StateId InputStateId;
  typedef typename Arc::StateId OutputStateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  
  PruneSpecialClass(const Fst<Arc> &ifst,
                    VectorFst<Arc> *ofst,
                    Weight beam,
                    size_t max_states):
      ifst_(ifst), ofst_(ofst), beam_(beam), max_states_(max_states),
      best_weight_(Weight::Zero()) {
    KALDI_ASSERT(beam != Weight::One());
    KALDI_ASSERT(queue_.size() == 0);
    ofst_->DeleteStates(); // make sure it's empty.
    if (ifst_.Start() == kNoStateId)
      return;
    ofst_->SetStart(ProcessState(ifst_.Start(), Weight::One()));

    while (!queue_.empty()) {
      Task task = queue_.top();
      queue_.pop();
      if (Done(task)) break;
      else ProcessTask(task);
    }
    Connect(ofst);
    if (beam_ != Weight::One())
      Prune(ofst, beam_);
  }
  
  struct Task {
    InputStateId istate;
    OutputStateId ostate; // could be looked up; this is for speed.
    size_t position; // arc position, or -1 if final-prob.
    Weight weight;
    
    Task(InputStateId istate, OutputStateId ostate, size_t position,
         Weight weight): istate(istate), ostate(ostate), position(position),
                         weight(weight) { }
    bool operator < (const Task &other) const {
      return Compare(weight, other.weight) < 0;
    }
  };

  bool Done(const Task &task) {
    if (beam_ != Weight::One() && best_weight_ != Weight::Zero() &&
        Compare(task.weight, Times(best_weight_, beam_)) < 0)
      return true;
    if (max_states_ > 0 &&
        static_cast<size_t>(ofst_->NumStates()) > max_states_)
      return true;
    return false;
  }
  

  // This function assumes "state" has not been seen before, so we need to
  // create a new output-state for it and add tasks.  It returns the
  // output-state id.  "weight" is the best cost from the start-state to this
  // state.
  inline OutputStateId ProcessState(InputStateId istate, const Weight &weight) {
    OutputStateId ostate = ofst_->AddState();
    state_map_[istate] = ostate;
    for (ArcIterator<Fst<Arc> > aiter(ifst_, istate); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      Task new_task(istate, ostate, aiter.Position(),
                    Times(weight, arc.weight));
      KALDI_ASSERT(Compare(arc.weight, Weight::One()) != 1);
      queue_.push(new_task);
    }
    Weight final = ifst_.Final(istate);
    if (final != Weight::Zero()) {
      Task final_task(istate, ostate, static_cast<size_t>(-1),
                      Times(weight, final));
      KALDI_ASSERT(Compare(final, Weight::One()) != 1);
      queue_.push(final_task);
    }
    return ostate;
  }

  // Returns the output-state id corresponding to "istate".  This assumes we are
  // processing a task corresponding to an arc to "istate", and the cost from
  // the start-state to this state is "weight".  Since we process tasks in
  // order, if this is the first time we see this istate, then this is the best
  // cost from the start-state to this state, and it can be used in setting the
  // priority costs in ProcessState().
  inline OutputStateId GetOutputStateId(InputStateId istate,
                                        const Weight &weight) {
    typedef typename unordered_map<InputStateId, OutputStateId>::iterator IterType;
    IterType iter = state_map_.find(istate);
    if (iter == state_map_.end())
      return ProcessState(istate, weight);
    else 
      return iter->second;
  }
  
  void ProcessTask(const Task &task) {
    if (task.position == static_cast<size_t>(-1)) {
      ofst_->SetFinal(task.ostate, ifst_.Final(task.istate));
      if (best_weight_ == Weight::Zero())
        best_weight_ = task.weight; // best-path cost through FST, used for
                                    // beam-pruning.
    } else {
      ArcIterator<Fst<Arc> > aiter(ifst_, task.istate);
      aiter.Seek(task.position); // if we spend most of our time here, we may
                                 // need to store the arc in the Task.
      const Arc &arc = aiter.Value();
      InputStateId next_istate = arc.nextstate;
      OutputStateId next_ostate = GetOutputStateId(next_istate, task.weight);
      Arc oarc(arc.ilabel, arc.olabel, arc.weight, next_ostate);
      ofst_->AddArc(task.ostate, oarc);
    }
  }
  
 private:
  const Fst<Arc> &ifst_;
  VectorFst<Arc> *ofst_;
  Weight beam_;
  size_t max_states_;

  unordered_map<InputStateId, OutputStateId> state_map_;
  std::priority_queue<Task> queue_;
  Weight best_weight_; // if not Zero(), then we have now processed a successful path
                       // through ifst_, and this is the weight.
  
};

template<class Arc>
void PruneSpecial(const Fst<Arc> &ifst,
                  VectorFst<Arc> *ofst,
                  typename Arc::Weight beam,
                  size_t max_states) {
  PruneSpecialClass<Arc> c(ifst, ofst, beam, max_states);
}



}


#endif
