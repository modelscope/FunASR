// lat/phone-align-lattice.cc

// Copyright 2012-2013  Microsoft Corporation
//                      Johns Hopkins University (Author: Daniel Povey)

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


#include "lat/phone-align-lattice.h"
#include "util/stl-utils.h"

namespace kaldi {

class LatticePhoneAligner {
 public:
  typedef CompactLatticeArc::StateId StateId;
  typedef CompactLatticeArc::Label Label;

  class ComputationState { /// The state of the computation in which,
    /// along a single path in the lattice, we work out the phone
    /// boundaries and output phone-aligned arcs. [These may or may not have
    /// words on them; the word symbols are not aligned with anything.
   public:

    /// Advance the computation state by adding the symbols and weights
    /// from this arc.  Gets rid of the weight and puts it in "weight" which
    /// will be put on the output arc; this keeps the state-space small.
    void Advance(const CompactLatticeArc &arc, const PhoneAlignLatticeOptions &opts,
                 LatticeWeight *weight) {
      const std::vector<int32> &string = arc.weight.String();
      transition_ids_.insert(transition_ids_.end(),
                             string.begin(), string.end());
      if (arc.ilabel != 0 && !opts.replace_output_symbols) // note: arc.ilabel==arc.olabel (acceptor)
        word_labels_.push_back(arc.ilabel);
      *weight = Times(weight_, arc.weight.Weight());
      weight_ = LatticeWeight::One();
    }

    /// If it can output a whole phone, it will do so, will put it in arc_out,
    /// and return true; else it will return false.  If it detects an error
    /// condition and *error = false, it will set *error to true and print
    /// a warning.  In this case it will still output phone arcs, they will
    /// just be inaccurate.  Of course once *error is set, something has gone
    /// wrong so don't trust the output too fully.
    /// Note: the "next_state" of the arc will not be set, you have to do that
    /// yourself.
    bool OutputPhoneArc(const TransitionInformation &tmodel,
                        const PhoneAlignLatticeOptions &opts,
                        CompactLatticeArc *arc_out,
                        bool *error);

    /// This will succeed (and output the arc) if we have >1 word in words_;
    /// the arc won't have any transition-ids on it.  This is intended to fix
    /// a particular pathology where too many words were pending and we had
    /// blowup.
    bool OutputWordArc(const TransitionInformation &tmodel,
                       const PhoneAlignLatticeOptions &opts,
                       CompactLatticeArc *arc_out,
                       bool *error);

    bool IsEmpty() { return (transition_ids_.empty() && word_labels_.empty()); }

    /// FinalWeight() will return "weight" if both transition_ids
    /// and word_labels are empty, otherwise it will return
    /// Weight::Zero().
    LatticeWeight FinalWeight() { return (IsEmpty() ? weight_ : LatticeWeight::Zero()); }

    /// This function may be called when you reach the end of
    /// the lattice and this structure hasn't voluntarily
    /// output words using "OutputArc".  If IsEmpty() == false,
    /// then you can call this function and it will output
    /// an arc.  The only
    /// non-error state in which this happens, is when a word
    /// (or silence) has ended, but we don't know that it's
    /// ended because we haven't seen the first transition-id
    /// from the next word.  Otherwise (error state), the output
    /// will consist of partial words, and this will only
    /// happen for lattices that were somehow broken, i.e.
    /// had not reached the final state.
    void OutputArcForce(const TransitionInformation &tmodel,
                        const PhoneAlignLatticeOptions &opts,
                        CompactLatticeArc *arc_out,
                        bool *error);

    size_t Hash() const {
      VectorHasher<int32> vh;
      return vh(transition_ids_) + 90647 * vh(word_labels_);
      // 90647 is an arbitrary largish prime number.
      // We don't bother including the weight in the hash--
      // we don't really expect duplicates with the same vectors
      // but different weights, and anyway, this is only an
      // efficiency issue.
    }

    // Just need an arbitrary complete order.
    bool operator == (const ComputationState &other) const {
      return (transition_ids_ == other.transition_ids_
              && word_labels_ == other.word_labels_
              && weight_ == other.weight_);
    }

    ComputationState(): weight_(LatticeWeight::One()) { } // initial state.
    ComputationState(const ComputationState &other):
        transition_ids_(other.transition_ids_), word_labels_(other.word_labels_),
        weight_(other.weight_) { }
   private:
    std::vector<int32> transition_ids_;
    std::vector<int32> word_labels_;
    LatticeWeight weight_; // contains two floats.
  };


  struct Tuple {
    Tuple(StateId input_state, ComputationState comp_state):
        input_state(input_state), comp_state(comp_state) {}
    StateId input_state;
    ComputationState comp_state;
  };

  struct TupleHash {
    size_t operator() (const Tuple &state) const {
      return state.input_state + 102763 * state.comp_state.Hash();
      // 102763 is just an arbitrary prime number
    }
  };
  struct TupleEqual {
    bool operator () (const Tuple &state1, const Tuple &state2) const {
      // treat this like operator ==
      return (state1.input_state == state2.input_state
              && state1.comp_state == state2.comp_state);
    }
  };

  typedef unordered_map<Tuple, StateId, TupleHash, TupleEqual> MapType;

  StateId GetStateForTuple(const Tuple &tuple, bool add_to_queue) {
    MapType::iterator iter = map_.find(tuple);
    if (iter == map_.end()) { // not in map.
      StateId output_state = lat_out_->AddState();
      map_[tuple] = output_state;
      if (add_to_queue)
        queue_.push_back(std::make_pair(tuple, output_state));
      return output_state;
    } else {
      return iter->second;
    }
  }

  void ProcessFinal(Tuple tuple, StateId output_state) {
    // ProcessFinal is only called if the input_state has
    // final-prob of One().  [else it should be zero.  This
    // is because we called CreateSuperFinal().]

    if (tuple.comp_state.IsEmpty()) { // computation state doesn't have
      // anything pending.
      std::vector<int32> empty_vec;
      CompactLatticeWeight cw(tuple.comp_state.FinalWeight(), empty_vec);
      lat_out_->SetFinal(output_state, Plus(lat_out_->Final(output_state), cw));
    } else {
      // computation state has something pending, i.e. input or
      // output symbols that need to be flushed out.  Note: OutputArc() would
      // have returned false or we wouldn't have been called, so we have to
      // force it out.
      CompactLatticeArc lat_arc;
      // Note: the next call will change the computation-state of the tuple,
      // so it becomes a different tuple.
      tuple.comp_state.OutputArcForce(tmodel_, opts_, &lat_arc, &error_);
      lat_arc.nextstate = GetStateForTuple(tuple, true); // true == add to queue.
      // The final-prob stuff will get called again from ProcessQueueElement().
      // Note: because we did CreateSuperFinal(), this final-state on the input
      // lattice will have no output arcs (and unit final-prob), so there will be
      // no complications with processing the arcs from this state (there won't
      // be any).
      KALDI_ASSERT(output_state != lat_arc.nextstate);
      lat_out_->AddArc(output_state, lat_arc);
    }
  }


  void ProcessQueueElement() {
    KALDI_ASSERT(!queue_.empty());
    Tuple tuple = queue_.back().first;
    StateId output_state = queue_.back().second;
    queue_.pop_back();

    // First thing is-- we see whether the computation-state has something
    // pending that it wants to output.  In this case we don't do
    // anything further.  This is a chosen behavior similar to the
    // epsilon-sequencing rules encoded by the filters in
    // composition.
    CompactLatticeArc lat_arc;
    if (tuple.comp_state.OutputPhoneArc(tmodel_, opts_, &lat_arc, &error_) ||
        tuple.comp_state.OutputWordArc(tmodel_, opts_, &lat_arc, &error_)) {
      // note: the functions OutputPhoneArc() and OutputWordArc() change the
      // tuple (when they return true).
      lat_arc.nextstate = GetStateForTuple(tuple, true); // true == add to
                                                         // queue, if not
                                                         // already present.
      KALDI_ASSERT(output_state != lat_arc.nextstate);
      lat_out_->AddArc(output_state, lat_arc);
    } else {
      // when there's nothing to output, we'll process arcs from the input-state.
      // note: it would in a sense be valid to do both (i.e. process the stuff
      // above, and also these), but this is a bit like the epsilon-sequencing
      // stuff in composition: we avoid duplicate arcs by doing it this way.

      if (lat_.Final(tuple.input_state) != CompactLatticeWeight::Zero()) {
        KALDI_ASSERT(lat_.Final(tuple.input_state) == CompactLatticeWeight::One());
        // ... since we did CreateSuperFinal.
        ProcessFinal(tuple, output_state);
      }
      // Now process the arcs.  Note: final-states shouldn't have any arcs.
      for(fst::ArcIterator<CompactLattice> aiter(lat_, tuple.input_state);
          !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        Tuple next_tuple(tuple);
        LatticeWeight weight;
        next_tuple.comp_state.Advance(arc, opts_, &weight);
        next_tuple.input_state = arc.nextstate;
        StateId next_output_state = GetStateForTuple(next_tuple, true); // true == add to queue,
        // if not already present.
        // We add an epsilon arc here (as the input and output happens
        // separately)... the epsilons will get removed later.
        KALDI_ASSERT(next_output_state != output_state);
        lat_out_->AddArc(output_state,
                         CompactLatticeArc(0, 0,
                            CompactLatticeWeight(weight, std::vector<int32>()),
                            next_output_state));
      }
    }
  }

  LatticePhoneAligner(const CompactLattice &lat,
                      const TransitionInformation &tmodel,
                      const PhoneAlignLatticeOptions &opts,
                     CompactLattice *lat_out):
      lat_(lat), tmodel_(tmodel), opts_(opts), lat_out_(lat_out),
      error_(false) {
    fst::CreateSuperFinal(&lat_); // Creates a super-final state, so the
    // only final-probs are One().
  }

  // Removes epsilons; also removes unreachable states...
  // not sure if these would exist if original was connected.
  // This also replaces the temporary symbols for the silence
  // and partial-words, with epsilons, if we wanted epsilons.
  void RemoveEpsilonsFromLattice() {
    RmEpsilon(lat_out_, true); // true = connect.
  }

  bool AlignLattice() {
    lat_out_->DeleteStates();
    if (lat_.Start() == fst::kNoStateId) {
      KALDI_WARN << "Trying to word-align empty lattice.";
      return false;
    }
    ComputationState initial_comp_state;
    Tuple initial_tuple(lat_.Start(), initial_comp_state);
    StateId start_state = GetStateForTuple(initial_tuple, true); // True = add this to queue.
    lat_out_->SetStart(start_state);

    while (!queue_.empty())
      ProcessQueueElement();

    if (opts_.remove_epsilon)
      RemoveEpsilonsFromLattice();

    return !error_;
  }

  CompactLattice lat_;
  const TransitionInformation &tmodel_;
  const PhoneAlignLatticeOptions &opts_;
  CompactLattice *lat_out_;

  std::vector<std::pair<Tuple, StateId> > queue_;
  MapType map_; // map from tuples to StateId.
  bool error_;
};

bool LatticePhoneAligner::ComputationState::OutputPhoneArc(
    const TransitionInformation &tmodel,
    const PhoneAlignLatticeOptions &opts,
    CompactLatticeArc *arc_out,
    bool *error) {
  if (transition_ids_.empty()) return false;
  int32 phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  // we assume the start of transition_ids_ is the start of the phone;
  // this is a precondition.
  size_t len = transition_ids_.size(), i;
  // Keep going till we reach a "final" transition-id; note, if
  // reorder==true, we have to go a bit further after this.
  for (i = 0; i < len; i++) {
    int32 tid = transition_ids_[i];
    int32 this_phone = tmodel.TransitionIdToPhone(tid);
    if (this_phone != phone && ! *error) { // error condition: should have
                                           // reached final transition-id first.
      *error = true;
      KALDI_WARN << phone << " -> " << this_phone;
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
    }
    if (tmodel.IsFinal(tid))
      break;
  }
  if (i == len) return false; // fell off loop.
  i++; // go past the one for which IsFinal returned true.
  if (opts.reorder) // we have to consume the following self-loop transition-ids.
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false; // we don't know if it ends here... so can't output arc.

  // interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(),
                              transition_ids_.begin()+i);

  Label output_label = 0;
  if (!word_labels_.empty()) {
    output_label = word_labels_[0];
    word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1);
  }
  if (opts.replace_output_symbols)
    output_label = phone;
  *arc_out = CompactLatticeArc(output_label, output_label,
                               CompactLatticeWeight(weight_, tids_out),
                               fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(), transition_ids_.begin()+i);
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}

bool LatticePhoneAligner::ComputationState::OutputWordArc(
    const TransitionInformation &tmodel,
    const PhoneAlignLatticeOptions &opts,
    CompactLatticeArc *arc_out,
    bool *error) {
  // output a word but no phones.
  if (word_labels_.size() < 2) return false;

  int32 output_label = word_labels_[0];
  word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1);

  *arc_out = CompactLatticeArc(output_label, output_label,
                               CompactLatticeWeight(weight_, std::vector<int32>()),
                               fst::kNoStateId);
  weight_ = LatticeWeight::One(); // we just output the weight, so set it to one.
  return true;
}


void LatticePhoneAligner::ComputationState::OutputArcForce(
    const TransitionInformation &tmodel,
    const PhoneAlignLatticeOptions &opts,
    CompactLatticeArc *arc_out,
    bool *error) {
  KALDI_ASSERT(!IsEmpty());

  int32 phone = -1; // This value -1 will never be used,
  // although it might not be obvious from superficially checking
  // the code.  IsEmpty() would be true if we had transition_ids_.empty()
  // and opts.replace_output_symbols, so we would already die by assertion;
  // in fact, this function would never be called.

  if (!transition_ids_.empty()) { // Do some checking here.
    int32 tid = transition_ids_[0];
    phone = tmodel.TransitionIdToPhone(tid);
    int32 num_final = 0;
    for (int32 i = 0; i < transition_ids_.size(); i++) { // A check.
      int32 this_tid = transition_ids_[i];
      int32 this_phone = tmodel.TransitionIdToPhone(this_tid);
      bool is_final = tmodel.IsFinal(this_tid); // should be exactly one.
      if (is_final) num_final++;
      if (this_phone != phone && ! *error) {
        KALDI_WARN << "Mismatch in phone: error in lattice or mismatched transition model?";
        *error = true;
      }
    }
    if (num_final != 1 && ! *error) {
      KALDI_WARN << "Problem phone-aligning lattice: saw " << num_final
                 << " final-states in last phone in lattice (forced out?) "
                 << "Producing partial lattice.";
      *error = true;
    }
  }

  Label output_label = 0;
  if (!word_labels_.empty()) {
    output_label = word_labels_[0];
    word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1);
  }
  if (opts.replace_output_symbols)
    output_label = phone;
  *arc_out = CompactLatticeArc(output_label, output_label,
                               CompactLatticeWeight(weight_, transition_ids_),
                               fst::kNoStateId);
  transition_ids_.clear();
  weight_ = LatticeWeight::One(); // we just output the weight.
}

bool PhoneAlignLattice(const CompactLattice &lat,
                       const TransitionInformation &tmodel,
                       const PhoneAlignLatticeOptions &opts,
                       CompactLattice *lat_out) {
  LatticePhoneAligner aligner(lat, tmodel, opts, lat_out);
  return aligner.AlignLattice();
}


}  // namespace kaldi
