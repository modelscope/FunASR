// lat/word-align-lattice.cc

// Copyright 2011-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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


#include "lat/word-align-lattice.h"
#include "util/stl-utils.h"

namespace kaldi {

class LatticeWordAligner {
 public:
  typedef CompactLatticeArc::StateId StateId;
  typedef CompactLatticeArc::Label Label;

  class ComputationState { /// The state of the computation in which,
    /// along a single path in the lattice, we work out the word
    /// boundaries and output aligned arcs.
   public:

    /// Advance the computation state by adding the symbols and weights
    /// from this arc.  We'll put the weight on the output arc; this helps
    /// keep the state-space smaller.
    void Advance(const CompactLatticeArc &arc, LatticeWeight *weight) {
      const std::vector<int32> &string = arc.weight.String();
      transition_ids_.insert(transition_ids_.end(),
                             string.begin(), string.end());
      if (arc.ilabel != 0) // note: arc.ilabel==arc.olabel (acceptor)
        word_labels_.push_back(arc.ilabel);
      *weight = Times(weight_, arc.weight.Weight());
      weight_ = LatticeWeight::One();
    }

    /// If it can output a whole word, it will do so, will put it in arc_out,
    /// and return true; else it will return false.  If it detects an error
    /// condition and *error = false, it will set *error to true and print
    /// a warning.  In this case it may or may not [output an arc and return true],
    /// depending on what we think is most likely the right thing to do.  Of
    /// course once *error is set, something has gone wrong so don't trust
    /// the output too fully.
    /// Note: the "next_state" of the arc will not be set, you have to do that
    /// yourself.
    bool OutputArc(const WordBoundaryInfo &info,
                   const TransitionInformation &tmodel,
                   CompactLatticeArc *arc_out,
                   bool *error) {
      // order of this ||-expression doesn't matter for
      // function behavior, only for efficiency, since the
      // cases are disjoint.
      return OutputNormalWordArc(info, tmodel, arc_out, error) ||
          OutputSilenceArc(info, tmodel, arc_out, error) ||
          OutputOnePhoneWordArc(info, tmodel, arc_out, error);
    }

    bool OutputSilenceArc(const WordBoundaryInfo &info,
                          const TransitionInformation &tmodel,
                          CompactLatticeArc *arc_out,
                          bool *error);
    bool OutputOnePhoneWordArc(const WordBoundaryInfo &info,
                               const TransitionInformation &tmodel,
                               CompactLatticeArc *arc_out,
                               bool *error);
    bool OutputNormalWordArc(const WordBoundaryInfo &info,
                             const TransitionInformation &tmodel,
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
    void OutputArcForce(const WordBoundaryInfo &info,
                        const TransitionInformation &tmodel,
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
      tuple.comp_state.OutputArcForce(info_, tmodel_, &lat_arc, &error_);
      // True in the next line means add it to the queue.
      lat_arc.nextstate = GetStateForTuple(tuple, true);
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
    if (tuple.comp_state.OutputArc(info_, tmodel_, &lat_arc, &error_)) {
      // note: this function changes the tuple (when it returns true).
      lat_arc.nextstate = GetStateForTuple(tuple, true); // true == add to queue,
      // if not already present.
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
      // Now process the arcs.  Note: final-state shouldn't have any arcs.
      for (fst::ArcIterator<CompactLattice> aiter(lat_, tuple.input_state);
          !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        Tuple next_tuple(tuple);
        LatticeWeight weight;
        next_tuple.comp_state.Advance(arc, &weight);
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

  LatticeWordAligner(const CompactLattice &lat,
                     const TransitionInformation &tmodel,
                     const WordBoundaryInfo &info,
                     int32 max_states,
                     CompactLattice *lat_out):
      lat_(lat), tmodel_(tmodel), info_in_(info), info_(info),
      max_states_(max_states), lat_out_(lat_out),
      error_(false) {
    bool test = true;
    uint64 props = lat_.Properties(fst::kIDeterministic|fst::kIEpsilons, test);
    if (props != fst::kIDeterministic) {
      KALDI_WARN << "[Lattice has input epsilons and/or is not input-deterministic "
                 << "(in Mohri sense)]-- i.e. lattice is not deterministic.  "
                 << "Word-alignment may be slow and-or blow up in memory.";
    }
    fst::CreateSuperFinal(&lat_); // Creates a super-final state, so the
    // only final-probs are One().

    // Inside this class, we don't want to use zero for the silence
    // or partial-word labels, as this will interfere with the RmEpsilon
    // stage, where we don't want the arcs corresponding to silence or
    // partial words to be removed-- only the arcs with nothing at all
    // on them.
    if (info_.partial_word_label == 0 || info_.silence_label == 0) {
      int32 unused_label = 1 + HighestNumberedOutputSymbol(lat);
      if (info_.partial_word_label >= unused_label)
        unused_label = info_.partial_word_label + 1;
      if (info_.silence_label >= unused_label)
        unused_label = info_.silence_label + 1;
      KALDI_ASSERT(unused_label > 0);
      if (info_.partial_word_label == 0)
        info_.partial_word_label = unused_label++;
      if (info_.silence_label == 0)
        info_.silence_label = unused_label;
    }
  }

  // Removes epsilons; also removes unreachable states...
  // not sure if these would exist if original was connected.
  // This also replaces the temporary symbols for the silence
  // and partial-words, with epsilons, if we wanted epsilons.
  void RemoveEpsilonsFromLattice() {
    // Remove epsilon arcs from output lattice.
    RmEpsilon(lat_out_, true); // true = connect.
    std::vector<int32> syms_to_remove;
    if (info_in_.partial_word_label == 0)
      syms_to_remove.push_back(info_.partial_word_label);
    if (info_in_.silence_label == 0)
      syms_to_remove.push_back(info_.silence_label);
    if (!syms_to_remove.empty()) {
      RemoveSomeInputSymbols(syms_to_remove, lat_out_);
      Project(lat_out_, fst::PROJECT_INPUT);
    }
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

    while (!queue_.empty()) {
      if (max_states_ > 0 && lat_out_->NumStates() > max_states_) {
        KALDI_WARN << "Number of states in lattice exceeded max-states of "
                   << max_states_ << ", original lattice had "
                   << lat_.NumStates() << " states.  Returning what we have.";
        RemoveEpsilonsFromLattice();
        return false;
      }
      ProcessQueueElement();
    }

    RemoveEpsilonsFromLattice();

    return !error_;
  }

  CompactLattice lat_;
  const TransitionInformation &tmodel_;
  const WordBoundaryInfo &info_in_;
  WordBoundaryInfo info_;
  int32 max_states_;
  CompactLattice *lat_out_;

  std::vector<std::pair<Tuple, StateId> > queue_;



  MapType map_; // map from tuples to StateId.
  bool error_;

};

bool LatticeWordAligner::ComputationState::OutputSilenceArc(
    const WordBoundaryInfo &info, const TransitionInformation &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {
  if (transition_ids_.empty()) return false;
  int32 phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  if (info.TypeOfPhone(phone) != WordBoundaryInfo::kNonWordPhone) return false;

  // we assume the start of transition_ids_ is the start of the phone [silence];
  // this is a precondition.
  size_t len = transition_ids_.size(), i;
  // Keep going till we reach a "final" transition-id; note, if
  // reorder==true, we have to go a bit further after this.
  for (i = 0; i < len; i++) {
    int32 tid = transition_ids_[i];
    int32 this_phone = tmodel.TransitionIdToPhone(tid);
    if (this_phone != phone && ! *error) { // error condition: should have reached final transition-id first.
      *error = true;
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
    }
    if (tmodel.IsFinal(tid))
      break;
  }
  if (i == len) return false; // fell off loop.
  i++; // go past the one for which IsFinal returned true.
  if (info.reorder) // we have to consume the following self-loop transition-ids.
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false; // we don't know if it ends here... so can't output arc.

  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != phone
      && ! *error) { // another check.
    KALDI_WARN << "Phone changed unexpectedly in lattice "
        "[broken lattice or mismatched model?]";
  }
  // interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(), transition_ids_.begin()+i);

  // consumed transition ids from our internal state.
  *arc_out = CompactLatticeArc(info.silence_label, info.silence_label,
                               CompactLatticeWeight(weight_, tids_out), fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(), transition_ids_.begin()+i); // delete these
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}


bool LatticeWordAligner::ComputationState::OutputOnePhoneWordArc(
    const WordBoundaryInfo &info, const TransitionInformation &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {
  if (transition_ids_.empty()) return false;
  if (word_labels_.empty()) return false;
  int32 phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  if (info.TypeOfPhone(phone) != WordBoundaryInfo::kWordBeginAndEndPhone)
    return false;
  // we assume the start of transition_ids_ is the start of the phone.
  // this is a precondition.
  size_t len = transition_ids_.size(), i;
  for (i = 0; i < len; i++) {
    int32 tid = transition_ids_[i];
    int32 this_phone = tmodel.TransitionIdToPhone(tid);
    if (this_phone != phone && ! *error) { // error condition: should have reached final transition-id first.
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
      // just continue, ignoring this-- we'll probably output something...
    }
    if (tmodel.IsFinal(tid))
      break;
  }
  if (i == len) return false; // fell off loop.
  i++; // go past the one for which IsFinal returned true.
  if (info.reorder) // we have to consume the following self-loop transition-ids.
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false; // we don't know if it ends here... so can't output arc.

  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != phone
      && ! *error) { // another check.
    KALDI_WARN << "Phone changed unexpectedly in lattice "
        "[broken lattice or mismatched model?]";
    *error = true;
  }

  // interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(),
                              transition_ids_.begin() + i);

  // consumed transition ids from our internal state.
  int32 word = word_labels_[0];
  *arc_out = CompactLatticeArc(word, word,
                               CompactLatticeWeight(weight_, tids_out), fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(),
                        transition_ids_.begin() + i); // delete these
  // Remove the word that we just output.
  word_labels_.erase(word_labels_.begin(), word_labels_.begin() + 1);
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}


/// This function tries to see if it can output a normal word arc--
/// one with at least two phones in it.
bool LatticeWordAligner::ComputationState::OutputNormalWordArc(
    const WordBoundaryInfo &info, const TransitionInformation &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {
  if (transition_ids_.empty()) return false;
  if (word_labels_.empty()) return false;
  int32 begin_phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  if (info.TypeOfPhone(begin_phone) != WordBoundaryInfo::kWordBeginPhone)
    return false;
  // we assume the start of transition_ids_ is the start of the phone.
  // this is a precondition.
  size_t len = transition_ids_.size(), i;

  // Eat up the transition-ids of this word-begin phone until we get to the
  // "final" transition-id.  [there may be self-loops following this though,
  // if reorder==true]
  for (i = 0; i < len && !tmodel.IsFinal(transition_ids_[i]); i++);
  if (i == len) return false;
  i++; // Skip over this final-transition.
  if (info.reorder) // Skip over any reordered self-loops for this final-transition
    for (; i < len && tmodel.IsSelfLoop(transition_ids_[i]); i++);
  if (i == len) return false;
  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != begin_phone
      && ! *error) { // another check.
    KALDI_WARN << "Phone changed unexpectedly in lattice "
        "[broken lattice or mismatched model?]";
    *error = true;
  }
  // Now keep going till we hit a word-ending phone.
  // Note: we don't expect anything except word-internal phones
  // here, but we'll just print a warning if we get something
  // else.
  for (; i < len; i++) {
    int32 this_phone = tmodel.TransitionIdToPhone(transition_ids_[i]);
    if (info.TypeOfPhone(this_phone) == WordBoundaryInfo::kWordEndPhone)
      break;
    if (info.TypeOfPhone(this_phone) != WordBoundaryInfo::kWordInternalPhone
        && !*error) {
      KALDI_WARN << "Unexpected phone " << this_phone
                 << " found inside a word.";
      *error = true;
    }
  }
  if (i == len) return false;

  // OK, we hit a word-ending phone.  Continue till we get to
  // a "final-transition".

  // this variable just used for checks.
  int32 final_phone = tmodel.TransitionIdToPhone(transition_ids_[i]);
  for (; i < len; i++) {
    int32 this_phone = tmodel.TransitionIdToPhone(transition_ids_[i]);
    if (this_phone != final_phone && ! *error) {
      *error = true;
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
    }
    if (tmodel.IsFinal(transition_ids_[i])) break;
  }
  if (i == len) return false;
  i++;
  // We got to the final-transition of the final phone;
  // if reorder==true, continue eating up the self-loop.
  if (info.reorder == true)
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false;
  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != final_phone
      && ! *error) {
    *error = true;
    KALDI_WARN << "Phone changed while following final self-loop "
        "[broken lattice or mismatched model or wrong --reorder option?]";
  }

  // OK, we're ready to output the word.
  // Interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(),
                              transition_ids_.begin() + i);

  // consumed transition ids from our internal state.
  int32 word = word_labels_[0];
  *arc_out = CompactLatticeArc(word, word,
                               CompactLatticeWeight(weight_, tids_out),
                               fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(),
                        transition_ids_.begin() + i); // delete these
  // Remove the word that we just output.
  word_labels_.erase(word_labels_.begin(),
                     word_labels_.begin() + 1);
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}

// Returns true if this vector of transition-ids could be a valid
// word.  Note: the checks are not 100% exhaustive.
static bool IsPlausibleWord(const WordBoundaryInfo &info,
                            const TransitionInformation &tmodel,
                            const std::vector<int32> &transition_ids) {
  if (transition_ids.empty()) return false;
  int32 first_phone = tmodel.TransitionIdToPhone(transition_ids.front()),
      last_phone = tmodel.TransitionIdToPhone(transition_ids.back());
  if ( (info.TypeOfPhone(first_phone) == WordBoundaryInfo::kWordBeginAndEndPhone
        && first_phone == last_phone)
       ||
       (info.TypeOfPhone(first_phone) == WordBoundaryInfo::kWordBeginPhone &&
        info.TypeOfPhone(last_phone) == WordBoundaryInfo::kWordEndPhone) ) {
    if (! info.reorder) {
      return (tmodel.IsFinal(transition_ids.back()));
    } else {
      int32 i = transition_ids.size() - 1;
      while (i > 0 && tmodel.IsSelfLoop(transition_ids[i])) i--;
      return tmodel.IsFinal(transition_ids[i]);
    }
  } else return false;
}


void LatticeWordAligner::ComputationState::OutputArcForce(
    const WordBoundaryInfo &info, const TransitionInformation &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {

  KALDI_ASSERT(!IsEmpty());
  if (!word_labels_.empty()
      && !transition_ids_.empty()) { // We have at least one word to
    // output, and some transition-ids.  We assume that the normal OutputArc was called
    // and failed, so this means we didn't see the end of that
    // word.
    int32 word = word_labels_[0];
    if (! *error && !IsPlausibleWord(info, tmodel, transition_ids_)) {
      *error = true;
      KALDI_WARN << "Invalid word at end of lattice [partial lattice, forced out?]";
    }
    CompactLatticeWeight cw(weight_, transition_ids_);
    *arc_out = CompactLatticeArc(word, word, cw, fst::kNoStateId);
    weight_ = LatticeWeight::One();
    transition_ids_.clear();
    word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1);
  } else if (!word_labels_.empty() && transition_ids_.empty()) {
    // We won't create arcs with these word labels on, as most likely
    // this will cause errors down the road.  This is an error
    // condition anyway, in some sense.
    if (! *error) {
      *error = true;
      KALDI_WARN << "Discarding word-ids at the end of a sentence, "
          "that don't have alignments.";
    }
    CompactLatticeWeight cw(weight_, transition_ids_);
    // This creates an epsilon arc with a weight on it, but
    // no transition-ids since the vector is empty.
    // The word labels are discarded.
    *arc_out = CompactLatticeArc(0, 0, cw, fst::kNoStateId);
    weight_ = LatticeWeight::One();
    word_labels_.clear();
  } else if (!transition_ids_.empty() && word_labels_.empty()) {
    // Transition-ids but no word label-- either silence or partial word.
    int32 first_phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
    if (info.TypeOfPhone(first_phone) == WordBoundaryInfo::kNonWordPhone) {
      // first phone is silence...
      if (first_phone != tmodel.TransitionIdToPhone(transition_ids_.back())
          && ! *error) {
        *error = true;
        // Phone changed-- this is a code error, because the regular OutputArc
        // should have output an arc (a silence arc) if that phone finished.
        // So we make it fatal.
        KALDI_ERR << "Broken silence arc at end of utterance (the phone "
            "changed); code error";
      }
      if (!*error) { // Check that it ends at the end state of silence; error otherwise.
        int32 i = transition_ids_.size() - 1;
        if (info.reorder)
          while (tmodel.IsSelfLoop(transition_ids_[i]) && i > 0)
            i--;
        if (!tmodel.IsFinal(transition_ids_[i])) {
          *error = true;
          KALDI_WARN << "Broken silence arc at end of utterance (does not "
              "reach end of silence)";
        }
      }
      CompactLatticeWeight cw(weight_, transition_ids_);
      *arc_out = CompactLatticeArc(info.silence_label, info.silence_label,
                                   cw, fst::kNoStateId);
    } else {
      // Not silence phone -- treat as partial word (with no word label).
      // This is in itself an error condition, i.e. the lattice was maybe
      // forced out.
      if (! *error) {
        *error = true;
        KALDI_WARN << "Partial word detected at end of utterance";
      }
      CompactLatticeWeight cw(weight_, transition_ids_);
      *arc_out = CompactLatticeArc(info.partial_word_label, info.partial_word_label,
                                   cw, fst::kNoStateId);
    }
    transition_ids_.clear();
    weight_ = LatticeWeight::One();
  } else {
    KALDI_ERR << "Code error, word-aligning lattice"; // this shouldn't
    // be able to happen; we don't call this function of they're both empty.
  }
}

// This code will eventually be removed.
void WordBoundaryInfo::SetOptions(const std::string int_list, PhoneType phone_type) {
  KALDI_ASSERT(!int_list.empty() && phone_type != kNoPhone);
  std::vector<int32> phone_list;
  if (!kaldi::SplitStringToIntegers(int_list, ":",
                                    false,
                                    &phone_list)
      || phone_list.empty())
    KALDI_ERR << "Invalid argument to --*-phones option: " << int_list;
  for (size_t i= 0; i < phone_list.size(); i++) {
    if (phone_to_type.size() <= phone_list[i])
      phone_to_type.resize(phone_list[i]+1, kNoPhone);
    if (phone_to_type[phone_list[i]] != kNoPhone)
      KALDI_ERR << "Phone " << phone_list[i] << "was given two incompatible "
          "assignments.";
    phone_to_type[phone_list[i]] = phone_type;
  }
}

// This initializer will be deleted eventually.
WordBoundaryInfo::WordBoundaryInfo(const WordBoundaryInfoOpts &opts) {
  SetOptions(opts.wbegin_phones, kWordBeginPhone);
  SetOptions(opts.wend_phones, kWordEndPhone);
  SetOptions(opts.wbegin_and_end_phones, kWordBeginAndEndPhone);
  SetOptions(opts.winternal_phones, kWordInternalPhone);
  SetOptions(opts.silence_phones, (opts.silence_has_olabels ?
                                   kWordBeginAndEndPhone : kNonWordPhone));
  reorder = opts.reorder;
  silence_label = opts.silence_label;
  partial_word_label = opts.partial_word_label;
}

WordBoundaryInfo::WordBoundaryInfo(const WordBoundaryInfoNewOpts &opts) {
  reorder = opts.reorder;
  silence_label = opts.silence_label;
  partial_word_label = opts.partial_word_label;
}

WordBoundaryInfo::WordBoundaryInfo(const WordBoundaryInfoNewOpts &opts,
                                   std::string word_boundary_file) {
  reorder = opts.reorder;
  silence_label = opts.silence_label;
  partial_word_label = opts.partial_word_label;
  bool binary_in;
  Input ki(word_boundary_file, &binary_in);
  KALDI_ASSERT(!binary_in && "Not expecting binary word-boundary file.");
  Init(ki.Stream());
}

void WordBoundaryInfo::Init(std::istream &stream) {
  std::string line;
  while (std::getline(stream, line)) {
    std::vector<std::string> split_line;
    SplitStringToVector(line, " \t\r", true, &split_line);// split the line by space or tab
    int32 p = 0;
    if (split_line.size() != 2 ||
        !ConvertStringToInteger(split_line[0], &p))
      KALDI_ERR << "Invalid line in word-boundary file: " << line;
    KALDI_ASSERT(p > 0);
    if (phone_to_type.size() <= static_cast<size_t>(p))
      phone_to_type.resize(p+1, kNoPhone);
    std::string t = split_line[1];
    if (t == "nonword") phone_to_type[p] = kNonWordPhone;
    else if (t == "begin") phone_to_type[p] = kWordBeginPhone;
    else if (t == "singleton") phone_to_type[p] = kWordBeginAndEndPhone;
    else if (t == "end") phone_to_type[p] = kWordEndPhone;
    else if (t == "internal") phone_to_type[p] = kWordInternalPhone;
    else
      KALDI_ERR << "Invalid line in word-boundary file: " << line;
  }
  if (phone_to_type.empty())
    KALDI_ERR << "Empty word-boundary file";
}

bool WordAlignLattice(const CompactLattice &lat,
                      const TransitionInformation &tmodel,
                      const WordBoundaryInfo &info,
                      int32 max_states,
                      CompactLattice *lat_out) {
  LatticeWordAligner aligner(lat, tmodel, info, max_states, lat_out);
  return aligner.AlignLattice();
}



class WordAlignedLatticeTester {
 public:
  WordAlignedLatticeTester(const CompactLattice &lat,
                           const TransitionInformation &tmodel,
                           const WordBoundaryInfo &info,
                           const CompactLattice &aligned_lat):
      lat_(lat), tmodel_(tmodel), info_(info), aligned_lat_(aligned_lat) { }

  void Test() {
    // First test that each aligned arc is valid.
    typedef CompactLattice::StateId StateId ;
    for (StateId s = 0; s < aligned_lat_.NumStates(); s++) {
      for (fst::ArcIterator<CompactLattice> iter(aligned_lat_, s);
           !iter.Done();
           iter.Next()) {
        TestArc(iter.Value());
      }
      if (aligned_lat_.Final(s) != CompactLatticeWeight::Zero()) {
        TestFinal(aligned_lat_.Final(s));
      }
    }
    TestEquivalent();
  }
 private:
  void TestArc(const CompactLatticeArc &arc) {
    if (! (TestArcSilence(arc) || TestArcNormalWord(arc) || TestArcOnePhoneWord(arc)
           || TestArcEmpty(arc)))
      KALDI_ERR << "Invalid arc in aligned CompactLattice: "
                << arc.ilabel << " " << arc.olabel << " " << arc.nextstate
                << " " << arc.weight;
  }
  bool TestArcEmpty(const CompactLatticeArc &arc) {
    if (arc.ilabel != 0) return false; // Check there is no label.  Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    return tids.empty();
  }
  bool TestArcSilence(const CompactLatticeArc &arc) {
    // This only applies when silence doesn't have word labels.
    if (arc.ilabel !=  info_.silence_label) return false; // Check the label is
    // the silence label. Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    int32 first_phone = tmodel_.TransitionIdToPhone(tids.front());
    if (info_.TypeOfPhone(first_phone) != WordBoundaryInfo::kNonWordPhone)
      return false;
    for (size_t i = 0; i < tids.size(); i++)
      if (tmodel_.TransitionIdToPhone(tids[i]) != first_phone) return false;

    if (!info_.reorder) return tmodel_.IsFinal(tids.back());
    else {
      for (size_t i = 0; i < tids.size(); i++) {
        if (tmodel_.IsFinal(tids[i])) { // got the "final" transition, which is
          // reordered to actually not be final.  Make sure that all the
          // rest of the transition ids are the self-loop of that same
          // transition-state.
          for (size_t j = i+1; j < tids.size(); j++) {
              if (!tmodel_.TransitionIdsEquivalent(tids[j], tids[i])) return false;
          }
          return true;
        }
      }
      return false; // fell off loop.  No final-state present.
    }
  }

  bool TestArcOnePhoneWord(const CompactLatticeArc &arc) {
    if (arc.ilabel == 0) return false; // Check there's a label.  Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    int32 first_phone = tmodel_.TransitionIdToPhone(tids.front());
    if (info_.TypeOfPhone(first_phone) !=
        WordBoundaryInfo::kWordBeginAndEndPhone) return false;
    for (size_t i = 0; i < tids.size(); i++)
      if (tmodel_.TransitionIdToPhone(tids[i]) != first_phone) return false;

    if (!info_.reorder) return tmodel_.IsFinal(tids.back());
    else {
      for (size_t i = 0; i < tids.size(); i++) {
        if (tmodel_.IsFinal(tids[i])) { // got the "final" transition, which is
          // reordered to actually not be final.  Make sure that all the
          // rest of the transition ids are the self-loop of that same
          // transition-state.
          for (size_t j = i+1; j < tids.size(); j++) {
            if (!tmodel_.TransitionIdsEquivalent(tids[j], tids[i])) return false;
          }
          return true;
        }
      }
      return false; // fell off loop.  No final-state present.
    }
  }

  bool TestArcNormalWord(const CompactLatticeArc &arc) {
    if (arc.ilabel == 0) return false; // Check there's a label.  Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    int32 first_phone = tmodel_.TransitionIdToPhone(tids.front());
    if (info_.TypeOfPhone(first_phone) != WordBoundaryInfo::kWordBeginPhone)
      return false;
    size_t i;
    { // first phone.
      int num_final = 0;
      for (i = 0; i < tids.size(); i++) {
        if (tmodel_.TransitionIdToPhone(tids[i]) != first_phone) break;
        if (tmodel_.IsFinal(tids[i])) num_final++;
      }
      if (num_final != 1)
        return false; // Something went wrong-- perhaps we
      // got two beginning phones in a row.
    }
    { // middle phones.  Skip over them.
      while (i < tids.size() &&
             info_.TypeOfPhone(tmodel_.TransitionIdToPhone(tids[i]))
             == WordBoundaryInfo::kWordInternalPhone)
        i++;
    }
    if (i == tids.size()) return false;
    int32 final_phone = tmodel_.TransitionIdToPhone(tids[i]);
    if (info_.TypeOfPhone(final_phone) != WordBoundaryInfo::kWordEndPhone)
      return false; // not word-ending.
    for (size_t j = i; j < tids.size(); j++) // make sure only this final phone till end.
      if (tmodel_.TransitionIdToPhone(tids[j]) != final_phone)
        return false; // Other phones after final phone.

    for (size_t j = i; j < tids.size(); j++) {
      if (tmodel_.IsFinal(tids[j])) { // Found "final transition"..   Note:
        // may be "reordered" with its self loops.
        if (!info_.reorder) return (j+1 == tids.size());
        else {
          // Make sure the only thing that follows this is self-loops
          // of the final transition-state.
          for (size_t k = j + 1; k < tids.size(); k++)
              if (!tmodel_.TransitionIdsEquivalent(tids[k], tids[j])
                  || !tmodel_.IsSelfLoop(tids[k]))
              return false;
          return true;
        }
      }
    }
    return false; // Found no final state.
  }

  bool TestArcPartialWord(const CompactLatticeArc &arc) {
    if (arc.ilabel != info_.partial_word_label) return false; // label should
    // be the partial-word label.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    return true; // We're pretty liberal when it comes to partial words here.
  }

  void TestFinal(const CompactLatticeWeight &w) {
    if (!w.String().empty())
      KALDI_ERR << "Expect to have no strings on final-weights of lattices.";
  }
  void TestEquivalent() {
    CompactLattice aligned_lat(aligned_lat_);
    if (info_.silence_label != 0) { // remove silence labels.
      std::vector<int32> to_remove;
      to_remove.push_back(info_.silence_label);
      RemoveSomeInputSymbols(to_remove, &aligned_lat);
      Project(&aligned_lat, fst::PROJECT_INPUT);
    }

    if (!RandEquivalent(lat_, aligned_lat, 5/*paths*/, 1.0e+10/*delta*/, Rand()/*seed*/,
                        200/*path length (max?)*/))
      KALDI_ERR << "Equivalence test failed (testing word-alignment of lattices.) "
                << "Make sure your model and lattices match!";
  }

  const CompactLattice &lat_;
  const TransitionInformation &tmodel_;
  const WordBoundaryInfo &info_;
  const CompactLattice &aligned_lat_;
};




/// You should only test a lattice if WordAlignLattice returned true (i.e. it
/// succeeded and it wasn't a forced-out lattice); otherwise the test will most
/// likely fail.
void TestWordAlignedLattice(const CompactLattice &lat,
                            const TransitionInformation &tmodel,
                            const WordBoundaryInfo &info,
                            const CompactLattice &aligned_lat) {
  WordAlignedLatticeTester t(lat, tmodel, info, aligned_lat);
  t.Test();
}






}  // namespace kaldi
