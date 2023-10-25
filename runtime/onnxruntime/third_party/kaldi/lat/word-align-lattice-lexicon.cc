// lat/word-align-lattice-lexicon.cc

// Copyright 2013 Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/word-align-lattice-lexicon.h"
#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/stl-utils.h"

namespace kaldi {

const int kTemporaryEpsilon = -2;
const int kNumStatesOffset = 1000; // relates to how we apply the
// max-states to the lattices; relates to the --max-expand option which
// stops this blowing up for pathological cases or in case of a mismatch.

class LatticeLexiconWordAligner {
 public:
  typedef CompactLatticeArc::StateId StateId;
  typedef CompactLatticeArc::Label Label;
  typedef WordAlignLatticeLexiconInfo::ViabilityMap ViabilityMap;
  typedef WordAlignLatticeLexiconInfo::LexiconMap LexiconMap;
  typedef WordAlignLatticeLexiconInfo::NumPhonesMap NumPhonesMap;

  /*
    The Freshness enum is applied to phone and word-sequences in the computation
    state; it is related to the epsilon sequencing problem.  If a phone or word
    is new (added by the latest transition), it is fresh.  We are only concerned
    with the freshness of the left-most word (i.e. word index 0) in words_, and
    the freshness of that can take only two values, kNotFresh or kFresh.  As
    regards the phones_ variable, the difference between kFresh and kAllFresh
    is: if we just appended a phone it's kFresh, but if we just shifted off a
    phone or phones by outputting a nonempty word it's kAllFresh, meaning that
    all sub-sequences of the phone sequence are new.  Note: if a phone or
    word-sequence is empty the freshness of that sequence does not matter or is
    not defined; we'll let it default to kNotFresh.
   */
  typedef enum {
    kNotFresh,
    kFresh,
    kAllFresh
  } Freshness;

  class ComputationState {
    /// The state of the computation in which,
    /// along a single path in the lattice, we work out the word
    /// boundaries and output aligned arcs.
   public:

    /// Advance the computation state by adding the symbols and weights from
    /// this arc.  Outputs weight to "leftover_weight" and sets the weight to
    /// 1.0 (this helps keep the state space small).  Note: because we
    /// previously did PhoneAlignLattice, we can assume this arc corresponds to
    /// exactly one or zero phones.
    void Advance(const CompactLatticeArc &arc,
                 const TransitionInformation &tmodel,
                 LatticeWeight *leftover_weight);

    /// Returns true if, assuming we were to add one or more phones by calling
    /// Advance one or more times on this, we might be able later to
    /// successfully call TakeTransition.  It's a kind of co-accessibility test
    /// that avoids us creating an exponentially large number of states that
    /// would contribute nothing to the final output.
    bool ViableIfAdvanced(const ViabilityMap &viability_map) const;

    int32 NumPhones() const { return phones_.size(); }
    int32 NumWords() const { return words_.size(); }
    int32 PendingWord() const { KALDI_ASSERT(!words_.empty()); return words_[0]; }
    Freshness WordFreshness() const { return word_fresh_; }
    Freshness PhoneFreshness() const { return phone_fresh_; }

    /// This may be called at the end of a lattice, if it was forced
    /// out.  Note: we will only use "partial_word_label" if there are
    /// phones without corresponding words; otherwise we'll use the
    /// word label that was there.
    void TakeForcedTransition(int32 partial_word_label,
                              ComputationState *next_state,
                              CompactLatticeArc *arc_out) const;

    /// Take a transition, if possible; consume "num_phones" phones and (if
    /// word_id != 0) the word "word_id" which must be the first word in words_.
    /// Returns true if we could take the transition.
    bool TakeTransition(const LexiconMap &lexicon_map,
                        int32 word_id,
                        int32 num_phones,
                        ComputationState *next_state,
                        CompactLatticeArc *arc_out) const;

    bool IsEmpty() const { return (transition_ids_.empty() && words_.empty()); }

    /// FinalWeight() will return "weight" if both transition_ids
    /// and word_labels are empty, otherwise it will return
    /// Weight::Zero().
    LatticeWeight FinalWeight() const {
      return (IsEmpty() ? weight_ : LatticeWeight::Zero());
    }

    size_t Hash() const {
      VectorHasher<int32> vh;
      const int32 p1 = 11117, p2 = 90647, p3 = 3967, p4 = 3557; // primes.
      size_t ans = 0;
      for (int32 i = 0; i < static_cast<int32>(transition_ids_.size()); i++) {
        ans *= p1;
        ans += vh(transition_ids_[i]);
      }
      ans += p2 * vh(words_)
          + static_cast<int32>(word_fresh_) * p3
          + static_cast<int32>(phone_fresh_) * p4;
      // phones_ is determined by transition-id sequence so we don't
      // need to include it in the hash.
      return ans;
    }

    bool operator == (const ComputationState &other) const {
      // phones_ is determined by transition-id sequence so don't
      // need to compare it.
      return (transition_ids_ == other.transition_ids_ &&
              words_ == other.words_ &&
              weight_ == other.weight_ &&
              phone_fresh_ == other.phone_fresh_ &&
              word_fresh_ == other.word_fresh_);
    }

    ComputationState(): phone_fresh_(kNotFresh), word_fresh_(kNotFresh),
                        weight_(LatticeWeight::One()) { } // initial state.

    ComputationState(const ComputationState &other):
        phones_(other.phones_), words_(other.words_),
        phone_fresh_(other.phone_fresh_), word_fresh_(other.word_fresh_),
        transition_ids_(other.transition_ids_), weight_(other.weight_) { }
   private:
    std::vector<int32> phones_; // sequence of pending phones
    std::vector<int32> words_; // sequence of pending words.

    // The following variables tell us whether the phones_ and/or words_
    // variables were modified by the last operation on the computation state.
    // This is used to make sure we don't have multiple ways of handling the
    // same sequence, by taking transitions at multiple points (see code for
    // more details).  It's related to the epsilon sequencing problem.
    // See the declaration above of the enum "Freshness".
    Freshness phone_fresh_;
    Freshness word_fresh_;

    std::vector<std::vector<int32> > transition_ids_; // sequence of transition-ids for each phone..

    LatticeWeight weight_; // contains two floats.
  };


  static void AppendVectors(
      std::vector<std::vector<int32> >::const_iterator input_begin,
      std::vector<std::vector<int32> >::const_iterator input_end,
      std::vector<int32> *output);

  struct Tuple {
    Tuple(StateId input_state, ComputationState comp_state):
        input_state(input_state), comp_state(comp_state) {}
    Tuple() {}
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

  // This function may alter queue_.
  StateId GetStateForTuple(const Tuple &tuple) {
    MapType::iterator iter = map_.find(tuple);
    if (iter == map_.end()) { // not in map.
      StateId output_state = lat_out_->AddState();
      map_[tuple] = output_state;
      queue_.push_back(std::make_pair(tuple, output_state));
      return output_state;
    } else {
      return iter->second;
    }
  }

  // This function may alter queue_, via GetStateForTuple.
  void ProcessTransition(StateId prev_output_state, // state-id of from-state in output lattice
                         const Tuple &next_tuple,
                         CompactLatticeArc *arc) { // arc to add (must first modify it by adding "nextstate")
    arc->nextstate = GetStateForTuple(next_tuple); // adds it to queue_ if new.
    lat_out_->AddArc(prev_output_state, *arc);
  }

  // Process any epsilon transitions out of this state.  This refers to
  // filler-words, such as silence, which have epsilon as the symbol in the
  // original lattice, or no symbol at all (typically the original lattice
  // will be determinized with epsilon-removal so there is no separate arc,
  // just one or more extra phones that don't match up with any word.
  void ProcessEpsilonTransitions(const Tuple &tuple, StateId output_state);

  // Process any non-epsilon transitions out of this state in the output lattice.
  void ProcessWordTransitions(const Tuple &tuple, StateId output_state);

  // Take any transitions that correspond to advancing along arcs arc in the
  // original FST.
  void PossiblyAdvanceArc(const Tuple &tuple, StateId output_state);

  /// Process all final-probs (normal case, no forcing-out).
  /// returns true if we had at least one final-prob.
  bool ProcessFinal();

  /// This function returns true if the state "output_state" in the output
  /// lattice has arcs out that have either a non-epsilon symbol or transition-ids
  /// in the string of the weight.
  bool HasNonEpsArcsOut(StateId output_state);

  /// Creates arcs from all the tuples that were final in the original lattice
  /// but have no arcs out of them in the output lattice that consume words or
  /// phones-- does so by "forcing out" any words and phones there are pending
  /// in the computation states.  This function is only called if no states were
  /// "naturally" final; this will only happen for lattices that were forced out
  /// during decoding.
  void ProcessFinalForceOut();

  // Process all final-probs -- a wrapper function that handles the forced-out case.
  void ProcessFinalWrapper() {
    if (final_queue_.empty()) {
      KALDI_WARN << "No final-probs to process.";
      error_ = true;
      return;
    }
    if (ProcessFinal()) return;
    error_ = true;
    KALDI_WARN << "Word-aligning lattice: lattice was forced out, will have partial words at end.";

    ProcessFinalForceOut();

    if (ProcessFinal()) return;
    KALDI_WARN << "Word-aligning lattice: had no final-states even after forcing out "
               << "(result will be empty).  This probably indicates wrong input.";
    return;
  }

  void ProcessQueueElement() {
    KALDI_ASSERT(!queue_.empty());
    Tuple tuple = queue_.back().first;
    StateId output_state = queue_.back().second;
    queue_.pop_back();

    ProcessEpsilonTransitions(tuple, output_state);
    ProcessWordTransitions(tuple, output_state);
    PossiblyAdvanceArc(tuple, output_state);

    // Note: we'll do a bit more filtering in ProcessFinal(), meaning
    // that we won't necessarily give a final-prob to all of the things
    // that go onto final_queue_.
    if (lat_in_.Final(tuple.input_state) != CompactLatticeWeight::Zero())
      final_queue_.push_back(std::make_pair(tuple, output_state));
  }

  LatticeLexiconWordAligner(const CompactLattice &lat,
                            const TransitionInformation &tmodel,
                            const WordAlignLatticeLexiconInfo &lexicon_info,
                            int32 max_states,
                            int32 partial_word_label,
                            CompactLattice *lat_out):
      lat_in_(lat), tmodel_(tmodel), lexicon_info_(lexicon_info),
      max_states_(max_states),
      lat_out_(lat_out),
      partial_word_label_(partial_word_label == 0 ?
                          kTemporaryEpsilon : partial_word_label),
      error_(false) {
    // lat_in_ is after PhoneAlignLattice, it is not deterministic and contains epsilons

    fst::CreateSuperFinal(&lat_in_); // Creates a super-final state, so the
    // only final-probs are One().  Note: the member lat_in_ is not a reference.

  }

  // Removes epsilons; also removes unreachable states...
  // not sure if these would exist if original was connected.
  // This also replaces the temporary symbols for the silence
  // and partial-words, with epsilons, if we wanted epsilons.
  void RemoveEpsilonsFromLattice() {
    Connect(lat_out_);
    RmEpsilon(lat_out_, true); // true = connect.
    std::vector<int32> syms_to_remove;
    syms_to_remove.push_back(kTemporaryEpsilon);
    RemoveSomeInputSymbols(syms_to_remove, lat_out_);
    Project(lat_out_, fst::PROJECT_INPUT);
  }

  bool AlignLattice() {
    lat_out_->DeleteStates();
    if (lat_in_.Start() == fst::kNoStateId) {
      KALDI_WARN << "Trying to word-align empty lattice.";
      return false;
    }
    ComputationState initial_comp_state;
    Tuple initial_tuple(lat_in_.Start(), initial_comp_state);
    StateId start_state = GetStateForTuple(initial_tuple);
    lat_out_->SetStart(start_state);

    while (!queue_.empty()) {
      if (max_states_ > 0 && lat_out_->NumStates() > max_states_) {
        KALDI_WARN << "Number of states in lattice exceeded max-states of "
                   << max_states_ << ", original lattice had "
                   << lat_in_.NumStates() << " states.  Returning empty lattice.";
        lat_out_->DeleteStates();
        return false;
      }
      ProcessQueueElement();
    }
    ProcessFinalWrapper();

    RemoveEpsilonsFromLattice();

    return !error_;
  }

  CompactLattice lat_in_;
  const TransitionInformation &tmodel_;
  const WordAlignLatticeLexiconInfo &lexicon_info_;
  int32 max_states_;
  CompactLattice *lat_out_;

  std::vector<std::pair<Tuple, StateId> > queue_;

  std::vector<std::pair<Tuple, StateId> > final_queue_; // as queue_, but
  // just contains states that may have final-probs to process.  We process these
  // all at once, at the end.

  MapType map_; // map from tuples to StateId.
  int32 partial_word_label_;
  bool error_;
};

// static
void LatticeLexiconWordAligner::AppendVectors(
    std::vector<std::vector<int32> >::const_iterator input_begin,
    std::vector<std::vector<int32> >::const_iterator input_end,
    std::vector<int32> *output) {
  size_t size = 0;
  for (std::vector<std::vector<int32> >::const_iterator iter = input_begin;
       iter != input_end;
       ++iter)
    size += iter->size();
  output->clear();
  output->reserve(size);
  for (std::vector<std::vector<int32> >::const_iterator iter = input_begin;
       iter != input_end;
       ++iter)
    output->insert(output->end(), iter->begin(), iter->end());
}

void LatticeLexiconWordAligner::ProcessEpsilonTransitions(
    const Tuple &tuple, StateId output_state) {
  const ComputationState &comp_state = tuple.comp_state;
  StateId input_state = tuple.input_state;
  StateId zero_word = 0;
  NumPhonesMap::const_iterator iter =
      lexicon_info_.num_phones_map_.find(zero_word);
  if (iter == lexicon_info_.num_phones_map_.end()) {
    return; // No epsilons to match; this can only happen if the lexicon
    // we were provided had no lines with 0 as the first entry, i.e.  no
    // optional silences or the like.
  }
  // Now decide what range of phone-lengths we must process.  This is all
  // about only getting a single opportunity to process any given sequence of
  // phones.
  int32 min_num_phones, max_num_phones;

  if (comp_state.PhoneFreshness() == kAllFresh) {
    // All sub-sequences of the phone sequence are fresh because we just
    // shifted some phones off, so we do this for all lengths.  We can limit
    // ourselves to the range of possible lengths for the epsilon symbol,
    // in the lexicon.
    min_num_phones = iter->second.first;
    max_num_phones = std::min(iter->second.second, comp_state.NumPhones());
  } else if (comp_state.PhoneFreshness() == kFresh) {
    // only last phone is "fresh", so only consider the sequence of all
    // phones including the last one.
    int32 num_phones = comp_state.NumPhones();
    if (num_phones >= iter->second.first &&
        num_phones <= iter->second.second) {
      min_num_phones = num_phones;
      max_num_phones = num_phones;
    } else {
      return;
    }
  } else { // kNotFresh
    return;
  }

  if (min_num_phones == 0)
    KALDI_ERR << "Lexicon error: epsilon transition that produces no output:";

  for (int32 num_phones = min_num_phones;
       num_phones <= max_num_phones;
       num_phones++) {
    Tuple next_tuple;
    next_tuple.input_state = input_state; // We're not taking a transition in the
    // input FST so this stays the same.
    CompactLatticeArc arc;
    if (comp_state.TakeTransition(lexicon_info_.lexicon_map_,
                                  zero_word,
                                  num_phones,
                                  &next_tuple.comp_state,
                                  &arc)) {
      ProcessTransition(output_state, next_tuple, &arc);
    }
  }
}

void LatticeLexiconWordAligner::ProcessWordTransitions(
    const Tuple &tuple, StateId output_state) {
  const ComputationState &comp_state = tuple.comp_state;
  StateId input_state = tuple.input_state;
  if (comp_state.NumWords() > 0) {
    int32 min_num_phones, max_num_phones;
    int32 word_id = comp_state.PendingWord();

    if (comp_state.WordFreshness() == kFresh ||
        comp_state.PhoneFreshness() == kAllFresh) {
      // Just saw word, or shifted phones,
      // so 1st opportunity to process phone-sequences of all possible sizes,
      // with this word.
      NumPhonesMap::const_iterator iter =
          lexicon_info_.num_phones_map_.find(word_id);
      if (iter == lexicon_info_.num_phones_map_.end()) {
        KALDI_ERR << "Word " << word_id << " is not present in the lexicon.";
      }
      min_num_phones = iter->second.first;
      max_num_phones = std::min(iter->second.second,
                                comp_state.NumPhones());
    } else if (comp_state.PhoneFreshness() == kFresh) {
      // just the latest phone is new -> just try to process the
      // phone-sequence of all the phones we have.
      min_num_phones = comp_state.NumPhones();
      max_num_phones = min_num_phones;
    } else {
      return; // Nothing to do, since neither the word nor the phones are fresh.
    }

    for (int32 num_phones = min_num_phones;
         num_phones <= max_num_phones;
         num_phones++) {
      Tuple next_tuple;
      next_tuple.input_state = input_state; // We're not taking a transition in the
      // input FST so this stays the same.
      CompactLatticeArc arc;
      if (comp_state.TakeTransition(lexicon_info_.lexicon_map_,
                                    word_id,
                                    num_phones,
                                    &next_tuple.comp_state,
                                    &arc)) {
        ProcessTransition(output_state, next_tuple, &arc);
      }
    }
  }
}


void LatticeLexiconWordAligner::PossiblyAdvanceArc(
    const Tuple &tuple, StateId output_state) {
  if (tuple.comp_state.ViableIfAdvanced(lexicon_info_.viability_map_)) {
    for(fst::ArcIterator<CompactLattice> aiter(lat_in_, tuple.input_state);
        !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc_in = aiter.Value();
      Tuple next_tuple(arc_in.nextstate, tuple.comp_state);
      LatticeWeight arc_weight;
      next_tuple.comp_state.Advance(arc_in, tmodel_, &arc_weight);
      // Note: GetStateForTuple will add the tuple to the queue,
      // if necessary.

      StateId next_output_state = GetStateForTuple(next_tuple);
      CompactLatticeArc arc_out(0, 0,
                                CompactLatticeWeight(arc_weight,
                                                     std::vector<int32>()),
                                next_output_state);
      lat_out_->AddArc(output_state,
                       arc_out);
    }
  }
}

bool LatticeLexiconWordAligner::ProcessFinal() {
  bool saw_final = false;
  // Find final-states...
  for (size_t i = 0; i < final_queue_.size(); i++) {
    const Tuple &tuple = final_queue_[i].first;
    StateId output_state = final_queue_[i].second;
    KALDI_ASSERT(lat_in_.Final(tuple.input_state) == CompactLatticeWeight::One());
    LatticeWeight final_weight = tuple.comp_state.FinalWeight();
    if (final_weight != LatticeWeight::Zero()) {
      // note: final_weight is only nonzero if there are no
      // pending transition-ids, so there is no string component.
      std::vector<int32> empty_vec;
      lat_out_->SetFinal(output_state,
                         CompactLatticeWeight(final_weight, empty_vec));
      saw_final = true;
    }
  }
  return saw_final;
}

bool LatticeLexiconWordAligner::HasNonEpsArcsOut(StateId output_state) {
  for (fst::ArcIterator<CompactLattice> aiter(*lat_out_, output_state);
       !aiter.Done(); aiter.Next()) {
    const CompactLatticeArc &arc = aiter.Value();
    if (arc.ilabel != 0 || arc.olabel != 0 || !arc.weight.String().empty())
      return true;
  }
  return false;
}

void LatticeLexiconWordAligner::ProcessFinalForceOut() {
  KALDI_ASSERT(queue_.empty());
  std::vector<std::pair<Tuple, StateId> > new_final_queue_;
  new_final_queue_.reserve(final_queue_.size());
  for (size_t i = 0; i < final_queue_.size();i++) { // note: all the states will
    // be final in the orig. lattice
    const Tuple &tuple = final_queue_[i].first;
    StateId output_state = final_queue_[i].second;

    if (!HasNonEpsArcsOut(output_state)) { // This if-statement
      // avoids forcing things out too early, when they had words
      // that could naturally have been put out.  [without it,
      // we'd have multiple alternate paths at the end.]

      CompactLatticeArc arc;
      Tuple next_tuple;
      next_tuple.input_state = tuple.input_state;
      tuple.comp_state.TakeForcedTransition(partial_word_label_,
                                            &next_tuple.comp_state,
                                            &arc);
      // Note: the following call may add to queue_, but we'll clear it,
      // we don't want to process these states.
      StateId new_state = GetStateForTuple(next_tuple);
      arc.nextstate = new_state;
      lat_out_->AddArc(output_state, arc);
      new_final_queue_.push_back(std::make_pair(next_tuple, new_state));
    }
  }
  queue_.clear();
  std::swap(final_queue_, new_final_queue_);
}

void LatticeLexiconWordAligner::ComputationState::Advance(
    const CompactLatticeArc &arc, const TransitionInformation &tmodel,
    LatticeWeight *weight) {
  const std::vector<int32> &tids = arc.weight.String();
  int32 phone;
  if (tids.empty()) phone = 0;
  else {
    phone = tmodel.TransitionIdToPhone(tids.front());
    KALDI_ASSERT(phone == tmodel.TransitionIdToPhone(tids.back()) &&
                 "Error: lattice is not phone-aligned.");
  }
  if (arc.ilabel != 0) { // note: arc.ilabel==arc.olabel (acceptor)
    words_.push_back(arc.ilabel);
    // Note: the word freshness only applies to the word in position 0,
    // so only if the word-sequence is now of size 1, is it fresh.
    if (words_.size() == 1) word_fresh_ = kFresh;
    else word_fresh_ = kNotFresh;
  } else { // No word added -> word not fresh.
    word_fresh_ = kNotFresh;
  }
  if (phone != 0) {
    phones_.push_back(phone);
    transition_ids_.push_back(tids);
    phone_fresh_ = kFresh;
  } else {
    phone_fresh_ = kNotFresh;
  }
  *weight = Times(weight_, arc.weight.Weight()); // will go on arc in output lattice
  weight_ = LatticeWeight::One();
}


bool LatticeLexiconWordAligner::ComputationState::ViableIfAdvanced(
    const ViabilityMap &viability_map) const {
  /* This will ideally to return true if and only if we can ever take
     any kind of transition out of this state after "advancing" it by adding
     words and/or phones.  It's OK to return true in some cases where the
     condition is false, though, if it's a pain to check, because the result
     will just be doing extra work for nothing (those states won't be
     co-accessible in the output).
  */
  if (phones_.empty()) return true;
  if (words_.empty()) return true;
  else {
    // neither phones_ or words_ is empty.  Return true if a longer sequence
    // than this phone sequence can have either zero (<eps>/epsilon) or the
    // first element of words_, as an entry in the lexicon with that phone
    // sequence.
    ViabilityMap::const_iterator iter = viability_map.find(phones_);
    if (iter == viability_map.end()) return false;
    else {
      const std::vector<int32> &this_set = iter->second; // sorted vector.
      // Return true if either 0 or words_[0] is in the set.  If 0 is
      // in the set, it will be the 1st element of the vector, because it's
      // the lowest element.
      return (this_set.front() == 0 ||
              std::binary_search(this_set.begin(), this_set.end(), words_[0]));
    }
  }
}


void LatticeLexiconWordAligner::ComputationState::TakeForcedTransition(
    int32 partial_word_label,
    ComputationState *next_state,
    CompactLatticeArc *arc_out) const {
  KALDI_ASSERT(!IsEmpty());

  next_state->phones_.clear();
  next_state->words_.clear();
  next_state->transition_ids_.clear();
  // neither of the following variables should matter, actually,
  // they will never be inspected.  So just set them to kFresh for consistency,
  // so they end up at the same place in the tuple-map_.
  next_state->word_fresh_ = kFresh;
  next_state->phone_fresh_ = kFresh;
  next_state->weight_ = LatticeWeight::One();

  int32 word_id;
  if (words_.size() >= 1) {
    word_id = words_[0];
    if (words_.size() > 1)
      KALDI_WARN << "Word-aligning lattice: discarding extra word at end of lattice"
                 << "(forced-out).";
  } else {
    word_id = partial_word_label;
  }
  KALDI_ASSERT(word_id != 0);  // any zeros would have been replaced with
                               // 'temporary epsilon' = 2.
  std::vector<int32> appended_transition_ids;
  AppendVectors(transition_ids_.begin(),
                transition_ids_.end(),
                &appended_transition_ids);
  arc_out->ilabel = word_id;
  arc_out->olabel = word_id;
  arc_out->weight = CompactLatticeWeight(weight_,
                                         appended_transition_ids);
  // arc_out->nextstate will be set by the calling code.
}


bool LatticeLexiconWordAligner::ComputationState::TakeTransition(
    const LexiconMap &lexicon_map, int32 word_id, int32 num_phones,
    ComputationState *next_state, CompactLatticeArc *arc_out) const {
  KALDI_ASSERT(word_id == 0 || (!words_.empty() && word_id == words_[0]));
  KALDI_ASSERT(num_phones <= static_cast<int32>(phones_.size()));

  std::vector<int32> lexicon_key;
  lexicon_key.reserve(1 + num_phones);
  lexicon_key.push_back(word_id); // put 1st word in lexicon_key.
  lexicon_key.insert(lexicon_key.end(),
                     phones_.begin(), phones_.begin() + num_phones);
  LexiconMap::const_iterator iter = lexicon_map.find(lexicon_key);
  if (iter == lexicon_map.end()) { // no such entry
    return false;
  } else { // Entry exists.  We'll create an arc.
    next_state->phones_.assign(phones_.begin() + num_phones, phones_.end());
    next_state->words_.assign(words_.begin() + (word_id == 0 ? 0 : 1),
                              words_.end());
    next_state->transition_ids_.assign(transition_ids_.begin() + num_phones,
                                       transition_ids_.end());
    next_state->word_fresh_ =
        (word_id != 0 && !next_state->words_.empty()) ? kFresh : kNotFresh;
    next_state->phone_fresh_ =
        (next_state->phones_.empty() || num_phones == 0) ? kNotFresh : kAllFresh;

    // this next thing is a bit hard to explain.  If we just consumed a word with
    // no phones, we treat the phones as fresh.  The idea is that if we need to
    // both consume a word with no phones and a phone with no words (e.g.
    // an empty word and then silence), we need to have the phones marked
    // as fresh in order for this to be possible.
    if (num_phones == 0 && word_id != 0 && !next_state->phones_.empty())
      next_state->phone_fresh_ = kAllFresh;

    next_state->weight_ = LatticeWeight::One();

    if (GetVerboseLevel() >= 5) {
      std::ostringstream ostr;
      for (size_t i = 0; i < num_phones; i++)
        ostr << phones_[i] << " ";
      KALDI_VLOG(5) << "Taking arc with word = " << word_id
                    << " and phones = " << ostr.str()
                    << ", output-word = " << iter->second
                    << ", dest-state has num-words = " << next_state->words_.size()
                    << " and num-phones = " << next_state->phones_.size();
    }

    // Set arc_out:
    Label word_id = iter->second; // word_id will typically be
    // the same as words_[0], i.e. the
    // word we consumed.

    KALDI_ASSERT(word_id != 0);  // we replaced zeros with 'temporary epsilon' = -2.

    std::vector<int32> appended_transition_ids;
    AppendVectors(transition_ids_.begin(),
                  transition_ids_.begin() + num_phones,
                  &appended_transition_ids);
    arc_out->ilabel = word_id;
    arc_out->olabel = word_id;
    arc_out->weight = CompactLatticeWeight(weight_,
                                           appended_transition_ids);
    // arc_out->nextstate will be set in the calling code.
    return true;
  }
}

void WordAlignLatticeLexiconInfo::UpdateViabilityMap(
    const std::vector<int32> &lexicon_entry) {
  int32 word = lexicon_entry[0];  // note: word may be zero.
  int32 num_phones = static_cast<int32>(lexicon_entry.size()) - 2;
  std::vector<int32> phones;
  if (num_phones > 0)
    phones.reserve(num_phones - 1);
  // for each nonempty sequence of phones that is a strict prefix of the phones
  // in the lexicon entry (i.e. lexicon_entry [2 ... ]), add the word to the set
  // in viability_map_[phones].
  for (int32 n = 0; n < num_phones - 1; n++) {
    phones.push_back(lexicon_entry[n + 2]); // first phone is at position 2.
    // n+1 is the length of the sequence of phones
    viability_map_[phones].push_back(word);
  }
}

void WordAlignLatticeLexiconInfo::FinalizeViabilityMap() {
  for (ViabilityMap::iterator iter = viability_map_.begin();
       iter != viability_map_.end();
       ++iter) {
    std::vector<int32> &words = iter->second;
    SortAndUniq(&words);
    KALDI_ASSERT(words[0] >= 0 && "Error: negative labels in lexicon.");
  }
}

/// Update the map from a vector (orig-word-symbol phone1 phone2 ... ) to the
/// new word-symbol.  The new word-symbol must always be nonzero; we'll replace
/// it with kTemporaryEpsilon = -2, if it was zero.
void WordAlignLatticeLexiconInfo::UpdateLexiconMap(
    const std::vector<int32> &lexicon_entry) {
  KALDI_ASSERT(lexicon_entry.size() >= 2);
  std::vector<int32> key;
  key.reserve(lexicon_entry.size() - 1);
  // add the original word:
  key.push_back(lexicon_entry[0]);
  // add the phones:
  key.insert(key.end(), lexicon_entry.begin() + 2, lexicon_entry.end());
  int32 new_word = lexicon_entry[1]; // This will typically be the same as
  // the original word at lexicon_entry[0] but is allowed to differ.
  if (new_word == 0) new_word = kTemporaryEpsilon; // replace 0's with -2;
  // we'll revert the change at the end.
  if (lexicon_map_.count(key) != 0) {
    if (lexicon_map_[key] == new_word)
      KALDI_WARN << "Duplicate entry in lexicon map for word " << lexicon_entry[0];
    else
      KALDI_ERR << "Duplicate entry in lexicon map for word " << lexicon_entry[0]
                << " with inconsistent to-word.";
  }
  lexicon_map_[key] = new_word;

  if (lexicon_entry[0] != lexicon_entry[1]) {
    // Add reverse lexicon entry, this time with no 0 -> -2 mapping.
    key[0] = lexicon_entry[1];
    // Note: we ignore the situation where there are conflicting
    // entries in reverse_lexicon_map_, as we never actually inspect
    // the contents so it won't matter.
    reverse_lexicon_map_[key] = lexicon_entry[0];
  }
}

void WordAlignLatticeLexiconInfo::UpdateNumPhonesMap(
    const std::vector<int32> &lexicon_entry) {
  int32 num_phones = static_cast<int32>(lexicon_entry.size()) - 2;
  int32 word = lexicon_entry[0];
  if (num_phones_map_.count(word) == 0)
    num_phones_map_[word] = std::make_pair(num_phones, num_phones);
  else {
    std::pair<int32, int32> &pr = num_phones_map_[word];
    pr.first = std::min(pr.first, num_phones); // update min-num-phones
    pr.second = std::max(pr.second, num_phones); // update max-num-phones
    if (pr.first == 0 && word == 0)
      KALDI_ERR << "Zero word with empty pronunciation is not allowed.";
  }
}

/// Entry contains new-word-id phone1 phone2 ...
/// equivalent to all but the 1st entry on a line of the input file.
bool WordAlignLatticeLexiconInfo::IsValidEntry(const std::vector<int32> &entry) const {
  KALDI_ASSERT(!entry.empty());
  LexiconMap::const_iterator iter = lexicon_map_.find(entry);
  if (iter != lexicon_map_.end()) {
    int32 tgt_word = (iter->second == kTemporaryEpsilon ? 0 : iter->second);
    if (tgt_word == entry[0]) return true; // symmetric entry.
    // this means that that there would be an output-word with this
    // value, and this sequence of phones.
  }
  // For entries that were not symmetric:
  return (reverse_lexicon_map_.count(entry) != 0);
}

int32 WordAlignLatticeLexiconInfo::EquivalenceClassOf(int32 word) const {
  unordered_map<int32, int32>::const_iterator iter =
      equivalence_map_.find(word);
  if (iter == equivalence_map_.end()) return word; // not in map.
  else return iter->second;
}

void WordAlignLatticeLexiconInfo::UpdateEquivalenceMap(
    const std::vector<std::vector<int32> > &lexicon) {
  std::vector<std::pair<int32, int32> > equiv_pairs; // pairs of
  // (lower,higher) words that are equivalent.
  for (size_t i = 0; i < lexicon.size(); i++) {
    KALDI_ASSERT(lexicon[i].size() >= 2);
    int32 w1 = lexicon[i][0], w2 = lexicon[i][1];
    if (w1 == w2) continue; // They are the same; this provides no information
                            // about equivalence, since any word is equivalent
                            // to itself.
    if (w1 > w2) std::swap(w1, w2); // make sure w1 < w2.
    equiv_pairs.push_back(std::make_pair(w1, w2));
  }
  SortAndUniq(&equiv_pairs);
  equivalence_map_.clear();
  for (size_t i = 0; i < equiv_pairs.size(); i++) {
    int32 w1 = equiv_pairs[i].first, w2 = equiv_pairs[i].second,
        w1dash = EquivalenceClassOf(w1);
    equivalence_map_[w2] = w1dash;
  }
}


WordAlignLatticeLexiconInfo::WordAlignLatticeLexiconInfo(
    const std::vector<std::vector<int32> > &lexicon) {
  for (size_t i = 0; i < lexicon.size(); i++) {
    const std::vector<int32> &lexicon_entry = lexicon[i];
    KALDI_ASSERT(lexicon_entry.size() >= 2);
    UpdateViabilityMap(lexicon_entry);
    UpdateLexiconMap(lexicon_entry);
    UpdateNumPhonesMap(lexicon_entry);
  }
  FinalizeViabilityMap();
  UpdateEquivalenceMap(lexicon);
}

// This is the wrapper function for users to call.
bool WordAlignLatticeLexicon(const CompactLattice &lat,
                             const TransitionInformation &tmodel,
                             const WordAlignLatticeLexiconInfo &lexicon_info,
                             const WordAlignLatticeLexiconOpts &opts,
                             CompactLattice *lat_out) {
  PhoneAlignLatticeOptions phone_align_opts;
  phone_align_opts.reorder = opts.reorder;
  phone_align_opts.replace_output_symbols = false;
  phone_align_opts.remove_epsilon = false;

  // Input Lattice should be deterministic and w/o epsilons.
  bool test = true;
  uint64 props = lat.Properties(fst::kIDeterministic|fst::kIEpsilons, test);
  if (props != fst::kIDeterministic) {
    KALDI_WARN << "[Lattice has input epsilons and/or is not input-deterministic "
               << "(in Mohri sense)]-- i.e. lattice is not deterministic.  "
               << "Word-alignment may be slow and-or blow up in memory.";
  }

  CompactLattice phone_aligned_lat;
  bool ans = PhoneAlignLattice(lat, tmodel, phone_align_opts,
                               &phone_aligned_lat);
  // 'phone_aligned_lat' is no longer deterministic and contains epsilons.

  int32 max_states;
  if (opts.max_expand <= 0) {
    max_states = -1;
  } else {
    // The 1000 is a fixed offset to give it more wiggle room for very
    // small inputs.
    max_states = kNumStatesOffset + opts.max_expand * phone_aligned_lat.NumStates();
  }

  // If ans == false, we hope this is due to a forced-out lattice, and we try to
  // continue.
  LatticeLexiconWordAligner aligner(phone_aligned_lat, tmodel, lexicon_info,
                                    max_states, opts.partial_word_label, lat_out);
  // We'll let the calling code warn if this is false; it will know the utterance-id.
  ans = aligner.AlignLattice() && ans;
  return ans;
}

bool ReadLexiconForWordAlign (std::istream &is,
                              std::vector<std::vector<int32> > *lexicon) {
  lexicon->clear();
  std::string line;
  while (std::getline(is, line)) {
    std::vector<int32> this_entry;
    if (!SplitStringToIntegers(line, " \t\r", false, &this_entry) ||
        this_entry.size() < 2) {
      KALDI_WARN << "Lexicon line '" << line  << "' is invalid";
      return false;
    }
    lexicon->push_back(this_entry);
  }
  return (!lexicon->empty());
}

}  // namespace kaldi

