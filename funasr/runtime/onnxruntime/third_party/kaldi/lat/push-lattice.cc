// lat/push-lattice.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli
//                2014  Guoguo Chen

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


#include "lat/push-lattice.h"
#include "util/stl-utils.h"

namespace fst {


template<class Weight, class IntType> class CompactLatticePusher {  
 public:
  typedef CompactLatticeWeightTpl<Weight, IntType> CompactWeight;
  typedef ArcTpl<CompactWeight> CompactArc;
  typedef typename CompactArc::StateId StateId;
  
  CompactLatticePusher(MutableFst<CompactArc> *clat): clat_(clat) { }
  bool Push() {
    if (clat_->Properties(kTopSorted, true) == 0) {
      if (!TopSort(clat_)) {
        KALDI_WARN << "Topological sorting of state-level lattice failed "
            "(probably your lexicon has empty words or your LM has epsilon cycles; this "
            " is a bad idea.)";
        return false;
      }
    }
    ComputeShifts();
    ApplyShifts();
    return true;
  }

  // Gets the string of length [end - begin], starting at this
  // state and taking arc "arc_idx" (and thereafter an arbitrary sequence).
  // Note: here, arc_idx == -1 means take an arbitrary path.
  static void GetString(const ExpandedFst<CompactArc> &clat,
                        StateId state,
                        size_t arc_idx,
                        typename std::vector<IntType>::iterator begin,
                        typename std::vector<IntType>::iterator end) {
    CompactWeight final = clat.Final(state);
    size_t len = end - begin;
    KALDI_ASSERT(len >= 0);
    if (len == 0) return;
    if (arc_idx == -1 && final != CompactWeight::Zero()) {
      const std::vector<IntType> &string = final.String();
      KALDI_ASSERT(string.size() >= len &&
                   "Either code error, or paths in lattice have inconsistent lengths");
      std::copy(string.begin(), string.begin() + len, begin);
      return;
    }

    ArcIterator<ExpandedFst<CompactArc> > aiter(clat, state);
    if (arc_idx != -1)
      aiter.Seek(arc_idx);
    KALDI_ASSERT(!aiter.Done() &&
                 "Either code error, or paths in lattice are inconsistent in length.");

    const CompactArc &arc = aiter.Value();
    size_t arc_len = arc.weight.String().size();
    if (arc_len >= len) {
      std::copy(arc.weight.String().begin(), arc.weight.String().begin() + len, begin);
    } else {
      std::copy(arc.weight.String().begin(), arc.weight.String().end(), begin);
      // Recurse.
      GetString(clat, arc.nextstate, -1, begin + arc_len, end);
    }
  }

  void CheckForConflict(const CompactWeight &final,
                        StateId state,
                        int32 *shift) {
    if (shift == NULL) return;
    // At input, "shift" has the maximum value that we could shift back assuming
    // there is no conflict between the values of the strings.  We need to check
    // if there is conflict, and if so, reduce the "shift".
    bool is_final = (final != CompactWeight::Zero());
    size_t num_arcs = clat_->NumArcs(state);
    if (num_arcs + (is_final ? 1 : 0) > 1 && *shift > 0) {
      // There is potential for conflict between string values, because >1
      // [arc or final-prob].  Find the longest shift up to and including the
      // current shift, that gives no conflict.

      std::vector<IntType> string(*shift), compare_string(*shift);
      size_t arc;
      if (is_final) {
        KALDI_ASSERT(final.String().size() >= *shift);
        std::copy(final.String().begin(), final.String().begin() + *shift,
                  string.begin());
        arc = 0;
      } else {
        // set "string" to string if we take 1st arc.
        GetString(*clat_, state, 0, string.begin(), string.end());
        arc = 1;
      }
      for (; arc < num_arcs; arc++) { // for the other arcs..
        GetString(*clat_, state, arc,
                  compare_string.begin(), compare_string.end());
        std::pair<typename std::vector<IntType>::iterator,
            typename std::vector<IntType>::iterator> pr =
                std::mismatch(string.begin(), string.end(),
                              compare_string.begin());
        if (pr.first != string.end()) { // There was a mismatch.  Reduce the shift
          // to a value where they will match.
          *shift = pr.first - string.begin();
          string.resize(*shift);
          compare_string.resize(*shift);
        }
      }
    }
  }

  void ComputeShifts() {
    StateId num_states = clat_->NumStates();
    shift_vec_.resize(num_states, 0);
    
    // The for loop will only work if StateId is signed, so assert this.
    KALDI_COMPILE_TIME_ASSERT(static_cast<StateId>(-1) < static_cast<StateId>(0));
    // We rely on the topological sorting, so clat_->Start() should be zero or
    // at least any preceding states should be non-accessible.  We leave the
    // shift at zero for the start state because we can't shift to before that.
    for (StateId state = num_states - 1; state > clat_->Start(); state--) {
      size_t num_arcs = clat_->NumArcs(state);
      CompactWeight final = clat_->Final(state);
      if (num_arcs == 0) {
        // we can shift back by the number of transition-ids on the
        // final-prob, if any.
        shift_vec_[state] = final.String().size();
      } else { // We have arcs ...
        int32 shift = std::numeric_limits<int32>::max();
        size_t num_arcs = 0;
        bool is_final = (final != CompactWeight::Zero());
        if (is_final)
          shift = std::min(shift, static_cast<int32>(final.String().size()));
        for (ArcIterator<MutableFst<CompactArc> > aiter(*clat_, state);
             !aiter.Done(); aiter.Next(), num_arcs++) {
          const CompactArc &arc (aiter.Value());
          shift = std::min(shift, shift_vec_[arc.nextstate] +
                           static_cast<int32>(arc.weight.String().size()));
        }
        CheckForConflict(final, state, &shift);
        shift_vec_[state] = shift;
      }
    }
  }

  void ApplyShifts() {
    StateId num_states = clat_->NumStates();
    for (StateId state = 0; state < num_states; state++) {
      int32 shift = shift_vec_[state];
      std::vector<IntType> string;
      for (MutableArcIterator<MutableFst<CompactArc> > aiter(clat_, state);
           !aiter.Done(); aiter.Next()) {
        CompactArc arc(aiter.Value());
        KALDI_ASSERT(arc.nextstate > state && "Cyclic lattice");

        string = arc.weight.String();
        size_t orig_len = string.size(), next_shift = shift_vec_[arc.nextstate];
        // extend "string" by next_shift.
        string.resize(string.size() + next_shift);
        // The next command sets the last "next_shift" elements of 'string' to
        // the string starting from arc.nextstate (taking an arbitrary path).
        GetString(*clat_, arc.nextstate, -1,
                  string.begin() + orig_len, string.end());
        // Remove the first "shift" elements of this string and set the
        // arc-weight string to this.
        arc.weight.SetString(std::vector<IntType>(string.begin() + shift,
                                                string.end()));
        aiter.SetValue(arc);
      }
      
      CompactWeight final = clat_->Final(state);
      if (final != CompactWeight::Zero()) {
        // Erase first "shift" elements of final-prob.
        final.SetString(std::vector<IntType>(final.String().begin() + shift,
                                           final.String().end()));
        clat_->SetFinal(state, final);
      }
    }
  }

 private:
  MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat_;

  // For each state s, shift_vec_[s] >= 0  is how much we will shift the
  // transition-ids back at this state.
  std::vector<int32> shift_vec_;
};

template<class Weight, class IntType>
bool PushCompactLatticeStrings(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat) {
  CompactLatticePusher<Weight, IntType> pusher(clat);
  return pusher.Push();
}

template<class Weight, class IntType>
bool PushCompactLatticeWeights(
   MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat) {
  if (clat->Properties(kTopSorted, true) == 0) {
    if (!TopSort(clat)) {
      KALDI_WARN << "Topological sorting of state-level lattice failed "
          "(probably your lexicon has empty words or your LM has epsilon cycles; this "
          " is a bad idea.)";
      return false;
    }
  }
  typedef CompactLatticeWeightTpl<Weight, IntType> CompactWeight;
  typedef ArcTpl<CompactWeight> CompactArc;
  typedef typename CompactArc::StateId StateId;

  StateId num_states = clat->NumStates();
  if (num_states == 0) {
    KALDI_WARN << "Pushing weights of empty compact lattice";
    return true; // this is technically success because an empty
                 // lattice is already pushed.
  }
  std::vector<Weight> weight_to_end(num_states); // Note: LatticeWeight
                                                 // contains two floats.
  for (StateId s = num_states - 1; s >= 0; s--) {
    Weight this_weight_to_end = clat->Final(s).Weight();
    for (ArcIterator<MutableFst<CompactArc> > aiter(*clat, s);
         !aiter.Done(); aiter.Next()) {
      const CompactArc &arc = aiter.Value();
      KALDI_ASSERT(arc.nextstate > s && "Cyclic lattices not allowed.");
      this_weight_to_end = Plus(this_weight_to_end,
                                Times(aiter.Value().weight.Weight(),
                                      weight_to_end[arc.nextstate]));
    }
    if (this_weight_to_end == Weight::Zero()) {
      KALDI_WARN << "Lattice has non-coaccessible states.";
    }
    weight_to_end[s] = this_weight_to_end;
  }
  weight_to_end[0] = Weight::One(); // We leave the "leftover weight" on
                                    // the start state, which won't
                                    // necessarily end up summing to one.
  for (StateId s = 0; s < num_states; s++) {
    Weight this_weight_to_end = weight_to_end[s];
    if (this_weight_to_end == Weight::Zero())
      continue;
    for (MutableArcIterator<MutableFst<CompactArc> > aiter(clat, s);
         !aiter.Done(); aiter.Next()) {
      CompactArc arc = aiter.Value();
      Weight next_weight_to_end = weight_to_end[arc.nextstate];
      if (next_weight_to_end != Weight::Zero()) {
        arc.weight.SetWeight(Times(arc.weight.Weight(),
                                   Divide(next_weight_to_end,
                                          this_weight_to_end)));
        aiter.SetValue(arc);
      }
    }
    CompactWeight final_weight = clat->Final(s);
    if (final_weight != CompactWeight::Zero()) {
      final_weight.SetWeight(Divide(final_weight.Weight(), this_weight_to_end));
      clat->SetFinal(s, final_weight);
    }
  }
  
  return true;
}

// Instantiate for CompactLattice.
template
bool PushCompactLatticeStrings<kaldi::LatticeWeight, kaldi::int32>(
   MutableFst<kaldi::CompactLatticeArc> *clat);

template
bool PushCompactLatticeWeights<kaldi::LatticeWeight, kaldi::int32>(
   MutableFst<kaldi::CompactLatticeArc> *clat);

}  // namespace fst
