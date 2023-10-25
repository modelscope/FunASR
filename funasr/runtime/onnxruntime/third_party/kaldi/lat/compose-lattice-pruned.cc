// lat/compose-lattice-pruned.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)
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

#include "lat/compose-lattice-pruned.h"
#include "lat/lattice-functions.h"

namespace kaldi {

/**
   PrunedCompactLatticeComposer implements an algorithm for pruned composition.
   It uses a heuristic (like the heuristics used in A*) to estimate the
   cost to the end of a graph, of the best path that we might get if
   we expand a particular transition out of a particular state.  This enables
   us to use a priority queue to expand arcs in the composed result in an
   order roughly from most promising to least promising.

   Because some of the quantities used in the heuristic are hard to efficiently
   keep updated as the composed output is incrementally added to, we
   periodically recompute these quantities (c.f. RecomputePruningInfo()).
   In order to prevent this periodic recomputation from dominating the time
   taken to produce the lattice, we recompute these things on a schedule where,
   between each computation, we allow the size of the output to grow by
   a constant factor (default: 1.5).  Since the time taken to do the
   recomputation of quantities used in the heuristic takes time linear in the
   size of the so-far existing composed output, doing so on this type of schedule
   will add no more than a constant factor to the runtime.

 */
class PrunedCompactLatticeComposer {
 public:
  PrunedCompactLatticeComposer(
      const ComposeLatticePrunedOptions &opts,
      const CompactLattice &clat,
      fst::DeterministicOnDemandFst<fst::StdArc> *det_fst,
      CompactLattice* composed_clat);

  // Does the composition.  You must call this just once per object.
  void Compose();

 private:

  // Gets the num-arcs limit for this iteration of the algorithm, which will be
  // opts_.initial_num_arcs if there are currently no arcs; or otherwise
  // opts_.growth_ration * the current number of arcs (subject to the
  // opts_.max_arcs limit if we have already reached a final-state).  This helps
  // ensure that we call RecomputePruningInfo() on an appropriate schedule.
  int32 GetCurrentArcLimit() const;

  // This function, called just once at the start, computes all the static
  // information about the input lattice 'clat', in lat_state_info_.  (however,
  // the 'composed_states' members are just set to the empty vector for now.
  void ComputeLatticeStateInfo();

  // Called just once at the start, this sets up the first state in the
  // composed output.
  void AddFirstState();

  // This function processes the next un-expanded transition (or final-state)
  // out of the composed state numbered 'composed_state_to_expand'.
  void ProcessQueueElement(int32 composed_state_to_expand);

  // This is a part of ProcessQueueElements() that has been broken out
  // for clarity. it process the arc_index'th arc out of this source state.
  void ProcessTransition(int32 composed_src_state,
                         int32 arc_index);

  // This function recomputes certain members of the ComposedStateInfo relating
  // to the output states: namely, 'forward_cost', 'backward_cost' and
  // 'delta_backward_cost'.  In between calls to this function, we try to
  // keep those quantities as accurate as possible, but they aren't
  // completely accurate (see comments by their declarations for more info).
  void RecomputePruningInfo();

  // Sets '*composed_states' to a list of the states that currently
  // exist in the composed output, in topologically sorted order.
  // At exit, *composed_states will be a permutation of numbers
  // [0, 1, ...  clat_out_->NumStates() - 1], beginning with the
  // start-state 0.
  void GetTopsortedStateList(std::vector<int32> *composed_states) const;

  // Called from RecomputePruningInfo(), this computes all the 'forward_cost'
  // and 'prev_composed_state' members of the ComposedStateInfo.
  //   @param [in] composed_states  This is expected to be a list,
  //         in topological order, of all currently existing composed states,
  //         as produced by GetTopsortedStateList().
  void ComputeForwardCosts(const std::vector<int32> &composed_states);

  // Called from RecomputePruningInfo(), this computes all the 'backward_cost'
  // members of the ComposedStateInfo.  It also sets 'output_best_cost_'.
  // 'composed_states' is expected to be a list, in topological order, of all
  // currently existing composed states, as produced by GetTopsortedStateList().
  void ComputeBackwardCosts(const std::vector<int32> &composed_states);

  // Called from RecomputePruningInfo(), this computes all the
  // 'delta_backward_cost' members of the ComposedStateInfo.  'composed_states'
  // is expected to be a list, in topological order, of all currently existing
  // composed states, as produced by GetTopsortedStateList().  It also computes
  // the 'expected_cost_offset' values for all states, and uses them recreate
  // 'composed_state_queue_'.
  void ComputeDeltaBackwardCosts(const std::vector<int32> &composed_states);


  // This struct contains information about a state of the input lattice.
  struct LatticeStateInfo {
    // 'backward_cost' is the total cost of the best path from this state to
    //  the final state in the source lattice, including the final-prob.
    double backward_cost;

    // 'arc_delta_costs' is an array, one for each arc (and the final-prob, if
    // present), showing how much the cost to the final-state for the best path
    // starting in this state and exiting through each arc (or final-prob),
    // differs from 'backward_cost'.  Specifically, it contains pairs
    // (delta_cost, arc_index), where delta_cost >= 0 and arc_index is
    // either the index into this state's array of arcs (for arcs), or -1
    // if this represents the final-prob.
    //
    // 'arc_delta_costs' will be sorted, so that the first element has
    // .first=0.0 and the delta-costs will be increasing order.  This means that
    // we expand them from the start of the array, in order to process the best
    // arcs first.
    // lat_state_info_[i].arc_delta_costs.size() will equal will equal
    // clat_.NumStates(i), plus one if clat_.Final(i) != Zero().
    std::vector<std::pair<BaseFloat, int32> > arc_delta_costs;


    // 'composed_states' is a list of the state-ids in the composed output
    // that correspond to this state in the lattice, so we expect
    // that composed_state_info_[composed_states[i]].lat_state
    // equals the index of this lattice state.  This is helpful in
    // accessing the states in the output lattice in topological
    // order.
    std::vector<int32> composed_states;
  };

  // This struct contains information about a state of the composed
  // output.
  struct ComposedStateInfo {
    // 'lat_state' and 'lm_state' form the pair of states in the two FSTs
    // that this state corresponds to.  The unordered map 'pair_to_state_' maps these
    // state-pairs to the index of the composed state (the state-index in clat_out_).
    int32 lat_state;
    int32 lm_state;

    // the number of arcs on the path from the start state to this state, in the
    // composed lattice, by which this state was first reached.
    int32 depth;

    // If you have just called RecomputePruningInfo(), then
    // 'forward_cost' will equal the cost of the best path from the start-state
    // to this state, in the composed output.
    //
    // In between calls to RecomputePruningInfo() it may not always be fully up
    // to date; instead it will be an upper bound on what it would be if you had
    // just called RecomputePruningInfo(); it will be the cost of some path but
    // not necessarily the best path.
    double forward_cost;

    // 'backward_cost' relates to the cost from this state to the final-state in
    // the composed FST.  (By this we mean, more precisely, the cost of the best
    // path from this state to any final state, including the final-prob in that
    // final state).
    //
    // If we have just called RecomputePruningInfo(), then the following rules
    // specify what the value of 'backward_cost' will be:
    //   - If a final state is reachable from this state, backward_cost
    //     will contain the cost of the best path from this state to the
    //     final state (including the corresponding final-prob).
    //   - Otherwise, it will contain +infinity.
    //
    // If RecomputePruningInfo() has not just been called), it may contain any
    // value that is >= the value the the rules above specify (since, for
    // existing states, we don't modify it between calls to
    // RecomputePruningInfo()).  For states that have been added since
    // RecomputePruningInfo() was last called, it will be infinity.
    double backward_cost;

    // 'delta_backward_cost' is a quantity that is used in our heuristic of the
    // cost to an end-state from expanding a previously un-expanded arc.  It is
    // an estimate of the difference between the backward cost in this struct
    // (this->backward_cost) and the backward cost in the input lattice
    // (LatticeStateInfo::backward_cost).  This term reflects the anticipated
    // extra costs from 'det_fst_', which, while fairly close to zero, may be
    // substantial enough to want to correct for.
    //
    // The following is the value that 'delta_backward_cost' will have if
    // RecomputePruningInfo() has just been called:
    //   - If backward_cost is finite (this state in the composed result can
    //    reach the final state via currently expanded states), then
    //    delta_backward_cost is this->backward_cost minus
    //    lat_state_info_[lat_state].backward_cost.  (It will mostly, but
    //    not always, be <= 0, reflecting that the new LM is better than
    //    the old LM).
    //  - On the other hand, if backward_cost is infinite: delta_backward_cost
    //     is set to the delta_backward_cost of the previous state on the best
    //     path from the start state of the composed result to this state (or
    //     zero if this is the start state).
    //
    // If RecomputePruningInfo() has not just been called, then:
    //  - For states created since RecomputePruningInfo() was last called,
    //    delta_backward_cost will be inherited from the source state from
    //    which the new state was expanded.
    //  - For other states, delta_backward_cost will be unchanged since
    //    RecomputePruningInfo() was last called.
    // The above rules may make the delta_backward_cost a less accurate, but
    // still probably reasonable, heuristic.  What it is a heuristic for,
    // is: if we were to successfully reach an end-state of the composed output
    // from this state, what would be the resulting backward_cost
    // minus lat_state_info_[lat_state].backward_cost.
    BaseFloat delta_backward_cost;

    // 'prev_composed_state' is the previous state on the best path from
    // the start-state to the current state (or -1 if this is the start state).
    // It is computed in RecomputePruningInfo() when setting up 'forward_cost',
    // and then used to compute delta_backward_cost.  It is not otherwise
    // used.
    int32 prev_composed_state;

    // 'sorted_arc_index' is an index into the 'arc_delta_costs' array which is
    // a member of the LatticeStateInfo object corresponding to the lattice
    // state 'lat_state'.  It corresponds to the next arc (or final-prob) out of
    // the input lattice that we have yet to expand in the composition; or -1 if
    // we have expanded all of them.  When we first reach a composed state,
    // 'sorted_arc_index' will be zero; then it will increase one at a time as
    // we expand arcs until either the composition terminates or we have
    // expanded all the arcs and it becomes -1.
    int32 sorted_arc_index;

    // 'arc_delta_cost' is a derived quantity that we store here for easier
    // access.  Suppose this_lat_info is lat_state_info_[lat_state]; then
    // if sorted_arc_index >= 0, then:
    //    arc_delta_cost == this_lat_info.arc_delta_costs[sorted_arc_index].first
    // else: arc_delta_cost == +infinity.
    //
    // what 'arc_delta_cost' represents (or is a heuristic for), is the expected
    // cost of a path to the final-state leaving through the arc we're about to
    // expand, minus the expected cost of any path to the final-state starting
    // from this state.
    BaseFloat arc_delta_cost;

    // view 'expected_cost_offset' a phantom field of this struct, that has
    // been optimized out.  It's clearer if we act like it's a field, but
    // actually it's not stored.
    //
    // 'expected_cost_offset' is a derived quantity that reflects the expected
    // cost (according to our heuristic) of the best path we might encounter
    // when expanding the next previously unseen arc (or final-prob),
    // corresponding to 'sorted_arc_index'.  (This is the expected cost of a
    // successful path, from the beginning to the end of the lattice, but
    // constrained to be a path that contains the arc we're about to expand).
    //
    // The 'offset' part is about subtracting the best cost of the lattice, so we
    // can cast to float without too much loss of accuracy:
    //   expected_cost_offset = expected_cost - lat_best_cost_.
    //
    // We define expected_cost_offset by defining the 'expected_cost' part;
    // for clarity:
    //   First, let lat_backward_cost equal the backward_cost of the LatticeStateInfo
    //   corresponding to 'lat_state', i.e.
    //   lat_backward_cost = lat_state_info_[lat_state].backward_cost.  Then:
    //  expected_cost = forward_cost + lat_backward_cost +
    //                  delta_backward_cost + arc_delta_cost.
    // expected_cost_offset will always equal the above minus lat_best_cost_.
    //
    // The formula for expected_cost above is a pretty good heuristic for what
    // the cost to the end-state will be.  If the costs in det_fst_ were zero,
    // then the expression (forward_cost + lat_backward_cost + arc_delta_cost)
    // would be exact, and we would expand things in the ideal, best-first
    // order.  "delta_backward_cost" is a reasonable approximation for the extra
    // costs from 'det_fst_'.
    // BaseFloat expected_cost_offset;
  };

  // This bool variable is initialized to false, and will be updated to true
  // the first time a Final() function is called on the det_fst_. Then we will
  // immediately call RecomputeRruningInfo() so that the output_best_cost_ is
  // changed from +inf to a finite value, to be used in beam search. This is the
  // only time the RecomputeRruningInfo() function is called manually; otherwise
  // it always follows an automatic schedule based on the num-arcs of the output
  // lattice.
  bool output_reached_final_;

  // This variable, which we set initially to -1000, makes sure that in the
  // beginning of the algorithm, we always prioritize exploring the lattice
  // in a depth-first way. Once we find a path reaching a final state, this
  // variable will be reset to 0.
  // The reason we do this is because the beam-search depends on a good estimate
  // of the composed-best-cost, which before we reach a final state, we instead
  // borrow the value from best-cost from the input lattice, which is usually
  // systematically worse than the RNNLM scores, and makes the algorithm spend
  // a lot of time before reaching any final state, especially if the input
  // lattices are large.
  float depth_penalty_;
  const ComposeLatticePrunedOptions &opts_;
  const CompactLattice &clat_in_;
  fst::DeterministicOnDemandFst<fst::StdArc> *det_fst_;
  CompactLattice *clat_out_;

  // This counter keeps track of the number of arcs in the output lattice
  // clat_out_.  When it exceeds max_arcs,
  int32 num_arcs_out_;

  std::vector<LatticeStateInfo> lat_state_info_;

  // 'lat_best_cost' is the cost of the best path in the input lattice,
  // equal to lat_state_info_[0].backward_cost (we check that 0 is the
  // start state in the input lattice).
  double lat_best_cost_;

  // 'output_best_cost_' is the cost of the best successful path in the output
  // lattice 'clat_out_'; or +infinity if 'clat_out_' does not yet have any
  // successful paths.  It is updated only when RecomputePruningInfo() is
  // called.
  double output_best_cost_;


  // current_cutoff_ is a value used in deciding which composed states
  // need to be included in the queue.  Each time RecomputePruningInfo()
  // called, current_cutoff_ is set to
  //    (output_best_cost_ - lat_best_cost_ + opts_.lattice_compose_beam).
  // It will be +infinity if the output lattice doesn't yet have any
  // successful paths.  It decreases with time.  You can compare the
  // phantom 'expected_cost_offset' members of ComposedStateInfo with this
  // value; if they are more than this value, then there is no need
  // to enter the corresponding state into the queue.
  BaseFloat current_cutoff_;

  typedef std::priority_queue<std::pair<BaseFloat, int32>,
                      std::vector<std::pair<BaseFloat, int32> >,
                      std::greater<std::pair<BaseFloat, int32> > > QueueType;

  // composed_state_queue_ is a priority queue of the composed states
  // that we are intending to expand.  It contains pairs
  //   (expected_cost_offset, composed_state_index),
  // where expected_cost_offset == the phantom variable
  //       composed_state_info_[composed_state_index].expected_cost_offset.
  // We process the states from lowest cost first.
  // Every time RecomputePruningInfo() is called, this is cleared and repopulated
  // (since the states' expected_cost_offset values may change), and in between
  // calls to RecomputePruningInfo(), we do insert elements for newly created
  // states.
  QueueType composed_state_queue_;


  std::vector<ComposedStateInfo> composed_state_info_;

  // This maps a pair (lat_state, lm_state) to the index of the
  // state in the composed FST.  That would correspond to a state-id in
  // clat_out_, and also to an index into 'composed_state_info_'.
  unordered_map<std::pair<int32,int32>,
                int32, PairHasher<int32> > pair_to_state_;

  // This contains the set of state-indexes of the input lattice that already
  // have states in the composed output (i.e. is in accessed_lat_states_ if and
  // only if !lat_state_info_[i].composed_states.empty().  The point is to be
  // able to enumerate, in order or in reverse order, just those states in the
  // lattice that appear in the composed output (it's an efficiency thing that
  // will matter more for early iterations of the composition, when we need
  // to access the output lattice in topological order).
  std::set<int32> accessed_lat_states_;
};


void PrunedCompactLatticeComposer::GetTopsortedStateList(
    std::vector<int32> *composed_states) const {
  composed_states->clear();
  composed_states->reserve(clat_out_->NumStates());
  std::set<int32>::const_iterator iter = accessed_lat_states_.begin(),
      end = accessed_lat_states_.end();
  for (; iter != end; ++iter) {
    int32 lat_state = *iter;
    const LatticeStateInfo &input_lat_info = lat_state_info_[lat_state];
    composed_states->insert(composed_states->end(),
                            input_lat_info.composed_states.begin(),
                            input_lat_info.composed_states.end());
  }
  KALDI_ASSERT((*composed_states)[0] == 0 &&
               static_cast<int32>(composed_states->size()) ==
               clat_out_->NumStates());
}

int32 PrunedCompactLatticeComposer::GetCurrentArcLimit() const {
  int32 current_num_arcs = num_arcs_out_;
  if (current_num_arcs == 0) {
    return opts_.initial_num_arcs;
  } else {
    KALDI_ASSERT(opts_.growth_ratio > 1.0);
    int32 ans = static_cast<int32>(current_num_arcs *
                                   opts_.growth_ratio);
    if (ans == current_num_arcs)  // make sure the target increases.
      ans = current_num_arcs + 1;
    // if we have already reached a final state, then
    // apply the max_arcs limit.
    if (output_best_cost_ - output_best_cost_ == 0.0 &&
        ans > opts_.max_arcs)
      ans = opts_.max_arcs;
    return ans;
  }

}


void PrunedCompactLatticeComposer::RecomputePruningInfo() {
  std::vector<int32> all_composed_states;
  GetTopsortedStateList(&all_composed_states);
  ComputeForwardCosts(all_composed_states);
  ComputeBackwardCosts(all_composed_states);
  ComputeDeltaBackwardCosts(all_composed_states);
}

void PrunedCompactLatticeComposer::ComputeForwardCosts(
    const std::vector<int32> &composed_states) {
  KALDI_ASSERT(composed_states[0] == 0);

  // Note: when we initialized composed_state_info_[0]
  // we set forward_cost = 0.0, prev_composed_state = -1.

  std::vector<ComposedStateInfo>::iterator
      state_iter = composed_state_info_.begin(),
      state_end = composed_state_info_.end();

  state_iter->depth = 0;  // start state has depth 0
  ++state_iter;  // Skip over the start state.
  // Set all other forward_cost fields to infinity and prev_composed_state to
  // -1.
  for (; state_iter != state_end; ++state_iter) {
    state_iter->forward_cost = std::numeric_limits<double>::infinity();
    state_iter->prev_composed_state = -1;
  }

  std::vector<int32>::const_iterator state_index_iter = composed_states.begin(),
      state_index_end = composed_states.end();
  for (; state_index_iter != state_index_end; ++state_index_iter) {
    int32 composed_state_index = *state_index_iter;
    const ComposedStateInfo &info = composed_state_info_[
        composed_state_index];
    double forward_cost = info.forward_cost;
    // The next line is a check for infinity.  If infinities have appeared, it
    // either means there is a bug in the algorithm or there were infinities or
    // NaN's in the lattice.
    KALDI_ASSERT(forward_cost - forward_cost == 0.0);
    fst::ArcIterator<CompactLattice> aiter(*clat_out_,
                                           composed_state_index);
    for (; !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      double arc_cost = ConvertToCost(arc.weight),
          next_forward_cost = forward_cost + arc_cost;
      ComposedStateInfo &next_info = composed_state_info_[arc.nextstate];
      if (next_info.forward_cost > next_forward_cost) {
        next_info.forward_cost = next_forward_cost;
        next_info.prev_composed_state = composed_state_index;
        next_info.depth = composed_state_info_[composed_state_index].depth + 1;
      }
    }
  }
}

void PrunedCompactLatticeComposer::ComputeBackwardCosts(
    const std::vector<int32> &composed_states) {
  // Access the composed states in reverse topological order from latest to
  // earliest.
  std::vector<int32>::const_reverse_iterator iter = composed_states.rbegin(),
      end = composed_states.rend();
  for (; iter != end; ++iter) {
    int32 composed_state_index = *iter;
    ComposedStateInfo &info = composed_state_info_[composed_state_index];
    double backward_cost =
        ConvertToCost(clat_out_->Final(composed_state_index));
    fst::ArcIterator<CompactLattice> aiter(*clat_out_,
                                           composed_state_index);
    for (; !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      double arc_cost = ConvertToCost(arc.weight),
         next_backward_cost = composed_state_info_[arc.nextstate].backward_cost,
         this_backward_cost = arc_cost + next_backward_cost;
      if (this_backward_cost < backward_cost)
        backward_cost = this_backward_cost;
    }
    // It's OK if at this point, backward_cost is still +infinity.  This means
    // that this state cannot reach the end yet, which means we have not yet
    // expanded any path from this state all the way to a final-state of the
    // output.
    info.backward_cost = backward_cost;
  }
  output_best_cost_ = composed_state_info_[0].backward_cost;
  // See the declaration of current_cutoff_ for more information.  Note: on
  // early iterations, before any path reaches a final state of the composed
  // lattice, current_cutoff_ may be +infinity, and this is OK.
  current_cutoff_ =
      output_best_cost_ - lat_best_cost_ + opts_.lattice_compose_beam;
}

void PrunedCompactLatticeComposer::ComputeDeltaBackwardCosts(
    const std::vector<int32> &composed_states) {

  int32 num_states = clat_out_->NumStates();
  for (int32 composed_state_index = 0; composed_state_index < num_states;
       ++composed_state_index) {
    ComposedStateInfo &info = composed_state_info_[composed_state_index];
    int32 lat_state = info.lat_state;
    // Note: delta_backward_cost will be +infinity at this stage if the
    // backward_cost was +infinity.  This is OK; we'll set them all to
    // finite values later in this function.
    info.delta_backward_cost =
        info.backward_cost - lat_state_info_[lat_state].backward_cost + info.depth * depth_penalty_;
  }

  // 'queue_elements' is a list of items (expected_cost_offset,
  // composed_state_index) that we are going to add to composed_state_queue_,
  // after clearing it.  It's more efficient to accumulate them as a vector
  // and add them all at once, than adding them one by one (search online for
  // "heapify" if this seems confusing).
  std::vector<std::pair<BaseFloat, int32> > queue_elements;
  queue_elements.reserve(num_states);

  double lat_best_cost = lat_best_cost_;
  BaseFloat current_cutoff = current_cutoff_;
  std::vector<int32>::const_iterator iter = composed_states.begin(),
      end = composed_states.end();
  for (; iter != end; ++iter) {
    int32 composed_state_index = *iter;
    ComposedStateInfo &info = composed_state_info_[composed_state_index];
    if (info.delta_backward_cost - info.delta_backward_cost != 0) {
      // if info.delta_backward_cost is +infinity...
      int32 prev_composed_state = info.prev_composed_state;
      if (prev_composed_state < 0) {
        KALDI_ASSERT(composed_state_index == 0);
        info.delta_backward_cost = 0.0;
      } else {
        const ComposedStateInfo &prev_info =
            composed_state_info_[prev_composed_state];
        // Check that prev_info.delta_backward_cost is finite.
        KALDI_ASSERT(prev_info.delta_backward_cost -
                     prev_info.delta_backward_cost == 0.0);
        info.delta_backward_cost = prev_info.delta_backward_cost + depth_penalty_;
      }
    }
    double lat_backward_cost = lat_state_info_[info.lat_state].backward_cost;
    // See the formula by where expected_cost_offset is declared in the
    // struct for explanation.
    BaseFloat expected_cost_offset =
        info.forward_cost + lat_backward_cost + info.delta_backward_cost +
        info.arc_delta_cost - lat_best_cost;
    // If info.expected_cost_offset were real, we'd set it here:
    //info.expected_cost_offset = expected_cost_offset;

    // At this point expected_cost_offset may be infinite, if arc_delta_cost was
    // infinite (reflecting that we processed all the arcs, and the final-state
    // if applicable, of the lattice state corresponding to this composed state.
    if (expected_cost_offset < current_cutoff) {
      queue_elements.push_back(std::pair<BaseFloat, int32>(
          expected_cost_offset, composed_state_index));
    }
  }

  // Reinitialize composed_state_queue_ from 'queue_elements'.
  QueueType temp_queue(queue_elements.begin(), queue_elements.end());
  composed_state_queue_.swap(temp_queue);
}

void PrunedCompactLatticeComposer::ComputeLatticeStateInfo() {
  KALDI_ASSERT(clat_in_.Properties(fst::kTopSorted, true) ==
               fst::kTopSorted && clat_in_.NumStates() > 0 &&
               clat_in_.Start()  == 0);
  int32 num_lat_states = clat_in_.NumStates();
  lat_state_info_.resize(num_lat_states);

  for (int32 s = num_lat_states - 1; s >= 0; s--) {
    LatticeStateInfo &info = lat_state_info_[s];
    std::vector<std::pair<double, int32> > arc_costs;
    double backward_cost = ConvertToCost(clat_in_.Final(s));
    if (backward_cost != std::numeric_limits<double>::infinity())
      arc_costs.push_back(std::pair<BaseFloat,int32>(backward_cost, -1));
    fst::ArcIterator<CompactLattice> aiter(clat_in_, s);
    int32 arc_index = 0;
    for (; !aiter.Done(); aiter.Next(), ++arc_index)  {
      const CompactLatticeArc &arc = aiter.Value();
      KALDI_ASSERT(arc.nextstate > s);
      backward_cost = lat_state_info_[arc.nextstate].backward_cost +
          ConvertToCost(arc.weight);
      KALDI_ASSERT(backward_cost - backward_cost == 0.0 &&
                   "Possibly not all states of input lattice are co-accessible?");
      arc_costs.push_back(std::pair<BaseFloat,int32>(backward_cost, arc_index));
    }
    std::sort(arc_costs.begin(), arc_costs.end());
    KALDI_ASSERT(!arc_costs.empty() &&
                 "Possibly not all states of input lattice are co-accessible?");
    backward_cost = arc_costs[0].first;
    info.backward_cost = backward_cost;  // this is the state's backward_cost,
                                         // reflecting the best path to the end.
    info.arc_delta_costs.resize(arc_costs.size());
    std::vector<std::pair<double, int32> >::const_iterator
        src_iter = arc_costs.begin(), src_end = arc_costs.end();
    std::vector<std::pair<BaseFloat, int32> >::iterator
        dest_iter = info.arc_delta_costs.begin();
    for (; src_iter != src_end; ++src_iter, ++dest_iter) {
      dest_iter->first = BaseFloat(src_iter->first - backward_cost);
      dest_iter->second = src_iter->second;
    }
  }
  lat_best_cost_ = lat_state_info_[0].backward_cost;
}

PrunedCompactLatticeComposer::PrunedCompactLatticeComposer(
      const ComposeLatticePrunedOptions &opts,
      const CompactLattice &clat_in,
      fst::DeterministicOnDemandFst<fst::StdArc> *det_fst,
      CompactLattice* composed_clat): output_reached_final_(false),
    opts_(opts), clat_in_(clat_in), det_fst_(det_fst),
    clat_out_(composed_clat),
    num_arcs_out_(0),
    output_best_cost_(std::numeric_limits<double>::infinity()),
    current_cutoff_(std::numeric_limits<double>::infinity()) {
  clat_out_->DeleteStates();
  depth_penalty_ = -1000;
}


void PrunedCompactLatticeComposer::AddFirstState() {
  int32 state_id = clat_out_->AddState();
  clat_out_->SetStart(state_id);
  KALDI_ASSERT(state_id == 0);
  composed_state_info_.resize(1);
  ComposedStateInfo &composed_state = composed_state_info_[0];
  composed_state.lat_state = 0;
  composed_state.lm_state = det_fst_->Start();
  composed_state.depth = 0;
  composed_state.forward_cost = 0.0;
  composed_state.backward_cost = std::numeric_limits<double>::infinity();
  composed_state.delta_backward_cost = 0.0;
  composed_state.prev_composed_state = -1;
  composed_state.sorted_arc_index = 0;
  composed_state.arc_delta_cost = 0.0; // the first arc_delta_cost is always 0.0
                                       // due to sorting; no need to look it up.
  lat_state_info_[0].composed_states.push_back(state_id);
  accessed_lat_states_.insert(state_id);
  pair_to_state_[std::pair<int32, int32>(0, det_fst_->Start())] = state_id;

  BaseFloat expected_cost_offset = 0.0;  // the formula simplifies to zero
                                         // in this case.
  composed_state_queue_.push(
      std::pair<BaseFloat, int32>(expected_cost_offset,
                                  state_id));  // actually (0.0, 0).

}


void PrunedCompactLatticeComposer::ProcessQueueElement(
    int32 src_composed_state) {
  KALDI_ASSERT(static_cast<size_t>(src_composed_state) <
               composed_state_info_.size());

  ComposedStateInfo &src_composed_state_info = composed_state_info_[
      src_composed_state];
  int32 lat_state = src_composed_state_info.lat_state;
  const LatticeStateInfo &lat_state_info =
      lat_state_info_[lat_state];

  int32 sorted_arc_index = src_composed_state_info.sorted_arc_index,
      num_sorted_arcs = lat_state_info.arc_delta_costs.size();
  // note: num_sorted_arcs will be the number of arcs from this
  // lattice state; plus one if there is a final-prob.
  KALDI_ASSERT(sorted_arc_index >= 0);

  { // this block update the state's 'sorted_arc_index', 'arc_delta_cost' and
    // 'expected_cost_offset' to reflect the fact that (by the time we exit from
    // this function) we will have processed this arc (or the final-prob);
    // it also re-inserts this state into the queue, if appropriate.
    BaseFloat expected_cost_offset;
    if (sorted_arc_index + 1 == num_sorted_arcs) {
      src_composed_state_info.sorted_arc_index = -1;
      src_composed_state_info.arc_delta_cost =
          std::numeric_limits<BaseFloat>::infinity();
      expected_cost_offset =
          std::numeric_limits<BaseFloat>::infinity();
    } else {
      src_composed_state_info.sorted_arc_index = sorted_arc_index + 1;
      src_composed_state_info.arc_delta_cost =
          lat_state_info.arc_delta_costs[sorted_arc_index+1].first;
      expected_cost_offset =
          (src_composed_state_info.forward_cost +
           lat_state_info.backward_cost +
           src_composed_state_info.delta_backward_cost +
           src_composed_state_info.arc_delta_cost - lat_best_cost_);
    }
    // We do '<' here rather than '<=', so that if current_cutoff_ is infinity
    // and expected_cost_offset is infinity (because we've exhausted all the
    // transitions from this state, and sorted_arc_index is now -1), we don't
    // add this element to the queue.
    if (expected_cost_offset < current_cutoff_) {
      // this state has another exit arc (or final prob) that is good
      // enough to re-enter into the queue.  Note: if we are processing
      // an arc out of this state and the destination state is new,
      // we may also add something new to the queue at that time.

      // the following call should be equivalent to
      // composed_state_queue_.push(std::pair<BaseFloat,int32>(...)) with
      // the same pair of args.
      composed_state_queue_.emplace(
          expected_cost_offset, src_composed_state);
    }
  }

  int32 arc_index = lat_state_info.arc_delta_costs[sorted_arc_index].second;
  if (arc_index < 0) {  // This (arc_index == -1) means it is not really an arc
                        // index; it's a final-prob.
    int32 lm_state = src_composed_state_info.lm_state;
    BaseFloat lm_final_cost = det_fst_->Final(lm_state).Value();
    if (lm_final_cost != std::numeric_limits<BaseFloat>::infinity()) {
      // If there is a final-prob on this LM state (note: there always will be
      // for conventional language models), then add the final-prob of this
      // state...
      CompactLattice::Weight final_weight = clat_in_.Final(lat_state);
      // assume 'final_weight' is not Zero(); otherwise the final-prob should
      // not have been present in 'arc_delta_costs'.
      Lattice::Weight final_lat_weight = final_weight.Weight();
      final_lat_weight.SetValue1(final_lat_weight.Value1() +
                                 lm_final_cost);
      final_weight.SetWeight(final_lat_weight);
      clat_out_->SetFinal(src_composed_state, final_weight);
      double final_cost = ConvertToCost(final_lat_weight);
      if (final_cost < src_composed_state_info.backward_cost)
        src_composed_state_info.backward_cost = final_cost;
      if (!output_reached_final_) {
        output_reached_final_ = true;
        depth_penalty_ = 0.0;
        RecomputePruningInfo();
      }
    }
  } else {
    // It really was an arc.  This code is very complicated, so we make it its
    // own function.
    ProcessTransition(src_composed_state, arc_index);
  }
}

void PrunedCompactLatticeComposer::ProcessTransition(int32 src_composed_state,
                                                     int32 arc_index) {
  // Make src_composed_state a const pointer not a reference, as we may have to
  // modify the pointer if composed_state_info_ is resized.
  const ComposedStateInfo *src_info = &(composed_state_info_[
      src_composed_state]);
  int32 src_lat_state = src_info->lat_state;
  // Get the arc we are going to expand.
  fst::ArcIterator<CompactLattice> aiter(clat_in_, src_lat_state);
  aiter.Seek(arc_index);
  const CompactLatticeArc &lat_arc = aiter.Value();
  // Note: this code is for CompactLatticeArc, in which the ilabel and olabel
  // are the same, but we're writing it in such a way that it will naturally
  // generalize to LatticeArc, so there are separate variables for the ilabel
  // and the olabel.
  int32 dest_lat_state = lat_arc.nextstate,
      ilabel = lat_arc.ilabel,
      olabel = lat_arc.olabel;
  // Note: we expect that ilabel == olabel, since this is a CompactLattice, but this
  // may not be so if we extend this to work with Lattice.
  fst::StdArc lm_arc;

  // the input lattice might have epsilons
  if (olabel == 0) {
    lm_arc.ilabel = 0;
    lm_arc.olabel = 0;
    lm_arc.nextstate = src_info->lm_state;
    lm_arc.weight = fst::StdArc::Weight(0.0);
  } else if (!det_fst_->GetArc(src_info->lm_state, olabel, &lm_arc)) {
    // for normal language models we don't expect this to happen, but the
    // appropriate behavior is to do nothing; the composed arc does not exist,
    // so there is no arc to add and no new state to create.
    return;
  }
  int32 dest_lm_state = lm_arc.nextstate;
  // The following assertion is necessary because CompactLattice cannot support
  // different ilabel vs. olabel; and also it's an expectation about
  // language-models.
  KALDI_ASSERT(lm_arc.ilabel == lm_arc.olabel);

  LatticeStateInfo &dest_lat_state_info =
      lat_state_info_[dest_lat_state];

  int32 dest_composed_state;
  ComposedStateInfo *dest_info;

  { // The next block works out 'dest_composed_state' and
    // 'dest_info', and if the destination state did not already
    // exist, creates a new composed state.
    typedef std::unordered_map<std::pair<int32,int32>, int32,
        PairHasher<int32> > MapType;
    int32 new_composed_state = clat_out_->NumStates();
    std::pair<const std::pair<int32,int32>, int32> value(
        std::pair<int32,int32>(dest_lat_state, dest_lm_state), new_composed_state);
    std::pair<MapType::iterator, bool> ret =
        pair_to_state_.insert(value);
    if (ret.second) {
      // Successfully inserted: this dest-state did not already exist.  Most of
      // the rest of this block deals with the consequences of adding a new
      // state.
      int32 ans = clat_out_->AddState();
      KALDI_ASSERT(ans == new_composed_state);
      dest_composed_state = new_composed_state;
      composed_state_info_.resize(dest_composed_state + 1);
      dest_info = &(composed_state_info_[dest_composed_state]);
      // Re-assign src_composed_state as the vector might have been reallocated.
      src_info = &(composed_state_info_[src_composed_state]);
      if (dest_lat_state_info.composed_states.empty())
        accessed_lat_states_.insert(dest_lat_state);
      dest_lat_state_info.composed_states.push_back(new_composed_state);
      dest_info->lat_state = dest_lat_state;
      dest_info->lm_state = dest_lm_state;
      dest_info->depth = src_info->depth + 1;
      dest_info->forward_cost =
          src_info->forward_cost +
          ConvertToCost(lat_arc.weight) + lm_arc.weight.Value();
      dest_info->backward_cost =
          std::numeric_limits<double>::infinity();
      dest_info->delta_backward_cost =
          src_info->delta_backward_cost + dest_info->depth * depth_penalty_;
      // The 'prev_composed_state' field will not be read again until after it's
      // overwritten; we set it as below only for debugging purposes (the
      // negation is also for debugging purposes).
      dest_info->prev_composed_state = -src_composed_state;
      dest_info->sorted_arc_index = 0;
      dest_info->arc_delta_cost = 0.0;
      // Note: in the expression below, which can be understood with reference
      // to the comment by the declaration of the phantom variable
      // 'expected_cost_offset', 'arc_delta_cost' is known to equal 0.0 so it
      // has been removed.
      BaseFloat expected_cost_offset =
          (dest_info->forward_cost +
           dest_lat_state_info.backward_cost +
           dest_info->delta_backward_cost -
           lat_best_cost_);
      if (expected_cost_offset < current_cutoff_) {
        // the following call should be equivalent to
        // composed_state_queue_.push(std::pair<BaseFloat,int32>(...)) with
        // the same pair of args.
        composed_state_queue_.emplace(expected_cost_offset,
                                      dest_composed_state);
      }
    } else { // the destination composed state already existed.
      dest_composed_state = ret.first->second;
      dest_info = &(composed_state_info_[dest_composed_state]);
    }
  }
  // Add the arc from the src to dest state in the composed output.
  CompactLatticeArc new_arc;
  new_arc.nextstate = dest_composed_state;
  // Actually the ilabel and olabel are the same, but writing it this way will
  // generalize better to type Lattice if we need that later.
  new_arc.ilabel = ilabel;
  new_arc.olabel = olabel;
  new_arc.weight = lat_arc.weight;
  // 'weight' is the weight part, as opposed to the string part.
  LatticeArc::Weight weight = new_arc.weight.Weight();
  // include the LM-arc's weight in the weight of the new arc.
  weight.SetValue1(fst::Times(weight.Value1(), lm_arc.weight).Value());
  new_arc.weight.SetWeight(weight);
  clat_out_->AddArc(src_composed_state, new_arc);
  num_arcs_out_++;
}

static int32 TotalNumArcs(const CompactLattice &clat) {
  int32 num_states = clat.NumStates(),
      num_arcs = 0;
  for (int32 s = 0; s < num_states; s++)
    num_arcs += clat.NumArcs(s);
  return num_arcs;
}


void PrunedCompactLatticeComposer::Compose() {
  if (clat_in_.NumStates() == 0) {
    KALDI_WARN << "Input lattice to composition is empty.";
    return;
  }
  ComputeLatticeStateInfo();
  AddFirstState();
  // while (we have not reached final state  ||
  //        num-arcs produced < target num-arcs) { ...
  while (output_best_cost_ == std::numeric_limits<double>::infinity() ||
         num_arcs_out_ < opts_.max_arcs) {
    RecomputePruningInfo();
    int32 this_iter_arc_limit = GetCurrentArcLimit();
    while (num_arcs_out_ < this_iter_arc_limit &&
           !composed_state_queue_.empty()) {
      int32 src_composed_state = composed_state_queue_.top().second;
      composed_state_queue_.pop();
      ProcessQueueElement(src_composed_state);
    }
    if (composed_state_queue_.empty())
      break;
  }

  fst::Connect(clat_out_);
  TopSortCompactLatticeIfNeeded(clat_out_);

  if (GetVerboseLevel() >= 2) {
    int32 num_arcs_in = TotalNumArcs(clat_in_),
        orig_num_arcs_out = num_arcs_out_,
        num_arcs_out = TotalNumArcs(*clat_out_),
        num_states_in = clat_in_.NumStates(),
        orig_num_states_out = composed_state_info_.size(),
        num_states_out = clat_out_->NumStates();
    std::ostringstream os;
    os << "Input lattice had " << num_arcs_in << '/' << num_states_in
       << " arcs/states; output lattice has " << num_arcs_out << '/'
       << num_states_out;
    if (num_arcs_out != orig_num_arcs_out) {
      os << " (before pruning: " << orig_num_arcs_out << '/'
         << orig_num_states_out << ")";
    }
    if (!composed_state_queue_.empty()) {
      // Below, composed_state_queue_.top().first + lat_best_cost is an
      // expected-cost of the best path from the composed output that we *did
      // not* expand.  This, minus the best cost in the output compact lattice,
      // can be interpreted as the beam that we effecctively pruned the output
      // lattice to.
      BaseFloat effective_beam =
          composed_state_queue_.top().first + lat_best_cost_ - output_best_cost_;
      os << ". Effective beam was " << effective_beam;
    }
    KALDI_VLOG(2) << os.str();
  }

  if (clat_out_->NumStates() == 0) {
    KALDI_WARN << "Composed lattice has no states: something went wrong.";
  }
}

void ComposeCompactLatticePruned(
    const ComposeLatticePrunedOptions &opts,
    const CompactLattice &clat,
    fst::DeterministicOnDemandFst<fst::StdArc> *det_fst,
    CompactLattice* composed_clat) {
  PrunedCompactLatticeComposer composer(opts, clat, det_fst, composed_clat);
  composer.Compose();
}

} // namespace kaldi
