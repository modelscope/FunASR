// lat/determinize-lattice-pruned.cc

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

#include <vector>
#include <climits>
#include "fstext/determinize-lattice.h" // for LatticeStringRepository
#include "fstext/fstext-utils.h"
#include "lat/lattice-functions.h"  // for PruneLattice
//#include "lat/minimize-lattice.h"   // for minimization
//#include "lat/push-lattice.h"       // for minimization
#include "lat/determinize-lattice-pruned.h"

namespace fst {

using std::vector;
using std::pair;
using std::greater;

// class LatticeDeterminizerPruned is templated on the same types that
// CompactLatticeWeight is templated on: the base weight (Weight), typically
// LatticeWeightTpl<float> etc. but could also be e.g. TropicalWeight, and the
// IntType, typically int32, used for the output symbols in the compact
// representation of strings [note: the output symbols would usually be
// p.d.f. id's in the anticipated use of this code] It has a special requirement
// on the Weight type: that there should be a Compare function on the weights
// such that Compare(w1, w2) returns -1 if w1 < w2, 0 if w1 == w2, and +1 if w1 >
// w2.  This requires that there be a total order on the weights.

template<class Weight, class IntType> class LatticeDeterminizerPruned {
 public:
  // Output to Gallic acceptor (so the strings go on weights, and there is a 1-1 correspondence
  // between our states and the states in ofst.  If destroy == true, release memory as we go
  // (but we cannot output again).

  typedef CompactLatticeWeightTpl<Weight, IntType> CompactWeight;
  typedef ArcTpl<CompactWeight> CompactArc; // arc in compact, acceptor form of lattice
  typedef ArcTpl<Weight> Arc; // arc in non-compact version of lattice

  // Output to standard FST with CompactWeightTpl<Weight> as its weight type (the
  // weight stores the original output-symbol strings).  If destroy == true,
  // release memory as we go (but we cannot output again).
  void Output(MutableFst<CompactArc>  *ofst, bool destroy = true) {
    KALDI_ASSERT(determinized_);
    typedef typename Arc::StateId StateId;
    StateId nStates = static_cast<StateId>(output_states_.size());
    if (destroy)
      FreeMostMemory();
    ofst->DeleteStates();
    ofst->SetStart(kNoStateId);
    if (nStates == 0) {
      return;
    }
    for (StateId s = 0;s < nStates;s++) {
      OutputStateId news = ofst->AddState();
      KALDI_ASSERT(news == s);
    }
    ofst->SetStart(0);
    // now process transitions.
    for (StateId this_state_id = 0; this_state_id < nStates; this_state_id++) {
      OutputState &this_state = *(output_states_[this_state_id]);
      vector<TempArc> &this_vec(this_state.arcs);
      typename vector<TempArc>::const_iterator iter = this_vec.begin(), end = this_vec.end();

      for (;iter != end; ++iter) {
        const TempArc &temp_arc(*iter);
        CompactArc new_arc;
        vector<Label> olabel_seq;
        repository_.ConvertToVector(temp_arc.string, &olabel_seq);
        CompactWeight weight(temp_arc.weight, olabel_seq);
        if (temp_arc.nextstate == kNoStateId) {  // is really final weight.
          ofst->SetFinal(this_state_id, weight);
        } else {  // is really an arc.
          new_arc.nextstate = temp_arc.nextstate;
          new_arc.ilabel = temp_arc.ilabel;
          new_arc.olabel = temp_arc.ilabel;  // acceptor.  input == output.
          new_arc.weight = weight;  // includes string and weight.
          ofst->AddArc(this_state_id, new_arc);
        }
      }
      // Free up memory.  Do this inside the loop as ofst is also allocating memory,
      // and we want to reduce the maximum amount ever allocated.
      if (destroy) { vector<TempArc> temp; temp.swap(this_vec); }
    }
    if (destroy) {
      FreeOutputStates();
      repository_.Destroy();
    }
  }

  // Output to standard FST with Weight as its weight type.  We will create extra
  // states to handle sequences of symbols on the output.  If destroy == true,
  // release memory as we go (but we cannot output again).
  void  Output(MutableFst<Arc> *ofst, bool destroy = true) {
    // Outputs to standard fst.
    OutputStateId nStates = static_cast<OutputStateId>(output_states_.size());
    ofst->DeleteStates();
    if (nStates == 0) {
      ofst->SetStart(kNoStateId);
      return;
    }
    if (destroy)
      FreeMostMemory();
    // Add basic states-- but we will add extra ones to account for strings on output.
    for (OutputStateId s = 0; s< nStates;s++) {
      OutputStateId news = ofst->AddState();
      KALDI_ASSERT(news == s);
    }
    ofst->SetStart(0);
    for (OutputStateId this_state_id = 0; this_state_id < nStates; this_state_id++) {
      OutputState &this_state = *(output_states_[this_state_id]);
      vector<TempArc> &this_vec(this_state.arcs);

      typename vector<TempArc>::const_iterator iter = this_vec.begin(), end = this_vec.end();
      for (; iter != end; ++iter) {
        const TempArc &temp_arc(*iter);
        vector<Label> seq;
        repository_.ConvertToVector(temp_arc.string, &seq);

        if (temp_arc.nextstate == kNoStateId) {  // Really a final weight.
          // Make a sequence of states going to a final state, with the strings
          // as labels.  Put the weight on the first arc.
          OutputStateId cur_state = this_state_id;
          for (size_t i = 0; i < seq.size(); i++) {
            OutputStateId next_state = ofst->AddState();
            Arc arc;
            arc.nextstate = next_state;
            arc.weight = (i == 0 ? temp_arc.weight : Weight::One());
            arc.ilabel = 0;  // epsilon.
            arc.olabel = seq[i];
            ofst->AddArc(cur_state, arc);
            cur_state = next_state;
          }
          ofst->SetFinal(cur_state, (seq.size() == 0 ? temp_arc.weight : Weight::One()));
        } else {  // Really an arc.
          OutputStateId cur_state = this_state_id;
          // Have to be careful with this integer comparison (i+1 < seq.size()) because unsigned.
          // i < seq.size()-1 could fail for zero-length sequences.
          for (size_t i = 0; i+1 < seq.size();i++) {
            // for all but the last element of seq, create new state.
            OutputStateId next_state = ofst->AddState();
            Arc arc;
            arc.nextstate = next_state;
            arc.weight = (i == 0 ? temp_arc.weight : Weight::One());
            arc.ilabel = (i == 0 ? temp_arc.ilabel : 0);  // put ilabel on first element of seq.
            arc.olabel = seq[i];
            ofst->AddArc(cur_state, arc);
            cur_state = next_state;
          }
          // Add the final arc in the sequence.
          Arc arc;
          arc.nextstate = temp_arc.nextstate;
          arc.weight = (seq.size() <= 1 ? temp_arc.weight : Weight::One());
          arc.ilabel = (seq.size() <= 1 ? temp_arc.ilabel : 0);
          arc.olabel = (seq.size() > 0 ? seq.back() : 0);
          ofst->AddArc(cur_state, arc);
        }
      }
      // Free up memory.  Do this inside the loop as ofst is also allocating memory
      if (destroy) { vector<TempArc> temp; temp.swap(this_vec); }
    }
    if (destroy) {
      FreeOutputStates();
      repository_.Destroy();
    }
  }


  // Initializer.  After initializing the object you will typically
  // call Determinize() and then call one of the Output functions.
  // Note: ifst.Copy() will generally do a
  // shallow copy.  We do it like this for memory safety, rather than
  // keeping a reference or pointer to ifst_.
  LatticeDeterminizerPruned(const ExpandedFst<Arc> &ifst,
                            double beam,
                            DeterminizeLatticePrunedOptions opts):
      num_arcs_(0), num_elems_(0), ifst_(ifst.Copy()), beam_(beam), opts_(opts),
      equal_(opts_.delta), determinized_(false),
      minimal_hash_(3, hasher_, equal_), initial_hash_(3, hasher_, equal_) {
    KALDI_ASSERT(Weight::Properties() & kIdempotent); // this algorithm won't
    // work correctly otherwise.
  }

  void FreeOutputStates() {
    for (size_t i = 0; i < output_states_.size(); i++)
      delete output_states_[i];
    vector<OutputState*> temp;
    temp.swap(output_states_);
  }

  // frees all memory except the info (in output_states_[ ]->arcs)
  // that we need to output the FST.
  void FreeMostMemory() {
    if (ifst_) {
      delete ifst_;
      ifst_ = NULL;
    }
    { MinimalSubsetHash tmp; tmp.swap(minimal_hash_); }

    for (size_t i = 0; i < output_states_.size(); i++) {
      vector<Element> empty_subset;
      empty_subset.swap(output_states_[i]->minimal_subset);
    }

    for (typename InitialSubsetHash::iterator iter = initial_hash_.begin();
         iter != initial_hash_.end(); ++iter)
      delete iter->first;
    { InitialSubsetHash tmp; tmp.swap(initial_hash_); }
    { vector<char> tmp;  tmp.swap(isymbol_or_final_); }
    { // Free up the queue.  I'm not sure how to make sure all
      // the memory is really freed (no swap() function)... doesn't really
      // matter much though.
      while (!queue_.empty()) {
        Task *t = queue_.top();
        delete t;
        queue_.pop();
      }
    }
    { vector<pair<Label, Element> > tmp; tmp.swap(all_elems_tmp_); }
  }

  ~LatticeDeterminizerPruned() {
    FreeMostMemory();
    FreeOutputStates();
    // rest is deleted by destructors.
  }

  void RebuildRepository() { // rebuild the string repository,
    // freeing stuff we don't need.. we call this when memory usage
    // passes a supplied threshold.  We need to accumulate all the
    // strings we need the repository to "remember", then tell it
    // to clean the repository.
    std::vector<StringId> needed_strings;
    for (size_t i = 0; i < output_states_.size(); i++) {
      AddStrings(output_states_[i]->minimal_subset, &needed_strings);
      for (size_t j = 0; j < output_states_[i]->arcs.size(); j++)
        needed_strings.push_back(output_states_[i]->arcs[j].string);
    }

    { // the queue doesn't allow us access to the underlying vector,
      // so we have to resort to a temporary collection.
      std::vector<Task*> tasks;
      while (!queue_.empty()) {
        Task *task = queue_.top();
        queue_.pop();
        tasks.push_back(task);
        AddStrings(task->subset, &needed_strings);
      }
      for (size_t i = 0; i < tasks.size(); i++)
        queue_.push(tasks[i]);
    }

    // the following loop covers strings present in initial_hash_.
    for (typename InitialSubsetHash::const_iterator
             iter = initial_hash_.begin();
         iter != initial_hash_.end(); ++iter) {
      const vector<Element> &vec = *(iter->first);
      Element elem = iter->second;
      AddStrings(vec, &needed_strings);
      needed_strings.push_back(elem.string);
    }
    std::sort(needed_strings.begin(), needed_strings.end());
    needed_strings.erase(std::unique(needed_strings.begin(),
                                     needed_strings.end()),
                         needed_strings.end()); // uniq the strings.
    KALDI_LOG << "Rebuilding repository.";

    repository_.Rebuild(needed_strings);
  }

  bool CheckMemoryUsage() {
    int32 repo_size = repository_.MemSize(),
        arcs_size = num_arcs_ * sizeof(TempArc),
        elems_size = num_elems_ * sizeof(Element),
        total_size = repo_size + arcs_size + elems_size;
    if (opts_.max_mem > 0 && total_size > opts_.max_mem) { // We passed the memory threshold.
      // This is usually due to the repository getting large, so we
      // clean this out.
      RebuildRepository();
      int32 new_repo_size = repository_.MemSize(),
          new_total_size = new_repo_size + arcs_size + elems_size;

      KALDI_VLOG(2) << "Rebuilt repository in determinize-lattice: repository shrank from "
                    << repo_size << " to " << new_repo_size << " bytes (approximately)";

      if (new_total_size > static_cast<int32>(opts_.max_mem * 0.8)) {
        // Rebuilding didn't help enough-- we need a margin to stop
        // having to rebuild too often.  We'll just return to the user at
        // this point, with a partial lattice that's pruned tighter than
        // the specified beam.  Here we figure out what the effective
        // beam was.
        double effective_beam = beam_;
        if (!queue_.empty()) { // Note: queue should probably not be empty; we're
          // just being paranoid here.
          Task *task = queue_.top();
          double total_weight = backward_costs_[ifst_->Start()]; // best weight of FST.
          effective_beam = task->priority_cost - total_weight;
        }
        KALDI_WARN << "Did not reach requested beam in determinize-lattice: "
                   << "size exceeds maximum " << opts_.max_mem
                   << " bytes; (repo,arcs,elems) = (" << repo_size << ","
                   << arcs_size << "," << elems_size
                   << "), after rebuilding, repo size was " << new_repo_size
                   << ", effective beam was " << effective_beam
                   << " vs. requested beam " << beam_;
        return false;
      }
    }
    return true;
  }

  bool Determinize(double *effective_beam) {
    KALDI_ASSERT(!determinized_);
    // This determinizes the input fst but leaves it in the "special format"
    // in "output_arcs_".  Must be called after Initialize().  To get the
    // output, call one of the Output routines.

    InitializeDeterminization(); // some start-up tasks.
    while (!queue_.empty()) {
      Task *task = queue_.top();
      // Note: the queue contains only tasks that are "within the beam".
      // We also have to check whether we have reached one of the user-specified
      // maximums, of estimated memory, arcs, or states.  The condition for
      // ending is:
      // num-states is more than user specified, OR
      // num-arcs is more than user specified, OR
      // memory passed a user-specified threshold and cleanup failed
      //  to get it below that threshold.
      size_t num_states = output_states_.size();
      if ((opts_.max_states > 0 && num_states > opts_.max_states) ||
          (opts_.max_arcs > 0 && num_arcs_ > opts_.max_arcs) ||
          (num_states % 10 == 0 && !CheckMemoryUsage())) { // note: at some point
        // it was num_states % 100, not num_states % 10, but I encountered an example
        // where memory was exhausted before we reached state #100.
        KALDI_VLOG(1) << "Lattice determinization terminated but not "
                      << " because of lattice-beam.  (#states, #arcs) is ( "
                      << output_states_.size() << ", " << num_arcs_
                      << " ), versus limits ( " << opts_.max_states << ", "
                      << opts_.max_arcs << " ) (else, may be memory limit).";
        break;
        // we terminate the determinization here-- whatever we already expanded is
        // what we'll return...  because we expanded stuff in order of total
        // (forward-backward) weight, the stuff we returned first is the most
        // important.
      }
      queue_.pop();
      ProcessTransition(task->state, task->label, &(task->subset));
      delete task;
    }
    determinized_ = true;
    if (effective_beam != NULL) {
      if (queue_.empty()) *effective_beam = beam_;
      else
        *effective_beam = queue_.top()->priority_cost -
            backward_costs_[ifst_->Start()];
    }
    return (queue_.empty()); // return success if queue was empty, i.e. we processed
    // all tasks and did not break out of the loop early due to reaching a memory,
    // arc or state limit.
  }
 private:

  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;  // use this when we don't know if it's input or output.
  typedef typename Arc::StateId InputStateId;  // state in the input FST.
  typedef typename Arc::StateId OutputStateId;  // same as above but distinguish
                                                // states in output Fst.

  typedef LatticeStringRepository<IntType> StringRepositoryType;
  typedef const typename StringRepositoryType::Entry* StringId;

  // Element of a subset [of original states]
  struct Element {
    StateId state; // use StateId as this is usually InputStateId but in one case
                   // OutputStateId.
    StringId string;
    Weight weight;
    bool operator != (const Element &other) const {
      return (state != other.state || string != other.string ||
              weight != other.weight);
    }
    // This operator is only intended for the priority_queue in the function
    // EpsilonClosure().
    bool operator > (const Element &other) const {
      return state > other.state;
    }
    // This operator is only intended to support sorting in EpsilonClosure()
    bool operator < (const Element &other) const {
      return state < other.state;
    }
  };

  // Arcs in the format we temporarily create in this class (a representation, essentially of
  // a Gallic Fst).
  struct TempArc {
    Label ilabel;
    StringId string;  // Look it up in the StringRepository, it's a sequence of Labels.
    OutputStateId nextstate;  // or kNoState for final weights.
    Weight weight;
  };

  // Hashing function used in hash of subsets.
  // A subset is a pointer to vector<Element>.
  // The Elements are in sorted order on state id, and without repeated states.
  // Because the order of Elements is fixed, we can use a hashing function that is
  // order-dependent.  However the weights are not included in the hashing function--
  // we hash subsets that differ only in weight to the same key.  This is not optimal
  // in terms of the O(N) performance but typically if we have a lot of determinized
  // states that differ only in weight then the input probably was pathological in some way,
  // or even non-determinizable.
  //   We don't quantize the weights, in order to avoid inexactness in simple cases.
  // Instead we apply the delta when comparing subsets for equality, and allow a small
  // difference.

  class SubsetKey {
   public:
    size_t operator ()(const vector<Element> * subset) const {  // hashes only the state and string.
      size_t hash = 0, factor = 1;
      for (typename vector<Element>::const_iterator iter= subset->begin(); iter != subset->end(); ++iter) {
        hash *= factor;
        hash += iter->state + reinterpret_cast<size_t>(iter->string);
        factor *= 23531;  // these numbers are primes.
      }
      return hash;
    }
  };

  // This is the equality operator on subsets.  It checks for exact match on state-id
  // and string, and approximate match on weights.
  class SubsetEqual {
   public:
    bool operator ()(const vector<Element> * s1, const vector<Element> * s2) const {
      size_t sz = s1->size();
      KALDI_ASSERT(sz>=0);
      if (sz != s2->size()) return false;
      typename vector<Element>::const_iterator iter1 = s1->begin(),
          iter1_end = s1->end(), iter2=s2->begin();
      for (; iter1 < iter1_end; ++iter1, ++iter2) {
        if (iter1->state != iter2->state ||
           iter1->string != iter2->string ||
            ! ApproxEqual(iter1->weight, iter2->weight, delta_)) return false;
      }
      return true;
    }
    float delta_;
    SubsetEqual(float delta): delta_(delta) {}
    SubsetEqual(): delta_(kDelta) {}
  };

  // Operator that says whether two Elements have the same states.
  // Used only for debug.
  class SubsetEqualStates {
   public:
    bool operator ()(const vector<Element> * s1, const vector<Element> * s2) const {
      size_t sz = s1->size();
      KALDI_ASSERT(sz>=0);
      if (sz != s2->size()) return false;
      typename vector<Element>::const_iterator iter1 = s1->begin(),
          iter1_end = s1->end(), iter2=s2->begin();
      for (; iter1 < iter1_end; ++iter1, ++iter2) {
        if (iter1->state != iter2->state) return false;
      }
      return true;
    }
  };

  // Define the hash type we use to map subsets (in minimal
  // representation) to OutputStateId.
  typedef unordered_map<const vector<Element>*, OutputStateId,
                        SubsetKey, SubsetEqual> MinimalSubsetHash;

  // Define the hash type we use to map subsets (in initial
  // representation) to OutputStateId, together with an
  // extra weight. [note: we interpret the Element.state in here
  // as an OutputStateId even though it's declared as InputStateId;
  // these types are the same anyway].
  typedef unordered_map<const vector<Element>*, Element,
                        SubsetKey, SubsetEqual> InitialSubsetHash;


  // converts the representation of the subset from canonical (all states) to
  // minimal (only states with output symbols on arcs leaving them, and final
  // states).  Output is not necessarily normalized, even if input_subset was.
  void ConvertToMinimal(vector<Element> *subset) {
    KALDI_ASSERT(!subset->empty());
    typename vector<Element>::iterator cur_in = subset->begin(),
        cur_out = subset->begin(), end = subset->end();
    while (cur_in != end) {
      if(IsIsymbolOrFinal(cur_in->state)) {  // keep it...
        *cur_out = *cur_in;
        cur_out++;
      }
      cur_in++;
    }
    subset->resize(cur_out - subset->begin());
  }

  // Takes a minimal, normalized subset, and converts it to an OutputStateId.
  // Involves a hash lookup, and possibly adding a new OutputStateId.
  // If it creates a new OutputStateId, it creates a new record for it, works
  // out its final-weight, and puts stuff on the queue relating to its
  // transitions.
  OutputStateId MinimalToStateId(const vector<Element> &subset,
                                 const double forward_cost) {
    typename MinimalSubsetHash::const_iterator iter
        = minimal_hash_.find(&subset);
    if (iter != minimal_hash_.end()) { // Found a matching subset.
      OutputStateId state_id = iter->second;
      const OutputState &state = *(output_states_[state_id]);
      // Below is just a check that the algorithm is working...
      if (forward_cost < state.forward_cost - 0.1) {
        // for large weights, this check could fail due to roundoff.
        KALDI_WARN << "New cost is less (check the difference is small) "
                   << forward_cost << ", "
                   << state.forward_cost;
      }
      return state_id;
    }
    OutputStateId state_id = static_cast<OutputStateId>(output_states_.size());
    OutputState *new_state = new OutputState(subset, forward_cost);
    minimal_hash_[&(new_state->minimal_subset)] = state_id;
    output_states_.push_back(new_state);
    num_elems_ += subset.size();
    // Note: in the previous algorithm, we pushed the new state-id onto the queue
    // at this point.  Here, the queue happens elsewhere, and we directly process
    // the state (which result in stuff getting added to the queue).
    ProcessFinal(state_id); // will work out the final-prob.
    ProcessTransitions(state_id); // will process transitions and add stuff to the queue.
    return state_id;
  }


  // Given a normalized initial subset of elements (i.e. before epsilon closure),
  // compute the corresponding output-state.
  OutputStateId InitialToStateId(const vector<Element> &subset_in,
                                 double forward_cost,
                                 Weight *remaining_weight,
                                 StringId *common_prefix) {
    typename InitialSubsetHash::const_iterator iter
        = initial_hash_.find(&subset_in);
    if (iter != initial_hash_.end()) { // Found a matching subset.
      const Element &elem = iter->second;
      *remaining_weight = elem.weight;
      *common_prefix = elem.string;
      if (elem.weight == Weight::Zero())
        KALDI_WARN << "Zero weight!";
      return elem.state;
    }
    // else no matching subset-- have to work it out.
    vector<Element> subset(subset_in);
    // Follow through epsilons.  Will add no duplicate states.  note: after
    // EpsilonClosure, it is the same as "canonical" subset, except not
    // normalized (actually we never compute the normalized canonical subset,
    // only the normalized minimal one).
    EpsilonClosure(&subset); // follow epsilons.
    ConvertToMinimal(&subset); // remove all but emitting and final states.

    Element elem; // will be used to store remaining weight and string, and
                 // OutputStateId, in initial_hash_;
    NormalizeSubset(&subset, &elem.weight, &elem.string); // normalize subset; put
    // common string and weight in "elem".  The subset is now a minimal,
    // normalized subset.

    forward_cost += ConvertToCost(elem.weight);
    OutputStateId ans = MinimalToStateId(subset, forward_cost);
    *remaining_weight = elem.weight;
    *common_prefix = elem.string;
    if (elem.weight == Weight::Zero())
      KALDI_WARN << "Zero weight!";

    // Before returning "ans", add the initial subset to the hash,
    // so that we can bypass the epsilon-closure etc., next time
    // we process the same initial subset.
    vector<Element> *initial_subset_ptr = new vector<Element>(subset_in);
    elem.state = ans;
    initial_hash_[initial_subset_ptr] = elem;
    num_elems_ += initial_subset_ptr->size(); // keep track of memory usage.
    return ans;
  }

  // returns the Compare value (-1 if a < b, 0 if a == b, 1 if a > b) according
  // to the ordering we defined on strings for the CompactLatticeWeightTpl.
  // see function
  // inline int Compare (const CompactLatticeWeightTpl<WeightType,IntType> &w1,
  //                     const CompactLatticeWeightTpl<WeightType,IntType> &w2)
  // in lattice-weight.h.
  // this is the same as that, but optimized for our data structures.
  inline int Compare(const Weight &a_w, StringId a_str,
                     const Weight &b_w, StringId b_str) const {
    int weight_comp = fst::Compare(a_w, b_w);
    if (weight_comp != 0) return weight_comp;
    // now comparing strings.
    if (a_str == b_str) return 0;
    vector<IntType> a_vec, b_vec;
    repository_.ConvertToVector(a_str, &a_vec);
    repository_.ConvertToVector(b_str, &b_vec);
    // First compare their lengths.
    int a_len = a_vec.size(), b_len = b_vec.size();
    // use opposite order on the string lengths (c.f. Compare in
    // lattice-weight.h)
    if (a_len > b_len) return -1;
    else if (a_len < b_len) return 1;
    for(int i = 0; i < a_len; i++) {
      if (a_vec[i] < b_vec[i]) return -1;
      else if (a_vec[i] > b_vec[i]) return 1;
    }
    KALDI_ASSERT(0); // because we checked if a_str == b_str above, shouldn't reach here
    return 0;
  }

  // This function computes epsilon closure of subset of states by following epsilon links.
  // Called by InitialToStateId and Initialize.
  // Has no side effects except on the string repository.  The "output_subset" is not
  // necessarily normalized (in the sense of there being no common substring), unless
  // input_subset was.
  void EpsilonClosure(vector<Element> *subset) {
    // at input, subset must have only one example of each StateId.  [will still
    // be so at output].  This function follows input-epsilons, and augments the
    // subset accordingly.

    std::priority_queue<Element, vector<Element>, greater<Element> > queue;
    unordered_map<InputStateId, Element> cur_subset;
    typedef typename unordered_map<InputStateId, Element>::iterator MapIter;
    typedef typename vector<Element>::const_iterator VecIter;

    for (VecIter iter = subset->begin(); iter != subset->end(); ++iter) {
      queue.push(*iter);
      cur_subset[iter->state] = *iter;
    }

    // find whether input fst is known to be sorted on input label.
    bool sorted = ((ifst_->Properties(kILabelSorted, false) & kILabelSorted) != 0);
    bool replaced_elems = false; // relates to an optimization, see below.
    int counter = 0; // stops infinite loops here for non-lattice-determinizable input
    // (e.g. input with negative-cost epsilon loops); useful in testing.
    while (queue.size() != 0) {
      Element elem = queue.top();
      queue.pop();

      // The next if-statement is a kind of optimization.  It's to prevent us
      // unnecessarily repeating the processing of a state.  "cur_subset" always
      // contains only one Element with a particular state.  The issue is that
      // whenever we modify the Element corresponding to that state in "cur_subset",
      // both the new (optimal) and old (less-optimal) Element will still be in
      // "queue".  The next if-statement stops us from wasting compute by
      // processing the old Element.
      if (replaced_elems && cur_subset[elem.state] != elem)
        continue;
      if (opts_.max_loop > 0 && counter++ > opts_.max_loop) {
        KALDI_ERR << "Lattice determinization aborted since looped more than "
                  << opts_.max_loop << " times during epsilon closure.";
      }
      for (ArcIterator<ExpandedFst<Arc> > aiter(*ifst_, elem.state); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (sorted && arc.ilabel != 0) break;  // Break from the loop: due to sorting there will be no
        // more transitions with epsilons as input labels.
        if (arc.ilabel == 0
            && arc.weight != Weight::Zero()) {  // Epsilon transition.
          Element next_elem;
          next_elem.state = arc.nextstate;
          next_elem.weight = Times(elem.weight, arc.weight);
          // next_elem.string is not set up yet... create it only
          // when we know we need it (this is an optimization)

          MapIter iter = cur_subset.find(next_elem.state);
          if (iter == cur_subset.end()) {
            // was no such StateId: insert and add to queue.
            next_elem.string = (arc.olabel == 0 ? elem.string :
                                repository_.Successor(elem.string, arc.olabel));
            cur_subset[next_elem.state] = next_elem;
            queue.push(next_elem);
          } else {
            // was not inserted because one already there.  In normal
            // determinization we'd add the weights.  Here, we find which one
            // has the better weight, and keep its corresponding string.
            int comp = fst::Compare(next_elem.weight, iter->second.weight);
            if (comp == 0) { // A tie on weights.  This should be a rare case;
                             // we don't optimize for it.
              next_elem.string = (arc.olabel == 0 ? elem.string :
                                  repository_.Successor(elem.string,
                                                        arc.olabel));
              comp = Compare(next_elem.weight, next_elem.string,
                             iter->second.weight, iter->second.string);
            }
            if(comp == 1) { // next_elem is better, so use its (weight, string)
              next_elem.string = (arc.olabel == 0 ? elem.string :
                                  repository_.Successor(elem.string, arc.olabel));
              iter->second.string = next_elem.string;
              iter->second.weight = next_elem.weight;
              queue.push(next_elem);
              replaced_elems = true;
            }
            // else it is the same or worse, so use original one.
          }
        }
      }
    }

    { // copy cur_subset to subset.
      subset->clear();
      subset->reserve(cur_subset.size());
      MapIter iter = cur_subset.begin(), end = cur_subset.end();
      for (; iter != end; ++iter) subset->push_back(iter->second);
      // sort by state ID, because the subset hash function is order-dependent(see SubsetKey)
      std::sort(subset->begin(), subset->end());
    }
  }


  // This function works out the final-weight of the determinized state.
  // called by ProcessSubset.
  // Has no side effects except on the variable repository_, and
  // output_states_[output_state_id].arcs

  void ProcessFinal(OutputStateId output_state_id) {
    OutputState &state = *(output_states_[output_state_id]);
    const vector<Element> &minimal_subset = state.minimal_subset;
    // processes final-weights for this subset.  state.minimal_subset_ may be
    // empty if the graphs is not connected/trimmed, I think, do don't check
    // that it's nonempty.
    StringId final_string = repository_.EmptyString();  // set it to keep the
    // compiler happy; if it doesn't get set in the loop, we won't use the value anyway.
    Weight final_weight = Weight::Zero();
    bool is_final = false;
    typename vector<Element>::const_iterator iter = minimal_subset.begin(), end = minimal_subset.end();
    for (; iter != end; ++iter) {
      const Element &elem = *iter;
      Weight this_final_weight = Times(elem.weight, ifst_->Final(elem.state));
      StringId this_final_string = elem.string;
      if (this_final_weight != Weight::Zero() &&
         (!is_final || Compare(this_final_weight, this_final_string,
                               final_weight, final_string) == 1)) { // the new
        // (weight, string) pair is more in semiring than our current
        // one.
        is_final = true;
        final_weight = this_final_weight;
        final_string = this_final_string;
      }
    }
    if (is_final &&
        ConvertToCost(final_weight) + state.forward_cost <= cutoff_) {
      // store final weights in TempArc structure, just like a transition.
      // Note: we only store the final-weight if it's inside the pruning beam, hence
      // the stuff with Compare.
      TempArc temp_arc;
      temp_arc.ilabel = 0;
      temp_arc.nextstate = kNoStateId;  // special marker meaning "final weight".
      temp_arc.string = final_string;
      temp_arc.weight = final_weight;
      state.arcs.push_back(temp_arc);
      num_arcs_++;
    }
  }

  // NormalizeSubset normalizes the subset "elems" by
  // removing any common string prefix (putting it in common_str),
  // and dividing by the total weight (putting it in tot_weight).
  void NormalizeSubset(vector<Element> *elems,
                       Weight *tot_weight,
                       StringId *common_str) {
    if(elems->empty()) { // just set common_str, tot_weight
      // to defaults and return...
      KALDI_WARN << "empty subset";
      *common_str = repository_.EmptyString();
      *tot_weight = Weight::Zero();
      return;
    }
    size_t size = elems->size();
    vector<IntType> common_prefix;
    repository_.ConvertToVector((*elems)[0].string, &common_prefix);
    Weight weight = (*elems)[0].weight;
    for(size_t i = 1; i < size; i++) {
      weight = Plus(weight, (*elems)[i].weight);
      repository_.ReduceToCommonPrefix((*elems)[i].string, &common_prefix);
    }
    KALDI_ASSERT(weight != Weight::Zero()); // we made sure to ignore arcs with zero
    // weights on them, so we shouldn't have zero here.
    size_t prefix_len = common_prefix.size();
    for(size_t i = 0; i < size; i++) {
      (*elems)[i].weight = Divide((*elems)[i].weight, weight, DIVIDE_LEFT);
      (*elems)[i].string =
          repository_.RemovePrefix((*elems)[i].string, prefix_len);
    }
    *common_str = repository_.ConvertFromVector(common_prefix);
    *tot_weight = weight;
  }

  // Take a subset of Elements that is sorted on state, and
  // merge any Elements that have the same state (taking the best
  // (weight, string) pair in the semiring).
  void MakeSubsetUnique(vector<Element> *subset) {
    typedef typename vector<Element>::iterator IterType;

    // This KALDI_ASSERT is designed to fail (usually) if the subset is not sorted on
    // state.
    KALDI_ASSERT(subset->size() < 2 || (*subset)[0].state <= (*subset)[1].state);

    IterType cur_in = subset->begin(), cur_out = cur_in, end = subset->end();
    size_t num_out = 0;
    // Merge elements with same state-id
    while (cur_in != end) {  // while we have more elements to process.
      // At this point, cur_out points to location of next place we want to put an element,
      // cur_in points to location of next element we want to process.
      if (cur_in != cur_out) *cur_out = *cur_in;
      cur_in++;
      while (cur_in != end && cur_in->state == cur_out->state) {
        if (Compare(cur_in->weight, cur_in->string,
                   cur_out->weight, cur_out->string) == 1) {
          // if *cur_in > *cur_out in semiring, then take *cur_in.
          cur_out->string = cur_in->string;
          cur_out->weight = cur_in->weight;
        }
        cur_in++;
      }
      cur_out++;
      num_out++;
    }
    subset->resize(num_out);
  }

  // ProcessTransition was called from "ProcessTransitions" in the non-pruned
  // code, but now we in effect put the calls to ProcessTransition on a priority
  // queue, and it now gets called directly from Determinize().  This function
  // processes a transition from state "ostate_id".  The set "subset" of Elements
  // represents a set of next-states with associated weights and strings, each
  // one arising from an arc from some state in a determinized-state; the
  // next-states are unique (there is only one Entry assocated with each)
  void ProcessTransition(OutputStateId ostate_id, Label ilabel, vector<Element> *subset) {

    double forward_cost = output_states_[ostate_id]->forward_cost;
    StringId common_str;
    Weight tot_weight;
    NormalizeSubset(subset, &tot_weight, &common_str);
    forward_cost += ConvertToCost(tot_weight);

    OutputStateId nextstate;
    {
      Weight next_tot_weight;
      StringId next_common_str;
      nextstate = InitialToStateId(*subset,
                                   forward_cost,
                                   &next_tot_weight,
                                   &next_common_str);
      common_str = repository_.Concatenate(common_str, next_common_str);
      tot_weight = Times(tot_weight, next_tot_weight);
    }

    // Now add an arc to the next state (would have been created if necessary by
    // InitialToStateId).
    TempArc temp_arc;
    temp_arc.ilabel = ilabel;
    temp_arc.nextstate = nextstate;
    temp_arc.string = common_str;
    temp_arc.weight = tot_weight;
    output_states_[ostate_id]->arcs.push_back(temp_arc);  // record the arc.
    num_arcs_++;
  }


  // "less than" operator for pair<Label, Element>.   Used in ProcessTransitions.
  // Lexicographical order, which only compares the state when ordering the
  // "Element" member of the pair.

  class PairComparator {
   public:
    inline bool operator () (const pair<Label, Element> &p1, const pair<Label, Element> &p2) {
      if (p1.first < p2.first) return true;
      else if (p1.first > p2.first) return false;
      else {
        return p1.second.state < p2.second.state;
      }
    }
  };


  // ProcessTransitions processes emitting transitions (transitions with
  // ilabels) out of this subset of states.  It actualy only creates records
  // ("Task") that get added to the queue.  The transitions will be processed in
  // priority order from Determinize().  This function soes not consider final
  // states.  Partitions the emitting transitions up by ilabel (by sorting on
  // ilabel), and for each unique ilabel, it creates a Task record that contains
  // the information we need to process the transition.

  void ProcessTransitions(OutputStateId output_state_id) {
    const vector<Element> &minimal_subset = output_states_[output_state_id]->minimal_subset;
    // it's possible that minimal_subset could be empty if there are
    // unreachable parts of the graph, so don't check that it's nonempty.
    vector<pair<Label, Element> > &all_elems(all_elems_tmp_); // use class member
    // to avoid memory allocation/deallocation.
    {
      // Push back into "all_elems", elements corresponding to all
      // non-epsilon-input transitions out of all states in "minimal_subset".
      typename vector<Element>::const_iterator iter = minimal_subset.begin(), end = minimal_subset.end();
      for (;iter != end; ++iter) {
        const Element &elem = *iter;
        for (ArcIterator<ExpandedFst<Arc> > aiter(*ifst_, elem.state); ! aiter.Done(); aiter.Next()) {
          const Arc &arc = aiter.Value();
          if (arc.ilabel != 0
              && arc.weight != Weight::Zero()) {  // Non-epsilon transition -- ignore epsilons here.
            pair<Label, Element> this_pr;
            this_pr.first = arc.ilabel;
            Element &next_elem(this_pr.second);
            next_elem.state = arc.nextstate;
            next_elem.weight = Times(elem.weight, arc.weight);
            if (arc.olabel == 0) // output epsilon
              next_elem.string = elem.string;
            else
              next_elem.string = repository_.Successor(elem.string, arc.olabel);
            all_elems.push_back(this_pr);
          }
        }
      }
    }
    PairComparator pc;
    std::sort(all_elems.begin(), all_elems.end(), pc);
    // now sorted first on input label, then on state.
    typedef typename vector<pair<Label, Element> >::const_iterator PairIter;
    PairIter cur = all_elems.begin(), end = all_elems.end();
    while (cur != end) {
      // The old code (non-pruned) called ProcessTransition; here, instead,
      // we'll put the calls into a priority queue.
      Task *task = new Task;
      // Process ranges that share the same input symbol.
      Label ilabel = cur->first;
      task->state = output_state_id;
      task->priority_cost = std::numeric_limits<double>::infinity();
      task->label = ilabel;
      while (cur != end && cur->first == ilabel) {
        task->subset.push_back(cur->second);
        const Element &element = cur->second;
        // Note: we'll later include the term "forward_cost" in the
        // priority_cost.
        task->priority_cost = std::min(task->priority_cost,
                                       ConvertToCost(element.weight) +
                                       backward_costs_[element.state]);
        cur++;
      }

      // After the command below, the "priority_cost" is a value comparable to
      // the total-weight of the input FST, like a total-path weight... of
      // course, it will typically be less (in the semiring) than that.
      // note: we represent it just as a double.
      task->priority_cost += output_states_[output_state_id]->forward_cost;

      if (task->priority_cost > cutoff_) {
        // This task would never get done as it's past the pruning cutoff.
        delete task;
      } else {
        MakeSubsetUnique(&(task->subset)); // remove duplicate Elements with the same state.
        queue_.push(task); // Push the task onto the queue.  The queue keeps it
        // in prioritized order, so we always process the one with the "best"
        // weight (highest in the semiring).

        { // this is a check.
          double best_cost = backward_costs_[ifst_->Start()],
              tolerance = 0.01 + 1.0e-04 * std::abs(best_cost);
          if (task->priority_cost < best_cost - tolerance) {
            KALDI_WARN << "Cost below best cost was encountered:"
                       << task->priority_cost << " < " << best_cost;
          }
        }
      }
    }
    all_elems.clear(); // as it's a reference to a class variable; we want it to stay
    // empty.
  }


  bool IsIsymbolOrFinal(InputStateId state) { // returns true if this state
    // of the input FST either is final or has an osymbol on an arc out of it.
    // Uses the vector isymbol_or_final_ as a cache for this info.
    KALDI_ASSERT(state >= 0);
    if (isymbol_or_final_.size() <= state)
      isymbol_or_final_.resize(state+1, static_cast<char>(OSF_UNKNOWN));
    if (isymbol_or_final_[state] == static_cast<char>(OSF_NO))
      return false;
    else if (isymbol_or_final_[state] == static_cast<char>(OSF_YES))
      return true;
    // else work it out...
    isymbol_or_final_[state] = static_cast<char>(OSF_NO);
    if (ifst_->Final(state) != Weight::Zero())
      isymbol_or_final_[state] = static_cast<char>(OSF_YES);
    for (ArcIterator<ExpandedFst<Arc> > aiter(*ifst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0 && arc.weight != Weight::Zero()) {
        isymbol_or_final_[state] = static_cast<char>(OSF_YES);
        return true;
      }
    }
    return IsIsymbolOrFinal(state); // will only recurse once.
  }

  void ComputeBackwardWeight() {
    // Sets up the backward_costs_ array, and the cutoff_ variable.
    KALDI_ASSERT(beam_ > 0);

    // Only handle the toplogically sorted case.
    backward_costs_.resize(ifst_->NumStates());
    for (StateId s = ifst_->NumStates() - 1; s >= 0; s--) {
      double &cost = backward_costs_[s];
      cost = ConvertToCost(ifst_->Final(s));
      for (ArcIterator<ExpandedFst<Arc> > aiter(*ifst_, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        cost = std::min(cost,
                        ConvertToCost(arc.weight) + backward_costs_[arc.nextstate]);
      }
    }

    if (ifst_->Start() == kNoStateId) return; // we'll be returning
    // an empty FST.

    double best_cost = backward_costs_[ifst_->Start()];
    if (best_cost == std::numeric_limits<double>::infinity())
      KALDI_WARN << "Total weight of input lattice is zero.";
    cutoff_ = best_cost + beam_;
  }

  void InitializeDeterminization() {
    // We insist that the input lattice be topologically sorted.  This is not a
    // fundamental limitation of the algorithm (which in principle should be
    // applicable to even cyclic FSTs), but it helps us more efficiently
    // compute the backward_costs_ array.  There may be some other reason we
    // require this, that escapes me at the moment.
    KALDI_ASSERT(ifst_->Properties(kTopSorted, true) != 0);
    ComputeBackwardWeight();
#if !(__GNUC__ == 4 && __GNUC_MINOR__ == 0)
    if(ifst_->Properties(kExpanded, false) != 0) { // if we know the number of
      // states in ifst_, it might be a bit more efficient
      // to pre-size the hashes so we're not constantly rebuilding them.
      StateId num_states =
          down_cast<const ExpandedFst<Arc>*, const Fst<Arc> >(ifst_)->NumStates();
      minimal_hash_.rehash(num_states/2 + 3);
      initial_hash_.rehash(num_states/2 + 3);
    }
#endif
    InputStateId start_id = ifst_->Start();
    if (start_id != kNoStateId) {
      /* Create determinized-state corresponding to the start state....
         Unlike all the other states, we don't "normalize" the representation
         of this determinized-state before we put it into minimal_hash_.  This is actually
         what we want, as otherwise we'd have problems dealing with any extra weight
         and string and might have to create a "super-initial" state which would make
         the output nondeterministic.  Normalization is only needed to make the
         determinized output more minimal anyway, it's not needed for correctness.
         Note, we don't put anything in the initial_hash_.  The initial_hash_ is only
         a lookaside buffer anyway, so this isn't a problem-- it will get populated
         later if it needs to be.
      */
      vector<Element> subset(1);
      subset[0].state = start_id;
      subset[0].weight = Weight::One();
      subset[0].string = repository_.EmptyString();  // Id of empty sequence.
      EpsilonClosure(&subset); // follow through epsilon-input links
      ConvertToMinimal(&subset); // remove all but final states and
      // states with input-labels on arcs out of them.
      // Weight::One() is the "forward-weight" of this determinized state...
      // i.e. the minimal cost from the start of the determinized FST to this
      // state [One() because it's the start state].
      OutputState *initial_state = new OutputState(subset, 0);
      KALDI_ASSERT(output_states_.empty());
      output_states_.push_back(initial_state);
      num_elems_ += subset.size();
      OutputStateId initial_state_id = 0;
      minimal_hash_[&(initial_state->minimal_subset)] = initial_state_id;
      ProcessFinal(initial_state_id);
      ProcessTransitions(initial_state_id); // this will add tasks to
      // the queue, which we'll start processing in Determinize().
    }
  }

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeDeterminizerPruned);

  struct OutputState {
    vector<Element> minimal_subset;
    vector<TempArc> arcs; // arcs out of the state-- those that have been processed.
    // Note: the final-weight is included here with kNoStateId as the state id.  We
    // always process the final-weight regardless of the beam; when producing the
    // output we may have to ignore some of these.
    double forward_cost; // Represents minimal cost from start-state
    // to this state.  Used in prioritization of tasks, and pruning.
    // Note: we know this minimal cost from when we first create the OutputState;
    // this is because of the priority-queue we use, that ensures that the
    // "best" path into the state will be expanded first.
    OutputState(const vector<Element> &minimal_subset,
                double forward_cost): minimal_subset(minimal_subset),
                                      forward_cost(forward_cost) { }
  };

  vector<OutputState*> output_states_; // All the info about the output states.

  int num_arcs_; // keep track of memory usage: number of arcs in output_states_[ ]->arcs
  int num_elems_; // keep track of memory usage: number of elems in output_states_ and
  // the keys of initial_hash_

  const ExpandedFst<Arc> *ifst_;
  std::vector<double> backward_costs_; // This vector stores, for every state in ifst_,
  // the minimal cost to the end-state (i.e. the sum of weights; they are guaranteed to
  // have "take-the-minimum" semantics).  We get the double from the ConvertToCost()
  // function on the lattice weights.

  double beam_;
  double cutoff_; // beam plus total-weight of input (and note, the weight is
  // guaranteed to be "tropical-like" so the sum does represent a min-cost.

  DeterminizeLatticePrunedOptions opts_;
  SubsetKey hasher_;  // object that computes keys-- has no data members.
  SubsetEqual equal_;  // object that compares subsets-- only data member is delta_.
  bool determinized_; // set to true when user called Determinize(); used to make
  // sure this object is used correctly.
  MinimalSubsetHash minimal_hash_;  // hash from Subset to OutputStateId.  Subset is "minimal
                                    // representation" (only include final and states and states with
                                    // nonzero ilabel on arc out of them.  Owns the pointers
                                    // in its keys.
  InitialSubsetHash initial_hash_;   // hash from Subset to Element, which
                                     // represents the OutputStateId together
                                     // with an extra weight and string.  Subset
                                     // is "initial representation".  The extra
                                     // weight and string is needed because after
                                     // we convert to minimal representation and
                                     // normalize, there may be an extra weight
                                     // and string.  Owns the pointers
                                     // in its keys.

  struct Task {
    OutputStateId state; // State from which we're processing the transition.
    Label label; // Label on the transition we're processing out of this state.
    vector<Element> subset; // Weighted subset of states (with strings)-- not normalized.
    double priority_cost; // Cost used in deciding priority of tasks.  Note:
    // we assume there is a ConvertToCost() function that converts the semiring to double.
  };

  struct TaskCompare {
    inline int operator() (const Task *t1, const Task *t2) {
      // view this like operator <, which is the default template parameter
      // to std::priority_queue.
      // returns true if t1 is worse than t2.
      return (t1->priority_cost > t2->priority_cost);
    }
  };

  // This priority queue contains "Task"s to be processed; these correspond
  // to transitions out of determinized states.  We process these in priority
  // order according to the best weight of any path passing through these
  // determinized states... it's possible to work this out.
  std::priority_queue<Task*, vector<Task*>, TaskCompare> queue_;

  vector<pair<Label, Element> > all_elems_tmp_; // temporary vector used in ProcessTransitions.

  enum IsymbolOrFinal { OSF_UNKNOWN = 0, OSF_NO = 1, OSF_YES = 2 };

  vector<char> isymbol_or_final_; // A kind of cache; it says whether
  // each state is (emitting or final) where emitting means it has at least one
  // non-epsilon output arc.  Only accessed by IsIsymbolOrFinal()

  LatticeStringRepository<IntType> repository_;  // defines a compact and fast way of
  // storing sequences of labels.

  void AddStrings(const vector<Element> &vec,
                  vector<StringId> *needed_strings) {
    for (typename std::vector<Element>::const_iterator iter = vec.begin();
         iter != vec.end(); ++iter)
      needed_strings->push_back(iter->string);
  }
};


// normally Weight would be LatticeWeight<float> (which has two floats),
// or possibly TropicalWeightTpl<float>, and IntType would be int32.
// Caution: there are two versions of the function DeterminizeLatticePruned,
// with identical code but different output FST types.
template<class Weight, class IntType>
bool DeterminizeLatticePruned(
    const ExpandedFst<ArcTpl<Weight> >&ifst,
    double beam,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > >*ofst,
    DeterminizeLatticePrunedOptions opts) {
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  if (ifst.NumStates() == 0) {
    ofst->DeleteStates();
    return true;
  }
  KALDI_ASSERT(opts.retry_cutoff >= 0.0 && opts.retry_cutoff < 1.0);
  int32 max_num_iters = 10;  // avoid the potential for infinite loops if
                             // retrying.
  VectorFst<ArcTpl<Weight> > temp_fst;

  for (int32 iter = 0; iter < max_num_iters; iter++) {
    LatticeDeterminizerPruned<Weight, IntType> det(iter == 0 ? ifst : temp_fst,
                                                   beam, opts);
    double effective_beam;
    bool ans = det.Determinize(&effective_beam);
    // if it returns false it will typically still produce reasonable output,
    // just with a narrower beam than "beam".  If the user specifies an infinite
    // beam we don't do this beam-narrowing.
    if (effective_beam >= beam * opts.retry_cutoff ||
        beam == std::numeric_limits<double>::infinity() ||
        iter + 1 == max_num_iters) {
      det.Output(ofst);
      return ans;
    } else {
      // The code below to set "beam" is a heuristic.
      // If effective_beam is very small, we want to reduce by a lot.
      // But never change the beam by more than a factor of two.
      if (effective_beam < 0.0) effective_beam = 0.0;
      double new_beam = beam * sqrt(effective_beam / beam);
      if (new_beam < 0.5 * beam) new_beam = 0.5 * beam;
      beam = new_beam;
      if (iter == 0) temp_fst = ifst;
      kaldi::PruneLattice(beam, &temp_fst);
      KALDI_LOG << "Pruned state-level lattice with beam " << beam
                << " and retrying determinization with that beam.";
    }
  }
  return false; // Suppress compiler warning; this code is unreachable.
}


// normally Weight would be LatticeWeight<float> (which has two floats),
// or possibly TropicalWeightTpl<float>, and IntType would be int32.
// Caution: there are two versions of the function DeterminizeLatticePruned,
// with identical code but different output FST types.
template<class Weight>
bool DeterminizeLatticePruned(const ExpandedFst<ArcTpl<Weight> > &ifst,
                              double beam,
                              MutableFst<ArcTpl<Weight> > *ofst,
                              DeterminizeLatticePrunedOptions opts) {
  typedef int32 IntType;
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  KALDI_ASSERT(opts.retry_cutoff >= 0.0 && opts.retry_cutoff < 1.0);
  if (ifst.NumStates() == 0) {
    ofst->DeleteStates();
    return true;
  }
  int32 max_num_iters = 10;  // avoid the potential for infinite loops if
                             // retrying.
  VectorFst<ArcTpl<Weight> > temp_fst;

  for (int32 iter = 0; iter < max_num_iters; iter++) {
    LatticeDeterminizerPruned<Weight, IntType> det(iter == 0 ? ifst : temp_fst,
                                                   beam, opts);
    double effective_beam;
    bool ans = det.Determinize(&effective_beam);
    // if it returns false it will typically still
    // produce reasonable output, just with a
    // narrower beam than "beam".
    if (effective_beam >= beam * opts.retry_cutoff ||
        iter + 1 == max_num_iters) {
      det.Output(ofst);
      return ans;
    } else {
      // The code below to set "beam" is a heuristic.
      // If effective_beam is very small, we want to reduce by a lot.
      // But never change the beam by more than a factor of two.
      if (effective_beam < 0)
        effective_beam = 0;
      double new_beam = beam * sqrt(effective_beam / beam);
      if (new_beam < 0.5 * beam) new_beam = 0.5 * beam;
      KALDI_WARN << "Effective beam " << effective_beam << " was less than beam "
                 << beam << " * cutoff " << opts.retry_cutoff << ", pruning raw "
                 << "lattice with new beam " << new_beam << " and retrying.";
      beam = new_beam;
      if (iter == 0) temp_fst = ifst;
      kaldi::PruneLattice(beam, &temp_fst);
    }
  }
  return false; // Suppress compiler warning; this code is unreachable.
}

//template<class Weight>
//typename ArcTpl<Weight>::Label DeterminizeLatticeInsertPhones(
//    const kaldi::TransitionInformation &trans_model,
//    MutableFst<ArcTpl<Weight> > *fst) {
//  // Define some types.
//  typedef ArcTpl<Weight> Arc;
//  typedef typename Arc::StateId StateId;
//  typedef typename Arc::Label Label;
//
//  // Work out the first phone symbol. This is more related to the phone
//  // insertion function, so we put it here and make it the returning value of
//  // DeterminizeLatticeInsertPhones().
//  Label first_phone_label = HighestNumberedInputSymbol(*fst) + 1;
//
//  // Insert phones here.
//  for (StateIterator<MutableFst<Arc> > siter(*fst);
//       !siter.Done(); siter.Next()) {
//    StateId state = siter.Value();
//    if (state == fst->Start())
//      continue;
//    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, state);
//         !aiter.Done(); aiter.Next()) {
//      Arc arc = aiter.Value();
//
//      // Note: the words are on the input symbol side and transition-id's are on
//      // the output symbol side.
//      if ((arc.olabel != 0)
//          && (trans_model.TransitionIdIsStartOfPhone(arc.olabel))
//          && (!trans_model.IsSelfLoop(arc.olabel))) {
//        Label phone =
//            static_cast<Label>(trans_model.TransitionIdToPhone(arc.olabel));
//
//        // Skips <eps>.
//        KALDI_ASSERT(phone != 0);
//
//        if (arc.ilabel == 0) {
//          // If there is no word on the arc, insert the phone directly.
//          arc.ilabel = first_phone_label + phone;
//        } else {
//          // Otherwise, add an additional arc.
//          StateId additional_state = fst->AddState();
//          StateId next_state = arc.nextstate;
//          arc.nextstate = additional_state;
//          fst->AddArc(additional_state,
//                      Arc(first_phone_label + phone, 0,
//                          Weight::One(), next_state));
//        }
//      }
//
//      aiter.SetValue(arc);
//    }
//  }
//
//  return first_phone_label;
//}
//
//template<class Weight>
//void DeterminizeLatticeDeletePhones(
//    typename ArcTpl<Weight>::Label first_phone_label,
//    MutableFst<ArcTpl<Weight> > *fst) {
//  // Define some types.
//  typedef ArcTpl<Weight> Arc;
//  typedef typename Arc::StateId StateId;
//  typedef typename Arc::Label Label;
//
//  // Delete phones here.
//  for (StateIterator<MutableFst<Arc> > siter(*fst);
//       !siter.Done(); siter.Next()) {
//    StateId state = siter.Value();
//    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, state);
//         !aiter.Done(); aiter.Next()) {
//      Arc arc = aiter.Value();
//
//      if (arc.ilabel >= first_phone_label)
//        arc.ilabel = 0;
//
//      aiter.SetValue(arc);
//    }
//  }
//}
//// instantiate for type LatticeWeight
//template
//void DeterminizeLatticeDeletePhones(
//    ArcTpl<kaldi::LatticeWeight>::Label first_phone_label,
//    MutableFst<ArcTpl<kaldi::LatticeWeight> > *fst);
//
///** This function does a first pass determinization with phone symbols inserted
//    at phone boundary. It uses a transition model to work out the transition-id
//    to phone map. First, phones will be inserted into the word level lattice.
//    Second, determinization will be applied on top of the phone + word lattice.
//    Finally, the inserted phones will be removed, converting the lattice back to
//    a word level lattice. The output lattice of this pass is not deterministic,
//    since we remove the phone symbols as a last step. It is supposed to be
//    followed by another pass of determinization at the word level. It could also
//    be useful for some other applications such as fMLLR estimation, confidence
//    estimation, discriminative training, etc.
//*/
//template<class Weight, class IntType>
//bool DeterminizeLatticePhonePrunedFirstPass(
//    const kaldi::TransitionInformation &trans_model,
//    double beam,
//    MutableFst<ArcTpl<Weight> > *fst,
//    const DeterminizeLatticePrunedOptions &opts) {
//  // First, insert the phones.
//  typename ArcTpl<Weight>::Label first_phone_label =
//      DeterminizeLatticeInsertPhones(trans_model, fst);
//  TopSort(fst);
//
//  // Second, do determinization with phone inserted.
//  bool ans = DeterminizeLatticePruned<Weight>(*fst, beam, fst, opts);
//
//  // Finally, remove the inserted phones.
//  DeterminizeLatticeDeletePhones(first_phone_label, fst);
//  TopSort(fst);
//
//  return ans;
//}
//
//// "Destructive" version of DeterminizeLatticePhonePruned() where the input
//// lattice might be modified.
//template<class Weight, class IntType>
//bool DeterminizeLatticePhonePruned(
//    const kaldi::TransitionInformation &trans_model,
//    MutableFst<ArcTpl<Weight> > *ifst,
//    double beam,
//    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
//    DeterminizeLatticePhonePrunedOptions opts) {
//  // Returning status.
//  bool ans = true;
//
//  // Make sure at least one of opts.phone_determinize and opts.word_determinize
//  // is not false, otherwise calling this function doesn't make any sense.
//  if ((opts.phone_determinize || opts.word_determinize) == false) {
//    KALDI_WARN << "Both --phone-determinize and --word-determinize are set to "
//               << "false, copying lattice without determinization.";
//    // We are expecting the words on the input side.
//    ConvertLattice<Weight, IntType>(*ifst, ofst, false);
//    return ans;
//  }
//
//  // Determinization options.
//  DeterminizeLatticePrunedOptions det_opts;
//  det_opts.delta = opts.delta;
//  det_opts.max_mem = opts.max_mem;
//
//  // If --phone-determinize is true, do the determinization on phone + word
//  // lattices.
//  if (opts.phone_determinize) {
//    KALDI_VLOG(3) << "Doing first pass of determinization on phone + word "
//                  << "lattices.";
//    ans = DeterminizeLatticePhonePrunedFirstPass<Weight, IntType>(
//        trans_model, beam, ifst, det_opts) && ans;
//
//    // If --word-determinize is false, we've finished the job and return here.
//    if (!opts.word_determinize) {
//      // We are expecting the words on the input side.
//      ConvertLattice<Weight, IntType>(*ifst, ofst, false);
//      return ans;
//    }
//  }
//
//  // If --word-determinize is true, do the determinization on word lattices.
//  if (opts.word_determinize) {
//    KALDI_VLOG(3) << "Doing second pass of determinization on word lattices.";
//    ans = DeterminizeLatticePruned<Weight, IntType>(
//        *ifst, beam, ofst, det_opts) && ans;
//  }
//
//  // If --minimize is true, push and minimize after determinization.
//  if (opts.minimize) {
//    KALDI_VLOG(3) << "Pushing and minimizing on word lattices.";
//    ans = PushCompactLatticeStrings<Weight, IntType>(ofst) && ans;
//    ans = PushCompactLatticeWeights<Weight, IntType>(ofst) && ans;
//    ans = MinimizeCompactLattice<Weight, IntType>(ofst) && ans;
//  }
//
//  return ans;
//}
//
//// Normal verson of DeterminizeLatticePhonePruned(), where the input lattice
//// will be kept as unchanged.
//template<class Weight, class IntType>
//bool DeterminizeLatticePhonePruned(
//    const kaldi::TransitionInformation &trans_model,
//    const ExpandedFst<ArcTpl<Weight> > &ifst,
//    double beam,
//    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
//    DeterminizeLatticePhonePrunedOptions opts) {
//  VectorFst<ArcTpl<Weight> > temp_fst(ifst);
//  return DeterminizeLatticePhonePruned(trans_model, &temp_fst,
//                                       beam, ofst, opts);
//}
//
//bool DeterminizeLatticePhonePrunedWrapper(
//    const kaldi::TransitionInformation &trans_model,
//    MutableFst<kaldi::LatticeArc> *ifst,
//    double beam,
//    MutableFst<kaldi::CompactLatticeArc> *ofst,
//    DeterminizeLatticePhonePrunedOptions opts) {
//  bool ans = true;
//  Invert(ifst);
//  if (ifst->Properties(fst::kTopSorted, true) == 0) {
//    if (!TopSort(ifst)) {
//      // Cannot topologically sort the lattice -- determinization will fail.
//      KALDI_ERR << "Topological sorting of state-level lattice failed (probably"
//                << " your lexicon has empty words or your LM has epsilon cycles"
//                << ").";
//    }
//  }
//  ILabelCompare<kaldi::LatticeArc> ilabel_comp;
//  ArcSort(ifst, ilabel_comp);
//  ans = DeterminizeLatticePhonePruned<kaldi::LatticeWeight, kaldi::int32>(
//      trans_model, ifst, beam, ofst, opts);
//  Connect(ofst);
//  return ans;
//}

// Instantiate the templates for the types we might need.
// Note: there are actually four templates, each of which
// we instantiate for a single type.
template
bool DeterminizeLatticePruned<kaldi::LatticeWeight>(
    const ExpandedFst<kaldi::LatticeArc> &ifst,
    double prune,
    MutableFst<kaldi::CompactLatticeArc> *ofst,
    DeterminizeLatticePrunedOptions opts);

template
bool DeterminizeLatticePruned<kaldi::LatticeWeight>(
    const ExpandedFst<kaldi::LatticeArc> &ifst,
    double prune,
    MutableFst<kaldi::LatticeArc> *ofst,
    DeterminizeLatticePrunedOptions opts);

//template
//bool DeterminizeLatticePhonePruned<kaldi::LatticeWeight, kaldi::int32>(
//    const kaldi::TransitionInformation &trans_model,
//    const ExpandedFst<kaldi::LatticeArc> &ifst,
//    double prune,
//    MutableFst<kaldi::CompactLatticeArc> *ofst,
//    DeterminizeLatticePhonePrunedOptions opts);
//
//template
//bool DeterminizeLatticePhonePruned<kaldi::LatticeWeight, kaldi::int32>(
//    const kaldi::TransitionInformation &trans_model,
//    MutableFst<kaldi::LatticeArc> *ifst,
//    double prune,
//    MutableFst<kaldi::CompactLatticeArc> *ofst,
//    DeterminizeLatticePhonePrunedOptions opts);

}
