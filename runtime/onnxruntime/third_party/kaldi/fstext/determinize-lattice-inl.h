// fstext/determinize-lattice-inl.h

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_FSTEXT_DETERMINIZE_LATTICE_INL_H_
#define KALDI_FSTEXT_DETERMINIZE_LATTICE_INL_H_
// Do not include this file directly.  It is included by determinize-lattice.h

#include <vector>
#include <climits>

namespace fst {

// This class maps back and forth from/to integer id's to sequences of strings.
// used in determinization algorithm.  It is constructed in such a way that
// finding the string-id of the successor of (string, next-label) has constant time.

// Note: class IntType, typically int32, is the type of the element in the
// string (typically a template argument of the CompactLatticeWeightTpl).

template<class IntType> class LatticeStringRepository {
 public:
  struct Entry {
    const Entry *parent; // NULL for empty string.
    IntType i;
    inline bool operator == (const Entry &other) const {
      return (parent == other.parent && i == other.i);
    }
    Entry() { }
    Entry(const Entry &e): parent(e.parent), i(e.i) {}
  };
  // Note: all Entry* pointers returned in function calls are
  // owned by the repository itself, not by the caller!

  // Interface guarantees empty string is NULL.
  inline const Entry *EmptyString() { return NULL; }

  // Returns string of "parent" with i appended.  Pointer
  // owned by repository
  const Entry *Successor(const Entry *parent, IntType i) {
    new_entry_->parent = parent;
    new_entry_->i = i;

    std::pair<typename SetType::iterator, bool> pr = set_.insert(new_entry_);
    if (pr.second) { // Was successfully inserted (was not there).  We need to
                     // replace the element we inserted, which resides on the
                     // stack, with one from the heap.
      const Entry *ans = new_entry_;
      new_entry_ = new Entry();
      return ans;
    } else { // Was not inserted because an equivalent Entry already
             // existed.
      return *pr.first;
    }
  }

  const Entry *Concatenate (const Entry *a, const Entry *b) {
    if (a == NULL) return b;
    else if (b == NULL) return a;
    std::vector<IntType> v;
    ConvertToVector(b, &v);
    const Entry *ans = a;
    for(size_t i = 0; i < v.size(); i++)
      ans = Successor(ans, v[i]);
    return ans;
  }
  const Entry *CommonPrefix (const Entry *a, const Entry *b) {
    std::vector<IntType> a_vec, b_vec;
    ConvertToVector(a, &a_vec);
    ConvertToVector(b, &b_vec);
    const Entry *ans = NULL;
    for(size_t i = 0; i < a_vec.size() && i < b_vec.size() &&
            a_vec[i] == b_vec[i]; i++)
      ans = Successor(ans, a_vec[i]);
    return ans;
  }

  // removes any elements from b that are not part of
  // a common prefix with a.
  void ReduceToCommonPrefix(const Entry *a,
                            std::vector<IntType> *b) {
    size_t a_size = Size(a), b_size = b->size();
    while (a_size> b_size) {
      a = a->parent;
      a_size--;
    }
    if (b_size > a_size)
      b_size = a_size;
    typename std::vector<IntType>::iterator b_begin = b->begin();
    while (a_size != 0) {
      if (a->i != *(b_begin + a_size - 1))
        b_size = a_size - 1;
      a = a->parent;
      a_size--;
    }
    if (b_size != b->size())
      b->resize(b_size);
  }

  // removes the first n elements of a.
  const Entry *RemovePrefix(const Entry *a, size_t n) {
    if (n==0) return a;
    std::vector<IntType> a_vec;
    ConvertToVector(a, &a_vec);
    assert(a_vec.size() >= n);
    const Entry *ans = NULL;
    for(size_t i = n; i < a_vec.size(); i++)
      ans = Successor(ans, a_vec[i]);
    return ans;
  }



  // Returns true if a is a prefix of b.  If a is prefix of b,
  // time taken is |b| - |a|.  Else, time taken is |b|.
  bool IsPrefixOf(const Entry *a, const Entry *b) const {
    if(a == NULL) return true; // empty string prefix of all.
    if (a == b) return true;
    if (b == NULL) return false;
    return IsPrefixOf(a, b->parent);
  }


  inline size_t Size(const Entry *entry) const {
    size_t ans = 0;
    while (entry != NULL) {
      ans++;
      entry = entry->parent;
    }
    return ans;
  }

  void ConvertToVector(const Entry *entry, std::vector<IntType> *out) const {
    size_t length = Size(entry);
    out->resize(length);
    if (entry != NULL) {
      typename std::vector<IntType>::reverse_iterator iter = out->rbegin();
    while (entry != NULL) {
      *iter = entry->i;
      entry = entry->parent;
        ++iter;
    }
  }
  }

  const Entry *ConvertFromVector(const std::vector<IntType> &vec) {
    const Entry *e = NULL;
    for(size_t i = 0; i < vec.size(); i++)
      e = Successor(e, vec[i]);
    return e;
  }

  LatticeStringRepository() { new_entry_ = new Entry; }

  void Destroy() {
    for (typename SetType::iterator iter = set_.begin();
         iter != set_.end();
         ++iter)
      delete *iter;
    SetType tmp;
    tmp.swap(set_);
    if (new_entry_) {
      delete new_entry_;
      new_entry_ = NULL;
    }
  }

  // Rebuild will rebuild this object, guaranteeing only
  // to preserve the Entry values that are in the vector pointed
  // to (this list does not have to be unique).  The point of
  // this is to save memory.
  void Rebuild(const std::vector<const Entry*> &to_keep) {
    SetType tmp_set;
    for (typename std::vector<const Entry*>::const_iterator
             iter = to_keep.begin();
         iter != to_keep.end(); ++iter)
      RebuildHelper(*iter, &tmp_set);
    // Now delete all elems not in tmp_set.
    for (typename SetType::iterator iter = set_.begin();
         iter != set_.end(); ++iter) {
      if (tmp_set.count(*iter) == 0)
        delete (*iter); // delete the Entry; not needed.
    }
    set_.swap(tmp_set);
  }

  ~LatticeStringRepository() { Destroy(); }
  int32 MemSize() const {
    return set_.size() * sizeof(Entry) * 2; // this is a lower bound
    // on the size this structure might take.
  }
 private:
  class EntryKey { // Hash function object.
   public:
    inline size_t operator()(const Entry *entry) const {
      size_t prime = 49109;
      return static_cast<size_t>(entry->i)
          + prime * reinterpret_cast<size_t>(entry->parent);
    }
  };
  class EntryEqual {
   public:
    inline bool operator()(const Entry *e1, const Entry *e2) const {
      return (*e1 == *e2);
    }
  };
  typedef std::unordered_set<const Entry*, EntryKey, EntryEqual> SetType;

  void RebuildHelper(const Entry *to_add, SetType *tmp_set) {
    while(true) {
      if (to_add == NULL) return;
      typename SetType::iterator iter = tmp_set->find(to_add);
      if (iter == tmp_set->end()) { // not in tmp_set.
        tmp_set->insert(to_add);
        to_add = to_add->parent; // and loop.
      } else {
        return;
      }
    }
  }

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeStringRepository);
  Entry *new_entry_; // We always have a pre-allocated Entry ready to use,
                     // to avoid unnecessary news and deletes.
  SetType set_;

};




// class LatticeDeterminizer is templated on the same types that
// CompactLatticeWeight is templated on: the base weight (Weight), typically
// LatticeWeightTpl<float> etc. but could also be e.g. TropicalWeight, and the
// IntType, typically int32, used for the output symbols in the compact
// representation of strings [note: the output symbols would usually be
// p.d.f. id's in the anticipated use of this code] It has a special requirement
// on the Weight type: that there should be a Compare function on the weights
// such that Compare(w1, w2) returns -1 if w1 < w2, 0 if w1 == w2, and +1 if w1 >
// w2.  This requires that there be a total order on the weights.

template<class Weight, class IntType> class LatticeDeterminizer {
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
    assert(determinized_);
    typedef typename Arc::StateId StateId;
    StateId nStates = static_cast<StateId>(output_arcs_.size());
    if (destroy)
      FreeMostMemory();
    ofst->DeleteStates();
    ofst->SetStart(kNoStateId);
    if (nStates == 0) {
      return;
    }
    for (StateId s = 0;s < nStates;s++) {
      OutputStateId news = ofst->AddState();
      assert(news == s);
    }
    ofst->SetStart(0);
    // now process transitions.
    for (StateId this_state = 0; this_state < nStates; this_state++) {
      std::vector<TempArc> &this_vec(output_arcs_[this_state]);
      typename std::vector<TempArc>::const_iterator iter = this_vec.begin(), end = this_vec.end();

      for (;iter != end; ++iter) {
        const TempArc &temp_arc(*iter);
        CompactArc new_arc;
        std::vector<Label> seq;
        repository_.ConvertToVector(temp_arc.string, &seq);
        CompactWeight weight(temp_arc.weight, seq);
        if (temp_arc.nextstate == kNoStateId) {  // is really final weight.
          ofst->SetFinal(this_state, weight);
        } else {  // is really an arc.
          new_arc.nextstate = temp_arc.nextstate;
          new_arc.ilabel = temp_arc.ilabel;
          new_arc.olabel = temp_arc.ilabel;  // acceptor.  input == output.
          new_arc.weight = weight;  // includes string and weight.
          ofst->AddArc(this_state, new_arc);
        }
      }
      // Free up memory.  Do this inside the loop as ofst is also allocating memory
      if (destroy) { std::vector<TempArc> temp; std::swap(temp, this_vec); }
    }
    if (destroy) { std::vector<std::vector<TempArc> > temp; std::swap(temp, output_arcs_); }
  }

  // Output to standard FST with Weight as its weight type.  We will create extra
  // states to handle sequences of symbols on the output.  If destroy == true,
  // release memory as we go (but we cannot output again).
  void  Output(MutableFst<Arc> *ofst, bool destroy = true) {
    // Outputs to standard fst.
    OutputStateId nStates = static_cast<OutputStateId>(output_arcs_.size());
    ofst->DeleteStates();
    if (nStates == 0) {
      ofst->SetStart(kNoStateId);
      return;
    }
    if (destroy)
      FreeMostMemory();
    // Add basic states-- but we will add extra ones to account for strings on output.
    for (OutputStateId s = 0;s < nStates;s++) {
      OutputStateId news = ofst->AddState();
      assert(news == s);
    }
    ofst->SetStart(0);
    for (OutputStateId this_state = 0; this_state < nStates; this_state++) {
      std::vector<TempArc> &this_vec(output_arcs_[this_state]);

      typename std::vector<TempArc>::const_iterator iter = this_vec.begin(), end = this_vec.end();
      for (; iter != end; ++iter) {
        const TempArc &temp_arc(*iter);
        std::vector<Label> seq;
        repository_.ConvertToVector(temp_arc.string, &seq);

        if (temp_arc.nextstate == kNoStateId) {  // Really a final weight.
          // Make a sequence of states going to a final state, with the strings
          // as labels.  Put the weight on the first arc.
          OutputStateId cur_state = this_state;
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
          OutputStateId cur_state = this_state;
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
      if (destroy) {
        std::vector<TempArc> temp; temp.swap(this_vec);
      }
    }
    if (destroy) {
      std::vector<std::vector<TempArc> > temp;
      temp.swap(output_arcs_);
      repository_.Destroy();
    }
  }


  // Initializer.  After initializing the object you will typically
  // call Determinize() and then call one of the Output functions.
  // Note: ifst.Copy() will generally do a
  // shallow copy.  We do it like this for memory safety, rather than
  // keeping a reference or pointer to ifst_.
  LatticeDeterminizer(const Fst<Arc> &ifst,
                      DeterminizeLatticeOptions opts):
      num_arcs_(0), num_elems_(0), ifst_(ifst.Copy()), opts_(opts),
      equal_(opts_.delta), determinized_(false),
      minimal_hash_(3, hasher_, equal_), initial_hash_(3, hasher_, equal_) {
    KALDI_ASSERT(Weight::Properties() & kIdempotent); // this algorithm won't
    // work correctly otherwise.
  }

  // frees all except output_arcs_, which contains the important info
  // we need to output the FST.
  void FreeMostMemory() {
    if (ifst_) {
      delete ifst_;
      ifst_ = NULL;
    }
    for (typename MinimalSubsetHash::iterator iter = minimal_hash_.begin();
        iter != minimal_hash_.end(); ++iter)
      delete iter->first;
    { MinimalSubsetHash tmp; tmp.swap(minimal_hash_); }
    for (typename InitialSubsetHash::iterator iter = initial_hash_.begin();
        iter != initial_hash_.end(); ++iter)
      delete iter->first;
    { InitialSubsetHash tmp; tmp.swap(initial_hash_); }
    { std::vector<std::vector<Element>* > output_states_tmp;
      output_states_tmp.swap(output_states_); }
    { std::vector<char> tmp;  tmp.swap(isymbol_or_final_); }
    { std::vector<OutputStateId> tmp; tmp.swap(queue_); }
    { std::vector<std::pair<Label, Element> > tmp; tmp.swap(all_elems_tmp_); }
  }

  ~LatticeDeterminizer() {
    FreeMostMemory(); // rest is deleted by destructors.
  }
  void RebuildRepository() { // rebuild the string repository,
    // freeing stuff we don't need.. we call this when memory usage
    // passes a supplied threshold.  We need to accumulate all the
    // strings we need the repository to "remember", then tell it
    // to clean the repository.
    std::vector<StringId> needed_strings;
    for (size_t i = 0; i < output_arcs_.size(); i++)
      for (size_t j = 0; j < output_arcs_[i].size(); j++)
        needed_strings.push_back(output_arcs_[i][j].string);

    // the following loop covers strings present in minimal_hash_
    // which are also accessible via output_states_.
    for (size_t i = 0; i < output_states_.size(); i++)
      for (size_t j = 0; j < output_states_[i]->size(); j++)
        needed_strings.push_back((*(output_states_[i]))[j].string);

    // the following loop covers strings present in initial_hash_.
    for (typename InitialSubsetHash::const_iterator
             iter = initial_hash_.begin();
         iter != initial_hash_.end(); ++iter) {
      const std::vector<Element> &vec = *(iter->first);
      Element elem = iter->second;
      for (size_t i = 0; i < vec.size(); i++)
        needed_strings.push_back(vec[i].string);
      needed_strings.push_back(elem.string);
    }

    std::sort(needed_strings.begin(), needed_strings.end());
    needed_strings.erase(std::unique(needed_strings.begin(),
                                     needed_strings.end()),
                         needed_strings.end()); // uniq the strings.
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
        // having to rebuild too often.
        KALDI_WARN << "Failure in determinize-lattice: size exceeds maximum "
                   << opts_.max_mem << " bytes; (repo,arcs,elems) = ("
                   << repo_size << "," << arcs_size << "," << elems_size
                   << "), after rebuilding, repo size was " << new_repo_size;
        return false;
      }
    }
    return true;
  }

  // Returns true on success.  Can fail for out-of-memory
  // or max-states related reasons.
  bool Determinize(bool *debug_ptr) {
    assert(!determinized_);
    // This determinizes the input fst but leaves it in the "special format"
    // in "output_arcs_".  Must be called after Initialize().  To get the
    // output, call one of the Output routines.
    try {
      InitializeDeterminization(); // some start-up tasks.
      while (!queue_.empty()) {
        OutputStateId out_state = queue_.back();
        queue_.pop_back();
        ProcessState(out_state);
        if (debug_ptr && *debug_ptr) Debug();  // will exit.
        if (!CheckMemoryUsage()) return false;
      }
      return (determinized_ = true);
    } catch (const std::bad_alloc &) {
      int32 repo_size = repository_.MemSize(),
          arcs_size = num_arcs_ * sizeof(TempArc),
          elems_size = num_elems_ * sizeof(Element),
          total_size = repo_size + arcs_size + elems_size;
      KALDI_WARN << "Memory allocation error doing lattice determinization; using "
          << total_size << " bytes (max = " << opts_.max_mem
          << " (repo,arcs,elems) = ("
          << repo_size << "," << arcs_size << "," << elems_size << ")";
      return (determinized_ = false);
    } catch (const std::runtime_error &) {
      KALDI_WARN << "Caught exception doing lattice determinization";
      return (determinized_ = false);
    }
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
    size_t operator ()(const std::vector<Element> * subset) const {  // hashes only the state and string.
      size_t hash = 0, factor = 1;
      for (typename std::vector<Element>::const_iterator iter= subset->begin(); iter != subset->end(); ++iter) {
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
    bool operator ()(const std::vector<Element> * s1, const std::vector<Element> * s2) const {
      size_t sz = s1->size();
      assert(sz>=0);
      if (sz != s2->size()) return false;
      typename std::vector<Element>::const_iterator iter1 = s1->begin(),
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
    bool operator ()(const std::vector<Element> * s1, const std::vector<Element> * s2) const {
      size_t sz = s1->size();
      assert(sz>=0);
      if (sz != s2->size()) return false;
      typename std::vector<Element>::const_iterator iter1 = s1->begin(),
          iter1_end = s1->end(), iter2=s2->begin();
      for (; iter1 < iter1_end; ++iter1, ++iter2) {
        if (iter1->state != iter2->state) return false;
      }
      return true;
    }
  };

  // Define the hash type we use to map subsets (in minimal
  // representation) to OutputStateId.
  typedef std::unordered_map<const std::vector<Element>*, OutputStateId,
                        SubsetKey, SubsetEqual> MinimalSubsetHash;

  // Define the hash type we use to map subsets (in initial
  // representation) to OutputStateId, together with an
  // extra weight. [note: we interpret the Element.state in here
  // as an OutputStateId even though it's declared as InputStateId;
  // these types are the same anyway].
  typedef std::unordered_map<const std::vector<Element>*, Element,
                        SubsetKey, SubsetEqual> InitialSubsetHash;


  // converts the representation of the subset from canonical (all states) to
  // minimal (only states with output symbols on arcs leaving them, and final
  // states).  Output is not necessarily normalized, even if input_subset was.
  void ConvertToMinimal(std::vector<Element> *subset) {
    assert(!subset->empty());
    typename std::vector<Element>::iterator cur_in = subset->begin(),
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
  // If it creates a new OutputStateId, it adds it to the queue.
  OutputStateId MinimalToStateId(const std::vector<Element> &subset) {
    typename MinimalSubsetHash::const_iterator iter
        = minimal_hash_.find(&subset);
    if (iter != minimal_hash_.end()) // Found a matching subset.
      return iter->second;
    OutputStateId ans = static_cast<OutputStateId>(output_arcs_.size());
    std::vector<Element> *subset_ptr = new std::vector<Element>(subset);
    output_states_.push_back(subset_ptr);
    num_elems_ += subset_ptr->size();
    output_arcs_.push_back(std::vector<TempArc>());
    minimal_hash_[subset_ptr] = ans;
    queue_.push_back(ans);
    return ans;
  }


  // Given a normalized initial subset of elements (i.e. before epsilon closure),
  // compute the corresponding output-state.
  OutputStateId InitialToStateId(const std::vector<Element> &subset_in,
                                 Weight *remaining_weight,
                                 StringId *common_prefix) {
    typename InitialSubsetHash::const_iterator iter
        = initial_hash_.find(&subset_in);
    if (iter != initial_hash_.end()) { // Found a matching subset.
      const Element &elem = iter->second;
      *remaining_weight = elem.weight;
      *common_prefix = elem.string;
      if (elem.weight == Weight::Zero())
        KALDI_WARN << "Zero weight!"; // TEMP
      return elem.state;
    }
    // else no matching subset-- have to work it out.
    std::vector<Element> subset(subset_in);
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

    OutputStateId ans = MinimalToStateId(subset);
    *remaining_weight = elem.weight;
    *common_prefix = elem.string;
    if (elem.weight == Weight::Zero())
      KALDI_WARN << "Zero weight!"; // TEMP

    // Before returning "ans", add the initial subset to the hash,
    // so that we can bypass the epsilon-closure etc., next time
    // we process the same initial subset.
    std::vector<Element> *initial_subset_ptr = new std::vector<Element>(subset_in);
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
    std::vector<IntType> a_vec, b_vec;
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
    assert(0); // because we checked if a_str == b_str above, shouldn't reach here
    return 0;
  }


  // This function computes epsilon closure of subset of states by following epsilon links.
  // Called by InitialToStateId and Initialize.
  // Has no side effects except on the string repository.  The "output_subset" is not
  // necessarily normalized (in the sense of there being no common substring), unless
  // input_subset was.
  void EpsilonClosure(std::vector<Element> *subset) {
    // at input, subset must have only one example of each StateId.  [will still
    // be so at output].  This function follows input-epsilons, and augments the
    // subset accordingly.

    std::deque<Element> queue;
    std::unordered_map<InputStateId, Element> cur_subset;
    typedef typename std::unordered_map<InputStateId, Element>::iterator MapIter;
    typedef typename std::vector<Element>::const_iterator VecIter;

    for (VecIter iter = subset->begin(); iter != subset->end(); ++iter) {
      queue.push_back(*iter);
      cur_subset[iter->state] = *iter;
    }

    // find whether input fst is known to be sorted on input label.
    bool sorted = ((ifst_->Properties(kILabelSorted, false) & kILabelSorted) != 0);
    bool replaced_elems = false; // relates to an optimization, see below.
    int counter = 0; // stops infinite loops here for non-lattice-determinizable input;
    // useful in testing.
    while (queue.size() != 0) {
      Element elem = queue.front();
      queue.pop_front();

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
                  << opts_.max_loop << " times during epsilon closure";
      }
      for (ArcIterator<Fst<Arc> > aiter(*ifst_, elem.state); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (sorted && arc.ilabel != 0) break;  // Break from the loop: due to sorting there will be no
        // more transitions with epsilons as input labels.
        if (arc.ilabel == 0
            && arc.weight != Weight::Zero()) {  // Epsilon transition.
          Element next_elem;
          next_elem.state = arc.nextstate;
          next_elem.weight = Times(elem.weight, arc.weight);
          // now must append strings
          if (arc.olabel == 0)
            next_elem.string = elem.string;
          else
            next_elem.string = repository_.Successor(elem.string, arc.olabel);

          MapIter iter = cur_subset.find(next_elem.state);
          if (iter == cur_subset.end()) {
            // was no such StateId: insert and add to queue.
            cur_subset[next_elem.state] = next_elem;
            queue.push_back(next_elem);
          } else {
            // was not inserted because one already there.  In normal determinization we'd
            // add the weights.  Here, we find which one has the better weight, and
            // keep its corresponding string.
            int comp = Compare(next_elem.weight, next_elem.string,
                               iter->second.weight, iter->second.string);
            if(comp == 1) { // next_elem is better, so use its (weight, string)
              iter->second.string = next_elem.string;
              iter->second.weight = next_elem.weight;
              queue.push_back(next_elem);
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
  // Has no side effects except on the variable repository_, and output_arcs_.

  void ProcessFinal(OutputStateId output_state) {
    const std::vector<Element> &minimal_subset = *(output_states_[output_state]);
    // processes final-weights for this subset.

    // minimal_subset may be empty if the graphs is not connected/trimmed, I think,
    // do don't check that it's nonempty.
    bool is_final = false;
    StringId final_string = NULL;  // = NULL to keep compiler happy.
    Weight final_weight = Weight::Zero();
    typename std::vector<Element>::const_iterator iter = minimal_subset.begin(), end = minimal_subset.end();
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
    if (is_final) {
      // store final weights in TempArc structure, just like a transition.
      TempArc temp_arc;
      temp_arc.ilabel = 0;
      temp_arc.nextstate = kNoStateId;  // special marker meaning "final weight".
      temp_arc.string = final_string;
      temp_arc.weight = final_weight;
      output_arcs_[output_state].push_back(temp_arc);
      num_arcs_++;
    }
  }

  // NormalizeSubset normalizes the subset "elems" by
  // removing any common string prefix (putting it in common_str),
  // and dividing by the total weight (putting it in tot_weight).
  void NormalizeSubset(std::vector<Element> *elems,
                       Weight *tot_weight,
                       StringId *common_str) {
    if(elems->empty()) { // just set common_str, tot_weight
      KALDI_WARN << "[empty subset]"; // TEMP
      // to defaults and return...
      *common_str = repository_.EmptyString();
      *tot_weight = Weight::Zero();
      return;
    }
    size_t size = elems->size();
    std::vector<IntType> common_prefix;
    repository_.ConvertToVector((*elems)[0].string, &common_prefix);
    Weight weight = (*elems)[0].weight;
    for (size_t i = 1; i < size; i++) {
      weight = Plus(weight, (*elems)[i].weight);
      repository_.ReduceToCommonPrefix((*elems)[i].string, &common_prefix);
    }
    assert(weight != Weight::Zero()); // we made sure to ignore arcs with zero
    // weights on them, so we shouldn't have zero here.
    size_t prefix_len = common_prefix.size();
    for (size_t i = 0; i < size; i++) {
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
  void MakeSubsetUnique(std::vector<Element> *subset) {
    typedef typename std::vector<Element>::iterator IterType;

    // This assert is designed to fail (usually) if the subset is not sorted on
    // state.
    assert(subset->size() < 2 || (*subset)[0].state <= (*subset)[1].state);

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

  // ProcessTransition is called from "ProcessTransitions".  Broken out for
  // clarity.  Processes a transition from state "state".  The set of Elements
  // represents a set of next-states with associated weights and strings, each
  // one arising from an arc from some state in a determinized-state; the
  // next-states are not necessarily unique (i.e. there may be >1 entry
  // associated with each), and any such sets of Elements have to be merged
  // within this routine (we take the [weight, string] pair that's better in the
  // semiring).
  void ProcessTransition(OutputStateId state, Label ilabel, std::vector<Element> *subset) {
    MakeSubsetUnique(subset); // remove duplicates with the same state.

    StringId common_str;
    Weight tot_weight;
    NormalizeSubset(subset, &tot_weight, &common_str);

    OutputStateId nextstate;
    {
      Weight next_tot_weight;
      StringId next_common_str;
      nextstate = InitialToStateId(*subset,
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
    output_arcs_[state].push_back(temp_arc);  // record the arc.
    num_arcs_++;
  }


  // "less than" operator for pair<Label, Element>.   Used in ProcessTransitions.
  // Lexicographical order, which only compares the state when ordering the
  // "Element" member of the pair.

  class PairComparator {
   public:
    inline bool operator () (const std::pair<Label, Element> &p1, const std::pair<Label, Element> &p2) {
      if (p1.first < p2.first) return true;
      else if (p1.first > p2.first) return false;
      else {
        return p1.second.state < p2.second.state;
      }
    }
  };


  // ProcessTransitions processes emitting transitions (transitions
  // with ilabels) out of this subset of states.
  // Does not consider final states.  Breaks the emitting transitions up by ilabel,
  // and creates a new transition in the determinized FST for each unique ilabel.
  // Does this by creating a big vector of pairs <Label, Element> and then sorting them
  // using a lexicographical ordering, and calling ProcessTransition for each range
  // with the same ilabel.
  // Side effects on repository, and (via ProcessTransition) on Q_, hash_,
  // and output_arcs_.

  void ProcessTransitions(OutputStateId output_state) {
    const std::vector<Element> &minimal_subset = *(output_states_[output_state]);
    // it's possible that minimal_subset could be empty if there are
    // unreachable parts of the graph, so don't check that it's nonempty.
    std::vector<std::pair<Label, Element> > &all_elems(all_elems_tmp_); // use class member
    // to avoid memory allocation/deallocation.
    {
      // Push back into "all_elems", elements corresponding to all
      // non-epsilon-input transitions out of all states in "minimal_subset".
      typename std::vector<Element>::const_iterator iter = minimal_subset.begin(), end = minimal_subset.end();
      for (;iter != end; ++iter) {
        const Element &elem = *iter;
        for (ArcIterator<Fst<Arc> > aiter(*ifst_, elem.state); ! aiter.Done(); aiter.Next()) {
          const Arc &arc = aiter.Value();
          if (arc.ilabel != 0
              && arc.weight != Weight::Zero()) {  // Non-epsilon transition -- ignore epsilons here.
            std::pair<Label, Element> this_pr;
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
    typedef typename std::vector<std::pair<Label, Element> >::const_iterator PairIter;
    PairIter cur = all_elems.begin(), end = all_elems.end();
    std::vector<Element> this_subset;
    while (cur != end) {
      // Process ranges that share the same input symbol.
      Label ilabel = cur->first;
      this_subset.clear();
      while (cur != end && cur->first == ilabel) {
        this_subset.push_back(cur->second);
        cur++;
      }
      // We now have a subset for this ilabel.
      assert(!this_subset.empty()); // temp.
      ProcessTransition(output_state, ilabel, &this_subset);
    }
    all_elems.clear(); // as it's a class variable-- want it to stay
    // emtpy.
  }



  // ProcessState does the processing of a determinized state, i.e. it creates
  // transitions out of it and the final-probability if any.
  void ProcessState(OutputStateId output_state) {
    ProcessFinal(output_state);
    ProcessTransitions(output_state);
  }


  void Debug() {  // this function called if you send a signal
    // SIGUSR1 to the process (and it's caught by the handler in
    // fstdeterminizestar).  It prints out some traceback
    // info and exits.

    KALDI_WARN << "Debug function called (probably SIGUSR1 caught)";
    // free up memory from the hash as we need a little memory
    { MinimalSubsetHash hash_tmp; hash_tmp.swap(minimal_hash_); }

    if (output_arcs_.size() <= 2) {
      KALDI_ERR << "Nothing to trace back";
    }
    size_t max_state = output_arcs_.size() - 2;  // Don't take the last
    // one as we might be halfway into constructing it.

    std::vector<OutputStateId> predecessor(max_state+1, kNoStateId);
    for (size_t i = 0; i < max_state; i++) {
      for (size_t j = 0; j < output_arcs_[i].size(); j++) {
        OutputStateId nextstate = output_arcs_[i][j].nextstate;
        // Always find an earlier-numbered predecessor; this
        // is always possible because of the way the algorithm
        // works.
        if (nextstate <= max_state && nextstate > i)
          predecessor[nextstate] = i;
      }
    }
    std::vector<std::pair<Label, StringId> > traceback;
    // 'traceback' is a pair of (ilabel, olabel-seq).
    OutputStateId cur_state = max_state;  // A recently constructed state.

    while (cur_state != 0 && cur_state != kNoStateId) {
      OutputStateId last_state = predecessor[cur_state];
      std::pair<Label, StringId> p;
      size_t i;
      for (i = 0; i < output_arcs_[last_state].size(); i++) {
        if (output_arcs_[last_state][i].nextstate == cur_state) {
          p.first = output_arcs_[last_state][i].ilabel;
          p.second = output_arcs_[last_state][i].string;
          traceback.push_back(p);
          break;
        }
      }
      KALDI_ASSERT(i != output_arcs_[last_state].size());  // Or fell off loop.
      cur_state = last_state;
    }
    if (cur_state == kNoStateId)
      KALDI_WARN << "Traceback did not reach start state "
                 << "(possibly debug-code error)";

    std::stringstream ss;
    ss << "Traceback follows in format "
       << "ilabel (olabel olabel) ilabel (olabel) ... :";
    for (ssize_t i = traceback.size() - 1; i >= 0; i--) {
      ss << ' ' << traceback[i].first << " ( ";
      std::vector<Label> seq;
      repository_.ConvertToVector(traceback[i].second, &seq);
      for (size_t j = 0; j < seq.size(); j++)
        ss << seq[j] << ' ';
      ss << ')';
    }
    KALDI_ERR << ss.str();
  }

  bool IsIsymbolOrFinal(InputStateId state) { // returns true if this state
    // of the input FST either is final or has an osymbol on an arc out of it.
    // Uses the vector isymbol_or_final_ as a cache for this info.
    assert(state >= 0);
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
    for (ArcIterator<Fst<Arc> > aiter(*ifst_, state);
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

  void InitializeDeterminization() {
    if(ifst_->Properties(kExpanded, false) != 0) { // if we know the number of
      // states in ifst_, it might be a bit more efficient
      // to pre-size the hashes so we're not constantly rebuilding them.
#if !(__GNUC__ == 4 && __GNUC_MINOR__ == 0)
      StateId num_states =
          down_cast<const ExpandedFst<Arc>*, const Fst<Arc> >(ifst_)->NumStates();
      minimal_hash_.rehash(num_states/2 + 3);
      initial_hash_.rehash(num_states/2 + 3);
#endif
    }
    InputStateId start_id = ifst_->Start();
    if (start_id != kNoStateId) {
      /* Insert determinized-state corresponding to the start state into hash and
         queue.  Unlike all the other states, we don't "normalize" the representation
         of this determinized-state before we put it into minimal_hash_.  This is actually
         what we want, as otherwise we'd have problems dealing with any extra weight
         and string and might have to create a "super-initial" state which would make
         the output nondeterministic.  Normalization is only needed to make the
         determinized output more minimal anyway, it's not needed for correctness.
         Note, we don't put anything in the initial_hash_.  The initial_hash_ is only
         a lookaside buffer anyway, so this isn't a problem-- it will get populated
         later if it needs to be.
      */
      Element elem;
      elem.state = start_id;
      elem.weight = Weight::One();
      elem.string = repository_.EmptyString();  // Id of empty sequence.
      std::vector<Element> subset;
      subset.push_back(elem);
      EpsilonClosure(&subset); // follow through epsilon-inputs links
      ConvertToMinimal(&subset); // remove all but final states and
      // states with input-labels on arcs out of them.
      std::vector<Element> *subset_ptr = new std::vector<Element>(subset);
      assert(output_arcs_.empty() && output_states_.empty());
      // add the new state...
      output_states_.push_back(subset_ptr);
      output_arcs_.push_back(std::vector<TempArc>());
      OutputStateId initial_state = 0;
      minimal_hash_[subset_ptr] = initial_state;
      queue_.push_back(initial_state);
    }
  }

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeDeterminizer);


  std::vector<std::vector<Element>* > output_states_; // maps from output state to
                                            // minimal representation [normalized].
                                            // View pointers as owned in
                                            // minimal_hash_.
  std::vector<std::vector<TempArc> > output_arcs_;  // essentially an FST in our format.

  int num_arcs_; // keep track of memory usage: number of arcs in output_arcs_
  int num_elems_; // keep track of memory usage: number of elems in output_states_

  const Fst<Arc> *ifst_;
  DeterminizeLatticeOptions opts_;
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
  std::vector<OutputStateId> queue_; // Queue of output-states to process.  Starts with
  // state 0, and increases and then (hopefully) decreases in length during
  // determinization.  LIFO queue (queue discipline doesn't really matter).

  std::vector<std::pair<Label, Element> > all_elems_tmp_; // temporary vector used in ProcessTransitions.

  enum IsymbolOrFinal { OSF_UNKNOWN = 0, OSF_NO = 1, OSF_YES = 2 };

  std::vector<char> isymbol_or_final_; // A kind of cache; it says whether
  // each state is (emitting or final) where emitting means it has at least one
  // non-epsilon output arc.  Only accessed by IsIsymbolOrFinal()

  LatticeStringRepository<IntType> repository_;  // defines a compact and fast way of
  // storing sequences of labels.
};


// normally Weight would be LatticeWeight<float> (which has two floats),
// or possibly TropicalWeightTpl<float>, and IntType would be int32.
template<class Weight, class IntType>
bool DeterminizeLattice(const Fst<ArcTpl<Weight> > &ifst,
                        MutableFst<ArcTpl<Weight> > *ofst,
                        DeterminizeLatticeOptions opts,
                        bool *debug_ptr) {
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  LatticeDeterminizer<Weight, IntType> det(ifst, opts);
  if (!det.Determinize(debug_ptr))
    return false;
  det.Output(ofst);
  return true;
}


// normally Weight would be LatticeWeight<float> (which has two floats),
// or possibly TropicalWeightTpl<float>, and IntType would be int32.
template<class Weight, class IntType>
bool DeterminizeLattice(const Fst<ArcTpl<Weight> >&ifst,
                        MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > >*ofst,
                        DeterminizeLatticeOptions opts,
                        bool *debug_ptr) {
  ofst->SetInputSymbols(ifst.InputSymbols());
  ofst->SetOutputSymbols(ifst.OutputSymbols());
  LatticeDeterminizer<Weight, IntType> det(ifst, opts);
  if (!det.Determinize(debug_ptr))
    return false;
  det.Output(ofst);
  return true;
}

}  // namespace fst

#endif  // KALDI_FSTEXT_DETERMINIZE_LATTICE_INL_H_
