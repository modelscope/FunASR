// fstext/pre-determinize-inl.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_FSTEXT_PRE_DETERMINIZE_INL_H_
#define KALDI_FSTEXT_PRE_DETERMINIZE_INL_H_


/* Do not include this file directly.  It is an implementation file included by PreDeterminize.h */

/*
  Predeterminization

    This is a function that makes an FST compactly determinizable by inserting symbols on the input
  side as necessary for disambiguation.  Note that we do not treat epsilon as a real symbol
  when measuring determinizability in this sense.   The extra symbols are added to the vocabulary,
  on the input side; these are of the form (prefix)1, (prefix)2, and so on without limit, where
  (prefix) is some prefix the user provides, e.g. '#' (the function checks that this will not
  lead to conflicts with symbols already in the FST).  The function tells us how many such
  symbols it created.

   Note that there is a paper "Generalized optimization algorithm for speech recognition
   transducers" by Allauzen and Mohri, that deals with a similar issue, but this is a very
   different algorithm that only aims to ensure determinizability, but not *compact*
   determinizability.

   Our algorithm is slightly heuristic, and probably not optimal, but does ensure that the
   output is compactly determinizable, possibly at the expense of inserting unnecessary
   symbols.  We considered more sophisticated algorithms, but these were extremely
   complicated and would give the same output for the kinds of inputs that we envisage.

   Suppose the input FST is T.  We want to ensure that in det(T), if we consider the
   states of det(T) as weighted subsets of states of T, each state of T only appears once
   in any given subset.  This ensures that det(T) is no larger than T in an appropriate
   sense.  The way we do this is as follows.  We identify all states in T that have
   multiple input transitions (counting "being an initial state" as an input transition).
   Let's call these "problematic" states.  For a problematic state p we stipulate that it
   can never appear in any state of det(T) unless that state equals (p, \bar{1}) [i.e. p,
   unweighted].  In order to ensure this, we insert input symbols on the transitions to these
   problematic states (this may necessitate adding extra states).
      We also stipulate that the path through det(T) should always be sufficient to tell us
   the path through T (and we insert extra symbols sufficient to make this so).  This is to
   simplify the algorithm, so that we don't have to consider the output symbols or weights
   when predeterminizing.

   The algorithm is as follows.

    (A) Definitions

      (i)  Define a *problematic state* as a state that either has multiple input transitions,
           or is an initial state and has at least one input transition.

     (ii)  For an arc a, define:
            i[a] = input symbol on a
            o[a] = output symbol on a
            n[a] = dest-state of a
            p[a] = origin-state of a

           For a state q, define
            E[q] = set of transitions leaving q.
           For a set of states Q, define
            E[Q] = set of transitions leaving some q in Q

    (iii)  For a state s, define Closure(s) as the union of state s, and all states t
           that are reachable via sequences of arcs a such that i[a]=epsilon and n[a] is
           not problematic.

           For a set of states S, define Closure(S) as the union of the closures of
           states s in S.

    (B) Inputs and outputs.

     (i) Inputs and preconditions.  Input is an FST, which should have a symbol table compiled into
         it, and a prefix (e.g. #) for symbols to be added.  We check that the input FST is trim,
         and that it does not have any symbols that appear on its arcs, that are equal to the prefix
         followed by digits.

    (ii) Outputs: The algorithm modifies the FST that is given to it, and returns the number of
         the highest numbered "extra symbol" inserted.  The extra symbols are numbered #1, #2 and
         so on without limit (as integers).  They are inserted into the symbol table in a sequential
         way by calling AvailableKey()
         for each in turn (this is stipulated in case we need to keep other symbol tables in sync).

     (C) Sub-algorithm: Closure(S).  This requires the array p(s), defined below, which is true
         if s is problematic.  This also requires, for efficiency, that the arcs be sorted on input
         label.
            Input: a set of states S.  [plus, the fst and the array p].
            Output: a set of states T.
            Algorithm:
                set T <-- S, Q <-- S.
                while Q is nonempty:
                  pop a state s from Q.
                  for each transition a from state s with epsilon on the input label [we can
                    find these efficiently using the sorting on arcs]:
                      If p(n[a]) is false and n[a] is not in T:
                         Insert n[a] into T.
                         Add n[a] to Q.
                return T.


     (D) Main algorithm.


       (i) (a) Check preconditions (FST is trim)
           (b) Make sure there is just one final state (insert epsilon transitions as necessary).
           (c) Sort arcs on input label (so epsilon arcs are at the start of arc lists).


      (ii) Work out the set of problematic states by constructing a boolean array indexed by
           states, i.e.
             p(s)
           which is true if the state is problematic.  We can do this by constructing an array
           t(s) to store the number of transitions into each state [adding one for the initial state],
           and then setting p(s) = true if t(s) > 1.

           Also create a boolean array d(s), defined for states, and set d(s) = false.
           This array is purely for sanity-checking that we are processing each state exactly once.

     (iii) Set up an array of integers m(a), indexed by arcs (how exactly we store these is
           implementation-dependent, but this will probably be a hash from (state, arc-index) to
           integers.  m(a) will store the extra symbol, if any, to be added to that arc (or -1
           if no such symbol; we can also simply have the arc not present in the hash).  The
           initial value of m(a) is -1 (if array), or undefined (if hash).

      (iv) Initialize a set of sets-of-states S, and a queue of pairs Q, as follows.
            The pairs in Q are a pair of (set-of-states, integer), where the integer
            is the number of "special symbols" already used up for that state.

            Note that we use a special indexing for the sets in both S and Q, rather than
            using std::set.  We use a sorted vector of StateId's.  And in S, we index them
            by the lowest-numbered state-id.  Because each state is supposed to only ever
            be a member of one set, if there is an attempt to add another, different set
            with the same lowest-numbered state-id, we detect an error.

            Let I be the single initial state (OpenFST only supports one).
            We set:
              S = { Closure(I) }
              Push (Closure(I), 0)  onto Q.
            Then for each state s such that p(s) = true, and s is not an initial state:
              S <-- S u { Closure(s) }
              Push (Closure(s), 0)  onto Q.

       (v) While Q is nonempty:

          (a) Pop pair (A, n) from Q (queue discipline is arbitrary).

          (b) For each state s in A, check that d(s) is false, and set d(s) to true.
              This is for sanity checking only.

          (c)
             Let S_\eps be the set of epsilon-transitions from members of A to problematic
             states (i.e. S_\eps = \{ a \in E[A]: i[a]=\epsilon, p(n[a]) = true \}).

             Next, we will define, for each t \neq \epsilon, S_t as the set of
               transitions from some state s in S with t as the input label, i.e.:
               S_t = \{ a \in E[A]: i[a] = t \}
               We further define T_t and U_t as the subsets of S where the destination
                 state is problematic and non-problematic respectively, i.e:
               T_t = \{ a \in E[A]: i[a] = t, p(n[a]) = true \}
               U_t = \{ a \in E[A]: i[a] = t, p(n[a]) = false \}

             The easiest way to obtain these sets is probably to have a hash indexed by
               t that maps to a list of pairs (state, arc-offset) that stores S_t.
               From this we can work out the sizes of T_t and U_t on the fly.

         (d)
             for each transition a in S_\eps:
                m(a) <-- n # Will put symbol n on this transition.
                n <-- n+1  # Note, same n as in pair (A, n)

         (e)
             next,
             for each t\neq epsilon s.t. S_t is nonempty,

                if |S_t| > 1 #if-statement is because if |S_t|=|T_t|=1, no need for prefix.
                   k = 0
                   for each transition a in T_t:
                     set m(a) to k.
                     set k = k+1

                if |U_t| > 0
                   Let V_t be the set of destination-states of arcs in U_t.
                   if Closure(V_t) is not in S:
                     insert Closure(V_t) into S, and add the pair (Closure(V_t), k) to Q.

       (vi) Check that for each state in the FST, d(s) = true.

      (vii) Let n = max_a m(a).  This is the highest-numbered extra symbol (extra symbols
            start from zero, in this numbering which doesn't correspond to the symbol-table
            numbering).  Here we add n+1 extra symbols to the symbol table and store
            the mappings from 0, 1, ... n to the symbol-id.

     (viii) Set up a hash h from (state, int) to (state-id) such that
             t = h(s, k)
            will be the state-id of a newly-created state that has a transition to state s
            with input-label #k.

      (ix) For each arc a such that m(a) != 0:
             If i[a] = epsilon (the input label is epsilon):
                Change i[a] to #m(a). [i.e. prefix then digit m(a)]
             Otherwise:
                If t = h(n[a], m(a)) is not defined [where n[a] is the dest-state]:
                  create a new state t with a transition to n[a], with input-label #m(a) and
                  no output-label or weight.  Set h(n[a], m(a)) = t.
                Change n[a] to h(n[a], m(a)).


*/
namespace fst {

namespace pre_determinize_helpers {

// make it inline to avoid having to put it in a .cc file which most functions here
// could not go in.
inline bool HasBannedPrefixPlusDigits(SymbolTable *symTable, std::string prefix, std::string *bad_sym) {
  // returns true if the symbol table contains any string consisting of this
  // (possibly empty) prefix followed by a nonempty sequence of digits (0 to 9).
  // requires symTable to be non-NULL.
  // if bad_sym != NULL, puts the first bad symbol it finds in *bad_sym.
  assert(symTable != NULL);
  const char *prefix_ptr = prefix.c_str();
  size_t prefix_len = strlen(prefix_ptr);  // allowed to be zero but not encouraged.
  for (SymbolTableIterator siter(*symTable); !siter.Done(); siter.Next()) {
    const std::string &sym = siter.Symbol();
    if (!strncmp(prefix_ptr, sym.c_str(), prefix_len)) {  // has prefix.
      if (isdigit(sym[prefix_len])) {  // we don't allow prefix followed by a digit, as a symbol.
        // Has at least one digit.
        size_t pos;
        for (pos = prefix_len;sym[pos] != '\0'; pos++)
          if (!isdigit(sym[pos])) break;
        if (sym[pos] == '\0') {  // All remaining characters were digits.
          if (bad_sym != NULL) *bad_sym = sym;
          return true;
        }
      } // else OK because prefix was followed by '\0' or a non-digit.
    }
  }
  return false;  // doesn't have banned symbol.
}

template<class T> void CopySetToVector(const std::set<T> s, std::vector<T> *v) {
  // adds members of s to v, in sorted order from lowest to highest
  // (because the set was in sorted order).
  assert(v != NULL);
  v->resize(s.size());
  typename std::set<T>::const_iterator siter = s.begin();
  typename std::vector<T>::iterator viter = v->begin();
  for (;  siter != s.end(); ++siter, ++viter) {
    assert(viter != v->end());
    *viter = *siter;
  }
}

// Warning.  This function calls 'new'.
template<class T>
std::vector<T>* InsertMember(const std::vector<T> m, std::vector<std::vector<T>*> *S) {
  assert(m.size() > 0);
  T idx = m[0];
  assert(idx>=(T)0 && idx < (T)S->size());
  if ( (*S)[idx] != NULL) {
    assert( *((*S)[idx]) == m );
    // The vectors should be the same.  Otherwise this is a bug in the algorithm.
    // It could either be a programming error or a deeper conceptual bug.
    return NULL;  // nothing was inserted.
  } else {
    std::vector<T> *ret = (*S)[idx] = new std::vector<T>(m);  // New copy of m.
    return ret;  // was inserted.
  }
}


// See definition of Closure(S) in item A(iii) in the comment above. it's the set of states
// that are reachable from S via sequences of arcs a such that i[a]=epsilon and n[a] is
// not problematic.  We assume that the fst is sorted on input label (so epsilon arcs first)
// The algorithm is described in section (C) above.  We use the same variable for S and T.
template<class Arc> void Closure(MutableFst<Arc> *fst, std::set<typename Arc::StateId> *S,
                                 const std::vector<bool> &pVec) {
  typedef typename Arc::StateId StateId;
  std::vector<StateId> Q;
  CopySetToVector(*S, &Q);
  while (Q.size() != 0) {
    StateId s = Q.back();
    Q.pop_back();
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, s); ! aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) break;  // Break from the loop: due to sorting there will be no
      // more transitions with epsilons as input labels.
      if (!pVec[arc.nextstate]) {  // Next state is not problematic -> we can use this transition.
        std::pair< typename std::set<StateId>::iterator, bool > p = S->insert(arc.nextstate);
        if (p.second) {  // True means: was inserted into S (wasn't already there).
          Q.push_back(arc.nextstate);
        }
      }
    }
  }
} // end function Closure.

} // end namespace pre_determinize_helpers.


template<class Arc, class Int>
void PreDeterminize(MutableFst<Arc> *fst,
                    typename Arc::Label first_new_sym,
                    std::vector<Int> *symsOut) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef size_t ArcId;  // Our own typedef, not standard OpenFst.  Use size_t
  // for compatibility with argument of ArcIterator::Seek().
  typedef typename Arc::Weight Weight;
  assert(first_new_sym > 0);
  assert(fst != NULL);
  if (fst->Start() == kNoStateId) return;  // for empty FST, nothing to do.
  assert(symsOut != NULL && symsOut->size() == 0);  // we will output the symbols we add into this.

  {  // (D)(i)(a): check is trim (i.e. connected, in OpenFST parlance).
    KALDI_VLOG(2) <<  "PreDeterminize: Checking FST properties";
    uint64 props = fst->Properties(kAccessible|kCoAccessible, true);  // true-> computes properties if unknown at time when called.
    if (props != (kAccessible|kCoAccessible)) {  // All states are not both accessible and co-accessible...
      KALDI_ERR << "PreDeterminize: FST is not trim";
    }
  }

  {  // (D)(i)(b): make single final state.
    KALDI_VLOG(2) <<  "PreDeterminize: creating single final state";
    CreateSuperFinal(fst);
  }

  {  // (D)(i)(c): sort arcs on input.
    KALDI_VLOG(2) <<  "PreDeterminize: sorting arcs on input";
    ILabelCompare<Arc> icomp;
    ArcSort(fst, icomp);
  }

  StateId n_states = 0, max_state = 0;  // Compute n_states, max_state = highest-numbered state.
  {  // compute nStates, maxStates.
    for (StateIterator<MutableFst<Arc> > iter(*fst); ! iter.Done(); iter.Next()) {
      StateId state = iter.Value();
      assert(state>=0);
      n_states++;
      if (state > max_state) max_state = state;
    }
    KALDI_VLOG(2) <<  "PreDeterminize: n_states = "<<(n_states)<<", max_state ="<<(max_state);
  }

  std::vector<bool> p_vec(max_state+1, false);  // compute this next.
  {  // D(ii): computing the array p. ["problematic states, i.e. states with >1 input transition,
    // counting being the initial state as an input transition"].
    std::vector<bool> seen_vec(max_state+1, false);  // rather than counting incoming transitions we just have a bool that says we saw at least one.

    seen_vec[fst->Start()] = true;
    for (StateIterator<MutableFst<Arc> > siter(*fst); ! siter.Done(); siter.Next()) {
      for (ArcIterator<MutableFst<Arc> > aiter(*fst, siter.Value()); ! aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        assert(arc.nextstate>=0&&arc.nextstate<max_state+1);
        if (seen_vec[arc.nextstate])
          p_vec[arc.nextstate] = true;  // now have >1 transition in, so problematic.
        else
          seen_vec[arc.nextstate] = true;
      }
    }
  }
  // D(iii): set up m(a)
  std::map<std::pair<StateId, ArcId>, size_t> m_map;
  // This is the array m, indexed by arcs.  It maps to the index of the symbol we add.


  // WARNING: we should be sure to clean up this memory before exiting.  Do not return
  // or throw an exception from this function, later than this point, without cleaning up!
  // Note that the vectors are shared between Q and S (they "belong to" S.
  std::vector<std::vector<StateId>* > S(max_state+1, (std::vector<StateId>*)(void*)0);
  std::vector<std::pair<std::vector<StateId>*, size_t> > Q;

  // D(iv): initialize S and Q.
  {
    std::vector<StateId> all_seed_states;  // all "problematic" states, plus initial state (if not problematic).
    if (!p_vec[fst->Start()])
      all_seed_states.push_back(fst->Start());
    for (StateId s = 0;s<=max_state; s++)
      if (p_vec[s]) all_seed_states.push_back(s);

    for (size_t idx = 0;idx < all_seed_states.size(); idx++) {
      StateId s = all_seed_states[idx];
      std::set<StateId> closure_s;
      closure_s.insert(s);  // insert "seed" state.
      pre_determinize_helpers::Closure(fst, &closure_s, p_vec);  // follow epsilons to non-problematic states.
      // Closure in this case whis will usually not add anything, for typical topologies in speech
      std::vector<StateId> closure_s_vec;
      pre_determinize_helpers::CopySetToVector(closure_s, &closure_s_vec);
      KALDI_ASSERT(closure_s_vec.size() != 0);
      std::vector<StateId> *ptr = pre_determinize_helpers::InsertMember(closure_s_vec, &S);
      KALDI_ASSERT(ptr != NULL);  // Or conceptual bug or programming error.
      Q.push_back(std::pair<std::vector<StateId>*, size_t>(ptr, 0));
    }
  }

  std::vector<bool> d_vec(max_state+1, false);  // "done vector".  Purely for debugging.


  size_t num_extra_det_states = 0;

  // (D)(v)
  while (Q.size() != 0) {

    // (D)(v)(a)
    std::pair<std::vector<StateId>*, size_t> cur_pair(Q.back());
    Q.pop_back();
    const std::vector<StateId> &A(*cur_pair.first);
    size_t n =cur_pair.second;  // next special symbol to add.

    // (D)(v)(b)
    for (size_t idx = 0;idx < A.size(); idx++) {
      assert(d_vec[A[idx]] == false && "This state has been seen before.  Algorithm error.");
      d_vec[A[idx]] = true;
    }

    // From here is (D)(v)(c).  We work out S_\eps and S_t (for t\neq eps)
    // simultaneously at first.
    std::map<Label, std::set<std::pair<std::pair<StateId, ArcId>, StateId> > > arc_hash;
    // arc_hash is a hash with info of all arcs from states in the set A to
    // non-problematic states.
    // It is a map from ilabel to pair(pair(start-state, arc-offset), end-state).
    // Here, arc-offset reflects the order in which we accessed the arc using the
    // ArcIterator (zero for the first arc).


    {  // This block sets up arc_hash
      for (size_t idx = 0;idx < A.size(); idx++) {
        StateId s = A[idx];
        assert(s>=0 && s<=max_state);
        ArcId arc_id = 0;
        for (ArcIterator<MutableFst<Arc> > aiter(*fst, s); ! aiter.Done(); aiter.Next(), ++arc_id) {
          const Arc &arc = aiter.Value();

          std::pair<std::pair<StateId, ArcId>, StateId>
              this_pair(std::pair<StateId, ArcId>(s, arc_id), arc.nextstate);
          bool inserted = (arc_hash[arc.ilabel].insert(this_pair)).second;
          assert(inserted);  // Otherwise we had a duplicate.
        }
      }
    }

    // (D)(v)(d)
    if (arc_hash.count(0) == 1) {  // We have epsilon transitions out.
      std::set<std::pair<std::pair<StateId, ArcId>, StateId> >  &eps_set = arc_hash[0];
      typedef typename std::set<std::pair<std::pair<StateId, ArcId>, StateId> >::iterator set_iter_t;
      for (set_iter_t siter = eps_set.begin(); siter != eps_set.end(); ++siter) {
        const std::pair<std::pair<StateId, ArcId>, StateId>  &this_pr = *siter;
        if (p_vec[this_pr.second]) {  // Eps-transition to problematic state.
          assert(m_map.count(this_pr.first) == 0);
          m_map[this_pr.first] = n;
          n++;
        }
      }
    }

    // (D)(v)(e)
    {
      typedef typename std::map<Label, std::set<std::pair<std::pair<StateId, ArcId>, StateId> > >::iterator map_iter_t;
      typedef typename std::set<std::pair<std::pair<StateId, ArcId>, StateId> >::iterator set_iter_t2;
      for (map_iter_t miter = arc_hash.begin(); miter != arc_hash.end(); ++miter) {
        Label t = miter->first;
        std::set<std::pair<std::pair<StateId, ArcId>, StateId> >  &S_t = miter->second;
        if (t != 0) {  // For t != epsilon,
          std::set<StateId> V_t;  // set of destination non-problem states.  Will create this set now.

          // exists_noproblem is true iff |U_t| > 0.
          size_t k = 0;

          // First loop "for each transition a in T_t" (i.e. transitions to problematic states)
          // The if-statement if (|S_t|>1) is pushed inside the loop, as the loop also computes
          // the set V_t.
          for (set_iter_t2 siter = S_t.begin(); siter != S_t.end(); ++siter) {
            const std::pair<std::pair<StateId, ArcId>, StateId>  &this_pr = *siter;
            if (p_vec[this_pr.second]) {  // only consider problematic states (just set T_t)
              if (S_t.size() > 1) {  // This is where we pushed the if-statement in.
                assert(m_map.count(this_pr.first) == 0);
                m_map[this_pr.first] = k;
                k++;
                num_extra_det_states++;
              }
            } else {  // Create the set V_t.
              V_t.insert(this_pr.second);
            }
          }
          if (V_t.size() != 0) {
            pre_determinize_helpers::Closure(fst, &V_t, p_vec);  // follow epsilons to non-problematic states.
            std::vector<StateId> closure_V_t_vec;
            pre_determinize_helpers::CopySetToVector(V_t, &closure_V_t_vec);
            std::vector<StateId> *ptr = pre_determinize_helpers::InsertMember(closure_V_t_vec, &S);
            if (ptr != NULL) {  // was inserted.
              Q.push_back(std::pair<std::vector<StateId>*, size_t>(ptr, k));
            }
          }
        }
      }
    }
  } // end while (Q.size() != 0)


  {  // (D)(vi): Check that for each state in the FST, d(s) = true.
    for (StateIterator<MutableFst<Arc> > siter(*fst); ! siter.Done(); siter.Next()) {
      StateId val = siter.Value();
      assert(d_vec[val] == true);
    }
  }

  {  // (D)(vii): compute symbol-table ID's.
    // sets up symsOut array.
    int64 n = -1;
    for (typename std::map<std::pair<StateId, ArcId>, size_t>::iterator m_iter = m_map.begin();
        m_iter != m_map.end();
        ++m_iter) {
      n = std::max(n, (int64) m_iter->second);  // m_iter->second is of type size_t.
    }
    // At this point n is the highest symbol-id (type size_t) of symbols we must add.
    n++;  // This is now the number of symbols we must add.
    for (size_t i = 0;static_cast<int64>(i)<n;i++) symsOut->push_back(first_new_sym + i);
  }

  // (D)(viii): set up hash.
  std::map<std::pair<StateId, size_t>, StateId> h_map;

  {  // D(ix): add extra symbols!  This is where the work gets done.

    // Core part of this is below, search for (*)
    size_t n_states_added = 0;

    for (typename std::map<std::pair<StateId, ArcId>, size_t>::iterator m_iter = m_map.begin();
        m_iter != m_map.end();
        ++m_iter) {
      StateId state = m_iter->first.first;
      ArcId arcpos = m_iter->first.second;
      size_t m_a = m_iter->second;

      MutableArcIterator<MutableFst<Arc> > aiter(fst, state);
      aiter.Seek(arcpos);
      Arc arc = aiter.Value();

      // (*) core part here.
      if (arc.ilabel == 0)
        arc.ilabel = (*symsOut)[m_a];
      else {
        std::pair<StateId, size_t> pr(arc.nextstate, m_a);
        if (!h_map.count(pr)) {
          n_states_added++;
          StateId newstate = fst->AddState();
          assert(newstate>=0);
          Arc new_arc( (*symsOut)[m_a], (Label)0, Weight::One(), arc.nextstate);
          fst->AddArc(newstate, new_arc);
          h_map[pr] = newstate;
        }
        arc.nextstate = h_map[pr];
      }
      aiter.SetValue(arc);
    }

    KALDI_VLOG(2) <<  "Added " <<(n_states_added)<<" new states and added/changed "<<(m_map.size())<<" arcs";

  }
  // Now free up memory.
  for (size_t i = 0;i < S.size();i++)
    delete S[i];
} // end function PreDeterminize


template<class Label> void CreateNewSymbols(SymbolTable *input_sym_table, int nSym,
                                            std::string prefix, std::vector<Label> *symsOut) {
  // Creates nSym new symbols named (prefix)0, (prefix)1 and so on.
  // Crashes if it cannot create them because one or more of them were in the symbol
  // table already.
  assert(symsOut && symsOut->size() == 0);
  for (int i = 0;i < nSym;i++) {
    std::stringstream ss; ss << prefix << i;
    std::string str = ss.str();
    if (input_sym_table->Find(str) != -1) {  // should not be present.
    }
    assert(symsOut);
    symsOut->push_back( (Label) input_sym_table->AddSymbol(str));
  }
}


// see pre-determinize.h for documentation.
template<class Arc> void AddSelfLoops(MutableFst<Arc> *fst, std::vector<typename Arc::Label> &isyms,
                                      std::vector<typename Arc::Label> &osyms) {
  assert(fst != NULL);
  assert(isyms.size() == osyms.size());
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  size_t n = isyms.size();
  if (n == 0) return;  // Nothing to do.

  // {
  // the following declarations and statements are for quick detection of these
  // symbols, which is purely for debugging/checking purposes.
  Label  isyms_min = *std::min_element(isyms.begin(), isyms.end()),
         isyms_max = *std::max_element(isyms.begin(), isyms.end()),
         osyms_min = *std::min_element(osyms.begin(), osyms.end()),
         osyms_max = *std::max_element(osyms.begin(), osyms.end());
  std::set<Label> isyms_set, osyms_set;
  for (size_t i = 0; i < isyms.size(); i++) {
    assert(isyms[i] > 0 && osyms[i] > 0);  // should not have epsilon or invalid symbols.
    isyms_set.insert(isyms[i]);
    osyms_set.insert(osyms[i]);
  }
  assert(isyms_set.size() == n && osyms_set.size() == n);
  // } end block.

  for (StateIterator<MutableFst<Arc> > siter(*fst); ! siter.Done(); siter.Next()) {
    StateId state = siter.Value();
    bool this_state_needs_self_loops = (fst->Final(state) != Weight::Zero());
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, state); ! aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      // If one of the following asserts fails, it means that the input FST already had the symbols
      // we are inserting.  This is contrary to the preconditions of this algorithm.
      assert(!(arc.ilabel>=isyms_min && arc.ilabel<=isyms_max && isyms_set.count(arc.ilabel) != 0));
      assert(!(arc.olabel>=osyms_min && arc.olabel<=osyms_max && osyms_set.count(arc.olabel) != 0));
      if (arc.olabel != 0) // Has non-epsilon output label -> need self loops.
        this_state_needs_self_loops = true;
    }
    if (this_state_needs_self_loops) {
      for (size_t i = 0;i < n;i++) {
        Arc arc;
        arc.ilabel = isyms[i];
        arc.olabel = osyms[i];
        arc.weight = Weight::One();
        arc.nextstate = state;
        fst->AddArc(state, arc);
      }
    }
  }
}

template<class Arc>
int64 DeleteISymbols(MutableFst<Arc> *fst, std::vector<typename Arc::Label> isyms) {

  // We could do this using the Mapper concept, but this is much easier to understand.

  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;

  int64 num_deleted = 0;

  if (isyms.size() == 0) return 0;
  Label  isyms_min = *std::min_element(isyms.begin(), isyms.end()),
         isyms_max = *std::max_element(isyms.begin(), isyms.end());
  bool isyms_consecutive = (isyms_max+1-isyms_min == static_cast<Label>(isyms.size()));
  std::set<Label> isyms_set;
  if (!isyms_consecutive)
    for (size_t i = 0;i < isyms.size();i++)
      isyms_set.insert(isyms[i]);

  for (StateIterator<MutableFst<Arc> > siter(*fst); ! siter.Done(); siter.Next()) {
    StateId state = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, state); ! aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel >= isyms_min && arc.ilabel <= isyms_max) {
        if (isyms_consecutive || isyms_set.count(arc.ilabel) != 0) {
          num_deleted++;
          Arc mod_arc (arc);
          mod_arc.ilabel = 0;  // change label to epsilon.
          aiter.SetValue(mod_arc);
        }
      }
    }
  }
  return num_deleted;
}

template<class Arc>
typename Arc::StateId CreateSuperFinal(MutableFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  assert(fst != NULL);
  StateId num_states = fst->NumStates();
  std::vector<StateId> final_states;
  for (StateId s = 0; s < num_states; s++) {
    if (fst->Final(s) != Weight::Zero()) {
      final_states.push_back(s);
    }
  }
  if (final_states.size() == 1) {
    if (fst->Final(final_states[0]) == Weight::One()) {
      ArcIterator<MutableFst<Arc> > iter(*fst, final_states[0]);
      if (iter.Done()) {
        // We already have a final state w/ no transitions out and unit weight.
        // So we're done.
        return final_states[0];
      }
    }
  }

  StateId final_state = fst->AddState();
  fst->SetFinal(final_state, Weight::One());
  for (size_t idx = 0; idx < final_states.size(); idx++) {
    StateId s = final_states[idx];
    Weight weight = fst->Final(s);
    fst->SetFinal(s, Weight::Zero());
    Arc arc;
    arc.ilabel = 0;
    arc.olabel = 0;
    arc.nextstate = final_state;
    arc.weight = weight;
    fst->AddArc(s, arc);
  }
  return final_state;
}


}  // namespace fst

#endif  // KALDI_FSTEXT_PRE_DETERMINIZE_INL_H_
