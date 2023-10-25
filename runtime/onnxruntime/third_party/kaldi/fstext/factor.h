// fstext/factor.h

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

#ifndef KALDI_FSTEXT_FACTOR_H_
#define KALDI_FSTEXT_FACTOR_H_

/*
   This header declares the Factor function, which takes an FST and
   compresses it by detecting linear chains of states, and creating
   special input symbols that represent these chains.  It outputs enough
   information to be able to reconstruct the original sequences [i.e.
   the mapping between the new symbols, and sequences of the original
   symbols].  It ensures that the original symbols all have the same
   number as a corresponding "new" symbol representing a sequence of length
   one; this enables certain optimizations later on.
*/


#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "util/const-integer-set.h"

namespace fst {


/**
   Factor identifies linear chains of states with an olabel (if any)
   only on the first arc of the chain, and possibly a sequence of
   ilabels; it outputs an FST with different symbols on the input
   that represent sequences of the original input symbols; it outputs
   the mapping from the new symbol to sequences of original symbols,
   as "symbols" [zero is reserved for epsilon].

   As a side effect it also sorts the FST in depth-first order.  Factor will
   usually do the best job when the olabels have been pushed to the left,
   i.e. if you make a call like

     Push<Arc, REWEIGHT_TO_INITIAL>(fsta, &fstb, kPushLabels);

   This is because it only creates a chain with olabels on the first arc of the
   chain (or a chain with no olabels). [it's possible to construct cases where
   pushing makes things worse, though].  After Factor, the composition of *ofst
   with the result of calling CreateFactorFst(*symbols) should be equivalent to
   fst.  Alternatively, calling ExpandInputSequences with ofst and *symbols
   would produce something equivalent to fst.
*/

template<class Arc, class I>
void Factor(const Fst<Arc> &fst, MutableFst<Arc> *ofst,
            std::vector<std::vector<I> > *symbols);


/// This is a more conventional interface of Factor that outputs
/// the result as two FSTs.
template<class Arc>
void Factor(const Fst<Arc> &fst, MutableFst<Arc> *ofst1,
            MutableFst<Arc> *ofst2);



/// ExpandInputSequences expands out the input symbols into sequences of input
/// symbols.  It creates linear chains of states for each arc that had >1
/// augmented symbol on it.  It also sets the input symbol table to NULL, since
/// in case you did have a symbol table there it would no longer be valid.  It
/// leaves any weight and output symbols on the first arc of the chain.
template<class Arc, class I>
void ExpandInputSequences(const std::vector<std::vector<I> > &sequences,
                          MutableFst<Arc> *fst);


/// The function CreateFactorFst will create an FST that expands out the
/// "factors" that are the indices of the "sequences" array, into linear sequences
/// of symbols.  There is a single start and end state (state 0), and for each
/// nonzero index i into the array "sequences", there is an arc from state 0 that
/// has output-label i, and enters a chain of states with output epsilons and input
/// labels corresponding to the remaining elements of the sequences, terminating
/// again in state 0.  This FST is output-deterministic and sorted on olabel.
/// Composing an FST on the left with the output of this function, should be the
/// same as calling "ExpandInputSequences".  Use TableCompose (see table-matcher.h)
/// for efficiency.
template<class Arc, class I>
void CreateFactorFst(const std::vector<std::vector<I> > &sequences,
                     MutableFst<Arc> *fst);


/// CreateMapFst will create an FST representing this symbol_map.  The
/// FST has a single loop state with single-arc loops with
/// isymbol = symbol_map[i], osymbol = i.  The resulting FST applies this
/// map to the input symbols of something we compose with it on the right.
/// Must have symbol_map[0] == 0.
template<class Arc, class I>
void CreateMapFst(const std::vector<I> &symbol_map,
                  MutableFst<Arc> *fst);


enum  StatePropertiesEnum
{ kStateFinal = 0x1,
  kStateInitial = 0x2,
  kStateArcsIn = 0x4,
  kStateMultipleArcsIn = 0x8,
  kStateArcsOut = 0x10,
  kStateMultipleArcsOut = 0x20,
  kStateOlabelsOut = 0x40,
  kStateIlabelsOut = 0x80 };

typedef unsigned char StatePropertiesType;

/**
   This function works out various properties of the states in the
   FST, using the bit properties defined in StatePropertiesEnum. */
template<class Arc>
void GetStateProperties(const Fst<Arc> &fst,
                        typename Arc::StateId max_state,
                        std::vector<StatePropertiesType> *props);



template<class Arc>
class DfsOrderVisitor {
  // visitor class that gives the user the dfs order,
  // c.f. dfs-visit.h.  Used in factor-fst-impl.h
  typedef typename Arc::StateId StateId;
 public:
  DfsOrderVisitor(std::vector<StateId> *order): order_(order) { order->clear(); }
  void InitVisit(const Fst<Arc> &fst) {}
  bool InitState(StateId s, StateId) { order_->push_back(s); return true; }
  bool TreeArc(StateId, const Arc&) { return true; }
  bool BackArc(StateId, const Arc&) { return true; }
  bool ForwardOrCrossArc(StateId, const Arc&) { return true; }
  void FinishState(StateId, StateId, const Arc *) { }
  void FinishVisit() { }
 private:
  std::vector<StateId> *order_;
};



}  // namespace fst

#include "factor-inl.h"

#endif
