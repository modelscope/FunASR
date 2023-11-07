// fstext/fstext-utils.h

// Copyright 2009-2011  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)
//                2013  Guoguo Chen
//                2014  Telepoint Global Hosting Service, LLC. (Author: David Snyder)

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

#ifndef KALDI_FSTEXT_FSTEXT_UTILS_H_
#define KALDI_FSTEXT_FSTEXT_UTILS_H_
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "fstext/determinize-star.h"
#include "fstext/remove-eps-local.h"
#include "base/kaldi-common.h" // for error reporting macros.
#include "util/text-utils.h" // for SplitStringToVector
#include "fst/script/print-impl.h"

namespace fst {

/// Returns the highest numbered output symbol id of the FST (or zero
/// for an empty FST.
template<class Arc>
typename Arc::Label HighestNumberedOutputSymbol(const Fst<Arc> &fst);

/// Returns the highest numbered input symbol id of the FST (or zero
/// for an empty FST.
template<class Arc>
typename Arc::Label HighestNumberedInputSymbol(const Fst<Arc> &fst);

/// Returns the total number of arcs in an FST.
template<class Arc>
typename Arc::StateId NumArcs(const ExpandedFst<Arc> &fst);

/// GetInputSymbols gets the list of symbols on the input of fst
/// (including epsilon, if include_eps == true), as a sorted, unique
/// list.
template<class Arc, class I>
void GetInputSymbols(const Fst<Arc> &fst,
                     bool include_eps,
                     std::vector<I> *symbols);

/// GetOutputSymbols gets the list of symbols on the output of fst
/// (including epsilon, if include_eps == true)
template<class Arc, class I>
void GetOutputSymbols(const Fst<Arc> &fst,
                      bool include_eps,
                      std::vector<I> *symbols);

/// ClearSymbols sets all the symbols on the input and/or
/// output side of the FST to zero, as specified.
/// It does not alter the symbol tables.
template<class Arc>
void ClearSymbols(bool clear_input,
                  bool clear_output,
                  MutableFst<Arc> *fst);

template<class I>
void GetSymbols(const SymbolTable &symtab,
                bool include_eps,
                std::vector<I> *syms_out);



inline
void DeterminizeStarInLog(VectorFst<StdArc> *fst, float delta = kDelta, bool *debug_ptr = NULL,
                          int max_states = -1);


// e.g. of using this function: PushInLog<REWEIGHT_TO_INITIAL>(fst, kPushWeights|kPushLabels);

template<ReweightType rtype> // == REWEIGHT_TO_{INITIAL, FINAL}
void PushInLog(VectorFst<StdArc> *fst, uint32 ptype, float delta = kDelta) {

  // PushInLog pushes the FST
  // and returns a new pushed FST (labels and weights pushed to the left).
  VectorFst<LogArc> *fst_log = new VectorFst<LogArc>;  // Want to determinize in log semiring.
  Cast(*fst, fst_log);
  VectorFst<StdArc> tmp;
  *fst = tmp;  // free up memory.
  VectorFst<LogArc> *fst_pushed_log = new VectorFst<LogArc>;
  Push<LogArc, rtype>(*fst_log, fst_pushed_log, ptype, delta);
  Cast(*fst_pushed_log, fst);
  delete fst_log;
  delete fst_pushed_log;
}

// Minimizes after encoding; applicable to all FSTs.  It is like what you get
// from the Minimize() function, except it will not push the weights, or the
// symbols.  This is better for our recipes, as we avoid ever pushing the
// weights.  However, it will only minimize optimally if your graphs are such
// that the symbols are as far to the left as they can go, and the weights
// in combinable paths are the same... hard to formalize this, but it's something
// that is satisified by our normal FSTs.
template<class Arc>
void MinimizeEncoded(VectorFst<Arc> *fst, float delta = kDelta) {

  Map(fst, QuantizeMapper<Arc>(delta));
  EncodeMapper<Arc> encoder(kEncodeLabels | kEncodeWeights, ENCODE);
  Encode(fst, &encoder);
  internal::AcceptorMinimize(fst);
  Decode(fst, encoder);
}


/// GetLinearSymbolSequence gets the symbol sequence from a linear FST.
/// If the FST is not just a linear sequence, it returns false.   If it is
/// a linear sequence (including the empty FST), it returns true.  In this
/// case it outputs the symbol
/// sequences as "isymbols_out" and "osymbols_out" (removing epsilons), and
/// the total weight as "tot_weight". The total weight will be Weight::Zero()
/// if the FST is empty.  If any of the output pointers are NULL, it does not
/// create that output.
template<class Arc, class I>
bool GetLinearSymbolSequence(const Fst<Arc> &fst,
                             std::vector<I> *isymbols_out,
                             std::vector<I> *osymbols_out,
                             typename Arc::Weight *tot_weight_out);


/// This function converts an FST with a special structure, which is
/// output by the OpenFst functions ShortestPath and RandGen, and converts
/// them into a std::vector of separate FSTs.  This special structure is that
/// the only state that has more than one (arcs-out or final-prob) is the
/// start state.  fsts_out is resized to the appropriate size.
template<class Arc>
void ConvertNbestToVector(const Fst<Arc> &fst,
                          std::vector<VectorFst<Arc> > *fsts_out);


/// Takes the n-shortest-paths (using ShortestPath), but outputs
/// the result as a vector of up to n fsts.  This function will
/// size the "fsts_out" vector to however many paths it got
/// (which will not exceed n).  n must be >= 1.
template<class Arc>
void NbestAsFsts(const Fst<Arc> &fst,
                 size_t n,
                 std::vector<VectorFst<Arc> > *fsts_out);




/// Creates unweighted linear acceptor from symbol sequence.
template<class Arc, class I>
void MakeLinearAcceptor(const std::vector<I> &labels, MutableFst<Arc> *ofst);



/// Creates an unweighted acceptor with a linear structure, with alternatives
/// at each position.  Epsilon is treated like a normal symbol here.
/// Each position in "labels" must have at least one alternative.
template<class Arc, class I>
void MakeLinearAcceptorWithAlternatives(const std::vector<std::vector<I> > &labels,
                                        MutableFst<Arc> *ofst);


/// Does PreDeterminize and DeterminizeStar and then removes the disambiguation symbols.
/// This is a form of determinization that will never blow up.
/// Note that ifst is non-const and can be considered to be destroyed by this
/// operation.
/// Does not do epsilon removal (RemoveEpsLocal)-- this is so it's safe to cast to
/// log and do this, and maintain equivalence in tropical.

template<class Arc>
void SafeDeterminizeWrapper(MutableFst<Arc> *ifst, MutableFst<Arc> *ofst, float delta = kDelta);


/// SafeDeterminizeMinimizeWapper is as SafeDeterminizeWrapper except that it also
/// minimizes (encoded minimization, which is safe).  This algorithm will destroy "ifst".
template<class Arc>
void SafeDeterminizeMinimizeWrapper(MutableFst<Arc> *ifst, VectorFst<Arc> *ofst, float delta = kDelta);


/// SafeDeterminizeMinimizeWapperInLog is as SafeDeterminizeMinimizeWrapper except
/// it first casts tothe log semiring.
void SafeDeterminizeMinimizeWrapperInLog(VectorFst<StdArc> *ifst, VectorFst<StdArc> *ofst, float delta = kDelta);



/// RemoveSomeInputSymbols removes any symbol that appears in "to_remove", from
/// the input side of the FST, replacing them with epsilon.
template<class Arc, class I>
void RemoveSomeInputSymbols(const std::vector<I> &to_remove,
                            MutableFst<Arc> *fst);

// MapInputSymbols will replace any input symbol i that is between 0 and
// symbol_map.size()-1, with symbol_map[i].  It removes the input symbol
// table of the FST.
template<class Arc, class I>
void MapInputSymbols(const std::vector<I> &symbol_map,
                     MutableFst<Arc> *fst);


template<class Arc>
void RemoveWeights(MutableFst<Arc> *fst);




/// Returns true if and only if the FST is such that the input symbols
/// on arcs entering any given state all have the same value.
/// if "start_is_epsilon", treat start-state as an epsilon input arc
/// [i.e. ensure only epsilon can enter start-state].
template<class Arc>
bool PrecedingInputSymbolsAreSame(bool start_is_epsilon, const Fst<Arc> &fst);


/// This is as PrecedingInputSymbolsAreSame, but with a functor f that maps labels to classes.
/// The function tests whether the symbols preceding any given state are in the same
/// class.
/// Formally, f is of a type F that has an operator of type
/// F::Result F::operator() (F::Arg a) const;
/// where F::Result is an integer type and F::Arc can be constructed from Arc::Label.
/// this must apply to valid labels and also to kNoLabel (so we can have a marker for
/// the invalid labels.
template<class Arc, class F>
bool PrecedingInputSymbolsAreSameClass(bool start_is_epsilon, const Fst<Arc> &fst, const F &f);


/// Returns true if and only if the FST is such that the input symbols
/// on arcs exiting any given state all have the same value.
/// If end_is_epsilon, treat end-state as an epsilon output arc [i.e. ensure
/// end-states cannot have non-epsilon output transitions.]
template<class Arc>
bool FollowingInputSymbolsAreSame(bool end_is_epsilon, const Fst<Arc> &fst);


template<class Arc, class F>
bool FollowingInputSymbolsAreSameClass(bool end_is_epsilon, const Fst<Arc> &fst, const F &f);


/// MakePrecedingInputSymbolsSame ensures that all arcs entering any given fst
/// state have the same input symbol.  It does this by detecting states
/// that have differing input symbols going in, and inserting, for each of
/// the preceding arcs with non-epsilon input symbol, a new dummy state that
/// has an epsilon link to the fst state.
/// If "start_is_epsilon", ensure that start-state can have only epsilon-links
/// into it.
template<class Arc>
void MakePrecedingInputSymbolsSame(bool start_is_epsilon, MutableFst<Arc> *fst);


/// As MakePrecedingInputSymbolsSame, but takes a functor object that maps labels to classes.
template<class Arc, class F>
void MakePrecedingInputSymbolsSameClass(bool start_is_epsilon, MutableFst<Arc> *fst, const F &f);


/// MakeFollowingInputSymbolsSame ensures that all arcs exiting any given fst
/// state have the same input symbol.  It does this by detecting states that have
/// differing input symbols on arcs that exit it, and inserting, for each of the
/// following arcs with non-epsilon input symbol, a new dummy state that has an
/// input-epsilon link from the fst state.  The output symbol and weight stay on the
/// link to the dummy state (in order to keep the FST output-deterministic and
/// stochastic, if it already was).
/// If end_is_epsilon, treat "being a final-state" like having an epsilon output
/// link.
template<class Arc>
void MakeFollowingInputSymbolsSame(bool end_is_epsilon, MutableFst<Arc> *fst);

/// As MakeFollowingInputSymbolsSame, but takes a functor object that maps labels to classes.
template<class Arc, class F>
void MakeFollowingInputSymbolsSameClass(bool end_is_epsilon, MutableFst<Arc> *fst, const F &f);




/// MakeLoopFst creates an FST that has a state that is both initial and
/// final (weight == Weight::One()), and for each non-NULL pointer fsts[i],
/// it has an arc out whose output-symbol is i and which goes to a
/// sub-graph whose input language is equivalent to fsts[i], where the
/// final-state becomes a transition to the loop-state.  Each fst in "fsts"
/// should be an acceptor.  The fst MakeLoopFst returns is output-deterministic,
/// but not output-epsilon free necessarily, and arcs are sorted on output label.
/// Note: if some of the pointers in the input vector "fsts" have the same
/// value, "MakeLoopFst" uses this to speed up the computation.

/// Formally: suppose I is the set of indexes i such that fsts[i] != NULL.
/// Let L[i] be the language that the acceptor fsts[i] accepts.
/// Let the language K be the set of input-output pairs i:l such
/// that i in I and l in L[i].  Then the FST returned by MakeLoopFst
/// accepts the language K*, where * is the Kleene closure (CLOSURE_STAR)
/// of K.

/// We could have implemented this via a combination of "project",
/// "concat", "union" and "closure".  But that FST would have been
/// less well optimized and would have a lot of final-states.

template<class Arc>
VectorFst<Arc>* MakeLoopFst(const std::vector<const ExpandedFst<Arc> *> &fsts);


/// ApplyProbabilityScale is applicable to FSTs in the log or tropical semiring.
/// It multiplies the arc and final weights by "scale" [this is not the Mul
/// operation of the semiring, it's actual multiplication, which is equivalent
/// to taking a power in the semiring].
template<class Arc>
void ApplyProbabilityScale(float scale, MutableFst<Arc> *fst);





/// EqualAlign is similar to RandGen, but it generates a sequence with exactly "length"
/// input symbols.  It returns true on success, false on failure (failure is partly
/// random but should never happen in practice for normal speech models.)
/// It generates a random path through the input FST, finds out which subset of the
/// states it visits along the way have self-loops with inupt symbols on them, and
/// outputs a path with exactly enough self-loops to have the requested number
/// of input symbols.
/// Note that EqualAlign does not use the probabilities on the FST.  It just uses
/// equal probabilities in the first stage of selection (since the output will anyway
/// not be a truly random sample from the FST).
/// The input fst "ifst" must be connected or this may enter an infinite loop.
template<class Arc>
bool EqualAlign(const Fst<Arc> &ifst, typename Arc::StateId length,
                int rand_seed, MutableFst<Arc> *ofst, int num_retries = 10);



// RemoveUselessArcs removes arcs such that there is no input symbol
// sequence for which the best path through the FST would contain
// those arcs [for these purposes, epsilon is not treated as a real symbol].
// This is mainly geared towards decoding-graph FSTs which may contain
// transitions that have less likely words on them that would never be
// taken.  We do not claim that this algorithm removes all such arcs;
// it just does the best job it can.
// Only works for tropical (not log) semiring as it uses
// NaturalLess.
template<class Arc>
void RemoveUselessArcs(MutableFst<Arc> *fst);


// PhiCompose is a version of composition where
// the right hand FST (fst2) is treated as a backoff
// LM, with the phi symbol (e.g. #0) treated as a
// "failure transition", only taken when we don't
// have a match for the requested symbol.
template<class Arc>
void PhiCompose(const Fst<Arc> &fst1,
                const Fst<Arc> &fst2,
                typename Arc::Label phi_label,
                MutableFst<Arc> *fst);

// PropagateFinal propagates final-probs through
// "phi" transitions (note that here, phi_label may
// be epsilon if you want).  If you have a backoff LM
// with special symbols ("phi") on the backoff arcs
// instead of epsilon, you may use PhiCompose to compose
// with it, but this won't do the right thing w.r.t.
// final probabilities.  You should first call PropagateFinal
// on the FST with phi's i it (fst2 in PhiCompose above),
// to fix this.  If a state does not have a final-prob,
// but has a phi transition, it makes the state's final-prob
// (phi-prob * final-prob-of-dest-state), and does this
// recursively i.e. follows phi transitions on the dest state
// first.  It behaves as if there were a super-final state
// with a special symbol leading to it, from each currently
// final state.  Note that this may not behave as desired
// if there are epsilons in your FST; it might be better
// to remove those before calling this function.

template<class Arc>
void PropagateFinal(typename Arc::Label phi_label,
                    MutableFst<Arc> *fst);

// RhoCompose is a version of composition where
// the right hand FST (fst2) has speciall "rho transitions"
// which are taken whenever no normal transition matches; these
// transitions will be rewritten with whatever symbol was on
// the first FST.
template<class Arc>
void RhoCompose(const Fst<Arc> &fst1,
                const Fst<Arc> &fst2,
                typename Arc::Label rho_label,
                MutableFst<Arc> *fst);

/** This function returns true if, in the semiring of the FST, the sum (within
    the semiring) of all the arcs out of each state in the FST is one, to within
    delta.  After MakeStochasticFst, this should be true (for a connected FST).

    @param fst [in] the FST that we are testing.
    @param delta [in] the tolerance to within which we test equality to 1.
    @param min_sum [out] if non, NULL, contents will be set to the minimum sum of weights.
    @param max_sum [out] if non, NULL, contents will be set to the maximum sum of weights.
    @return Returns true if the FST is stochastic, and false otherwise.
*/

template<class Arc>
bool IsStochasticFst(const Fst<Arc> &fst,
                     float delta = kDelta,  // kDelta = 1.0/1024.0 by default.
                     typename Arc::Weight *min_sum = NULL,
                     typename Arc::Weight *max_sum = NULL);




// IsStochasticFstInLog makes sure it's stochastic after casting to log.
inline bool IsStochasticFstInLog(const Fst<StdArc> &fst,
                                 float delta = kDelta,  // kDelta = 1.0/1024.0 by default.
                                 StdArc::Weight *min_sum = NULL,
                                 StdArc::Weight *max_sum = NULL);


} // end namespace fst


#include "fstext/fstext-utils-inl.h"

#endif
