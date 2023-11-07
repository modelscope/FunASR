// lat/determinize-lattice-pruned.h

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

#ifndef KALDI_LAT_DETERMINIZE_LATTICE_PRUNED_H_
#define KALDI_LAT_DETERMINIZE_LATTICE_PRUNED_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "fstext/lattice-weight.h"
#include "itf/transition-information.h"
#include "itf/options-itf.h"
#include "lat/kaldi-lattice.h"

namespace fst {

/// \addtogroup fst_extensions
///  @{


// For example of usage, see test-determinize-lattice-pruned.cc

/*
   DeterminizeLatticePruned implements a special form of determinization with
   epsilon removal, optimized for a phase of lattice generation.  This algorithm
   also does pruning at the same time-- the combination is more efficient as it
   somtimes prevents us from creating a lot of states that would later be pruned
   away.  This allows us to increase the lattice-beam and not have the algorithm
   blow up.  Also, because our algorithm processes states in order from those
   that appear on high-scoring paths down to those that appear on low-scoring
   paths, we can easily terminate the algorithm after a certain specified number
   of states or arcs.

   The input is an FST with weight-type BaseWeightType (usually a pair of floats,
   with a lexicographical type of order, such as LatticeWeightTpl<float>).
   Typically this would be a state-level lattice, with input symbols equal to
   words, and output-symbols equal to p.d.f's (so like the inverse of HCLG).  Imagine representing this as an
   acceptor of type CompactLatticeWeightTpl<float>, in which the input/output
   symbols are words, and the weights contain the original weights together with
   strings (with zero or one symbol in them) containing the original output labels
   (the p.d.f.'s).  We determinize this using acceptor determinization with
   epsilon removal.  Remember (from lattice-weight.h) that
   CompactLatticeWeightTpl has a special kind of semiring where we always take
   the string corresponding to the best cost (of type BaseWeightType), and
   discard the other.  This corresponds to taking the best output-label sequence
   (of p.d.f.'s) for each input-label sequence (of words).  We couldn't use the
   Gallic weight for this, or it would die as soon as it detected that the input
   FST was non-functional.  In our case, any acyclic FST (and many cyclic ones)
   can be determinized.
   We assume that there is a function
      Compare(const BaseWeightType &a, const BaseWeightType &b)
   that returns (-1, 0, 1) according to whether (a < b, a == b, a > b) in the
   total order on the BaseWeightType... this information should be the
   same as NaturalLess would give, but it's more efficient to do it this way.
   You can define this for things like TropicalWeight if you need to instantiate
   this class for that weight type.

   We implement this determinization in a special way to make it efficient for
   the types of FSTs that we will apply it to.  One issue is that if we
   explicitly represent the strings (in CompactLatticeWeightTpl) as vectors of
   type vector<IntType>, the algorithm takes time quadratic in the length of
   words (in states), because propagating each arc involves copying a whole
   vector (of integers representing p.d.f.'s).  Instead we use a hash structure
   where each string is a pointer (Entry*), and uses a hash from (Entry*,
   IntType), to the successor string (and a way to get the latest IntType and the
   ancestor Entry*).  [this is the class LatticeStringRepository].

   Another issue is that rather than representing a determinized-state as a
   collection of (state, weight), we represent it in a couple of reduced forms.
   Suppose a determinized-state is a collection of (state, weight) pairs; call
   this the "canonical representation".  Note: these collections are always
   normalized to remove any common weight and string part.  Define end-states as
   the subset of states that have an arc out of them with a label on, or are
   final.  If we represent a determinized-state a the set of just its (end-state,
   weight) pairs, this will be a valid and more compact representation, and will
   lead to a smaller set of determinized states (like early minimization).  Call
   this collection of (end-state, weight) pairs the "minimal representation".  As
   a mechanism to reduce compute, we can also consider another representation.
   In the determinization algorithm, we start off with a set of (begin-state,
   weight) pairs (where the "begin-states" are initial or have a label on the
   transition into them), and the "canonical representation" consists of the
   epsilon-closure of this set (i.e. follow epsilons).  Call this set of
   (begin-state, weight) pairs, appropriately normalized, the "initial
   representation".  If two initial representations are the same, the "canonical
   representation" and hence the "minimal representation" will be the same.  We
   can use this to reduce compute.  Note that if two initial representations are
   different, this does not preclude the other representations from being the same.

*/


struct DeterminizeLatticePrunedOptions {
  float delta; // A small offset used to measure equality of weights.
  int max_mem; // If >0, determinization will fail and return false
  // when the algorithm's (approximate) memory consumption crosses this threshold.
  int max_loop; // If >0, can be used to detect non-determinizable input
  // (a case that wouldn't be caught by max_mem).
  int max_states;
  int max_arcs;
  float retry_cutoff;
  DeterminizeLatticePrunedOptions(): delta(kDelta),
                                     max_mem(-1),
                                     max_loop(-1),
                                     max_states(-1),
                                     max_arcs(-1),
                                     retry_cutoff(0.5) { }
  void Register (kaldi::OptionsItf *opts) {
    opts->Register("delta", &delta, "Tolerance used in determinization");
    opts->Register("max-mem", &max_mem, "Maximum approximate memory usage in "
                   "determinization (real usage might be many times this)");
    opts->Register("max-arcs", &max_arcs, "Maximum number of arcs in "
                   "output FST (total, not per state");
    opts->Register("max-states", &max_states, "Maximum number of arcs in output "
                   "FST (total, not per state");
    opts->Register("max-loop", &max_loop, "Option used to detect a particular "
                   "type of determinization failure, typically due to invalid input "
                   "(e.g., negative-cost loops)");
    opts->Register("retry-cutoff", &retry_cutoff, "Controls pruning un-determinized "
                   "lattice and retrying determinization: if effective-beam < "
                   "retry-cutoff * beam, we prune the raw lattice and retry.  Avoids "
                   "ever getting empty output for long segments.");
  }
};

struct DeterminizeLatticePhonePrunedOptions {
  // delta: a small offset used to measure equality of weights.
  float delta;
  // max_mem: if > 0, determinization will fail and return false when the
  // algorithm's (approximate) memory consumption crosses this threshold.
  int max_mem;
  // phone_determinize: if true, do a first pass determinization on both phones
  // and words.
  bool phone_determinize;
  // word_determinize: if true, do a second pass determinization on words only.
  bool word_determinize;
  // minimize: if true, push and minimize after determinization.
  bool minimize;
  DeterminizeLatticePhonePrunedOptions(): delta(kDelta),
                                          max_mem(50000000),
                                          phone_determinize(true),
                                          word_determinize(true),
                                          minimize(false) {}
  void Register (kaldi::OptionsItf *opts) {
    opts->Register("delta", &delta, "Tolerance used in determinization");
    opts->Register("max-mem", &max_mem, "Maximum approximate memory usage in "
                   "determinization (real usage might be many times this).");
    opts->Register("phone-determinize", &phone_determinize, "If true, do an "
                   "initial pass of determinization on both phones and words (see"
                   " also --word-determinize)");
    opts->Register("word-determinize", &word_determinize, "If true, do a second "
                   "pass of determinization on words only (see also "
                   "--phone-determinize)");
    opts->Register("minimize", &minimize, "If true, push and minimize after "
                   "determinization.");
  }
};

/**
    This function implements the normal version of DeterminizeLattice, in which the
    output strings are represented using sequences of arcs, where all but the
    first one has an epsilon on the input side.  It also prunes using the beam
    in the "prune" parameter.  The input FST must be topologically sorted in order
    for the algorithm to work. For efficiency it is recommended to sort ilabel as well.
    Returns true on success, and false if it had to terminate the determinization
    earlier than specified by the "prune" beam-- that is, if it terminated because
    of the max_mem, max_loop or max_arcs constraints in the options.
    CAUTION: you may want to use the version below which outputs to CompactLattice.
*/
template<class Weight>
bool DeterminizeLatticePruned(
    const ExpandedFst<ArcTpl<Weight> > &ifst,
    double prune,
    MutableFst<ArcTpl<Weight> > *ofst,
    DeterminizeLatticePrunedOptions opts = DeterminizeLatticePrunedOptions());


/*  This is a version of DeterminizeLattice with a slightly more "natural" output format,
    where the output sequences are encoded using the CompactLatticeArcTpl template
    (i.e. the sequences of output symbols are represented directly as strings The input
    FST must be topologically sorted in order for the algorithm to work. For efficiency
    it is recommended to sort the ilabel for the input FST as well.
    Returns true on normal success, and false if it had to terminate the determinization
    earlier than specified by the "prune" beam-- that is, if it terminated because
    of the max_mem, max_loop or max_arcs constraints in the options.
    CAUTION: if Lattice is the input, you need to Invert() before calling this,
    so words are on the input side.
*/
template<class Weight, class IntType>
bool DeterminizeLatticePruned(
    const ExpandedFst<ArcTpl<Weight> >&ifst,
    double prune,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
    DeterminizeLatticePrunedOptions opts = DeterminizeLatticePrunedOptions());

/** This function takes in lattices and inserts phones at phone boundaries. It
    uses the transition model to work out the transition_id to phone map. The
    returning value is the starting index of the phone label. Typically we pick
    (maximum_output_label_index + 1) as this value. The inserted phones are then
    mapped to (returning_value + original_phone_label) in the new lattice. The
    returning value will be used by DeterminizeLatticeDeletePhones() where it
    works out the phones according to this value.
*/
template<class Weight>
typename ArcTpl<Weight>::Label DeterminizeLatticeInsertPhones(
    const kaldi::TransitionInformation &trans_model,
    MutableFst<ArcTpl<Weight> > *fst);

/** This function takes in lattices and deletes "phones" from them. The "phones"
    here are actually any label that is larger than first_phone_label because
    when we insert phones into the lattice, we map the original phone label to
    (first_phone_label + original_phone_label). It is supposed to be used
    together with DeterminizeLatticeInsertPhones()
*/
template<class Weight>
void DeterminizeLatticeDeletePhones(
    typename ArcTpl<Weight>::Label first_phone_label,
    MutableFst<ArcTpl<Weight> > *fst);

/** This function is a wrapper of DeterminizeLatticePhonePrunedFirstPass() and
    DeterminizeLatticePruned(). If --phone-determinize is set to true, it first
    calls DeterminizeLatticePhonePrunedFirstPass() to do the initial pass of
    determinization on the phone + word lattices. If --word-determinize is set
    true, it then does a second pass of determinization on the word lattices by
    calling DeterminizeLatticePruned(). If both are set to false, then it gives
    a warning and copying the lattices without determinization.

    Note: the point of doing first a phone-level determinization pass and then
    a word-level determinization pass is that it allows us to determinize
    deeper lattices without "failing early" and returning a too-small lattice
    due to the max-mem constraint.  The result should be the same as word-level
    determinization in general, but for deeper lattices it is a bit faster,
    despite the fact that we now have two passes of determinization by default.
*/
template<class Weight, class IntType>
bool DeterminizeLatticePhonePruned(
    const kaldi::TransitionInformation &trans_model,
    const ExpandedFst<ArcTpl<Weight> > &ifst,
    double prune,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
    DeterminizeLatticePhonePrunedOptions opts
      = DeterminizeLatticePhonePrunedOptions());

/** "Destructive" version of DeterminizeLatticePhonePruned() where the input
    lattice might be changed.
*/
template<class Weight, class IntType>
bool DeterminizeLatticePhonePruned(
    const kaldi::TransitionInformation &trans_model,
    MutableFst<ArcTpl<Weight> > *ifst,
    double prune,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
    DeterminizeLatticePhonePrunedOptions opts
      = DeterminizeLatticePhonePrunedOptions());

/** This function is a wrapper of DeterminizeLatticePhonePruned() that works for
    Lattice type FSTs.  It simplifies the calling process by calling
    TopSort() Invert() and ArcSort() for you.
    Unlike other determinization routines, the function
    requires "ifst" to have transition-id's on the input side and words on the
    output side.
    This function can be used as the top-level interface to all the determinization
    code.
*/
bool DeterminizeLatticePhonePrunedWrapper(
    const kaldi::TransitionInformation &trans_model,
    MutableFst<kaldi::LatticeArc> *ifst,
    double prune,
    MutableFst<kaldi::CompactLatticeArc> *ofst,
    DeterminizeLatticePhonePrunedOptions opts
      = DeterminizeLatticePhonePrunedOptions());

/// @} end "addtogroup fst_extensions"

} // end namespace fst

#endif
