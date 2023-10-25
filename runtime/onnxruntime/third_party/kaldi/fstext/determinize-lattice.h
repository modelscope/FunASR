// fstext/determinize-lattice.h

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

#ifndef KALDI_FSTEXT_DETERMINIZE_LATTICE_H_
#define KALDI_FSTEXT_DETERMINIZE_LATTICE_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "fstext/lattice-weight.h"

namespace fst {

/// \addtogroup fst_extensions
///  @{


// For example of usage, see test-determinize-lattice.cc

/*
   DeterminizeLattice implements a special form of determinization
   with epsilon removal, optimized for a phase of lattice generation.
   Its input is an FST with weight-type BaseWeightType (usually a pair of floats,
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

struct DeterminizeLatticeOptions {
  float delta; // A small offset used to measure equality of weights.
  int max_mem; // If >0, determinization will fail and return false
  // when the algorithm's (approximate) memory consumption crosses this threshold.
  int max_loop; // If >0, can be used to detect non-determinizable input
  // (a case that wouldn't be caught by max_mem).
  DeterminizeLatticeOptions(): delta(kDelta),
                               max_mem(-1),
                               max_loop(-1) { }
};

/**
    This function implements the normal version of DeterminizeLattice, in which
    the output strings are represented using sequences of arcs, where all but
    the first one has an epsilon on the input side.  The debug_ptr argument is
    an optional pointer to a bool that, if it becomes true while the algorithm
    is executing, the algorithm will print a traceback and terminate (used in
    fstdeterminizestar.cc debug non-terminating determinization).  More
    efficient if ifst is arc-sorted on input label.  If the number of arcs gets
    more than max_states, it will throw std::runtime_error (otherwise this code
    does not use exceptions).  This is mainly useful for debug.  */
template<class Weight, class IntType>
bool DeterminizeLattice(
    const Fst<ArcTpl<Weight> > &ifst,
    MutableFst<ArcTpl<Weight> > *ofst,
    DeterminizeLatticeOptions opts = DeterminizeLatticeOptions(),
    bool *debug_ptr = NULL);


/*  This is a version of DeterminizeLattice with a slightly more "natural" output format,
    where the output sequences are encoded using the CompactLatticeArcTpl template
    (i.e. the sequences of output symbols are represented directly as strings)
    More efficient if ifst is arc-sorted on input label.
    If the #arcs gets more than max_arcs, it will throw std::runtime_error (otherwise
    this code does not use exceptions).  This is mainly useful for debug.
*/
template<class Weight, class IntType>
bool DeterminizeLattice(
    const Fst<ArcTpl<Weight> >&ifst,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
    DeterminizeLatticeOptions opts = DeterminizeLatticeOptions(),
    bool *debug_ptr = NULL);




/// @} end "addtogroup fst_extensions"

} // end namespace fst

#include "fstext/determinize-lattice-inl.h"

#endif
