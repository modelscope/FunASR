// lat/compose-lattice-pruned.h

// Copyright      2017  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_LAT_COMPOSE_LATTICE_PRUNED_H_
#define KALDI_LAT_COMPOSE_LATTICE_PRUNED_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "fstext/lattice-weight.h"
#include "itf/options-itf.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {


/*
   This header implements pruned lattice composition, via the functions
   ComposeCompactLatticePruned (we may later add ComposeLatticePruned if
   needed).

   ComposeCompactLatticePruned does composition of a CompactLattice with a
   DeterministicOnDemandFst<StdArc>, producing a CompactLattice.  It's
   intended for language model rescoring of lattices.

   The scenario is that you have produced a Lattice or CompactLattice via
   conventional decoding, and you want to replace (or partially replace) the
   language model scores in the lattice (which will probably will come from the
   LM used to generate the HCLG.fst) with the language model scores from a
   larger language model.

   The simpler alternative to using ComposeCompactLatticePruned is to use
   ComposeCompactLatticeDeterministic.  The advantages of ComposedCompactLatticePruned are:

     (1) The alternative might be too slow, because when you compose a lattice
         with a high-order n-gram language model (or an RNNLM with a high-order
         n-gram approximation) it can generate a lot more arcs than were present
         in the original lattice.

     (2) For RNNLM rescoring, the n-gram approximation may not always
         be choosing a very good history.  In the n-gram approximation,
         the LM score for a particular word given a history is taken
         from a history that is the same as the desired history up to
         the last, say, 4 words, but beyond that may differ.  The
         advantage of ComposeCompactLatticePruned functions over the alternative is
         that it will often take, in a suitable sense, the "best" history
         (instead of an arbitrary history); this happens simply because the
         paths that are expected to be the best paths are visited first.


   We now describe how you are expected to get the thing to compose with,
   i.e. the DeterministicOnDemandFst<StdArc> that corrects the LM weights.  It
   will normally contain the LM used to create the original HCLG, with a
   negative weight, composed with the LM you want to use, with a positive
   weights (these weights might not be -1 and 1 if there is interpolation in the
   picture).  The LM we want to use will often be e.g. a 4-gram ARPA-type LM
   (stored as a regular FST or, more compactly, as a .carpa file which is a
   ConstArpaFst), or it will be some kind of RNNLM.  You would use a
   ComposeDeterministicOnDemandFst<StdArc> to combine the "base" language model
   (with a negative weight, using either ConstArpaLm or
   BackoffDeterministicOnDemandFst wrapped in ScaleDeterministicOnDemandFst)
   with the RNNLM language model (the name of FST TBD, Hainan needs to write
   this).
*/




// This options class is used for ComposeCompactLatticePruned,
// and if in future we write a function ComposeLatticePruned, we'll
// use the same options class.
// Note: the binary that uses this may want to use an --acoustic-scale
// option, in case the acoustics need to be scaled down before this
// composition, because it will make a difference to which paths
// are explored in the lattice.
struct ComposeLatticePrunedOptions {
  // 'lattice_compose_beam' is a beam that determines
  // how much of a given composition space we will expand (at least,
  // until we hit the limit imposed by 'max_arcs'..  This
  // beam is applied using heuristically-estimated expected costs
  // to the end of the lattice, so if you specify, for example,
  // beam=5.0, it doesn't guarantee that all paths with best-cost
  // within 5.0 of the best path in the composed output will be
  // retained (However, this would be exact if the LM we were
  // rescoring with had zero costs).
  float lattice_compose_beam;

  // 'max_arcs' is the maximum number of arcs that we are willing to expand per
  // lattice; once this limit is reached, we terminate the composition (however,
  // this limit is not applied until at least one path to a final-state has been
  // produced).
  int32 max_arcs;

  // 'initial_num_arcs' is the number of arcs we use on the first outer
  // iteration of the algorithm.  This is so unimportant that we do not expose
  // it on the command line.
  int32 initial_num_arcs;

  // 'growth_ratio' determines how much we allow the num-arcs to grow on each
  // outer iteration of the algorithm.  1.5 is a reasonable value; if it is set
  // too small, too much time will be taken in RecomputePruningInfo(), and if
  // too large, the paths searched may be less optimal than they could be (the
  // heuristics will be less accurate).
  BaseFloat growth_ratio;

  ComposeLatticePrunedOptions(): lattice_compose_beam(6.0),
                                 max_arcs(100000),
                                 initial_num_arcs(100),
                                 growth_ratio(1.5) { }
  void Register(OptionsItf *po) {
    po->Register("lattice-compose-beam", &lattice_compose_beam,
                 "Beam used in pruned lattice composition, which determines how "
                 "large the composed lattice may be.");
    po->Register("max-arcs", &max_arcs, "Maximum number of arcs we allow in "
                 "any given lattice, during pruned composition (limits max size "
                 "of lattices; also see lattice-compose-beam).");
    po->Register("growth-ratio", &growth_ratio, "Factor used in the lattice "
                 "composition algorithm; must be >1.0.  Affects speed vs. "
                 "the optimality of the best-first composition.");
  }
};


/**
   Does pruned composition of a lattice 'clat' with a DeterministicOnDemandFst
   'det_fst'; implements LM rescoring.

   @param [in] opts Class containing options
   @param [in] clat   The input lattice, which is expected to already have a
                   reasonable acoustic scale applied (e.g. 0.1 if it's a normal
                   cross-entropy system, but 1.0 for a chain system); this scale
                   affects the pruning.
   @param [in] det_fst   The on-demand FST that we are composing with; its
                   ilabels will correspond to words and it should be an acceptor
                   in practice (ilabel == olabel).  Will often contain a
                   weighted difference of language model scores, with scores
                   of the form alpha * new - alpha * old, where alpha
                   is the interpolation weight for the 'new' language model
                   (e.g. 0.5 or 0.8).  It's non-const because 'det_fst' is
                   on-demand.
   @param [out] composed_clat  The output, which is a result of composing
                   'clat' with '*det_fst'.  Notionally, '*det_fst' is on the
                   right, although both are acceptors so it doesn't really
                   matter in practice.
                   Although the two FSTs are of different types, the code
                   manually does the conversion.  The weights in '*det_fst'
                   will be interpreted as graph weights (Value1()) in the
                   lattice semiring.
 */
void ComposeCompactLatticePruned(
    const ComposeLatticePrunedOptions &opts,
    const CompactLattice &clat,
    fst::DeterministicOnDemandFst<fst::StdArc> *det_fst,
    CompactLattice* composed_clat);




} // namespace kaldi

#endif
