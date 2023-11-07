// lat/lattice-functions-transition-model.h

// Copyright 2009-2012   Saarland University (author: Arnab Ghoshal)
//           2012-2013   Johns Hopkins University (Author: Daniel Povey);
//                       Bagher BabaAli
//                2014   Guoguo Chen

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

#ifndef KALDI_LAT_LATTICE_FUNCTIONS_TRANSITION_MODEL_H_
#define KALDI_LAT_LATTICE_FUNCTIONS_TRANSITION_MODEL_H_

#include <map>
#include <vector>

#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h"
#include "lat/word-align-lattice-lexicon.h"

namespace kaldi {

/**
   This function can be used to compute posteriors for MMI, with a positive contribution
   for the numerator and a negative one for the denominator.  This function is not actually
   used in our normal MMI training recipes, where it's instead done using various command
   line programs that each do a part of the job.  This function was written for use in
   neural-net MMI training.

   @param [in] trans    The transition model. Used to map the
                        transition-ids to phones or pdfs.
   @param [in] lat      The denominator lattice
   @param [in] num_ali  The numerator alignment
   @param [in] drop_frames   If "drop_frames" is true, it will not compute any
                        posteriors on frames where the num and den have disjoint
                        pdf-ids.
   @param [in] convert_to_pdf_ids   If "convert_to_pdfs_ids" is true, it will
                        convert the output to be at the level of pdf-ids, not
                        transition-ids.
   @param [in] cancel   If "cancel" is true, it will cancel out any positive and
                        negative parts from the same transition-id (or pdf-id,
                        if convert_to_pdf_ids == true).
   @param [out] arc_post   The output MMI posteriors of transition-ids (or
                        pdf-ids if convert_to_pdf_ids == true) at each frame
                        i.e. the difference between the numerator
                        and denominator posteriors.

   It returns the forward-backward likelihood of the lattice. */
BaseFloat LatticeForwardBackwardMmi(
    const TransitionModel &trans,
    const Lattice &lat,
    const std::vector<int32> &num_ali,
    bool drop_frames,
    bool convert_to_pdf_ids,
    bool cancel,
    Posterior *arc_post);

/// This function takes a CompactLattice that should only contain a single
/// linear sequence (e.g. derived from lattice-1best), and that should have been
/// processed so that the arcs in the CompactLattice align correctly with the
/// word boundaries (e.g. by lattice-align-words).  It outputs 4 vectors of the
/// same size, which give, for each word in the lattice (in sequence), the word
/// label, the begin time and length in frames, and the pronunciation (sequence
/// of phones).  This is done even for zero words, corresponding to optional
/// silences -- if you don't want them, just ignore them in the output.
/// This function will print a warning and return false, if the lattice
/// did not have the correct format (e.g. if it is empty or it is not
/// linear).
bool CompactLatticeToWordProns(
    const TransitionModel &tmodel,
    const CompactLattice &clat,
    std::vector<int32> *words,
    std::vector<int32> *begin_times,
    std::vector<int32> *lengths,
    std::vector<std::vector<int32> > *prons,
    std::vector<std::vector<int32> > *phone_lengths);


bool TestWordAlignedLattice(const WordAlignLatticeLexiconInfo &lexicon_info,
                            const TransitionModel &tmodel,
                            CompactLattice clat,
                            CompactLattice aligned_clat,
                            bool allow_duplicate_paths);

}  // namespace kaldi

#endif // KALDI_LAT_LATTICE_FUNCTIONS_TRANSITION_MODEL_H_
