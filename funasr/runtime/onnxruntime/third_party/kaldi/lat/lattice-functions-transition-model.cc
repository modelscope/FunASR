// lat/lattice-functions-transition-model.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli
//                2013  Cisco Systems (author: Neha Agrawal) [code modified
//                      from original code in ../gmmbin/gmm-rescore-lattice.cc]
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

#include "lat/lattice-functions-transition-model.h"

#include "hmm/hmm-utils.h"
#include "hmm/transition-model.h"
#include "lat/lattice-functions.h"

namespace kaldi {

BaseFloat LatticeForwardBackwardMmi(
    const TransitionModel &tmodel,
    const Lattice &lat,
    const std::vector<int32> &num_ali,
    bool drop_frames,
    bool convert_to_pdf_ids,
    bool cancel,
    Posterior *post) {
  // First compute the MMI posteriors.

  Posterior den_post;
  BaseFloat ans = LatticeForwardBackward(lat,
                                         &den_post,
                                         NULL);

  Posterior num_post;
  AlignmentToPosterior(num_ali, &num_post);

  // Now negate the MMI posteriors and add the numerator
  // posteriors.
  ScalePosterior(-1.0, &den_post);

  if (convert_to_pdf_ids) {
    Posterior num_tmp;
    ConvertPosteriorToPdfs(tmodel, num_post, &num_tmp);
    num_tmp.swap(num_post);
    Posterior den_tmp;
    ConvertPosteriorToPdfs(tmodel, den_post, &den_tmp);
    den_tmp.swap(den_post);
  }

  MergePosteriors(num_post, den_post,
                  cancel, drop_frames, post);

  return ans;
}


bool CompactLatticeToWordProns(
    const TransitionModel &tmodel,
    const CompactLattice &clat,
    std::vector<int32> *words,
    std::vector<int32> *begin_times,
    std::vector<int32> *lengths,
    std::vector<std::vector<int32> > *prons,
    std::vector<std::vector<int32> > *phone_lengths) {
  words->clear();
  begin_times->clear();
  lengths->clear();
  prons->clear();
  phone_lengths->clear();
  typedef CompactLattice::Arc Arc;
  typedef Arc::Label Label;
  typedef CompactLattice::StateId StateId;
  typedef CompactLattice::Weight Weight;
  using namespace fst;
  StateId state = clat.Start();
  int32 cur_time = 0;
  if (state == kNoStateId) {
    KALDI_WARN << "Empty lattice.";
    return false;
  }
  while (1) {
    Weight final = clat.Final(state);
    size_t num_arcs = clat.NumArcs(state);
    if (final != Weight::Zero()) {
      if (num_arcs != 0) {
        KALDI_WARN << "Lattice is not linear.";
        return false;
      }
      if (! final.String().empty()) {
        KALDI_WARN << "Lattice has alignments on final-weight: probably "
            "was not word-aligned (alignments will be approximate)";
      }
      return true;
    } else {
      if (num_arcs != 1) {
        KALDI_WARN << "Lattice is not linear: num-arcs = " << num_arcs;
        return false;
      }
      fst::ArcIterator<CompactLattice> aiter(clat, state);
      const Arc &arc = aiter.Value();
      Label word_id = arc.ilabel; // Note: ilabel==olabel, since acceptor.
      // Also note: word_id may be zero; we output it anyway.
      int32 length = arc.weight.String().size();
      words->push_back(word_id);
      begin_times->push_back(cur_time);
      lengths->push_back(length);
      const std::vector<int32> &arc_alignment = arc.weight.String();
      std::vector<std::vector<int32> > split_alignment;
      SplitToPhones(tmodel, arc_alignment, &split_alignment);
      std::vector<int32> phones(split_alignment.size());
      std::vector<int32> plengths(split_alignment.size());
      for (size_t i = 0; i < split_alignment.size(); i++) {
        KALDI_ASSERT(!split_alignment[i].empty());
        phones[i] = tmodel.TransitionIdToPhone(split_alignment[i][0]);
        plengths[i] = split_alignment[i].size();
      }
      prons->push_back(phones);
      phone_lengths->push_back(plengths);

      cur_time += length;
      state = arc.nextstate;
    }
  }
}

// Returns true if this vector of transition-ids could be a valid
// word.  Note: for testing, we assume that the lexicon always
// has the same input-word and output-word.  The other case is complex
// to test.
static bool IsPlausibleWord(const WordAlignLatticeLexiconInfo &lexicon_info,
                            const TransitionModel &tmodel,
                            int32 word_id,
                            const std::vector<int32> &transition_ids) {

  std::vector<std::vector<int32> > split_alignment; // Split into phones.
  if (!SplitToPhones(tmodel, transition_ids, &split_alignment)) {
    KALDI_WARN << "Could not split word into phones correctly (forced-out?)";
  }
  std::vector<int32> phones(split_alignment.size());
  for (size_t i = 0; i < split_alignment.size(); i++) {
    KALDI_ASSERT(!split_alignment[i].empty());
    phones[i] = tmodel.TransitionIdToPhone(split_alignment[i][0]);
  }
  std::vector<int32> lexicon_entry;
  lexicon_entry.push_back(word_id);
  lexicon_entry.insert(lexicon_entry.end(), phones.begin(), phones.end());

  if (!lexicon_info.IsValidEntry(lexicon_entry)) {
    std::ostringstream ostr;
    for (size_t i = 0; i < lexicon_entry.size(); i++)
      ostr << lexicon_entry[i] << ' ';
    KALDI_WARN << "Invalid arc in aligned lattice (code error?) lexicon-entry is " << ostr.str();
    return false;
  } else {
    return true;
  }
}

/// Testing code; map word symbols in the lattice "lat" using the equivalence-classes
/// obtained from the lexicon, using the function EquivalenceClassOf in the lexicon_info
/// object.
static void MapSymbols(const WordAlignLatticeLexiconInfo &lexicon_info,
                       CompactLattice *lat) {
  typedef CompactLattice::StateId StateId;
  for (StateId s = 0; s < lat->NumStates(); s++) {
    for (fst::MutableArcIterator<CompactLattice> aiter(lat, s);
         !aiter.Done(); aiter.Next()) {
      CompactLatticeArc arc (aiter.Value());
      KALDI_ASSERT(arc.ilabel == arc.olabel);
      arc.ilabel = lexicon_info.EquivalenceClassOf(arc.ilabel);
      arc.olabel = arc.ilabel;
      aiter.SetValue(arc);
    }
  }
}

bool TestWordAlignedLattice(const WordAlignLatticeLexiconInfo &lexicon_info,
                            const TransitionModel &tmodel,
                            CompactLattice clat,
                            CompactLattice aligned_clat,
                            bool allow_duplicate_paths) {
  int32 max_err = 5, num_err = 0;
  { // We test whether the forward-backward likelihoods differ; this is intended
    // to detect when we have duplicate paths in the aligned lattice, for some path
    // in the input lattice (e.g. due to epsilon-sequencing problems).
    Posterior post;
    Lattice lat, aligned_lat;
    ConvertLattice(clat, &lat);
    ConvertLattice(aligned_clat, &aligned_lat);
    TopSort(&lat);
    TopSort(&aligned_lat);
    BaseFloat like_before = LatticeForwardBackward(lat, &post),
        like_after = LatticeForwardBackward(aligned_lat, &post);
    if (fabs(like_before - like_after) >
        1.0e-04 * (fabs(like_before) + fabs(like_after))) {
      KALDI_WARN << "Forward-backward likelihoods differ in word-aligned lattice "
                 << "testing, " << like_before << " != " << like_after;
      if (!allow_duplicate_paths)
        num_err++;
    }
  }

  // Do a check on the arcs of the aligned lattice, that each arc corresponds
  // to an entry in the lexicon.
  for (CompactLattice::StateId s = 0; s < aligned_clat.NumStates(); s++) {
    for (fst::ArcIterator<CompactLattice> aiter(aligned_clat, s);
         !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc (aiter.Value());
      KALDI_ASSERT(arc.ilabel == arc.olabel);
      int32 word_id = arc.ilabel;
      const std::vector<int32> &tids = arc.weight.String();
      if (word_id == 0 && tids.empty()) continue; // We allow epsilon arcs.

      if (num_err < max_err)
        if (!IsPlausibleWord(lexicon_info, tmodel, word_id, tids))
          num_err++;
      // Note: IsPlausibleWord will warn if there is an error.
    }
    if (!aligned_clat.Final(s).String().empty()) {
      KALDI_WARN << "Aligned lattice has nonempty string on its final-prob.";
      return false;
    }
  }

  // Next we'll do an equivalence test.
  // First map symbols into equivalence classes, so that we don't wrongly fail
  // due to the capability of the framework to map words to other words.
  // (e.g. mapping <eps> on silence arcs to SIL).

  MapSymbols(lexicon_info, &clat);
  MapSymbols(lexicon_info, &aligned_clat);

  /// Check equivalence.
  int32 num_paths = 5, seed = Rand(), max_path_length = -1;
  BaseFloat delta = 0.2; // some lattices have large costs -> use large delta.

  FLAGS_v = GetVerboseLevel(); // set the OpenFst verbose level to the Kaldi
                               // verbose level.
  if (!RandEquivalent(clat, aligned_clat, num_paths, delta, seed, max_path_length)) {
    KALDI_WARN << "Equivalence test failed during lattice alignment.";
    return false;
  }
  FLAGS_v = 0;

  return (num_err == 0);
}

}  // namespace kaldi
