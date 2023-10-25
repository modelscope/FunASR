// lat/lattice-functions.h

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


#ifndef KALDI_LAT_LATTICE_FUNCTIONS_H_
#define KALDI_LAT_LATTICE_FUNCTIONS_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
// #include "hmm/posterior.h"
#include "fstext/fstext-lib.h"
// #include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
// #include "itf/decodable-itf.h"

namespace kaldi {

// /**
//    This function extracts the per-frame log likelihoods from a linear
//    lattice (which we refer to as an 'nbest' lattice elsewhere in Kaldi code).
//    The dimension of *per_frame_loglikes will be set to the
//    number of input symbols in 'nbest'.  The elements of
//    '*per_frame_loglikes' will be set to the .Value2() elements of the lattice
//    weights, which represent the acoustic costs; you may want to scale this
//    vector afterward by -1/acoustic_scale to get the original loglikes.
//    If there are acoustic costs on input-epsilon arcs or the final-prob in 'nbest'
//    (and this should not normally be the case in situations where it makes
//    sense to call this function), they will be included to the cost of the
//    preceding input symbol, or the following input symbol for input-epsilons
//    encountered prior to any input symbol.  If 'nbest' has no input symbols,
//    'per_frame_loglikes' will be set to the empty vector.
// **/
// void GetPerFrameAcousticCosts(const Lattice &nbest,
//                               Vector<BaseFloat> *per_frame_loglikes);
//
// /// This function iterates over the states of a topologically sorted lattice and
// /// counts the time instance corresponding to each state. The times are returned
// /// in a vector of integers 'times' which is resized to have a size equal to the
// /// number of states in the lattice. The function also returns the maximum time
// /// in the lattice (this will equal the number of frames in the file).
// int32 LatticeStateTimes(const Lattice &lat, std::vector<int32> *times);
//
// /// As LatticeStateTimes, but in the CompactLattice format.  Note: must
// /// be topologically sorted.  Returns length of the utterance in frames, which
// /// might not be the same as the maximum time in the lattice, due to frames
// /// in the final-prob.
// int32 CompactLatticeStateTimes(const CompactLattice &clat,
//                                std::vector<int32> *times);
//
// /// This function does the forward-backward over lattices and computes the
// /// posterior probabilities of the arcs. It returns the total log-probability
// /// of the lattice.  The Posterior quantities contain pairs of (transition-id, weight)
// /// on each frame.
// /// If the pointer "acoustic_like_sum" is provided, this value is set to
// /// the sum over the arcs, of the posterior of the arc times the
// /// acoustic likelihood [i.e. negated acoustic score] on that link.
// /// This is used in combination with other quantities to work out
// /// the objective function in MMI discriminative training.
// BaseFloat LatticeForwardBackward(const Lattice &lat,
//                                  Posterior *arc_post,
//                                  double *acoustic_like_sum = NULL);
//
// // This function is something similar to LatticeForwardBackward(), but it is on
// // the CompactLattice lattice format. Also we only need the alpha in the forward
// // path, not the posteriors.
// bool ComputeCompactLatticeAlphas(const CompactLattice &lat,
//                                  std::vector<double> *alpha);
//
// // A sibling of the function CompactLatticeAlphas()... We compute the beta from
// // the backward path here.
// bool ComputeCompactLatticeBetas(const CompactLattice &lat,
//                                 std::vector<double> *beta);
//
//
// // Computes (normal or Viterbi) alphas and betas; returns (total-prob, or
// // best-path negated cost) Note: in either case, the alphas and betas are
// // negated costs.  Requires that lat be topologically sorted.  This code
// // will work for either CompactLattice or Latice.
// template<typename LatticeType>
// double ComputeLatticeAlphasAndBetas(const LatticeType &lat,
//                                     bool viterbi,
//                                     std::vector<double> *alpha,
//                                     std::vector<double> *beta);
//
//
// /// Topologically sort the compact lattice if not already topologically sorted.
// /// Will crash if the lattice cannot be topologically sorted.
// void TopSortCompactLatticeIfNeeded(CompactLattice *clat);
//
//
// /// Topologically sort the lattice if not already topologically sorted.
// /// Will crash if lattice cannot be topologically sorted.
// void TopSortLatticeIfNeeded(Lattice *clat);
//
// /// Returns the depth of the lattice, defined as the average number of arcs (or
// /// final-prob strings) crossing any given frame.  Returns 1 for empty lattices.
// /// Requires that clat is topologically sorted!
// BaseFloat CompactLatticeDepth(const CompactLattice &clat,
//                               int32 *num_frames = NULL);
//
// /// This function returns, for each frame, the number of arcs crossing that
// /// frame.
// void CompactLatticeDepthPerFrame(const CompactLattice &clat,
//                                  std::vector<int32> *depth_per_frame);
//
//
// /// This function limits the depth of the lattice, per frame: that means, it
// /// does not allow more than a specified number of arcs active on any given
// /// frame.  This can be used to reduce the size of the "very deep" portions of
// /// the lattice.
// void CompactLatticeLimitDepth(int32 max_arcs_per_frame,
//                               CompactLattice *clat);
//
//
// /// Given a lattice, and a transition model to map pdf-ids to phones,
// /// outputs for each frame the set of phones active on that frame.  If
// /// sil_phones (which must be sorted and uniq) is nonempty, it excludes
// /// phones in this list.
// void LatticeActivePhones(const Lattice &lat, const TransitionModel &trans,
//                          const std::vector<int32> &sil_phones,
//                          std::vector<std::set<int32> > *active_phones);
//
// /// Given a lattice, and a transition model to map pdf-ids to phones,
// /// replace the output symbols (presumably words), with phones; we
// /// use the TransitionModel to work out the phone sequence.  Note
// /// that the phone labels are not exactly aligned with the phone
// /// boundaries.  We put a phone label to coincide with any transition
// /// to the final, nonemitting state of a phone (this state always exists,
// /// we ensure this in HmmTopology::Check()).  This would be the last
// /// transition-id in the phone if reordering is not done (but typically
// /// we do reorder).
// /// Also see PhoneAlignLattice, in phone-align-lattice.h.
// void ConvertLatticeToPhones(const TransitionModel &trans_model,
//                             Lattice *lat);

/// Prunes a lattice or compact lattice.  Returns true on success, false if
/// there was some kind of failure.
template<class LatticeType>
bool PruneLattice(BaseFloat beam, LatticeType *lat);

//
// /// Given a lattice, and a transition model to map pdf-ids to phones,
// /// replace the sequences of transition-ids with sequences of phones.
// /// Note that this is different from ConvertLatticeToPhones, in that
// /// we replace the transition-ids not the words.
// void ConvertCompactLatticeToPhones(const TransitionModel &trans_model,
//                                    CompactLattice *clat);
//
// /// Boosts LM probabilities by b * [number of frame errors]; equivalently, adds
// /// -b*[number of frame errors] to the graph-component of the cost of each arc/path.
// /// There is a frame error if a particular transition-id on a particular frame
// /// corresponds to a phone not matching transcription's alignment for that frame.
// /// This is used in "margin-inspired" discriminative training, esp. Boosted MMI.
// /// The TransitionModel is used to map transition-ids in the lattice
// /// input-side to phones; the phones appearing in
// /// "silence_phones" are treated specially in that we replace the frame error f
// /// (either zero or 1) for a frame, with the minimum of f or max_silence_error.
// /// For the normal recipe, max_silence_error would be zero.
// /// Returns true on success, false if there was some kind of mismatch.
// /// At input, silence_phones must be sorted and unique.
// bool LatticeBoost(const TransitionModel &trans,
//                   const std::vector<int32> &alignment,
//                   const std::vector<int32> &silence_phones,
//                   BaseFloat b,
//                   BaseFloat max_silence_error,
//                   Lattice *lat);
//
//
// /**
//    This function implements either the MPFE (minimum phone frame error) or SMBR
//    (state-level minimum bayes risk) forward-backward, depending on whether
//    "criterion" is "mpfe" or "smbr".  It returns the MPFE
//    criterion of SMBR criterion for this utterance, and outputs the posteriors (which
//    may be positive or negative) into "post".
//
//    @param [in] trans    The transition model. Used to map the
//                         transition-ids to phones or pdfs.
//    @param [in] silence_phones   A list of integer ids of silence phones. The
//                         silence frames i.e. the frames where num_ali
//                         corresponds to a silence phones are treated specially.
//                         The behavior is determined by 'one_silence_class'
//                         being false (traditional behavior) or true.
//                         Usually in our setup, several phones including
//                         the silence, vocalized noise, non-spoken noise
//                         and unk are treated as "silence phones"
//    @param [in] lat      The denominator lattice
//    @param [in] num_ali  The numerator alignment
//    @param [in] criterion    The objective function. Must be "mpfe" or "smbr"
//                         for MPFE (minimum phone frame error) or sMBR
//                         (state minimum bayes risk) training.
//    @param [in] one_silence_class   Determines how the silence frames are treated.
//                         Setting this to false gives the old traditional behavior,
//                         where the silence frames (according to num_ali) are
//                         treated as incorrect. However, this means that the
//                         insertions are not penalized by the objective.
//                         Setting this to true gives the new behaviour, where we
//                         treat silence as any other phone, except that all pdfs
//                         of silence phones are collapsed into a single class for
//                         the frame-error computation. This can possible reduce
//                         the insertions in the trained model. This is closer to
//                         the WER metric that we actually care about, since WER is
//                         generally computed after filtering out noises, but
//                         does penalize insertions.
//     @param [out] post   The "MBR posteriors" i.e. derivatives w.r.t to the
//                         pseudo log-likelihoods of states at each frame.
// */
// BaseFloat LatticeForwardBackwardMpeVariants(
//     const TransitionModel &trans,
//     const std::vector<int32> &silence_phones,
//     const Lattice &lat,
//     const std::vector<int32> &num_ali,
//     std::string criterion,
//     bool one_silence_class,
//     Posterior *post);
//
// /**
//    This function can be used to compute posteriors for MMI, with a positive contribution
//    for the numerator and a negative one for the denominator.  This function is not actually
//    used in our normal MMI training recipes, where it's instead done using various command
//    line programs that each do a part of the job.  This function was written for use in
//    neural-net MMI training.
//
//    @param [in] trans    The transition model. Used to map the
//                         transition-ids to phones or pdfs.
//    @param [in] lat      The denominator lattice
//    @param [in] num_ali  The numerator alignment
//    @param [in] drop_frames   If "drop_frames" is true, it will not compute any
//                         posteriors on frames where the num and den have disjoint
//                         pdf-ids.
//    @param [in] convert_to_pdf_ids   If "convert_to_pdfs_ids" is true, it will
//                         convert the output to be at the level of pdf-ids, not
//                         transition-ids.
//    @param [in] cancel   If "cancel" is true, it will cancel out any positive and
//                         negative parts from the same transition-id (or pdf-id,
//                         if convert_to_pdf_ids == true).
//    @param [out] arc_post   The output MMI posteriors of transition-ids (or
//                         pdf-ids if convert_to_pdf_ids == true) at each frame
//                         i.e. the difference between the numerator
//                         and denominator posteriors.
//
//    It returns the forward-backward likelihood of the lattice. */
// BaseFloat LatticeForwardBackwardMmi(
//     const TransitionModel &trans,
//     const Lattice &lat,
//     const std::vector<int32> &num_ali,
//     bool drop_frames,
//     bool convert_to_pdf_ids,
//     bool cancel,
//     Posterior *arc_post);
//
//
// /// This function takes a CompactLattice that should only contain a single
// /// linear sequence (e.g. derived from lattice-1best), and that should have been
// /// processed so that the arcs in the CompactLattice align correctly with the
// /// word boundaries (e.g. by lattice-align-words).  It outputs 3 vectors of the
// /// same size, which give, for each word in the lattice (in sequence), the word
// /// label and the begin time and length in frames.  This is done even for zero
// /// (epsilon) words, generally corresponding to optional silence-- if you don't
// /// want them, just ignore them in the output.
// /// This function will print a warning and return false, if the lattice
// /// did not have the correct format (e.g. if it is empty or it is not
// /// linear).
// bool CompactLatticeToWordAlignment(const CompactLattice &clat,
//                                    std::vector<int32> *words,
//                                    std::vector<int32> *begin_times,
//                                    std::vector<int32> *lengths);
//
// /// This function takes a CompactLattice that should only contain a single
// /// linear sequence (e.g. derived from lattice-1best), and that should have been
// /// processed so that the arcs in the CompactLattice align correctly with the
// /// word boundaries (e.g. by lattice-align-words).  It outputs 4 vectors of the
// /// same size, which give, for each word in the lattice (in sequence), the word
// /// label, the begin time and length in frames, and the pronunciation (sequence
// /// of phones).  This is done even for zero words, corresponding to optional
// /// silences -- if you don't want them, just ignore them in the output.
// /// This function will print a warning and return false, if the lattice
// /// did not have the correct format (e.g. if it is empty or it is not
// /// linear).
// bool CompactLatticeToWordProns(
//     const TransitionModel &tmodel,
//     const CompactLattice &clat,
//     std::vector<int32> *words,
//     std::vector<int32> *begin_times,
//     std::vector<int32> *lengths,
//     std::vector<std::vector<int32> > *prons,
//     std::vector<std::vector<int32> > *phone_lengths);
//
//
// /// A form of the shortest-path/best-path algorithm that's specially coded for
// /// CompactLattice.  Requires that clat be acyclic.
// void CompactLatticeShortestPath(const CompactLattice &clat,
//                                 CompactLattice *shortest_path);
//
// /// This function expands a CompactLattice to ensure high-probability paths
// /// have unique histories. Arcs with posteriors larger than epsilon get splitted.
// void ExpandCompactLattice(const CompactLattice &clat,
//                           double epsilon,
//                           CompactLattice *expand_clat);
//
// /// For each state, compute forward and backward best (viterbi) costs and its
// /// traceback states (for generating best paths later). The forward best cost
// /// for a state is the cost of the best path from the start state to the state.
// /// The traceback state of this state is its predecessor state in the best path.
// /// The backward best cost for a state is the cost of the best path from the
// /// state to a final one. Its traceback state is the successor state in the best
// /// path in the forward direction.
// /// Note: final weights of states are in backward_best_cost_and_pred.
// /// Requires the input CompactLattice clat be acyclic.
// typedef std::vector<std::pair<double,
//         CompactLatticeArc::StateId> > CostTraceType;
// void CompactLatticeBestCostsAndTracebacks(
//     const CompactLattice &clat,
//     CostTraceType *forward_best_cost_and_pred,
//     CostTraceType *backward_best_cost_and_pred);
//
// /// This function adds estimated neural language model scores of words in a
// /// minimal list of hypotheses that covers a lattice, to the graph scores on the
// /// arcs. The list of hypotheses are generated by latbin/lattice-path-cover.
// typedef unordered_map<std::pair<int32, int32>, double, PairHasher<int32> > MapT;
// void AddNnlmScoreToCompactLattice(const MapT &nnlm_scores,
//                                   CompactLattice *clat);
//
// /// This function add the word insertion penalty to graph score of each word
// /// in the compact lattice
// void AddWordInsPenToCompactLattice(BaseFloat word_ins_penalty,
//                                    CompactLattice *clat);
//
// /// This function *adds* the negated scores obtained from the Decodable object,
// /// to the acoustic scores on the arcs.  If you want to replace them, you should
// /// use ScaleCompactLattice to first set the acoustic scores to zero.  Returns
// /// true on success, false on error (typically some kind of mismatched inputs).
// bool RescoreCompactLattice(DecodableInterface *decodable,
//                            CompactLattice *clat);
//
//
// /// This function returns the number of words in the longest sentence in a
// /// CompactLattice (i.e. the the maximum of any path, of the count of
// /// olabels on that path).
// int32 LongestSentenceLength(const Lattice &lat);
//
// /// This function returns the number of words in the longest sentence in a
// /// CompactLattice, i.e. the the maximum of any path, of the count of
// /// labels on that path... note, in CompactLattice, the ilabels and olabels
// /// are identical because it is an acceptor.
// int32 LongestSentenceLength(const CompactLattice &lat);
//
//
// /// This function is like RescoreCompactLattice, but it is modified to avoid
// /// computing probabilities on most frames where all the pdf-ids are the same.
// /// (it needs the transition-model to work out whether two transition-ids map to
// /// the same pdf-id, and it assumes that the lattice has transition-ids on it).
// /// The naive thing would be to just set all probabilities to zero on frames
// /// where all the pdf-ids are the same (because this value won't affect the
// /// lattice posterior).  But this would become confusing when we compute
// /// corpus-level diagnostics such as the MMI objective function.  Instead,
// /// imagine speedup_factor = 100 (it must be >= 1.0)... with probability (1.0 /
// /// speedup_factor) we compute those likelihoods and multiply them by
// /// speedup_factor; otherwise we set them to zero.  This gives the right
// /// expected probability so our corpus-level diagnostics will be about right.
// bool RescoreCompactLatticeSpeedup(
//     const TransitionModel &tmodel,
//     BaseFloat speedup_factor,
//     DecodableInterface *decodable,
//     CompactLattice *clat);
//
//
// /// This function *adds* the negated scores obtained from the Decodable object,
// /// to the acoustic scores on the arcs.  If you want to replace them, you should
// /// use ScaleCompactLattice to first set the acoustic scores to zero.  Returns
// /// true on success, false on error (e.g. some kind of mismatched inputs).
// /// The input labels, if nonzero, are interpreted as transition-ids or whatever
// /// other index the Decodable object expects.
// bool RescoreLattice(DecodableInterface *decodable,
//                     Lattice *lat);
//
// /// This function Composes a CompactLattice format lattice with a
// /// DeterministicOnDemandFst<fst::StdFst> format fst, and outputs another
// /// CompactLattice format lattice. The first element (the one that corresponds
// /// to LM weight) in CompactLatticeWeight is used for composition.
// ///
// /// Note that the DeterministicOnDemandFst interface is not "const", therefore
// /// we cannot use "const" for <det_fst>.
// void ComposeCompactLatticeDeterministic(
//     const CompactLattice& clat,
//     fst::DeterministicOnDemandFst<fst::StdArc>* det_fst,
//     CompactLattice* composed_clat);
//
// /// This function computes the mapping from the pair
// /// (frame-index, transition-id) to the pair
// /// (sum-of-acoustic-scores, num-of-occurences) over all occurences of the
// /// transition-id in that frame.
// /// frame-index in the lattice.
// /// This function is useful for retaining the acoustic scores in a
// /// non-compact lattice after a process like determinization where the
// /// frame-level acoustic scores are typically lost.
// /// The function ReplaceAcousticScoresFromMap is used to restore the
// /// acoustic scores computed by this function.
// ///
// ///   @param [in] lat   Input lattice. Expected to be top-sorted. Otherwise the
// ///                     function will crash.
// ///   @param [out] acoustic_scores
// ///                     Pointer to a map from the pair (frame-index,
// ///                     transition-id) to a pair (sum-of-acoustic-scores,
// ///                     num-of-occurences).
// ///                     Usually the acoustic scores for a pdf-id (and hence
// ///                     transition-id) on a frame will be the same for all the
// ///                     occurences of the pdf-id in that frame.
// ///                     But if not, we will take the average of the acoustic
// ///                     scores. Hence, we store both the sum-of-acoustic-scores
// ///                     and the num-of-occurences of the transition-id in that
// ///                     frame.
// void ComputeAcousticScoresMap(
//     const Lattice &lat,
//     unordered_map<std::pair<int32, int32>, std::pair<BaseFloat, int32>,
//                                         PairHasher<int32> > *acoustic_scores);
//
// /// This function restores acoustic scores computed using the function
// /// ComputeAcousticScoresMap into the lattice.
// ///
// ///   @param [in] acoustic_scores
// ///                      A map from the pair (frame-index, transition-id) to a
// ///                      pair (sum-of-acoustic-scores, num-of-occurences) of
// ///                      the occurences of the transition-id in that frame.
// ///                      See the comments for ComputeAcousticScoresMap for
// ///                      details.
// ///   @param [out] lat   Pointer to the output lattice.
// void ReplaceAcousticScoresFromMap(
//     const unordered_map<std::pair<int32, int32>, std::pair<BaseFloat, int32>,
//                                         PairHasher<int32> > &acoustic_scores,
//     Lattice *lat);

}  // namespace kaldi

#endif  // KALDI_LAT_LATTICE_FUNCTIONS_H_
