// lat/sausages.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
//           2015  Guoguo Chen
//           2019  Dogan Can

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


#ifndef KALDI_LAT_SAUSAGES_H_
#define KALDI_LAT_SAUSAGES_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/// The implementation of the Minimum Bayes Risk decoding method described in
///  "Minimum Bayes Risk decoding and system combination based on a recursion for
///  edit distance", Haihua Xu, Daniel Povey, Lidia Mangu and Jie Zhu, Computer
///  Speech and Language, 2011
/// This is a slightly more principled way to do Minimum Bayes Risk (MBR) decoding
/// than the standard "Confusion Network" method.  Note: MBR decoding aims to
/// minimize the expected word error rate, assuming the lattice encodes the
/// true uncertainty about what was spoken; standard Viterbi decoding gives the
/// most likely utterance, which corresponds to minimizing the expected sentence
/// error rate.
///
/// In addition to giving the MBR output, we also provide a way to get a
/// "Confusion Network" or informally "sausage"-like structure.  This is a
/// linear sequence of bins, and in each bin, there is a distribution over
/// words (or epsilon, meaning no word).  This is useful for estimating
/// confidence.  Note: due to the way these sausages are made, typically there
/// will be, between each bin representing a high-confidence word, a bin
/// in which epsilon (no word) is the most likely word.  Inside these bins
/// is where we put possible insertions.

struct MinimumBayesRiskOptions {
  /// Boolean configuration parameter: if true, we actually update the hypothesis
  /// to do MBR decoding (if false, our output is the MAP decoded output, but we
  /// output the stats too (i.e. the confidences)).
  bool decode_mbr;
  /// Boolean configuration parameter: if true, the 1-best path will 'keep' the <eps> bins,
  bool print_silence;

  MinimumBayesRiskOptions() : decode_mbr(true), print_silence(false)
  { }
  void Register(OptionsItf *opts) {
    opts->Register("decode-mbr", &decode_mbr, "If true, do Minimum Bayes Risk "
                   "decoding (else, Maximum a Posteriori)");
    opts->Register("print-silence", &print_silence, "Keep the inter-word '<eps>' "
                   "bins in the 1-best output (ctm, <eps> can be a 'silence' or a 'deleted' word)");
  }
};

/// This class does the word-level Minimum Bayes Risk computation, and gives you
/// either the 1-best MBR output together with the expected Bayes Risk,
/// or a sausage-like structure.
class MinimumBayesRisk {
 public:
  /// Initialize with compact lattice-- any acoustic scaling etc., is assumed
  /// to have been done already.
  /// This does the whole computation.  You get the output with
  /// GetOneBest(), GetBayesRisk(), and GetSausageStats().
  MinimumBayesRisk(const CompactLattice &clat,
                   MinimumBayesRiskOptions opts = MinimumBayesRiskOptions());

  // Uses the provided <words> as <R_> instead of using the lattice best path.
  // Note that the default value of opts.decode_mbr is true. If you provide 1-best
  // hypothesis from MAP decoding, the output ctm from MBR decoding may be
  // mismatched with the provided <words> (<words> would be used as the starting
  // point of optimization).
  MinimumBayesRisk(const CompactLattice &clat,
                   const std::vector<int32> &words,
                   MinimumBayesRiskOptions opts = MinimumBayesRiskOptions());
  // Uses the provided <words> as <R_> and <times> of bins instead of using the lattice best path.
  // Note that the default value of opts.decode_mbr is true. If you provide 1-best
  // hypothesis from MAP decoding, the output ctm from MBR decoding may be
  // mismatched with the provided <words> (<words> would be used as the starting
  // point of optimization).
  MinimumBayesRisk(const CompactLattice &clat,
                   const std::vector<int32> &words,
                   const std::vector<std::pair<BaseFloat,BaseFloat> > &times,
                   MinimumBayesRiskOptions opts = MinimumBayesRiskOptions());

  const std::vector<int32> &GetOneBest() const { // gets one-best (with no epsilons)
    return R_;
  }

  const std::vector<std::vector<std::pair<BaseFloat, BaseFloat> > > GetTimes() const {
    return times_; // returns average (start,end) times for each word in each
    // bin. These are raw averages without any processing, i.e. time intervals
    // from different bins can overlap.
  }

  const std::vector<std::pair<BaseFloat, BaseFloat> > GetSausageTimes() const {
    return sausage_times_; // returns average (start,end) times for each bin.
    // This is typically the weighted average of the times in GetTimes() but can
    // be slightly different if the times for the bins overlap, in which case
    // the times returned by this method do not overlap unlike the times
    // returned by GetTimes().
  }

  const std::vector<std::pair<BaseFloat, BaseFloat> > &GetOneBestTimes() const {
    return one_best_times_; // returns average (start,end) times for each word
    // corresponding to an entry in the one-best output.  This is typically the
    // appropriate subset of the times in GetTimes() but can be slightly
    // different if the times for the one-best words overlap, in which case
    // the times returned by this method do not overlap unlike the times
    // returned by GetTimes().
  }

  /// Outputs the confidences for the one-best transcript.
  const std::vector<BaseFloat> &GetOneBestConfidences() const {
    return one_best_confidences_;
  }

  /// Returns the expected WER over this sentence (assuming model correctness).
  BaseFloat GetBayesRisk() const { return L_; }

  const std::vector<std::vector<std::pair<int32, BaseFloat> > > &GetSausageStats() const {
    return gamma_;
  }

 private:
  void PrepareLatticeAndInitStats(CompactLattice *clat);

  /// Minimum-Bayes-Risk Decode. Top-level algorithm.  Figure 6 of the paper.
  void MbrDecode();

  /// Without the 'penalize' argument this gives us the basic edit-distance
  /// function l(a,b), as in the paper.
  /// With the 'penalize' argument it can be interpreted as the edit distance
  /// plus the 'delta' from the paper, except that we make a kind of conceptual
  /// bug-fix and only apply the delta if the edit-distance was not already
  /// zero.  This bug-fix was necessary in order to force all the stats to show
  /// up, that should show up, and applying the bug-fix makes the sausage stats
  /// significantly less sparse.
  inline double l(int32 a, int32 b, bool penalize = false) {
    if (a == b) return 0.0;
    else return (penalize ? 1.0 + delta() : 1.0);
  }

  /// returns r_q, in one-based indexing, as in the paper.
  inline int32 r(int32 q) { return R_[q-1]; }


  /// Figure 4 of the paper; called from AccStats (Fig. 5)
  double EditDistance(int32 N, int32 Q,
                      Vector<double> &alpha,
                      Matrix<double> &alpha_dash,
                      Vector<double> &alpha_dash_arc);

  /// Figure 5 of the paper.  Outputs to gamma_ and L_.
  void AccStats();

  /// Removes epsilons (symbol 0) from a vector
  static void RemoveEps(std::vector<int32> *vec);

  // Ensures that between each word in "vec" and at the beginning and end, is
  // epsilon (0).  (But if no words in vec, just one epsilon)
  static void NormalizeEps(std::vector<int32> *vec);

  // delta() is a constant used in the algorithm, which penalizes
  // the use of certain epsilon transitions in the edit-distance which would cause
  // words not to show up in the accumulated edit-distance statistics.
  // There has been a conceptual bug-fix versus the way it was presented in
  // the paper: we now add delta only if the edit-distance was not already
  // zero.
  static inline BaseFloat delta() { return 1.0e-05; }


  /// Function used to increment map.
  static inline void AddToMap(int32 i, double d, std::map<int32, double> *gamma) {
    if (d == 0) return;
    std::pair<const int32, double> pr(i, d);
    std::pair<std::map<int32, double>::iterator, bool> ret = gamma->insert(pr);
    if (!ret.second) // not inserted, so add to contents.
      ret.first->second += d;
  }

  struct Arc {
    int32 word;
    int32 start_node;
    int32 end_node;
    BaseFloat loglike;
  };

  MinimumBayesRiskOptions opts_;


  /// Arcs in the topologically sorted acceptor form of the word-level lattice,
  /// with one final-state.  Contains (word-symbol, log-likelihood on arc ==
  /// negated cost).  Indexed from zero.
  std::vector<Arc> arcs_;

  /// For each node in the lattice, a list of arcs entering that node. Indexed
  /// from 1 (first node == 1).
  std::vector<std::vector<int32> > pre_;

  std::vector<int32> state_times_; // time of each state in the word lattice,
  // indexed from 1 (same index as into pre_)

  std::vector<int32> R_; // current 1-best word sequence, normalized to have
  // epsilons between each word and at the beginning and end.  R in paper...
  // caution: indexed from zero, not from 1 as in paper.

  double L_; // current averaged edit-distance between lattice and R_.
  // \hat{L} in paper.

  std::vector<std::vector<std::pair<int32, BaseFloat> > > gamma_;
  // The stats we accumulate; these are pairs of (posterior, word-id), and note
  // that word-id may be epsilon.  Caution: indexed from zero, not from 1 as in
  // paper.  We sort in reverse order on the second member (posterior), so more
  // likely word is first.

  std::vector<std::vector<std::pair<BaseFloat, BaseFloat> > > times_;
  // The average start and end times for words in each confusion-network bin.
  // This is like an average over arcs, of the tau_b and tau_e quantities in
  // Appendix C of the paper.  Indexed from zero, like gamma_ and R_.

  std::vector<std::pair<BaseFloat, BaseFloat> > sausage_times_;
  // The average start and end times for each confusion-network bin.  This
  // is like an average over words, of the tau_b and tau_e quantities in
  // Appendix C of the paper.  Indexed from zero, like gamma_ and R_.

  std::vector<std::pair<BaseFloat, BaseFloat> > one_best_times_;
  // The average start and end times for words in the one best output.  This
  // is like an average over the arcs, of the tau_b and tau_e quantities in
  // Appendix C of the paper. Indexed from zero, like gamma_ and R_.

  std::vector<BaseFloat> one_best_confidences_;
  // vector of confidences for the 1-best output (which could be
  // the MAP output if opts_.decode_mbr == false, or the MBR output otherwise).
  // Indexed by the same index as one_best_times_.

  struct GammaCompare{
    // should be like operator <.  But we want reverse order
    // on the 2nd element (posterior), so it'll be like operator
    // > that looks first at the posterior.
    bool operator () (const std::pair<int32, BaseFloat> &a,
                      const std::pair<int32, BaseFloat> &b) const {
      if (a.second > b.second) return true;
      else if (a.second < b.second) return false;
      else return a.first > b.first;
    }
  };
};

}  // namespace kaldi

#endif  // KALDI_LAT_SAUSAGES_H_
