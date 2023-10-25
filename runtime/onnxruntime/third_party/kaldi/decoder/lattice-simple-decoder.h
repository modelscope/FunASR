// decoder/lattice-simple-decoder.h

// Copyright 2009-2012  Microsoft Corporation
//           2012-2014  Johns Hopkins University (Author: Daniel Povey)
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

#ifndef KALDI_DECODER_LATTICE_SIMPLE_DECODER_H_
#define KALDI_DECODER_LATTICE_SIMPLE_DECODER_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"

#include <algorithm>

namespace kaldi {

struct LatticeSimpleDecoderConfig {
  BaseFloat beam;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
  // command-line program.
  bool prune_lattice;
  BaseFloat beam_ratio;
  BaseFloat prune_scale;   // Note: we don't make this configurable on the command line,
                           // it's not a very important parameter.  It affects the
                           // algorithm that prunes the tokens as we go.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  LatticeSimpleDecoderConfig(): beam(16.0),
                                lattice_beam(10.0),
                                prune_interval(25),
                                determinize_lattice(true),
                                beam_ratio(0.9),
                                prune_scale(0.1) { }
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (in a special sense, keeping only "
                   "best pdf-sequence for each word-sequence).");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && lattice_beam > 0.0 && prune_interval > 0);
  }
};


/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class LatticeSimpleDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  // instantiate this class once for each thing you have to decode.
  LatticeSimpleDecoder(const fst::Fst<fst::StdArc> &fst,
                       const LatticeSimpleDecoderConfig &config):
      fst_(fst), config_(config), num_toks_(0) { config.Check(); }
  
  ~LatticeSimpleDecoder() { ClearActiveTokens(); }

  const LatticeSimpleDecoderConfig &GetOptions() const {
    return config_;
  }

  // Returns true if any kind of traceback is available (not necessarily from
  // a final state).
  bool Decode(DecodableInterface *decodable);


  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }
  
  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding();

  /// This function may be optionally called after AdvanceDecoding(), when you
  /// do not plan to decode any further.  It does an extra pruning step that
  /// will help to prune the lattices output by GetLattice and (particularly)
  /// GetRawLattice more accurately, particularly toward the end of the
  /// utterance.  It does this by using the final-probs in pruning (if any
  /// final-state survived); it also does a final pruning step that visits all
  /// states (the pruning that is done during decoding may fail to prune states
  /// that are within kPruningScale = 0.1 outside of the beam).  If you call
  /// this, you cannot call AdvanceDecoding again (it will fail), and you
  /// cannot call GetLattice() and related functions with use_final_probs =
  /// false.
  /// Used to be called PruneActiveTokensFinal().
  void FinalizeDecoding();

  /// FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.  If it not
  /// too positive (e.g. < 5 is my first guess, but this is not tested) you can
  /// take it as a good indication that we reached the final-state with
  /// reasonable likelihood.
  BaseFloat FinalRelativeCost() const;
  
  // Outputs an FST corresponding to the single best path
  // through the lattice.  Returns true if result is nonempty
  // (using the return status is deprecated, it will become void).
  // If "use_final_probs" is true AND we reached the final-state
  // of the graph then it will include those as final-probs, else
  // it will treat all final-probs as one.
  bool GetBestPath(Lattice *lat,
                   bool use_final_probs = true) const;

  // Outputs an FST corresponding to the raw, state-level
  // tracebacks.  Returns true if result is nonempty
  // (using the return status is deprecated, it will become void).
  // If "use_final_probs" is true AND we reached the final-state
  // of the graph then it will include those as final-probs, else
  // it will treat all final-probs as one.
  bool GetRawLattice(Lattice *lat,
                     bool use_final_probs = true) const;

  // This function is now deprecated, since now we do determinization from
  // outside the LatticeTrackingDecoder class.
  // Outputs an FST corresponding to the lattice-determinized
  // lattice (one path per word sequence).  [will become deprecated,
  // users should determinize themselves.]
  bool GetLattice(CompactLattice *clat,
                  bool use_final_probs = true) const;
  
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }  
 private:
  struct Token;
  // ForwardLinks are the links from a token to a token on the next frame.
  // or sometimes on the current frame (for input-epsilon links).
  struct ForwardLink {
    Token *next_tok; // the next token [or NULL if represents final-state]
    Label ilabel; // ilabel on link.
    Label olabel; // olabel on link.
    BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    ForwardLink *next; // next in singly-linked list of forward links from a
                       // token.
    ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                BaseFloat graph_cost, BaseFloat acoustic_cost, 
                ForwardLink *next):
        next_tok(next_tok), ilabel(ilabel), olabel(olabel),
        graph_cost(graph_cost), acoustic_cost(acoustic_cost), 
        next(next) { }
  };  
  
  // Token is what's resident in a particular state at a particular time.
  // In this decoder a Token actually contains *forward* links.
  // When first created, a Token just has the (total) cost.    We add forward
  // links from it when we process the next frame.
  struct Token {
    BaseFloat tot_cost; // would equal weight.Value()... cost up to this point.
    BaseFloat extra_cost; // >= 0.  After calling PruneForwardLinks, this equals
    // the minimum difference between the cost of the best path this is on,
    // and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).
    
    ForwardLink *links; // Head of singly linked list of ForwardLinks
    
    Token *next; // Next in list of tokens for this frame.
    
    Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
          Token *next): tot_cost(tot_cost), extra_cost(extra_cost), links(links),
                        next(next) { }
    Token() {}
    void DeleteForwardLinks() {
      ForwardLink *l = links, *m; 
      while (l != NULL) {
        m = l->next;
        delete l;
        l = m;
      }
      links = NULL;
    }
  };
  
  // head and tail of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList(): toks(NULL), must_prune_forward_links(true),
                 must_prune_tokens(true) { }
  };
  

  // FindOrAddToken either locates a token in cur_toks_, or if necessary inserts a new,
  // empty token (i.e. with no forward links) for the current frame.  [note: it's
  // inserted if necessary into cur_toks_ and also into the singly linked list
  // of tokens active on this frame (whose head is at active_toks_[frame]).
  //
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  inline Token *FindOrAddToken(StateId state, int32 frame_plus_one,
                               BaseFloat tot_cost, bool emitting, bool *changed);
  
  // delta is the amount by which the extra_costs must
  // change before it sets "extra_costs_changed" to true.  If delta is larger,
  // we'll tend to go back less far toward the beginning of the file.
  void PruneForwardLinks(int32 frame, bool *extra_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta);

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses the final-probs
  // for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();
  
  // Prune away any tokens on this frame that have no forward links. [we don't do
  // this in PruneForwardLinks because it would give us a problem with dangling
  // pointers].
  void PruneTokensForFrame(int32 frame);
  
  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneActiveTokens(BaseFloat delta);

  void ProcessEmitting(DecodableInterface *decodable);

  void ProcessNonemitting();

  void ClearActiveTokens(); // a cleanup routine, at utt end/begin

  // This function computes the final-costs for tokens active on the final
  // frame.  It outputs to final-costs, if non-NULL, a map from the Token*
  // pointer to the final-prob of the corresponding state, or zero for all states if
  // none were final.  It outputs to final_relative_cost, if non-NULL, the
  // difference between the best forward-cost including the final-prob cost, and
  // the best forward-cost without including the final-prob cost (this will
  // usually be positive), or infinity if there were no final-probs.  It outputs
  // to final_best_cost, if non-NULL, the lowest for any token t active on the
  // final frame, of t + final-cost[t], where final-cost[t] is the final-cost
  // in the graph of the state corresponding to token t, or zero if there
  // were no final-probs active on the final frame.
  // You cannot call this after FinalizeDecoding() has been called; in that
  // case you should get the answer from class-member variables.
  void ComputeFinalCosts(unordered_map<Token*, BaseFloat> *final_costs,
                         BaseFloat *final_relative_cost,
                         BaseFloat *final_best_cost) const;
  

  // PruneCurrentTokens deletes the tokens from the "toks" map, but not
  // from the active_toks_ list, which could cause dangling forward pointers
  // (will delete it during regular pruning operation).
  void PruneCurrentTokens(BaseFloat beam, unordered_map<StateId, Token*> *toks);
  
  unordered_map<StateId, Token*> cur_toks_;
  unordered_map<StateId, Token*> prev_toks_;
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame_plus_one
  const fst::Fst<fst::StdArc> &fst_;
  LatticeSimpleDecoderConfig config_;
  int32 num_toks_; // current total #toks allocated...
  bool warned_;


  /// decoding_finalized_ is true if someone called FinalizeDecoding().  [note,
  /// calling this is optional].  If true, it's forbidden to decode more.  Also,
  /// if this is set, then the output of ComputeFinalCosts() is in the next
  /// three variables.  The reason we need to do this is that after
  /// FinalizeDecoding() calls PruneTokensForFrame() for the final frame, some
  /// of the tokens on the last frame are freed, so we free the list from
  /// cur_toks_ to avoid having dangling pointers hanging around.
  bool decoding_finalized_;
  /// For the meaning of the next 3 variables, see the comment for
  /// decoding_finalized_ above., and ComputeFinalCosts().
  unordered_map<Token*, BaseFloat> final_costs_;
  BaseFloat final_relative_cost_;
  BaseFloat final_best_cost_;  
};


} // end namespace kaldi.


#endif
