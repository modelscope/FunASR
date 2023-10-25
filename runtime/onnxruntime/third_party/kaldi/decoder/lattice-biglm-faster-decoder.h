// decoder/lattice-biglm-faster-decoder.h

// Copyright 2009-2011  Microsoft Corporation, Mirko Hannemann,
//              Gilles Boulianne

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

#ifndef KALDI_DECODER_LATTICE_BIGLM_FASTER_DECODER_H_
#define KALDI_DECODER_LATTICE_BIGLM_FASTER_DECODER_H_


#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "decoder/lattice-faster-decoder.h" // for options.


namespace kaldi {

// The options are the same as for lattice-faster-decoder.h for now.
typedef LatticeFasterDecoderConfig LatticeBiglmFasterDecoderConfig;

/** This is as LatticeFasterDecoder, but does online composition between
    HCLG and the "difference language model", which is a deterministic
    FST that represents the difference between the language model you want
    and the language model you compiled HCLG with.  The class
    DeterministicOnDemandFst follows through the epsilons in G for you
    (assuming G is a standard backoff language model) and makes it look
    like a determinized FST.
*/

class LatticeBiglmFasterDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  // A PairId will be constructed as: (StateId in fst) + (StateId in lm_diff_fst) << 32;
  typedef uint64 PairId;
  typedef Arc::Weight Weight;
  // instantiate this class once for each thing you have to decode.
  LatticeBiglmFasterDecoder(
      const fst::Fst<fst::StdArc> &fst,      
      const LatticeBiglmFasterDecoderConfig &config,
      fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst):
      fst_(fst), lm_diff_fst_(lm_diff_fst), config_(config),
      warned_noarc_(false), num_toks_(0) {
    config.Check();
    KALDI_ASSERT(fst.Start() != fst::kNoStateId &&
                 lm_diff_fst->Start() != fst::kNoStateId);
    toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  }
  void SetOptions(const LatticeBiglmFasterDecoderConfig &config) { config_ = config; } 
  LatticeBiglmFasterDecoderConfig GetOptions() { return config_; } 
  ~LatticeBiglmFasterDecoder() {
    DeleteElems(toks_.Clear());    
    ClearActiveTokens();
  }

  // Returns true if any kind of traceback is available (not necessarily from
  // a final state).
  bool Decode(DecodableInterface *decodable) {
    // clean up from last time:
    DeleteElems(toks_.Clear());
    ClearActiveTokens();
    warned_ = false;
    final_active_ = false;
    final_costs_.clear();
    num_toks_ = 0;
    PairId start_pair = ConstructPair(fst_.Start(), lm_diff_fst_->Start());
    active_toks_.resize(1);
    Token *start_tok = new Token(0.0, 0.0, NULL, NULL);
    active_toks_[0].toks = start_tok;
    toks_.Insert(start_pair, start_tok);
    num_toks_++;
    ProcessNonemitting(0);
    
    // We use 1-based indexing for frames in this decoder (if you view it in
    // terms of features), but note that the decodable object uses zero-based
    // numbering, which we have to correct for when we call it.
    for (int32 frame = 1; !decodable->IsLastFrame(frame-2); frame++) {
      active_toks_.resize(frame+1); // new column

      ProcessEmitting(decodable, frame);
      
      ProcessNonemitting(frame);

      if (decodable->IsLastFrame(frame-1))
        PruneActiveTokensFinal(frame);
      else if (frame % config_.prune_interval == 0)
        PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.        
    }
    // Returns true if we have any kind of traceback available (not necessarily
    // to the end state; query ReachedFinal() for that).
    return !final_costs_.empty();
  }

  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const { return final_active_; }


  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool GetBestPath(fst::MutableFst<LatticeArc> *ofst, 
                   bool use_final_probs = true) const {
    fst::VectorFst<LatticeArc> fst;
    if (!GetRawLattice(&fst, use_final_probs)) return false;
    // std::cout << "Raw lattice is:\n";
    // fst::FstPrinter<LatticeArc> fstprinter(fst, NULL, NULL, NULL, false, true);
    // fstprinter.Print(&std::cout, "standard output");
    ShortestPath(fst, ofst);
    return true;
  }

  // Outputs an FST corresponding to the raw, state-level
  // tracebacks.
  bool GetRawLattice(fst::MutableFst<LatticeArc> *ofst,
                     bool use_final_probs = true) const {
    typedef LatticeArc Arc;
    typedef Arc::StateId StateId;
    // A PairId will be constructed as: (StateId in fst) + (StateId in lm_diff_fst) << 32;
    typedef uint64 PairId;
    typedef Arc::Weight Weight;
    typedef Arc::Label Label;
    ofst->DeleteStates();
    // num-frames plus one (since frames are one-based, and we have
    // an extra frame for the start-state).
    int32 num_frames = active_toks_.size() - 1;
    KALDI_ASSERT(num_frames > 0);
    unordered_map<Token*, StateId> tok_map(num_toks_/2 + 3); // bucket count
    // First create all states.
    for (int32 f = 0; f <= num_frames; f++) {
      if (active_toks_[f].toks == NULL) {
        KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                   << ": not producing lattice.\n";
        return false;
      }
      for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next)
        tok_map[tok] = ofst->AddState();
      // The next statement sets the start state of the output FST.
      // Because we always add new states to the head of the list
      // active_toks_[f].toks, and the start state was the first one
      // added, it will be the last one added to ofst.
      if (f == 0 && ofst->NumStates() > 0)
        ofst->SetStart(ofst->NumStates()-1);
    }
    KALDI_VLOG(3) << "init:" << num_toks_/2 + 3 << " buckets:" 
                  << tok_map.bucket_count() << " load:" << tok_map.load_factor() 
                  << " max:" << tok_map.max_load_factor();
    // Now create all arcs.
    StateId cur_state = 0; // we rely on the fact that we numbered these
    // consecutively (AddState() returns the numbers in order..)
    for (int32 f = 0; f <= num_frames; f++) {
      for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next,
               cur_state++) {
        for (ForwardLink *l = tok->links;
             l != NULL;
             l = l->next) {
          unordered_map<Token*, StateId>::const_iterator iter =
              tok_map.find(l->next_tok);
          StateId nextstate = iter->second;
          KALDI_ASSERT(iter != tok_map.end());
          Arc arc(l->ilabel, l->olabel,
                  Weight(l->graph_cost, l->acoustic_cost),
                  nextstate);
          ofst->AddArc(cur_state, arc);
        }
        if (f == num_frames) {
          if (use_final_probs && !final_costs_.empty()) {
            std::map<Token*, BaseFloat>::const_iterator iter =
                final_costs_.find(tok);
            if (iter != final_costs_.end())
              ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
          } else {
            ofst->SetFinal(cur_state, LatticeWeight::One());
          }
        }
      }
    }
    KALDI_ASSERT(cur_state == ofst->NumStates());
    return (cur_state != 0);
  }

  // This function is now deprecated, since now we do determinization from
  // outside the LatticeBiglmFasterDecoder class.
  // Outputs an FST corresponding to the lattice-determinized
  // lattice (one path per word sequence).
  bool GetLattice(fst::MutableFst<CompactLatticeArc> *ofst,
                  bool use_final_probs = true) const {
    Lattice raw_fst;
    if (!GetRawLattice(&raw_fst, use_final_probs)) return false;
    Invert(&raw_fst); // make it so word labels are on the input.
    if (!TopSort(&raw_fst)) // topological sort makes lattice-determinization more efficient
      KALDI_WARN << "Topological sorting of state-level lattice failed "
          "(probably your lexicon has empty words or your LM has epsilon cycles; this "
          " is a bad idea.)";
    // (in phase where we get backward-costs).
    fst::ILabelCompare<LatticeArc> ilabel_comp;
    ArcSort(&raw_fst, ilabel_comp); // sort on ilabel; makes
    // lattice-determinization more efficient.
    
    fst::DeterminizeLatticePrunedOptions lat_opts;
    lat_opts.max_mem = config_.det_opts.max_mem;
    
    DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
    raw_fst.DeleteStates(); // Free memory-- raw_fst no longer needed.
    Connect(ofst); // Remove unreachable states... there might be
    // a small number of these, in some cases.
    return true;
  }
  
 private:
  inline PairId ConstructPair(StateId fst_state, StateId lm_state) {
    return static_cast<PairId>(fst_state) + (static_cast<PairId>(lm_state) << 32);
  }
  
  static inline StateId PairToState(PairId state_pair) {
    return static_cast<StateId>(static_cast<uint32>(state_pair));
  }
  static inline StateId PairToLmState(PairId state_pair) {
    return static_cast<StateId>(static_cast<uint32>(state_pair >> 32));
  }
  
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
    inline ForwardLink(Token *next_tok, Label ilabel, Label olabel,
                       BaseFloat graph_cost, BaseFloat acoustic_cost, 
                       ForwardLink *next):
        next_tok(next_tok), ilabel(ilabel), olabel(olabel),
        graph_cost(graph_cost), acoustic_cost(acoustic_cost), 
        next(next) { }
  };  
  
  // Token is what's resident in a particular state at a particular time.
  // In this decoder a Token actually contains *forward* links.
  // When first created, a Token just has the (total) cost.    We add forward
  // links to it when we process the next frame.
  struct Token {
    BaseFloat tot_cost; // would equal weight.Value()... cost up to this point.
    BaseFloat extra_cost; // >= 0.  After calling PruneForwardLinks, this equals
    // the minimum difference between the cost of the best path, and the cost of
    // this is on, and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).
    
    ForwardLink *links; // Head of singly linked list of ForwardLinks
    
    Token *next; // Next in list of tokens for this frame.
    
    inline Token(BaseFloat tot_cost, BaseFloat extra_cost, ForwardLink *links,
                 Token *next): tot_cost(tot_cost), extra_cost(extra_cost),
                 links(links), next(next) { }
    inline void DeleteForwardLinks() {
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

  typedef HashList<PairId, Token*>::Elem Elem;
  
  void PossiblyResizeHash(size_t num_toks) {
    size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                        * config_.hash_ratio);
    if (new_sz > toks_.Size()) {
      toks_.SetSize(new_sz);
    }
  }

  // FindOrAddToken either locates a token in hash of toks_,
  // or if necessary inserts a new, empty token (i.e. with no forward links)
  // for the current frame.  [note: it's inserted if necessary into hash toks_
  // and also into the singly linked list of tokens active on this frame
  // (whose head is at active_toks_[frame]).
  inline Elem *FindOrAddToken(PairId state_pair, int32 frame,
      BaseFloat tot_cost, bool emitting, bool *changed) {
    // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
    // if the token was newly created or the cost changed.
    KALDI_ASSERT(frame < active_toks_.size());
    Token *&toks = active_toks_[frame].toks;
    Elem *e_found = toks_.Insert(state_pair, NULL);
    if (e_found->val == NULL) { // no such token presently.
      const BaseFloat extra_cost = 0.0;
      // tokens on the currently final frame have zero extra_cost
      // as any of them could end up
      // on the winning path.
      Token *new_tok = new Token (tot_cost, extra_cost, NULL, toks);
      // NULL: no forward links yet
      toks = new_tok;
      num_toks_++;
      e_found->val = new_tok;
      if (changed) *changed = true;
      return e_found;
    } else {
      Token *tok = e_found->val; // There is an existing Token for this state.
      if (tok->tot_cost > tot_cost) { // replace old token
        tok->tot_cost = tot_cost;
        // we don't allocate a new token, the old stays linked in active_toks_
        // we only replace the tot_cost
        // in the current frame, there are no forward links (and no extra_cost)
        // only in ProcessNonemitting we have to delete forward links
        // in case we visit a state for the second time
        // those forward links, that lead to this replaced token before:
        // they remain and will hopefully be pruned later (PruneForwardLinks...)
        if (changed) *changed = true;
      } else {
        if (changed) *changed = false;
      }
      return e_found;
    }
  }
  
  // prunes outgoing links for all tokens in active_toks_[frame]
  // it's called by PruneActiveTokens
  // all links, that have link_extra_cost > lattice_beam are pruned
  void PruneForwardLinks(int32 frame, bool *extra_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta) {
    // delta is the amount by which the extra_costs must change
    // If delta is larger,  we'll tend to go back less far
    //    toward the beginning of the file.
    // extra_costs_changed is set to true if extra_cost was changed for any token
    // links_pruned is set to true if any link in any token was pruned

    *extra_costs_changed = false;
    *links_pruned = false;
    KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
    if (active_toks_[frame].toks == NULL ) { // empty list; should not happen.
      if (!warned_) {
        KALDI_WARN << "No tokens alive [doing pruning].. warning first "
            "time only for each utterance\n";
        warned_ = true;
      }
    }
    
    // We have to iterate until there is no more change, because the links
    // are not guaranteed to be in topological order.
    bool changed = true; // difference new minus old extra cost >= delta ?
    while (changed) {
      changed = false;
      for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
        ForwardLink *link, *prev_link=NULL;
        // will recompute tok_extra_cost for tok.
        BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
        // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
        for (link = tok->links; link != NULL; ) {
          // See if we need to excise this link...
          Token *next_tok = link->next_tok;
          BaseFloat link_extra_cost = next_tok->extra_cost +
              ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
               - next_tok->tot_cost); // difference in brackets is >= 0
          // link_exta_cost is the difference in score between the best paths
          // through link source state and through link destination state
          KALDI_ASSERT(link_extra_cost == link_extra_cost); // check for NaN
          if (link_extra_cost > config_.lattice_beam) { // excise link
            ForwardLink *next_link = link->next;
            if (prev_link != NULL) prev_link->next = next_link;
            else tok->links = next_link;
            delete link;
            link = next_link; // advance link but leave prev_link the same.
            *links_pruned = true;
          } else { // keep the link and update the tok_extra_cost if needed.
            if (link_extra_cost < 0.0) { // this is just a precaution.
              if (link_extra_cost < -0.01)
                KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
              link_extra_cost = 0.0;
            }
            if (link_extra_cost < tok_extra_cost)
              tok_extra_cost = link_extra_cost;
            prev_link = link; // move to next link
            link = link->next;
          }
        } // for all outgoing links
        if (fabs(tok_extra_cost - tok->extra_cost) > delta)
          changed = true;  // difference new minus old is bigger than delta
        tok->extra_cost = tok_extra_cost;
        // will be +infinity or <= lattice_beam_.
        // infinity indicates, that no forward link survived pruning
      } // for all Token on active_toks_[frame]
      if (changed) *extra_costs_changed = true;

      // Note: it's theoretically possible that aggressive compiler
      // optimizations could cause an infinite loop here for small delta and
      // high-dynamic-range scores.
    } // while changed
  }

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal(int32 frame) {
    KALDI_ASSERT(static_cast<size_t>(frame+1) == active_toks_.size());
    if (active_toks_[frame].toks == NULL ) // empty list; should not happen.
      KALDI_WARN << "No tokens alive at end of file\n";

    // First go through, working out the best token (do it in parallel
    // including final-probs and not including final-probs; we'll take
    // the one with final-probs if it's valid).
    const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
    BaseFloat best_cost_final = infinity,
        best_cost_nofinal = infinity;
    unordered_map<Token*, BaseFloat> tok_to_final_cost;
    Elem *cur_toks = toks_.Clear(); // swapping prev_toks_ / cur_toks_
    for (Elem *e = cur_toks, *e_tail; e != NULL;  e = e_tail) {
      PairId state_pair = e->key;
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      Token *tok = e->val;
      BaseFloat final_cost = fst_.Final(state).Value() +
          lm_diff_fst_->Final(lm_state).Value();
      tok_to_final_cost[tok] = final_cost;
      best_cost_final = std::min(best_cost_final, tok->tot_cost + final_cost);
      best_cost_nofinal = std::min(best_cost_nofinal, tok->tot_cost);
      e_tail = e->tail;
      toks_.Delete(e);
    }
    final_active_ = (best_cost_final != infinity);
    
    // Now go through tokens on this frame, pruning forward links...  may have
    // to iterate a few times until there is no more change, because the list is
    // not in topological order.

    bool changed = true;
    BaseFloat delta = 1.0e-05;
    while (changed) {
      changed = false;
      for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
        ForwardLink *link, *prev_link=NULL;
        // will recompute tok_extra_cost.  It has a term in it that corresponds
        // to the "final-prob", so instead of initializing tok_extra_cost to infinity
        // below we set it to the difference between the (score+final_prob) of this token,
        // and the best such (score+final_prob).
        BaseFloat tok_extra_cost;
        if (final_active_) {
          BaseFloat final_cost = tok_to_final_cost[tok];
          tok_extra_cost = (tok->tot_cost + final_cost) - best_cost_final;
        } else 
          tok_extra_cost = tok->tot_cost - best_cost_nofinal;
      
        for (link = tok->links; link != NULL; ) {
          // See if we need to excise this link...
          Token *next_tok = link->next_tok;
          BaseFloat link_extra_cost = next_tok->extra_cost +
              ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
               - next_tok->tot_cost);
          if (link_extra_cost > config_.lattice_beam) { // excise link
            ForwardLink *next_link = link->next;
            if (prev_link != NULL) prev_link->next = next_link;
            else tok->links = next_link;
            delete link;
            link = next_link; // advance link but leave prev_link the same.
          } else { // keep the link and update the tok_extra_cost if needed.
            if (link_extra_cost < 0.0) { // this is just a precaution.
              if (link_extra_cost < -0.01)
                KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
              link_extra_cost = 0.0;
            }
            if (link_extra_cost < tok_extra_cost)
              tok_extra_cost = link_extra_cost;
            prev_link = link;
            link = link->next;
          }
        }
        // prune away tokens worse than lattice_beam above best path.  This step
        // was not necessary in the non-final case because then, this case
        // showed up as having no forward links.  Here, the tok_extra_cost has
        // an extra component relating to the final-prob.
        if (tok_extra_cost > config_.lattice_beam)
          tok_extra_cost = infinity;
          // to be pruned in PruneTokensForFrame

        if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
          changed = true;
        tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
      }
    } // while changed

    // Now put surviving Tokens in the final_costs_ hash, which is a class
    // member (unlike tok_to_final_costs).
    for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {    
      if (tok->extra_cost != infinity) {
        // If the token was not pruned away, 
        if (final_active_) {
          BaseFloat final_cost = tok_to_final_cost[tok];         
          if (final_cost != infinity)
            final_costs_[tok] = final_cost;
        } else {
          final_costs_[tok] = 0;
        }
      }
    }
  }
  
  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32 frame) {
    KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
    Token *&toks = active_toks_[frame].toks;
    if (toks == NULL)
      KALDI_WARN << "No tokens alive [doing pruning]\n";
    Token *tok, *next_tok, *prev_tok = NULL;
    for (tok = toks; tok != NULL; tok = next_tok) {
      next_tok = tok->next;
      if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
        // token is unreachable from end of graph; (no forward links survived)
        // excise tok from list and delete tok.
        if (prev_tok != NULL) prev_tok->next = tok->next;
        else toks = tok->next;
        delete tok;
        num_toks_--;
      } else { // fetch next Token
        prev_tok = tok;
      }
    }
  }
  
  // Go backwards through still-alive tokens, pruning them.  note: cur_frame is
  // where hash toks_ are (so we do not want to mess with it because these tokens
  // don't yet have forward pointers), but we do all previous frames, unless we
  // know that we can safely ignore them because the frame after them was unchanged.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.
  // for a larger delta, we will recurse less far back
  void PruneActiveTokens(int32 cur_frame, BaseFloat delta) {
    int32 num_toks_begin = num_toks_;
    for (int32 frame = cur_frame-1; frame >= 0; frame--) {
      // Reason why we need to prune forward links in this situation:
      // (1) we have never pruned them (new TokenList)
      // (2) we have not yet pruned the forward links to the next frame,
      // after any of those tokens have changed their extra_cost.
      if (active_toks_[frame].must_prune_forward_links) {
        bool extra_costs_changed = false, links_pruned = false;
        PruneForwardLinks(frame, &extra_costs_changed, &links_pruned, delta);
        if (extra_costs_changed && frame > 0) // any token has changed extra_cost
          active_toks_[frame-1].must_prune_forward_links = true;
        if (links_pruned) // any link was pruned
          active_toks_[frame].must_prune_tokens = true;
        active_toks_[frame].must_prune_forward_links = false; // job done
      }
      if (frame+1 < cur_frame &&      // except for last frame (no forward links)
         active_toks_[frame+1].must_prune_tokens) {
        PruneTokensForFrame(frame+1);
        active_toks_[frame+1].must_prune_tokens = false;
      }
    }
    KALDI_VLOG(3) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                  << " to " << num_toks_;
  }

  // Version of PruneActiveTokens that we call on the final frame.
  // Takes into account the final-prob of tokens.
  void PruneActiveTokensFinal(int32 cur_frame) {
    // returns true if there were final states active
    // else returns false and treats all states as final while doing the pruning
    // (this can be useful if you want partial lattice output,
    // although it can be dangerous, depending what you want the lattices for).
    // final_active_ and final_probs_ (a hash) are set internally
    // by PruneForwardLinksFinal
    int32 num_toks_begin = num_toks_;
    PruneForwardLinksFinal(cur_frame); // prune final frame (with final-probs)
    // sets final_active_ and final_probs_
    for (int32 frame = cur_frame-1; frame >= 0; frame--) {
      bool b1, b2; // values not used.
      BaseFloat dontcare = 0.0; // delta of zero means we must always update
      PruneForwardLinks(frame, &b1, &b2, dontcare);
      PruneTokensForFrame(frame+1);
    }
    PruneTokensForFrame(0); 
    KALDI_VLOG(3) << "PruneActiveTokensFinal: pruned tokens from " << num_toks_begin
                  << " to " << num_toks_;
  }
  
  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem) {
    BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
    // positive == high cost == bad.
    size_t count = 0;
    if (config_.max_active == std::numeric_limits<int32>::max()) {
      for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
        BaseFloat w = static_cast<BaseFloat>(e->val->tot_cost);
        if (w < best_weight) {
          best_weight = w;
          if (best_elem) *best_elem = e;
        }
      }
      if (tok_count != NULL) *tok_count = count;
      if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
      return best_weight + config_.beam;
    } else {
      tmp_array_.clear();
      for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
        BaseFloat w = e->val->tot_cost;
        tmp_array_.push_back(w);
        if (w < best_weight) {
          best_weight = w;
          if (best_elem) *best_elem = e;
        }
      }
      if (tok_count != NULL) *tok_count = count;
      if (tmp_array_.size() <= static_cast<size_t>(config_.max_active)) {
        if (adaptive_beam) *adaptive_beam = config_.beam;
        return best_weight + config_.beam;
      } else {
        // the lowest elements (lowest costs, highest likes)
        // will be put in the left part of tmp_array.
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin()+config_.max_active,
                         tmp_array_.end());
        // return the tighter of the two beams.
        BaseFloat ans = std::min(best_weight + config_.beam,
                                 *(tmp_array_.begin()+config_.max_active));
        if (adaptive_beam)
          *adaptive_beam = std::min(config_.beam,
                                    ans - best_weight + config_.beam_delta);
        return ans;
      }
    }
  }

  inline StateId PropagateLm(StateId lm_state,
                             Arc *arc) { // returns new LM state.
    if (arc->olabel == 0) {
      return lm_state; // no change in LM state if no word crossed.
    } else { // Propagate in the LM-diff FST.
      Arc lm_arc;
      bool ans = lm_diff_fst_->GetArc(lm_state, arc->olabel, &lm_arc);
      if (!ans) { // this case is unexpected for statistical LMs.
        if (!warned_noarc_) {
          warned_noarc_ = true;
          KALDI_WARN << "No arc available in LM (unlikely to be correct "
              "if a statistical language model); will not warn again";
        }
        arc->weight = Weight::Zero();
        return lm_state; // doesn't really matter what we return here; will
        // be pruned.
      } else {
        arc->weight = Times(arc->weight, lm_arc.weight);
        arc->olabel = lm_arc.olabel; // probably will be the same.
        return lm_arc.nextstate; // return the new LM state.
      }      
    }
  }
  
  void ProcessEmitting(DecodableInterface *decodable, int32 frame) {
    // Processes emitting arcs for one frame.  Propagates from prev_toks_ to cur_toks_.
    Elem *last_toks = toks_.Clear(); // swapping prev_toks_ / cur_toks_
    Elem *best_elem = NULL;
    BaseFloat adaptive_beam;
    size_t tok_cnt;
    BaseFloat cur_cutoff = GetCutoff(last_toks, &tok_cnt, &adaptive_beam, &best_elem);
    PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.    
    
    BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
    // pruning "online" before having seen all tokens

    // First process the best token to get a hopefully
    // reasonably tight bound on the next cutoff.
    if (best_elem) {
      PairId state_pair = best_elem->key;
      StateId state = PairToState(state_pair), // state in "fst"
          lm_state = PairToLmState(state_pair);
      Token *tok = best_elem->val;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          PropagateLm(lm_state, &arc); // may affect "arc.weight".
          // We don't need the return value (the new LM state).
          arc.weight = Times(arc.weight,
                             Weight(-decodable->LogLikelihood(frame-1, arc.ilabel)));
          BaseFloat new_weight = arc.weight.Value() + tok->tot_cost;
          if (new_weight + adaptive_beam < next_cutoff)
            next_cutoff = new_weight + adaptive_beam;
        }
      }
    }
    
    // the tokens are now owned here, in last_toks, and the hash is empty.
    // 'owned' is a complex thing here; the point is we need to call DeleteElem
    // on each elem 'e' to let toks_ know we're done with them.
    for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {
      // loop this way because we delete "e" as we go.
      PairId state_pair = e->key;
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      Token *tok = e->val;
      if (tok->tot_cost <=  cur_cutoff) {
        for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
             !aiter.Done();
             aiter.Next()) {
          const Arc &arc_ref = aiter.Value();
          if (arc_ref.ilabel != 0) {  // propagate..
            Arc arc(arc_ref);
            StateId next_lm_state = PropagateLm(lm_state, &arc);
            BaseFloat ac_cost = -decodable->LogLikelihood(frame-1, arc.ilabel),
                graph_cost = arc.weight.Value(),
                cur_cost = tok->tot_cost,
                tot_cost = cur_cost + ac_cost + graph_cost;
            if (tot_cost >= next_cutoff) continue;
            else if (tot_cost + adaptive_beam < next_cutoff)
              next_cutoff = tot_cost + adaptive_beam; // prune by best current token
            PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
            Elem *e_next = FindOrAddToken(next_pair, frame, tot_cost, true, NULL);
            // true: emitting, NULL: no change indicator needed
          
            // Add ForwardLink from tok to next_tok (put on head of list tok->links)
            tok->links = new ForwardLink(e_next->val, arc.ilabel, arc.olabel, 
                                         graph_cost, ac_cost, tok->links);
          }
        } // for all arcs
      }
      e_tail = e->tail;
      toks_.Delete(e); // delete Elem
    }
  }

  void ProcessNonemitting(int32 frame) {
    // note: "frame" is the same as emitting states just processed.
    
    // Processes nonemitting arcs for one frame.  Propagates within toks_.
    // Note-- this queue structure is is not very optimal as
    // it may cause us to process states unnecessarily (e.g. more than once),
    // but in the baseline code, turning this vector into a set to fix this
    // problem did not improve overall speed.

    KALDI_ASSERT(queue_.empty());
    BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
    for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
      queue_.push_back(e);
      // for pruning with current best token
      best_cost = std::min(best_cost, static_cast<BaseFloat>(e->val->tot_cost));
    }
    if (queue_.empty()) {
      if (!warned_) {
        KALDI_ERR << "Error in ProcessNonemitting: no surviving tokens: frame is "
                  << frame;
        warned_ = true;
      }
    }
    BaseFloat cutoff = best_cost + config_.beam;
    
    while (!queue_.empty()) {
      const Elem *e = queue_.back();
      queue_.pop_back();

      PairId state_pair = e->key;
      Token *tok = e->val;  // would segfault if state not in
                            // toks_ but this can't happen.
      BaseFloat cur_cost = tok->tot_cost;
      if (cur_cost >= cutoff) // Don't bother processing successors.
        continue;
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      // If "tok" has any existing forward links, delete them,
      // because we're about to regenerate them.  This is a kind
      // of non-optimality (remember, this is the simple decoder),
      // but since most states are emitting it's not a huge issue.
      tok->DeleteForwardLinks(); // necessary when re-visiting
      tok->links = NULL;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        const Arc &arc_ref = aiter.Value();
        if (arc_ref.ilabel == 0) {  // propagate nonemitting only...
          Arc arc(arc_ref);
          StateId next_lm_state = PropagateLm(lm_state, &arc);          
          BaseFloat graph_cost = arc.weight.Value(),
              tot_cost = cur_cost + graph_cost;
          if (tot_cost < cutoff) {
            bool changed;
            PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
            Elem *e_new = FindOrAddToken(next_pair, frame, tot_cost,
                                         false, &changed); // false: non-emit
            
            tok->links = new ForwardLink(e_new->val, 0, arc.olabel,
                                         graph_cost, 0, tok->links);
            
            // "changed" tells us whether the new token has a different
            // cost from before, or is new [if so, add into queue].
            if (changed) queue_.push_back(e_new);
          }
        }
      } // for all arcs
    } // while queue not empty
  }


  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<PairId, Token*> toks_;
  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<const Elem* > queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.
  const fst::Fst<fst::StdArc> &fst_;
  fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst_;  
  LatticeBiglmFasterDecoderConfig config_;
  bool warned_noarc_;  
  int32 num_toks_; // current total #toks allocated...
  bool warned_;
  bool final_active_; // use this to say whether we found active final tokens
  // on the last frame.
  std::map<Token*, BaseFloat> final_costs_; // A cache of final-costs
  // of tokens on the last frame-- it's just convenient to store it this way.
  
  // It might seem unclear why we call DeleteElems(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void DeleteElems(Elem *list) {
    for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
      e_tail = e->tail;
      toks_.Delete(e);
    }
    toks_.Clear();
  }
  
  void ClearActiveTokens() { // a cleanup routine, at utt end/begin
    for (size_t i = 0; i < active_toks_.size(); i++) {
      // Delete all tokens alive on this frame, and any forward
      // links they may have.
      for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
        tok->DeleteForwardLinks();
        Token *next_tok = tok->next;
        delete tok;
        num_toks_--;
        tok = next_tok;
      }
    }
    active_toks_.clear();
    KALDI_ASSERT(num_toks_ == 0);
  }
};

} // end namespace kaldi.

#endif
