// decoder/lattice-faster-decoder.cc

// Copyright 2009-2012  Microsoft Corporation  Mirko Hannemann
//           2013-2018  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen

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

#include "decoder/lattice-faster-decoder.h"
#include "lat/lattice-functions.h"

namespace kaldi {
int niter = 0;
// instantiate this class once for each thing you have to decode.
template <typename FST, typename Token>
LatticeFasterDecoderTpl<FST, Token>::LatticeFasterDecoderTpl(
    const FST &fst, const LatticeFasterDecoderConfig &config)
    : fst_(&fst),
      delete_fst_(false),
      config_(config),
      num_toks_(0),
      token_pool_(config.memory_pool_tokens_block_size),
      forward_link_pool_(config.memory_pool_links_block_size) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}

template <typename FST, typename Token>
LatticeFasterDecoderTpl<FST, Token>::LatticeFasterDecoderTpl(
    const LatticeFasterDecoderConfig &config, FST *fst)
    : fst_(fst),
      delete_fst_(true),
      config_(config),
      num_toks_(0),
      token_pool_(config.memory_pool_tokens_block_size),
      forward_link_pool_(config.memory_pool_links_block_size) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}

template <typename FST, typename Token>
LatticeFasterDecoderTpl<FST, Token>::~LatticeFasterDecoderTpl() {
  DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (delete_fst_) delete fst_;
}

template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::InitDecoding() {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  StateId start_state = fst_->Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  active_toks_.resize(1);
  Token *start_tok =
      new (token_pool_.Allocate()) Token(0.0, 0.0, NULL, NULL, NULL);
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;
  ProcessNonemitting(config_.beam);
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template <typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::Decode(DecodableInterface *decodable) {
  InitDecoding();
  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  AdvanceDecoding(decodable);
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}


// Outputs an FST corresponding to the single best path through the lattice.
template <typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::GetBestPath(Lattice *olat,
                                       bool use_final_probs) const {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  return (olat->NumStates() != 0);
}


// Outputs an FST corresponding to the raw, state-level lattice
template <typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::GetRawLattice(
    Lattice *ofst,
    bool use_final_probs) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  const int32 bucket_count = num_toks_/2 + 3;
  unordered_map<Token*, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token*> token_list;
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  KALDI_VLOG(4) << "init:" << num_toks_/2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLinkT *l = tok->links;
           l != NULL;
           l = l->next) {
        typename unordered_map<Token*, StateId>::const_iterator
            iter = tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          typename unordered_map<Token*, BaseFloat>::const_iterator
              iter = final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  return (ofst->NumStates() > 0);
}


// This function is now deprecated, since now we do determinization from outside
// the LatticeFasterDecoder class.  Outputs an FST corresponding to the
// lattice-determinized lattice (one path per word sequence).
template <typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::GetLattice(CompactLattice *ofst,
                                           bool use_final_probs) const {
  Lattice raw_fst;
  GetRawLattice(&raw_fst, use_final_probs);
  Invert(&raw_fst);  // make it so word labels are on the input.
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp);  // sort on ilabel; makes
  // lattice-determinization more efficient.

  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = config_.det_opts.max_mem;

  DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates();  // Free memory-- raw_fst no longer needed.
  Connect(ofst);  // Remove unreachable states... there might be
  // a small number of these, in some cases.
  // Note: if something went wrong and the raw lattice was empty,
  // we should still get to this point in the code without warnings or failures.
  return (ofst->NumStates() != 0);
}

template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

/*
  A note on the definition of extra_cost.

  extra_cost is used in pruning tokens, to save memory.

  extra_cost can be thought of as a beta (backward) cost assuming
  we had set the betas on currently-active tokens to all be the negative
  of the alphas for those tokens.  (So all currently active tokens would
  be on (tied) best paths).

  We can use the extra_cost to accurately prune away tokens that we know will
  never appear in the lattice.  If the extra_cost is greater than the desired
  lattice beam, the token would provably never appear in the lattice, so we can
  prune away the token.

  (Note: we don't update all the extra_costs every time we update a frame; we
  only do it every 'config_.prune_interval' frames).
 */

// FindOrAddToken either locates a token in hash of toks_,
// or if necessary inserts a new, empty token (i.e. with no forward links)
// for the current frame.  [note: it's inserted if necessary into hash toks_
// and also into the singly linked list of tokens active on this frame
// (whose head is at active_toks_[frame]).
template <typename FST, typename Token>
inline typename LatticeFasterDecoderTpl<FST, Token>::Elem*
LatticeFasterDecoderTpl<FST, Token>::FindOrAddToken(
      StateId state, int32 frame_plus_one, BaseFloat tot_cost,
      Token *backpointer, bool *changed, StateId bias_lm_state) {
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  Elem *e_found = toks_.Insert(state, NULL);
  if (e_found->val == NULL) {  // no such token presently.
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    Token *new_tok = new (token_pool_.Allocate())
        Token(tot_cost, extra_cost, NULL, toks, backpointer);
    // NULL: no forward links yet
    new_tok->bias_lm_state = bias_lm_state;
    toks = new_tok;
    num_toks_++;
    e_found->val = new_tok;
    if (changed) *changed = true;
    return e_found;
  } else {
    Token *tok = e_found->val;  // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) {  // replace old token
      tok->bias_lm_state = bias_lm_state;
      tok->tot_cost = tot_cost;
      // SetBackpointer() just does tok->backpointer = backpointer in
      // the case where Token == BackpointerToken, else nothing.
      tok->SetBackpointer(backpointer);
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
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) {  // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);  // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost);  // check for NaN
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          forward_link_pool_.Free(link);
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            //if (link_extra_cost < -0.01)
            //  KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL)  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef typename unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;
  // We call DeleteElems() as a nicety, not because it's really necessary;
  // otherwise there would be a time, after calling PruneTokensForFrame() on the
  // final frame, when toks_.GetList() or toks_.Clear() would contain pointers
  // to nonexistent tokens.
  DeleteElems(toks_.Clear());

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        IterType iter = final_costs_.find(tok);
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      BaseFloat tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          forward_link_pool_.Free(link);
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            //if (link_extra_cost < -0.01)
            //  KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
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
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}

template <typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}


// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneTokensForFrame(int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      token_pool_.Free(tok);
      num_toks_--;
    } else {  // fetch next Token
      prev_tok = tok;
    }
  }
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneActiveTokens(BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f+1 < cur_frame_plus_one &&      // except for last f (no forward links)
        active_toks_[f+1].must_prune_tokens) {
      PruneTokensForFrame(f+1);
      active_toks_[f+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ComputeFinalCosts(
    unordered_map<Token*, BaseFloat> *final_costs,
    BaseFloat *final_relative_cost,
    BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
  if (final_costs != NULL)
    final_costs->clear();
  const Elem *final_toks = toks_.GetList();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity,
      best_cost_with_final = infinity;

  while (final_toks != NULL) {
    StateId state = final_toks->key;
    Token *tok = final_toks->val;
    const Elem *next = final_toks->tail;
    BaseFloat final_cost = fst_->Final(state).Value();
    BaseFloat cost = tok->tot_cost,
        cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
    final_toks = next;
  }
  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}

template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::AdvanceDecoding(DecodableInterface *decodable,
                                                int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc> >::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    }
  }


  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

/// Gets the weight cutoff.  Also counts the active tokens.
template <typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::GetCutoff(Elem *list_head, size_t *tok_count,
                                          BaseFloat *adaptive_beam, Elem **best_elem) {
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
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

    BaseFloat beam_cutoff = best_weight + config_.beam,
        min_active_cutoff = std::numeric_limits<BaseFloat>::infinity(),
        max_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

    KALDI_VLOG(6) << "Number of tokens active on frame " << NumFramesDecoded()
                  << " is " << tmp_array_.size();

    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_weight;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_weight + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

template <typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::ProcessEmitting(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 frame = active_toks_.size() - 1; // frame is the frame-index
                                         // (zero-based) used to get likelihoods
                                         // from the decodable object.
  active_toks_.resize(active_toks_.size() + 1);

  Elem *final_toks = toks_.Clear(); // analogous to swapping prev_toks_ / cur_toks_
                                   // in simple-decoder.h.   Removes the Elems from
                                   // being indexed in the hash in toks_.
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  size_t tok_cnt;
  BaseFloat cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem);
  KALDI_VLOG(6) << "Adaptive beam on frame " << NumFramesDecoded() << " is "
                << adaptive_beam;

  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.

  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  BaseFloat cost_offset = 0.0; // Used to keep probabilities in a good
                               // dynamic range.


  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.  The only
  // products of the next block are "next_cutoff" and "cost_offset".
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    cost_offset = - tok->tot_cost;
    for (fst::ArcIterator<FST> aiter(*fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate..
        if (arc.nextstate == state) continue;
        BaseFloat new_weight = arc.weight.Value() + cost_offset -
            decodable->LogLikelihood(frame, arc.ilabel) + tok->tot_cost;

        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  cost_offsets_.resize(frame + 1, 0.0);
  cost_offsets_[frame] = cost_offset;

  // the tokens are now owned here, in final_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call DeleteElem
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = final_toks, *e_tail; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;

    if (tok->tot_cost <= cur_cutoff) {
      for (fst::ArcIterator<FST> aiter(*fst_, state);
           !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          if (arc.nextstate == state) continue;
          StateId new_bias_state = 0;
          BaseFloat ac_cost = cost_offset -
              decodable->LogLikelihood(frame, arc.ilabel),
              graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;


          if (bias_lm_) {
            float bias_lm_score = 0.0f;
            int frm = frame;
            int stat = state;
            int ilab = arc.ilabel-1;
            if (arc.ilabel - 1 == 0) {
              new_bias_state = tok->bias_lm_state;
            } else {
              bias_lm_score = bias_lm_->BiasLmScore(tok->bias_lm_state, 
                                                    arc.ilabel - 1, new_bias_state);
            }
            graph_cost -= bias_lm_score;
            tot_cost -= bias_lm_score;
          }

          if (tot_cost >= next_cutoff) {
            continue;
          } else if (tot_cost + adaptive_beam < next_cutoff) {
            next_cutoff = tot_cost + adaptive_beam; // prune by best current token
          }
          // Note: the frame indexes into active_toks_ are one-based,
          // hence the + 1.
          Elem *e_next = FindOrAddToken(arc.nextstate,
                                        frame + 1, tot_cost, tok, NULL, new_bias_state);
          // NULL: no change indicator needed

          // Add ForwardLink from tok to next_tok (put on head of list tok->links)
          tok->links = new (forward_link_pool_.Allocate())
              ForwardLinkT(e_next->val, arc.ilabel, arc.olabel, graph_cost,
                           ac_cost, tok->links);
        }
      } // for all arcs
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
  return next_cutoff;
}

template <typename FST, typename Token>
std::string LatticeFasterDecoderTpl<FST, Token>::GetTokResult(Token *tok) {
  if (!tok) { return ""; }
  std::string res;
  ForwardLinkT* link;
  std::vector<int> phn_id;
  tok->GetLabelSeq(tok, phn_id);
  for (int i = 0; i < phn_id.size(); i++) {
    res = bias_lm_->GetPhoneLabel(phn_id[i]) + res;
  }
  return res;
}

// static inline
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::DeleteForwardLinks(Token *tok) {
  ForwardLinkT *l = tok->links, *m;
  while (l != NULL) {
    m = l->next;
    forward_link_pool_.Free(l);
    l = m;
  }
  tok->links = NULL;
}

template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.

  KALDI_ASSERT(queue_.empty());

  if (toks_.GetList() == NULL) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens: frame is " << frame;
      warned_ = true;
    }
  }

  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
    StateId state = e->key;
    if (fst_->NumInputEpsilons(state) != 0)
      queue_.push_back(e);
  }

  while (!queue_.empty()) {
    const Elem *e = queue_.back();
    queue_.pop_back();

    StateId state = e->key;
    Token *tok = e->val;  // would segfault if e is a NULL pointer but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    if (cur_cost >= cutoff) // Don't bother processing successors.
      continue;
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    DeleteForwardLinks(tok); // necessary when re-visiting
    tok->links = NULL;
    for (fst::ArcIterator<FST> aiter(*fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        BaseFloat graph_cost = arc.weight.Value(),
            tot_cost = cur_cost + graph_cost;
        if (tot_cost < cutoff) {
          bool changed;

          Elem *e_new = FindOrAddToken(arc.nextstate, frame + 1, tot_cost,
                                          tok, &changed);
          if (bias_lm_ && changed) {
            e_new->val->bias_lm_state = tok->bias_lm_state;
          }
          
          tok->links = new (forward_link_pool_.Allocate()) ForwardLinkT(
              e_new->val, 0, arc.olabel, graph_cost, 0, tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed && fst_->NumInputEpsilons(arc.nextstate) != 0)
            queue_.push_back(e_new);
        }
      }
    } // for all arcs
  } // while queue not empty
}


template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::DeleteElems(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
      DeleteForwardLinks(tok);
      Token *next_tok = tok->next;
      token_pool_.Free(tok);
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}

// static
template <typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::TopSortTokens(
    Token *tok_list, std::vector<Token*> *topsorted_list) {
  unordered_map<Token*, int32> token2pos;
  typedef typename unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token*> reprocess;

  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
    Token *tok = iter->first;
    int32 pos = iter->second;
    for (ForwardLinkT *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        IterType following_iter = token2pos.find(link->next_tok);
        if (following_iter != token2pos.end()) { // another token on this frame,
                                                 // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the next Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_tok);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (typename unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (typename std::vector<Token*>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLinkT *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_tok);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_tok);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}

// Instantiate the template for the combination of token types and FST types
// that we'll need.
template class LatticeFasterDecoderTpl<fst::Fst<fst::StdArc>, decoder::StdToken>;
template class LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, decoder::StdToken >;
template class LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, decoder::StdToken >;

//template class LatticeFasterDecoderTpl<fst::ConstGrammarFst, decoder::StdToken>;
//template class LatticeFasterDecoderTpl<fst::VectorGrammarFst, decoder::StdToken>;

template class LatticeFasterDecoderTpl<fst::Fst<fst::StdArc> , decoder::BackpointerToken>;
template class LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, decoder::BackpointerToken >;
template class LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, decoder::BackpointerToken >;
//template class LatticeFasterDecoderTpl<fst::ConstGrammarFst, decoder::BackpointerToken>;
//template class LatticeFasterDecoderTpl<fst::VectorGrammarFst, decoder::BackpointerToken>;


} // end namespace kaldi.
