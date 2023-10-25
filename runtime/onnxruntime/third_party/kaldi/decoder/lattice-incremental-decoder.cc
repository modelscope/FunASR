// decoder/lattice-incremental-decoder.cc

// Copyright      2019  Zhehuai Chen,  Daniel Povey

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

#include "decoder/lattice-incremental-decoder.h"
#include "lat/lattice-functions.h"
#include "base/timer.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
template <typename FST, typename Token>
LatticeIncrementalDecoderTpl<FST, Token>::LatticeIncrementalDecoderTpl(
    const FST &fst, const TransitionInformation &trans_model,
    const LatticeIncrementalDecoderConfig &config)
    : fst_(&fst),
      delete_fst_(false),
      num_toks_(0),
      config_(config),
      determinizer_(trans_model, config) {
  config.Check();
  toks_.SetSize(1000); // just so on the first frame we do something reasonable.
}

template <typename FST, typename Token>
LatticeIncrementalDecoderTpl<FST, Token>::LatticeIncrementalDecoderTpl(
    const LatticeIncrementalDecoderConfig &config, FST *fst,
    const TransitionInformation &trans_model)
    : fst_(fst),
      delete_fst_(true),
      num_toks_(0),
      config_(config),
      determinizer_(trans_model, config) {
  config.Check();
  toks_.SetSize(1000); // just so on the first frame we do something reasonable.
}

template <typename FST, typename Token>
LatticeIncrementalDecoderTpl<FST, Token>::~LatticeIncrementalDecoderTpl() {
  DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (delete_fst_) delete fst_;
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::InitDecoding() {
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
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, NULL);
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;

  determinizer_.Init();
  num_frames_in_lattice_ = 0;
  token2label_map_.clear();
  next_token_label_ = LatticeIncrementalDeterminizer::kTokenLabelOffset;
  ProcessNonemitting(config_.beam);
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::UpdateLatticeDeterminization() {
  if (NumFramesDecoded() - num_frames_in_lattice_ <
      config_.determinize_max_delay)
    return;


  /* Make sure the token-pruning is active.  Note: PruneActiveTokens() has
     internal logic that prevents it from doing unnecessary work if you
     call it and then immediately call it again. */
  PruneActiveTokens(config_.lattice_beam * config_.prune_scale);

  int32 first = num_frames_in_lattice_ + config_.determinize_min_chunk_size,
      last = NumFramesDecoded(),
      fewest_tokens = std::numeric_limits<int32>::max(),
      best_frame = -1;
  for (int32 t = last; t >= first; t--) {
    /* Make sure PruneActiveTokens() has computed num_toks for all these
       frames... */
    KALDI_ASSERT(active_toks_[t].num_toks != -1);
    if (active_toks_[t].num_toks < fewest_tokens) {
      //  <= because we want the latest one in case of ties.
      fewest_tokens = active_toks_[t].num_toks;
      best_frame = t;
    }
  }
  /* Skip this update if we have too many tokens, determinization will take too long,
     postpone it to the next update */
  if (fewest_tokens > config_.determinize_max_active)
      return;

  /* OK, determinize the chunk that spans from num_frames_in_lattice_ to
     best_frame. */
  bool use_final_probs = false;
  GetLattice(best_frame, use_final_probs);
  return;
}
// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template <typename FST, typename Token>
bool LatticeIncrementalDecoderTpl<FST, Token>::Decode(DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.

  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    UpdateLatticeDeterminization();

    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
  Timer timer;
  FinalizeDecoding();
  bool use_final_probs = true;
  GetLattice(NumFramesDecoded(), use_final_probs);
  KALDI_VLOG(2) << "Delay time during and after FinalizeDecoding()"
                << "(secs): " << timer.Elapsed();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}


template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz =
      static_cast<size_t>(static_cast<BaseFloat>(num_toks) * config_.hash_ratio);
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


  Define the 'forward cost' of a token as zero for any token on the frame
  we're currently decoding; and for other frames, as the shortest-path cost
  between that token and a token on the frame we're currently decoding.
  (by "currently decoding" I mean the most recently processed frame).

  Then define the extra_cost of a token (always >= 0) as the forward-cost of
  the token minus the smallest forward-cost of any token on the same frame.

  We can use the extra_cost to accurately prune away tokens that we know will
  never appear in the lattice.  If the extra_cost is greater than the desired
  lattice beam, the token would provably never appear in the lattice, so we can
  prune away the token.

  The advantage of storing the extra_cost rather than the forward-cost, is that
  it is less costly to keep the extra_cost up-to-date when we process new frames.
  When we process a new frame, *all* the previous frames' forward-costs would change;
  but in general the extra_cost will change only for a finite number of frames.
  (Actually we don't update all the extra_costs every time we update a frame; we
  only do it every 'config_.prune_interval' frames).
 */

// FindOrAddToken either locates a token in hash of toks_,
// or if necessary inserts a new, empty token (i.e. with no forward links)
// for the current frame.  [note: it's inserted if necessary into hash toks_
// and also into the singly linked list of tokens active on this frame
// (whose head is at active_toks_[frame]).
template <typename FST, typename Token>
inline Token *LatticeIncrementalDecoderTpl<FST, Token>::FindOrAddToken(
    StateId state, int32 frame_plus_one, BaseFloat tot_cost, Token *backpointer,
    bool *changed) {
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  Elem *e_found = toks_.Find(state);
  if (e_found == NULL) { // no such token presently.
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    Token *new_tok = new Token(tot_cost, extra_cost, NULL, toks, backpointer);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    toks_.Insert(state, new_tok);
    if (changed) *changed = true;
    return new_tok;
  } else {
    Token *tok = e_found->val;      // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) { // replace old token
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
    return tok;
  }
}

// prunes outgoing links for all tokens in active_toks_[frame]
// it's called by PruneActiveTokens
// all links, that have link_extra_cost > lattice_beam are pruned
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed, bool *links_pruned,
    BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) { // empty list; should not happen.
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
    for (Token *tok = active_toks_[frame_plus_one].toks; tok != NULL;
         tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL;) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost =
            next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost) -
             next_tok->tot_cost); // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost); // check for NaN
        if (link_extra_cost > config_.lattice_beam) {     // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL)
            prev_link->next = next_link;
          else
            tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
          *links_pruned = true;
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) tok_extra_cost = link_extra_cost;
          prev_link = link; // move to next link
          link = link->next;
        }
      } // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true; // difference new minus old is bigger than delta
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
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL) // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef typename unordered_map<Token *, BaseFloat>::const_iterator IterType;
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
    for (Token *tok = active_toks_[frame_plus_one].toks; tok != NULL;
         tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this
      // token,
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
      for (link = tok->links; link != NULL;) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost =
            next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost) -
             next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL)
            prev_link->next = next_link;
          else
            tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else {            // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) tok_extra_cost = link_extra_cost;
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

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta)) changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}

template <typename FST, typename Token>
BaseFloat LatticeIncrementalDecoderTpl<FST, Token>::FinalRelativeCost() const {
  BaseFloat relative_cost;
  ComputeFinalCosts(NULL, &relative_cost, NULL);
  return relative_cost;
}

// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneTokensForFrame(
    int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL) KALDI_WARN << "No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  int32 num_toks = 0;
  for (tok = toks; tok != NULL; tok = next_tok, num_toks++) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL)
        prev_tok->next = tok->next;
      else
        toks = tok->next;
      delete tok;
      num_toks_--;
    } else { // fetch next Token
      prev_tok = tok;
    }
  }
  active_toks_[frame_plus_one].num_toks = num_toks;
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneActiveTokens(BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;

  if (active_toks_[cur_frame_plus_one].num_toks == -1){
    // The current frame's tokens don't get pruned so they don't get counted
    // (the count is needed by the incremental determinization code).
    // Fix this.
    int this_frame_num_toks = 0;
    for (Token *t = active_toks_[cur_frame_plus_one].toks; t != NULL; t = t->next)
      this_frame_num_toks++;
    active_toks_[cur_frame_plus_one].num_toks = this_frame_num_toks;
 }

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
        active_toks_[f - 1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f + 1 < cur_frame_plus_one && // except for last f (no forward links)
        active_toks_[f + 1].must_prune_tokens) {
      PruneTokensForFrame(f + 1);
      active_toks_[f + 1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::ComputeFinalCosts(
    unordered_map<Token *, BaseFloat> *final_costs, BaseFloat *final_relative_cost,
    BaseFloat *final_best_cost) const {
  if (decoding_finalized_) {
    // If we finalized decoding, the list toks_ will no longer exist, so return
    // something we already computed.
    if (final_costs) *final_costs = final_costs_;
    if (final_relative_cost) *final_relative_cost = final_relative_cost_;
    if (final_best_cost) *final_best_cost = final_best_cost_;
    return;
  }
  if (final_costs != NULL) final_costs->clear();
  const Elem *final_toks = toks_.GetList();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity, best_cost_with_final = infinity;

  while (final_toks != NULL) {
    StateId state = final_toks->key;
    Token *tok = final_toks->val;
    const Elem *next = final_toks->tail;
    BaseFloat final_cost = fst_->Final(state).Value();
    BaseFloat cost = tok->tot_cost, cost_with_final = cost + final_cost;
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
void LatticeIncrementalDecoderTpl<FST, Token>::AdvanceDecoding(
    DecodableInterface *decodable, int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc> >::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<
              LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>, Token> *>(
              this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<
              LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>, Token> *>(
              this);
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
    target_frames_decoded =
        std::min(target_frames_decoded, NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
  UpdateLatticeDeterminization();
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes the final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2;              // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin << " to " << num_toks_;
}

/// Gets the weight cutoff.  Also counts the active tokens.
template <typename FST, typename Token>
BaseFloat LatticeIncrementalDecoderTpl<FST, Token>::GetCutoff(
    Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam, Elem **best_elem) {
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
      std::nth_element(tmp_array_.begin(), tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0)
        min_active_cutoff = best_weight;
      else {
        std::nth_element(tmp_array_.begin(), tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active)
                             ? tmp_array_.begin() + config_.max_active
                             : tmp_array_.end());
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
BaseFloat LatticeIncrementalDecoderTpl<FST, Token>::ProcessEmitting(
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

  PossiblyResizeHash(tok_cnt); // This makes sure the hash is always big enough.

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
    cost_offset = -tok->tot_cost;
    for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) { // propagate..
        BaseFloat new_weight = arc.weight.Value() + cost_offset -
                               decodable->LogLikelihood(frame, arc.ilabel) +
                               tok->tot_cost;
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
      for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) { // propagate..
          BaseFloat ac_cost =
                        cost_offset - decodable->LogLikelihood(frame, arc.ilabel),
                    graph_cost = arc.weight.Value(), cur_cost = tok->tot_cost,
                    tot_cost = cur_cost + ac_cost + graph_cost;
          if (tot_cost >= next_cutoff)
            continue;
          else if (tot_cost + adaptive_beam < next_cutoff)
            next_cutoff = tot_cost + adaptive_beam; // prune by best current token
          // Note: the frame indexes into active_toks_ are one-based,
          // hence the + 1.
          Token *next_tok =
              FindOrAddToken(arc.nextstate, frame + 1, tot_cost, tok, NULL);
          // NULL: no change indicator needed

          // Add ForwardLink from tok to next_tok (put on head of list tok->links)
          tok->links = new ForwardLinkT(next_tok, arc.ilabel, arc.olabel, graph_cost,
                                        ac_cost, tok->links);
        }
      } // for all arcs
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
  return next_cutoff;
}

// static inline
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::DeleteForwardLinks(Token *tok) {
  ForwardLinkT *l = tok->links, *m;
  while (l != NULL) {
    m = l->next;
    delete l;
    l = m;
  }
  tok->links = NULL;
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
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

  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    StateId state = e->key;
    if (fst_->NumInputEpsilons(state) != 0) queue_.push_back(state);
  }

  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();

    Token *tok =
        toks_.Find(state)
            ->val; // would segfault if state not in toks_ but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    if (cur_cost >= cutoff) // Don't bother processing successors.
      continue;
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    DeleteForwardLinks(tok); // necessary when re-visiting
    tok->links = NULL;
    for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) { // propagate nonemitting only...
        BaseFloat graph_cost = arc.weight.Value(), tot_cost = cur_cost + graph_cost;
        if (tot_cost < cutoff) {
          bool changed;

          Token *new_tok =
              FindOrAddToken(arc.nextstate, frame + 1, tot_cost, tok, &changed);

          tok->links =
              new ForwardLinkT(new_tok, 0, arc.olabel, graph_cost, 0, tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed && fst_->NumInputEpsilons(arc.nextstate) != 0)
            queue_.push_back(arc.nextstate);
        }
      }
    } // for all arcs
  }   // while queue not empty
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::DeleteElems(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<
    FST, Token>::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL;) {
      DeleteForwardLinks(tok);
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}


template <typename FST, typename Token>
const CompactLattice& LatticeIncrementalDecoderTpl<FST, Token>::GetLattice(
    int32 num_frames_to_include,
    bool use_final_probs) {
  KALDI_ASSERT(num_frames_to_include >= num_frames_in_lattice_ &&
               num_frames_to_include <= NumFramesDecoded());


  if (num_frames_in_lattice_ > 0 &&
      determinizer_.GetLattice().NumStates() == 0) {
    /* Something went wrong, lattice is empty and will continue to be empty.
       User-level code should detect and deal with this.
     */
    num_frames_in_lattice_ = num_frames_to_include;
    return determinizer_.GetLattice();
  }

  if (decoding_finalized_ && !use_final_probs) {
    // This is not supported
    KALDI_ERR << "You cannot get the lattice without final-probs after "
        "calling FinalizeDecoding().";
  }
  if (use_final_probs && num_frames_to_include != NumFramesDecoded()) {
    /* This is because we only remember the relation between HCLG states and
       Tokens for the current frame; the Token does not have a `state` field. */
    KALDI_ERR << "use-final-probs may no be true if you are not "
        "getting a lattice for all frames decoded so far.";
  }


  if (num_frames_to_include > num_frames_in_lattice_) {
    /* Make sure the token-pruning is up to date.   If we just pruned the tokens,
       this will do very little work. */
    PruneActiveTokens(config_.lattice_beam * config_.prune_scale);

    if (determinizer_.GetLattice().NumStates() == 0 ||
        determinizer_.GetLattice().Final(0) != CompactLatticeWeight::Zero()) {
      num_frames_in_lattice_ = 0;
      determinizer_.Init();
    }

    Lattice chunk_lat;

    unordered_map<Label, LatticeArc::StateId> token_label2state;
    if (num_frames_in_lattice_ != 0) {
      determinizer_.InitializeRawLatticeChunk(&chunk_lat,
                                              &token_label2state);
    }

    // tok_map will map from Token* to state-id in chunk_lat.
    // The cur and prev versions alternate on different frames.
    unordered_map<Token*, StateId> &tok2state_map(temp_token_map_);
    tok2state_map.clear();

    unordered_map<Token*, Label> &next_token2label_map(token2label_map_temp_);
    next_token2label_map.clear();

    { // Deal with the last frame in the chunk, the one numbered `num_frames_to_include`.
      // (Yes, this is backwards).   We allocate token labels, and set tokens as
      // final, but don't add any transitions.  This may leave some states
      // disconnected (e.g. due to chains of nonemitting arcs), but it's OK; we'll
      // fix it when we generate the next chunk of lattice.
      int32 frame = num_frames_to_include;
      // Allocate state-ids for all tokens on this frame.

      for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
        /* If we included the final-costs at this stage, they will cause
           non-final states to be pruned out from the end of the lattice. */
        BaseFloat final_cost;
        {  // This block computes final_cost
          if (decoding_finalized_) {
            if (final_costs_.empty()) {
              final_cost = 0.0;  /* No final-state survived, so treat all as final
                                  * with probability One(). */
            } else {
              auto iter = final_costs_.find(tok);
              if (iter == final_costs_.end())
                final_cost = std::numeric_limits<BaseFloat>::infinity();
              else
                final_cost = iter->second;
            }
          } else {
            /* this is a `fake` final-cost used to guide pruning.  It's as if we
               set the betas (backward-probs) on the final frame to the
               negatives of the corresponding alphas, so all tokens on the last
               frae will be on a best path..  the extra_cost for each token
               always corresponds to its alpha+beta on this assumption.  We want
               the final_cost here to correspond to the beta (backward-prob), so
               we get that by final_cost = extra_cost - tot_cost.
               [The tot_cost is the forward/alpha cost.]
            */
            final_cost = tok->extra_cost - tok->tot_cost;
          }
        }

        StateId state = chunk_lat.AddState();
        tok2state_map[tok] = state;
        if (final_cost < std::numeric_limits<BaseFloat>::infinity()) {
          next_token2label_map[tok] = AllocateNewTokenLabel();
          StateId token_final_state = chunk_lat.AddState();
          LatticeArc::Label ilabel = 0,
              olabel = (next_token2label_map[tok] = AllocateNewTokenLabel());
          chunk_lat.AddArc(state,
                           LatticeArc(ilabel, olabel,
                                      LatticeWeight::One(),
                                      token_final_state));
          chunk_lat.SetFinal(token_final_state, LatticeWeight(final_cost, 0.0));
        }
      }
    }

    // Go in reverse order over the remaining frames so we can create arcs as we
    // go, and their destination-states will already be in the map.
    for (int32 frame = num_frames_to_include;
         frame >= num_frames_in_lattice_; frame--) {
      // The conditional below is needed for the last frame of the utterance.
      BaseFloat cost_offset = (frame < cost_offsets_.size() ?
                               cost_offsets_[frame] : 0.0);

      // For the first frame of the chunk, we need to make sure the states are
      // the ones created by InitializeRawLatticeChunk() (where not pruned away).
      if (frame == num_frames_in_lattice_ && num_frames_in_lattice_ != 0) {
        for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
          auto iter = token2label_map_.find(tok);
          KALDI_ASSERT(iter != token2label_map_.end());
          Label token_label = iter->second;
          auto iter2 = token_label2state.find(token_label);
          if (iter2 != token_label2state.end()) {
            StateId state = iter2->second;
            tok2state_map[tok] = state;
          } else {
            // Some states may have been pruned out, but we should still allocate
            // them.  They might have been part of chains of nonemitting arcs
            // where the state became disconnected because the last chunk didn't
            // include arcs starting at this frame.
            StateId state = chunk_lat.AddState();
            tok2state_map[tok] = state;
          }
        }
      } else if (frame != num_frames_to_include) {  // We already created states
                                                    // for the last frame.
        for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
          StateId state = chunk_lat.AddState();
          tok2state_map[tok] = state;
        }
      }
      for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
        auto iter = tok2state_map.find(tok);
        KALDI_ASSERT(iter != tok2state_map.end());
        StateId cur_state = iter->second;
        for (ForwardLinkT *l = tok->links; l != NULL; l = l->next) {
          auto next_iter = tok2state_map.find(l->next_tok);
          if (next_iter == tok2state_map.end()) {
            // Emitting arcs from the last frame we're including -- ignore
            // these.
            KALDI_ASSERT(frame == num_frames_to_include);
            continue;
          }
          StateId next_state = next_iter->second;
          BaseFloat this_offset = (l->ilabel != 0 ? cost_offset : 0);
          LatticeArc arc(l->ilabel, l->olabel,
                         LatticeWeight(l->graph_cost, l->acoustic_cost - this_offset),
                         next_state);
          // Note: the epsilons get redundantly included at the end and beginning
          // of successive chunks.  These will get removed in the determinization.
          chunk_lat.AddArc(cur_state, arc);
        }
      }
    }
    if (num_frames_in_lattice_ == 0) {
      // This block locates the start token.  NOTE: we use the fact that in the
      // linked list of tokens, things are added at the head, so the start state
      // must be at the tail.  If this data structure is changed in future, we
      // might need to explicitly store the start token as a class member.
      Token *tok = active_toks_[0].toks;
      if (tok == NULL) {
        KALDI_WARN << "No tokens exist on start frame";
        return determinizer_.GetLattice();  // will be empty.
      }
      while (tok->next != NULL)
        tok = tok->next;
      Token *start_token = tok;
      auto iter = tok2state_map.find(start_token);
      KALDI_ASSERT(iter != tok2state_map.end());
      StateId start_state = iter->second;
      chunk_lat.SetStart(start_state);
    }
    token2label_map_.swap(next_token2label_map);

    // bool finished_before_beam =
    determinizer_.AcceptRawLatticeChunk(&chunk_lat);
    // We are ignoring the return status, which say whether it finished before the beam.

    num_frames_in_lattice_ = num_frames_to_include;

    if (determinizer_.GetLattice().NumStates() == 0)
      return determinizer_.GetLattice();   // Something went wrong, lattice is empty.
  }

  unordered_map<Token*, BaseFloat> token2final_cost;
  unordered_map<Label, BaseFloat> token_label2final_cost;
  if (use_final_probs) {
    ComputeFinalCosts(&token2final_cost, NULL, NULL);
    for (const auto &p: token2final_cost) {
      Token *tok = p.first;
      BaseFloat cost = p.second;
      auto iter = token2label_map_.find(tok);
      if (iter != token2label_map_.end()) {
        /* Some tokens may not have survived the pruned determinization. */
        Label token_label = iter->second;
        bool ret = token_label2final_cost.insert({token_label, cost}).second;
        KALDI_ASSERT(ret); /* Make sure it was inserted. */
      }
    }
  }
  /* Note: these final-probs won't affect the next chunk, only the lattice
     returned from GetLattice().  They are kind of temporaries. */
  determinizer_.SetFinalCosts(token_label2final_cost.empty() ? NULL :
                              &token_label2final_cost);

  return determinizer_.GetLattice();
}


template <typename FST, typename Token>
int32 LatticeIncrementalDecoderTpl<FST, Token>::GetNumToksForFrame(int32 frame) {
  int32 r = 0;
  for (Token *tok = active_toks_[frame].toks; tok; tok = tok->next) r++;
  return r;
}



/* This utility function adds an arc to a Lattice, but where the source is a
   CompactLatticeArc.  If the CompactLatticeArc has a string with length greater
   than 1, this will require adding extra states to `lat`.
 */
static void AddCompactLatticeArcToLattice(
    const CompactLatticeArc &clat_arc,
    LatticeArc::StateId src_state,
    Lattice *lat) {
  const std::vector<int32> &string = clat_arc.weight.String();
  size_t N = string.size();
  if (N == 0) {
    LatticeArc arc;
    arc.ilabel = 0;
    arc.olabel = clat_arc.ilabel;
    arc.nextstate = clat_arc.nextstate;
    arc.weight = clat_arc.weight.Weight();
    lat->AddArc(src_state, arc);
  } else {
    LatticeArc::StateId cur_state = src_state;
    for (size_t i = 0; i < N; i++) {
      LatticeArc arc;
      arc.ilabel = string[i];
      arc.olabel = (i == 0 ? clat_arc.ilabel : 0);
      arc.nextstate = (i + 1 == N ? clat_arc.nextstate : lat->AddState());
      arc.weight = (i == 0 ? clat_arc.weight.Weight() : LatticeWeight::One());
      lat->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
  }
}


void LatticeIncrementalDeterminizer::Init() {
  non_final_redet_states_.clear();
  clat_.DeleteStates();
  final_arcs_.clear();
  forward_costs_.clear();
  arcs_in_.clear();
}

CompactLattice::StateId LatticeIncrementalDeterminizer::AddStateToClat() {
  CompactLattice::StateId ans = clat_.AddState();
  forward_costs_.push_back(std::numeric_limits<BaseFloat>::infinity());
  KALDI_ASSERT(forward_costs_.size() == ans + 1);
  arcs_in_.resize(ans + 1);
  return ans;
}

void LatticeIncrementalDeterminizer::AddArcToClat(
    CompactLattice::StateId state,
    const CompactLatticeArc &arc) {
  BaseFloat forward_cost = forward_costs_[state] +
      ConvertToCost(arc.weight);
  if (forward_cost == std::numeric_limits<BaseFloat>::infinity())
    return;
  int32 arc_idx = clat_.NumArcs(state);
  clat_.AddArc(state, arc);
  arcs_in_[arc.nextstate].push_back({state, arc_idx});
  if (forward_cost < forward_costs_[arc.nextstate])
    forward_costs_[arc.nextstate] = forward_cost;
}

// See documentation in header
void LatticeIncrementalDeterminizer::IdentifyTokenFinalStates(
    const CompactLattice &chunk_clat,
    std::unordered_map<CompactLattice::StateId, CompactLatticeArc::Label> *token_map) const {
  token_map->clear();
  using StateId = CompactLattice::StateId;
  using Label = CompactLatticeArc::Label;

  StateId num_states = chunk_clat.NumStates();
  for (StateId state = 0; state < num_states; state++) {
    for (fst::ArcIterator<CompactLattice> aiter(chunk_clat, state);
       !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      if (arc.olabel >= kTokenLabelOffset && arc.olabel < kMaxTokenLabel) {
        StateId nextstate = arc.nextstate;
        auto r = token_map->insert({nextstate, arc.olabel});
        // Check consistency of labels on incoming arcs
        KALDI_ASSERT(r.first->second == arc.olabel);
      }
    }
  }
}




void LatticeIncrementalDeterminizer::GetNonFinalRedetStates() {
  using StateId = CompactLattice::StateId;
  non_final_redet_states_.clear();
  non_final_redet_states_.reserve(final_arcs_.size());

  std::vector<StateId> state_queue;
  for (const CompactLatticeArc &arc: final_arcs_) {
    // Note: we abuse the .nextstate field to store the state which is really
    // the source of that arc.
    StateId redet_state = arc.nextstate;
    if (forward_costs_[redet_state] != std::numeric_limits<BaseFloat>::infinity()) {
      // if it is accessible..
      if (non_final_redet_states_.insert(redet_state).second) {
        // it was not already there
        state_queue.push_back(redet_state);
      }
    }
  }
  // Add any states that are reachable from the states above.
  while (!state_queue.empty()) {
    StateId s = state_queue.back();
    state_queue.pop_back();
    for (fst::ArcIterator<CompactLattice> aiter(clat_, s); !aiter.Done();
         aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      StateId nextstate = arc.nextstate;
      if (non_final_redet_states_.insert(nextstate).second)
        state_queue.push_back(nextstate); // it was not already there
    }
  }
}


void LatticeIncrementalDeterminizer::InitializeRawLatticeChunk(
    Lattice *olat,
    unordered_map<Label, LatticeArc::StateId> *token_label2state) {
  using namespace fst;

  olat->DeleteStates();
  LatticeArc::StateId start_state = olat->AddState();
  olat->SetStart(start_state);
  token_label2state->clear();

  // redet_state_map maps from state-ids in clat_ to state-ids in olat.  This
  // will be the set of states from which the arcs to final-states in the
  // canonical appended lattice leave (physically, these are in the .nextstate
  // elements of arcs_, since we use that field for the source state), plus any
  // states reachable from those states.
  unordered_map<CompactLattice::StateId, LatticeArc::StateId> redet_state_map;

  for (CompactLattice::StateId redet_state: non_final_redet_states_)
    redet_state_map[redet_state] = olat->AddState();

  // First, process any arcs leaving the non-final redeterminized states that
  // are not to final-states.  (What we mean by "not to final states" is, not to
  // stats that are final in the `canonical appended lattice`.. they may
  // actually be physically final in clat_, because we make clat_ what we want
  // to return to the user.
  for (CompactLattice::StateId redet_state: non_final_redet_states_) {
    LatticeArc::StateId lat_state = redet_state_map[redet_state];

    for (ArcIterator<CompactLattice> aiter(clat_, redet_state);
         !aiter.Done(); aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      CompactLattice::StateId nextstate = arc.nextstate;
      LatticeArc::StateId lat_nextstate = olat->NumStates();
      auto r = redet_state_map.insert({nextstate, lat_nextstate});
      if (r.second) {  // Was inserted.
        LatticeArc::StateId s = olat->AddState();
        KALDI_ASSERT(s == lat_nextstate);
      } else {
        // was not inserted -> was already there.
        lat_nextstate = r.first->second;
      }
      CompactLatticeArc clat_arc(arc);
      clat_arc.nextstate = lat_nextstate;
      AddCompactLatticeArcToLattice(clat_arc, lat_state, olat);
    }
    clat_.DeleteArcs(redet_state);
    clat_.SetFinal(redet_state, CompactLatticeWeight::Zero());
  }

  for (const CompactLatticeArc &arc: final_arcs_) {
    // We abuse the `nextstate` field to store the source state.
    CompactLattice::StateId src_state = arc.nextstate;
    auto iter = redet_state_map.find(src_state);
    if (forward_costs_[src_state] == std::numeric_limits<BaseFloat>::infinity())
      continue;  /* Unreachable state */
    KALDI_ASSERT(iter != redet_state_map.end());
    LatticeArc::StateId src_lat_state = iter->second;
    Label token_label = arc.ilabel;  // will be == arc.olabel.
    KALDI_ASSERT(token_label >= kTokenLabelOffset &&
                 token_label < kMaxTokenLabel);
    auto r = token_label2state->insert({token_label,
            olat->NumStates()});
    LatticeArc::StateId dest_lat_state = r.first->second;
    if (r.second) { // was inserted
      LatticeArc::StateId new_state = olat->AddState();
      KALDI_ASSERT(new_state == dest_lat_state);
    }
    CompactLatticeArc new_arc;
    new_arc.nextstate = dest_lat_state;
    /*  We convert the token-label to epsilon; it's not needed anymore. */
    new_arc.ilabel = new_arc.olabel = 0;
    new_arc.weight = arc.weight;
    AddCompactLatticeArcToLattice(new_arc, src_lat_state, olat);
  }

  // Now deal with the initial-probs.  Arcs from initial-states to
  // redeterminized-states in the raw lattice have an olabel that identifies the
  // id of that redeterminized-state in clat_, and a cost that is derived from
  // its entry in forward_costs_.  These forward-probs are used to get the
  // pruned lattice determinization to behave correctly, and will be canceled
  // out later on.
  //
  // In the paper this is the second-from-last bullet in Sec. 5.2.  NOTE: in the
  // paper we state that we only include such arcs for "each redeterminized
  // state that is either initial in det(A) or that has an arc entering it from
  // a state that is not a redeterminized state."  In fact, we include these
  // arcs for all redeterminized states.  I realized that it won't make a
  // difference to the outcome, and it's easier to do it this way.
  for (CompactLattice::StateId state_id: non_final_redet_states_) {
    BaseFloat forward_cost = forward_costs_[state_id];
    LatticeArc arc;
    arc.ilabel = 0;
    // The olabel (which appears where the word-id would) is what
    // we call a 'state-label'.  It identifies a state in clat_.
    arc.olabel = state_id + kStateLabelOffset;
    // It doesn't matter what field we put forward_cost in (or whether we
    // divide it among them both; the effect on pruning is the same, and
    // we will cancel it out later anyway.
    arc.weight = LatticeWeight(forward_cost, 0);
    auto iter = redet_state_map.find(state_id);
    KALDI_ASSERT(iter != redet_state_map.end());
    arc.nextstate = iter->second;
    olat->AddArc(start_state, arc);
  }
}

void LatticeIncrementalDeterminizer::GetRawLatticeFinalCosts(
    const Lattice &raw_fst,
    std::unordered_map<Label, BaseFloat> *old_final_costs) {
  LatticeArc::StateId raw_fst_num_states = raw_fst.NumStates();
  for (LatticeArc::StateId s = 0; s < raw_fst_num_states; s++) {
    for (fst::ArcIterator<Lattice> aiter(raw_fst, s); !aiter.Done();
         aiter.Next()) {
      const LatticeArc &value = aiter.Value();
      if (value.olabel >= (Label)kTokenLabelOffset &&
          value.olabel < (Label)kMaxTokenLabel) {
        LatticeWeight final_weight = raw_fst.Final(value.nextstate);
        if (final_weight != LatticeWeight::Zero() &&
            final_weight.Value2() != 0) {
          KALDI_ERR << "Label " << value.olabel << " from state " << s
                    << " looks like a token-label but its next-state "
                    << value.nextstate <<
              " has unexpected final-weight " << final_weight.Value1() << ','
                    << final_weight.Value2();
        }
        auto r = old_final_costs->insert({value.olabel,
                final_weight.Value1()});
        if (!r.second && r.first->second != final_weight.Value1()) {
          // For any given token-label, all arcs in raw_fst with that
          // olabel should go to the same state, so this should be
          // impossible.
          KALDI_ERR << "Unexpected mismatch in final-costs for tokens, "
                    << r.first->second << " vs " << final_weight.Value1();
        }
      }
    }
  }
}


bool LatticeIncrementalDeterminizer::ProcessArcsFromChunkStartState(
    const CompactLattice &chunk_clat,
    std::unordered_map<CompactLattice::StateId, CompactLattice::StateId> *state_map) {
  using StateId = CompactLattice::StateId;
  StateId clat_num_states = clat_.NumStates();

  // Process arcs leaving the start state of chunk_clat.  These arcs will have
  // state-labels on them (unless this is the first chunk).
  // For destination-states of those arcs, work out which states in
  // clat_ they correspond to and update their forward_costs.
  for (fst::ArcIterator<CompactLattice> aiter(chunk_clat, chunk_clat.Start());
       !aiter.Done(); aiter.Next()) {
    const CompactLatticeArc &arc = aiter.Value();
    Label label = arc.ilabel;  // ilabel == olabel; would be the olabel
                               // in a Lattice.
    if (!(label >= kStateLabelOffset &&
          label - kStateLabelOffset < clat_num_states)) {
      // The label was not a state-label.  This should only be possible on the
      // first chunk.
      KALDI_ASSERT(state_map->empty());
      return true;  // this is the first chunk.
    }
    StateId clat_state = label - kStateLabelOffset;
    StateId chunk_state = arc.nextstate;
    auto p = state_map->insert({chunk_state, clat_state});
    StateId dest_clat_state = p.first->second;
    // We deleted all its arcs in InitializeRawLatticeChunk
    KALDI_ASSERT(clat_.NumArcs(clat_state) == 0);
    /*
      In almost all cases, dest_clat_state and clat_state will be the same state;
      but there may be situations where two arcs with different state-labels
      left the start state and entered the same next-state in chunk_clat; and in
      these cases, they will be different.

      We didn't address this issue in the paper (or actually realize it could be
      a problem).  What we do is pick one of the clat_states as the "canonical"
      one, and redirect all incoming transitions of the others to enter the
      "canonical" one.  (Search below for new_in_arc.nextstate =
      dest_clat_state).
     */
    if (clat_state != dest_clat_state) {
      // Check that the start state isn't getting merged with any other state.
      // If this were possible, we'd need to deal with it specially, but it
      // can't be, because to be merged, 2 states must have identical arcs
      // leaving them with identical weights, so we'd need to have another state
      // on frame 0 identical to the start state, which is not possible if the
      // lattice is deterministic and epsilon-free.
      KALDI_ASSERT(clat_state != 0 && dest_clat_state != 0);
    }

    // in_weight is an extra weight that we'll include on arcs entering this
    // state from the previous chunk.  We need to cancel out
    // `forward_costs[clat_state]`, which was included in the corresponding arc
    // in the raw lattice for pruning purposes; and we need to include the
    // weight on the arc from the start-state of `chunk_clat` to this state.
    CompactLatticeWeight extra_weight_in = arc.weight;
    extra_weight_in.SetWeight(
        fst::Times(extra_weight_in.Weight(),
                   LatticeWeight(-forward_costs_[clat_state], 0.0)));

    // We don't allow state 0 to be a redeterminized-state; calling code assures
    // this.  Search for `determinizer_.GetLattice().Final(0) !=
    // CompactLatticeWeight::Zero())` to find that calling code.
    KALDI_ASSERT(clat_state != 0);

    // Note: 0 is the start state of clat_.  This was checked.
    forward_costs_[clat_state] = (clat_state == 0 ? 0 :
                                  std::numeric_limits<BaseFloat>::infinity());
    std::vector<std::pair<StateId, int32> > arcs_in;
    arcs_in.swap(arcs_in_[clat_state]);
    for (auto p: arcs_in) {
      // Note: we'll be doing `continue` below if this input arc came from
      // another redeterminized-state, because we did DeleteArcs() for them in
      // InitializeRawLatticeChunk().  Those arcs will be transferred
      // from chunk_clat later on.
      CompactLattice::StateId src_state = p.first;
      int32 arc_pos = p.second;

      if (arc_pos >= (int32)clat_.NumArcs(src_state))
        continue;
      fst::MutableArcIterator<CompactLattice> aiter(&clat_, src_state);
      aiter.Seek(arc_pos);
      if (aiter.Value().nextstate != clat_state)
        continue;  // This arc record has become invalidated.
      CompactLatticeArc new_in_arc(aiter.Value());
      // In most cases we will have dest_clat_state == clat_state, so the next
      // line won't change the value of .nextstate
      new_in_arc.nextstate = dest_clat_state;
      new_in_arc.weight = fst::Times(new_in_arc.weight, extra_weight_in);
      aiter.SetValue(new_in_arc);

      BaseFloat new_forward_cost = forward_costs_[src_state] +
          ConvertToCost(new_in_arc.weight);
      if (new_forward_cost < forward_costs_[dest_clat_state])
        forward_costs_[dest_clat_state] = new_forward_cost;
      arcs_in_[dest_clat_state].push_back(p);
    }
  }
  return false;  // this is not the first chunk.
}

void LatticeIncrementalDeterminizer::TransferArcsToClat(
    const CompactLattice &chunk_clat,
    bool is_first_chunk,
    const std::unordered_map<CompactLattice::StateId, CompactLattice::StateId> &state_map,
    const std::unordered_map<CompactLattice::StateId, Label> &chunk_state_to_token,
    const std::unordered_map<Label, BaseFloat> &old_final_costs) {
  using StateId = CompactLattice::StateId;
  StateId chunk_num_states = chunk_clat.NumStates();

  // Now transfer arcs from chunk_clat to clat_.
  for (StateId chunk_state = (is_first_chunk ? 0 : 1);
       chunk_state < chunk_num_states; chunk_state++) {
    auto iter = state_map.find(chunk_state);
    if (iter == state_map.end()) {
      KALDI_ASSERT(chunk_state_to_token.count(chunk_state) != 0);
      // Don't process token-final states.  Anyway they have no arcs leaving
      // them.
      continue;
    }
    StateId clat_state = iter->second;

    // We know that this point that `clat_state` is not a token-final state
    // (see glossary for definition) as if it were, we would have done
    // `continue` above.
    //
    // Only in the last chunk of the lattice would be there be a final-prob on
    // states that are not `token-final states`; these final-probs would
    // normally all be Zero() at this point.  So in almost all cases the following
    // call will do nothing.
    clat_.SetFinal(clat_state, chunk_clat.Final(chunk_state));

    // Process arcs leaving this state.
    for (fst::ArcIterator<CompactLattice> aiter(chunk_clat, chunk_state);
         !aiter.Done(); aiter.Next()) {
      CompactLatticeArc arc(aiter.Value());

      auto next_iter = state_map.find(arc.nextstate);
      if (next_iter != state_map.end()) {
        // The normal case (when the .nextstate has a corresponding
        // state in clat_) is very simple.  Just copy the arc over.
        arc.nextstate = next_iter->second;
        KALDI_ASSERT(arc.ilabel < kTokenLabelOffset ||
                     arc.ilabel > kMaxTokenLabel);
        AddArcToClat(clat_state, arc);
      } else {
        // This is the case when the arc is to a `token-final` state (see
        // glossary.)

        // TODO: remove the following slightly excessive assertion?
        KALDI_ASSERT(chunk_clat.Final(arc.nextstate) != CompactLatticeWeight::Zero() &&
                     arc.olabel >= (Label)kTokenLabelOffset &&
                     arc.olabel < (Label)kMaxTokenLabel &&
                     chunk_state_to_token.count(arc.nextstate) != 0 &&
                     old_final_costs.count(arc.olabel) != 0);

        // Include the final-cost of the next state (which should be final)
        // in arc.weight.
        arc.weight = fst::Times(arc.weight,
                                chunk_clat.Final(arc.nextstate));

        auto cost_iter = old_final_costs.find(arc.olabel);
        KALDI_ASSERT(cost_iter != old_final_costs.end());
        BaseFloat old_final_cost = cost_iter->second;

        // `arc` is going to become an element of final_arcs_.  These
        // contain information about transitions from states in clat_ to
        // `token-final` states (i.e. states that have a token-label on the arc
        // to them and that are final in the canonical compact lattice).
        // We subtract the old_final_cost as it was just a temporary cost
        // introduced for pruning purposes.
        arc.weight.SetWeight(fst::Times(arc.weight.Weight(),
                                        LatticeWeight{-old_final_cost, 0.0}));
        // In a slight abuse of the Arc data structure, the nextstate is set to
        // the source state.  The label (ilabel == olabel) indicates the
        // token it is associated with.
        arc.nextstate = clat_state;
        final_arcs_.push_back(arc);
      }
    }
  }

}

bool LatticeIncrementalDeterminizer::AcceptRawLatticeChunk(
    Lattice *raw_fst) {
  using Label = CompactLatticeArc::Label;
  using StateId = CompactLattice::StateId;

  // old_final_costs is a map from a `token-label` (see glossary) to the
  // associated final-prob in a final-state of `raw_fst`, that is associated
  // with that Token.  These are Tokens that were active at the end of the
  // chunk.  The final-probs may arise from beta (backward) costs, introduced
  // for pruning purposes, and/or from final-probs in HCLG.  Those costs will
  // not be included in anything we store permamently in this class; they used
  // only to guide pruned determinization, and we will use `old_final_costs`
  // later to cancel them out.
  std::unordered_map<Label, BaseFloat> old_final_costs;
  GetRawLatticeFinalCosts(*raw_fst, &old_final_costs);

  CompactLattice chunk_clat;
  bool determinized_till_beam = DeterminizeLatticePhonePrunedWrapper(
      trans_model_, raw_fst, config_.lattice_beam, &chunk_clat,
      config_.det_opts);

  TopSortCompactLatticeIfNeeded(&chunk_clat);

  std::unordered_map<StateId, Label> chunk_state_to_token;
  IdentifyTokenFinalStates(chunk_clat,
                           &chunk_state_to_token);

  StateId chunk_num_states = chunk_clat.NumStates();
  if (chunk_num_states == 0) {
    // This will be an error but user-level calling code can detect it from the
    // lattice being empty.
    KALDI_WARN << "Empty lattice, something went wrong.";
    clat_.DeleteStates();
    return false;
  }

  StateId start_state = chunk_clat.Start();  // would be 0.
  KALDI_ASSERT(start_state == 0);

  // Process arcs leaving the start state of chunk_clat. Unless this is the
  // first chunk in the lattice, all arcs leaving the start state of chunk_clat
  // will have `state labels` on them (identifying redeterminized-states in
  // clat_), and will transition to a state in `chunk_clat` that we can identify
  // with that redeterminized-state.

  // state_map maps from (non-initial, non-token-final state s in chunk_clat) to
  // a state in clat_.
  std::unordered_map<StateId, StateId> state_map;


  bool is_first_chunk = ProcessArcsFromChunkStartState(chunk_clat, &state_map);

  // Remove any existing arcs in clat_ that leave redeterminized-states, and
  // make those states non-final.  Below, we'll add arcs leaving those states
  // (and possibly new final-probs.)
  for (StateId clat_state: non_final_redet_states_) {
    clat_.DeleteArcs(clat_state);
    clat_.SetFinal(clat_state, CompactLatticeWeight::Zero());
  }

  // The previous final-arc info is no longer relevant; we'll recreate it below.
  final_arcs_.clear();

  // assume chunk_lat.Start() == 0; we asserted it above.  Allocate state-ids
  // for all remaining states in chunk_clat, except for token-final states.
  for (StateId state = (is_first_chunk ? 0 : 1);
       state < chunk_num_states; state++) {
    if (chunk_state_to_token.count(state) != 0)
      continue;  // these `token-final` states don't get a state allocated.

    StateId new_clat_state = clat_.NumStates();
    if (state_map.insert({state, new_clat_state}).second) {
      // If it was inserted then we need to actually allocate that state
      StateId s = AddStateToClat();
      KALDI_ASSERT(s == new_clat_state);
    }   // else do nothing; it would have been a redeterminized-state and no
  }     // allocation is needed since they already exist in clat_. and
        // in state_map.

  if (is_first_chunk) {
    auto iter = state_map.find(start_state);
    KALDI_ASSERT(iter != state_map.end());
    CompactLattice::StateId clat_start_state = iter->second;
    KALDI_ASSERT(clat_start_state == 0);  // topological order.
    clat_.SetStart(clat_start_state);
    forward_costs_[clat_start_state] = 0.0;
  }

  TransferArcsToClat(chunk_clat, is_first_chunk,
                     state_map, chunk_state_to_token, old_final_costs);

  GetNonFinalRedetStates();

  return determinized_till_beam;
}



void LatticeIncrementalDeterminizer::SetFinalCosts(
    const unordered_map<Label, BaseFloat> *token_label2final_cost) {
  if (final_arcs_.empty()) {
    KALDI_WARN << "SetFinalCosts() called when final_arcs_.empty()... possibly "
        "means you are calling this after Finalize()?  Not allowed: could "
        "indicate a code error.  Or possibly decoding failed somehow.";
  }

  /*
    prefinal states a terminology that does not appear in the paper.  What it
    means is: the set of states that have an arc with a Token-label as the label
    leaving them in the canonical appended lattice.
  */
  std::unordered_set<int32> &prefinal_states(temp_);
  prefinal_states.clear();
  for (const auto &arc: final_arcs_) {
    /* Caution: `state` is actually the state the arc would
       leave from in the canonical appended lattice; we just store
       that in the .nextstate field. */
    CompactLattice::StateId state = arc.nextstate;
    prefinal_states.insert(state);
  }

  for (int32 state: prefinal_states)
    clat_.SetFinal(state, CompactLatticeWeight::Zero());


  for (const CompactLatticeArc &arc: final_arcs_) {
    Label token_label = arc.ilabel;
    /* Note: we store the source state in the .nextstate field. */
    CompactLattice::StateId src_state = arc.nextstate;
    BaseFloat graph_final_cost;
    if (token_label2final_cost == NULL) {
      graph_final_cost = 0.0;
    } else {
      auto iter = token_label2final_cost->find(token_label);
      if (iter == token_label2final_cost->end())
        continue;
      else
        graph_final_cost = iter->second;
    }
    /* It might seem odd to set a final-prob on the src-state of the arc..
       the point is that the symbol on the arc is a token-label, which should not
       appear in the lattice the user sees, so after that token-label is removed
       the arc would just become a final-prob.
    */
    clat_.SetFinal(src_state,
                   fst::Plus(clat_.Final(src_state),
                             fst::Times(arc.weight,
                                        CompactLatticeWeight(
                                            LatticeWeight(graph_final_cost, 0), {}))));
  }
}




// Instantiate the template for the combination of token types and FST types
// that we'll need.
template class LatticeIncrementalDecoderTpl<fst::Fst<fst::StdArc>, decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>,
                                            decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>,
                                            decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::ConstGrammarFst ,
                                            decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::VectorGrammarFst,
                                            decoder::StdToken>;

template class LatticeIncrementalDecoderTpl<fst::Fst<fst::StdArc>,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::ConstGrammarFst,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::VectorGrammarFst,
                                            decoder::BackpointerToken>;

} // end namespace kaldi.
