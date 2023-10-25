// decoder/lattice-faster-online-decoder.cc

// Copyright 2009-2012  Microsoft Corporation  Mirko Hannemann
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2014  IMSL, PKU-HKUST (author: Wei Shi)
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

// see note at the top of lattice-faster-decoder.cc, about how to maintain this
// file in sync with lattice-faster-decoder.cc

#include "decoder/lattice-faster-online-decoder.h"
#include "lat/lattice-functions.h"

namespace kaldi {

template <typename FST>
bool LatticeFasterOnlineDecoderTpl<FST>::TestGetBestPath(
    bool use_final_probs) const {
  Lattice lat1;
  {
    Lattice raw_lat;
    this->GetRawLattice(&raw_lat, use_final_probs);
    ShortestPath(raw_lat, &lat1);
  }
  Lattice lat2;
  GetBestPath(&lat2, use_final_probs);
  BaseFloat delta = 0.1;
  int32 num_paths = 1;
  if (!fst::RandEquivalent(lat1, lat2, num_paths, delta, rand())) {
    KALDI_WARN << "Best-path test failed";
    return false;
  } else {
    return true;
  }
}


// Outputs an FST corresponding to the single best path through the lattice.
template <typename FST>
bool LatticeFasterOnlineDecoderTpl<FST>::GetBestPath(Lattice *olat,
                                                     bool use_final_probs) const {
  olat->DeleteStates();
  BaseFloat final_graph_cost;
  BestPathIterator iter = BestPathEnd(use_final_probs, &final_graph_cost);
  if (iter.Done())
    return false;  // would have printed warning.
  StateId state = olat->AddState();
  olat->SetFinal(state, LatticeWeight(final_graph_cost, 0.0));
  while (!iter.Done()) {
    LatticeArc arc;
    iter = TraceBackBestPath(iter, &arc);
    arc.nextstate = state;
    StateId new_state = olat->AddState();
    olat->AddArc(new_state, arc);
    state = new_state;
  }
  olat->SetStart(state);
  return true;
}

template <typename FST>
typename LatticeFasterOnlineDecoderTpl<FST>::BestPathIterator LatticeFasterOnlineDecoderTpl<FST>::BestPathEnd(
    bool use_final_probs,
    BaseFloat *final_cost_out) const {
  if (this->decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "BestPathEnd() with use_final_probs == false";
  KALDI_ASSERT(this->NumFramesDecoded() > 0 &&
               "You cannot call BestPathEnd if no frames were decoded.");

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (this->decoding_finalized_ ? this->final_costs_ :final_costs_local);
  if (!this->decoding_finalized_ && use_final_probs)
    this->ComputeFinalCosts(&final_costs_local, NULL, NULL);

  // Singly linked list of tokens on last frame (access list through "next"
  // pointer).
  BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_final_cost = 0;
  Token *best_tok = NULL;
  for (Token *tok = this->active_toks_.back().toks;
       tok != NULL; tok = tok->next) {
    BaseFloat cost = tok->tot_cost, final_cost = 0.0;
    if (use_final_probs && !final_costs.empty()) {
      // if we are instructed to use final-probs, and any final tokens were
      // active on final frame, include the final-prob in the cost of the token.
      typename unordered_map<Token*, BaseFloat>::const_iterator
          iter = final_costs.find(tok);
      if (iter != final_costs.end()) {
        final_cost = iter->second;
        cost += final_cost;
      } else {
        cost = std::numeric_limits<BaseFloat>::infinity();
      }
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_tok = tok;
      best_final_cost = final_cost;
    }
  }
  if (best_tok == NULL) {  // this should not happen, and is likely a code error or
    // caused by infinities in likelihoods, but I'm not making
    // it a fatal error for now.
    KALDI_WARN << "No final token found.";
  }
  if (final_cost_out)
    *final_cost_out = best_final_cost;
  return BestPathIterator(best_tok, this->NumFramesDecoded() - 1);
}


template <typename FST>
typename LatticeFasterOnlineDecoderTpl<FST>::BestPathIterator LatticeFasterOnlineDecoderTpl<FST>::TraceBackBestPath(
    BestPathIterator iter, LatticeArc *oarc) const {
  KALDI_ASSERT(!iter.Done() && oarc != NULL);
  Token *tok = static_cast<Token*>(iter.tok);
  int32 cur_t = iter.frame, step_t = 0;
  if (tok->backpointer != NULL) {
    // retrieve the correct forward link(with the best link cost)
    BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
    ForwardLinkT *link;
    for (link = tok->backpointer->links;
         link != NULL; link = link->next) {
      if (link->next_tok == tok) { // this is a link to "tok"
        BaseFloat graph_cost = link->graph_cost, 
                  acoustic_cost = link->acoustic_cost;
        BaseFloat cost = graph_cost + acoustic_cost;
        if (cost < best_cost) {
          oarc->ilabel = link->ilabel;
          oarc->olabel = link->olabel;
          if (link->ilabel != 0) {
            KALDI_ASSERT(static_cast<size_t>(cur_t) < this->cost_offsets_.size());
            acoustic_cost -= this->cost_offsets_[cur_t];
            step_t = -1;
          } else {
            step_t = 0;
          }
          oarc->weight = LatticeWeight(graph_cost, acoustic_cost);
          best_cost = cost;
        }
      }
    }
    if (link == NULL &&
        best_cost == std::numeric_limits<BaseFloat>::infinity()) { // Did not find correct link.
      KALDI_ERR << "Error tracing best-path back (likely "
                << "bug in token-pruning algorithm)";
    }
  } else {
    oarc->ilabel = 0;
    oarc->olabel = 0;
    oarc->weight = LatticeWeight::One(); // zero costs.
  }
  return BestPathIterator(tok->backpointer, cur_t + step_t);
}

template <typename FST>
bool LatticeFasterOnlineDecoderTpl<FST>::GetRawLatticePruned(
    Lattice *ofst,
    bool use_final_probs,
    BaseFloat beam) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (this->decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (this->decoding_finalized_ ? this->final_costs_ : final_costs_local);
  if (!this->decoding_finalized_ && use_final_probs)
    this->ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = this->active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  for (int32 f = 0; f <= num_frames; f++) {
    if (this->active_toks_[f].toks == NULL) {
      KALDI_WARN << "No tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
  }
  unordered_map<Token*, StateId> tok_map;
  std::queue<std::pair<Token*, int32> > tok_queue;
  // First initialize the queue and states.  Put the initial state on the queue;
  // this is the last token in the list active_toks_[0].toks.
  for (Token *tok = this->active_toks_[0].toks;
       tok != NULL; tok = tok->next) {
    if (tok->next == NULL) {
      tok_map[tok] = ofst->AddState();
      ofst->SetStart(tok_map[tok]);
      std::pair<Token*, int32> tok_pair(tok, 0);  // #frame = 0
      tok_queue.push(tok_pair);
    }
  }

  // Next create states for "good" tokens
  while (!tok_queue.empty()) {
    std::pair<Token*, int32> cur_tok_pair = tok_queue.front();
    tok_queue.pop();
    Token *cur_tok = cur_tok_pair.first;
    int32 cur_frame = cur_tok_pair.second;
    KALDI_ASSERT(cur_frame >= 0 &&
                 cur_frame <= this->cost_offsets_.size());

    typename unordered_map<Token*, StateId>::const_iterator iter =
        tok_map.find(cur_tok);
    KALDI_ASSERT(iter != tok_map.end());
    StateId cur_state = iter->second;

    for (ForwardLinkT *l = cur_tok->links;
         l != NULL;
         l = l->next) {
      Token *next_tok = l->next_tok;
      if (next_tok->extra_cost < beam) {
        // so both the current and the next token are good; create the arc
        int32 next_frame = l->ilabel == 0 ? cur_frame : cur_frame + 1;
        StateId nextstate;
        if (tok_map.find(next_tok) == tok_map.end()) {
          nextstate = tok_map[next_tok] = ofst->AddState();
          tok_queue.push(std::pair<Token*, int32>(next_tok, next_frame));
        } else {
          nextstate = tok_map[next_tok];
        }
        BaseFloat cost_offset = (l->ilabel != 0 ?
                                 this->cost_offsets_[cur_frame] : 0);
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
    }
    if (cur_frame == num_frames) {
      if (use_final_probs && !final_costs.empty()) {
        typename unordered_map<Token*, BaseFloat>::const_iterator iter =
            final_costs.find(cur_tok);
        if (iter != final_costs.end())
          ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
      } else {
        ofst->SetFinal(cur_state, LatticeWeight::One());
      }
    }
  }
  return (ofst->NumStates() != 0);
}



// Instantiate the template for the FST types that we'll need.
template class LatticeFasterOnlineDecoderTpl<fst::Fst<fst::StdArc> >;
template class LatticeFasterOnlineDecoderTpl<fst::VectorFst<fst::StdArc> >;
template class LatticeFasterOnlineDecoderTpl<fst::ConstFst<fst::StdArc> >;
//template class LatticeFasterOnlineDecoderTpl<fst::ConstGrammarFst >;
//template class LatticeFasterOnlineDecoderTpl<fst::VectorGrammarFst >;


} // end namespace kaldi.
