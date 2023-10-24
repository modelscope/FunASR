// decoder/lattice-incremental-online-decoder.cc

// Copyright      2019  Zhehuai Chen

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

#include "decoder/lattice-incremental-decoder.h"
#include "decoder/lattice-incremental-online-decoder.h"
#include "lat/lattice-functions.h"
#include "base/timer.h"

namespace kaldi {

// Outputs an FST corresponding to the single best path through the lattice.
template <typename FST>
bool LatticeIncrementalOnlineDecoderTpl<FST>::GetBestPath(Lattice *olat,
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
typename LatticeIncrementalOnlineDecoderTpl<FST>::BestPathIterator LatticeIncrementalOnlineDecoderTpl<FST>::BestPathEnd(
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
  if (final_cost_out != NULL)
    *final_cost_out = best_final_cost;
  return BestPathIterator(best_tok, this->NumFramesDecoded() - 1);
}


template <typename FST>
typename LatticeIncrementalOnlineDecoderTpl<FST>::BestPathIterator LatticeIncrementalOnlineDecoderTpl<FST>::TraceBackBestPath(
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
      if (link->next_tok == tok) { // this is the a to "tok"
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

// Instantiate the template for the FST types that we'll need.
template class LatticeIncrementalOnlineDecoderTpl<fst::Fst<fst::StdArc> >;
template class LatticeIncrementalOnlineDecoderTpl<fst::VectorFst<fst::StdArc> >;
template class LatticeIncrementalOnlineDecoderTpl<fst::ConstFst<fst::StdArc> >;
template class LatticeIncrementalOnlineDecoderTpl<fst::ConstGrammarFst >;
template class LatticeIncrementalOnlineDecoderTpl<fst::VectorGrammarFst >;

} // end namespace kaldi.
