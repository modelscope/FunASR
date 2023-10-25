// decoder/biglm-faster-decoder.h

// Copyright 2009-2011 Microsoft Corporation,  Gilles Boulianne

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

#ifndef KALDI_DECODER_BIGLM_FASTER_DECODER_H_
#define KALDI_DECODER_BIGLM_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc
#include "decoder/faster-decoder.h" // for options class
#include "fstext/deterministic-fst.h"

namespace kaldi {

struct BiglmFasterDecoderOptions: public FasterDecoderOptions {
  BiglmFasterDecoderOptions() {
    min_active = 200;
  }
};

/** This is as FasterDecoder, but does online composition between
    HCLG and the "difference language model", which is a deterministic
    FST that represents the difference between the language model you want
    and the language model you compiled HCLG with.  The class
    DeterministicOnDemandFst follows through the epsilons in G for you
    (assuming G is a standard backoff language model) and makes it look
    like a determinized FST.  Actually, in practice,
    DeterministicOnDemandFst operates in a mode where it composes two
    G's together; one has negated likelihoods and works by removing the
    LM probabilities that you made HCLG with, and one is the language model
    you want to use.
*/
class BiglmFasterDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  // A PairId will be constructed as: (StateId in fst) + (StateId in lm_diff_fst) << 32;
  typedef uint64 PairId;
  typedef Arc::Weight Weight;
  
  // This constructor is the same as for FasterDecoder, except the second
  // argument (lm_diff_fst) is new; it's an FST (actually, a
  // DeterministicOnDemandFst) that represents the difference in LM scores
  // between the LM we want and the LM the decoding-graph "fst" was built with.
  // See e.g. gmm-decode-biglm-faster.cc for an example of how this is called.
  // Basically, we are using fst o lm_diff_fst (where o is composition)
  // as the decoding graph.  Instead of having everything indexed by the state in
  // "fst", we now index by the pair of states in (fst, lm_diff_fst).
  // Whenever we cross a word, we need to propagate the state within
  // lm_diff_fst.
  BiglmFasterDecoder(const fst::Fst<fst::StdArc> &fst,
                     const BiglmFasterDecoderOptions &opts,
                     fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst):
      fst_(fst), lm_diff_fst_(lm_diff_fst), opts_(opts), warned_noarc_(false) {
    KALDI_ASSERT(opts_.hash_ratio >= 1.0);  // less doesn't make much sense.
    KALDI_ASSERT(opts_.max_active > 1);
    KALDI_ASSERT(fst.Start() != fst::kNoStateId &&
                 lm_diff_fst->Start() != fst::kNoStateId);
    toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  }
  
  void SetOptions(const BiglmFasterDecoderOptions &opts) { opts_ = opts; }

  ~BiglmFasterDecoder() {
    ClearToks(toks_.Clear());
  }

  void Decode(DecodableInterface *decodable) {
    // clean up from last time:
    ClearToks(toks_.Clear());
    PairId start_pair = ConstructPair(fst_.Start(), lm_diff_fst_->Start());
    Arc dummy_arc(0, 0, Weight::One(), fst_.Start()); // actually, the last element of
    // the Arcs (fst_.Start(), here) is never needed.
    toks_.Insert(start_pair, new Token(dummy_arc, NULL));
    ProcessNonemitting(std::numeric_limits<float>::max());
    for (int32 frame = 0; !decodable->IsLastFrame(frame-1); frame++) {
      BaseFloat weight_cutoff = ProcessEmitting(decodable, frame);
      ProcessNonemitting(weight_cutoff);
    }
  }

  bool ReachedFinal() {
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      PairId state_pair = e->key;
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      Weight this_weight =
          Times(e->val->weight_,
                Times(fst_.Final(state), lm_diff_fst_->Final(lm_state)));
      if (this_weight != Weight::Zero())
        return true;
    }
    return false;
  }

  bool GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                   bool use_final_probs = true) {
    // GetBestPath gets the decoding output.  If "use_final_probs" is true
    // AND we reached a final state, it limits itself to final states;
    // otherwise it gets the most likely token not taking into
    // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
    // nothing was available.  It returns true if it got output (thus, fst_out
    // will be nonempty).
    fst_out->DeleteStates();
    Token *best_tok = NULL;
    Weight best_final = Weight::Zero(); // set only if is_final == true.  The
    // final-prob corresponding to the best final token (i.e. the one with best
    // weight best_weight, below).
    bool is_final = ReachedFinal();
    if (!is_final) {
      for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
        if (best_tok == NULL || *best_tok < *(e->val) )
          best_tok = e->val;
    } else {
      Weight best_weight = Weight::Zero();
      for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
        Weight fst_final = fst_.Final(PairToState(e->key)),
            lm_final = lm_diff_fst_->Final(PairToLmState(e->key)),
            final = Times(fst_final, lm_final);
        Weight this_weight = Times(e->val->weight_, final);
        if (this_weight != Weight::Zero() &&
           this_weight.Value() < best_weight.Value()) {
          best_weight = this_weight;
          best_final = final;
          best_tok = e->val;
        }
      }
    }
    if (best_tok == NULL) return false;  // No output.

    std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

    for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
      BaseFloat tot_cost = tok->weight_.Value() -
          (tok->prev_ ? tok->prev_->weight_.Value() : 0.0),
          graph_cost = tok->arc_.weight.Value(),
          ac_cost = tot_cost - graph_cost;
      LatticeArc l_arc(tok->arc_.ilabel,
                       tok->arc_.olabel,
                       LatticeWeight(graph_cost, ac_cost),
                       tok->arc_.nextstate);
      arcs_reverse.push_back(l_arc);
    }
    KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.
    
    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);
    for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
      LatticeArc arc = arcs_reverse[i];
      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
    if (is_final && use_final_probs) {
      fst_out->SetFinal(cur_state, LatticeWeight(best_final.Value(), 0.0));
    } else {
      fst_out->SetFinal(cur_state, LatticeWeight::One());
    }
    RemoveEpsLocal(fst_out);
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

  class Token {
   public:
    Arc arc_; // contains only the graph part of the cost,
    // including the part in "fst" (== HCLG) plus lm_diff_fst.
    // We can work out the acoustic part from difference between
    // "weight_" and prev->weight_.
    Token *prev_;
    int32 ref_count_;
    Weight weight_; // weight up to current point.
    inline Token(const Arc &arc, Weight &ac_weight, Token *prev):
        arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        weight_ = Times(Times(prev->weight_, arc.weight), ac_weight);
      } else {
        weight_ = Times(arc.weight, ac_weight);
      }
    }
    inline Token(const Arc &arc, Token *prev):
        arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        weight_ = Times(prev->weight_, arc.weight);
      } else {
        weight_ = arc.weight;
      }
    }
    inline bool operator < (const Token &other) {
      return weight_.Value() > other.weight_.Value();
      // This makes sense for log + tropical semiring.
    }

    inline ~Token() {
      KALDI_ASSERT(ref_count_ == 1);
      if (prev_ != NULL) TokenDelete(prev_);
    }
    inline static void TokenDelete(Token *tok) {
      if (tok->ref_count_ == 1) { 
        delete tok;
      } else {
        tok->ref_count_--;
      }
    }
  };
  typedef HashList<PairId, Token*>::Elem Elem;


  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem) {
    BaseFloat best_weight = 1.0e+10;  // positive == high cost == bad.
    size_t count = 0;
    if (opts_.max_active == std::numeric_limits<int32>::max() &&
        opts_.min_active == 0) {
      for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
        BaseFloat w = static_cast<BaseFloat>(e->val->weight_.Value());
        if (w < best_weight) {
          best_weight = w;
          if (best_elem) *best_elem = e;
        }
      }
      if (tok_count != NULL) *tok_count = count;
      if (adaptive_beam != NULL) *adaptive_beam = opts_.beam;
      return best_weight + opts_.beam;
    } else {
      tmp_array_.clear();
      for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
        BaseFloat w = e->val->weight_.Value();
        tmp_array_.push_back(w);
        if (w < best_weight) {
          best_weight = w;
          if (best_elem) *best_elem = e;
        }
      }
      if (tok_count != NULL) *tok_count = count;

      BaseFloat beam_cutoff = best_weight + opts_.beam,
        min_active_cutoff = std::numeric_limits<BaseFloat>::infinity(),
        max_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

      if (tmp_array_.size() > static_cast<size_t>(opts_.max_active)) {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + opts_.max_active,
                         tmp_array_.end());
        max_active_cutoff = tmp_array_[opts_.max_active];
      }
      if (tmp_array_.size() > static_cast<size_t>(opts_.min_active)) {
        if (opts_.min_active == 0) min_active_cutoff = best_weight;
        else {
          std::nth_element(tmp_array_.begin(),
                           tmp_array_.begin() + opts_.min_active,
                           tmp_array_.size() > static_cast<size_t>(opts_.max_active) ?
                           tmp_array_.begin() + opts_.max_active :
                           tmp_array_.end());
          min_active_cutoff = tmp_array_[opts_.min_active];
        }
      }

      if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
        if (adaptive_beam)
          *adaptive_beam = max_active_cutoff - best_weight + opts_.beam_delta;
        return max_active_cutoff;
      } else if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
        if (adaptive_beam)
          *adaptive_beam = min_active_cutoff - best_weight + opts_.beam_delta;
        return min_active_cutoff;
      } else {
        *adaptive_beam = opts_.beam;
        return beam_cutoff;
      }
    }
  }

  void PossiblyResizeHash(size_t num_toks) {
    size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                        * opts_.hash_ratio);
    if (new_sz > toks_.Size()) {
      toks_.SetSize(new_sz);
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

  // ProcessEmitting returns the likelihood cutoff used.
  BaseFloat ProcessEmitting(DecodableInterface *decodable, int frame) {
    Elem *last_toks = toks_.Clear();
    size_t tok_cnt;
    BaseFloat adaptive_beam;
    Elem *best_elem = NULL;
    BaseFloat weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                        &adaptive_beam, &best_elem);
    PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.
    
    // This is the cutoff we use after adding in the log-likes (i.e.
    // for the next frame).  This is a bound on the cutoff we will use
    // on the next frame.
    BaseFloat next_weight_cutoff = 1.0e+10;

    // First process the best token to get a hopefully
    // reasonably tight bound on the next cutoff.
    if (best_elem) {
      PairId state_pair = best_elem->key;
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      Token *tok = best_elem->val;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();        
        if (arc.ilabel != 0) {  // we'd propagate..
          PropagateLm(lm_state, &arc); // may affect "arc.weight".
          // We don't need the return value (the new LM state).
          BaseFloat ac_cost = - decodable->LogLikelihood(frame, arc.ilabel),
              new_weight = arc.weight.Value() + tok->weight_.Value() + ac_cost;
          if (new_weight + adaptive_beam < next_weight_cutoff)
            next_weight_cutoff = new_weight + adaptive_beam;
        }
      }
    }

    // the tokens are now owned here, in last_toks, and the hash is empty.
    // 'owned' is a complex thing here; the point is we need to call toks_.Delete(e)
    // on each elem 'e' to let toks_ know we're done with them.
    for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {  // loop this way
      // because we delete "e" as we go.
      PairId state_pair = e->key;
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      Token *tok = e->val;
      if (tok->weight_.Value() < weight_cutoff) {  // not pruned.
        KALDI_ASSERT(state == tok->arc_.nextstate);
        for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
            !aiter.Done();
            aiter.Next()) {
          Arc arc = aiter.Value();
          if (arc.ilabel != 0) {  // propagate.
            StateId next_lm_state = PropagateLm(lm_state, &arc);
            Weight ac_weight(-decodable->LogLikelihood(frame, arc.ilabel));
            BaseFloat new_weight = arc.weight.Value() + tok->weight_.Value()
                + ac_weight.Value();
            if (new_weight < next_weight_cutoff) {  // not pruned..
              PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
              Token *new_tok = new Token(arc, ac_weight, tok);
              Elem *e_found = toks_.Insert(next_pair, new_tok);
              if (new_weight + adaptive_beam < next_weight_cutoff)
                next_weight_cutoff = new_weight + adaptive_beam;
              if (e_found->val != new_tok) {
                if (*(e_found->val) < *new_tok) {
                  Token::TokenDelete(e_found->val);
                  e_found->val = new_tok;
                } else {
                  Token::TokenDelete(new_tok);
                }
              }
            }
          }
        }
      }
      e_tail = e->tail;
      Token::TokenDelete(e->val);
      toks_.Delete(e);
    }
    return next_weight_cutoff;
  }

  // TODO: first time we go through this, could avoid using the queue.
  void ProcessNonemitting(BaseFloat cutoff) {
    // Processes nonemitting arcs for one frame. 
    KALDI_ASSERT(queue_.empty());
    for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
      queue_.push_back(e);
    while (!queue_.empty()) {
      const Elem *e = queue_.back();
      queue_.pop_back();
      PairId state_pair = e->key;
      Token *tok = e->val;  // would segfault if state not
      // in toks_ but this can't happen.
      if (tok->weight_.Value() > cutoff) { // Don't bother processing successors.
        continue;
      }
      KALDI_ASSERT(tok != NULL);
      StateId state = PairToState(state_pair),
          lm_state = PairToLmState(state_pair);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        const Arc &arc_ref = aiter.Value();
        if (arc_ref.ilabel == 0) {  // propagate nonemitting only...
          Arc arc(arc_ref);
          StateId next_lm_state = PropagateLm(lm_state, &arc);
          PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
          Token *new_tok = new Token(arc, tok);
          if (new_tok->weight_.Value() > cutoff) {  // prune
            Token::TokenDelete(new_tok);
          } else {
            Elem *e_found = toks_.Insert(next_pair, new_tok);
            if (e_found->val == new_tok) {
              queue_.push_back(e_found);
            } else {
              if ( *(e_found->val) < *new_tok ) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
                queue_.push_back(e_found);
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
  }

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by PairId.
  HashList<PairId, Token*> toks_;
  const fst::Fst<fst::StdArc> &fst_;
  fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst_;
  BiglmFasterDecoderOptions opts_;
  bool warned_noarc_;
  std::vector<const Elem* > queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks(Elem *list) {
    for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
      Token::TokenDelete(e->val);
      e_tail = e->tail;
      toks_.Delete(e);
    }
  }
  KALDI_DISALLOW_COPY_AND_ASSIGN(BiglmFasterDecoder);
};


} // end namespace kaldi.


#endif
