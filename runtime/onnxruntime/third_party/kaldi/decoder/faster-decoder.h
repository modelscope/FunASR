// decoder/faster-decoder.h

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_FASTER_DECODER_H_
#define KALDI_DECODER_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc

namespace kaldi {

struct FasterDecoderOptions {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  FasterDecoderOptions(): beam(16.0),
                          max_active(std::numeric_limits<int32>::max()),
                          min_active(20), // This decoder mostly used for
                                          // alignment, use small default.
                          beam_delta(0.5),
                          hash_ratio(2.0) { }
  void Register(OptionsItf *opts, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active,
                   "Decoder min active states (don't prune if #active less than this).");
    if (full) {
      opts->Register("beam-delta", &beam_delta,
                     "Increment used in decoder [obscure setting]");
      opts->Register("hash-ratio", &hash_ratio,
                     "Setting used in decoder to control hash behavior");
    }
  }
};

class FasterDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                const FasterDecoderOptions &config);

  void SetOptions(const FasterDecoderOptions &config) { config_ = config; }

  ~FasterDecoder() { ClearToks(toks_.Clear()); }

  void Decode(DecodableInterface *decodable);

  /// Returns true if a final state was active on the last frame.
  bool ReachedFinal() const;

  /// GetBestPath gets the decoding traceback. If "use_final_probs" is true
  /// AND we reached a final state, it limits itself to final states;
  /// otherwise it gets the most likely token not taking into account
  /// final-probs. Returns true if the output best path was not the empty
  /// FST (will only return false in unusual circumstances where
  /// no tokens survived).
  bool GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                   bool use_final_probs = true);

  /// As a new alternative to Decode(), you can call InitDecoding
  /// and then (possibly multiple times) AdvanceDecoding().
  void InitDecoding();


  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32 max_num_frames = -1);

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

 protected:

  class Token {
   public:
    Arc arc_; // contains only the graph part of the cost;
    // we can work out the acoustic part from difference between
    // "cost_" and prev->cost_.
    Token *prev_;
    int32 ref_count_;
    // if you are looking for weight_ here, it was removed and now we just have
    // cost_, which corresponds to ConvertToCost(weight_).
    double cost_;
    inline Token(const Arc &arc, BaseFloat ac_cost, Token *prev):
        arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight.Value() + ac_cost;
      } else {
        cost_ = arc.weight.Value() + ac_cost;
      }
    }
    inline Token(const Arc &arc, Token *prev):
        arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight.Value();
      } else {
        cost_ = arc.weight.Value();
      }
    }
    inline bool operator < (const Token &other) {
      return cost_ > other.cost_;
    }

    inline static void TokenDelete(Token *tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == NULL) return;
        else tok = prev;
      }
#ifdef KALDI_PARANOID
      KALDI_ASSERT(tok->ref_count_ > 0);
#endif
    }
  };
  typedef HashList<StateId, Token*>::Elem Elem;


  /// Gets the weight cutoff.  Also counts the active tokens.
  double GetCutoff(Elem *list_head, size_t *tok_count,
                   BaseFloat *adaptive_beam, Elem **best_elem);

  void PossiblyResizeHash(size_t num_toks);

  // ProcessEmitting returns the likelihood cutoff used.
  // It decodes the frame num_frames_decoded_ of the decodable object
  // and then increments num_frames_decoded_
  double ProcessEmitting(DecodableInterface *decodable);

  // TODO: first time we go through this, could avoid using the queue.
  void ProcessNonemitting(double cutoff);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<StateId, Token*> toks_;
  const fst::Fst<fst::StdArc> &fst_;
  FasterDecoderOptions config_;
  std::vector<const Elem* > queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks(Elem *list);

  KALDI_DISALLOW_COPY_AND_ASSIGN(FasterDecoder);
};


} // end namespace kaldi.


#endif
