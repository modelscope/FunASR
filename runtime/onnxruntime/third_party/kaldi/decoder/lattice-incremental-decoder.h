// decoder/lattice-incremental-decoder.h

// Copyright      2019  Zhehuai Chen, Daniel Povey

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

#ifndef KALDI_DECODER_LATTICE_INCREMENTAL_DECODER_H_
#define KALDI_DECODER_LATTICE_INCREMENTAL_DECODER_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "decoder/grammar-fst.h"
#include "decoder/lattice-faster-decoder.h"

namespace kaldi {
/**
   The normal decoder, lattice-faster-decoder.h, sometimes has an issue when
   doing real-time applications with long utterances, that each time you get the
   lattice the lattice determinization can take a considerable amount of time;
   this introduces latency.  This version of the decoder spreads the work of
   lattice determinization out throughout the decoding process.

   NOTE:

   Please see https://www.danielpovey.com/files/ *TBD* .pdf for a technical
   explanation of what is going on here.

   GLOSSARY OF TERMS:
      chunk: We do the determinization on chunks of frames; these
          may coincide with the chunks on which the user calls
          AdvanceDecoding().  The basic idea is to extract chunks
          of the raw lattice and determinize them individually, but
          it gets much more complicated than that.  The chunks
          should normally be at least as long as a word (let's say,
          at least 20 frames), or the overhead of this algorithm
          might become excessive and affect RTF.

      raw lattice chunk: A chunk of raw (i.e. undeterminized) lattice
          that we will determinize.  In the paper this corresponds
          to the FST B that is described in Section 5.2.

      token_label, state_label / token-label, state-label:

          In the paper these are both referred to as `state labels` (these are
          special, large integer id's that refer to states in the undeterminized
          lattice and in the the determinized lattice); but we use two separate
          terms here, for more clarity, when referring to the undeterminized
          vs. determinized lattice.

           token_label conceptually refers to states in the
           raw lattice, but we don't materialize the entire
           raw lattice as a physical FST and and these tokens
           are actually tokens (template type Token) held by
           the decoder

           state_label when used in this code refers specifically
           to labels that identify states in the determinized
           lattice (i.e. state indexes in lat_).

       token-final state
          A state in a raw lattice or in a determinized chunk that has an arc
          entering it that has a `token-label` on it (as defined above).
          These states will have nonzero final-probs.

       redeterminized-non-splice-state, aka ns_redet:
         A redeterminized state which is not also a splice state;
         refer to the paper for explanation.  In the already-determinized
         part this means a redeterminized state which is not final.

       canonical appended lattice:  This is the appended compact lattice
         that we conceptually have (i.e. what we described in the paper).
         The difference from the "actual appended lattice" stored
         in LatticeIncrementalDeterminizer::clat_ is that the
         actual appended lattice has all its final-arcs replaced with
         final-probs, and we keep the real final-arcs "on the side" in a
         separate data structure.  The final-probs in clat_ aren't
         necessarily related to the costs on the final-arcs; instead
         they can have arbitrary values passed in by the user (e.g.
         if we want to include final-probs).  This means that the
         clat_ can be returned without modification to the user who wants
         a partially determinized result.

       final-arc:  An arc in the canonical appended CompactLattice which
         goes to a final-state.  These arcs will have `state-labels` as
         their labels.

 */
struct LatticeIncrementalDecoderConfig {
  // All the configuration values until det_opts are the same as in
  // LatticeFasterDecoder.  For clarity we repeat them rather than inheriting.
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat prune_scale; // Note: we don't make this configurable on the command line,
                         // it's not a very important parameter.  It affects the
                         // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeIncrementalDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeIncremental.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  // The configuration values from this point on are specific to the
  // incremental determinization.  See where they are registered for
  // explanation.
  // Caution: these are only inspected in UpdateLatticeDeterminization().
  // If you call
  int32 determinize_max_delay;
  int32 determinize_min_chunk_size;
  int32 determinize_max_active;


  LatticeIncrementalDecoderConfig()
      : beam(16.0),
        max_active(std::numeric_limits<int32>::max()),
        min_active(200),
        lattice_beam(10.0),
        prune_interval(25),
        beam_delta(0.5),
        hash_ratio(2.0),
        prune_scale(0.01),
        determinize_max_delay(60),
        determinize_min_chunk_size(20),
        determinize_max_active(200) {
    det_opts.minimize = false;
  }
  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active,
                   "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam,
                   "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("prune-interval", &prune_interval,
                   "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("beam-delta", &beam_delta,
                   "Increment used in decoding-- this "
                   "parameter is obscure and relates to a speedup in the way the "
                   "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio,
                   "Setting used in decoder to "
                   "control hash behavior");
    opts->Register("determinize-max-delay", &determinize_max_delay,
                   "Maximum frames of delay between decoding a frame and "
                   "determinizing it");
    opts->Register("determinize-min-chunk-size", &determinize_min_chunk_size,
                   "Minimum chunk size used in determinization");
    opts->Register("determinize-max-active", &determinize_max_active,
                   "Maximum number of active tokens to update determinization");

  }
  void Check() const {
    if (!(beam > 0.0 && max_active > 1 && lattice_beam > 0.0 &&
          min_active <= max_active && prune_interval > 0 &&
          beam_delta > 0.0 && hash_ratio >= 1.0 &&
          prune_scale > 0.0 && prune_scale < 1.0 &&
          determinize_max_delay > determinize_min_chunk_size &&
          determinize_min_chunk_size > 0 &&
          determinize_max_active >= 0))
        KALDI_ERR << "Invalid options given to decoder";
    /* Minimization of the chunks is not compatible withour algorithm (or at
       least, would require additional complexity to implement.) */
    if (det_opts.minimize || !det_opts.word_determinize)
      KALDI_ERR << "Invalid determinization options given to decoder.";
  }
};



/**
   This class is used inside LatticeIncrementalDecoderTpl; it handles
   some of the details of incremental determinization.
   https://www.danielpovey.com/files/ *TBD*.pdf for the paper.

*/
class LatticeIncrementalDeterminizer {
 public:
  using Label = typename LatticeArc::Label;  /* Actualy the same labels appear
                                                in both lattice and compact
                                                lattice, so we don't use the
                                                specific type all the time but
                                                just say 'Label' */
  LatticeIncrementalDeterminizer(
      const TransitionInformation &trans_model,
      const LatticeIncrementalDecoderConfig &config):
      trans_model_(trans_model), config_(config) { }

  // Resets the lattice determinization data for new utterance
  void Init();

  // Returns the current determinized lattice.
  const CompactLattice &GetDeterminizedLattice() const { return clat_; }

  /**
     Starts the process of creating a raw lattice chunk.  (Search the glossary
     for "raw lattice chunk").  This just sets up the initial states and
     redeterminized-states in the chunk.  Relates to sec. 5.2 in the paper,
     specifically the initial-state i and the redeterminized-states.

     After calling this, the caller would add the remaining arcs and states
     to `olat` and then call AcceptRawLatticeChunk() with the result.

        @param [out] olat    The lattice to be (partially) created

        @param [out] token_label2state  This function outputs to here
                a map from `token-label` to the state we created for
                it in *olat.  See glossary for `token-label`.
                The keys actually correspond to the .nextstate fields
                in the arcs in final_arcs_; values are states in `olat`.
                See the last bullet point before Sec. 5.3 in the paper.
  */
  void InitializeRawLatticeChunk(
      Lattice *olat,
      unordered_map<Label, LatticeArc::StateId> *token_label2state);

  /**
     This function accepts the raw FST (state-level lattice) corresponding to a
     single chunk of the lattice, determinizes it and appends it to this->clat_.
     Unless this was the

     Note: final-probs in `raw_fst` are treated specially: they are used to
     guide the pruned determinization, but when you call GetLattice() it will be
     -- except for pruning effects-- as if all nonzero final-probs in `raw_fst`
     were: One() if final_costs == NULL; else the value present in `final_costs`.

       @param [in] raw_fst  (Consumed destructively).  The input
                  raw (state-level) lattice.  Would correspond to the
                  FST A in the paper if first_frame == 0, and B
                  otherwise.

     @return returns false if determinization finished earlier than the beam
         or the determinized lattice was empty; true otherwise.

     NOTE: if this is not the final chunk, you will probably want to call
     SetFinalCosts() directly after calling this.
  */
  bool AcceptRawLatticeChunk(Lattice *raw_fst);

  /*
    Sets final-probs in `clat_`.  Must only be called if the final chunk
    has not been processed.  (The final chunk is whenever GetLattice() is
    called with finalize == true).

    The reason this is a separate function from AcceptRawLatticeChunk() is that
    there may be situations where a user wants to get the latice with
    final-probs in it, after previously getting it without final-probs; or
    vice versa.  By final-probs, we mean the Final() probabilities in the
    HCLG (decoding graph; this->fst_).

       @param [in] token_label2final_cost   A map from the token-label
              corresponding to Tokens active on the final frame of the
              lattice in the object, to the final-cost we want to use for
              those tokens.  If NULL, it means all Tokens should be treated
              as final with probability One().  If non-NULL, and a particular
              token-label is not a key of this map, it means that Token
              corresponded to a state that was not final in HCLG; and
              such tokens will be treated as non-final.  However,
              if this would result in no states in the lattice being final,
              we will treat all Tokens as final with probability One(),
              a warning will be printed (this should not happen.)
  */
  void SetFinalCosts(const unordered_map<Label, BaseFloat> *token_label2final_cost = NULL);

  const CompactLattice &GetLattice() { return clat_; }

  // kStateLabelOffset is what we add to state-ids in clat_ to produce labels
  // to identify them in the raw lattice chunk
  // kTokenLabelOffset is where we start allocating labels corresponding to Tokens
  // (these correspond with raw lattice states);
  enum  { kStateLabelOffset = (int)1e8,  kTokenLabelOffset = (int)2e8, kMaxTokenLabel = (int)3e8 };

 private:

  // [called from AcceptRawLatticeChunk()]
  // Gets the final costs from token-final states in the raw lattice (see
  // glossary for definition).  These final costs will be subtracted after
  // determinization; in the normal case they are `temporaries` used to guide
  // pruning.  NOTE: the index of the array is not the FST state that is final,
  // but the label on arcs entering it (these will be `token-labels`).  Each
  // token-final state will have the same label on all arcs entering it.
  //
  // `old_final_costs` is assumed to be empty at entry.
  void GetRawLatticeFinalCosts(const Lattice &raw_fst,
                               std::unordered_map<Label, BaseFloat> *old_final_costs);

  // Sets up non_final_redet_states_.  See documentation for that variable.
  void GetNonFinalRedetStates();

  /** [called from AcceptRawLatticeChunk()] Processes arcs that leave the
      start-state of `chunk_clat` (if this is not the first chunk); does nothing
      if this is the first chunk.  This includes using the `state-labels` to
      work out which states in clat_ these states correspond to, and writing
      that mapping to `state_map`.

      Also modifies forward_costs_, because it has to do a kind of reweighting
      of the clat states that are the values it puts in `state_map`, to take
      account of the probabilities on the arcs from the start state of
      chunk_clat to the states corresponding to those redeterminized-states
      (i.e. the states in clat corresponding to the values it puts in
      `*state_map`).  It also modifies arcs_in_, mostly because there
      are rare cases when we end up `merging` sets of those redeterminized-states,
      because the determinization process mapped them to a single state,
      and that means we need to reroute the arcs into members of that
      set into one single member (which will appear as a value in
      `*state_map`).

        @param [in] chunk_clat   The determinized chunk of lattice we are
                          processing
        @param [out] state_map    Mapping from states in chunk_clat to
                          the state in clat_ they correspond to.
        @return     Returns true if this is the first chunk.
  */
  bool ProcessArcsFromChunkStartState(
      const CompactLattice &chunk_clat,
      std::unordered_map<CompactLattice::StateId, CompactLattice::StateId> *state_map);

  /**
     This function, called from AcceptRawLatticeChunk(), transfers arcs from
     `chunk_clat` to clat_.  For those arcs that have `token-labels` on them,
     they don't get written to clat_ but instead are stored in the arcs_ array.

        @param [in] chunk_clat    The determinized lattice for the chunk
                         we are processing; this is the source of the arcs
                         we are moving.
        @param [in] is_first_chunk  True if this is the first chunk in the
                         utterance; it's needed because if it is, we
                         will also transfer arcs from the start state of
                         chunk_clat.
        @param [in] state_map  Map from state-ids in chunk_clat to state-ids
                         in clat_.
        @param [in] chunk_state_to_token  Map from `token-final states`
                         (see glossary) in chunk_clat, to the token-label
                         on arcs entering those states.
        @param [in] old_final_costs  Map from token-label to the
                         final-costs that were on the corresponding
                         token-final states in the undeterminized lattice;
                         these final-costs need to be removed when
                         we record the weights in final_arcs_, because
                         they were just temporary.
   */
  void TransferArcsToClat(
      const CompactLattice &chunk_clat,
      bool is_first_chunk,
      const std::unordered_map<CompactLattice::StateId, CompactLattice::StateId> &state_map,
      const std::unordered_map<CompactLattice::StateId, Label> &chunk_state_to_token,
      const std::unordered_map<Label, BaseFloat> &old_final_costs);



  /**
     Adds one arc to `clat_`.  It's like clat_.AddArc(state, arc), except
     it also modifies arcs_in_ and forward_costs_.
   */
  void AddArcToClat(CompactLattice::StateId state,
                    const CompactLatticeArc &arc);
  CompactLattice::StateId AddStateToClat();


  // Identifies token-final states in `chunk_clat`; see glossary above for
  // definition of `token-final`.  This function outputs a map from such states
  // in chunk_clat, to the `token-label` on arcs entering them.  (It is not
  // possible that the same state would have multiple arcs entering it with
  // different token-labels, or some arcs entering with one token-label and some
  // another, or be both initial and have such arcs; this is true due to how we
  // construct the raw lattice.)
  void IdentifyTokenFinalStates(
      const CompactLattice &chunk_clat,
      std::unordered_map<CompactLattice::StateId, CompactLatticeArc::Label> *token_map) const;

  // trans_model_ is needed by DeterminizeLatticePhonePrunedWrapper() which this
  // class calls.
  const TransitionInformation &trans_model_;
  // config_ is needed by DeterminizeLatticePhonePrunedWrapper() which this
  // class calls.
  const LatticeIncrementalDecoderConfig &config_;


  // Contains the set of redeterminized-states which are not final in the
  // canonical appended lattice.  Since the final ones don't physically appear
  // in clat_, this means the set of redeterminized-states which are physically
  // in clat_.  In code terms, this means set of .first elements in final_arcs,
  // plus whatever other states in clat_ are reachable from such states.
  std::unordered_set<CompactLattice::StateId> non_final_redet_states_;


  // clat_ is the appended lattice (containing all chunks processed so
  // far), except its `final-arcs` (i.e. arcs which in the canonical
  // lattice would go to final-states) are not present (they are stored
  // separately in final_arcs_) and states which in the canonical lattice
  // should have final-arcs leaving them will instead have a final-prob.
  CompactLattice clat_;


  // arcs_in_ is indexed by (state-id in clat_), and is a list of
  // arcs that come into this state, in the form (prev-state,
  // arc-index).  CAUTION: not all these input-arc records will always
  // be valid (some may be out-of-date, and may refer to an out-of-range
  // arc or an arc that does not point to this state).  But all
  // input arcs will always be listed.
  std::vector<std::vector<std::pair<CompactLattice::StateId, int32> > > arcs_in_;

  // final_arcs_ contains arcs which would appear in the canonical appended
  // lattice but for implementation reasons are not physically present in clat_.
  // These are arcs to final states in the canonical appended lattice.  The
  // .first elements are the source states in clat_ (these will all be elements
  // of non_final_redet_states_); the .nextstate elements of the arcs does not
  // contain a physical state, but contain state-labels allocated by
  // AllocateNewStateLabel().
  std::vector<CompactLatticeArc> final_arcs_;

  // forward_costs_, indexed by the state-id in clat_, stores the alpha
  // (forward) costs, i.e. the minimum cost from the start state to each state
  // in clat_.  This is relevant for pruned determinization.  The BaseFloat can
  // be thought of as the sum of a Value1() + Value2() in a LatticeWeight.
  std::vector<BaseFloat> forward_costs_;

  // temporary used in a function, kept here to avoid excessive reallocation.
  std::unordered_set<int32> temp_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDeterminizer);
};


/** This is an extention to the "normal" lattice-generating decoder.
   See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
    for more information.

   The main difference is the incremental determinization which will be
   discussed in the function GetLattice().  This means that the work of determinizatin
   isn't done all at once at the end of the file, but incrementally while decoding.
   See the comment at the top of this file for more explanation.

   The decoder is templated on the FST type and the token type.  The token type
   will normally be StdToken, but also may be BackpointerToken which is to support
   quick lookup of the current best path (see lattice-faster-online-decoder.h)

   The FST you invoke this decoder with is expected to be of type
   Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
   FST == StdFst and it notices that the actual FST type is
   fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
   will internally cast itself to one that is templated on those more specific
   types; this is an optimization for speed.
 */
template <typename FST, typename Token = decoder::StdToken>
class LatticeIncrementalDecoderTpl {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using ForwardLinkT = decoder::ForwardLink<Token>;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  LatticeIncrementalDecoderTpl(const FST &fst, const TransitionInformation &trans_model,
                               const LatticeIncrementalDecoderConfig &config);

  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  LatticeIncrementalDecoderTpl(const LatticeIncrementalDecoderConfig &config,
                               FST *fst, const TransitionInformation &trans_model);

  void SetOptions(const LatticeIncrementalDecoderConfig &config) { config_ = config; }

  const LatticeIncrementalDecoderConfig &GetOptions() const { return config_; }

  ~LatticeIncrementalDecoderTpl();

  /**
     CAUTION: it's unlikely that you will ever want to call this function.  In a
     scenario where you have the entire file and just want to decode it, there
     is no point using this decoder.

     An example of how to do decoding together with incremental
     determinization. It decodes until there are no more frames left in the
     "decodable" object.

     In this example, config_.determinize_max_delay, config_.determinize_min_chunk_size
     and config_.determinize_max_active are used to determine the time to
     call GetLattice().

     Users will probably want to use appropriate combinations of
     AdvanceDecoding() and GetLattice() to build their application; this just
     gives you some idea how.

     The function returns true if any kind of traceback is available (not
     necessarily from a final state).
  */
  bool Decode(DecodableInterface *decodable);

  /// says whether a final-state was active on the last frame.  If it was not,
  /// the lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }

  /**
     This decoder has no GetBestPath() function.
     If you need that functionality you should probably use lattice-incremental-online-decoder.h,
     which makes it very efficient to obtain the best path. */

  /**
     This GetLattice() function returns the lattice containing
     `num_frames_to_decode` frames; this will be all frames decoded so
     far, if you let num_frames_to_decode == NumFramesDecoded(),
     but it will generally be better to make it a few frames less than
     that to avoid the lattice having too many active states at
     the end.

     @param [in] num_frames_to_include  The number of frames that you want
                     to be included in the lattice.  Must be >=
                     NumFramesInLattice() and <= NumFramesDecoded().

     @param [in] use_final_probs  True if you want the final-probs
                    of HCLG to be included in the output lattice.  Must not
                    be set to true if num_frames_to_include !=
                    NumFramesDecoded().  Must be set to true if you have
                    previously called FinalizeDecoding().

                    (If no state was final on frame `num_frames_to_include`, the
                    final-probs won't be included regardless of
                    `use_final_probs`; you can test whether this
                    was the case by calling ReachedFinal().

      @return clat   The CompactLattice representing what has been decoded
                     up until `num_frames_to_include` (e.g., LatticeStateTimes()
                     on this lattice would return `num_frames_to_include`).

     See also UpdateLatticeDeterminizaton().  Caution: this const ref
     is only valid until the next time you call AdvanceDecoding() or
     GetLattice().

     CAUTION: the lattice may contain disconnnected states; you should
     call Connect() on the output before writing it out.
  */
  const CompactLattice &GetLattice(int32 num_frames_to_include,
                                   bool use_final_probs = false);

  /*
    Returns the number of frames in the currently-determinized part of the
    lattice which will be a number in [0, NumFramesDecoded()].  It will
    be the largest number that GetLattice() was called with, but note
    that GetLattice() may be called from UpdateLatticeDeterminization().

    Made available in case the user wants to give that same number to
    GetLattice().
   */
  int NumFramesInLattice() const { return num_frames_in_lattice_; }

  /**
     InitDecoding initializes the decoding, and should only be used if you
     intend to call AdvanceDecoding().  If you call Decode(), you don't need to
     call this.  You can also call InitDecoding if you have already decoded an
     utterance and want to start with a new utterance.
  */
  void InitDecoding();

  /**
     This will decode until there are no more frames ready in the decodable
     object.  You can keep calling it each time more frames become available
     (this is the normal pattern in a real-time/online decoding scenario).
     If max_num_frames is specified, it specifies the maximum number of frames
     the function will decode before returning.
  */
  void AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames = -1);


  /** FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
      more information.  It returns the difference between the best (final-cost
      plus cost) of any token on the final frame, and the best cost of any token
      on the final frame.  If it is infinity it means no final-states were
      present on the final frame.  It will usually be nonnegative.  If it not
      too positive (e.g. < 5 is my first guess, but this is not tested) you can
      take it as a good indication that we reached the final-state with
      reasonable likelihood. */
  BaseFloat FinalRelativeCost() const;

  /** Returns the number of frames decoded so far. */
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }

  /**
     Finalizes the decoding, doing an extra pruning step on the last frame
     that uses the final-probs.  May be called only once.
  */
  void FinalizeDecoding();

 protected:
  /* Some protected things are needed in LatticeIncrementalOnlineDecoderTpl. */

  /** NOTE: for parts the internal implementation that are shared with LatticeFasterDecoer,
      we have removed the comments.*/
  inline static void DeleteForwardLinks(Token *tok);
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    int32 num_toks;  /* Note: you can only trust `num_toks` if must_prune_tokens
                      * == false, because it is only set in
                      * PruneTokensForFrame(). */
    TokenList()
        : toks(NULL), must_prune_forward_links(true), must_prune_tokens(true),
          num_toks(-1) {}
  };
  using Elem = typename HashList<StateId, Token *>::Elem;
  void PossiblyResizeHash(size_t num_toks);
  inline Token *FindOrAddToken(StateId state, int32 frame_plus_one,
                               BaseFloat tot_cost, Token *backpointer, bool *changed);
  void PruneForwardLinks(int32 frame_plus_one, bool *extra_costs_changed,
                         bool *links_pruned, BaseFloat delta);
  void ComputeFinalCosts(unordered_map<Token *, BaseFloat> *final_costs,
                         BaseFloat *final_relative_cost,
                         BaseFloat *final_best_cost) const;
  void PruneForwardLinksFinal();
  void PruneTokensForFrame(int32 frame_plus_one);
  void PruneActiveTokens(BaseFloat delta);
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
                      Elem **best_elem);
  BaseFloat ProcessEmitting(DecodableInterface *decodable);
  void ProcessNonemitting(BaseFloat cost_cutoff);

  HashList<StateId, Token *> toks_;
  std::vector<TokenList> active_toks_;  // indexed by frame.
  std::vector<StateId> queue_;       // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_; // used in GetCutoff.
  const FST *fst_;
  bool delete_fst_;
  std::vector<BaseFloat> cost_offsets_;
  int32 num_toks_;
  bool warned_;
  bool decoding_finalized_;

  unordered_map<Token *, BaseFloat> final_costs_;
  BaseFloat final_relative_cost_;
  BaseFloat final_best_cost_;

  /***********************
      Variables below this point relate to the incremental
      determinization.
  *********************/
  LatticeIncrementalDecoderConfig config_;
  /** Much of the the incremental determinization algorithm is encapsulated in
      the determinize_ object.  */
  LatticeIncrementalDeterminizer determinizer_;


  /* Just a temporary used in a function; stored here to avoid reallocation. */
  unordered_map<Token*, StateId> temp_token_map_;

  /** num_frames_in_lattice_ is the highest `num_frames_to_include_` argument
      for any prior call to GetLattice(). */
  int32 num_frames_in_lattice_;

  // A map from Token to its token_label.  Will contain an entry for
  // each Token in active_toks_[num_frames_in_lattice_].
  unordered_map<Token*, Label> token2label_map_;

  // A temporary used in a function, kept here to avoid reallocation.
  unordered_map<Token*, Label> token2label_map_temp_;

  // we allocate a unique id for each Token
  Label next_token_label_;

  inline Label AllocateNewTokenLabel() { return next_token_label_++; }


  // There are various cleanup tasks... the the toks_ structure contains
  // singly linked lists of Token pointers, where Elem is the list type.
  // It also indexes them in a hash, indexed by state (this hash is only
  // maintained for the most recent frame).  toks_.Clear()
  // deletes them from the hash and returns the list of Elems.  The
  // function DeleteElems calls toks_.Delete(elem) for each elem in
  // the list, which returns ownership of the Elem to the toks_ structure
  // for reuse, but does not delete the Token pointer.  The Token pointers
  // are reference-counted and are ultimately deleted in PruneTokensForFrame,
  // but are also linked together on each frame by their own linked-list,
  // using the "next" pointer.  We delete them manually.
  void DeleteElems(Elem *list);

  void ClearActiveTokens();


  // Returns the number of active tokens on frame `frame`.  Can be used as part
  // of a heuristic to decide which frame to determinize until, if you are not
  // at the end of an utterance.
  int32 GetNumToksForFrame(int32 frame);

  /**
     UpdateLatticeDeterminization() ensures the work of determinization is kept
     up to date so that when you do need the lattice you can get it fast.  It
     uses the configuration values `determinize_max_delay`, `determinize_min_chunk_size`
     and `determinize_max_active`, to decide whether and when to call
     GetLattice().  You can safely call this as often as you want (e.g.  after
     each time you call AdvanceDecoding(); it won't do subtantially more work if
     it is called frequently.
  */
  void UpdateLatticeDeterminization();


  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeIncrementalDecoderTpl);
};

typedef LatticeIncrementalDecoderTpl<fst::StdFst, decoder::StdToken>
    LatticeIncrementalDecoder;


} // end namespace kaldi.

#endif
