// decoder/decoder-wrappers.h

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_DECODER_WRAPPERS_H_
#define KALDI_DECODER_DECODER_WRAPPERS_H_

#include "itf/options-itf.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/lattice-incremental-decoder.h"
#include "decoder/lattice-simple-decoder.h"

// This header contains declarations from various convenience functions that are called
// from binary-level programs such as gmm-decode-faster.cc, gmm-align-compiled.cc, and
// so on.

namespace kaldi {


struct AlignConfig {
  BaseFloat beam;
  BaseFloat retry_beam;
  bool careful;

  AlignConfig(): beam(200.0), retry_beam(0.0), careful(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("beam", &beam, "Decoding beam used in alignment");
    opts->Register("retry-beam", &retry_beam,
                   "Decoding beam for second try at alignment");
    opts->Register("careful", &careful,
                   "If true, do 'careful' alignment, which is better at detecting "
                   "alignment failure (involves loop to start of decoding graph).");
  }
};


/// AlignUtteranceWapper is a wrapper for alignment code used in training, that
/// is called from many different binaries, e.g. gmm-align, gmm-align-compiled,
/// sgmm-align, etc.  The writers for alignments and words will only be written
/// to if they are open.  The num_done, num_error, num_retried, tot_like and
/// frame_count pointers will (if non-NULL) be incremented or added to, not set,
/// by this function.
void AlignUtteranceWrapper(
    const AlignConfig &config,
    const std::string &utt,
    BaseFloat acoustic_scale,  // affects scores written to scores_writer, if
                               // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true, we add loop.
    DecodableInterface *decodable,  // not const but is really an input.
    Int32VectorWriter *alignment_writer,
    BaseFloatWriter *scores_writer,
    int32 *num_done,
    int32 *num_error,
    int32 *num_retried,
    double *tot_like,
    int64 *frame_count,
    BaseFloatVectorWriter *per_frame_acwt_writer = NULL);



/// This function modifies the decoding graph for what we call "careful
/// alignment".  The problem we are trying to solve is that if the decoding eats
/// up the words in the graph too fast, it can get stuck at the end, and produce
/// what looks like a valid alignment even though there was really a failure.
/// So what we want to do is to introduce, after the final-states of the graph,
/// a "blind alley" with no final-probs reachable, where the decoding can go to
/// get lost.  Our basic idea is to append the decoding-graph to itself using
/// the fst Concat operation; but in order that there should be final-probs at the end of
/// the first but not the second FST, we modify the right-hand argument to the
/// Concat operation so that it has none of the original final-probs, and add
/// a "pre-initial" state that is final.
void ModifyGraphForCarefulAlignment(
    fst::VectorFst<fst::StdArc> *fst);

/// TODO
template <typename FST>
bool DecodeUtteranceLatticeIncremental(
    LatticeIncrementalDecoderTpl<FST> &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.


/// This function DecodeUtteranceLatticeFaster is used in several decoders, and
/// we have moved it here.  Note: this is really "binary-level" code as it
/// involves table readers and writers; we've just put it here as there is no
/// other obvious place to put it.  If determinize == false, it writes to
/// lattice_writer, else to compact_lattice_writer.  The writers for
/// alignments and words will only be written to if they are open.
///
/// Caution: this will only link correctly if FST is either fst::Fst<fst::StdArc>,
/// or fst::GrammarFst, as the template function is defined in the .cc file and
/// only instantiated for those two types.
template <typename FST>
bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<FST> &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.


/// This class basically does the same job as the function
/// DecodeUtteranceLatticeFaster, but in a way that allows us
/// to build a multi-threaded command line program more easily.
/// The main computation takes place in operator (), and the output
/// happens in the destructor.
class DecodeUtteranceLatticeFasterClass {
 public:
  // Initializer sets various variables.
  // NOTE: we "take ownership" of "decoder" and "decodable".  These
  // are deleted by the destructor.  On error, "num_err" is incremented.
  DecodeUtteranceLatticeFasterClass(
      LatticeFasterDecoder *decoder,
      DecodableInterface *decodable,
      const TransitionInformation &trans_model,
      const fst::SymbolTable *word_syms,
      const std::string &utt,
      BaseFloat acoustic_scale,
      bool determinize,
      bool allow_partial,
      Int32VectorWriter *alignments_writer,
      Int32VectorWriter *words_writer,
      CompactLatticeWriter *compact_lattice_writer,
      LatticeWriter *lattice_writer,
      double *like_sum, // on success, adds likelihood to this.
      int64 *frame_sum, // on success, adds #frames to this.
      int32 *num_done, // on success (including partial decode), increments this.
      int32 *num_err,  // on failure, increments this.
      int32 *num_partial);  // If partial decode (final-state not reached), increments this.
  void operator () (); // The decoding happens here.
  ~DecodeUtteranceLatticeFasterClass(); // Output happens here.
 private:
  // The following variables correspond to inputs:
  LatticeFasterDecoder *decoder_;
  DecodableInterface *decodable_;
  const TransitionInformation *trans_model_;
  const fst::SymbolTable *word_syms_;
  std::string utt_;
  BaseFloat acoustic_scale_;
  bool determinize_;
  bool allow_partial_;
  Int32VectorWriter *alignments_writer_;
  Int32VectorWriter *words_writer_;
  CompactLatticeWriter *compact_lattice_writer_;
  LatticeWriter *lattice_writer_;
  double *like_sum_;
  int64 *frame_sum_;
  int32 *num_done_;
  int32 *num_err_;
  int32 *num_partial_;

  // The following variables are stored by the computation.
  bool computed_; // operator ()  was called.
  bool success_; // decoding succeeded (possibly partial)
  bool partial_; // decoding was partial.
  CompactLattice *clat_; // Stored output, if determinize_ == true.
  Lattice *lat_; // Stored output, if determinize_ == false.
};

// This function DecodeUtteranceLatticeSimple is used in several decoders, and
// we have moved it here.  Note: this is really "binary-level" code as it
// involves table readers and writers; we've just put it here as there is no
// other obvious place to put it.  If determinize == false, it writes to
// lattice_writer, else to compact_lattice_writer.  The writers for
// alignments and words will only be written to if they are open.
bool DecodeUtteranceLatticeSimple(
    LatticeSimpleDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.



} // end namespace kaldi.


#endif
