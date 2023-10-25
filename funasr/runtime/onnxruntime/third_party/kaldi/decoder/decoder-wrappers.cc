// decoder/decoder-wrappers.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/decoder-wrappers.h"
#include "decoder/faster-decoder.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/grammar-fst.h"
#include "lat/lattice-functions.h"

namespace kaldi {





DecodeUtteranceLatticeFasterClass::DecodeUtteranceLatticeFasterClass(
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
    int32 *num_partial):  // If partial decode (final-state not reached), increments this.
    decoder_(decoder), decodable_(decodable), trans_model_(&trans_model),
    word_syms_(word_syms), utt_(utt), acoustic_scale_(acoustic_scale),
    determinize_(determinize), allow_partial_(allow_partial),
    alignments_writer_(alignments_writer),
    words_writer_(words_writer),
    compact_lattice_writer_(compact_lattice_writer),
    lattice_writer_(lattice_writer),
    like_sum_(like_sum), frame_sum_(frame_sum),
    num_done_(num_done), num_err_(num_err),
    num_partial_(num_partial),
    computed_(false), success_(false), partial_(false),
    clat_(NULL), lat_(NULL) { }


void DecodeUtteranceLatticeFasterClass::operator () () {
  // Decoding and lattice determinization happens here.
  computed_ = true; // Just means this function was called-- a check on the
  // calling code.
  success_ = true;
  using fst::VectorFst;
  if (!decoder_->Decode(decodable_)) {
    KALDI_WARN << "Failed to decode utterance with id " << utt_;
    success_ = false;
  }
  if (!decoder_->ReachedFinal()) {
    if (allow_partial_) {
      KALDI_WARN << "Outputting partial output for utterance " << utt_
                 << " since no final-state reached\n";
      partial_ = true;
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt_
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      success_ = false;
    }
  }
  if (!success_) return;

  // Get lattice, and do determinization if requested.
  lat_ = new Lattice;
  decoder_->GetRawLattice(lat_);
  if (lat_->NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt_;
  fst::Connect(lat_);
  if (determinize_) {
    clat_ = new CompactLattice;
    if (!DeterminizeLatticePhonePrunedWrapper(
            *trans_model_,
            lat_,
            decoder_->GetOptions().lattice_beam,
            clat_,
            decoder_->GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt_;
    delete lat_;
    lat_ = NULL;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale_ != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale_), clat_);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale_ != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale_), lat_);
  }
}

DecodeUtteranceLatticeFasterClass::~DecodeUtteranceLatticeFasterClass() {
  if (!computed_)
    KALDI_ERR << "Destructor called without operator (), error in calling code.";

  if (!success_) {
    if (num_err_ != NULL) (*num_err_)++;
  } else { // successful decode.
    // Getting the one-best output is lightweight enough that we can do it in
    // the destructor (easier than adding more variables to the class, and
    // will rarely slow down the main thread.)
    double likelihood;
    LatticeWeight weight;
    int32 num_frames;
    { // First do some stuff with word-level traceback...
      // This is basically for diagnostics.
      fst::VectorFst<LatticeArc> decoded;
      decoder_->GetBestPath(&decoded);
      if (decoded.NumStates() == 0) {
        // Shouldn't really reach this point as already checked success.
        KALDI_ERR << "Failed to get traceback for utterance " << utt_;
      }
      std::vector<int32> alignment;
      std::vector<int32> words;
      GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
      num_frames = alignment.size();
      if (words_writer_->IsOpen())
        words_writer_->Write(utt_, words);
      if (alignments_writer_->IsOpen())
        alignments_writer_->Write(utt_, alignment);
      if (word_syms_ != NULL) {
        std::cerr << utt_ << ' ';
        for (size_t i = 0; i < words.size(); i++) {
          std::string s = word_syms_->Find(words[i]);
          if (s == "")
            KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
          std::cerr << s << ' ';
        }
        std::cerr << '\n';
      }
      likelihood = -(weight.Value1() + weight.Value2());
    }

    // Ouptut the lattices.
    if (determinize_) { // CompactLattice output.
      KALDI_ASSERT(compact_lattice_writer_ != NULL && clat_ != NULL);
      if (clat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        compact_lattice_writer_->Write(utt_, *clat_);
      }
      delete clat_;
      clat_ = NULL;
    } else {
      KALDI_ASSERT(lattice_writer_ != NULL && lat_ != NULL);
      if (lat_->NumStates() == 0) {
        KALDI_WARN << "Empty lattice for utterance " << utt_;
      } else {
        lattice_writer_->Write(utt_, *lat_);
      }
      delete lat_;
      lat_ = NULL;
    }

    // Print out logging information.
    KALDI_LOG << "Log-like per frame for utterance " << utt_ << " is "
              << (likelihood / num_frames) << " over "
              << num_frames << " frames.";
    KALDI_VLOG(2) << "Cost for utterance " << utt_ << " is "
                  << weight.Value1() << " + " << weight.Value2();

    // Now output the various diagnostic variables.
    if (like_sum_ != NULL) *like_sum_ += likelihood;
    if (frame_sum_ != NULL) *frame_sum_ += num_frames;
    if (num_done_ != NULL) (*num_done_)++;
    if (partial_ && num_partial_ != NULL) (*num_partial_)++;
  }
  // We were given ownership of these two objects that were passed in in
  // the initializer.
  delete decoder_;
  delete decodable_;
}

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
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr) { // puts utterance's like in like_ptr on success.
  using fst::VectorFst;
  if (!decoder.Decode(&decodable)) {
    KALDI_WARN << "Failed to decode utterance with id " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  // Get lattice
  CompactLattice clat = decoder.GetLattice(decoder.NumFramesDecoded(), true);
  if (clat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    CompactLattice decoded_clat;
    CompactLatticeShortestPath(clat, &decoded_clat);
    Lattice decoded;
    fst::ConvertLattice(decoded_clat, &decoded);

    if (decoded.Start() == fst::kNoStateId)
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    KALDI_ASSERT(num_frames == decoder.NumFramesDecoded());
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // We'll write the lattice without acoustic scaling.
  if (acoustic_scale != 0.0)
    fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
  Connect(&clat);
  compact_lattice_writer->Write(utt, clat);
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}


// Takes care of output.  Returns true on success.
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
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr) { // puts utterance's like in like_ptr on success.
  using fst::VectorFst;

  if (!decoder.Decode(&decodable)) {
    KALDI_WARN << "Failed to decode utterance with id " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded))
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  decoder.GetRawLattice(&lat);
  if (lat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
    lattice_writer->Write(utt, lat);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}

// Instantiate the template above for the two required FST types.
template bool DecodeUtteranceLatticeIncremental(
    LatticeIncrementalDecoderTpl<fst::Fst<fst::StdArc> > &decoder,
    DecodableInterface &decodable,
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);

template bool DecodeUtteranceLatticeIncremental(
    LatticeIncrementalDecoderTpl<fst::ConstGrammarFst > &decoder,
    DecodableInterface &decodable,
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);

template bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<fst::Fst<fst::StdArc> > &decoder,
    DecodableInterface &decodable,
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);

template bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<fst::ConstGrammarFst > &decoder,
    DecodableInterface &decodable,
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);


// Takes care of output.  Returns true on success.
bool DecodeUtteranceLatticeSimple(
    LatticeSimpleDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionInformation &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr) { // puts utterance's like in like_ptr on success.
  using fst::VectorFst;

  if (!decoder.Decode(&decodable)) {
    KALDI_WARN << "Failed to decode utterance with id " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight = LatticeWeight::Zero();
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded))
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  if (!decoder.GetRawLattice(&lat))
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
    lattice_writer->Write(utt, lat);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}


// see comment in header.
void ModifyGraphForCarefulAlignment(
    fst::VectorFst<fst::StdArc> *fst) {
  typedef fst::StdArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;
  typedef Arc::Weight Weight;
  StateId num_states = fst->NumStates();
  if (num_states == 0) {
    KALDI_WARN << "Empty FST input.";
    return;
  }
  Weight zero = Weight::Zero();
  // fst_rhs will be the right hand side of the Concat operation.
  fst::VectorFst<fst::StdArc> fst_rhs(*fst);
  // first remove the final-probs from fst_rhs.
  for (StateId state = 0; state < num_states; state++)
    fst_rhs.SetFinal(state, zero);
  StateId pre_initial = fst_rhs.AddState();
  Arc to_initial(0, 0, Weight::One(), fst_rhs.Start());
  fst_rhs.AddArc(pre_initial, to_initial);
  fst_rhs.SetStart(pre_initial);
  // make the pre_initial state final with probability one;
  // this is equivalent to keeping the final-probs of the first
  // FST when we do concat (otherwise they would get deleted).
  fst_rhs.SetFinal(pre_initial, Weight::One());
  fst::VectorFst<fst::StdArc> fst_concat;
  fst::Concat(fst, fst_rhs);
}


void AlignUtteranceWrapper(
    const AlignConfig &config,
    const std::string &utt,
    BaseFloat acoustic_scale,  // affects scores written to scores_writer, if
                               // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true.
    DecodableInterface *decodable,  // not const but is really an input.
    Int32VectorWriter *alignment_writer,
    BaseFloatWriter *scores_writer,
    int32 *num_done,
    int32 *num_error,
    int32 *num_retried,
    double *tot_like,
    int64 *frame_count,
    BaseFloatVectorWriter *per_frame_acwt_writer) {

  if ((config.retry_beam != 0 && config.retry_beam <= config.beam) ||
      config.beam <= 0.0) {
    KALDI_ERR << "Beams do not make sense: beam " << config.beam
              << ", retry-beam " << config.retry_beam;
  }

  if (fst->Start() == fst::kNoStateId) {
    KALDI_WARN << "Empty decoding graph for " << utt;
    if (num_error != NULL) (*num_error)++;
    return;
  }

  if (config.careful)
    ModifyGraphForCarefulAlignment(fst);

  FasterDecoderOptions decode_opts;
  decode_opts.beam = config.beam;

  FasterDecoder decoder(*fst, decode_opts);
  decoder.Decode(decodable);

  bool ans = decoder.ReachedFinal();  // consider only final states.

  if (!ans && config.retry_beam != 0.0) {
    if (num_retried != NULL) (*num_retried)++;
    KALDI_WARN << "Retrying utterance " << utt << " with beam "
               << config.retry_beam;
    decode_opts.beam = config.retry_beam;
    decoder.SetOptions(decode_opts);
    decoder.Decode(decodable);
    ans = decoder.ReachedFinal();
  }

  if (!ans) {  // Still did not reach final state.
    KALDI_WARN << "Did not successfully decode file " << utt << ", len = "
               << decodable->NumFramesReady();
    if (num_error != NULL) (*num_error)++;
    return;
  }

  fst::VectorFst<LatticeArc> decoded;  // linear FST.
  decoder.GetBestPath(&decoded);
  if (decoded.NumStates() == 0) {
    KALDI_WARN << "Error getting best path from decoder (likely a bug)";
    if (num_error != NULL) (*num_error)++;
    return;
  }

  std::vector<int32> alignment;
  std::vector<int32> words;
  LatticeWeight weight;

  GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
  BaseFloat like = -(weight.Value1()+weight.Value2()) / acoustic_scale;

  if (num_done != NULL) (*num_done)++;
  if (tot_like != NULL) (*tot_like) += like;
  if (frame_count != NULL) (*frame_count) += decodable->NumFramesReady();

  if (alignment_writer != NULL && alignment_writer->IsOpen())
    alignment_writer->Write(utt, alignment);

  if (scores_writer != NULL && scores_writer->IsOpen())
    scores_writer->Write(utt, -(weight.Value1()+weight.Value2()));

  Vector<BaseFloat> per_frame_loglikes;
  if (per_frame_acwt_writer != NULL && per_frame_acwt_writer->IsOpen()) {
    GetPerFrameAcousticCosts(decoded, &per_frame_loglikes);
    per_frame_loglikes.Scale(-1 / acoustic_scale);
    per_frame_acwt_writer->Write(utt, per_frame_loglikes);
  }
}

} // end namespace kaldi.
