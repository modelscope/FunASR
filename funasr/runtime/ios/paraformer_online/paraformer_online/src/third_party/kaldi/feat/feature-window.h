// feat/feature-window.h

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
//           2014-2016  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FEAT_FEATURE_WINDOW_H_
#define KALDI_FEAT_FEATURE_WINDOW_H_

#include <map>
#include <string>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

struct FrameExtractionOptions {
  BaseFloat samp_freq;
  BaseFloat frame_shift_ms;  // in milliseconds.
  BaseFloat frame_length_ms;  // in milliseconds.
  BaseFloat dither;  // Amount of dithering, 0.0 means no dither.
  BaseFloat preemph_coeff;  // Preemphasis coefficient.
  bool remove_dc_offset;  // Subtract mean of wave before FFT.
  std::string window_type;  // e.g. Hamming window
  bool round_to_power_of_two;
  BaseFloat blackman_coeff;
  bool snip_edges;
  bool allow_downsample;
  // May be "hamming", "rectangular", "povey", "hanning", "blackman"
  // "povey" is a window I made to be similar to Hamming but to go to zero at the
  // edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
  // I just don't think the Hamming window makes sense as a windowing function.
  FrameExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      dither(1.0),
      preemph_coeff(0.97),
      remove_dc_offset(true),
      window_type("povey"),
      round_to_power_of_two(true),
      blackman_coeff(0.42),
      snip_edges(true),
      allow_downsample(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("sample-frequency", &samp_freq,
                   "Waveform data sample frequency (must match the waveform file, "
                   "if specified there)");
    opts->Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
    opts->Register("frame-shift", &frame_shift_ms, "Frame shift in milliseconds");
    opts->Register("preemphasis-coefficient", &preemph_coeff,
                   "Coefficient for use in signal preemphasis");
    opts->Register("remove-dc-offset", &remove_dc_offset,
                   "Subtract mean from waveform on each frame");
    opts->Register("dither", &dither, "Dithering constant (0.0 means no dither)");
    opts->Register("window-type", &window_type, "Type of window "
                   "(\"hamming\"|\"hanning\"|\"povey\"|\"rectangular\""
                   "|\"blackmann\")");
    opts->Register("blackman-coeff", &blackman_coeff,
                   "Constant coefficient for generalized Blackman window.");
    opts->Register("round-to-power-of-two", &round_to_power_of_two,
                   "If true, round window size to power of two by zero-padding "
                   "input to FFT.");
    opts->Register("snip-edges", &snip_edges,
                   "If true, end effects will be handled by outputting only frames that "
                   "completely fit in the file, and the number of frames depends on the "
                   "frame-length.  If false, the number of frames depends only on the "
                   "frame-shift, and we reflect the data at the ends.");
    opts->Register("allow-downsample", &allow_downsample,
                   "If true, allow the input waveform to have a higher frequency than "
                   "the specified --sample-frequency (and we'll downsample).");
  }
  int32 WindowShift() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_shift_ms);
  }
  int32 WindowSize() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_length_ms);
  }
  int32 PaddedWindowSize() const {
    return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) :
                                    WindowSize());
  }
};


struct FeatureWindowFunction {
  FeatureWindowFunction() {}
  explicit FeatureWindowFunction(const FrameExtractionOptions &opts);
  FeatureWindowFunction(const FeatureWindowFunction &other):
      window(other.window) { }
  Vector<BaseFloat> window;
};


/**
   This function returns the number of frames that we can extract from a wave
   file with the given number of samples in it (assumed to have the same
   sampling rate as specified in 'opts').

      @param [in] num_samples  The number of samples in the wave file.
      @param [in] opts     The frame-extraction options class

      @param [in] flush   True if we are asserting that this number of samples is
             'all there is', false if we expecting more data to possibly come
             in.  This only makes a difference to the answer if opts.snips_edges
             == false.  For offline feature extraction you always want flush ==
             true.  In an online-decoding context, once you know (or decide) that
             no more data is coming in, you'd call it with flush == true at the
             end to flush out any remaining data.
*/
int32 NumFrames(int64 num_samples,
                const FrameExtractionOptions &opts,
                bool flush = true);

/*
   This function returns the index of the first sample of the frame indexed
   'frame'.  If snip-edges=true, it just returns frame * opts.WindowShift(); if
   snip-edges=false, the formula is a little more complicated and the result may
   be negative.
*/
int64 FirstSampleOfFrame(int32 frame,
                         const FrameExtractionOptions &opts);



void Dither(VectorBase<BaseFloat> *waveform, BaseFloat dither_value);

void Preemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff);

/**
  This function does all the windowing steps after actually
  extracting the windowed signal: depeding on the
  configuration, it does dithering, dc offset removal,
  preemphasis, and multiplication by the windowing function.
   @param [in] opts  The options class to be used
   @param [in] window_function  The windowing function-- should have
                    been initialized using 'opts'.
   @param [in,out] window  A vector of size opts.WindowSize().  Note:
      it will typically be a sub-vector of a larger vector of size
      opts.PaddedWindowSize(), with the remaining samples zero,
      as the FFT code is more efficient if it operates on data with
      power-of-two size.
   @param [out]   log_energy_pre_window If non-NULL, then after dithering and
      DC offset removal, this function will write to this pointer the log of
      the total energy (i.e. sum-squared) of the frame.
 */
void ProcessWindow(const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   VectorBase<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window = NULL);


/*
  ExtractWindow() extracts a windowed frame of waveform (possibly with a
  power-of-two, padded size, depending on the config), including all the
  proessing done by ProcessWindow().

  @param [in] sample_offset  If 'wave' is not the entire waveform, but
                   part of it to the left has been discarded, then the
                   number of samples prior to 'wave' that we have
                   already discarded.  Set this to zero if you are
                   processing the entire waveform in one piece, or
                   if you get 'no matching function' compilation
                   errors when updating the code.
  @param [in] wave  The waveform
  @param [in] f     The frame index to be extracted, with
                    0 <= f < NumFrames(sample_offset + wave.Dim(), opts, true)
  @param [in] opts  The options class to be used
  @param [in] window_function  The windowing function, as derived from the
                    options class.
  @param [out] window  The windowed, possibly-padded waveform to be
                     extracted.  Will be resized as needed.
  @param [out] log_energy_pre_window  If non-NULL, the log-energy of
                   the signal prior to pre-emphasis and multiplying by
                   the windowing function will be written to here.
*/
void ExtractWindow(int64 sample_offset,
                   const VectorBase<BaseFloat> &wave,
                   int32 f,
                   const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   Vector<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window = NULL);


// ExtractWaveformRemainder is useful if the waveform is coming in segments.
// It extracts the bit of the waveform at the end of this block that you
// would have to append the next bit of waveform to, if you wanted to have
// the same effect as everything being in one big block.
void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder);


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_WINDOW_H_
