// kaldi-native-fbank/csrc/feature-window.h
//
// Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-window.h

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_

#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/log.h"

namespace knf {

inline int32_t RoundUpToNearestPowerOfTwo(int32_t n) {
  // copied from kaldi/src/base/kaldi-math.cc
  KNF_CHECK_GT(n, 0);
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

struct FrameExtractionOptions {
  float samp_freq = 16000;
  float frame_shift_ms = 10.0f;   // in milliseconds.
  float frame_length_ms = 25.0f;  // in milliseconds.
  float dither = 1.0f;            // Amount of dithering, 0.0 means no dither.
  float preemph_coeff = 0.97f;    // Preemphasis coefficient.
  bool remove_dc_offset = true;   // Subtract mean of wave before FFT.
  std::string window_type = "povey";  // e.g. Hamming window
  // May be "hamming", "rectangular", "povey", "hanning", "sine", "blackman"
  // "povey" is a window I made to be similar to Hamming but to go to zero at
  // the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85) I just don't think the
  // Hamming window makes sense as a windowing function.
  bool round_to_power_of_two = true;
  float blackman_coeff = 0.42f;
  bool snip_edges = true;
  // bool allow_downsample = false;
  // bool allow_upsample = false;

  // Used for streaming feature extraction. It indicates the number
  // of feature frames to keep in the recycling vector. -1 means to
  // keep all feature frames.
  int32_t max_feature_vectors = -1;

  int32_t WindowShift() const {
    return static_cast<int32_t>(samp_freq * 0.001f * frame_shift_ms);
  }
  int32_t WindowSize() const {
    return static_cast<int32_t>(samp_freq * 0.001f * frame_length_ms);
  }
  int32_t PaddedWindowSize() const {
    return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize())
                                  : WindowSize());
  }
  std::string ToString() const {
    std::ostringstream os;
#define KNF_PRINT(x) os << #x << ": " << x << "\n"
    KNF_PRINT(samp_freq);
    KNF_PRINT(frame_shift_ms);
    KNF_PRINT(frame_length_ms);
    KNF_PRINT(dither);
    KNF_PRINT(preemph_coeff);
    KNF_PRINT(remove_dc_offset);
    KNF_PRINT(window_type);
    KNF_PRINT(round_to_power_of_two);
    KNF_PRINT(blackman_coeff);
    KNF_PRINT(snip_edges);
    // KNF_PRINT(allow_downsample);
    // KNF_PRINT(allow_upsample);
    KNF_PRINT(max_feature_vectors);
#undef KNF_PRINT
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const FrameExtractionOptions &opts);

class FeatureWindowFunction {
 public:
  FeatureWindowFunction() = default;
  explicit FeatureWindowFunction(const FrameExtractionOptions &opts);
  /**
   * @param wave Pointer to a 1-D array of shape [window_size].
   *             It is modified in-place: wave[i] = wave[i] * window_[i].
   * @param
   */
  void Apply(float *wave) const;

 private:
  std::vector<float> window_;  // of size opts.WindowSize()
};

int64_t FirstSampleOfFrame(int32_t frame, const FrameExtractionOptions &opts);

/**
   This function returns the number of frames that we can extract from a wave
   file with the given number of samples in it (assumed to have the same
   sampling rate as specified in 'opts').

      @param [in] num_samples  The number of samples in the wave file.
      @param [in] opts     The frame-extraction options class

      @param [in] flush   True if we are asserting that this number of samples
   is 'all there is', false if we expecting more data to possibly come in.  This
   only makes a difference to the answer
   if opts.snips_edges== false.  For offline feature extraction you always want
   flush == true.  In an online-decoding context, once you know (or decide) that
   no more data is coming in, you'd call it with flush == true at the end to
   flush out any remaining data.
*/
int32_t NumFrames(int64_t num_samples, const FrameExtractionOptions &opts,
                  bool flush = true);

/*
  ExtractWindow() extracts a windowed frame of waveform (possibly with a
  power-of-two, padded size, depending on the config), including all the
  processing done by ProcessWindow().

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
void ExtractWindow(int64_t sample_offset, const std::vector<float> &wave,
                   int32_t f, const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   std::vector<float> *window,
                   float *log_energy_pre_window = nullptr);

/**
  This function does all the windowing steps after actually
  extracting the windowed signal: depending on the
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
                   const FeatureWindowFunction &window_function, float *window,
                   float *log_energy_pre_window = nullptr);

// Compute the inner product of two vectors
float InnerProduct(const float *a, const float *b, int32_t n);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_WINDOW_H_
