/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// This file is copied/modified from kaldi/src/feat/mel-computations.h
#ifndef KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_
#define KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"

namespace knf {

struct MelBanksOptions {
  int32_t num_bins = 25;  // e.g. 25; number of triangular bins
  float low_freq = 20;    // e.g. 20; lower frequency cutoff

  // an upper frequency cutoff; 0 -> no cutoff, negative
  // ->added to the Nyquist frequency to get the cutoff.
  float high_freq = 0;

  float vtln_low = 100;  // vtln lower cutoff of warping function.

  // vtln upper cutoff of warping function: if negative, added
  // to the Nyquist frequency to get the cutoff.
  float vtln_high = -500;

  bool debug_mel = false;
  // htk_mode is a "hidden" config, it does not show up on command line.
  // Enables more exact compatibility with HTK, for testing purposes.  Affects
  // mel-energy flooring and reproduces a bug in HTK.
  bool htk_mode = false;

  std::string ToString() const {
    std::ostringstream os;
    os << "num_bins: " << num_bins << "\n";
    os << "low_freq: " << low_freq << "\n";
    os << "high_freq: " << high_freq << "\n";
    os << "vtln_low: " << vtln_low << "\n";
    os << "vtln_high: " << vtln_high << "\n";
    os << "debug_mel: " << debug_mel << "\n";
    os << "htk_mode: " << htk_mode << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const MelBanksOptions &opts);

class MelBanks {
 public:
  static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

  static float VtlnWarpFreq(
      float vtln_low_cutoff,
      float vtln_high_cutoff,  // discontinuities in warp func
      float low_freq,
      float high_freq,  // upper+lower frequency cutoffs in
      // the mel computation
      float vtln_warp_factor, float freq);

  static float VtlnWarpMelFreq(float vtln_low_cutoff, float vtln_high_cutoff,
                               float low_freq, float high_freq,
                               float vtln_warp_factor, float mel_freq);

  // TODO(fangjun): Remove vtln_warp_factor
  MelBanks(const MelBanksOptions &opts,
           const FrameExtractionOptions &frame_opts, float vtln_warp_factor);

  /// Compute Mel energies (note: not log energies).
  /// At input, "fft_energies" contains the FFT energies (not log).
  ///
  /// @param fft_energies 1-D array of size num_fft_bins/2+1
  /// @param mel_energies_out  1-D array of size num_mel_bins
  void Compute(const float *fft_energies, float *mel_energies_out) const;

  int32_t NumBins() const { return bins_.size(); }

 private:
  // center frequencies of bins, numbered from 0 ... num_bins-1.
  // Needed by GetCenterFreqs().
  std::vector<float> center_freqs_;

  // the "bins_" vector is a vector, one for each bin, of a pair:
  // (the first nonzero fft-bin), (the vector of weights).
  std::vector<std::pair<int32_t, std::vector<float>>> bins_;

  // TODO(fangjun): Remove debug_ and htk_mode_
  bool debug_;
  bool htk_mode_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_
