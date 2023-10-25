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

// This file is copied/modified from kaldi/src/feat/mel-computations.cc

#include "kaldi-native-fbank/csrc/mel-computations.h"

#include <algorithm>
#include <sstream>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"

namespace knf {

std::ostream &operator<<(std::ostream &os, const MelBanksOptions &opts) {
  os << opts.ToString();
  return os;
}

float MelBanks::VtlnWarpFreq(
    float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
    float vtln_high_cutoff,
    float low_freq,  // upper+lower frequency cutoffs in mel computation
    float high_freq, float vtln_warp_factor, float freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq
  /// and high_freq inclusive, with the following properties:
  ///  F(low_freq) == low_freq
  ///  F(high_freq) == high_freq
  /// The function is continuous and piecewise linear with two inflection
  ///   points.
  /// The lower inflection point (measured in terms of the unwarped
  ///  frequency) is at frequency l, determined as described below.
  /// The higher inflection point is at a frequency h, determined as
  ///   described below.
  /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
  /// If the higher inflection point (measured in terms of the unwarped
  ///   frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
  ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
  ///   max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
  ///   h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
  ///     = vtln_high_cutoff * min(1, vtln_warp_factor).
  /// If the lower inflection point (measured in terms of the unwarped
  ///   frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
  ///   This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
  ///                       = vtln_low_cutoff * max(1, vtln_warp_factor)

  if (freq < low_freq || freq > high_freq)
    return freq;  // in case this gets called
  // for out-of-range frequencies, just return the freq.

  KNF_CHECK_GT(vtln_low_cutoff, low_freq);
  KNF_CHECK_LT(vtln_high_cutoff, high_freq);

  float one = 1.0f;
  float l = vtln_low_cutoff * std::max(one, vtln_warp_factor);
  float h = vtln_high_cutoff * std::min(one, vtln_warp_factor);
  float scale = 1.0f / vtln_warp_factor;
  float Fl = scale * l;  // F(l);
  float Fh = scale * h;  // F(h);
  KNF_CHECK(l > low_freq && h < high_freq);
  // slope of left part of the 3-piece linear function
  float scale_left = (Fl - low_freq) / (l - low_freq);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  float scale_right = (high_freq - Fh) / (high_freq - h);

  if (freq < l) {
    return low_freq + scale_left * (freq - low_freq);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq + scale_right * (freq - high_freq);
  }
}

float MelBanks::VtlnWarpMelFreq(
    float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
    float vtln_high_cutoff,
    float low_freq,  // upper+lower frequency cutoffs in mel computation
    float high_freq, float vtln_warp_factor, float mel_freq) {
  return MelScale(VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff, low_freq,
                               high_freq, vtln_warp_factor,
                               InverseMelScale(mel_freq)));
}

MelBanks::MelBanks(const MelBanksOptions &opts,
                   const FrameExtractionOptions &frame_opts,
                   float vtln_warp_factor)
    : htk_mode_(opts.htk_mode) {
  int32_t num_bins = opts.num_bins;
  if (num_bins < 3) KNF_LOG(FATAL) << "Must have at least 3 mel bins";

  float sample_freq = frame_opts.samp_freq;
  int32_t window_length_padded = frame_opts.PaddedWindowSize();
  KNF_CHECK_EQ(window_length_padded % 2, 0);

  int32_t num_fft_bins = window_length_padded / 2;
  float nyquist = 0.5f * sample_freq;

  float low_freq = opts.low_freq, high_freq;
  if (opts.high_freq > 0.0f)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  if (low_freq < 0.0f || low_freq >= nyquist || high_freq <= 0.0f ||
      high_freq > nyquist || high_freq <= low_freq) {
    KNF_LOG(FATAL) << "Bad values in options: low-freq " << low_freq
                   << " and high-freq " << high_freq << " vs. nyquist "
                   << nyquist;
  }

  float fft_bin_width = sample_freq / window_length_padded;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]

  float mel_low_freq = MelScale(low_freq);
  float mel_high_freq = MelScale(high_freq);

  debug_ = opts.debug_mel;

  // divide by num_bins+1 in next line because of end-effects where the bins
  // spread out to the sides.
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);

  float vtln_low = opts.vtln_low, vtln_high = opts.vtln_high;
  if (vtln_high < 0.0f) {
    vtln_high += nyquist;
  }

  if (vtln_warp_factor != 1.0f &&
      (vtln_low < 0.0f || vtln_low <= low_freq || vtln_low >= high_freq ||
       vtln_high <= 0.0f || vtln_high >= high_freq || vtln_high <= vtln_low)) {
    KNF_LOG(FATAL) << "Bad values in options: vtln-low " << vtln_low
                   << " and vtln-high " << vtln_high << ", versus "
                   << "low-freq " << low_freq << " and high-freq " << high_freq;
  }

  bins_.resize(num_bins);
  center_freqs_.resize(num_bins);

  for (int32_t bin = 0; bin < num_bins; ++bin) {
    float left_mel = mel_low_freq + bin * mel_freq_delta,
          center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
          right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    if (vtln_warp_factor != 1.0f) {
      left_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, left_mel);
      center_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                   vtln_warp_factor, center_mel);
      right_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                  vtln_warp_factor, right_mel);
    }
    center_freqs_[bin] = InverseMelScale(center_mel);

    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    std::vector<float> this_bin(num_fft_bins);

    int32_t first_index = -1, last_index = -1;
    for (int32_t i = 0; i < num_fft_bins; ++i) {
      float freq = (fft_bin_width * i);  // Center frequency of this fft
                                         // bin.
      float mel = MelScale(freq);
      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel)
          weight = (mel - left_mel) / (center_mel - left_mel);
        else
          weight = (right_mel - mel) / (right_mel - center_mel);
        this_bin[i] = weight;
        if (first_index == -1) first_index = i;
        last_index = i;
      }
    }
    KNF_CHECK(first_index != -1 && last_index >= first_index &&
              "You may have set num_mel_bins too large.");

    bins_[bin].first = first_index;
    int32_t size = last_index + 1 - first_index;
    bins_[bin].second.insert(bins_[bin].second.end(),
                             this_bin.begin() + first_index,
                             this_bin.begin() + first_index + size);

    // Replicate a bug in HTK, for testing purposes.
    if (opts.htk_mode && bin == 0 && mel_low_freq != 0.0f) {
      bins_[bin].second[0] = 0.0;
    }
  }  // for (int32_t bin = 0; bin < num_bins; ++bin) {

  if (debug_) {
    std::ostringstream os;
    for (size_t i = 0; i < bins_.size(); i++) {
      os << "bin " << i << ", offset = " << bins_[i].first << ", vec = ";
      for (auto k : bins_[i].second) os << k << ", ";
      os << "\n";
    }
    KNF_LOG(INFO) << os.str();
  }
}

// "power_spectrum" contains fft energies.
void MelBanks::Compute(const float *power_spectrum,
                       float *mel_energies_out) const {
  int32_t num_bins = bins_.size();

  for (int32_t i = 0; i < num_bins; i++) {
    int32_t offset = bins_[i].first;
    const auto &v = bins_[i].second;
    float energy = 0;
    for (int32_t k = 0; k != v.size(); ++k) {
      energy += v[k] * power_spectrum[k + offset];
    }

    // HTK-like flooring- for testing purposes (we prefer dither)
    if (htk_mode_ && energy < 1.0) {
      energy = 1.0;
    }

    mel_energies_out[i] = energy;

    // The following assert was added due to a problem with OpenBlas that
    // we had at one point (it was a bug in that library).  Just to detect
    // it early.
    KNF_CHECK_EQ(energy, energy);  // check that energy is not nan
  }

  if (debug_) {
    fprintf(stderr, "MEL BANKS:\n");
    for (int32_t i = 0; i < num_bins; i++)
      fprintf(stderr, " %f", mel_energies_out[i]);
    fprintf(stderr, "\n");
  }
}

}  // namespace knf
