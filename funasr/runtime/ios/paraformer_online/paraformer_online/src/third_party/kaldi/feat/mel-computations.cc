// feat/mel-computations.cc

// Copyright 2009-2011  Phonexia s.r.o.;  Karel Vesely;  Microsoft Corporation

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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <iostream>

#include "feat/feature-functions.h"
#include "feat/feature-window.h"
#include "feat/mel-computations.h"

namespace kaldi {


MelBanks::MelBanks(const MelBanksOptions &opts,
                   const FrameExtractionOptions &frame_opts,
                   BaseFloat vtln_warp_factor):
    htk_mode_(opts.htk_mode) {
  int32 num_bins = opts.num_bins;
  if (num_bins < 3) KALDI_ERR << "Must have at least 3 mel bins";
  BaseFloat sample_freq = frame_opts.samp_freq;
  int32 window_length_padded = frame_opts.PaddedWindowSize();
  KALDI_ASSERT(window_length_padded % 2 == 0);
  int32 num_fft_bins = window_length_padded / 2;
  BaseFloat nyquist = 0.5 * sample_freq;

  BaseFloat low_freq = opts.low_freq, high_freq;
  if (opts.high_freq > 0.0)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  if (low_freq < 0.0 || low_freq >= nyquist
      || high_freq <= 0.0 || high_freq > nyquist
      || high_freq <= low_freq)
    KALDI_ERR << "Bad values in options: low-freq " << low_freq
              << " and high-freq " << high_freq << " vs. nyquist "
              << nyquist;

  BaseFloat fft_bin_width = sample_freq / window_length_padded;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]

  BaseFloat mel_low_freq = MelScale(low_freq);
  BaseFloat mel_high_freq = MelScale(high_freq);

  debug_ = opts.debug_mel;

  // divide by num_bins+1 in next line because of end-effects where the bins
  // spread out to the sides.
  BaseFloat mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins+1);

  BaseFloat vtln_low = opts.vtln_low,
      vtln_high = opts.vtln_high;
  if (vtln_high < 0.0) {
    vtln_high += nyquist;
  }

  if (vtln_warp_factor != 1.0 &&
      (vtln_low < 0.0 || vtln_low <= low_freq
       || vtln_low >= high_freq
       || vtln_high <= 0.0 || vtln_high >= high_freq
       || vtln_high <= vtln_low))
    KALDI_ERR << "Bad values in options: vtln-low " << vtln_low
              << " and vtln-high " << vtln_high << ", versus "
              << "low-freq " << low_freq << " and high-freq "
              << high_freq;

  bins_.resize(num_bins);
  center_freqs_.Resize(num_bins);

  for (int32 bin = 0; bin < num_bins; bin++) {
    BaseFloat left_mel = mel_low_freq + bin * mel_freq_delta,
        center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
        right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    if (vtln_warp_factor != 1.0) {
      left_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, left_mel);
      center_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, center_mel);
      right_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                  vtln_warp_factor, right_mel);
    }
    center_freqs_(bin) = InverseMelScale(center_mel);
    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    Vector<BaseFloat> this_bin(num_fft_bins);
    int32 first_index = -1, last_index = -1;
    for (int32 i = 0; i < num_fft_bins; i++) {
      BaseFloat freq = (fft_bin_width * i);  // Center frequency of this fft
                                             // bin.
      BaseFloat mel = MelScale(freq);
      if (mel > left_mel && mel < right_mel) {
        BaseFloat weight;
        if (mel <= center_mel)
          weight = (mel - left_mel) / (center_mel - left_mel);
        else
         weight = (right_mel-mel) / (right_mel-center_mel);
        this_bin(i) = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }
    KALDI_ASSERT(first_index != -1 && last_index >= first_index
                 && "You may have set --num-mel-bins too large.");

    bins_[bin].first = first_index;
    int32 size = last_index + 1 - first_index;
    bins_[bin].second.Resize(size);
    bins_[bin].second.CopyFromVec(this_bin.Range(first_index, size));

    // Replicate a bug in HTK, for testing purposes.
    if (opts.htk_mode && bin == 0 && mel_low_freq != 0.0)
      bins_[bin].second(0) = 0.0;

  }
  if (debug_) {
    for (size_t i = 0; i < bins_.size(); i++) {
      KALDI_LOG << "bin " << i << ", offset = " << bins_[i].first
                << ", vec = " << bins_[i].second;
    }
  }
}

MelBanks::MelBanks(const MelBanks &other):
    center_freqs_(other.center_freqs_),
    bins_(other.bins_),
    debug_(other.debug_),
    htk_mode_(other.htk_mode_) { }

BaseFloat MelBanks::VtlnWarpFreq(BaseFloat vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
                                 BaseFloat vtln_high_cutoff,
                                 BaseFloat low_freq,  // upper+lower frequency cutoffs in mel computation
                                 BaseFloat high_freq,
                                 BaseFloat vtln_warp_factor,
                                 BaseFloat freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq and
  /// high_freq inclusive, with the following properties:
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


  if (freq < low_freq || freq > high_freq) return freq;  // in case this gets called
  // for out-of-range frequencies, just return the freq.

  KALDI_ASSERT(vtln_low_cutoff > low_freq &&
               "be sure to set the --vtln-low option higher than --low-freq");
  KALDI_ASSERT(vtln_high_cutoff < high_freq &&
               "be sure to set the --vtln-high option lower than --high-freq [or negative]");
  BaseFloat one = 1.0;
  BaseFloat l = vtln_low_cutoff * std::max(one, vtln_warp_factor);
  BaseFloat h = vtln_high_cutoff * std::min(one, vtln_warp_factor);
  BaseFloat scale = 1.0 / vtln_warp_factor;
  BaseFloat Fl = scale * l;  // F(l);
  BaseFloat Fh = scale * h;  // F(h);
  KALDI_ASSERT(l > low_freq && h < high_freq);
  // slope of left part of the 3-piece linear function
  BaseFloat scale_left = (Fl - low_freq) / (l - low_freq);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  BaseFloat scale_right = (high_freq - Fh) / (high_freq - h);

  if (freq < l) {
    return low_freq + scale_left * (freq - low_freq);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq + scale_right * (freq - high_freq);
  }
}

BaseFloat MelBanks::VtlnWarpMelFreq(BaseFloat vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
                                    BaseFloat vtln_high_cutoff,
                                    BaseFloat low_freq,  // upper+lower frequency cutoffs in mel computation
                                    BaseFloat high_freq,
                                    BaseFloat vtln_warp_factor,
                                    BaseFloat mel_freq) {
  return MelScale(VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                               low_freq, high_freq,
                               vtln_warp_factor, InverseMelScale(mel_freq)));
}


// "power_spectrum" contains fft energies.
void MelBanks::Compute(const VectorBase<BaseFloat> &power_spectrum,
                       VectorBase<BaseFloat> *mel_energies_out) const {
  int32 num_bins = bins_.size();
  KALDI_ASSERT(mel_energies_out->Dim() == num_bins);

  for (int32 i = 0; i < num_bins; i++) {
    int32 offset = bins_[i].first;
    const Vector<BaseFloat> &v(bins_[i].second);
    BaseFloat energy = VecVec(v, power_spectrum.Range(offset, v.Dim()));
    // HTK-like flooring- for testing purposes (we prefer dither)
    if (htk_mode_ && energy < 1.0) energy = 1.0;
    (*mel_energies_out)(i) = energy;

    // The following assert was added due to a problem with OpenBlas that
    // we had at one point (it was a bug in that library).  Just to detect
    // it early.
    KALDI_ASSERT(!KALDI_ISNAN((*mel_energies_out)(i)));
  }

  if (debug_) {
    fprintf(stderr, "MEL BANKS:\n");
    for (int32 i = 0; i < num_bins; i++)
      fprintf(stderr, " %f", (*mel_energies_out)(i));
    fprintf(stderr, "\n");
  }
}

void ComputeLifterCoeffs(BaseFloat Q, VectorBase<BaseFloat> *coeffs) {
  // Compute liftering coefficients (scaling on cepstral coeffs)
  // coeffs are numbered slightly differently from HTK: the zeroth
  // index is C0, which is not affected.
  for (int32 i = 0; i < coeffs->Dim(); i++)
    (*coeffs)(i) = 1.0 + 0.5 * Q * sin (M_PI * i / Q);
}


// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the demoninator
BaseFloat Durbin(int n, const BaseFloat *pAC, BaseFloat *pLP, BaseFloat *pTmp) {
  BaseFloat ki;                // reflection coefficient
  int i;
  int j;

  BaseFloat E = pAC[0];

  for (i = 0; i < n; i++) {
    // next reflection coefficient
    ki = pAC[i + 1];
    for (j = 0; j < i; j++)
      ki += pLP[j] * pAC[i - j];
    ki = ki / E;

    // new error
    BaseFloat c = 1 - ki * ki;
    if (c < 1.0e-5) // remove NaNs for constan signal
      c = 1.0e-5;
    E *= c;

    // new LP coefficients
    pTmp[i] = -ki;
    for (j = 0; j < i; j++)
      pTmp[j] = pLP[j] - ki * pLP[i - j - 1];

    for (j = 0; j <= i; j++)
      pLP[j] = pTmp[j];
  }

  return E;
}


void Lpc2Cepstrum(int n, const BaseFloat *pLPC, BaseFloat *pCepst) {
  for (int32 i = 0; i < n; i++) {
    double sum = 0.0;
    int j;
    for (j = 0; j < i; j++) {
      sum += static_cast<BaseFloat>(i - j) * pLPC[j] * pCepst[i - j - 1];
    }
    pCepst[i] = -pLPC[i] - sum / static_cast<BaseFloat>(i + 1);
  }
}

void GetEqualLoudnessVector(const MelBanks &mel_banks,
                            Vector<BaseFloat> *ans) {
  int32 n = mel_banks.NumBins();
  // Central frequency of each mel bin.
  const Vector<BaseFloat> &f0 = mel_banks.GetCenterFreqs();
  ans->Resize(n);
  for (int32 i = 0; i < n; i++) {
    BaseFloat fsq = f0(i) * f0(i);
    BaseFloat fsub = fsq / (fsq + 1.6e5);
    (*ans)(i) = fsub * fsub * ((fsq + 1.44e6) / (fsq + 9.61e6));
  }
}


// Compute LP coefficients from autocorrelation coefficients.
BaseFloat ComputeLpc(const VectorBase<BaseFloat> &autocorr_in,
                     Vector<BaseFloat> *lpc_out) {
  int32 n = autocorr_in.Dim() - 1;
  KALDI_ASSERT(lpc_out->Dim() == n);
  Vector<BaseFloat> tmp(n);
  BaseFloat ans = Durbin(n, autocorr_in.Data(),
                         lpc_out->Data(),
                         tmp.Data());
  if (ans <= 0.0)
    KALDI_WARN << "Zero energy in LPC computation";
  return -Log(1.0 / ans);  // forms the C0 value
}


}  // namespace kaldi
