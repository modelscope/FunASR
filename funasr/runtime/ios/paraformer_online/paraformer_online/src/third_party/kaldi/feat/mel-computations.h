// feat/mel-computations.h

// Copyright 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
//                2016  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FEAT_MEL_COMPUTATIONS_H_
#define KALDI_FEAT_MEL_COMPUTATIONS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"


namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

struct FrameExtractionOptions;  // defined in feature-window.h


struct MelBanksOptions {
  int32 num_bins;  // e.g. 25; number of triangular bins
  BaseFloat low_freq;  // e.g. 20; lower frequency cutoff
  BaseFloat high_freq;  // an upper frequency cutoff; 0 -> no cutoff, negative
  // ->added to the Nyquist frequency to get the cutoff.
  BaseFloat vtln_low;  // vtln lower cutoff of warping function.
  BaseFloat vtln_high;  // vtln upper cutoff of warping function: if negative, added
                        // to the Nyquist frequency to get the cutoff.
  bool debug_mel;
  // htk_mode is a "hidden" config, it does not show up on command line.
  // Enables more exact compatibibility with HTK, for testing purposes.  Affects
  // mel-energy flooring and reproduces a bug in HTK.
  bool htk_mode;
  explicit MelBanksOptions(int num_bins = 25)
      : num_bins(num_bins), low_freq(20), high_freq(0), vtln_low(100),
        vtln_high(-500), debug_mel(false), htk_mode(false) {}

  void Register(OptionsItf *opts) {
    opts->Register("num-mel-bins", &num_bins,
                   "Number of triangular mel-frequency bins");
    opts->Register("low-freq", &low_freq,
                   "Low cutoff frequency for mel bins");
    opts->Register("high-freq", &high_freq,
                   "High cutoff frequency for mel bins (if < 0, offset from Nyquist)");
    opts->Register("vtln-low", &vtln_low,
                   "Low inflection point in piecewise linear VTLN warping function");
    opts->Register("vtln-high", &vtln_high,
                   "High inflection point in piecewise linear VTLN warping function"
                   " (if negative, offset from high-mel-freq");
    opts->Register("debug-mel", &debug_mel,
                   "Print out debugging information for mel bin computation");
  }
};


class MelBanks {
 public:

  static inline BaseFloat InverseMelScale(BaseFloat mel_freq) {
    return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
  }

  static inline BaseFloat MelScale(BaseFloat freq) {
    return 1127.0f * logf (1.0f + freq / 700.0f);
  }

  static BaseFloat VtlnWarpFreq(BaseFloat vtln_low_cutoff,
                                BaseFloat vtln_high_cutoff,  // discontinuities in warp func
                                BaseFloat low_freq,
                                BaseFloat high_freq,  // upper+lower frequency cutoffs in
                                // the mel computation
                                BaseFloat vtln_warp_factor,
                                BaseFloat freq);

  static BaseFloat VtlnWarpMelFreq(BaseFloat vtln_low_cutoff,
                                   BaseFloat vtln_high_cutoff,
                                   BaseFloat low_freq,
                                   BaseFloat high_freq,
                                   BaseFloat vtln_warp_factor,
                                   BaseFloat mel_freq);


  MelBanks(const MelBanksOptions &opts,
           const FrameExtractionOptions &frame_opts,
           BaseFloat vtln_warp_factor);

  /// Compute Mel energies (note: not log enerties).
  /// At input, "fft_energies" contains the FFT energies (not log).
  void Compute(const VectorBase<BaseFloat> &fft_energies,
               VectorBase<BaseFloat> *mel_energies_out) const;

  int32 NumBins() const { return bins_.size(); }

  // returns vector of central freq of each bin; needed by plp code.
  const Vector<BaseFloat> &GetCenterFreqs() const { return center_freqs_; }

  // Copy constructor
  MelBanks(const MelBanks &other);
 private:
  // Disallow assignment
  MelBanks &operator = (const MelBanks &other);

  // center frequencies of bins, numbered from 0 ... num_bins-1.
  // Needed by GetCenterFreqs().
  Vector<BaseFloat> center_freqs_;

  // the "bins_" vector is a vector, one for each bin, of a pair:
  // (the first nonzero fft-bin), (the vector of weights).
  std::vector<std::pair<int32, Vector<BaseFloat> > > bins_;

  bool debug_;
  bool htk_mode_;
};


// Compute liftering coefficients (scaling on cepstral coeffs)
// coeffs are numbered slightly differently from HTK: the zeroth
// index is C0, which is not affected.
void ComputeLifterCoeffs(BaseFloat Q, VectorBase<BaseFloat> *coeffs);


// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the demoninator
// Returns log energy of residual (I think)
BaseFloat Durbin(int n, const BaseFloat *pAC, BaseFloat *pLP, BaseFloat *pTmp);

// Compute LP coefficients from autocorrelation coefficients.
// Returns log energy of residual (I think)
BaseFloat ComputeLpc(const VectorBase<BaseFloat> &autocorr_in,
                     Vector<BaseFloat> *lpc_out);

void Lpc2Cepstrum(int n, const BaseFloat *pLPC, BaseFloat *pCepst);



void GetEqualLoudnessVector(const MelBanks &mel_banks,
                            Vector<BaseFloat> *ans);

/// @} End of "addtogroup feat"
}  // namespace kaldi

#endif  // KALDI_FEAT_MEL_COMPUTATIONS_H_
