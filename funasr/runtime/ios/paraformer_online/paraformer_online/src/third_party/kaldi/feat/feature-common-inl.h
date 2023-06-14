// feat/feature-common-inl.h

// Copyright       2016  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FEAT_FEATURE_COMMON_INL_H_
#define KALDI_FEAT_FEATURE_COMMON_INL_H_

#include "feat/resample.h"
// Do not include this file directly.  It is included by feat/feature-common.h

namespace kaldi {

template <class F>
void OfflineFeatureTpl<F>::ComputeFeatures(
    const VectorBase<BaseFloat> &wave,
    BaseFloat sample_freq,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  BaseFloat new_sample_freq = computer_.GetFrameOptions().samp_freq;
  if (sample_freq == new_sample_freq)
    Compute(wave, vtln_warp, output);
  else {
    if (new_sample_freq < sample_freq) {
      if (! computer_.GetFrameOptions().allow_downsample)
        KALDI_ERR << "Waveform and config sample Frequency mismatch: "
                  << sample_freq << " .vs " << new_sample_freq
                  << " ( use --allow_downsample=true option to allow "
                  << " downsampling the waveform).";

      // Downsample the waveform.
      Vector<BaseFloat> downsampled_wave(wave);
      DownsampleWaveForm(sample_freq, wave,
                         new_sample_freq, &downsampled_wave);
      Compute(downsampled_wave, vtln_warp, output);
    } else
      KALDI_ERR << "The waveform is allowed to get downsampled."
                << "New sample Frequency " << new_sample_freq
                << " is larger than waveform original sampling frequency "
                << sample_freq;

  }
}

template <class F>
void OfflineFeatureTpl<F>::ComputeFeatures(
    const VectorBase<BaseFloat> &wave,
    BaseFloat sample_freq,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) const {
  OfflineFeatureTpl<F> temp(*this);
  // This const version of ComputeFeatures() is a wrapper that
  // calls the non-const ComputeFeatures() on a temporary object
  // that is a copy of *this.  It is not as efficient because of the
  // overhead of copying *this.
  temp.ComputeFeatures(wave, vtln_warp, output);
}

template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), computer_.GetFrameOptions()),
      cols_out = computer_.Dim();
  if (rows_out == 0) {
    output->Resize(0, 0);
    return;
  }
  output->Resize(rows_out, cols_out);
  Vector<BaseFloat> window;  // windowed waveform.
  bool use_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                  feature_window_function_, &window,
                  (use_raw_log_energy ? &raw_log_energy : NULL));

    SubVector<BaseFloat> output_row(*output, r);
    computer_.Compute(raw_log_energy, vtln_warp, &window, &output_row);
  }
}

template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) const {
  OfflineFeatureTpl<F> temp(*this);
  // call the non-const version of Compute() on a temporary copy of this object.
  // This is a workaround for const-ness that may sometimes be useful in
  // multi-threaded code, although it's not optimally efficient.
  temp.Compute(wave, vtln_warp, output);
}

} // end namespace kaldi

#endif
