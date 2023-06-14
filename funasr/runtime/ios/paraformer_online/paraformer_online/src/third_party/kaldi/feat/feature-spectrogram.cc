// feat/feature-spectrogram.cc

// Copyright 2009-2012  Karel Vesely
// Copyright 2012  Navdeep Jaitly

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


#include "feat/feature-spectrogram.h"


namespace kaldi {

SpectrogramComputer::SpectrogramComputer(const SpectrogramOptions &opts)
    : opts_(opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

SpectrogramComputer::SpectrogramComputer(const SpectrogramComputer &other):
    opts_(other.opts_), log_energy_floor_(other.log_energy_floor_), srfft_(NULL) {
  if (other.srfft_ != NULL)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*other.srfft_);
}

SpectrogramComputer::~SpectrogramComputer() {
  delete srfft_;
}

void SpectrogramComputer::Compute(BaseFloat signal_log_energy,
                                  BaseFloat vtln_warp,
                                  VectorBase<BaseFloat> *signal_frame,
                                  VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());


  // Compute energy after window function (not the raw one)
  if (!opts_.raw_energy)
    signal_log_energy = Log(std::max(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<BaseFloat>::epsilon()));

  if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two
    RealFft(signal_frame, true);

  // Convert the FFT into a power spectrum.
  ComputePowerSpectrum(signal_frame);
  SubVector<BaseFloat> power_spectrum(*signal_frame,
                                      0, signal_frame->Dim() / 2 + 1);

  power_spectrum.ApplyFloor(std::numeric_limits<BaseFloat>::epsilon());
  power_spectrum.ApplyLog();

  feature->CopyFromVec(power_spectrum);

  if (opts_.energy_floor > 0.0 && signal_log_energy < log_energy_floor_)
    signal_log_energy = log_energy_floor_;
  // The zeroth spectrogram component is always set to the signal energy,
  // instead of the square of the constant component of the signal.
  (*feature)(0) = signal_log_energy;
}

}  // namespace kaldi
