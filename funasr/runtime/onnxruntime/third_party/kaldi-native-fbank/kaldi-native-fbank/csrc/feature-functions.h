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

// This file is copied/modified from kaldi/src/feat/feature-functions.h
#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_FUNCTIONS_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_FUNCTIONS_H_

#include <vector>
namespace knf {

// ComputePowerSpectrum converts a complex FFT (as produced by the FFT
// functions in csrc/rfft.h), and converts it into
// a power spectrum.  If the complex FFT is a vector of size n (representing
// half of the complex FFT of a real signal of size n, as described there),
// this function computes in the first (n/2) + 1 elements of it, the
// energies of the fft bins from zero to the Nyquist frequency.  Contents of the
// remaining (n/2) - 1 elements are undefined at output.

void ComputePowerSpectrum(std::vector<float> *complex_fft);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_FUNCTIONS_H_
