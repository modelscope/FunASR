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

// This file is copied/modified from kaldi/src/feat/feature-functions.cc

#include "kaldi-native-fbank/csrc/feature-functions.h"

#include <cstdint>
#include <vector>

namespace knf {

void ComputePowerSpectrum(std::vector<float> *complex_fft) {
  int32_t dim = complex_fft->size();

  // now we have in complex_fft, first half of complex spectrum
  // it's stored as [real0, realN/2, real1, im1, real2, im2, ...]

  float *p = complex_fft->data();
  int32_t half_dim = dim / 2;
  float first_energy = p[0] * p[0];
  float last_energy = p[1] * p[1];  // handle this special case

  for (int32_t i = 1; i < half_dim; ++i) {
    float real = p[i * 2];
    float im = p[i * 2 + 1];
    p[i] = real * real + im * im;
  }
  p[0] = first_energy;
  p[half_dim] = last_energy;  // Will actually never be used, and anyway
  // if the signal has been bandlimited sensibly this should be zero.
}

}  // namespace knf
