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

#include <iostream>

#include "kaldi-native-fbank/csrc/online-feature.h"

int main() {
  knf::FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.mel_opts.num_bins = 10;

  knf::OnlineFbank fbank(opts);
  for (int32_t i = 0; i < 1600; ++i) {
    float s = (i * i - i / 2) / 32767.;
    fbank.AcceptWaveform(16000, &s, 1);
  }

  std::ostringstream os;

  int32_t n = fbank.NumFramesReady();
  for (int32_t i = 0; i != n; ++i) {
    const float *frame = fbank.GetFrame(i);
    for (int32_t k = 0; k != opts.mel_opts.num_bins; ++k) {
      os << frame[k] << ", ";
    }
    os << "\n";
  }

  std::cout << os.str() << "\n";

  return 0;
}
