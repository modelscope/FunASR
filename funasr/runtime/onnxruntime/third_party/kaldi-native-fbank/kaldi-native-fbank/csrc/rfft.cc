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

#include "kaldi-native-fbank/csrc/rfft.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kaldi-native-fbank/csrc/log.h"

// see fftsg.c
#ifdef __cplusplus
extern "C" void rdft(int n, int isgn, double *a, int *ip, double *w);
#else
void rdft(int n, int isgn, double *a, int *ip, double *w);
#endif

namespace knf {
class Rfft::RfftImpl {
 public:
  explicit RfftImpl(int32_t n) : n_(n), ip_(2 + std::sqrt(n / 2)), w_(n / 2) {
    KNF_CHECK_EQ(n & (n - 1), 0);
  }

  void Compute(float *in_out) {
    std::vector<double> d(in_out, in_out + n_);

    Compute(d.data());

    std::copy(d.begin(), d.end(), in_out);
  }

  void Compute(double *in_out) {
    // 1 means forward fft
    rdft(n_, 1, in_out, ip_.data(), w_.data());
  }

 private:
  int32_t n_;
  std::vector<int32_t> ip_;
  std::vector<double> w_;
};

Rfft::Rfft(int32_t n) : impl_(std::make_unique<RfftImpl>(n)) {}

Rfft::~Rfft() = default;

void Rfft::Compute(float *in_out) { impl_->Compute(in_out); }
void Rfft::Compute(double *in_out) { impl_->Compute(in_out); }

}  // namespace knf
