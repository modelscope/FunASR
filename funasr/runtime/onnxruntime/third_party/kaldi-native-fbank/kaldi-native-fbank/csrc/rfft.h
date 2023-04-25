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

#ifndef KALDI_NATIVE_FBANK_CSRC_RFFT_H_
#define KALDI_NATIVE_FBANK_CSRC_RFFT_H_

#include <memory>

namespace knf {

// n-point Real discrete Fourier transform
// where n is a power of 2. n >= 2
//
//  R[k] = sum_j=0^n-1 in[j]*cos(2*pi*j*k/n), 0<=k<=n/2
//  I[k] = sum_j=0^n-1 in[j]*sin(2*pi*j*k/n), 0<k<n/2
class Rfft {
 public:
  // @param n Number of fft bins. it should be a power of 2.
  explicit Rfft(int32_t n);
  ~Rfft();

  /** @param in_out A 1-D array of size n.
   *             On return:
   *               in_out[0] = R[0]
   *               in_out[1] = R[n/2]
   *               for 1 < k < n/2,
   *                 in_out[2*k] = R[k]
   *                 in_out[2*k+1] = I[k]
   *
   */
  void Compute(float *in_out);
  void Compute(double *in_out);

 private:
  class RfftImpl;
  std::unique_ptr<RfftImpl> impl_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_RFFT_H_
