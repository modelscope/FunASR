/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

#if 0
>>> import torch
>>> a = torch.tensor([1., -1, 3, 8, 20, 6, 0, 2])
>>> torch.fft.rfft(a)
tensor([ 39.0000+0.0000j, -28.1924-2.2929j,  18.0000+5.0000j,  -9.8076+3.7071j,
          9.0000+0.0000j])
#endif

TEST(Rfft, TestRfft) {
  knf::Rfft fft(8);
  for (int32_t i = 0; i != 10; ++i) {
    std::vector<float> d = {1, -1, 3, 8, 20, 6, 0, 2};
    fft.Compute(d.data());

    EXPECT_EQ(d[0], 39);
    EXPECT_EQ(d[1], 9);

    EXPECT_NEAR(d[2], -28.1924, 1e-3);
    EXPECT_NEAR(-d[3], -2.2929, 1e-3);

    EXPECT_NEAR(d[4], 18, 1e-3);
    EXPECT_NEAR(-d[5], 5, 1e-3);

    EXPECT_NEAR(d[6], -9.8076, 1e-3);
    EXPECT_NEAR(-d[7], 3.7071, 1e-3);
  }
}

}  // namespace knf
