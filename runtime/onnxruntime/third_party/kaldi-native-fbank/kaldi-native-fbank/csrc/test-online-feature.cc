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

#include "gtest/gtest.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
namespace knf {

TEST(RecyclingVector, TestUnlimited) {
  RecyclingVector v(-1);
  constexpr int32_t N = 100;
  for (int32_t i = 0; i != N; ++i) {
    std::unique_ptr<float[]> p(new float[3]{i, i + 1, i + 2});
    v.PushBack(std::move(p));
  }
  ASSERT_EQ(v.Size(), N);

  for (int32_t i = 0; i != N; ++i) {
    const float *t = v.At(i);
    for (int32_t k = 0; k != 3; ++k) {
      EXPECT_EQ(t[k], (i + k));
    }
  }
}

TEST(RecyclingVector, Testlimited) {
  constexpr int32_t K = 3;
  constexpr int32_t N = 10;
  RecyclingVector v(K);
  for (int32_t i = 0; i != N; ++i) {
    std::unique_ptr<float[]> p(new float[3]{i, i + 1, i + 2});
    v.PushBack(std::move(p));
  }

  ASSERT_EQ(v.Size(), N);

  for (int32_t i = N - K; i != N; ++i) {
    const float *t = v.At(i);

    for (int32_t k = 0; k != 3; ++k) {
      EXPECT_EQ(t[k], (i + k));
    }
  }
}
}  // namespace knf
