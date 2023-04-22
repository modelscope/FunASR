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

#ifndef KALDI_NATIVE_FBANK_PYTHON_CSRC_ONLINE_FEATURE_H_
#define KALDI_NATIVE_FBANK_PYTHON_CSRC_ONLINE_FEATURE_H_

#include "kaldi-native-fbank/python/csrc/kaldi-native-fbank.h"

namespace knf {

void PybindOnlineFeature(py::module &m);  // NOLINT

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_PYTHON_CSRC_ONLINE_FEATURE_H_
