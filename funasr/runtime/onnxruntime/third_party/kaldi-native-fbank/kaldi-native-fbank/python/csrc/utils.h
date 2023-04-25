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

#ifndef KALDI_NATIVE_FBANK_PYTHON_CSRC_UTILS_H_
#define KALDI_NATIVE_FBANK_PYTHON_CSRC_UTILS_H_

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/python/csrc/kaldi-native-fbank.h"

/*
 * This file contains code about `from_dict` and
 * `as_dict` for various options in kaldi-native-fbank.
 *
 * Regarding `from_dict`, users don't need to provide
 * all the fields in the options. If some fields
 * are not provided, it just uses the default one.
 *
 * If the provided dict in `from_dict` is empty,
 * all fields use their default values.
 */

namespace knf {

FrameExtractionOptions FrameExtractionOptionsFromDict(py::dict dict);
py::dict AsDict(const FrameExtractionOptions &opts);

MelBanksOptions MelBanksOptionsFromDict(py::dict dict);
py::dict AsDict(const MelBanksOptions &opts);

FbankOptions FbankOptionsFromDict(py::dict dict);
py::dict AsDict(const FbankOptions &opts);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_PYTHON_CSRC_UTILS_H_
