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

#include "kaldi-native-fbank/python/csrc/mel-computations.h"

#include <string>

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindMelBanksOptions(py::module &m) {  // NOLINT
  using PyClass = MelBanksOptions;
  py::class_<PyClass>(m, "MelBanksOptions")
      .def(py::init<>())
      .def_readwrite("num_bins", &PyClass::num_bins)
      .def_readwrite("low_freq", &PyClass::low_freq)
      .def_readwrite("high_freq", &PyClass::high_freq)
      .def_readwrite("vtln_low", &PyClass::vtln_low)
      .def_readwrite("vtln_high", &PyClass::vtln_high)
      .def_readwrite("debug_mel", &PyClass::debug_mel)
      .def_readwrite("htk_mode", &PyClass::htk_mode)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def("as_dict",
           [](const PyClass &self) -> py::dict { return AsDict(self); })
      .def_static("from_dict",
                  [](py::dict dict) -> PyClass {
                    return MelBanksOptionsFromDict(dict);
                  })
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return MelBanksOptionsFromDict(dict);
          }));
}

void PybindMelComputations(py::module &m) {  // NOLINT
  PybindMelBanksOptions(m);
}

}  // namespace knf
