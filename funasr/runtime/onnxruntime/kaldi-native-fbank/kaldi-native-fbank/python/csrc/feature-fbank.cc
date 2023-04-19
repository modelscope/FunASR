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

#include "kaldi-native-fbank/python/csrc/feature-fbank.h"

#include <memory>
#include <string>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindFbankOptions(py::module &m) {  // NOLINT
  using PyClass = FbankOptions;
  py::class_<PyClass>(m, "FbankOptions")
      .def(py::init<>())
      .def_readwrite("frame_opts", &PyClass::frame_opts)
      .def_readwrite("mel_opts", &PyClass::mel_opts)
      .def_readwrite("use_energy", &PyClass::use_energy)
      .def_readwrite("energy_floor", &PyClass::energy_floor)
      .def_readwrite("raw_energy", &PyClass::raw_energy)
      .def_readwrite("htk_compat", &PyClass::htk_compat)
      .def_readwrite("use_log_fbank", &PyClass::use_log_fbank)
      .def_readwrite("use_power", &PyClass::use_power)
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def("as_dict",
           [](const PyClass &self) -> py::dict { return AsDict(self); })
      .def_static(
          "from_dict",
          [](py::dict dict) -> PyClass { return FbankOptionsFromDict(dict); })
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass { return FbankOptionsFromDict(dict); }));
}

void PybindFeatureFbank(py::module &m) {  // NOLINT
  PybindFbankOptions(m);
}

}  // namespace knf
