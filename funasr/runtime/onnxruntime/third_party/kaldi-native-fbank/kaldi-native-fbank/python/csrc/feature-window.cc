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

#include "kaldi-native-fbank/python/csrc/feature-window.h"

#include <string>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/python/csrc/utils.h"

namespace knf {

static void PybindFrameExtractionOptions(py::module &m) {  // NOLINT
  using PyClass = FrameExtractionOptions;
  py::class_<PyClass>(m, "FrameExtractionOptions")
      .def(py::init<>())
      .def_readwrite("samp_freq", &PyClass::samp_freq)
      .def_readwrite("frame_shift_ms", &PyClass::frame_shift_ms)
      .def_readwrite("frame_length_ms", &PyClass::frame_length_ms)
      .def_readwrite("dither", &PyClass::dither)
      .def_readwrite("preemph_coeff", &PyClass::preemph_coeff)
      .def_readwrite("remove_dc_offset", &PyClass::remove_dc_offset)
      .def_readwrite("window_type", &PyClass::window_type)
      .def_readwrite("round_to_power_of_two", &PyClass::round_to_power_of_two)
      .def_readwrite("blackman_coeff", &PyClass::blackman_coeff)
      .def_readwrite("snip_edges", &PyClass::snip_edges)
      .def_readwrite("max_feature_vectors", &PyClass::max_feature_vectors)
      .def("as_dict",
           [](const PyClass &self) -> py::dict { return AsDict(self); })
      .def_static("from_dict",
                  [](py::dict dict) -> PyClass {
                    return FrameExtractionOptionsFromDict(dict);
                  })
#if 0
      .def_readwrite("allow_downsample",
                     &PyClass::allow_downsample)
      .def_readwrite("allow_upsample", &PyClass::allow_upsample)
#endif
      .def("__str__",
           [](const PyClass &self) -> std::string { return self.ToString(); })
      .def(py::pickle(
          [](const PyClass &self) -> py::dict { return AsDict(self); },
          [](py::dict dict) -> PyClass {
            return FrameExtractionOptionsFromDict(dict);
          }));
}

void PybindFeatureWindow(py::module &m) {  // NOLINT
  PybindFrameExtractionOptions(m);
}

}  // namespace knf
