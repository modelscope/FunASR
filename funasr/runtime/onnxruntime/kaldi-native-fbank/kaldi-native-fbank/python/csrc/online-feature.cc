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

#include "kaldi-native-fbank/python/csrc/online-feature.h"

#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
namespace knf {

template <typename C>
void PybindOnlineFeatureTpl(py::module &m,  // NOLINT
                            const std::string &class_name,
                            const std::string &class_help_doc = "") {
  using PyClass = OnlineGenericBaseFeature<C>;
  using Options = typename C::Options;
  py::class_<PyClass>(m, class_name.c_str(), class_help_doc.c_str())
      .def(py::init<const Options &>(), py::arg("opts"))
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("frame_shift_in_seconds",
                             &PyClass::FrameShiftInSeconds)
      .def_property_readonly("num_frames_ready", &PyClass::NumFramesReady)
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"))
      .def(
          "get_frame",
          [](py::object obj, int32_t frame) {
            auto *self = obj.cast<PyClass *>();
            const float *f = self->GetFrame(frame);
            return py::array_t<float>({self->Dim()},    // shape
                                      {sizeof(float)},  // stride in bytes
                                      f,                // ptr
                                      obj);  // it will increase the reference
                                             // count of **this** vector
          },
          py::arg("frame"))
      .def(
          "accept_waveform",
          [](PyClass &self, float sampling_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sampling_rate, waveform.data(),
                                waveform.size());
          },
          py::arg("sampling_rate"), py::arg("waveform"),
          py::call_guard<py::gil_scoped_release>())
      .def("input_finished", &PyClass::InputFinished);
}

void PybindOnlineFeature(py::module &m) {  // NOLINT
  PybindOnlineFeatureTpl<FbankComputer>(m, "OnlineFbank");
}

}  // namespace knf
