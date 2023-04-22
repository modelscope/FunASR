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

#include "kaldi-native-fbank/python/csrc/utils.h"

#include <string>

#include "kaldi-native-fbank/csrc/feature-window.h"

#define FROM_DICT(type, key)         \
  if (dict.contains(#key)) {         \
    opts.key = py::type(dict[#key]); \
  }

#define AS_DICT(key) dict[#key] = opts.key

namespace knf {

FrameExtractionOptions FrameExtractionOptionsFromDict(py::dict dict) {
  FrameExtractionOptions opts;

  FROM_DICT(float_, samp_freq);
  FROM_DICT(float_, frame_shift_ms);
  FROM_DICT(float_, frame_length_ms);
  FROM_DICT(float_, dither);
  FROM_DICT(float_, preemph_coeff);
  FROM_DICT(bool_, remove_dc_offset);
  FROM_DICT(str, window_type);
  FROM_DICT(bool_, round_to_power_of_two);
  FROM_DICT(float_, blackman_coeff);
  FROM_DICT(bool_, snip_edges);
  FROM_DICT(int_, max_feature_vectors);

  return opts;
}

py::dict AsDict(const FrameExtractionOptions &opts) {
  py::dict dict;

  AS_DICT(samp_freq);
  AS_DICT(frame_shift_ms);
  AS_DICT(frame_length_ms);
  AS_DICT(dither);
  AS_DICT(preemph_coeff);
  AS_DICT(remove_dc_offset);
  AS_DICT(window_type);
  AS_DICT(round_to_power_of_two);
  AS_DICT(blackman_coeff);
  AS_DICT(snip_edges);
  AS_DICT(max_feature_vectors);

  return dict;
}

MelBanksOptions MelBanksOptionsFromDict(py::dict dict) {
  MelBanksOptions opts;

  FROM_DICT(int_, num_bins);
  FROM_DICT(float_, low_freq);
  FROM_DICT(float_, high_freq);
  FROM_DICT(float_, vtln_low);
  FROM_DICT(float_, vtln_high);
  FROM_DICT(bool_, debug_mel);
  FROM_DICT(bool_, htk_mode);

  return opts;
}
py::dict AsDict(const MelBanksOptions &opts) {
  py::dict dict;

  AS_DICT(num_bins);
  AS_DICT(low_freq);
  AS_DICT(high_freq);
  AS_DICT(vtln_low);
  AS_DICT(vtln_high);
  AS_DICT(debug_mel);
  AS_DICT(htk_mode);

  return dict;
}

FbankOptions FbankOptionsFromDict(py::dict dict) {
  FbankOptions opts;

  if (dict.contains("frame_opts")) {
    opts.frame_opts = FrameExtractionOptionsFromDict(dict["frame_opts"]);
  }

  if (dict.contains("mel_opts")) {
    opts.mel_opts = MelBanksOptionsFromDict(dict["mel_opts"]);
  }

  FROM_DICT(bool_, use_energy);
  FROM_DICT(float_, energy_floor);
  FROM_DICT(bool_, raw_energy);
  FROM_DICT(bool_, htk_compat);
  FROM_DICT(bool_, use_log_fbank);
  FROM_DICT(bool_, use_power);

  return opts;
}

py::dict AsDict(const FbankOptions &opts) {
  py::dict dict;

  dict["frame_opts"] = AsDict(opts.frame_opts);
  dict["mel_opts"] = AsDict(opts.mel_opts);
  AS_DICT(use_energy);
  AS_DICT(energy_floor);
  AS_DICT(raw_energy);
  AS_DICT(htk_compat);
  AS_DICT(use_log_fbank);
  AS_DICT(use_power);

  return dict;
}

#undef FROM_DICT
#undef AS_DICT

}  // namespace knf
