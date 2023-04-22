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

// This file is copied/modified from kaldi/src/feat/feature-fbank.h

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_

#include <map>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

struct FbankOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  // append an extra dimension with energy to the filter banks
  bool use_energy = false;
  float energy_floor = 0.0f;  // active iff use_energy==true

  // If true, compute log_energy before preemphasis and windowing
  // If false, compute log_energy after preemphasis ans windowing
  bool raw_energy = true;  // active iff use_energy==true

  // If true, put energy last (if using energy)
  // If false, put energy first
  bool htk_compat = false;  // active iff use_energy==true

  // if true (default), produce log-filterbank, else linear
  bool use_log_fbank = true;

  // if true (default), use power in filterbank
  // analysis, else magnitude.
  bool use_power = true;

  FbankOptions() { mel_opts.num_bins = 23; }

  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";

    os << "mel_opts: \n";
    os << mel_opts << "\n";

    os << "use_energy: " << use_energy << "\n";
    os << "energy_floor: " << energy_floor << "\n";
    os << "raw_energy: " << raw_energy << "\n";
    os << "htk_compat: " << htk_compat << "\n";
    os << "use_log_fbank: " << use_log_fbank << "\n";
    os << "use_power: " << use_power << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const FbankOptions &opts);

class FbankComputer {
 public:
  using Options = FbankOptions;

  explicit FbankComputer(const FbankOptions &opts);
  ~FbankComputer();

  int32_t Dim() const {
    return opts_.mel_opts.num_bins + (opts_.use_energy ? 1 : 0);
  }

  // if true, compute log_energy_pre_window but after dithering and dc removal
  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const FbankOptions &GetOptions() const { return opts_; }

  /**
     Function that computes one frame of features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedsRawLogEnergy().
     @param [in] vtln_warp  The VTLN warping factor that the user wants
         to be applied when computing features for this utterance.  Will
         normally be 1.0, meaning no warping is to be done.  The value will
         be ignored for feature types that don't support VLTN, such as
         spectrogram features.
     @param [in] signal_frame  One frame of the signal,
       as extracted using the function ExtractWindow() using the options
       returned by this->GetFrameOptions().  The function will use the
       vector as a workspace, which is why it's a non-const pointer.
     @param [out] feature  Pointer to a vector of size this->Dim(), to which
         the computed feature will be written. It should be pre-allocated.
  */
  void Compute(float signal_raw_log_energy, float vtln_warp,
               std::vector<float> *signal_frame, float *feature);

 private:
  const MelBanks *GetMelBanks(float vtln_warp);

  FbankOptions opts_;
  float log_energy_floor_;
  std::map<float, MelBanks *> mel_banks_;  // float is VTLN coefficient.
  Rfft rfft_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_FBANK_H_
