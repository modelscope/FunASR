// feat/feature-fbank.h

// Copyright 2009-2012  Karel Vesely
//                2016  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_FEAT_FEATURE_FBANK_H_
#define KALDI_FEAT_FEATURE_FBANK_H_

#include <map>
#include <string>

#include "feat/feature-common.h"
#include "feat/feature-functions.h"
#include "feat/feature-window.h"
#include "feat/mel-computations.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


/// FbankOptions contains basic options for computing filterbank features.
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
struct FbankOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;
  bool use_energy;  // append an extra dimension with energy to the filter banks
  BaseFloat energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing
  bool htk_compat;  // If true, put energy last (if using energy)
  bool use_log_fbank;  // if true (default), produce log-filterbank, else linear
  bool use_power;  // if true (default), use power in filterbank analysis, else magnitude.

  FbankOptions(): mel_opts(23),
                 // defaults the #mel-banks to 23 for the FBANK computations.
                 // this seems to be common for 16khz-sampled data,
                 // but for 8khz-sampled data, 15 may be better.
                 use_energy(false),
                 energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
                 raw_energy(true),
                 htk_compat(false),
                 use_log_fbank(true),
                 use_power(true) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    mel_opts.Register(opts);
    opts->Register("use-energy", &use_energy,
                   "Add an extra dimension with energy to the FBANK output.");
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in FBANK computation");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
    opts->Register("htk-compat", &htk_compat, "If true, put energy last.  "
                   "Warning: not sufficient to get HTK compatible features (need "
                   "to change other parameters).");
    opts->Register("use-log-fbank", &use_log_fbank,
                   "If true, produce log-filterbank, else produce linear.");
    opts->Register("use-power", &use_power,
                   "If true, use power, else use magnitude.");
  }
};


/// Class for computing mel-filterbank features; see \ref feat_mfcc for more
/// information.
class FbankComputer {
 public:
  typedef FbankOptions Options;

  explicit FbankComputer(const FbankOptions &opts);
  FbankComputer(const FbankComputer &other);

  int32 Dim() const {
    return opts_.mel_opts.num_bins + (opts_.use_energy ? 1 : 0);
  }

  bool NeedRawLogEnergy() { return opts_.use_energy && opts_.raw_energy; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

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
         the computed feature will be written.
  */
  void Compute(BaseFloat signal_log_energy,
               BaseFloat vtln_warp,
               VectorBase<BaseFloat> *signal_frame,
               VectorBase<BaseFloat> *feature);

  ~FbankComputer();

 private:
  const MelBanks *GetMelBanks(BaseFloat vtln_warp);


  FbankOptions opts_;
  BaseFloat log_energy_floor_;
  std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
  SplitRadixRealFft<BaseFloat> *srfft_;
  // Disallow assignment.
  FbankComputer &operator =(const FbankComputer &other);
};

typedef OfflineFeatureTpl<FbankComputer> Fbank;

/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_FBANK_H_
