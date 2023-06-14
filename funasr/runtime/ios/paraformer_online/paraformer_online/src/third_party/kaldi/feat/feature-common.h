// feat/feature-common.h

// Copyright      2016   Johns Hopkins University (author: Daniel Povey)

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
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_FEAT_FEATURE_COMMON_H_
#define KALDI_FEAT_FEATURE_COMMON_H_

#include <map>
#include <string>
#include "feat/feature-window.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{



/// This class is only added for documentation, it is not intended to ever be
/// used.
struct ExampleFeatureComputerOptions {
  FrameExtractionOptions frame_opts;
  // .. more would go here.
};

/// This class is only added for documentation, it is not intended to ever be
/// used.  It documents the interface of the *Computer classes which wrap the
/// low-level feature extraction.  The template argument F of OfflineFeatureTpl must
/// follow this interface.  This interface is intended for features such as
/// MFCCs and PLPs which can be computed frame by frame.
class ExampleFeatureComputer {
 public:
  typedef ExampleFeatureComputerOptions Options;

  /// Returns a reference to the frame-extraction options class, which
  /// will be part of our own options class.
  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  /// Returns the feature dimension
  int32 Dim();

  /// Returns true if this function may inspect the raw log-energy of the signal
  /// (before windowing and pre-emphasis); it's safe to always return true, but
  /// setting it to false enables an optimization.
  bool NeedRawLogEnergy() { return true; }

  /// constructor from options class; it should not store a reference or pointer
  /// to the options class but should copy it.
  explicit ExampleFeatureComputer(const ExampleFeatureComputerOptions &opts):
      opts_(opts) { }

  /// Copy constructor; all of these classes must have one.
  ExampleFeatureComputer(const ExampleFeatureComputer &other);

  /**
     Function that computes one frame of features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedRawLogEnergy().
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

 private:
  // disallow assignment.
  ExampleFeatureComputer &operator = (const ExampleFeatureComputer &in);
  Options opts_;
};


/// This templated class is intended for offline feature extraction, i.e. where
/// you have access to the entire signal at the start.  It exists mainly to be
/// drop-in replacement for the old (pre-2016) classes Mfcc, Plp and so on, for
/// use in the offline case.  In April 2016 we reorganized the online
/// feature-computation code for greater modularity and to have correct support
/// for the snip-edges=false option.
template <class F>
class OfflineFeatureTpl {
 public:
  typedef typename F::Options Options;

  // Note: feature_window_function_ is the windowing function, which initialized
  // using the options class, that we cache at this level.
  OfflineFeatureTpl(const Options &opts):
      computer_(opts),
      feature_window_function_(computer_.GetFrameOptions()) { }

  // Internal (and back-compatibility) interface for computing features, which
  // requires that the user has already checked that the sampling frequency
  // of the waveform is equal to the sampling frequency specified in
  // the frame-extraction options.
  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output);

  // This const version of Compute() is a wrapper that
  // calls the non-const version on a temporary object.
  // It's less efficient than the non-const version.
  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output) const;

  /**
     Computes the features for one file (one sequence of features).
     This is the newer interface where you specify the sample frequency
     of the input waveform.
       @param [in] wave   The input waveform
       @param [in] sample_freq  The sampling frequency with which
                                'wave' was sampled.
                                if sample_freq is higher than the frequency
                                specified in the config, we will downsample
                                the waveform, but if lower, it's an error.
     @param [in] vtln_warp  The VTLN warping factor (will normally
                            be 1.0)
     @param [out]  output  The matrix of features, where the row-index
                           is the frame index.
  */
  void ComputeFeatures(const VectorBase<BaseFloat> &wave,
                       BaseFloat sample_freq,
                       BaseFloat vtln_warp,
                       Matrix<BaseFloat> *output);
  /**
     This const version of ComputeFeatures() is a wrapper that
     calls the non-const ComputeFeatures() on a temporary object
     that is a copy of *this.  It is not as efficient because of the
     overhead of copying *this.
  */
  void ComputeFeatures(const VectorBase<BaseFloat> &wave,
                       BaseFloat sample_freq,
                       BaseFloat vtln_warp,
                       Matrix<BaseFloat> *output) const;

  int32 Dim() const { return computer_.Dim(); }

  // Copy constructor.
  OfflineFeatureTpl(const OfflineFeatureTpl<F> &other):
      computer_(other.computer_),
      feature_window_function_(other.feature_window_function_) { }
  private:
  // Disallow assignment.
  OfflineFeatureTpl<F> &operator =(const OfflineFeatureTpl<F> &other);

  F computer_;
  FeatureWindowFunction feature_window_function_;
};

/// @} End of "addtogroup feat"
}  // namespace kaldi


#include "feat/feature-common-inl.h"

#endif  // KALDI_FEAT_FEATURE_COMMON_H_
