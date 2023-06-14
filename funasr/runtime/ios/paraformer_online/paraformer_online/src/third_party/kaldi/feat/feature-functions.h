// feat/feature-functions.h

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//                2014  IMSL, PKU-HKUST (author: Wei Shi)
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


#ifndef KALDI_FEAT_FEATURE_FUNCTIONS_H_
#define KALDI_FEAT_FEATURE_FUNCTIONS_H_

#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


// ComputePowerSpectrum converts a complex FFT (as produced by the FFT
// functions in matrix/matrix-functions.h), and converts it into
// a power spectrum.  If the complex FFT is a vector of size n (representing
// half the complex FFT of a real signal of size n, as described there),
// this function computes in the first (n/2) + 1 elements of it, the
// energies of the fft bins from zero to the Nyquist frequency.  Contents of the
// remaining (n/2) - 1 elements are undefined at output.
void ComputePowerSpectrum(VectorBase<BaseFloat> *complex_fft);


struct DeltaFeaturesOptions {
  int32 order;
  int32 window;  // e.g. 2; controls window size (window size is 2*window + 1)
  // the behavior at the edges is to replicate the first or last frame.
  // this is not configurable.

  DeltaFeaturesOptions(int32 order = 2, int32 window = 2):
      order(order), window(window) { }
  void Register(OptionsItf *opts) {
    opts->Register("delta-order", &order, "Order of delta computation");
    opts->Register("delta-window", &window,
                   "Parameter controlling window for delta computation (actual window"
                   " size for each delta order is 1 + 2*delta-window-size)");
  }
};

class DeltaFeatures {
 public:
  // This class provides a low-level function to compute delta features.
  // The function takes as input a matrix of features and a frame index
  // that it should compute the deltas on.  It puts its output in an object
  // of type VectorBase, of size (original-feature-dimension) * (opts.order+1).
  // This is not the most efficient way to do the computation, but it's
  // state-free and thus easier to understand

  explicit DeltaFeatures(const DeltaFeaturesOptions &opts);

  void Process(const MatrixBase<BaseFloat> &input_feats,
               int32 frame,
               VectorBase<BaseFloat> *output_frame) const;
 private:
  DeltaFeaturesOptions opts_;
  std::vector<Vector<BaseFloat> > scales_;  // a scaling window for each
  // of the orders, including zero: multiply the features for each
  // dimension by this window.
};

struct ShiftedDeltaFeaturesOptions {
  int32 window,           // The time delay and advance
        num_blocks,
        block_shift;      // Distance between consecutive blocks

  ShiftedDeltaFeaturesOptions():
      window(1), num_blocks(7), block_shift(3) { }
  void Register(OptionsItf *opts) {
    opts->Register("delta-window", &window, "Size of delta advance and delay.");
    opts->Register("num-blocks", &num_blocks, "Number of delta blocks in advance"
                   " of each frame to be concatenated");
    opts->Register("block-shift", &block_shift, "Distance between each block");
  }
};

class ShiftedDeltaFeatures {
 public:
  // This class provides a low-level function to compute shifted
  // delta cesptra (SDC).
  // The function takes as input a matrix of features and a frame index
  // that it should compute the deltas on.  It puts its output in an object
  // of type VectorBase, of size original-feature-dimension + (1  * num_blocks).

  explicit ShiftedDeltaFeatures(const ShiftedDeltaFeaturesOptions &opts);

  void Process(const MatrixBase<BaseFloat> &input_feats,
               int32 frame,
               SubVector<BaseFloat> *output_frame) const;
 private:
  ShiftedDeltaFeaturesOptions opts_;
  Vector<BaseFloat> scales_;  // a scaling window for each

};

// ComputeDeltas is a convenience function that computes deltas on a feature
// file.  If you want to deal with features coming in bit by bit you would have
// to use the DeltaFeatures class directly, and do the computation frame by
// frame.  Later we will have to come up with a nice mechanism to do this for
// features coming in.
void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features);

// ComputeShiftedDeltas computes deltas from a feature file by applying
// ShiftedDeltaFeatures over the frames. This function is provided for
// convenience, however, ShiftedDeltaFeatures can be used directly.
void ComputeShiftedDeltas(const ShiftedDeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features);

// SpliceFrames will normally be used together with LDA.
// It splices frames together to make a window.  At the
// start and end of an utterance, it duplicates the first
// and last frames.
// Will throw if input features are empty.
// left_context and right_context must be nonnegative.
// these both represent a number of frames (e.g. 4, 4 is
// a good choice).
void SpliceFrames(const MatrixBase<BaseFloat> &input_features,
                  int32 left_context,
                  int32 right_context,
                  Matrix<BaseFloat> *output_features);

// ReverseFrames reverses the frames in time (used for backwards decoding)
void ReverseFrames(const MatrixBase<BaseFloat> &input_features,
                  Matrix<BaseFloat> *output_features);


void InitIdftBases(int32 n_bases, int32 dimension, Matrix<BaseFloat> *mat_out);


// This is used for speaker-id.  Also see OnlineCmnOptions in ../online2/, which
// is online CMN with no latency, for online speech recognition.
struct SlidingWindowCmnOptions {
  int32 cmn_window;
  int32 min_window;
  int32 max_warnings;
  bool normalize_variance;
  bool center;

  SlidingWindowCmnOptions():
      cmn_window(600),
      min_window(100),
      max_warnings(5),
      normalize_variance(false),
      center(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("cmn-window", &cmn_window, "Window in frames for running "
                   "average CMN computation");
    opts->Register("min-cmn-window", &min_window, "Minimum CMN window "
                   "used at start of decoding (adds latency only at start). "
                   "Only applicable if center == false, ignored if center==true");
    opts->Register("max-warnings", &max_warnings, "Maximum warnings to report "
                   "per utterance. 0 to disable, -1 to show all.");
    opts->Register("norm-vars", &normalize_variance, "If true, normalize "
                   "variance to one."); // naming this as in apply-cmvn.cc
    opts->Register("center", &center, "If true, use a window centered on the "
                   "current frame (to the extent possible, modulo end effects). "
                   "If false, window is to the left.");
  }
  void Check() const;
};


/// Applies sliding-window cepstral mean and/or variance normalization.  See the
/// strings registering the options in the options class for information on how
/// this works and what the options are.  input and output must have the same
/// dimension.
void SlidingWindowCmn(const SlidingWindowCmnOptions &opts,
                      const MatrixBase<BaseFloat> &input,
                      MatrixBase<BaseFloat> *output);


/// @} End of "addtogroup feat"
}  // namespace kaldi



#endif  // KALDI_FEAT_FEATURE_FUNCTIONS_H_
