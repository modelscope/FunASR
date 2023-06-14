// feat/feature-functions.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2014  IMSL, PKU-HKUST (author: Wei Shi)

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


#include "feat/feature-functions.h"
#include "matrix/matrix-functions.h"


namespace kaldi {

void ComputePowerSpectrum(VectorBase<BaseFloat> *waveform) {
  int32 dim = waveform->Dim();

  // no, letting it be non-power-of-two for now.
  // KALDI_ASSERT(dim > 0 && (dim & (dim-1) == 0));  // make sure a power of two.. actually my FFT code
  // does not require this (dan) but this is better in case we use different code [dan].

  // RealFft(waveform, true);  // true == forward (not inverse) FFT; makes no difference here,
  // as we just want power spectrum.

  // now we have in waveform, first half of complex spectrum
  // it's stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32 half_dim = dim/2;
  BaseFloat first_energy = (*waveform)(0) * (*waveform)(0),
      last_energy = (*waveform)(1) * (*waveform)(1);  // handle this special case
  for (int32 i = 1; i < half_dim; i++) {
    BaseFloat real = (*waveform)(i*2), im = (*waveform)(i*2 + 1);
    (*waveform)(i) = real*real + im*im;
  }
  (*waveform)(0) = first_energy;
  (*waveform)(half_dim) = last_energy;  // Will actually never be used, and anyway
  // if the signal has been bandlimited sensibly this should be zero.
}


DeltaFeatures::DeltaFeatures(const DeltaFeaturesOptions &opts): opts_(opts) {
  KALDI_ASSERT(opts.order >= 0 && opts.order < 1000);  // just make sure we don't get binary junk.
  // opts will normally be 2 or 3.
  KALDI_ASSERT(opts.window > 0 && opts.window < 1000);  // again, basic sanity check.
  // normally the window size will be two.

  scales_.resize(opts.order+1);
  scales_[0].Resize(1);
  scales_[0](0) = 1.0;  // trivial window for 0th order delta [i.e. baseline feats]

  for (int32 i = 1; i <= opts.order; i++) {
    Vector<BaseFloat> &prev_scales = scales_[i-1],
        &cur_scales = scales_[i];
    int32 window = opts.window;  // this code is designed to still
    // work if instead we later make it an array and do opts.window[i-1],
    // or something like that. "window" is a parameter specifying delta-window
    // width which is actually 2*window + 1.
    KALDI_ASSERT(window != 0);
    int32 prev_offset = (static_cast<int32>(prev_scales.Dim()-1))/2,
        cur_offset = prev_offset + window;
    cur_scales.Resize(prev_scales.Dim() + 2*window);  // also zeros it.

    BaseFloat normalizer = 0.0;
    for (int32 j = -window; j <= window; j++) {
      normalizer += j*j;
      for (int32 k = -prev_offset; k <= prev_offset; k++) {
        cur_scales(j+k+cur_offset) +=
            static_cast<BaseFloat>(j) * prev_scales(k+prev_offset);
      }
    }
    cur_scales.Scale(1.0 / normalizer);
  }
}

void DeltaFeatures::Process(const MatrixBase<BaseFloat> &input_feats,
                            int32 frame,
                            VectorBase<BaseFloat> *output_frame) const {
  KALDI_ASSERT(frame < input_feats.NumRows());
  int32 num_frames = input_feats.NumRows(),
      feat_dim = input_feats.NumCols();
  KALDI_ASSERT(static_cast<int32>(output_frame->Dim()) == feat_dim * (opts_.order+1));
  output_frame->SetZero();
  for (int32 i = 0; i <= opts_.order; i++) {
    const Vector<BaseFloat> &scales = scales_[i];
    int32 max_offset = (scales.Dim() - 1) / 2;
    SubVector<BaseFloat> output(*output_frame, i*feat_dim, feat_dim);
    for (int32 j = -max_offset; j <= max_offset; j++) {
      // if asked to read
      int32 offset_frame = frame + j;
      if (offset_frame < 0) offset_frame = 0;
      else if (offset_frame >= num_frames)
        offset_frame = num_frames - 1;
      BaseFloat scale = scales(j + max_offset);
      if (scale != 0.0)
        output.AddVec(scale, input_feats.Row(offset_frame));
    }
  }
}

ShiftedDeltaFeatures::ShiftedDeltaFeatures(
  const ShiftedDeltaFeaturesOptions &opts): opts_(opts) {
  KALDI_ASSERT(opts.window > 0 && opts.window < 1000);

  // Default window is 1.
  int32 window = opts.window;
  KALDI_ASSERT(window != 0);
  scales_.Resize(1 + 2*window);  // also zeros it.
  BaseFloat normalizer = 0.0;
  for (int32 j = -window; j <= window; j++) {
    normalizer += j*j;
    scales_(j + window) += static_cast<BaseFloat>(j);
  }
  scales_.Scale(1.0 / normalizer);
}

void ShiftedDeltaFeatures::Process(const MatrixBase<BaseFloat> &input_feats,
                            int32 frame,
                            SubVector<BaseFloat> *output_frame) const {
  KALDI_ASSERT(frame < input_feats.NumRows());
  int32 num_frames = input_feats.NumRows(),
      feat_dim = input_feats.NumCols();
  KALDI_ASSERT(static_cast<int32>(output_frame->Dim())
               == feat_dim * (opts_.num_blocks + 1));
  output_frame->SetZero();

  // The original features
  SubVector<BaseFloat> output(*output_frame, 0, feat_dim);
  output.AddVec(1.0, input_feats.Row(frame));

  // Concatenate the delta-blocks. Each block is block_shift
  // (usually 3) frames apart.
  for (int32 i = 0; i < opts_.num_blocks; i++) {
    int32 max_offset = (scales_.Dim() - 1) / 2;
    SubVector<BaseFloat> output(*output_frame, (i + 1) * feat_dim, feat_dim);
    for (int32 j = -max_offset; j <= max_offset; j++) {
      int32 offset_frame = frame + j + i * opts_.block_shift;
      if (offset_frame < 0) offset_frame = 0;
      else if (offset_frame >= num_frames)
        offset_frame = num_frames - 1;
      BaseFloat scale = scales_(j + max_offset);
      if (scale != 0.0)
        output.AddVec(scale, input_feats.Row(offset_frame));
    }
  }
}

void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features) {
  output_features->Resize(input_features.NumRows(),
                          input_features.NumCols()
                          *(delta_opts.order + 1));
  DeltaFeatures delta(delta_opts);
  for (int32 r = 0; r < static_cast<int32>(input_features.NumRows()); r++) {
    SubVector<BaseFloat> row(*output_features, r);
    delta.Process(input_features, r, &row);
  }
}

void ComputeShiftedDeltas(const ShiftedDeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features) {
  output_features->Resize(input_features.NumRows(),
                          input_features.NumCols()
                          * (delta_opts.num_blocks + 1));
  ShiftedDeltaFeatures delta(delta_opts);

  for (int32 r = 0; r < static_cast<int32>(input_features.NumRows()); r++) {
    SubVector<BaseFloat> row(*output_features, r);
    delta.Process(input_features, r, &row);
  }
}


void InitIdftBases(int32 n_bases, int32 dimension, Matrix<BaseFloat> *mat_out) {
  BaseFloat angle = M_PI / static_cast<BaseFloat>(dimension - 1);
  BaseFloat scale = 1.0f / (2.0 * static_cast<BaseFloat>(dimension - 1));
  mat_out->Resize(n_bases, dimension);
  for (int32 i = 0; i < n_bases; i++) {
    (*mat_out)(i, 0) = 1.0 * scale;
    BaseFloat i_fl = static_cast<BaseFloat>(i);
    for (int32 j = 1; j < dimension - 1; j++) {
      BaseFloat j_fl = static_cast<BaseFloat>(j);
      (*mat_out)(i, j) = 2.0 * scale * cos(angle * i_fl * j_fl);
    }

    (*mat_out)(i, dimension -1)
        = scale * cos(angle * i_fl * static_cast<BaseFloat>(dimension-1));
  }
}

void SpliceFrames(const MatrixBase<BaseFloat> &input_features,
                  int32 left_context,
                  int32 right_context,
                  Matrix<BaseFloat> *output_features) {
  int32 T = input_features.NumRows(), D = input_features.NumCols();
  if (T == 0 || D == 0)
    KALDI_ERR << "SpliceFrames: empty input";
  KALDI_ASSERT(left_context >= 0 && right_context >= 0);
  int32 N = 1 + left_context + right_context;
  output_features->Resize(T, D*N);
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> dst_row(*output_features, t);
    for (int32 j = 0; j < N; j++) {
      int32 t2 = t + j - left_context;
      if (t2 < 0) t2 = 0;
      if (t2 >= T) t2 = T-1;
      SubVector<BaseFloat> dst(dst_row, j*D, D),
          src(input_features, t2);
      dst.CopyFromVec(src);
    }
  }
}

void ReverseFrames(const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features) {
  int32 T = input_features.NumRows(), D = input_features.NumCols();
  if (T == 0 || D == 0)
    KALDI_ERR << "ReverseFrames: empty input";
  output_features->Resize(T, D);
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> dst_row(*output_features, t);
    SubVector<BaseFloat> src_row(input_features, T-1-t);
    dst_row.CopyFromVec(src_row);
  }
}


void SlidingWindowCmnOptions::Check() const {
  KALDI_ASSERT(cmn_window > 0);
  if (center)
    KALDI_ASSERT(min_window > 0 && min_window <= cmn_window);
  // else ignored so value doesn't matter.
}

// Internal version of SlidingWindowCmn with double-precision arguments.
void SlidingWindowCmnInternal(const SlidingWindowCmnOptions &opts,
                              const MatrixBase<double> &input,
                              MatrixBase<double> *output) {
  opts.Check();
  int32 num_frames = input.NumRows(), dim = input.NumCols(),
        last_window_start = -1, last_window_end = -1,
        warning_count = 0;
  Vector<double> cur_sum(dim), cur_sumsq(dim);

  for (int32 t = 0; t < num_frames; t++) {
    int32 window_start, window_end; // note: window_end will be one
    // past the end of the window we use for normalization.
    if (opts.center) {
      window_start = t - (opts.cmn_window / 2);
      window_end = window_start + opts.cmn_window;
    } else {
      window_start = t - opts.cmn_window;
      window_end = t + 1;
    }
    if (window_start < 0) { // shift window right if starts <0.
      window_end -= window_start;
      window_start = 0; // or: window_start -= window_start
    }
    if (!opts.center) {
      if (window_end > t)
        window_end = std::max(t + 1, opts.min_window);
    }
    if (window_end > num_frames) {
      window_start -= (window_end - num_frames);
      window_end = num_frames;
      if (window_start < 0) window_start = 0;
    }
    if (last_window_start == -1) {
      SubMatrix<double> input_part(input,
                                      window_start, window_end - window_start,
                                      0, dim);
      cur_sum.AddRowSumMat(1.0, input_part , 0.0);
      if (opts.normalize_variance)
        cur_sumsq.AddDiagMat2(1.0, input_part, kTrans, 0.0);
    } else {
      if (window_start > last_window_start) {
        KALDI_ASSERT(window_start == last_window_start + 1);
        SubVector<double> frame_to_remove(input, last_window_start);
        cur_sum.AddVec(-1.0, frame_to_remove);
        if (opts.normalize_variance)
          cur_sumsq.AddVec2(-1.0, frame_to_remove);
      }
      if (window_end > last_window_end) {
        KALDI_ASSERT(window_end == last_window_end + 1);
        SubVector<double> frame_to_add(input, last_window_end);
        cur_sum.AddVec(1.0, frame_to_add);
        if (opts.normalize_variance)
          cur_sumsq.AddVec2(1.0, frame_to_add);
      }
    }
    int32 window_frames = window_end - window_start;
    last_window_start = window_start;
    last_window_end = window_end;

    KALDI_ASSERT(window_frames > 0);
    SubVector<double> input_frame(input, t),
        output_frame(*output, t);
    output_frame.CopyFromVec(input_frame);
    output_frame.AddVec(-1.0 / window_frames, cur_sum);

    if (opts.normalize_variance) {
      if (window_frames == 1) {
        output_frame.Set(0.0);
      } else {
        Vector<double> variance(cur_sumsq);
        variance.Scale(1.0 / window_frames);
        variance.AddVec2(-1.0 / (window_frames * window_frames), cur_sum);
        // now "variance" is the variance of the features in the window,
        // around their own mean.
        int32 num_floored;
	variance.ApplyFloor(1.0e-10, &num_floored);
        if (num_floored > 0 && num_frames > 1) {
          if (opts.max_warnings == warning_count) {
            KALDI_WARN << "Suppressing the remaining variance flooring "
                       << "warnings. Run program with --max-warnings=-1 to "
                       << "see all warnings.";
          }
          // If opts.max_warnings is a negative number, we won't restrict the
          // number of times that the warning is printed out.
          else if (opts.max_warnings < 0
                   || opts.max_warnings > warning_count) {
            KALDI_WARN << "Flooring when normalizing variance, floored "
                       << num_floored << " elements; num-frames was "
                       << window_frames;
          }
          warning_count++;
        }
        variance.ApplyPow(-0.5); // get inverse standard deviation.
        output_frame.MulElements(variance);
      }
    }
  }
}


void SlidingWindowCmn(const SlidingWindowCmnOptions &opts,
                      const MatrixBase<BaseFloat> &input,
                      MatrixBase<BaseFloat> *output) {
  KALDI_ASSERT(SameDim(input, *output) && input.NumRows() > 0);
  Matrix<double> input_dbl(input), output_dbl(input.NumRows(), input.NumCols());
  // call double-precision version
  SlidingWindowCmnInternal(opts, input_dbl, &output_dbl);
  output->CopyFromMat(output_dbl);
}



}  // namespace kaldi
