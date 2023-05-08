/**
 * Copyright     2013  Pegah Ghahremani
 *               2014  IMSL, PKU-HKUST (author: Wei Shi)
 *               2014  Yanqing Sun, Junjie Wang
 *               2014  Johns Hopkins University (author: Daniel Povey)
 * Copyright     2023  Xiaomi Corporation (authors: Fangjun Kuang)
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
// this file is copied and modified from
// kaldi/src/feat/resample.h
#pragma once 
#include <cstdint>
#include <vector>

namespace funasr {
/*
   We require that the input and output sampling rate be specified as
   integers, as this is an easy way to specify that their ratio be rational.
*/

class LinearResample {
 public:
  /// Constructor.  We make the input and output sample rates integers, because
  /// we are going to need to find a common divisor.  This should just remind
  /// you that they need to be integers.  The filter cutoff needs to be less
  /// than samp_rate_in_hz/2 and less than samp_rate_out_hz/2.  num_zeros
  /// controls the sharpness of the filter, more == sharper but less efficient.
  /// We suggest around 4 to 10 for normal use.
  LinearResample(int32_t samp_rate_in_hz, int32_t samp_rate_out_hz,
                 float filter_cutoff_hz, int32_t num_zeros);

  /// Calling the function Reset() resets the state of the object prior to
  /// processing a new signal; it is only necessary if you have called
  /// Resample(x, x_size, false, y) for some signal, leading to a remainder of
  /// the signal being called, but then abandon processing the signal before
  /// calling Resample(x, x_size, true, y) for the last piece.  Call it
  /// unnecessarily between signals will not do any harm.
  void Reset();

  /// This function does the resampling.  If you call it with flush == true and
  /// you have never called it with flush == false, it just resamples the input
  /// signal (it resizes the output to a suitable number of samples).
  ///
  /// You can also use this function to process a signal a piece at a time.
  /// suppose you break it into piece1, piece2, ... pieceN.  You can call
  /// \code{.cc}
  /// Resample(piece1, piece1_size, false, &output1);
  /// Resample(piece2, piece2_size, false, &output2);
  /// Resample(piece3, piece3_size, true, &output3);
  /// \endcode
  /// If you call it with flush == false, it won't output the last few samples
  /// but will remember them, so that if you later give it a second piece of
  /// the input signal it can process it correctly.
  /// If your most recent call to the object was with flush == false, it will
  /// have internal state; you can remove this by calling Reset().
  /// Empty input is acceptable.
  void Resample(const float *input, int32_t input_dim, bool flush,
                std::vector<float> *output);

  //// Return the input and output sampling rates (for checks, for example)
  int32_t GetInputSamplingRate() const { return samp_rate_in_; }
  int32_t GetOutputSamplingRate() const { return samp_rate_out_; }

 private:
  void SetIndexesAndWeights();

  float FilterFunc(float) const;

  /// This function outputs the number of output samples we will output
  /// for a signal with "input_num_samp" input samples.  If flush == true,
  /// we return the largest n such that
  /// (n/samp_rate_out_) is in the interval [ 0, input_num_samp/samp_rate_in_ ),
  /// and note that the interval is half-open.  If flush == false,
  /// define window_width as num_zeros / (2.0 * filter_cutoff_);
  /// we return the largest n such that (n/samp_rate_out_) is in the interval
  /// [ 0, input_num_samp/samp_rate_in_ - window_width ).
  int64_t GetNumOutputSamples(int64_t input_num_samp, bool flush) const;

  /// Given an output-sample index, this function outputs to *first_samp_in the
  /// first input-sample index that we have a weight on (may be negative),
  /// and to *samp_out_wrapped the index into weights_ where we can get the
  /// corresponding weights on the input.
  inline void GetIndexes(int64_t samp_out, int64_t *first_samp_in,
                         int32_t *samp_out_wrapped) const;

  void SetRemainder(const float *input, int32_t input_dim);

 private:
  // The following variables are provided by the user.
  int32_t samp_rate_in_;
  int32_t samp_rate_out_;
  float filter_cutoff_;
  int32_t num_zeros_;

  int32_t input_samples_in_unit_;  ///< The number of input samples in the
                                   ///< smallest repeating unit: num_samp_in_ =
                                   ///< samp_rate_in_hz / Gcd(samp_rate_in_hz,
                                   ///< samp_rate_out_hz)

  int32_t output_samples_in_unit_;  ///< The number of output samples in the
                                    ///< smallest repeating unit: num_samp_out_
                                    ///< = samp_rate_out_hz /
                                    ///< Gcd(samp_rate_in_hz, samp_rate_out_hz)

  /// The first input-sample index that we sum over, for this output-sample
  /// index.  May be negative; any truncation at the beginning is handled
  /// separately.  This is just for the first few output samples, but we can
  /// extrapolate the correct input-sample index for arbitrary output samples.
  std::vector<int32_t> first_index_;

  /// Weights on the input samples, for this output-sample index.
  std::vector<std::vector<float>> weights_;

  // the following variables keep track of where we are in a particular signal,
  // if it is being provided over multiple calls to Resample().

  int64_t input_sample_offset_;   ///< The number of input samples we have
                                  ///< already received for this signal
                                  ///< (including anything in remainder_)
  int64_t output_sample_offset_;  ///< The number of samples we have already
                                  ///< output for this signal.
  std::vector<float> input_remainder_;  ///< A small trailing part of the
                                        ///< previously seen input signal.
};
} // namespace funasr
