// feat/signal.cc

// Copyright 2015  Tom Ko

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/signal.h"

namespace kaldi {

void ElementwiseProductOfFft(const Vector<BaseFloat> &a, Vector<BaseFloat> *b) {
  int32 num_fft_bins = a.Dim() / 2;
  for (int32 i = 0; i < num_fft_bins; i++) {
    // do complex multiplication
    ComplexMul(a(2*i), a(2*i + 1), &((*b)(2*i)), &((*b)(2*i + 1)));
  }
}

void ConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal) {
  int32 signal_length = signal->Dim();
  int32 filter_length = filter.Dim();
  int32 output_length = signal_length + filter_length - 1;
  Vector<BaseFloat> signal_padded(output_length);
  signal_padded.SetZero();
  for (int32 i = 0; i < signal_length; i++) {
    for (int32 j = 0; j < filter_length; j++) {
        signal_padded(i + j) += (*signal)(i) * filter(j);
    }
  }
  signal->Resize(output_length);
  signal->CopyFromVec(signal_padded);
}


void FFTbasedConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal) {
  int32 signal_length = signal->Dim();
  int32 filter_length = filter.Dim();
  int32 output_length = signal_length + filter_length - 1;

  int32 fft_length = RoundUpToNearestPowerOfTwo(output_length);
  KALDI_VLOG(1) << "fft_length for full signal convolution is " << fft_length;

  SplitRadixRealFft<BaseFloat> srfft(fft_length);

  Vector<BaseFloat> filter_padded(fft_length);
  filter_padded.Range(0, filter_length).CopyFromVec(filter);
  srfft.Compute(filter_padded.Data(), true);

  Vector<BaseFloat> signal_padded(fft_length);
  signal_padded.Range(0, signal_length).CopyFromVec(*signal);
  srfft.Compute(signal_padded.Data(), true);

  ElementwiseProductOfFft(filter_padded, &signal_padded);

  srfft.Compute(signal_padded.Data(), false);
  signal_padded.Scale(1.0 / fft_length);

  signal->Resize(output_length);
  signal->CopyFromVec(signal_padded.Range(0, output_length));
}

void FFTbasedBlockConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal) {
  int32 signal_length = signal->Dim();
  int32 filter_length = filter.Dim();
  int32 output_length = signal_length + filter_length - 1;
  signal->Resize(output_length, kCopyData);

  KALDI_VLOG(1) << "Length of the filter is " << filter_length;

  int32 fft_length = RoundUpToNearestPowerOfTwo(4 * filter_length);
  KALDI_VLOG(1) << "Best FFT length is " << fft_length;

  int32 block_length = fft_length - filter_length + 1;
  KALDI_VLOG(1) << "Block size is " << block_length;
  SplitRadixRealFft<BaseFloat> srfft(fft_length);

  Vector<BaseFloat> filter_padded(fft_length);
  filter_padded.Range(0, filter_length).CopyFromVec(filter);
  srfft.Compute(filter_padded.Data(), true);

  Vector<BaseFloat> temp_pad(filter_length - 1);
  temp_pad.SetZero();
  Vector<BaseFloat> signal_block_padded(fft_length);

  for (int32 po = 0; po < output_length; po += block_length) {
    // get a block of the signal
    int32 process_length = std::min(block_length, output_length - po);
    signal_block_padded.SetZero();
    signal_block_padded.Range(0, process_length).CopyFromVec(signal->Range(po, process_length));

    srfft.Compute(signal_block_padded.Data(), true);

    ElementwiseProductOfFft(filter_padded, &signal_block_padded);

    srfft.Compute(signal_block_padded.Data(), false);
    signal_block_padded.Scale(1.0 / fft_length);

    // combine the block
    if (po + block_length < output_length) {       // current block is not the last block
      signal->Range(po, block_length).CopyFromVec(signal_block_padded.Range(0, block_length));
      signal->Range(po, filter_length - 1).AddVec(1.0, temp_pad);
      temp_pad.CopyFromVec(signal_block_padded.Range(block_length, filter_length - 1));
    } else {
      signal->Range(po, output_length - po).CopyFromVec(
                        signal_block_padded.Range(0, output_length - po));
      if (filter_length - 1 < output_length - po)
        signal->Range(po, filter_length - 1).AddVec(1.0, temp_pad);
      else
        signal->Range(po, output_length - po).AddVec(1.0, temp_pad.Range(0, output_length - po));
    }
  }
}
}

