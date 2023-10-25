// kaldi-native-fbank/csrc/feature-window.cc
//
// Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)

// This file is copied/modified from kaldi/src/feat/feature-window.cc

#include "kaldi-native-fbank/csrc/feature-window.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

namespace knf {

std::ostream &operator<<(std::ostream &os, const FrameExtractionOptions &opts) {
  os << opts.ToString();
  return os;
}

FeatureWindowFunction::FeatureWindowFunction(const FrameExtractionOptions &opts)
    : window_(opts.WindowSize()) {
  int32_t frame_length = opts.WindowSize();
  KNF_CHECK_GT(frame_length, 0);

  float *window_data = window_.data();

  double a = M_2PI / (frame_length - 1);
  for (int32_t i = 0; i < frame_length; i++) {
    double i_fl = static_cast<double>(i);
    if (opts.window_type == "hanning") {
      window_data[i] = 0.5 - 0.5 * cos(a * i_fl);
    } else if (opts.window_type == "sine") {
      // when you are checking ws wikipedia, please
      // note that 0.5 * a = M_PI/(frame_length-1)
      window_data[i] = sin(0.5 * a * i_fl);
    } else if (opts.window_type == "hamming") {
      window_data[i] = 0.54 - 0.46 * cos(a * i_fl);
    } else if (opts.window_type ==
               "povey") {  // like hamming but goes to zero at edges.
      window_data[i] = pow(0.5 - 0.5 * cos(a * i_fl), 0.85);
    } else if (opts.window_type == "rectangular") {
      window_data[i] = 1.0;
    } else if (opts.window_type == "blackman") {
      window_data[i] = opts.blackman_coeff - 0.5 * cos(a * i_fl) +
                       (0.5 - opts.blackman_coeff) * cos(2 * a * i_fl);
    } else {
      KNF_LOG(FATAL) << "Invalid window type " << opts.window_type;
    }
  }
}

void FeatureWindowFunction::Apply(float *wave) const {
  int32_t window_size = window_.size();
  const float *p = window_.data();
  for (int32_t k = 0; k != window_size; ++k) {
    wave[k] *= p[k];
  }
}

int64_t FirstSampleOfFrame(int32_t frame, const FrameExtractionOptions &opts) {
  int64_t frame_shift = opts.WindowShift();
  if (opts.snip_edges) {
    return frame * frame_shift;
  } else {
    int64_t midpoint_of_frame = frame_shift * frame + frame_shift / 2,
            beginning_of_frame = midpoint_of_frame - opts.WindowSize() / 2;
    return beginning_of_frame;
  }
}

int32_t NumFrames(int64_t num_samples, const FrameExtractionOptions &opts,
                  bool flush /*= true*/) {
  int64_t frame_shift = opts.WindowShift();
  int64_t frame_length = opts.WindowSize();
  if (opts.snip_edges) {
    // with --snip-edges=true (the default), we use a HTK-like approach to
    // determining the number of frames-- all frames have to fit completely into
    // the waveform, and the first frame begins at sample zero.
    if (num_samples < frame_length)
      return 0;
    else
      return (1 + ((num_samples - frame_length) / frame_shift));
    // You can understand the expression above as follows: 'num_samples -
    // frame_length' is how much room we have to shift the frame within the
    // waveform; 'frame_shift' is how much we shift it each time; and the ratio
    // is how many times we can shift it (integer arithmetic rounds down).
  } else {
    // if --snip-edges=false, the number of frames is determined by rounding the
    // (file-length / frame-shift) to the nearest integer.  The point of this
    // formula is to make the number of frames an obvious and predictable
    // function of the frame shift and signal length, which makes many
    // segmentation-related questions simpler.
    //
    // Because integer division in C++ rounds toward zero, we add (half the
    // frame-shift minus epsilon) before dividing, to have the effect of
    // rounding towards the closest integer.
    int32_t num_frames = (num_samples + (frame_shift / 2)) / frame_shift;

    if (flush) return num_frames;

    // note: 'end' always means the last plus one, i.e. one past the last.
    int64_t end_sample_of_last_frame =
        FirstSampleOfFrame(num_frames - 1, opts) + frame_length;

    // the following code is optimized more for clarity than efficiency.
    // If flush == false, we can't output frames that extend past the end
    // of the signal.
    while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
      num_frames--;
      end_sample_of_last_frame -= frame_shift;
    }
    return num_frames;
  }
}

void ExtractWindow(int64_t sample_offset, const std::vector<float> &wave,
                   int32_t f, const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   std::vector<float> *window,
                   float *log_energy_pre_window /*= nullptr*/) {
  KNF_CHECK(sample_offset >= 0 && wave.size() != 0);

  int32_t frame_length = opts.WindowSize();
  int32_t frame_length_padded = opts.PaddedWindowSize();

  int64_t num_samples = sample_offset + wave.size();
  int64_t start_sample = FirstSampleOfFrame(f, opts);
  int64_t end_sample = start_sample + frame_length;

  if (opts.snip_edges) {
    KNF_CHECK(start_sample >= sample_offset && end_sample <= num_samples);
  } else {
    KNF_CHECK(sample_offset == 0 || start_sample >= sample_offset);
  }

  if (window->size() != frame_length_padded) {
    window->resize(frame_length_padded);
  }

  // wave_start and wave_end are start and end indexes into 'wave', for the
  // piece of wave that we're trying to extract.
  int32_t wave_start = int32_t(start_sample - sample_offset);
  int32_t wave_end = wave_start + frame_length;

  if (wave_start >= 0 && wave_end <= wave.size()) {
    // the normal case-- no edge effects to consider.
    std::copy(wave.begin() + wave_start,
              wave.begin() + wave_start + frame_length, window->data());
  } else {
    // Deal with any end effects by reflection, if needed.  This code will only
    // be reached for about two frames per utterance, so we don't concern
    // ourselves excessively with efficiency.
    int32_t wave_dim = wave.size();
    for (int32_t s = 0; s < frame_length; ++s) {
      int32_t s_in_wave = s + wave_start;
      while (s_in_wave < 0 || s_in_wave >= wave_dim) {
        // reflect around the beginning or end of the wave.
        // e.g. -1 -> 0, -2 -> 1.
        // dim -> dim - 1, dim + 1 -> dim - 2.
        // the code supports repeated reflections, although this
        // would only be needed in pathological cases.
        if (s_in_wave < 0)
          s_in_wave = -s_in_wave - 1;
        else
          s_in_wave = 2 * wave_dim - 1 - s_in_wave;
      }
      (*window)[s] = wave[s_in_wave];
    }
  }

  ProcessWindow(opts, window_function, window->data(), log_energy_pre_window);
}

static void RemoveDcOffset(float *d, int32_t n) {
  float sum = 0;
  for (int32_t i = 0; i != n; ++i) {
    sum += d[i];
  }

  float mean = sum / n;

  for (int32_t i = 0; i != n; ++i) {
    d[i] -= mean;
  }
}

float InnerProduct(const float *a, const float *b, int32_t n) {
  float sum = 0;
  for (int32_t i = 0; i != n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

static void Preemphasize(float *d, int32_t n, float preemph_coeff) {
  if (preemph_coeff == 0.0) {
    return;
  }

  KNF_CHECK(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);

  for (int32_t i = n - 1; i > 0; --i) {
    d[i] -= preemph_coeff * d[i - 1];
  }
  d[0] -= preemph_coeff * d[0];
}

void ProcessWindow(const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function, float *window,
                   float *log_energy_pre_window /*= nullptr*/) {
  int32_t frame_length = opts.WindowSize();

//  // TODO(fangjun): Remove dither
//  KNF_CHECK_EQ(opts.dither, 0);

  // Add dither function
  // https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.cc
  if (opts.dither!=0) {
//    kaldi::RandomState rstate;
//    rstate.seed=0;
//    for (int32 i = 0; i < frame_length; i++)
//      window[i] += RandGauss(&rstate) * opts.dither;
  }

  if (opts.remove_dc_offset) {
    RemoveDcOffset(window, frame_length);
  }

  if (log_energy_pre_window != NULL) {
    float energy = std::max<float>(InnerProduct(window, window, frame_length),
                                   std::numeric_limits<float>::epsilon());
    *log_energy_pre_window = std::log(energy);
  }

  if (opts.preemph_coeff != 0.0) {
    Preemphasize(window, frame_length, opts.preemph_coeff);
  }

  window_function.Apply(window);
}

}  // namespace knf
