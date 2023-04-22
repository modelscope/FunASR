//
// Created by zhuzizyf(China Telecom Shanghai) on 4/22/23.
//


#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <vector>

using namespace std;

class OnlineFeature {

public:
  OnlineFeature(int sample_rate, knf::FbankOptions fbank_opts, int lfr_m_, int lfr_n_,
                std::vector<std::vector<float>> cmvns_);

  void extractFeats(vector<vector<float>> &vad_feats, vector<float> waves, bool input_finished);


private:
  void onlineFbank(vector<vector<float>> &vad_feats, vector<float> &waves);

  int OnlineLfrCmvn(vector<vector<float>> &vad_feats);

  static int compute_frame_num(int sample_length, int frame_sample_length, int frame_shift_sample_length) {
    int frame_num = static_cast<int>((sample_length - frame_sample_length) / frame_shift_sample_length + 1);

    if (frame_num >= 1 && sample_length >= frame_sample_length)
      return frame_num;
    else
      return 0;
  }

  void reset_cache() {
    reserve_waveforms_.clear();
    input_cache_.clear();
    lfr_splice_cache_.clear();
    input_finished_ = false;

  }

  knf::FbankOptions fbank_opts_;
  // The reserved waveforms by fbank
  std::vector<float> reserve_waveforms_;
  // waveforms reserved after last shift position
  std::vector<float> input_cache_;
  // lfr reserved cache
  std::vector<std::vector<float>> lfr_splice_cache_;
  std::vector<std::vector<float>> cmvns_;

  int sample_rate_ = 16000;
  int frame_sample_length_ = sample_rate_ / 1000 * 25;;
  int frame_shift_sample_length_ = sample_rate_ / 1000 * 10;
  int lfr_m_;
  int lfr_n_;
  bool input_finished_ = false;

};
