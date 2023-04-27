/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "online-feature.h"
#include <utility>

OnlineFeature::OnlineFeature(int sample_rate, knf::FbankOptions fbank_opts, int lfr_m, int lfr_n,
                             std::vector<std::vector<float>> cmvns)
  : sample_rate_(sample_rate),
    fbank_opts_(std::move(fbank_opts)),
    lfr_m_(lfr_m),
    lfr_n_(lfr_n),
    cmvns_(std::move(cmvns)) {
  frame_sample_length_ = sample_rate_ / 1000 * 25;;
  frame_shift_sample_length_ = sample_rate_ / 1000 * 10;
}

void OnlineFeature::ExtractFeats(vector<std::vector<float>> &vad_feats,
                                 vector<float> waves, bool input_finished) {
  input_finished_ = input_finished;
  OnlineFbank(vad_feats, waves);
  // cache deal & online lfr,cmvn
  if (vad_feats.size() > 0) {
    if (!reserve_waveforms_.empty()) {
      waves.insert(waves.begin(), reserve_waveforms_.begin(), reserve_waveforms_.end());
    }
    if (lfr_splice_cache_.empty()) {
      for (int i = 0; i < (lfr_m_ - 1) / 2; i++) {
        lfr_splice_cache_.emplace_back(vad_feats[0]);
      }
    }
    if (vad_feats.size() + lfr_splice_cache_.size() >= lfr_m_) {
      vad_feats.insert(vad_feats.begin(), lfr_splice_cache_.begin(), lfr_splice_cache_.end());
      int frame_from_waves = (waves.size() - frame_sample_length_) / frame_shift_sample_length_ + 1;
      int minus_frame = reserve_waveforms_.empty() ? (lfr_m_ - 1) / 2 : 0;
      int lfr_splice_frame_idxs = OnlineLfrCmvn(vad_feats);
      int reserve_frame_idx = lfr_splice_frame_idxs - minus_frame;
      reserve_waveforms_.clear();
      reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                waves.begin() + reserve_frame_idx * frame_shift_sample_length_,
                                waves.begin() + frame_from_waves * frame_shift_sample_length_);
      int sample_length = (frame_from_waves - 1) * frame_shift_sample_length_ + frame_sample_length_;
      waves.erase(waves.begin() + sample_length, waves.end());
    } else {
      reserve_waveforms_.clear();
      reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                waves.begin() + frame_sample_length_ - frame_shift_sample_length_, waves.end());
      lfr_splice_cache_.insert(lfr_splice_cache_.end(), vad_feats.begin(), vad_feats.end());
    }

  } else {
    if (input_finished_) {
      if (!reserve_waveforms_.empty()) {
        waves = reserve_waveforms_;
      }
      vad_feats = lfr_splice_cache_;
      OnlineLfrCmvn(vad_feats);
      ResetCache();
    }
  }

}

int OnlineFeature::OnlineLfrCmvn(vector<vector<float>> &vad_feats) {
  vector<vector<float>> out_feats;
  int T = vad_feats.size();
  int T_lrf = ceil((T - (lfr_m_ - 1) / 2) / lfr_n_);
  int lfr_splice_frame_idxs = T_lrf;
  vector<float> p;
  for (int i = 0; i < T_lrf; i++) {
    if (lfr_m_ <= T - i * lfr_n_) {
      for (int j = 0; j < lfr_m_; j++) {
        p.insert(p.end(), vad_feats[i * lfr_n_ + j].begin(), vad_feats[i * lfr_n_ + j].end());
      }
      out_feats.emplace_back(p);
      p.clear();
    } else {
      if (input_finished_) {
        int num_padding = lfr_m_ - (T - i * lfr_n_);
        for (int j = 0; j < (vad_feats.size() - i * lfr_n_); j++) {
          p.insert(p.end(), vad_feats[i * lfr_n_ + j].begin(), vad_feats[i * lfr_n_ + j].end());
        }
        for (int j = 0; j < num_padding; j++) {
          p.insert(p.end(), vad_feats[vad_feats.size() - 1].begin(), vad_feats[vad_feats.size() - 1].end());
        }
        out_feats.emplace_back(p);
      } else {
        lfr_splice_frame_idxs = i;
        break;
      }
    }
  }
  lfr_splice_frame_idxs = std::min(T - 1, lfr_splice_frame_idxs * lfr_n_);
  lfr_splice_cache_.clear();
  lfr_splice_cache_.insert(lfr_splice_cache_.begin(), vad_feats.begin() + lfr_splice_frame_idxs, vad_feats.end());

  // Apply cmvn
  for (auto &out_feat: out_feats) {
    for (int j = 0; j < cmvns_[0].size(); j++) {
      out_feat[j] = (out_feat[j] + cmvns_[0][j]) * cmvns_[1][j];
    }
  }
  vad_feats = out_feats;
  return lfr_splice_frame_idxs;
}

void OnlineFeature::OnlineFbank(vector<std::vector<float>> &vad_feats,
                                vector<float> &waves) {

  knf::OnlineFbank fbank(fbank_opts_);
  // cache merge
  waves.insert(waves.begin(), input_cache_.begin(), input_cache_.end());
  int frame_number = ComputeFrameNum(waves.size(), frame_sample_length_, frame_shift_sample_length_);
  // Send the audio after the last frame shift position to the cache
  input_cache_.clear();
  input_cache_.insert(input_cache_.begin(), waves.begin() + frame_number * frame_shift_sample_length_, waves.end());
  if (frame_number == 0) {
    return;
  }
  // Delete audio that haven't undergone fbank processing
  waves.erase(waves.begin() + (frame_number - 1) * frame_shift_sample_length_ + frame_sample_length_, waves.end());

  fbank.AcceptWaveform(sample_rate_, &waves[0], waves.size());
  int32_t frames = fbank.NumFramesReady();
  for (int32_t i = 0; i != frames; ++i) {
    const float *frame = fbank.GetFrame(i);
    vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
    vad_feats.emplace_back(frame_vector);
  }

}
