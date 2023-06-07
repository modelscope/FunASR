/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include <fstream>
#include "precomp.h"

namespace funasr {

void FsmnVadOnline::FbankKaldi(float sample_rate, std::vector<std::vector<float>> &vad_feats,
                               std::vector<float> &waves) {
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

    std::vector<float> buf(waves.size());
    for (int32_t i = 0; i != waves.size(); ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank.AcceptWaveform(sample_rate, buf.data(), buf.size());
    // fbank.AcceptWaveform(sample_rate, &waves[0], waves.size());
    int32_t frames = fbank.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float *frame = fbank.GetFrame(i);
        vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
        vad_feats.emplace_back(frame_vector);
    }
}

void FsmnVadOnline::ExtractFeats(float sample_rate, vector<std::vector<float>> &vad_feats,
                                 vector<float> &waves, bool input_finished) {
  FbankKaldi(sample_rate, vad_feats, waves);
  // cache deal & online lfr,cmvn
  if (vad_feats.size() > 0) {
    if (!reserve_waveforms_.empty()) {
      waves.insert(waves.begin(), reserve_waveforms_.begin(), reserve_waveforms_.end());
    }
    if (lfr_splice_cache_.empty()) {
      for (int i = 0; i < (lfr_m - 1) / 2; i++) {
        lfr_splice_cache_.emplace_back(vad_feats[0]);
      }
    }
    if (vad_feats.size() + lfr_splice_cache_.size() >= lfr_m) {
      vad_feats.insert(vad_feats.begin(), lfr_splice_cache_.begin(), lfr_splice_cache_.end());
      int frame_from_waves = (waves.size() - frame_sample_length_) / frame_shift_sample_length_ + 1;
      int minus_frame = reserve_waveforms_.empty() ? (lfr_m - 1) / 2 : 0;
      int lfr_splice_frame_idxs = OnlineLfrCmvn(vad_feats, input_finished);
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
    if (input_finished) {
      if (!reserve_waveforms_.empty()) {
        waves = reserve_waveforms_;
      }
      vad_feats = lfr_splice_cache_;
      OnlineLfrCmvn(vad_feats, input_finished);
    }
  }
  if(input_finished){
      Reset();
      ResetCache();
  }
}

int FsmnVadOnline::OnlineLfrCmvn(vector<vector<float>> &vad_feats, bool input_finished) {
    vector<vector<float>> out_feats;
    int T = vad_feats.size();
    int T_lrf = ceil((T - (lfr_m - 1) / 2) / lfr_n);
    int lfr_splice_frame_idxs = T_lrf;
    vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        if (lfr_m <= T - i * lfr_n) {
            for (int j = 0; j < lfr_m; j++) {
                p.insert(p.end(), vad_feats[i * lfr_n + j].begin(), vad_feats[i * lfr_n + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            if (input_finished) {
                int num_padding = lfr_m - (T - i * lfr_n);
                for (int j = 0; j < (vad_feats.size() - i * lfr_n); j++) {
                    p.insert(p.end(), vad_feats[i * lfr_n + j].begin(), vad_feats[i * lfr_n + j].end());
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
    lfr_splice_frame_idxs = std::min(T - 1, lfr_splice_frame_idxs * lfr_n);
    lfr_splice_cache_.clear();
    lfr_splice_cache_.insert(lfr_splice_cache_.begin(), vad_feats.begin() + lfr_splice_frame_idxs, vad_feats.end());

    // Apply cmvn
    for (auto &out_feat: out_feats) {
        for (int j = 0; j < means_list_.size(); j++) {
            out_feat[j] = (out_feat[j] + means_list_[j]) * vars_list_[j];
        }
    }
    vad_feats = out_feats;
    return lfr_splice_frame_idxs;
}

std::vector<std::vector<int>>
FsmnVadOnline::Infer(std::vector<float> &waves, bool input_finished) {
    std::vector<std::vector<float>> vad_feats;
    std::vector<std::vector<float>> vad_probs;
    ExtractFeats(vad_sample_rate_, vad_feats, waves, input_finished);
    fsmnvad_handle_->Forward(vad_feats, &vad_probs, &in_cache_, input_finished);

    std::vector<std::vector<int>> vad_segments;
    vad_segments = vad_scorer(vad_probs, waves, input_finished, true, vad_silence_duration_, vad_max_len_,
                              vad_speech_noise_thres_, vad_sample_rate_);
    return vad_segments;
}

void FsmnVadOnline::InitCache(){
  std::vector<float> cache_feats(128 * 19 * 1, 0);
  for (int i=0;i<4;i++){
    in_cache_.emplace_back(cache_feats);
  }
};

void FsmnVadOnline::Reset(){
  in_cache_.clear();
  InitCache();
};

void FsmnVadOnline::Test() {
}

void FsmnVadOnline::InitOnline(std::shared_ptr<Ort::Session> &vad_session,
                               Ort::Env &env,
                               std::vector<const char *> &vad_in_names,
                               std::vector<const char *> &vad_out_names,
                               knf::FbankOptions &fbank_opts,
                               std::vector<float> &means_list,
                               std::vector<float> &vars_list,
                               int vad_sample_rate,
                               int vad_silence_duration,
                               int vad_max_len,
                               double vad_speech_noise_thres) {
    vad_session_ = vad_session;
    vad_in_names_ = vad_in_names;
    vad_out_names_ = vad_out_names;
    fbank_opts_ = fbank_opts;
    means_list_ = means_list;
    vars_list_ = vars_list;
    vad_sample_rate_ = vad_sample_rate;
    vad_silence_duration_ = vad_silence_duration;
    vad_max_len_ = vad_max_len;
    vad_speech_noise_thres_ = vad_speech_noise_thres;
}

FsmnVadOnline::~FsmnVadOnline() {
}

FsmnVadOnline::FsmnVadOnline(FsmnVad* fsmnvad_handle):fsmnvad_handle_(std::move(fsmnvad_handle)),session_options_{}{
   InitCache();
   InitOnline(fsmnvad_handle_->vad_session_,
              fsmnvad_handle_->env_,
              fsmnvad_handle_->vad_in_names_,
              fsmnvad_handle_->vad_out_names_,
              fsmnvad_handle_->fbank_opts_,
              fsmnvad_handle_->means_list_,
              fsmnvad_handle_->vars_list_,
              fsmnvad_handle_->vad_sample_rate_,
              fsmnvad_handle_->vad_silence_duration_,
              fsmnvad_handle_->vad_max_len_,
              fsmnvad_handle_->vad_speech_noise_thres_);
}

} // namespace funasr
