/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once 
#include "precomp.h"

namespace funasr {
class FsmnVadOnline : public VadModel {
/**
 * Author: Speech Lab of DAMO Academy, Alibaba Group
 * Deep-FSMN for Large Vocabulary Continuous Speech Recognition
 * https://arxiv.org/abs/1803.05030
*/

public:
    explicit FsmnVadOnline(FsmnVad* fsmnvad_handle);
    ~FsmnVadOnline();
    void Test();
    std::vector<std::vector<int>> Infer(std::vector<float> &waves, bool input_finished);
    void ExtractFeats(float sample_rate, vector<vector<float>> &vad_feats, vector<float> &waves, bool input_finished);
    void Reset();

private:
    E2EVadModel vad_scorer = E2EVadModel();
    // std::unique_ptr<FsmnVad> fsmnvad_handle_;
    FsmnVad* fsmnvad_handle_ = nullptr;

    void FbankKaldi(float sample_rate, std::vector<std::vector<float>> &vad_feats,
                    std::vector<float> &waves);
    int OnlineLfrCmvn(vector<vector<float>> &vad_feats, bool input_finished);
    void InitVad(const std::string &vad_model, const std::string &vad_cmvn, const std::string &vad_config, int thread_num){}
    void InitCache();
    void InitOnline(std::shared_ptr<Ort::Session> &vad_session,
                    Ort::Env &env,
                    std::vector<const char *> &vad_in_names,
                    std::vector<const char *> &vad_out_names,
                    knf::FbankOptions &fbank_opts,
                    std::vector<float> &means_list,
                    std::vector<float> &vars_list,
                    int vad_sample_rate,
                    int vad_silence_duration,
                    int vad_max_len,
                    double vad_speech_noise_thres);

    static int ComputeFrameNum(int sample_length, int frame_sample_length, int frame_shift_sample_length) {
        int frame_num = static_cast<int>((sample_length - frame_sample_length) / frame_shift_sample_length + 1);
        if (frame_num >= 1 && sample_length >= frame_sample_length)
            return frame_num;
        else
            return 0;
    }
    void ResetCache() {
        reserve_waveforms_.clear();
        input_cache_.clear();
        lfr_splice_cache_.clear();
    }

    // from fsmnvad_handle_
    std::shared_ptr<Ort::Session> vad_session_ = nullptr;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::vector<const char *> vad_in_names_;
    std::vector<const char *> vad_out_names_;
    knf::FbankOptions fbank_opts_;
    std::vector<float> means_list_;
    std::vector<float> vars_list_;

    std::vector<std::vector<float>> in_cache_;
    // The reserved waveforms by fbank
    std::vector<float> reserve_waveforms_;
    // waveforms reserved after last shift position
    std::vector<float> input_cache_;
    // lfr reserved cache
    std::vector<std::vector<float>> lfr_splice_cache_;

    int vad_sample_rate_ = MODEL_SAMPLE_RATE;
    int vad_silence_duration_ = VAD_SILENCE_DURATION;
    int vad_max_len_ = VAD_MAX_LEN;
    double vad_speech_noise_thres_ = VAD_SPEECH_NOISE_THRES;
    int lfr_m = VAD_LFR_M;
    int lfr_n = VAD_LFR_N;
    int frame_sample_length_ = vad_sample_rate_ / 1000 * 25;;
    int frame_shift_sample_length_ = vad_sample_rate_ / 1000 * 10;
};

} // namespace funasr
