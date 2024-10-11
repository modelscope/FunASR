/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#ifndef VAD_SERVER_FSMNVAD_H
#define VAD_SERVER_FSMNVAD_H

#include "precomp.h"

namespace funasr {
class FsmnVad : public VadModel {
/**
 * Author: Speech Lab of DAMO Academy, Alibaba Group
 * Deep-FSMN for Large Vocabulary Continuous Speech Recognition
 * https://arxiv.org/abs/1803.05030
*/

public:
    FsmnVad();
    ~FsmnVad();
    void Test();
    void InitVad(const std::string &vad_model, const std::string &vad_cmvn, const std::string &vad_config, int thread_num);
    std::vector<std::vector<int>> Infer(std::vector<float> &waves, bool input_finished=true);
    void Forward(
        const std::vector<std::vector<float>> &chunk_feats,
        std::vector<std::vector<float>> *out_prob,
        std::vector<std::vector<float>> *in_cache,
        bool is_final);
    void Reset();

    int GetVadSampleRate() { return vad_sample_rate_; };
    
    std::shared_ptr<Ort::Session> vad_session_ = nullptr;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    vector<string> m_strInputNames, m_strOutputNames;
    std::vector<const char *> vad_in_names_;
    std::vector<const char *> vad_out_names_;
    std::vector<std::vector<float>> in_cache_;
    
    knf::FbankOptions fbank_opts_;
    std::vector<float> means_list_;
    std::vector<float> vars_list_;

    int vad_sample_rate_ = MODEL_SAMPLE_RATE;
    int vad_silence_duration_ = VAD_SILENCE_DURATION;
    int vad_max_len_ = VAD_MAX_LEN;
    double vad_speech_noise_thres_ = VAD_SPEECH_NOISE_THRES;
    int lfr_m = VAD_LFR_M;
    int lfr_n = VAD_LFR_N;

private:

    void ReadModel(const char* vad_model);
    void LoadConfigFromYaml(const char* filename);

    void FbankKaldi(float sample_rate, std::vector<std::vector<float>> &vad_feats,
                    std::vector<float> &waves);

    void LfrCmvn(std::vector<std::vector<float>> &vad_feats);
    void LoadCmvn(const char *filename);
    void InitCache();

};

} // namespace funasr
#endif //VAD_SERVER_FSMNVAD_H
