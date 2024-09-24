/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 */

#pragma once
#include "cam-sv-model.h"

namespace funasr
{
    class CamPPlusSv : public SvModel
    {
        /**
         * Author: Speech Lab of DAMO Academy, Alibaba Group
         * Deep-FSMN for Large Vocabulary Continuous Speech Recognition
         * https://arxiv.org/abs/1803.05030
         */

    public:
        CamPPlusSv();
        ~CamPPlusSv();    
        void InitSv(const std::string &model, const std::string &config, int thread_num);
        std::vector<std::vector<float>> Infer(sv_segment vad_seg);
        void Forward(
            const std::vector<std::vector<std::vector<float>>> &chunk_feats,
            std::vector<std::vector<float>> *out_prob);

        std::shared_ptr<Ort::Session> cam_session_ = nullptr;
        Ort::Env env_;
        Ort::SessionOptions session_options_;
        std::vector<const char *> cam_in_names_;
        std::vector<const char *> cam_out_names_;
        knf::FbankOptions fbank_opts_;

        int sample_rate_ = MODEL_SAMPLE_RATE;
        const double seg_dur = 1.5;
        const double seg_shift = 0.75;

    private:
        void ReadModel(const char *cam_model);
        void LoadConfigFromYaml(const char *filename);
        static void GetInputOutputInfo(
            const std::shared_ptr<Ort::Session> &session,
            std::vector<const char *> *in_names, std::vector<const char *> *out_names);

        void FbankKaldi(float sample_rate, std::vector<std::vector<float>> &vad_feats,
                        std::vector<float> &waves);
        void SubMean(std::vector<std::vector<float>>& voice_feats);
        std::vector<sv_segment> SegChunk(sv_segment& seg_data);
    };

} // namespace funasr
