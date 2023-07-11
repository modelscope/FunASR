/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once

#include "precomp.h"

namespace funasr {

    class ParaformerOnline : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * ParaformerOnline: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:
        //std::unique_ptr<knf::OnlineFbank> fbank_;
        knf::FbankOptions fbank_opts;

        Vocab* vocab = nullptr;
        vector<float> means_list;
        vector<float> vars_list;
        const float scale = 22.6274169979695;
        int32_t lfr_window_size = 7;
        int32_t lfr_window_shift = 6;

        void LoadCmvn(const char *filename);
        vector<float> ApplyLfr(const vector<float> &in);
        void ApplyCmvn(vector<float> *v);
        string GreedySearch( float* in, int n_len, int64_t token_nums);

        std::shared_ptr<Ort::Session> encoder_session = nullptr;
        std::shared_ptr<Ort::Session> decoder_session = nullptr;
        Ort::Env env_;
        Ort::SessionOptions session_options;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

    public:
        ParaformerOnline();
        ~ParaformerOnline();
        void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        void Reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string ForwardChunk(float* din, int len, int flag);
        string Forward(float* din, int len, int flag);
        string Rescoring();
    };

} // namespace funasr
