/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once

#include "precomp.h"

namespace funasr {

    class Paraformer : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:
        //std::unique_ptr<knf::OnlineFbank> fbank_;
        knf::FbankOptions fbank_opts;

        Vocab* vocab = nullptr;
        SegDict* seg_dict = nullptr;
        vector<float> means_list;
        vector<float> vars_list;
        const float scale = 22.6274169979695;
        int32_t lfr_window_size = 7;
        int32_t lfr_window_shift = 6;

        void LoadCmvn(const char *filename);
        vector<float> ApplyLfr(const vector<float> &in);
        void ApplyCmvn(vector<float> *v);
        string GreedySearch( float* in, int n_len, int64_t token_nums);

        std::shared_ptr<Ort::Session> m_session = nullptr;
        Ort::Env env_;
        Ort::SessionOptions session_options;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

        std::shared_ptr<Ort::Session> hw_m_session = nullptr;
        Ort::Env hw_env_;
        Ort::SessionOptions hw_session_options;
        vector<string> hw_m_strInputNames, hw_m_strOutputNames;
        vector<const char*> hw_m_szInputNames;
        vector<const char*> hw_m_szOutputNames;
        bool use_hotword;

    public:
        Paraformer();
        ~Paraformer();
        void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        void InitHwCompiler(const std::string &hw_model, int thread_num);
        void InitSegDict(const std::string &seg_dict_model);
        void Reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string ForwardChunk(float* din, int len, int flag);
        string Forward(float* din, int len, int flag, const std::vector<std::vector<float>> &hw_emb);
        std::vector<std::vector<float>> CompileHotwordEmbedding(std::string &hotwords);
        string Rescoring();
    };

} // namespace funasr
