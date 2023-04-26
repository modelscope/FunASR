/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once


#ifndef PARAFORMER_MODELIMP_H
#define PARAFORMER_MODELIMP_H

#include "precomp.h"

namespace paraformer {

    class Paraformer : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:
        //std::unique_ptr<knf::OnlineFbank> fbank_;
        knf::FbankOptions fbank_opts;

        std::unique_ptr<FsmnVad> vad_handle;
        std::unique_ptr<CTTransformer> punc_handle;

        Vocab* vocab;
        vector<float> means_list;
        vector<float> vars_list;
        const float scale = 22.6274169979695;
        int32_t lfr_window_size = 7;
        int32_t lfr_window_shift = 6;

        void LoadCmvn(const char *filename);
        vector<float> ApplyLfr(const vector<float> &in);
        void ApplyCmvn(vector<float> *v);

        string GreedySearch( float* in, int n_len, int64_t token_nums);

        std::shared_ptr<Ort::Session> m_session;
        Ort::Env env_;
        Ort::SessionOptions session_options;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;
        bool use_vad=false;
        bool use_punc=false;

    public:
        Paraformer(std::map<std::string, std::string>& model_path, int thread_num=0);
        ~Paraformer();
        void InitAM(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        void Reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string ForwardChunk(float* din, int len, int flag);
        string Forward(float* din, int len, int flag);
        string Rescoring();
        std::vector<std::vector<int>> VadSeg(std::vector<float>& pcm_data);
        string AddPunc(const char* sz_input);
        bool UseVad(){return use_vad;};
        bool UsePunc(){return use_punc;}; 
    };

} // namespace paraformer
#endif
