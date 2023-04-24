#pragma once


#ifndef PARAFORMER_MODELIMP_H
#define PARAFORMER_MODELIMP_H

#include "precomp.h"

namespace paraformer {

    class Paraformer : public Model {
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

        string GreedySearch( float* in, int n_len);

        std::shared_ptr<Ort::Session> m_session;
        Ort::Env env_;
        Ort::SessionOptions session_options;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

    public:
        Paraformer(const char* path, int thread_num=0, bool quantize=false, bool use_vad=false, bool use_punc=false);
        ~Paraformer();
        void Reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string ForwardChunk(float* din, int len, int flag);
        string Forward(float* din, int len, int flag);
        string Rescoring();
        std::vector<std::vector<int>> VadSeg(std::vector<float>& pcm_data);
        string AddPunc(const char* sz_input);
    };

} // namespace paraformer
#endif
