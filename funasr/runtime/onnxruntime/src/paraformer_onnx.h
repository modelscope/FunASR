#pragma once


#ifndef PARAFORMER_MODELIMP_H
#define PARAFORMER_MODELIMP_H

#include "precomp.h"

namespace paraformer {

    class ModelImp : public Model {
    private:
        //std::unique_ptr<knf::OnlineFbank> fbank_;
        knf::FbankOptions fbank_opts;

        std::unique_ptr<FsmnVad> vadHandle;

        Vocab* vocab;
        vector<float> means_list;
        vector<float> vars_list;
        const float scale = 22.6274169979695;
        int32_t lfr_window_size = 7;
        int32_t lfr_window_shift = 6;

        void load_cmvn(const char *filename);
        vector<float> ApplyLFR(const vector<float> &in);
        void ApplyCMVN(vector<float> *v);

        string greedy_search( float* in, int nLen);

        std::shared_ptr<Ort::Session> m_session;
        Ort::Env env_;
        Ort::SessionOptions sessionOptions;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

    public:
        ModelImp(const char* path, int nNumThread=0, bool quantize=false, bool use_vad=false);
        ~ModelImp();
        void reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string forward_chunk(float* din, int len, int flag);
        string forward(float* din, int len, int flag);
        string rescoring();
        std::vector<std::vector<int>> vad_seg(std::vector<float>& pcm_data);

    };

} // namespace paraformer
#endif
