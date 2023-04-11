#pragma once


#ifndef PARAFORMER_MODELIMP_H
#define PARAFORMER_MODELIMP_H

namespace paraformer {

    class ModelImp : public Model {
    private:
        //FeatureExtract* fe;

        Vocab* vocab;
        vector<float> means_list;
        vector<float> vars_list;
        const float scale = 22.6274169979695;

        void apply_lfr(Tensor<float>*& din);
        void apply_cmvn(Tensor<float>* din);
        void load_cmvn(const char *filename);

        string greedy_search( float* in, int nLen);

#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

        Ort::Session* m_session = nullptr;
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "paraformer");
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

    public:
        ModelImp(const char* path, int nNumThread=0, bool quantize=false);
        ~ModelImp();
        void reset();
        string forward_chunk(float* din, int len, int flag);
        string forward(float* din, int len, int flag);
        string rescoring();

    };

} // namespace paraformer
#endif
