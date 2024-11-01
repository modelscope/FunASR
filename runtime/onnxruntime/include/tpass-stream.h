#ifndef TPASS_STREAM_H
#define TPASS_STREAM_H

#include <memory>
#include <string>
#include <map>
#include "model.h"
#include "punc-model.h"
#include "vad-model.h"
#if !defined(__APPLE__)
#include "itn-model.h"
#endif

namespace funasr {
class TpassStream {
  public:
    TpassStream(std::map<std::string, std::string>& model_path, int thread_num);
    ~TpassStream(){};

    std::unique_ptr<VadModel> vad_handle = nullptr;
    std::unique_ptr<Model> asr_handle = nullptr;
    std::unique_ptr<PuncModel> punc_online_handle = nullptr;
#if !defined(__APPLE__)
    std::unique_ptr<ITNModel> itn_handle = nullptr;
#endif
    bool UseVad(){return use_vad;};
    bool UsePunc(){return use_punc;}; 
    bool UseITN(){return use_itn;};
    std::string GetModelType(){return model_type;};
    
  private:
    bool use_vad=false;
    bool use_punc=false;
    bool use_itn=false;
    std::string model_type = MODEL_PARA;
};

TpassStream *CreateTpassStream(std::map<std::string, std::string>& model_path, int thread_num=1);
} // namespace funasr
#endif
