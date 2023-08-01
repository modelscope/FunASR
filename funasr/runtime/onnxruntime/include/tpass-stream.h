#ifndef TPASS_STREAM_H
#define TPASS_STREAM_H

#include <memory>
#include <string>
#include <map>
#include "model.h"
#include "punc-model.h"
#include "vad-model.h"

namespace funasr {
class TpassStream {
  public:
    TpassStream(std::map<std::string, std::string>& model_path, int thread_num);
    ~TpassStream(){};

    // std::unique_ptr<VadModel> vad_handle = nullptr;
    std::unique_ptr<VadModel> vad_handle = nullptr;
    std::unique_ptr<VadModel> vad_online_handle = nullptr;
    std::unique_ptr<Model> asr_handle = nullptr;
    std::unique_ptr<Model> asr_online_handle = nullptr;
    std::unique_ptr<PuncModel> punc_online_handle = nullptr;
    bool UseVad(){return use_vad;};
    bool UsePunc(){return use_punc;}; 
    
  private:
    bool use_vad=false;
    bool use_punc=false;
};

TpassStream *CreateTpassStream(std::map<std::string, std::string>& model_path, int thread_num=1);
void CreateTpassOnlineStream(void* tpass_stream);
} // namespace funasr
#endif
