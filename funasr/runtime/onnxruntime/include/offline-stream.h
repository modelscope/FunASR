#ifndef OFFLINE_STREAM_H
#define OFFLINE_STREAM_H

#include <memory>
#include <string>
#include <map>
#include "model.h"
#include "punc-model.h"
#include "vad-model.h"

namespace funasr {
class OfflineStream {
  public:
    OfflineStream(std::map<std::string, std::string>& model_path, int thread_num);
    ~OfflineStream(){};

    std::unique_ptr<VadModel> vad_handle= nullptr;
    std::unique_ptr<Model> asr_handle= nullptr;
    std::unique_ptr<PuncModel> punc_handle= nullptr;
    bool UseVad(){return use_vad;};
    bool UsePunc(){return use_punc;}; 
    
  private:
    bool use_vad=false;
    bool use_punc=false;
};

OfflineStream *CreateOfflineStream(std::map<std::string, std::string>& model_path, int thread_num=1);
} // namespace funasr
#endif
