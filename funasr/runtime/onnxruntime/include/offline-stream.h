#ifndef OFFLINE_STREAM_H
#define OFFLINE_STREAM_H

#include <memory>
#include <string>
#include <map>
#include "model.h"
#include "punc-model.h"
#include "vad-model.h"
#include "itn-model.h"

namespace funasr {
class OfflineStream {
  public:
    OfflineStream(std::map<std::string, std::string>& model_path, int thread_num);
    ~OfflineStream(){};

    std::unique_ptr<VadModel> vad_handle= nullptr;
    std::unique_ptr<Model> asr_handle= nullptr;
    std::unique_ptr<PuncModel> punc_handle= nullptr;
    std::unique_ptr<ITNModel> itn_handle = nullptr;
    bool UseVad(){return use_vad;};
    bool UsePunc(){return use_punc;}; 
    bool UseITN(){return use_itn;};
    
  private:
    bool use_vad=false;
    bool use_punc=false;
    bool use_itn=false;
};

OfflineStream *CreateOfflineStream(std::map<std::string, std::string>& model_path, int thread_num=1);
} // namespace funasr
#endif
