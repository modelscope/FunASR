#ifndef OFFLINE_STREAM_H
#define OFFLINE_STREAM_H

#include <memory>
#include <string>
#include <map>
#include "model.h"
#include "punc-model.h"
#include "vad-model.h"
#include "cam-sv-model.h"
#if !defined(__APPLE__)
#include "itn-model.h"
#endif

namespace funasr {
class OfflineStream {
  public:
    OfflineStream(std::map<std::string, std::string>& model_path, int thread_num, bool use_gpu=false, int batch_size=1);
    ~OfflineStream(){};

    std::unique_ptr<VadModel> vad_handle= nullptr;
    std::unique_ptr<Model> asr_handle= nullptr;
    std::unique_ptr<PuncModel> punc_handle= nullptr;
    std::unique_ptr<SvModel> sv_handle = nullptr;
#if !defined(__APPLE__)
    std::unique_ptr<ITNModel> itn_handle = nullptr;
#endif
    bool UseVad(){return use_vad;};
    bool UsePunc(){return use_punc;}; 
    bool UseITN(){return use_itn;};
    bool UseSv(){return use_sv;};
    
  private:
    bool use_vad=false;
    bool use_punc=false;
    bool use_itn=false;
    bool use_sv=false;
};

OfflineStream *CreateOfflineStream(std::map<std::string, std::string>& model_path, int thread_num=1, bool use_gpu=false, int batch_size=1);
} // namespace funasr
#endif
