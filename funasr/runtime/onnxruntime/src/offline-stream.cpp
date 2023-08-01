#include "precomp.h"
#include <unistd.h>

namespace funasr {
OfflineStream::OfflineStream(std::map<std::string, std::string>& model_path, int thread_num)
{
    // VAD model
    if(model_path.find(VAD_DIR) != model_path.end()){
        string vad_model_path;
        string vad_cmvn_path;
        string vad_config_path;
    
        vad_model_path = PathAppend(model_path.at(VAD_DIR), MODEL_NAME);
        if(model_path.find(VAD_QUANT) != model_path.end() && model_path.at(VAD_QUANT) == "true"){
            vad_model_path = PathAppend(model_path.at(VAD_DIR), QUANT_MODEL_NAME);
        }
        vad_cmvn_path = PathAppend(model_path.at(VAD_DIR), VAD_CMVN_NAME);
        vad_config_path = PathAppend(model_path.at(VAD_DIR), VAD_CONFIG_NAME);
        if (access(vad_model_path.c_str(), F_OK) != 0 ||
            access(vad_cmvn_path.c_str(), F_OK) != 0 ||
            access(vad_config_path.c_str(), F_OK) != 0 )
        {
            LOG(INFO) << "VAD model file is not exist, skip load vad model.";
        }else{
            vad_handle = make_unique<FsmnVad>();
            vad_handle->InitVad(vad_model_path, vad_cmvn_path, vad_config_path, thread_num);
            use_vad = true;
        }
    }

    // AM model
    if(model_path.find(MODEL_DIR) != model_path.end()){
        string am_model_path;
        string am_cmvn_path;
        string am_config_path;
    
        am_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
        if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
            am_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
        }
        am_cmvn_path = PathAppend(model_path.at(MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(MODEL_DIR), AM_CONFIG_NAME);

        asr_handle = make_unique<Paraformer>();
        asr_handle->InitAsr(am_model_path, am_cmvn_path, am_config_path, thread_num);
    }

    // PUNC model
    if(model_path.find(PUNC_DIR) != model_path.end()){
        string punc_model_path;
        string punc_config_path;
    
        punc_model_path = PathAppend(model_path.at(PUNC_DIR), MODEL_NAME);
        if(model_path.find(PUNC_QUANT) != model_path.end() && model_path.at(PUNC_QUANT) == "true"){
            punc_model_path = PathAppend(model_path.at(PUNC_DIR), QUANT_MODEL_NAME);
        }
        punc_config_path = PathAppend(model_path.at(PUNC_DIR), PUNC_CONFIG_NAME);

        if (access(punc_model_path.c_str(), F_OK) != 0 ||
            access(punc_config_path.c_str(), F_OK) != 0 )
        {
            LOG(INFO) << "PUNC model file is not exist, skip load punc model.";
        }else{
            punc_handle = make_unique<CTTransformer>();
            punc_handle->InitPunc(punc_model_path, punc_config_path, thread_num);
            use_punc = true;
        }
    }
}

OfflineStream *CreateOfflineStream(std::map<std::string, std::string>& model_path, int thread_num)
{
    OfflineStream *mm;
    mm = new OfflineStream(model_path, thread_num);
    return mm;
}

} // namespace funasr