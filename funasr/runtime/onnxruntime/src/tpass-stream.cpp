#include "precomp.h"
#include <unistd.h>

namespace funasr {
TpassStream::TpassStream(std::map<std::string, std::string>& model_path, int thread_num)
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
    if(model_path.find(OFFLINE_MODEL_DIR) != model_path.end() && model_path.find(ONLINE_MODEL_DIR) != model_path.end()){
        // 2pass
        string am_model_path;
        string en_model_path;
        string de_model_path;
        string am_cmvn_path;
        string am_config_path;

        am_model_path = PathAppend(model_path.at(OFFLINE_MODEL_DIR), MODEL_NAME);
        en_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), ENCODER_NAME);
        de_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), DECODER_NAME);
        if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
            am_model_path = PathAppend(model_path.at(OFFLINE_MODEL_DIR), QUANT_MODEL_NAME);
            en_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), QUANT_ENCODER_NAME);
            de_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), QUANT_DECODER_NAME);
        }
        am_cmvn_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), AM_CONFIG_NAME);

        asr_handle = make_unique<Paraformer>();
        asr_handle->InitAsr(am_model_path, en_model_path, de_model_path, am_cmvn_path, am_config_path, thread_num);
    }else{
        LOG(ERROR) <<"Can not find offline-model-dir or online-model-dir";
        exit(-1);
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
            punc_online_handle = make_unique<CTTransformerOnline>();
            punc_online_handle->InitPunc(punc_model_path, punc_config_path, thread_num);
            use_punc = true;
        }
    }
}

TpassStream *CreateTpassStream(std::map<std::string, std::string>& model_path, int thread_num)
{
    TpassStream *mm;
    mm = new TpassStream(model_path, thread_num);
    return mm;
}

void CreateTpassOnlineStream(void* tpass_stream)
{
    funasr::TpassStream* tpass_obj = (funasr::TpassStream*)tpass_stream;
    if(tpass_obj->vad_handle){
        tpass_obj->vad_online_handle = make_unique<FsmnVadOnline>((FsmnVad*)(tpass_obj->vad_handle).get());
    }

    if(tpass_obj->asr_handle){
        tpass_obj->asr_online_handle = make_unique<ParaformerOnline>((Paraformer*)(tpass_obj->asr_handle).get());
    }else{
        LOG(ERROR)<<"asr_handle is null";
        exit(-1);
    }
}

} // namespace funasr