#include "precomp.h"

namespace funasr {
Model *CreateModel(std::map<std::string, std::string>& model_path, int thread_num, ASR_TYPE type)
{
    // offline
    if(type == ASR_OFFLINE){
        string am_model_path;
        string am_cmvn_path;
        string am_config_path;

        am_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
        if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
            am_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
        }
        am_cmvn_path = PathAppend(model_path.at(MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(MODEL_DIR), AM_CONFIG_NAME);

        Model *mm;
        mm = new Paraformer();
        mm->InitAsr(am_model_path, am_cmvn_path, am_config_path, thread_num);
        return mm;
    }else if(type == ASR_ONLINE){
        // online
        string en_model_path;
        string de_model_path;
        string am_cmvn_path;
        string am_config_path;

        en_model_path = PathAppend(model_path.at(MODEL_DIR), ENCODER_NAME);
        de_model_path = PathAppend(model_path.at(MODEL_DIR), DECODER_NAME);
        if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
            en_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_ENCODER_NAME);
            de_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_DECODER_NAME);
        }
        am_cmvn_path = PathAppend(model_path.at(MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(MODEL_DIR), AM_CONFIG_NAME);

        Model *mm;
        mm = new Paraformer();
        mm->InitAsr(en_model_path, de_model_path, am_cmvn_path, am_config_path, thread_num);
        return mm;
    }else{
        LOG(ERROR)<<"Wrong ASR_TYPE : " << type;
        exit(-1);
    }
}

Model *CreateModel(void* asr_handle)
{
    Model* mm;
    mm = new ParaformerOnline((Paraformer*)asr_handle);
    return mm;
}

} // namespace funasr