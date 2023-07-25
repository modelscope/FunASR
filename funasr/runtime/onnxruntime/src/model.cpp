#include "precomp.h"

namespace funasr {
Model *CreateModel(std::map<std::string, std::string>& model_path, int thread_num, int mode)
{
    if(mode == 0){
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
    }else if(mode == 1){
        // online
        string en_model_path;
        string de_model_path;
        string am_cmvn_path;
        string am_config_path;

        en_model_path = PathAppend(model_path.at(MODEL_DIR), "encoder.onnx");
        de_model_path = PathAppend(model_path.at(MODEL_DIR), "decoder.onnx");
        am_cmvn_path = PathAppend(model_path.at(MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(MODEL_DIR), AM_CONFIG_NAME);

        Model *mm;
        mm = new ParaformerOnline();
        mm->InitAsr(en_model_path, de_model_path, am_cmvn_path, am_config_path, thread_num);
        return mm;
    }
}

} // namespace funasr