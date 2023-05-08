#include "precomp.h"

Model *CreateModel(std::map<std::string, std::string>& model_path, int thread_num)
{
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
    mm = new paraformer::Paraformer();
    mm->InitAsr(am_model_path, am_cmvn_path, am_config_path, thread_num);
    return mm;
}
