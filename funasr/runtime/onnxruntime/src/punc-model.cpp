#include "precomp.h"

PuncModel *CreatePuncModel(std::map<std::string, std::string>& model_path, int thread_num)
{
    PuncModel *mm;
    mm = new CTTransformer();

    string punc_model_path;
    string punc_config_path;

    punc_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
    if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
        punc_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
    }
    punc_config_path = PathAppend(model_path.at(MODEL_DIR), PUNC_CONFIG_NAME);

    mm->InitPunc(punc_model_path, punc_config_path, thread_num);
    return mm;
}
