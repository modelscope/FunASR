
#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <map>
#include "funasrruntime.h"
namespace funasr {
class Model {
  public:
    virtual ~Model(){};
    virtual void Reset() = 0;
    virtual void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){};
    virtual void InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){};
    virtual void InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){};
    virtual std::string Forward(float *din, int len, bool input_finished){return "";};
    virtual std::string Rescoring() = 0;
};

Model *CreateModel(std::map<std::string, std::string>& model_path, int thread_num=1, ASR_TYPE type=ASR_OFFLINE);
Model *CreateModel(void* asr_handle, std::vector<int> chunk_size);

} // namespace funasr
#endif
