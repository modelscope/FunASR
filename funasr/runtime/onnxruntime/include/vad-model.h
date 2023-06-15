
#ifndef VAD_MODEL_H
#define VAD_MODEL_H

#include <string>
#include <map>
#include <vector>

namespace funasr {
class VadModel {
  public:
    virtual ~VadModel(){};
    virtual void InitVad(const std::string &vad_model, const std::string &vad_cmvn, const std::string &vad_config, int thread_num)=0;
    virtual std::vector<std::vector<int>> Infer(std::vector<float> &waves, bool input_finished=true)=0;
};

VadModel *CreateVadModel(std::map<std::string, std::string>& model_path, int thread_num);
VadModel *CreateVadModel(void* fsmnvad_handle);
} // namespace funasr
#endif
