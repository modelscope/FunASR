
#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <map>
namespace funasr {
class Model {
  public:
    virtual ~Model(){};
    virtual void Reset() = 0;
    virtual void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num)=0;
    virtual std::string ForwardChunk(float *din, int len, int flag) = 0;
    virtual std::string Forward(float *din, int len, int flag) = 0;
    virtual std::string Rescoring() = 0;
};

Model *CreateModel(std::map<std::string, std::string>& model_path,int thread_num=1);
} // namespace funasr
#endif
