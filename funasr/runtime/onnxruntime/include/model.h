
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
    virtual std::string Forward(float *din, int len, int flag, const std::vector<std::vector<float>> &hw_emb) = 0;
    virtual std::string Rescoring() = 0;
    virtual void InitHwCompiler(const std::string &hw_model, int thread_num) = 0;
    virtual void InitSegDict(const std::string &seg_dict_model) = 0;
    virtual std::vector<std::vector<float>> CompileHotwordEmbedding(std::string &hotwords) = 0;
};

Model *CreateModel(std::map<std::string, std::string>& model_path,int thread_num=1);
} // namespace funasr
#endif
