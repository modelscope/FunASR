
#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <map>

class Model {
  public:
    virtual ~Model(){};
    virtual void Reset() = 0;
    virtual std::string ForwardChunk(float *din, int len, int flag) = 0;
    virtual std::string Forward(float *din, int len, int flag) = 0;
    virtual std::string Rescoring() = 0;
    virtual std::vector<std::vector<int>> VadSeg(std::vector<float>& pcm_data)=0;
    virtual std::string AddPunc(const char* sz_input)=0;
    virtual bool UseVad() =0;
    virtual bool UsePunc() =0; 
};

Model *CreateModel(std::map<std::string, std::string>& model_path,int thread_num=1);
#endif
