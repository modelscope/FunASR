
#ifndef MODEL_H
#define MODEL_H

#include <string>

class Model {
  public:
    virtual ~Model(){};
    virtual void Reset() = 0;
    virtual std::string ForwardChunk(float *din, int len, int flag) = 0;
    virtual std::string Forward(float *din, int len, int flag) = 0;
    virtual std::string Rescoring() = 0;
    virtual std::vector<std::vector<int>> VadSeg(std::vector<float>& pcm_data)=0;
    virtual std::string AddPunc(const char* sz_input)=0;
};

Model *CreateModel(const char *path,int thread_num=1,bool quantize=false, bool use_vad=false, bool use_punc=false);
#endif
