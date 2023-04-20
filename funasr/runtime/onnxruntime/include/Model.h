
#ifndef MODEL_H
#define MODEL_H

#include <string>

class Model {
  public:
    virtual ~Model(){};
    virtual void reset() = 0;
    virtual std::string forward_chunk(float *din, int len, int flag) = 0;
    virtual std::string forward(float *din, int len, int flag) = 0;
    virtual std::string rescoring() = 0;
    virtual std::vector<std::vector<int>> vad_seg(std::vector<float>& pcm_data)=0;
};

Model *create_model(const char *path,int nThread=0,bool quantize=false, bool use_vad=false);
#endif
