/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
//
//  attention.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef attention_hpp
#define attention_hpp

#include <stdio.h>
#include "Tensor.h"
#include "model_params.hpp"
//#include "matrix.h"

//using namespace QNN_NS;
namespace paraformer_online {

typedef struct {
    float *data;
    int h;
    int w;
    int c;
} Matrix;

class MultiHeadedAttentionSANM {
    
public:
    MultiHeadedAttentionSANM(EncSelfAttnParams *params);
    ~MultiHeadedAttentionSANM();
    void forward(Tensor<float> *&feats, int *conv_im2col);
    
private:
    EncSelfAttnParams *params;
    
    void forward_qkv(Tensor<float> *din, Tensor<float> *dout,
                     float *weight, float *bias);
    void forward_fsmn(Tensor<float> *din, int *conv_im2col);
};



class MultiHeadedAttentionSANMDecoder {
public:
    MultiHeadedAttentionSANMDecoder(SubDecoderParams *params);
    ~MultiHeadedAttentionSANMDecoder();
    
    void forward_chunk(Tensor<float> *&feats, int *conv_im2col);
    
private:
    SubDecoderParams *params;
    
    Tensor<float> *fsmn_cache_;
    
    void forward_fsmn(Tensor<float> *din, Tensor<float> *fsmn_out);
    
    Matrix weights_m;
    
};



class MultiHeadedAttentionCrossAtt {
public:
    MultiHeadedAttentionCrossAtt(DecSelfAttnParams *params);
    ~MultiHeadedAttentionCrossAtt();
    void forward(Tensor<float> *&feats, Tensor<float> *enc);
    
private:
    DecSelfAttnParams *params;
    
    void linear_forward(Tensor<float> *din, Tensor<float> *dout, float *weight,
                     float *bias);
};

}

#endif /* attention_hpp */
