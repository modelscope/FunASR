/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  sanm_encoder.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef sanm_encoder_hpp
#define sanm_encoder_hpp

#include <stdio.h>
#include "Tensor.h"
#include "model_params.hpp"
#include "layer_norm.hpp"
#include "attention.hpp"
#include "feed_forward.hpp"
#include "params_config.h"

namespace paraformer_online {

class EncoderLayerSANM {
public:
    EncoderLayerSANM(SubEncoderParams *params, int size);
    ~EncoderLayerSANM();
    void forward(Tensor<float> *&feats, int *conv_im2col);
    
private:
    SubEncoderParams *params;
    LayerNorm *norm1_;
    MultiHeadedAttentionSANM *self_attn_;
    LayerNorm *norm2_;
    FeedForward *feedforward_;
};


class SANMEncoder {
    
public:
    SANMEncoder(EncoderParams *params);
    ~SANMEncoder();
    void forward_chunk(Tensor<float> *&feats);
    
private:
    EncoderParams *params;
    EncoderLayerSANM *encoders0_;
    EncoderLayerSANM *encoders_[encoder_num_blocks-1];
    LayerNorm *after_norm_;
    int *conv_im2col;
    
    Tensor<float> *encode_cache_;
    int start_idx_cache_;
    
    void get_poscode(Tensor<float> *poscode);
    void get_conv_im2col(int mm);
    void add_overlap_chunk(Tensor<float> *&feats);
};

}

#endif /* sanm_encoder_hpp */
