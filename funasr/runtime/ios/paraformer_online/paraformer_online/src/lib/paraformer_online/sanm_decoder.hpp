/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  sanm_decoder.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef sanm_decoder_hpp
#define sanm_decoder_hpp

#include <stdio.h>
#include "Tensor.h"
#include "model_params.hpp"
#include "layer_norm.hpp"
#include "feed_forward.hpp"
#include "attention.hpp"
#include "params_config.h"

namespace paraformer_online {

class DecoderLayerSANM {
public:
    DecoderLayerSANM(SubDecoderParams *params);
    ~DecoderLayerSANM();
    
    void forward(Tensor<float> *&feats, Tensor<float> *enc, int *conv_im2col);
    
private:
    SubDecoderParams *params;
    LayerNorm *norm1;
    FeedForwardDecoder *feedforward;
    LayerNorm *norm2;
    MultiHeadedAttentionSANMDecoder *self_attn;
    LayerNorm *norm3;
    MultiHeadedAttentionCrossAtt *src_attn;
};



class ParaformerSANMDecoder {
public:
    ParaformerSANMDecoder(DecoderParams *params);
    ~ParaformerSANMDecoder();
    
    void forward_chunk(Tensor<float> *&feats, Tensor<float> *enc);
private:
    DecoderParams *params;
    DecoderLayerSANM *sub_decoders[decoder_num_blocks];
    
    // 后面都放到attention
    LayerNorm *decoder3_norm;
    LayerNorm *after_norm;
    FeedForwardDecoder *feedforward;

    int *conv_im2col;
    void get_conv_im2col(int mm);
};

}

#endif /* sanm_decoder_hpp */
