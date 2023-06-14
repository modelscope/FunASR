/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  sanm_encoder.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "sanm_encoder.hpp"
#include <math.h>

namespace paraformer_online {

/* EncoderLayerSANM */

EncoderLayerSANM::EncoderLayerSANM(SubEncoderParams *params, int size) : params(params) {
    norm1_ = new LayerNorm(&params->norm1, 1e-12, size);
    self_attn_ = new MultiHeadedAttentionSANM(&params->self_attn);
    norm2_ = new LayerNorm(&params->norm2, 1e-12, output_size);
    feedforward_ = new FeedForward(&params->feedforward, 0);
}

EncoderLayerSANM::~EncoderLayerSANM() {
    delete norm1_;
    delete self_attn_;
    delete norm2_;
    delete feedforward_;
}

void EncoderLayerSANM::forward(Tensor<float> *&feats, int *conv_im2col) {
    int in_size = feats->size[3];
    Tensor<float> residual(feats);
    norm1_->forward(feats);
    
    self_attn_->forward(feats, conv_im2col);
    
    if (in_size == output_size) {
        feats->add(&residual);
    }
    
    Tensor<float> residual2(feats);
    norm2_->forward(feats);
    feedforward_->forward(feats);
    feats->add(&residual2);
}




/* SANMEncoder */
SANMEncoder::SANMEncoder(EncoderParams *params) : params(params) {
    encoders0_ = new EncoderLayerSANM(&params->sub_encoders0, lfr_m*n_mels);
    int i;
    for (i = 0; i < encoder_num_blocks-1; i++) {
        encoders_[i] = new EncoderLayerSANM(&params->sub_encoders[i], output_size);
    }
    after_norm_ = new LayerNorm(&params->after_norm, 1e-12, output_size);
    
    // TODO: write a fixed constant, equal to encoder input shape
    encode_cache_ = new Tensor<float>(10, lfr_m*n_mels);
    start_idx_cache_ = 0;
    conv_im2col = NULL;
}

SANMEncoder::~SANMEncoder() {
    free(conv_im2col);
    delete encoders0_;
    for (int i = 0; i < encoder_num_blocks-1; i++) {
        delete encoders_[i];
    }
    delete after_norm_;
    delete encode_cache_;
}

void SANMEncoder::get_conv_im2col(int mm)
{
    int idxs_size = mm * kernel_size;
    if (conv_im2col != NULL)
        free(conv_im2col);
    conv_im2col = (int *)malloc(sizeof(int) * idxs_size);
    int step = output_size;
    int i, j;
    int ii = 0;
    for (i = 0; i < mm; i++) {
        int start_idx = -floor(kernel_size/2) + i;
        for (j = 0; j < kernel_size; j++) {
            int val = start_idx + j;
            if (val >= 0 && val < mm)
                conv_im2col[ii++] = val * step;
            else
                conv_im2col[ii++] = -1;
        }
    }
}

void SANMEncoder::get_poscode(Tensor<float> *poscode)
{
    int timesteps = poscode->size[2];
    int feat_dim = poscode->size[3];
    int start_idx = start_idx_cache_;
    start_idx_cache_ += timesteps;
    int mm = start_idx_cache_;

    int i;
    float scale = -0.0330119726594128;
    
    Tensor<float> tmp(mm, lfr_m*n_mels);
    
    for (i = 0; i < feat_dim/2; i++) {
        float tmptime = exp(i * scale);
        int j;
        for (j = 0; j < mm; j++) {
            int sin_idx = j * feat_dim + i;
            int cos_idx = j * feat_dim + i + feat_dim/2;
            float coe = tmptime * (j + 1);
            tmp.buff[sin_idx] = sin(coe);
            tmp.buff[cos_idx] = cos(coe);
        }
    }
    
    for (i = start_idx; i < start_idx + timesteps; i++) {
        for (int j = 0; j < feat_dim; j++) {
            poscode->buff[(i-start_idx)*feat_dim+j] = tmp.buff[i*feat_dim+j];
        }
    }
}

void SANMEncoder::add_overlap_chunk(Tensor<float> *&feats) {
    int mm = feats->size[2];
    // pad cache
    mm = feats->size[2] + encode_cache_->size[2];
    int ch = feats->size[3];
    
    Tensor<float> *tmp = new Tensor<float>(mm, ch);
    memcpy(tmp->buff, encode_cache_->buff, encode_cache_->buff_size*sizeof(float));
    memcpy(tmp->buff+encode_cache_->buff_size, feats->buff, feats->buff_size*sizeof(float));
    
    // TODO: Support dynamics cache
    memcpy(encode_cache_->buff, feats->buff, feats->buff_size*sizeof(float));
    
    delete feats;
    feats = tmp;
}

void SANMEncoder::forward_chunk(Tensor<float> *&feats) {
    int mm = feats->size[2];
    int f_dim = feats->size[3];
    Tensor<float> poscode(mm, f_dim);
    
    get_poscode(&poscode);
    feats->add(&poscode);
    add_overlap_chunk(feats);
    get_conv_im2col(feats->size[2]);
    encoders0_->forward(feats, conv_im2col);
    
    int i;
    for (i = 0; i < encoder_num_blocks-1; i++) {
        encoders_[i]->forward(feats, conv_im2col);
    }
    
    after_norm_->forward(feats);
}

}
