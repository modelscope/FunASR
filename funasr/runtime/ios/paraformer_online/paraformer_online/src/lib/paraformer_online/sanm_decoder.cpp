/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  sanm_decoder.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "sanm_decoder.hpp"
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif

#include "util.h"

namespace paraformer_online {

DecoderLayerSANM::DecoderLayerSANM(SubDecoderParams *params) : params(params) {
    norm1 = new LayerNorm(&params->norm1, 1e-12, output_size);
    feedforward = new FeedForwardDecoder(&params->feedforward);
    norm2 = new LayerNorm(&params->norm2, 1e-12, output_size);
    self_attn = new MultiHeadedAttentionSANMDecoder(params);
    norm3 = new LayerNorm(&params->norm3, 1e-12, output_size);
    src_attn = new MultiHeadedAttentionCrossAtt(&params->src_attn);
}

DecoderLayerSANM::~DecoderLayerSANM() {
    delete norm1;
    delete feedforward;
    delete norm2;
    delete self_attn;
    delete norm3;
    delete src_attn;
}

void DecoderLayerSANM::forward(Tensor<float> *&feats, Tensor<float> *enc, int *conv_im2col) {
    Tensor<float> residual(feats);
    norm1->forward(feats);
    feedforward->forward(feats);
    norm2->forward(feats);
    self_attn->forward_chunk(feats, conv_im2col);
    feats->add(&residual);
    residual.reload(feats);
    norm3->forward(feats);
    src_attn->forward(feats, enc);
    feats->add(&residual);
}



/* ParaformerSANMDecoder */
ParaformerSANMDecoder::ParaformerSANMDecoder(DecoderParams *params) : params(params) {
    int i;
    for (i = 0; i < decoder_num_blocks; i++) {
        sub_decoders[i] = new DecoderLayerSANM(&params->sub_decoders[i]);
    }

    decoder3_norm = new LayerNorm(&params->sub_decoders3.norm1, 1e-12, output_size);
    feedforward = new FeedForwardDecoder(&params->sub_decoders3.feedforward);
    after_norm = new LayerNorm(&params->after_norm, 1e-12, output_size);
    
    conv_im2col = NULL;
}

ParaformerSANMDecoder::~ParaformerSANMDecoder() {
    int i;
    for (i = 0; i < decoder_num_blocks; i++) {
        delete sub_decoders[i];
    }

    delete decoder3_norm;
    delete feedforward;
    delete after_norm;
}

void ParaformerSANMDecoder::get_conv_im2col(int mm)
{
    mm += (kernel_size-1);
    int idxs_size = mm * kernel_size;
    if (conv_im2col != NULL)
        free(conv_im2col);
    conv_im2col = (int *)malloc(sizeof(int) * idxs_size);
    int step = output_size;
    int i, j;
    int ii = 0;
    for (i = 0; i < mm; i++) {
        int start_idx = i;
        for (j = 0; j < kernel_size; j++) {
            int val = start_idx + j;
            conv_im2col[ii++] = val * step;
        }
    }
}

void ParaformerSANMDecoder::forward_chunk(Tensor<float> *&feats, Tensor<float> *enc) {
    int mm = feats->size[2];
    get_conv_im2col(mm);
    
    int i;
    for (i = 0; i < decoder_num_blocks; i++) {
        sub_decoders[i]->forward(feats, enc, conv_im2col);
    }

    decoder3_norm->forward(feats);
    feedforward->forward(feats);
    after_norm->forward(feats);

    Tensor<float> *tmp = new Tensor<float>(mm, token_list);

    for (i = 0; i < mm; i++) {
        int offset = i * token_list;
        memcpy(tmp->buff + offset, params->linear_out_bias,
               token_list * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, token_list, 512, 1,
                feats->buff, 512, params->linear_out_weight, 512, 1, tmp->buff,
                8404);

    int j;
    for (j = 0; j < mm; j++) {
        int offset = j * 8404;
        log_softmax(tmp->buff + offset, 8404);
    }
    delete feats;
    free(conv_im2col);
    conv_im2col = NULL;
    feats = tmp;
    
}

}
