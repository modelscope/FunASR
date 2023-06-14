/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  feed_forward.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "feed_forward.hpp"
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif

#include "util.h"
#include "params_config.h"

namespace paraformer_online {

FeedForward::FeedForward(FeedForwardParams *params, int active_type)
    : params(params)
{
    if (active_type == 0) {
        activate = &relu;
    } else {
        activate = &swish;
    }
}

FeedForward::~FeedForward()
{
}

void FeedForward::forward(Tensor<float> *din)
{
    int nn = din->size[3];
    int mm = din->buff_size / nn;
    int i;
    Tensor<float> tmp(mm, linear_units);

    for (i = 0; i < mm; i++) {
        int offset = i * linear_units;
        memcpy(tmp.buff + offset, params->w1_bias, linear_units * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, linear_units, output_size, 1,
                din->buff, output_size, params->w1_weight, output_size, 1, tmp.buff, linear_units);

    activate(&tmp);

    for (i = 0; i < mm; i++) {
        int offset = i * output_size;
        memcpy(din->buff + offset, params->w2_bias, output_size * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, output_size, linear_units, 1,
                tmp.buff, linear_units, params->w2_weight, linear_units, 1, din->buff, output_size);
}





FeedForwardDecoder::FeedForwardDecoder(DecoderFeedForwardParams *params) : params(params) {
    activate = &relu;
    norm = new LayerNorm(&params->norm, 1e-12, linear_units);
}

FeedForwardDecoder::~FeedForwardDecoder() {
    delete norm;
}

void FeedForwardDecoder::forward(Tensor<float> *din) {
    int nn = din->size[3];
    int mm = din->buff_size / nn;
    int i;
    Tensor<float> tmp(mm, linear_units);

    for (i = 0; i < mm; i++) {
        int offset = i * linear_units;
        memcpy(tmp.buff + offset, params->w1_bias, linear_units * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, linear_units, output_size, 1,
                din->buff, output_size, params->w1_weight, output_size, 1, tmp.buff, linear_units);

    activate(&tmp);
    norm->forward(&tmp);

    din->zeros();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, output_size, linear_units, 1,
                tmp.buff, linear_units, params->w2_weight, linear_units, 1, din->buff, output_size);
}

}
