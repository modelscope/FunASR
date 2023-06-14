/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
//
//  cif.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "cif.hpp"
#include <vector>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif
#include "util.h"

namespace paraformer_online {

CifPredictorV2::CifPredictorV2(PredictorParams *params) : params(params) {
    cif_hidden_cache_ = new Tensor<float>(512);
    cif_alphas_cache_ = new Tensor<float>(1);
    for (int i = 0; i < 512; i++) {
        cif_hidden_cache_->buff[i] = 0.f;
    }
    cif_alphas_cache_->buff[0] = 0.f;
    
}

CifPredictorV2::~CifPredictorV2() {
    delete cif_hidden_cache_;
    delete cif_alphas_cache_;
}

void CifPredictorV2::get_conv_im2col(int mm)
{
    int idxs_size = mm * 3;
    conv_im2col = (int *)malloc(sizeof(int) * idxs_size);
    int step = 512;
    int i, j;
    int ii = 0;
    for (i = 0; i < mm; i++) {
        int start_idx = -1 + i;
        for (j = 0; j < 3; j++) {
            int val = start_idx + j;
            if (val >= 0 && val < mm)
                conv_im2col[ii++] = val * step;
            else
                conv_im2col[ii++] = -1;
        }
    }
}

void CifPredictorV2::cif_conv1d(Tensor<float> *&din)
{
    int mm = din->size[2];
    int v_offset = 0;

    Tensor<float> blasin(mm, 3);
    Tensor<float> *blasout = new Tensor<float>(mm, 512);
    int i, j;

    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        memcpy(blasout->buff + offset, params->cif_conv1d_bias,
               sizeof(float) * 512);
    }

    for (i = 0; i < 512; i++) {
        for (j = 0; j < mm * 3; j++) {
            int tmp_idx = conv_im2col[j];
            if (tmp_idx == -1)
                blasin.buff[j] = 0;
            else
                blasin.buff[j] = din->buff[tmp_idx + v_offset];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 3, 1,
                    blasin.buff, 3, params->cif_conv1d_weight + i * 512 * 3, 3,
                    1, blasout->buff, 512);

        v_offset++;
    }

    delete din;
    din = blasout;
}

void CifPredictorV2::forward_chunk(Tensor<float> *&din)
{
    int mm = din->size[2];
//    int nn = din->size[3];

    Tensor<float> alphas(mm, 1);
    Tensor<float> hidden(din);

    get_conv_im2col(mm);
    cif_conv1d(din);
    relu(din);

    int i, j;
    for (i = 0; i < mm; i++) {
        alphas.buff[i] = params->cif_output_bias[0];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 1, 512, 1,
                din->buff, 512, params->cif_output_weight, 512, 1, alphas.buff,
                1);

    sigmoid(&alphas);

//    relu(&alphas);
    
    // cache
    int chunk_size_pre = 5;
    for (i = 0; i < chunk_size_pre; i++)
        alphas.buff[i] = 0.0;
    
    bool is_final = false;
    int chunk_size_suf = 15;
    for (i = chunk_size_suf; i < alphas.buff_size; i++)
        alphas.buff[i] = 0.0;
    
    Tensor<float> alphas_catted(mm+cif_alphas_cache_->size[3], 1);
    Tensor<float> hidden_catted(mm+cif_hidden_cache_->size[2], hidden.size[3]); // (21, 512)
    
    memcpy(hidden_catted.buff, cif_hidden_cache_->buff, cif_hidden_cache_->buff_size*sizeof(float));
    memcpy(hidden_catted.buff+cif_hidden_cache_->buff_size, hidden.buff, hidden.buff_size*sizeof(float));
    alphas_catted.buff[0] = cif_alphas_cache_->buff[0];
    memcpy(alphas_catted.buff+1, alphas.buff, alphas.buff_size*sizeof(float));
    
    if (is_final) { // TODD: finish final part
        
    }
    
    mm = alphas_catted.buff_size;
    
    int frames_size = hidden_catted.size[3];
    Tensor<float> frames(frames_size);
    frames.zeros();

    std::vector<float> list_fire;
    std::vector<Tensor<float> *> list_frame;
    
    float threshold = 1.0;
    float intergrate = 0;
    for (i = 0; i < mm; i++) {
        float alpha = alphas_catted.buff[i];
        if (alpha + intergrate < threshold) {
            intergrate += alpha;
            list_fire.push_back(intergrate);
            for (j = 0; j < frames_size; j++) {
                frames.buff[j] += alpha * hidden_catted.buff[i*frames_size+j];
            }
        } else {
            for (j = 0; j < frames_size; j++) {
                frames.buff[j] += (threshold - intergrate) * hidden_catted.buff[i*frames_size+j];
            }
            Tensor<float> *tmp_frame = new Tensor<float>(frames_size);
            for (j = 0; j < frames_size; j++) {
                tmp_frame->buff[j] = frames.buff[j];
            }
            list_frame.push_back(tmp_frame);
            intergrate += alpha;
            list_fire.push_back(intergrate);
            intergrate -= threshold;
            for (j = 0; j < frames_size; j++) {
                frames.buff[j] = intergrate * hidden_catted.buff[i*frames_size+j];
            }
        }
    }
    
    // cache
    cif_alphas_cache_->buff[0] = intergrate;
    if (intergrate > 0.0) {
        for (i = 0; i < frames_size; i++) {
            cif_hidden_cache_->buff[i] = frames.buff[i] / intergrate;
        }
    } else {
        for (i = 0; i < frames_size; i++) {
            cif_hidden_cache_->buff[i] = frames.buff[i];
        }
    }
    
    int token_length = (int)list_frame.size();
    
    Tensor<float> *tout = new Tensor<float>(token_length, frames_size);
    tout->zeros();
    if (token_length == 0) {
        delete din;
        din = tout;
    }
    else {
        for (i = 0; i < token_length; i++) {
            Tensor<float> *t_frame = list_frame[i];
            for (j = 0; j < frames_size; j++) {
                tout->buff[i*frames_size+j] = t_frame->buff[j];
            }
            delete t_frame;
        }
        delete din;
        din = tout;
    }
}

}
