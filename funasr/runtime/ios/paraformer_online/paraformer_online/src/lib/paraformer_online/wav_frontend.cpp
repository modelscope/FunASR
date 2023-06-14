/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  wav_frontend.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "wav_frontend.hpp"
#include "feat_computer.h"
#include "predefine_coe.h"

const int input_cache_len = 320;
//const int n_mels = 80;
//const int lfr_n = 6;
//const int lfr_m = 7;

namespace paraformer_online {

WavFrontend::WavFrontend() {
    input_cache_ = new Tensor<float>(input_cache_len);
    input_cache_->zeros();
    lfr_splice_cache_ = new Tensor<float>((lfr_m-1)/2, n_mels);
    is_first_input_ = true;
    
    wav_segments_ = 0;
}

WavFrontend::~WavFrontend() {
    delete input_cache_;
    delete lfr_splice_cache_;
}

void WavFrontend::forward_fbank(float *wav, int len, Tensor<float> *&feats) {
    kaldi::SubVector<kaldi::BaseFloat> speech_vector(wav, len);
    kaldi::Matrix<kaldi::BaseFloat> output;
    kaldi::BaseFloat sample_rate = sampling_rate;
    
    FeatComputer computer;
    computer.compute_signals(speech_vector, &output, sample_rate);
    
    int rows = output.NumRows();
    int cols = output.NumCols();
    
    Tensor<float> *t_feats = new Tensor<float>(rows, cols);
    memcpy(t_feats->buff, output.Data(), rows*cols*sizeof(float));
    feats = t_feats;
}

void WavFrontend::apply_lfr(Tensor<float> *&feats)
{
    int mm = feats->size[2];
    int ll = ceil((mm - (lfr_m - 1) / 2.0) / lfr_n);
    Tensor<float> *tmp = new Tensor<float>(ll, lfr_m*n_mels);
    int i, j;
    int out_offset = 0;
    for (i = 0; i < ll; i++) {
        for (j = 0; j < lfr_m; j++) {
            int idx = i * lfr_n + j;
            if (idx < 0) {
                idx = 0;
            }
            if (idx >= mm) {
                idx = mm - 1;
            }
            memcpy(tmp->buff + out_offset, feats->buff + idx * n_mels,
                   sizeof(float) * n_mels);
            out_offset += n_mels;
        }
    }
    delete feats;
    feats = tmp;
}

void WavFrontend::apply_cmvn(Tensor<float> *feats)
{
    const float *var;
    const float *mean;
  
    int m = feats->size[2];
    int n = feats->size[3];

    var = (const float *)paraformer_cmvn_var_hex;
    mean = (const float *)paraformer_cmvn_mean_hex;
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            int idx = i * n + j;
            feats->buff[idx] = (feats->buff[idx] + mean[j]) * var[j];
        }
    }
}

void WavFrontend::forward(float *wav, int len, Tensor<float> *&outputs) {
    float *wav_cache;
    if (is_first_input_) {
        wav_cache = wav;
    } else {
        wav_cache = new float[len+input_cache_len];
        memcpy(wav_cache, input_cache_->buff, input_cache_len*sizeof(float));
        memcpy(wav_cache+input_cache_len, wav, len*sizeof(float));
        len += input_cache_len;
    }
    
    Tensor<float> *feats = nullptr;
    forward_fbank(wav_cache, len, feats);
    if (!is_first_input_) {
        delete wav_cache;
    }
    
    int t_lfr = is_first_input_ ? (lfr_m - 1) / 2 : 1;
    if (is_first_input_) { // splice cache
        for (int i = 0; i < t_lfr; i++) {
            memcpy(lfr_splice_cache_->buff+i*n_mels, feats->buff, n_mels*sizeof(float));
        }
    }
    
    Tensor<float> use_to_cache_feats(feats);
    
    // cat(cache, feats)
    Tensor<float> *catted_feats = new Tensor<float>(t_lfr+feats->size[2], n_mels);
    memcpy(catted_feats->buff, lfr_splice_cache_->buff, lfr_splice_cache_->buff_size*sizeof(float));
    memcpy(catted_feats->buff+lfr_splice_cache_->buff_size, feats->buff, feats->buff_size*sizeof(float));
    delete feats;
    
    apply_lfr(catted_feats);
    
    // cache lfr splice
    if (is_first_input_) {
        delete lfr_splice_cache_;
        lfr_splice_cache_ = new Tensor<float>(n_mels);
    }
    memcpy(lfr_splice_cache_->buff, use_to_cache_feats.buff+(use_to_cache_feats.size[2]-1)*n_mels, n_mels*sizeof(float));
    
    apply_cmvn(catted_feats);
    
    outputs = catted_feats;
    is_first_input_ = false;
}

}
