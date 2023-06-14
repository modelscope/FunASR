/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  wav_frontend.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef wav_frontend_hpp
#define wav_frontend_hpp

#include <stdio.h>
#include "Tensor.h"
#include "params_config.h"

namespace paraformer_online {

class WavFrontend {
public:
    WavFrontend();
    ~WavFrontend();
    void forward(float *wav, int len, Tensor<float> *&outputs);
    int wav_segments_; // debug
private:
    
    bool is_first_input_;
    Tensor<float> *input_cache_;
    Tensor<float> *lfr_splice_cache_;
    void forward_fbank(float *wav, int len, Tensor<float> *&feats);
    void apply_lfr(Tensor<float> *&feats);
    void apply_cmvn(Tensor<float>* feats);
};


}

#endif /* wav_frontend_hpp */
