/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  model_infer.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef model_infer_hpp
#define model_infer_hpp

#include <stdio.h>
#include <string>
#include "Tensor.h"
#include "wav_frontend.hpp"
#include "sanm_encoder.hpp"
#include "cif.hpp"
#include "model_params.hpp"
#include "Vocab.h"
#include "sanm_decoder.hpp"

namespace paraformer_online {

class ModelInfer {
    
public:
    ModelInfer(const char *path);
    ~ModelInfer();
    string forward(float *wav, int len);
    
private:
    WavFrontend *frontend_;
    SANMEncoder *encoder_;
    CifPredictorV2 *predictor_;
    ParaformerSANMDecoder *decoder_;
    
    ModelParamsHelper *p_helper_;
    funasr::Vocab *vocab_;
    
    string greedy_search(Tensor<float> *&feats);
};

}

#endif /* model_infer_hpp */
