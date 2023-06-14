/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
//
//  cif.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef cif_hpp
#define cif_hpp

#include <stdio.h>
#include "Tensor.h"
#include "model_params.hpp"

namespace paraformer_online {

class CifPredictorV2 {
    
public:
    CifPredictorV2(PredictorParams *params);
    ~CifPredictorV2();
    
    void forward_chunk(Tensor<float> *&feats);
    
private:
    PredictorParams *params;
    
    void cif_conv1d(Tensor<float> *&din);
    int *conv_im2col;
    
    Tensor<float> *cif_hidden_cache_;
    Tensor<float> *cif_alphas_cache_;

    void get_conv_im2col(int mm);
};

}

#endif /* cif_hpp */
