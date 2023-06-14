/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  layer_norm.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef layer_norm_hpp
#define layer_norm_hpp

#include <stdio.h>
#include "Tensor.h"
#include "model_params.hpp"

namespace paraformer_online {
class LayerNorm {
  private:
    NormParams *params;
    float error;
    int layer_size;
    void mean_var(float *din, float &mean, float &var);
    void norm(float *din, float mean, float var);

  public:
    LayerNorm(NormParams *params, float error, int layer_size);
    ~LayerNorm();
    void forward(Tensor<float> *din);
};
}

#endif /* layer_norm_hpp */
