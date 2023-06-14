/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  model_infer.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "model_infer.hpp"
#include "util.h"

namespace paraformer_online {

ModelInfer::ModelInfer(const char *path) {
    string wenet_path = pathAppend(path, "model.bin");
    string vocab_path = pathAppend(path, "config.yaml");
//    string mvn_name = "am.mvn";
//    string mvn_path = pathAppend(path, mvn_name);
    
    p_helper_ = new ModelParamsHelper(wenet_path.c_str(), 500);
    
    frontend_ = new WavFrontend();
    encoder_ = new SANMEncoder(&p_helper_->params.encoder);
    predictor_ = new CifPredictorV2(&p_helper_->params.predictor);
    decoder_ = new ParaformerSANMDecoder(&p_helper_->params.decoder);

    vocab_ = new funasr::Vocab(vocab_path.c_str());
    
}

ModelInfer::~ModelInfer() {
    delete p_helper_;
    delete frontend_;
    delete encoder_;
    delete predictor_;
    delete decoder_;
    delete vocab_;
}

string ModelInfer::greedy_search(Tensor<float> *&feats)
{
    vector<int> hyps;
    int Tmax = feats->size[2];
    int i;
    for (i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        findmax(feats->buff + i * 8404, 8404, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab_->Vector2StringV2(hyps);
}

string ModelInfer::forward(float *wav, int len) {
    Tensor<float> *feats = nullptr;
    frontend_->forward(wav, len, feats);
    encoder_->forward_chunk(feats); 
    Tensor<float> enc_out(feats);
    predictor_->forward_chunk(feats);
    
    if (feats->size[2] < 1) {
        delete feats;
        return "";
    }

    decoder_->forward_chunk(feats, &enc_out);

    string result = greedy_search(feats);
    delete feats;
    std::cout << "result: " << result << std::endl;
    
    return result;
}

}
