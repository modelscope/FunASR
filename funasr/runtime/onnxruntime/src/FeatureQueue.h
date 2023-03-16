
#ifndef FEATUREQUEUE_H
#define FEATUREQUEUE_H

#include "Tensor.h"
#include <queue>
#include <stdint.h>
using namespace std;


class FeatureQueue {
  private:
    queue<Tensor<float> *> feature_queue;
    Tensor<float> *buff;
    int buff_idx;
    int window_size;

  public:
    FeatureQueue();
    ~FeatureQueue();
    void reinit(int size);
    void reset();
    void push(float *din, int flag);
    Tensor<float> *pop();
    int size();
};

#endif
