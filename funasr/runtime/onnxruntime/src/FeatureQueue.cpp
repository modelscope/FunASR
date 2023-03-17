#include "precomp.h"
FeatureQueue::FeatureQueue()
{
    buff = new Tensor<float>(67, 80);
    window_size = 67;
    buff_idx = 0;
}

FeatureQueue::~FeatureQueue()
{
    delete buff;
}

void FeatureQueue::reinit(int size)
{
    delete buff;
    buff = new Tensor<float>(size, 80);
    buff_idx = 0;
    window_size = size;
}

void FeatureQueue::reset()
{
    buff_idx = 0;
}

void FeatureQueue::push(float *din, int flag)
{
    int offset = buff_idx * 80;
    memcpy(buff->buff + offset, din, 80 * sizeof(float));
    buff_idx++;

    if (flag == S_END) {
        Tensor<float> *tmp = new Tensor<float>(buff_idx, 80);
        memcpy(tmp->buff, buff->buff, buff_idx * 80 * sizeof(float));
        feature_queue.push(tmp);
        buff_idx = 0;
    } else if (buff_idx == window_size) {
        feature_queue.push(buff);
        Tensor<float> *tmp = new Tensor<float>(window_size, 80);
        memcpy(tmp->buff, buff->buff + (window_size - 3) * 80,
               3 * 80 * sizeof(float));
        buff_idx = 3;
        buff = tmp;
    }
}

Tensor<float> *FeatureQueue::pop()
{

    Tensor<float> *tmp = feature_queue.front();
    feature_queue.pop();
    return tmp;
}

int FeatureQueue::size()
{
    return feature_queue.size();
}
