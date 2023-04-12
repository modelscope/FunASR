
#ifndef FEATUREEXTRACT_H
#define FEATUREEXTRACT_H

#include <fftw3.h>
#include <stdint.h>

#include "FeatureQueue.h"
#include "SpeechWrap.h"
#include "Tensor.h"

class FeatureExtract {
  private:
    SpeechWrap speech;
    FeatureQueue fqueue;
    int mode;
    int fft_size = 512;
    int window_size = 400;
    int window_shift = 160;

    //void fftw_init();
    void melspect(float *din, float *dout);
    void global_cmvn(float *din);

  public:
    FeatureExtract(int mode);
    ~FeatureExtract();
    int size();
    //int status();
    void reset();
    void insert(fftwf_plan plan, float *din, int len, int flag);
    bool fetch(Tensor<float> *&dout);
};

#endif
