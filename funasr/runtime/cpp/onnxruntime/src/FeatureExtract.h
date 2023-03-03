
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

    float *fft_input;
    fftwf_complex *fft_out;
    fftwf_plan p;

    void fftw_init();
    void melspect(float *din, float *dout);
    void global_cmvn(float *din);

  public:
    FeatureExtract(int mode);
    ~FeatureExtract();
    int size();
    int status();
    void reset();
    void insert(float *din, int len, int flag);
    bool fetch(Tensor<float> *&dout);
};

#endif
