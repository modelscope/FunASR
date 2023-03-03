
#ifndef SPEECHWRAP_H
#define SPEECHWRAP_H

#include <stdint.h>

class SpeechWrap {
  private:
    float cache[400];
    int cache_size;
    float *in;
    int in_size;
    int total_size;
    int next_cache_size;

  public:
    SpeechWrap();
    ~SpeechWrap();
    void load(float *din, int len);
    void update(int offset);
    void reset();
    int size();
    float &operator[](int i);
};

#endif
