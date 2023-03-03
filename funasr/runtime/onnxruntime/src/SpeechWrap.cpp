#include "precomp.h"

SpeechWrap::SpeechWrap()
{
    cache_size = 0;
}

SpeechWrap::~SpeechWrap()
{
}

void SpeechWrap::reset()
{
    cache_size = 0;
}

void SpeechWrap::load(float *din, int len)
{
    in = din;
    in_size = len;
    total_size = cache_size + in_size;
}

int SpeechWrap::size()
{
    return total_size;
}

void SpeechWrap::update(int offset)
{
    int in_offset = offset - cache_size;
    cache_size = (total_size - offset);
    memcpy(cache, in + in_offset, cache_size * sizeof(float));
}

float &SpeechWrap::operator[](int i)
{
    return i < cache_size ? cache[i] : in[i - cache_size];
}
