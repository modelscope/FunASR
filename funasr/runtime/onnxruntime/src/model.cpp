#include "precomp.h"

Model *CreateModel(const char *path, int thread_num, bool quantize, bool use_vad, bool use_punc)
{
    Model *mm;

    mm = new paraformer::Paraformer(path, thread_num, quantize, use_vad, use_punc);

    return mm;
}
