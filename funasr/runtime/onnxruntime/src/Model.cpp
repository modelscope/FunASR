#include "precomp.h"

Model *create_model(const char *path, int nThread, bool quantize, bool use_vad, bool use_punc)
{
    Model *mm;

    mm = new paraformer::ModelImp(path, nThread, quantize, use_vad, use_punc);

    return mm;
}
