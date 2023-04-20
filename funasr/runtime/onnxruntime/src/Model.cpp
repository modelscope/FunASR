#include "precomp.h"

Model *create_model(const char *path, int nThread, bool quantize, bool use_vad)
{
    Model *mm;

    mm = new paraformer::ModelImp(path, nThread, quantize, use_vad);

    return mm;
}
