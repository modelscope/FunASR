#include "precomp.h"

Model *create_model(const char *path, int nThread, bool quantize)
{
    Model *mm;

    mm = new paraformer::ModelImp(path, nThread, quantize);

    return mm;
}
