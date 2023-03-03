#include "precomp.h"

Model *create_model(const char *path,int nThread)
{
    Model *mm;


    mm = new paraformer::ModelImp(path, nThread);

    return mm;
}
