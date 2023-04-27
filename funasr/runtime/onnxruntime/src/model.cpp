#include "precomp.h"

Model *CreateModel(std::map<std::string, std::string>& model_path, int thread_num)
{
    Model *mm;
    mm = new paraformer::Paraformer(model_path, thread_num);
    return mm;
}
