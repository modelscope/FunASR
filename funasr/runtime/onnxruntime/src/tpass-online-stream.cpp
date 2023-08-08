#include "precomp.h"
#include <unistd.h>

namespace funasr {
TpassOnlineStream::TpassOnlineStream(TpassStream* tpass_stream, std::vector<int> chunk_size){
    TpassStream* tpass_obj = (TpassStream*)tpass_stream;
    if(tpass_obj->vad_handle){
        vad_online_handle = make_unique<FsmnVadOnline>((FsmnVad*)(tpass_obj->vad_handle).get());
    }else{
        LOG(ERROR)<<"asr_handle is null";
        exit(-1);
    }

    if(tpass_obj->asr_handle){
        asr_online_handle = make_unique<ParaformerOnline>((Paraformer*)(tpass_obj->asr_handle).get(), chunk_size);
    }else{
        LOG(ERROR)<<"asr_handle is null";
        exit(-1);
    }
}

TpassOnlineStream* CreateTpassOnlineStream(void* tpass_stream, std::vector<int> chunk_size)
{
    TpassOnlineStream *mm;
    mm =new TpassOnlineStream((TpassStream*)tpass_stream, chunk_size);
    return mm;
}

} // namespace funasr