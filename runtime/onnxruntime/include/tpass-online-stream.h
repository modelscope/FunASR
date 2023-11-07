#ifndef TPASS_ONLINE_STREAM_H
#define TPASS_ONLINE_STREAM_H

#include <memory>
#include "tpass-stream.h"
#include "model.h"
#include "vad-model.h"

namespace funasr {
class TpassOnlineStream {
  public:
    TpassOnlineStream(TpassStream* tpass_stream, std::vector<int> chunk_size);
    ~TpassOnlineStream(){};

    std::unique_ptr<VadModel> vad_online_handle = nullptr;
    std::unique_ptr<Model> asr_online_handle = nullptr;
};
TpassOnlineStream* CreateTpassOnlineStream(void* tpass_stream, std::vector<int> chunk_size);
} // namespace funasr
#endif
