#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include <unordered_map>
#include <chrono>

#include "paraformer.grpc.pb.h"
#include "librapidasrapi.h"


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;


using paraformer::Request;
using paraformer::Response;
using paraformer::ASR;


class ASRServicer final : public ASR::Service {
  private:
    int init_flag;
    std::unordered_map<std::string, std::string> client_buffers;
    std::unordered_map<std::string, std::string> client_transcription;

  public:
    ASRServicer();
    void clear_states(const std::string& user);
    void clear_buffers(const std::string& user);
    void clear_transcriptions(const std::string& user);
    void disconnect(const std::string& user);
    grpc::Status Recognize(grpc::ServerContext* context, grpc::ServerReaderWriter<Response, Request>* stream);
    int nThreadNum = 4;
    RPASR_HANDLE AsrHanlde=RapidAsrInit("/data/asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/", nThreadNum);
	
};
