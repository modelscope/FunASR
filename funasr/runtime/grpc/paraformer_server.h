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
#include "libfunasrapi.h"


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

typedef struct
{
    std::string msg;
    float  snippet_time;
}FUNASR_RECOG_RESULT;


class ASRServicer final : public ASR::Service {
  private:
    int init_flag;

  public:
    ASRServicer(const char* model_path, int thread_num, bool quantize);
    grpc::Status Recognize(grpc::ServerContext* context, grpc::ServerReaderWriter<Response, Request>* stream);
    FUNASR_HANDLE AsrHanlde;
	
};
