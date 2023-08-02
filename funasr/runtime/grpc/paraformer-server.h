#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "paraformer.grpc.pb.h"
#include "funasrruntime.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "glog/logging.h"

using paraformer::Request;
using paraformer::Response;
using paraformer::ASR;

typedef struct
{
  std::string msg;
  float  snippet_time;
} FUNASR_RECOG_RESULT;

class GrpcEngine {
 public:
  GrpcEngine(grpc::ServerReaderWriter<Response, Request>* stream, std::shared_ptr<FUNASR_HANDLE> asr_handler);
  void operator()();

 private:
  grpc::ServerReaderWriter<Response, Request>* stream_;
  std::shared_ptr<FUNASR_HANDLE> asr_handler_;
  std::unordered_map<std::string, std::string> client_buffers;
};

class GrpcService final : public ASR::Service {
  public:
    GrpcService(std::map<std::string, std::string>& config, int num_thread);
    grpc::Status Recognize(grpc::ServerContext* context, grpc::ServerReaderWriter<Response, Request>* stream);

  private:
    std::map<std::string, std::string> config_;
    std::shared_ptr<FUNASR_HANDLE> asr_handler_;
};
