/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023 by burkliu(刘柏基) liubaiji@xverse.cn */

#include <string>
#include <thread>
#include <unistd.h>

#include "grpcpp/server_builder.h"
#include "paraformer.grpc.pb.h"
#include "funasrruntime.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "glog/logging.h"

using paraformer::WavFormat;
using paraformer::DecodeMode;
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
  void DecodeThreadFunc();
  void OnSpeechStart();
  void OnSpeechData();
  void OnSpeechEnd();

  grpc::ServerReaderWriter<Response, Request>* stream_;
  std::shared_ptr<Request> request_;
  std::shared_ptr<Response> response_;
  std::shared_ptr<FUNASR_HANDLE> asr_handler_;
  std::string audio_buffer_;
  std::shared_ptr<std::thread> decode_thread_ = nullptr;
  bool is_start_ = false;
  bool is_end_ = false;

  std::vector<int> chunk_size_ = {5, 10, 5};
  int sampling_rate_ = 16000;
  std::string encoding_;
  ASR_TYPE mode_ = ASR_TWO_PASS;
  int step_duration_ms_ = 100;
};

class GrpcService final : public ASR::Service {
  public:
    GrpcService(std::map<std::string, std::string>& config, int num_thread);
    grpc::Status Recognize(grpc::ServerContext* context, grpc::ServerReaderWriter<Response, Request>* stream);

  private:
    std::map<std::string, std::string> config_;
    std::shared_ptr<FUNASR_HANDLE> asr_handler_;
};
