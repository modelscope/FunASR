#include "paraformer-server.h"

using paraformer::Request;
using paraformer::Response;
using paraformer::ASR;

GrpcEngine::GrpcEngine(
  grpc::ServerReaderWriter<Response, Request>* stream,
  std::shared_ptr<FUNASR_HANDLE> asr_handler)
  : stream_(std::move(stream)),
    asr_handler_(std::move(asr_handler)) {}

void GrpcEngine::operator()() {
  Request request;
  while (stream_->Read(&request)) {
    Response respond;
    respond.set_user(request.user());
    respond.set_language(request.language());

    if (request.isend()) {
      std::cout << "asr end" << std::endl;
      respond.set_sentence(R"({"success": true, "detail": "asr end"})");
      respond.set_action("terminate");
      stream_->Write(respond);
    } else if (request.speaking()) {
      if (request.audio_data().size() > 0) {
        auto& buf = client_buffers[request.user()];
        buf.insert(buf.end(), request.audio_data().begin(), request.audio_data().end());
      }
      respond.set_sentence(R"({"success": true, "detail": "speaking"})");
      respond.set_action("speaking");
      stream_->Write(respond);
    } else {
      if (client_buffers.count(request.user()) == 0 && request.audio_data().size() == 0) {
        respond.set_sentence(R"({"success": true, "detail": "waiting_for_voice"})");
        respond.set_action("waiting");
        stream_->Write(respond);
      } else {
        auto begin_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (request.audio_data().size() > 0) {
          auto& buf = client_buffers[request.user()];
          buf.insert(buf.end(), request.audio_data().begin(), request.audio_data().end());
        }
        std::string tmp_data = this->client_buffers[request.user()];

        int data_len_int = tmp_data.length();
        std::string data_len = std::to_string(data_len_int);
        std::stringstream ss;
        ss << R"({"success": true, "detail": "decoding data: )" << data_len << R"( bytes")"  << R"("})";

        respond.set_sentence(ss.str());
        respond.set_action("decoding");
        stream_->Write(respond);

        // start recoginize
        std::string asr_result;
        if (tmp_data.length() < 800) { //min input_len for asr model
          asr_result = "";
          std::cout << "error: data_is_not_long_enough" << std::endl;
        } else {
          FUNASR_RESULT result = FunOfflineInferBuffer(*asr_handler_, tmp_data.c_str(), data_len_int, RASR_NONE, NULL, 16000);
          asr_result = ((FUNASR_RECOG_RESULT*) result)->msg;
        }

        auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::string delay_str = std::to_string(end_time - begin_time);
        std::cout << "user: " << request.user() << " , delay(ms): " << delay_str << ", text: " << asr_result << std::endl;
        std::stringstream ss2;
        ss2 << R"({"success": true, "detail": "finish_sentence","server_delay_ms":)" << delay_str << R"(,"text":")" << asr_result << R"("})";

        respond.set_sentence(ss2.str());
        respond.set_action("finish");
        stream_->Write(respond);
      }
    }
  }
}

GrpcService::GrpcService(std::map<std::string, std::string>& config, int num_thread)
  : config_(config) {

  asr_handler_ = std::make_shared<FUNASR_HANDLE>(std::move(FunOfflineInit(config_, num_thread)));
  std::cout << "GrpcService model loades" << std::endl;
}

grpc::Status GrpcService::Recognize(
  grpc::ServerContext* context,
  grpc::ServerReaderWriter<Response, Request>* stream) {

  LOG(INFO) << "Get Recognize request" << std::endl;
  GrpcEngine engine(
    stream,
    asr_handler_
  );

  std::thread t(std::move(engine));
  t.join();
  return grpc::Status::OK;
}

void GetValue(TCLAP::ValueArg<std::string>& value_arg, std::string key, std::map<std::string, std::string>& config) {
  if (value_arg.isSet()) {
    config.insert({key, value_arg.getValue()});
    LOG(INFO) << key << " : " << value_arg.getValue();
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  TCLAP::CmdLine cmd("paraformer-server", ' ', "1.0");
  TCLAP::ValueArg<std::string> model_dir("", MODEL_DIR, "the asr model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
  TCLAP::ValueArg<std::string> quantize("", QUANTIZE, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "false", "string");
  TCLAP::ValueArg<std::string> vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
  TCLAP::ValueArg<std::string> vad_quant("", VAD_QUANT, "false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "false", "string");
  TCLAP::ValueArg<std::string> punc_dir("", PUNC_DIR, "the punc model path, which contains model.onnx, punc.yaml", false, "", "string");
  TCLAP::ValueArg<std::string> punc_quant("", PUNC_QUANT, "false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "false", "string");
  TCLAP::ValueArg<std::string> port_id("", PORT_ID, "port id", true, "", "string");

  cmd.add(model_dir);
  cmd.add(quantize);
  cmd.add(vad_dir);
  cmd.add(vad_quant);
  cmd.add(punc_dir);
  cmd.add(punc_quant);
  cmd.add(port_id);
  cmd.parse(argc, argv);

  std::map<std::string, std::string> config;
  GetValue(model_dir, MODEL_DIR, config);
  GetValue(quantize, QUANTIZE, config);
  GetValue(vad_dir, VAD_DIR, config);
  GetValue(vad_quant, VAD_QUANT, config);
  GetValue(punc_dir, PUNC_DIR, config);
  GetValue(punc_quant, PUNC_QUANT, config);
  GetValue(port_id, PORT_ID, config);

  std::string port;
  try {
    port = config.at(PORT_ID);
  } catch(std::exception const &e) {
    std::cout << ("Error when read port.") << std::endl;
    exit(0);
  }
  std::string server_address;
  server_address = "0.0.0.0:" + port;
  GrpcService service(config, 1);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();

  return 0;
}
