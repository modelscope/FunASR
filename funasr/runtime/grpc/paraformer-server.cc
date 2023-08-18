/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023 by burkliu(刘柏基) liubaiji@xverse.cn */

#include "paraformer-server.h"

GrpcEngine::GrpcEngine(
  grpc::ServerReaderWriter<Response, Request>* stream,
  std::shared_ptr<FUNASR_HANDLE> asr_handler)
  : stream_(std::move(stream)),
    asr_handler_(std::move(asr_handler)) {

  request_ = std::make_shared<Request>();
}

void GrpcEngine::DecodeThreadFunc() {
  FUNASR_HANDLE tpass_online_handler = FunTpassOnlineInit(*asr_handler_, chunk_size_);
  int step = (sampling_rate_ * step_duration_ms_ / 1000) * 2; // int16 = 2bytes;
  std::vector<std::vector<std::string>> punc_cache(2);

  bool is_final = false;
  std::string online_result = "";
  std::string tpass_result = "";

  LOG(INFO) << "Decoder init, start decoding loop with mode";

  while (true) {
    if (audio_buffer_.length() > step || is_end_) {
      if (audio_buffer_.length() <= step && is_end_) {
        is_final = true;
        step = audio_buffer_.length();
      }

      FUNASR_RESULT result = FunTpassInferBuffer(*asr_handler_,
                                                 tpass_online_handler,
                                                 audio_buffer_.c_str(),
                                                 step,
                                                 punc_cache,
                                                 is_final,
                                                 sampling_rate_,
                                                 encoding_,
                                                 mode_);
      audio_buffer_ = audio_buffer_.substr(step);

      if (result) {
        std::string online_message = FunASRGetResult(result, 0);
        online_result += online_message;
        if(online_message != ""){
          Response response;
          response.set_mode(DecodeMode::online);
          response.set_text(online_message);
          response.set_is_final(is_final);
          stream_->Write(response);
          LOG(INFO) << "send online results: " << online_message;
        }
        std::string tpass_message = FunASRGetTpassResult(result, 0);
        tpass_result += tpass_message;
        if(tpass_message != ""){
          Response response;
          response.set_mode(DecodeMode::two_pass);
          response.set_text(tpass_message);
          response.set_is_final(is_final);
          stream_->Write(response);
          LOG(INFO) << "send offline results: " << tpass_message;
        }
        FunASRFreeResult(result);
      }

      if (is_final) {
        FunTpassOnlineUninit(tpass_online_handler);
        break;
      }
    }
    sleep(0.001);
  }
}

void GrpcEngine::OnSpeechStart() {
  if (request_->chunk_size_size() == 3) {
    for (int i = 0; i < 3; i++) {
      chunk_size_[i] = int(request_->chunk_size(i));
    }
  }
  std::string chunk_size_str;
  for (int i = 0; i < 3; i++) {
    chunk_size_str = " " + chunk_size_[i];
  }
  LOG(INFO) << "chunk_size is" << chunk_size_str;

  if (request_->sampling_rate() != 0) {
    sampling_rate_ = request_->sampling_rate();
  }
  LOG(INFO) << "sampling_rate is " << sampling_rate_;

  switch(request_->wav_format()) {
    case WavFormat::pcm: encoding_ = "pcm";
  }
  LOG(INFO) << "encoding is " << encoding_;

  std::string mode_str;
  switch(request_->mode()) {
    case DecodeMode::offline:
      mode_ = ASR_OFFLINE;
      mode_str = "offline";
      break;
    case DecodeMode::online:
      mode_ = ASR_ONLINE;
      mode_str = "online";
      break;
    case DecodeMode::two_pass:
      mode_ = ASR_TWO_PASS;
      mode_str = "two_pass";
      break;
  }
  LOG(INFO) << "decode mode is " << mode_str;
  
  decode_thread_ = std::make_shared<std::thread>(&GrpcEngine::DecodeThreadFunc, this);
  is_start_ = true;
}

void GrpcEngine::OnSpeechData() {
  audio_buffer_ += request_->audio_data();
}

void GrpcEngine::OnSpeechEnd() {
  is_end_ = true;
  LOG(INFO) << "Read all pcm data, wait for decoding thread";
  if (decode_thread_ != nullptr) {
    decode_thread_->join();
  }
}

void GrpcEngine::operator()() {
  try {
    LOG(INFO) << "start engine main loop";
    while (stream_->Read(request_.get())) {
      LOG(INFO) << "receive data";
      if (!is_start_) {
        OnSpeechStart();
      }
      OnSpeechData();
      if (request_->is_final()) {
        break;
      }
    }
    OnSpeechEnd();
    LOG(INFO) << "Connect finish";
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

GrpcService::GrpcService(std::map<std::string, std::string>& config, int onnx_thread)
  : config_(config) {

  asr_handler_ = std::make_shared<FUNASR_HANDLE>(std::move(FunTpassInit(config_, onnx_thread)));
  LOG(INFO) << "GrpcService model loaded";

  std::vector<int> chunk_size = {5, 10, 5};
  FUNASR_HANDLE tmp_online_handler = FunTpassOnlineInit(*asr_handler_, chunk_size);
  int sampling_rate = 16000;
  int buffer_len = sampling_rate * 1;
  std::string tmp_data(buffer_len, '0');
  std::vector<std::vector<std::string>> punc_cache(2);
  bool is_final = true;
  std::string encoding = "pcm";
  FUNASR_RESULT result = FunTpassInferBuffer(*asr_handler_,
                                             tmp_online_handler,
                                             tmp_data.c_str(),
                                             buffer_len,
                                             punc_cache,
                                             is_final,
                                             buffer_len,
                                             encoding,
                                             ASR_TWO_PASS);
  if (result) {
      FunASRFreeResult(result);
  }
  FunTpassOnlineUninit(tmp_online_handler);
  LOG(INFO) << "GrpcService model warmup";
}

grpc::Status GrpcService::Recognize(
  grpc::ServerContext* context,
  grpc::ServerReaderWriter<Response, Request>* stream) {
  LOG(INFO) << "Get Recognize request";
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
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);

  TCLAP::CmdLine cmd("funasr-onnx-2pass", ' ', "1.0");
  TCLAP::ValueArg<std::string>  model_dir("", MODEL_DIR, "the asr offline model path, which contains model.onnx, config.yaml, am.mvn", true, "", "string");
  TCLAP::ValueArg<std::string>  online_model_dir("", ONLINE_MODEL_DIR, "the asr online model path, which contains encoder.onnx, decoder.onnx, config.yaml, am.mvn", true, "", "string");
  TCLAP::ValueArg<std::string>  quantize("", QUANTIZE, "false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir", false, "false", "string");
  TCLAP::ValueArg<std::string>  vad_dir("", VAD_DIR, "the vad online model path, which contains model.onnx, vad.yaml, vad.mvn", false, "", "string");
  TCLAP::ValueArg<std::string>  vad_quant("", VAD_QUANT, "false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir", false, "true", "string");
  TCLAP::ValueArg<std::string>  punc_dir("", PUNC_DIR, "the punc online model path, which contains model.onnx, punc.yaml", false, "", "string");
  TCLAP::ValueArg<std::string>  punc_quant("", PUNC_QUANT, "false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir", false, "true", "string");
  TCLAP::ValueArg<std::int32_t>  onnx_thread("", "onnx-inter-thread", "onnxruntime SetIntraOpNumThreads", false, 1, "int32_t");
  TCLAP::ValueArg<std::string> port_id("", PORT_ID, "port id", true, "", "string");

  cmd.add(model_dir);
  cmd.add(online_model_dir);
  cmd.add(quantize);
  cmd.add(vad_dir);
  cmd.add(vad_quant);
  cmd.add(punc_dir);
  cmd.add(punc_quant);
  cmd.add(onnx_thread);
  cmd.add(port_id);
  cmd.parse(argc, argv);

  std::map<std::string, std::string> config;
  GetValue(model_dir, MODEL_DIR, config);
  GetValue(online_model_dir, ONLINE_MODEL_DIR, config);
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
    LOG(INFO) << ("Error when read port.");
    exit(0);
  }
  std::string server_address;
  server_address = "0.0.0.0:" + port;
  GrpcService service(config, onnx_thread);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;
  server->Wait();

  return 0;
}
