/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// io server
// Usage:funasr-ws-server  [--model_thread_num <int>] [--decoder_thread_num <int>]
//                    [--io_thread_num <int>] [--port <int>] [--listen_ip
//                    <string>] [--punc-quant <string>] [--punc-dir <string>]
//                    [--vad-quant <string>] [--vad-dir <string>] [--quantize
//                    <string>] --model-dir <string> [--] [--version] [-h]
#include "websocket-server.h"

using namespace std;
void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key,
              std::map<std::string, std::string>& model_path) {
    model_path.insert({key, value_arg.getValue()});
    LOG(INFO) << key << " : " << value_arg.getValue();
}
int main(int argc, char* argv[]) {
  try {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-ws-server", ' ', "1.0");
    TCLAP::ValueArg<std::string> model_dir(
        "", MODEL_DIR,
        "default: /workspace/models/asr, the asr model path, which contains model.onnx, config.yaml, am.mvn",
        false, "/workspace/models/asr", "string");
    TCLAP::ValueArg<std::string> quantize(
        "", QUANTIZE,
        "true (Default), load the model of model.onnx in model_dir. If set "
        "true, load the model of model_quant.onnx in model_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> vad_dir(
        "", VAD_DIR,
        "default: /workspace/models/vad, the vad model path, which contains model.onnx, vad.yaml, vad.mvn",
        false, "/workspace/models/vad", "string");
    TCLAP::ValueArg<std::string> vad_quant(
        "", VAD_QUANT,
        "true (Default), load the model of model.onnx in vad_dir. If set "
        "true, load the model of model_quant.onnx in vad_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> punc_dir(
        "", PUNC_DIR,
        "default: /workspace/models/punc, the punc model path, which contains model.onnx, punc.yaml", 
        false, "/workspace/models/punc",
        "string");
    TCLAP::ValueArg<std::string> punc_quant(
        "", PUNC_QUANT,
        "true (Default), load the model of model.onnx in punc_dir. If set "
        "true, load the model of model_quant.onnx in punc_dir",
        false, "true", "string");

    TCLAP::ValueArg<std::string> listen_ip("", "listen_ip", "listen_ip", false,
                                           "0.0.0.0", "string");
    TCLAP::ValueArg<int> port("", "port", "port", false, 8889, "int");
    TCLAP::ValueArg<int> io_thread_num("", "io_thread_num", "io_thread_num",
                                       false, 8, "int");
    TCLAP::ValueArg<int> decoder_thread_num(
        "", "decoder_thread_num", "decoder_thread_num", false, 8, "int");
    TCLAP::ValueArg<int> model_thread_num("", "model_thread_num",
                                          "model_thread_num", false, 1, "int");

    TCLAP::ValueArg<std::string> certfile("", "certfile", "certfile", false, "../../../ssl_key/server.crt",
                                          "string");
    TCLAP::ValueArg<std::string> keyfile("", "keyfile", "keyfile", false, "../../../ssl_key/server.key",
                                         "string");

    cmd.add(certfile);
    cmd.add(keyfile);

    cmd.add(model_dir);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_quant);

    cmd.add(listen_ip);
    cmd.add(port);
    cmd.add(io_thread_num);
    cmd.add(decoder_thread_num);
    cmd.add(model_thread_num);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(model_dir, MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);

    std::string s_listen_ip = listen_ip.getValue();
    int s_port = port.getValue();
    int s_io_thread_num = io_thread_num.getValue();
    int s_decoder_thread_num = decoder_thread_num.getValue();

    int s_model_thread_num = model_thread_num.getValue();

    asio::io_context io_decoder;  // context for decoding
    asio::io_context io_server;   // context for server

    std::vector<std::thread> decoder_threads;

    std::string s_certfile = certfile.getValue();
    std::string s_keyfile = keyfile.getValue();

    bool is_ssl = false;
    if (!s_certfile.empty()) {
      is_ssl = true;
    }

    auto conn_guard = asio::make_work_guard(
        io_decoder);  // make sure threads can wait in the queue
    auto server_guard = asio::make_work_guard(
        io_server);  // make sure threads can wait in the queue
    // create threads pool
    for (int32_t i = 0; i < s_decoder_thread_num; ++i) {
      decoder_threads.emplace_back([&io_decoder]() { io_decoder.run(); });
    }

    server server_;  // server for websocket
    wss_server wss_server_;
    if (is_ssl) {
      wss_server_.init_asio(&io_server);  // init asio
      wss_server_.set_reuse_addr(
          true);  // reuse address as we create multiple threads

      // list on port for accept
      wss_server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);
      WebSocketServer websocket_srv(
          io_decoder, is_ssl, nullptr, &wss_server_, s_certfile,
          s_keyfile);  // websocket server for asr engine
      websocket_srv.initAsr(model_path, s_model_thread_num);  // init asr model

    } else {
      server_.init_asio(&io_server);  // init asio
      server_.set_reuse_addr(
          true);  // reuse address as we create multiple threads

      // list on port for accept
      server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);
      WebSocketServer websocket_srv(
          io_decoder, is_ssl, &server_, nullptr, s_certfile,
          s_keyfile);  // websocket server for asr engine
      websocket_srv.initAsr(model_path, s_model_thread_num);  // init asr model
    }

    std::cout << "asr model init finished. listen on port:" << s_port
              << std::endl;

    // Start the ASIO network io_service run loop
    std::vector<std::thread> ts;
    // create threads for io network
    for (size_t i = 0; i < s_io_thread_num; i++) {
      ts.emplace_back([&io_server]() { io_server.run(); });
    }
    // wait for theads
    for (size_t i = 0; i < s_io_thread_num; i++) {
      ts[i].join();
    }

    // wait for theads
    for (auto& t : decoder_threads) {
      t.join();
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}