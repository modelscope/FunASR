/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// io server
// Usage:funasr-wss-server  [--model_thread_num <int>] [--decoder_thread_num <int>]
//                    [--io_thread_num <int>] [--port <int>] [--listen_ip
//                    <string>] [--punc-quant <string>] [--punc-dir <string>]
//                    [--vad-quant <string>] [--vad-dir <string>] [--quantize
//                    <string>] --model-dir <string> [--] [--version] [-h]
#include "websocket-server.h"
#include <unistd.h>

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

    TCLAP::CmdLine cmd("funasr-wss-server", ' ', "1.0");
    TCLAP::ValueArg<std::string> download_model_dir(
        "", "download-model-dir",
        "Download model from Modelscope to download_model_dir",
        false, "", "string");
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

    TCLAP::ValueArg<std::string> listen_ip("", "listen-ip", "listen ip", false,
                                           "0.0.0.0", "string");
    TCLAP::ValueArg<int> port("", "port", "port", false, 10095, "int");
    TCLAP::ValueArg<int> io_thread_num("", "io-thread-num", "io thread num",
                                       false, 8, "int");
    TCLAP::ValueArg<int> decoder_thread_num(
        "", "decoder-thread-num", "decoder thread num", false, 8, "int");
    TCLAP::ValueArg<int> model_thread_num("", "model-thread-num",
                                          "model thread num", false, 1, "int");

    TCLAP::ValueArg<std::string> certfile("", "certfile", 
        "default: ../../../ssl_key/server.crt, path of certficate for WSS connection. if it is empty, it will be in WS mode.", 
        false, "../../../ssl_key/server.crt", "string");
    TCLAP::ValueArg<std::string> keyfile("", "keyfile", 
        "default: ../../../ssl_key/server.key, path of keyfile for WSS connection", 
        false, "../../../ssl_key/server.key", "string");

    cmd.add(certfile);
    cmd.add(keyfile);

    cmd.add(download_model_dir);
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

    // Download model form Modelscope
    try{
        std::string s_download_model_dir = download_model_dir.getValue();
        // download model from modelscope when the model-dir is model ID or local path
        bool is_download = false;
        if(download_model_dir.isSet() && !s_download_model_dir.empty()){
            is_download = true;
            if (access(s_download_model_dir.c_str(), F_OK) != 0){
                LOG(ERROR) << s_download_model_dir << " do not exists."; 
                exit(-1);
            }
        }else{
            s_download_model_dir="./";
        }
        std::string s_vad_path = model_path[VAD_DIR];
        std::string s_vad_quant = model_path[VAD_QUANT];
        std::string s_asr_path = model_path[MODEL_DIR];
        std::string s_asr_quant = model_path[QUANTIZE];
        std::string s_punc_path = model_path[PUNC_DIR];
        std::string s_punc_quant = model_path[PUNC_QUANT];
        std::string python_cmd = "python -m funasr.export.export_model --type onnx --quantize True ";
        if(vad_dir.isSet() && !s_vad_path.empty()){
            std::string python_cmd_vad = python_cmd + " --model-name " + s_vad_path + " --export-dir " + s_download_model_dir;
            if(is_download){
                LOG(INFO) << "Download model: " <<  s_vad_path << " from modelscope: ";
            }else{
                LOG(INFO) << "Check local model: " <<  s_vad_path;
                if (access(s_vad_path.c_str(), F_OK) != 0){
                    LOG(ERROR) << s_vad_path << " do not exists."; 
                    exit(-1);
                }                
            }
            system(python_cmd_vad.c_str());
            std::string down_vad_path;
            std::string down_vad_model;            
            if(is_download){
                down_vad_path  = s_download_model_dir+"/"+s_vad_path;
                down_vad_model = s_download_model_dir+"/"+s_vad_path+"/model_quant.onnx";
            }else{
                down_vad_path  = s_vad_path;
                down_vad_model = s_vad_path+"/model_quant.onnx";
                if(s_vad_quant=="false" || s_vad_quant=="False" || s_vad_quant=="FALSE"){
                    down_vad_model = s_vad_path+"/model.onnx";
                }
            }
            if (access(down_vad_model.c_str(), F_OK) != 0){
                LOG(ERROR) << down_vad_model << " do not exists."; 
                exit(-1);
            }else{
                model_path[VAD_DIR]=down_vad_path;
                LOG(INFO) << "Set " << VAD_DIR << " : " << model_path[VAD_DIR];
            }
        }else{
            LOG(INFO) << "VAD model is not set, use default.";
        }

        if(model_dir.isSet() && !s_asr_path.empty()){
            std::string python_cmd_asr = python_cmd + " --model-name " + s_asr_path + " --export-dir " + s_download_model_dir;
            if(is_download){
                LOG(INFO) << "Download model: " <<  s_asr_path << " from modelscope: ";
            }else{
                LOG(INFO) << "Check local model: " <<  s_asr_path;
                if (access(s_asr_path.c_str(), F_OK) != 0){
                    LOG(ERROR) << s_asr_path << " do not exists."; 
                    exit(-1);
                }                
            }
            system(python_cmd_asr.c_str());
            std::string down_asr_path;
            std::string down_asr_model;     
            if(is_download){
                down_asr_path  = s_download_model_dir+"/"+s_asr_path;
                down_asr_model = s_download_model_dir+"/"+s_asr_path+"/model_quant.onnx";
            }else{
                down_asr_path  = s_asr_path;
                down_asr_model = s_asr_path+"/model_quant.onnx";
                if(s_asr_quant=="false" || s_asr_quant=="False" || s_asr_quant=="FALSE"){
                    down_asr_model = s_asr_path+"/model.onnx";
                }
            }
            if (access(down_asr_model.c_str(), F_OK) != 0){
              LOG(ERROR) << down_asr_model << " do not exists."; 
              exit(-1);
            }else{
              model_path[MODEL_DIR]=down_asr_path;
              LOG(INFO) << "Set " << MODEL_DIR << " : " << model_path[MODEL_DIR];
            }
        }else{
          LOG(INFO) << "ASR model is not set, use default.";
        }

        if(punc_dir.isSet() && !s_punc_path.empty()){
            std::string python_cmd_punc = python_cmd + " --model-name " + s_punc_path + " --export-dir " + s_download_model_dir;
            if(is_download){
                LOG(INFO) << "Download model: " <<  s_punc_path << " from modelscope: ";
            }else{
                LOG(INFO) << "Check local model: " <<  s_punc_path;
                if (access(s_punc_path.c_str(), F_OK) != 0){
                    LOG(ERROR) << s_punc_path << " do not exists."; 
                    exit(-1);
                }                
            }
            system(python_cmd_punc.c_str());
            std::string down_punc_path;
            std::string down_punc_model;            
            if(is_download){
                down_punc_path  = s_download_model_dir+"/"+s_punc_path;
                down_punc_model = s_download_model_dir+"/"+s_punc_path+"/model_quant.onnx";
            }else{
                down_punc_path  = s_punc_path;
                down_punc_model = s_punc_path+"/model_quant.onnx";
                if(s_punc_quant=="false" || s_punc_quant=="False" || s_punc_quant=="FALSE"){
                    down_punc_model = s_punc_path+"/model.onnx";
                }
            }
            if (access(down_punc_model.c_str(), F_OK) != 0){
              LOG(ERROR) << down_punc_model << " do not exists."; 
              exit(-1);
            }else{
              model_path[PUNC_DIR]=down_punc_path;
              LOG(INFO) << "Set " << PUNC_DIR << " : " << model_path[PUNC_DIR];
            }
        }else{
          LOG(INFO) << "PUNC model is not set, use default.";
        }
    } catch (std::exception const& e) {
        LOG(ERROR) << "Error: " << e.what();
    }

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
