/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// io server
// Usage:funasr-wss-server  [--model_thread_num <int>] [--decoder_thread_num
// <int>]
//                    [--io_thread_num <int>] [--port <int>] [--listen_ip
//                    <string>] [--punc-quant <string>] [--punc-dir <string>]
//                    [--vad-quant <string>] [--vad-dir <string>] [--quantize
//                    <string>] --model-dir <string> [--] [--version] [-h]
#include <unistd.h>
#include "websocket-server-2pass.h"

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
        "Download model from Modelscope to download_model_dir", false,
        "/workspace/models", "string");
    TCLAP::ValueArg<std::string> offline_model_dir(
        "", OFFLINE_MODEL_DIR,
        "default: /workspace/models/offline_asr, the asr model path, which "
        "contains model_quant.onnx, config.yaml, am.mvn",
        false, "/workspace/models/offline_asr", "string");
    TCLAP::ValueArg<std::string> online_model_dir(
        "", ONLINE_MODEL_DIR,
        "default: damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx, the asr model path, which "
        "contains model_quant.onnx, config.yaml, am.mvn",
        false, "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx", "string");

    TCLAP::ValueArg<std::string> offline_model_revision(
        "", "offline-model-revision", "ASR offline model revision", false,
        "v1.2.1", "string");

    TCLAP::ValueArg<std::string> online_model_revision(
        "", "online-model-revision", "ASR online model revision", false,
        "v1.0.6", "string");

    TCLAP::ValueArg<std::string> quantize(
        "", QUANTIZE,
        "true (Default), load the model of model_quant.onnx in model_dir. If "
        "set "
        "false, load the model of model.onnx in model_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> vad_dir(
        "", VAD_DIR,
        "default: /workspace/models/vad, the vad model path, which contains "
        "model_quant.onnx, vad.yaml, vad.mvn",
        false, "/workspace/models/vad", "string");
    TCLAP::ValueArg<std::string> vad_revision(
        "", "vad-revision", "VAD model revision", false, "v1.2.0", "string");
    TCLAP::ValueArg<std::string> vad_quant(
        "", VAD_QUANT,
        "true (Default), load the model of model_quant.onnx in vad_dir. If set "
        "false, load the model of model.onnx in vad_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> punc_dir(
        "", PUNC_DIR,
        "default: /workspace/models/punc, the punc model path, which contains "
        "model_quant.onnx, punc.yaml",
        false, "/workspace/models/punc", "string");
    TCLAP::ValueArg<std::string> punc_revision(
        "", "punc-revision", "PUNC model revision", false, "1.0.2", "string");
    TCLAP::ValueArg<std::string> punc_quant(
        "", PUNC_QUANT,
        "true (Default), load the model of model_quant.onnx in punc_dir. If "
        "set "
        "false, load the model of model.onnx in punc_dir",
        false, "true", "string");

    TCLAP::ValueArg<std::string> listen_ip("", "listen-ip", "listen ip", false,
                                           "0.0.0.0", "string");
    TCLAP::ValueArg<int> port("", "port", "port", false, 10095, "int");
    TCLAP::ValueArg<int> io_thread_num("", "io-thread-num", "io thread num",
                                       false, 8, "int");
    TCLAP::ValueArg<int> decoder_thread_num(
        "", "decoder-thread-num", "decoder thread num", false, 8, "int");
    TCLAP::ValueArg<int> model_thread_num("", "model-thread-num",
                                          "model thread num", false, 4, "int");

    TCLAP::ValueArg<std::string> certfile(
        "", "certfile",
        "default: ../../../ssl_key/server.crt, path of certficate for WSS "
        "connection. if it is empty, it will be in WS mode.",
        false, "../../../ssl_key/server.crt", "string");
    TCLAP::ValueArg<std::string> keyfile(
        "", "keyfile",
        "default: ../../../ssl_key/server.key, path of keyfile for WSS "
        "connection",
        false, "../../../ssl_key/server.key", "string");

    cmd.add(certfile);
    cmd.add(keyfile);

    cmd.add(download_model_dir);
    cmd.add(offline_model_dir);
    cmd.add(online_model_dir);
    cmd.add(offline_model_revision);
    cmd.add(online_model_revision);
    cmd.add(quantize);
    cmd.add(vad_dir);
    cmd.add(vad_revision);
    cmd.add(vad_quant);
    cmd.add(punc_dir);
    cmd.add(punc_revision);
    cmd.add(punc_quant);

    cmd.add(listen_ip);
    cmd.add(port);
    cmd.add(io_thread_num);
    cmd.add(decoder_thread_num);
    cmd.add(model_thread_num);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(offline_model_dir, OFFLINE_MODEL_DIR, model_path);
    GetValue(online_model_dir, ONLINE_MODEL_DIR, model_path);
    GetValue(quantize, QUANTIZE, model_path);
    GetValue(vad_dir, VAD_DIR, model_path);
    GetValue(vad_quant, VAD_QUANT, model_path);
    GetValue(punc_dir, PUNC_DIR, model_path);
    GetValue(punc_quant, PUNC_QUANT, model_path);

    GetValue(offline_model_revision, "offline-model-revision", model_path);
    GetValue(online_model_revision, "online-model-revision", model_path);
    GetValue(vad_revision, "vad-revision", model_path);
    GetValue(punc_revision, "punc-revision", model_path);

    // Download model form Modelscope
    try {
      std::string s_download_model_dir = download_model_dir.getValue();

      std::string s_vad_path = model_path[VAD_DIR];
      std::string s_vad_quant = model_path[VAD_QUANT];
      std::string s_offline_asr_path = model_path[OFFLINE_MODEL_DIR];
      std::string s_online_asr_path = model_path[ONLINE_MODEL_DIR];
      std::string s_asr_quant = model_path[QUANTIZE];
      std::string s_punc_path = model_path[PUNC_DIR];
      std::string s_punc_quant = model_path[PUNC_QUANT];

      std::string python_cmd =
          "python -m funasr.utils.runtime_sdk_download_tool --type onnx --quantize True ";

      if (vad_dir.isSet() && !s_vad_path.empty()) {
        std::string python_cmd_vad;
        std::string down_vad_path;
        std::string down_vad_model;

        if (access(s_vad_path.c_str(), F_OK) == 0) {
          // local
          python_cmd_vad = python_cmd + " --model-name " + s_vad_path +
                           " --export-dir ./ " + " --model_revision " +
                           model_path["vad-revision"];
          down_vad_path = s_vad_path;
        } else {
          // modelscope
          LOG(INFO) << "Download model: " << s_vad_path
                    << " from modelscope: "; 
		  python_cmd_vad = python_cmd + " --model-name " +
                s_vad_path +
                " --export-dir " + s_download_model_dir +
                " --model_revision " + model_path["vad-revision"]; 
		  down_vad_path  =
                s_download_model_dir +
                "/" + s_vad_path;
        }

        int ret = system(python_cmd_vad.c_str());
        if (ret != 0) {
          LOG(INFO) << "Failed to download model from modelscope. If you set local vad model path, you can ignore the errors.";
        }
        down_vad_model = down_vad_path + "/model_quant.onnx";
        if (s_vad_quant == "false" || s_vad_quant == "False" ||
            s_vad_quant == "FALSE") {
          down_vad_model = down_vad_path + "/model.onnx";
        }

        if (access(down_vad_model.c_str(), F_OK) != 0) {
          LOG(ERROR) << down_vad_model << " do not exists.";
          exit(-1);
        } else {
          model_path[VAD_DIR] = down_vad_path;
          LOG(INFO) << "Set " << VAD_DIR << " : " << model_path[VAD_DIR];
        }
      }
      else {
        LOG(INFO) << "VAD model is not set, use default.";
      }

      if (offline_model_dir.isSet() && !s_offline_asr_path.empty()) {
        std::string python_cmd_asr;
        std::string down_asr_path;
        std::string down_asr_model;

        if (access(s_offline_asr_path.c_str(), F_OK) == 0) {
          // local
          python_cmd_asr = python_cmd + " --model-name " + s_offline_asr_path +
                           " --export-dir ./ " + " --model_revision " +
                           model_path["offline-model-revision"];
          down_asr_path = s_offline_asr_path;
        } else {
          // modelscope
          LOG(INFO) << "Download model: " << s_offline_asr_path
                    << " from modelscope : "; 
          python_cmd_asr = python_cmd + " --model-name " +
                  s_offline_asr_path +
                  " --export-dir " + s_download_model_dir +
                  " --model_revision " + model_path["offline-model-revision"]; 
          down_asr_path
                = s_download_model_dir + "/" + s_offline_asr_path;
        }

        int ret = system(python_cmd_asr.c_str());
        if (ret != 0) {
          LOG(INFO) << "Failed to download model from modelscope. If you set local asr model path, you can ignore the errors.";
        }
        down_asr_model = down_asr_path + "/model_quant.onnx";
        if (s_asr_quant == "false" || s_asr_quant == "False" ||
            s_asr_quant == "FALSE") {
          down_asr_model = down_asr_path + "/model.onnx";
        }

        if (access(down_asr_model.c_str(), F_OK) != 0) {
          LOG(ERROR) << down_asr_model << " do not exists.";
          exit(-1);
        } else {
          model_path[MODEL_DIR] = down_asr_path;
          LOG(INFO) << "Set " << MODEL_DIR << " : " << model_path[MODEL_DIR];
        }
      } else {
        LOG(INFO) << "ASR Offline model is not set, use default.";
      }

      if (!s_online_asr_path.empty()) {
        std::string python_cmd_asr;
        std::string down_asr_path;
        std::string down_asr_model;

        if (access(s_online_asr_path.c_str(), F_OK) == 0) {
          // local
          python_cmd_asr = python_cmd + " --model-name " + s_online_asr_path +
                           " --export-dir ./ " + " --model_revision " +
                           model_path["online-model-revision"];
          down_asr_path = s_online_asr_path;
        } else {
          // modelscope
          LOG(INFO) << "Download model: " << s_online_asr_path
                    << " from modelscope : "; 
          python_cmd_asr = python_cmd + " --model-name " +
                    s_online_asr_path +
                    " --export-dir " + s_download_model_dir +
                    " --model_revision " + model_path["online-model-revision"]; 
          down_asr_path
                  = s_download_model_dir + "/" + s_online_asr_path;
        }

        int ret = system(python_cmd_asr.c_str());
        if (ret != 0) {
          LOG(INFO) << "Failed to download model from modelscope. If you set local asr model path,  you can ignore the errors.";
        }
        down_asr_model = down_asr_path + "/model_quant.onnx";
        if (s_asr_quant == "false" || s_asr_quant == "False" ||
            s_asr_quant == "FALSE") {
          down_asr_model = down_asr_path + "/model.onnx";
        }

        if (access(down_asr_model.c_str(), F_OK) != 0) {
          LOG(ERROR) << down_asr_model << " do not exists.";
          exit(-1);
        } else {
          model_path[MODEL_DIR] = down_asr_path;
          LOG(INFO) << "Set " << MODEL_DIR << " : " << model_path[MODEL_DIR];
        }
      } else {
        LOG(INFO) << "ASR online model is not set, use default.";
      }

      if (punc_dir.isSet() && !s_punc_path.empty()) {
        std::string python_cmd_punc;
        std::string down_punc_path;
        std::string down_punc_model;

        if (access(s_punc_path.c_str(), F_OK) == 0) {
          // local
          python_cmd_punc = python_cmd + " --model-name " + s_punc_path +
                            " --export-dir ./ " + " --model_revision " +
                            model_path["punc-revision"];
          down_punc_path = s_punc_path;
        } else {
          // modelscope
          LOG(INFO) << "Download model: " << s_punc_path
                    << " from modelscope : "; python_cmd_punc = python_cmd + " --model-name " +
                s_punc_path +
                " --export-dir " + s_download_model_dir +
                " --model_revision " + model_path["punc-revision "]; 
          down_punc_path  =
                s_download_model_dir +
                "/" + s_punc_path;
        }

        int ret = system(python_cmd_punc.c_str());
        if (ret != 0) {
          LOG(INFO) << "Failed to download model from modelscope. If you set local punc model path, you can ignore the errors.";
        }
        down_punc_model = down_punc_path + "/model_quant.onnx";
        if (s_punc_quant == "false" || s_punc_quant == "False" ||
            s_punc_quant == "FALSE") {
          down_punc_model = down_punc_path + "/model.onnx";
        }

        if (access(down_punc_model.c_str(), F_OK) != 0) {
          LOG(ERROR) << down_punc_model << " do not exists.";
          exit(-1);
        } else {
          model_path[PUNC_DIR] = down_punc_path;
          LOG(INFO) << "Set " << PUNC_DIR << " : " << model_path[PUNC_DIR];
        }
      } else {
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
