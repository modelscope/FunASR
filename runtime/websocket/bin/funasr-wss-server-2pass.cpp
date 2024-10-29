/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

#include "websocket-server-2pass.h"
#ifdef _WIN32
#include "win_func.h"
#else
#include <unistd.h>
#endif
#include <fstream>
#include "util.h"

// hotwords
std::unordered_map<std::string, int> hws_map_;
int fst_inc_wts_=20;
float global_beam_, lattice_beam_, am_scale_;

using namespace std;
void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key,
              std::map<std::string, std::string>& model_path) {
  model_path.insert({key, value_arg.getValue()});
  LOG(INFO) << key << " : " << value_arg.getValue();
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  #include <windows.h>
  SetConsoleOutputCP(65001);
#endif
  try {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    std::string tpass_version = "";
#ifdef _WIN32
    tpass_version = "0.1.0";
#endif
    TCLAP::CmdLine cmd("funasr-wss-server", ' ', tpass_version);
    TCLAP::ValueArg<std::string> download_model_dir(
        "", "download-model-dir",
        "Download model from Modelscope to download_model_dir", false,
        "/workspace/models", "string");
    TCLAP::ValueArg<std::string> offline_model_dir(
        "", OFFLINE_MODEL_DIR,
        "default: damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx, the asr model path, which "
        "contains model_quant.onnx, config.yaml, am.mvn",
        false, "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx", "string");
    TCLAP::ValueArg<std::string> online_model_dir(
        "", ONLINE_MODEL_DIR,
        "default: damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx, the asr model path, which "
        "contains model_quant.onnx, config.yaml, am.mvn",
        false, "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx", "string");

    TCLAP::ValueArg<std::string> offline_model_revision(
        "", "offline-model-revision", "ASR offline model revision", false,
        "v2.0.5", "string");

    TCLAP::ValueArg<std::string> online_model_revision(
        "", "online-model-revision", "ASR online model revision", false,
        "v2.0.5", "string");

    TCLAP::ValueArg<std::string> quantize(
        "", QUANTIZE,
        "true (Default), load the model of model_quant.onnx in model_dir. If "
        "set "
        "false, load the model of model.onnx in model_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> vad_dir(
        "", VAD_DIR,
        "default: damo/speech_fsmn_vad_zh-cn-16k-common-onnx, the vad model path, which contains "
        "model_quant.onnx, vad.yaml, vad.mvn",
        false, "damo/speech_fsmn_vad_zh-cn-16k-common-onnx", "string");
    TCLAP::ValueArg<std::string> vad_revision(
        "", "vad-revision", "VAD model revision", false, "v2.0.4", "string");
    TCLAP::ValueArg<std::string> vad_quant(
        "", VAD_QUANT,
        "true (Default), load the model of model_quant.onnx in vad_dir. If set "
        "false, load the model of model.onnx in vad_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> punc_dir(
        "", PUNC_DIR,
        "default: damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx, the punc model path, which contains "
        "model_quant.onnx, punc.yaml",
        false, "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx", "string");
    TCLAP::ValueArg<std::string> punc_revision(
        "", "punc-revision", "PUNC model revision", false, "v2.0.5", "string");
    TCLAP::ValueArg<std::string> punc_quant(
        "", PUNC_QUANT,
        "true (Default), load the model of model_quant.onnx in punc_dir. If "
        "set "
        "false, load the model of model.onnx in punc_dir",
        false, "true", "string");
    TCLAP::ValueArg<std::string> itn_dir(
        "", ITN_DIR,
        "default: thuduj12/fst_itn_zh, the itn model path, which contains "
        "zh_itn_tagger.fst, zh_itn_verbalizer.fst",
        false, "thuduj12/fst_itn_zh", "string");
    TCLAP::ValueArg<std::string> itn_revision(
        "", "itn-revision", "ITN model revision", false, "v1.0.1", "string");

    TCLAP::ValueArg<std::string> listen_ip("", "listen-ip", "listen ip", false,
                                           "0.0.0.0", "string");
    TCLAP::ValueArg<int> port("", "port", "port", false, 10095, "int");
    TCLAP::ValueArg<int> io_thread_num("", "io-thread-num", "io thread num",
                                       false, 2, "int");
    TCLAP::ValueArg<int> decoder_thread_num(
        "", "decoder-thread-num", "decoder thread num", false, 8, "int");
    TCLAP::ValueArg<int> model_thread_num("", "model-thread-num",
                                          "model thread num", false, 2, "int");

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

    TCLAP::ValueArg<float>    global_beam("", GLOB_BEAM, "the decoding beam for beam searching ", false, 3.0, "float");
    TCLAP::ValueArg<float>    lattice_beam("", LAT_BEAM, "the lattice generation beam for beam searching ", false, 3.0, "float");
    TCLAP::ValueArg<float>    am_scale("", AM_SCALE, "the acoustic scale for beam searching ", false, 10.0, "float");

    TCLAP::ValueArg<std::string> lm_dir("", LM_DIR,
        "the LM model path, which contains compiled models: TLG.fst, config.yaml ", false, "damo/speech_ngram_lm_zh-cn-ai-wesp-fst", "string");
    TCLAP::ValueArg<std::string> lm_revision(
        "", "lm-revision", "LM model revision", false, "v1.0.2", "string");
    TCLAP::ValueArg<std::string> hotword("", HOTWORD,
        "the hotword file, one hotword perline, Format: Hotword Weight (could be: 阿里巴巴 20)", 
        false, "/workspace/resources/hotwords.txt", "string");
    TCLAP::ValueArg<std::int32_t> fst_inc_wts("", FST_INC_WTS, 
        "the fst hotwords incremental bias", false, 20, "int32_t");

    // add file
    cmd.add(hotword);
    cmd.add(fst_inc_wts);
    cmd.add(global_beam);
    cmd.add(lattice_beam);
    cmd.add(am_scale);

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
    cmd.add(itn_dir);
    cmd.add(itn_revision);
    cmd.add(lm_dir);
    cmd.add(lm_revision);

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
    GetValue(itn_dir, ITN_DIR, model_path);
    GetValue(lm_dir, LM_DIR, model_path);
    GetValue(hotword, HOTWORD, model_path);

    GetValue(offline_model_revision, "offline-model-revision", model_path);
    GetValue(online_model_revision, "online-model-revision", model_path);
    GetValue(vad_revision, "vad-revision", model_path);
    GetValue(punc_revision, "punc-revision", model_path);
    GetValue(itn_revision, "itn-revision", model_path);
    GetValue(lm_revision, "lm-revision", model_path);

    global_beam_ = global_beam.getValue();
    lattice_beam_ = lattice_beam.getValue();
    am_scale_ = am_scale.getValue();

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
      std::string s_itn_path = model_path[ITN_DIR];
      std::string s_lm_path = model_path[LM_DIR];

      std::string python_cmd =
          "python -m funasr.download.runtime_sdk_download_tool --type onnx --quantize True ";

      if (!s_vad_path.empty()) {
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

      if (!s_offline_asr_path.empty()) {
        std::string python_cmd_asr;
        std::string down_asr_path;
        std::string down_asr_model;

        size_t found = s_offline_asr_path.find("speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404");
        if (found != std::string::npos) {
            model_path["offline-model-revision"]="v2.0.5";
        }

        found = s_offline_asr_path.find("speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404");
        if (found != std::string::npos) {
            model_path["offline-model-revision"]="v2.0.5";
        }

        found = s_offline_asr_path.find("speech_paraformer-large_asr_nat-en-16k-common-vocab10020");
        if (found != std::string::npos) {
            model_path["model-revision"]="v2.0.5";
            s_itn_path="";
            s_lm_path="";
        }
        found = s_offline_asr_path.find(MODEL_SVS);
        if (found != std::string::npos) {
            model_path["model-revision"]="v2.0.5";
            s_lm_path="";
            model_path[LM_DIR]="";
        }        

        if (access(s_offline_asr_path.c_str(), F_OK) == 0) {
          // local
          python_cmd_asr = python_cmd + " --model-name " + s_offline_asr_path +
                           " --export-dir ./ " + " --model_revision " +
                           model_path["offline-model-revision"];
          down_asr_path = s_offline_asr_path;
        } 
        else {
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
          model_path[OFFLINE_MODEL_DIR] = down_asr_path;
          LOG(INFO) << "Set " << OFFLINE_MODEL_DIR << " : " << model_path[OFFLINE_MODEL_DIR];
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
          model_path[ONLINE_MODEL_DIR] = down_asr_path;
          LOG(INFO) << "Set " << ONLINE_MODEL_DIR << " : " << model_path[ONLINE_MODEL_DIR];
        }
      } else {
        LOG(INFO) << "ASR online model is not set, use default.";
      }

      if (!s_lm_path.empty() && s_lm_path != "NONE" && s_lm_path != "none") {
          std::string python_cmd_lm;
          std::string down_lm_path;
          std::string down_lm_model;

          if (access(s_lm_path.c_str(), F_OK) == 0) {
              // local
              python_cmd_lm = python_cmd + " --model-name " + s_lm_path +
                                  " --export-dir ./ " + " --model_revision " +
                                  model_path["lm-revision"] + " --export False ";
              down_lm_path = s_lm_path;
          } else {
              // modelscope
              LOG(INFO) << "Download model: " << s_lm_path
                          << " from modelscope : "; 
              python_cmd_lm = python_cmd + " --model-name " +
                      s_lm_path +
                      " --export-dir " + s_download_model_dir +
                      " --model_revision " + model_path["lm-revision"]
                      + " --export False "; 
              down_lm_path  =
                      s_download_model_dir +
                      "/" + s_lm_path;
          }

          int ret = system(python_cmd_lm.c_str());
          if (ret != 0) {
              LOG(INFO) << "Failed to download model from modelscope. If you set local lm model path, you can ignore the errors.";
          }
          down_lm_model = down_lm_path + "/TLG.fst";

          if (access(down_lm_model.c_str(), F_OK) != 0) {
              LOG(ERROR) << down_lm_model << " do not exists.";
              exit(-1);
          } else {
              model_path[LM_DIR] = down_lm_path;
              LOG(INFO) << "Set " << LM_DIR << " : " << model_path[LM_DIR];
          }
      } else {
          LOG(INFO) << "LM model is not set, not executed.";
          model_path[LM_DIR] = "";
      }

      if (!s_punc_path.empty()) {
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
                    << " from modelscope : "; 
              python_cmd_punc = python_cmd + " --model-name " +
                s_punc_path +
                " --export-dir " + s_download_model_dir +
                " --model_revision " + model_path["punc-revision"]; 
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

      if (!s_itn_path.empty()) {
        std::string python_cmd_itn;
        std::string down_itn_path;
        std::string down_itn_model;

        if (access(s_itn_path.c_str(), F_OK) == 0) {
          // local
          python_cmd_itn = python_cmd + " --model-name " + s_itn_path +
                            " --export-dir ./ " + " --model_revision " +
                            model_path["itn-revision"] + " --export False ";
          down_itn_path = s_itn_path;
        } else {
          // modelscope
          LOG(INFO) << "Download model: " << s_itn_path
                    << " from modelscope : "; 
          python_cmd_itn = python_cmd + " --model-name " +
                s_itn_path +
                " --export-dir " + s_download_model_dir +
                " --model_revision " + model_path["itn-revision"]
                + " --export False "; 
          down_itn_path  =
                s_download_model_dir +
                "/" + s_itn_path;
        }

        int ret = system(python_cmd_itn.c_str());
        if (ret != 0) {
          LOG(INFO) << "Failed to download model from modelscope. If you set local itn model path, you can ignore the errors.";
        }
        down_itn_model = down_itn_path + "/zh_itn_tagger.fst";

        if (access(down_itn_model.c_str(), F_OK) != 0) {
          LOG(ERROR) << down_itn_model << " do not exists.";
          exit(-1);
        } else {
          model_path[ITN_DIR] = down_itn_path;
          LOG(INFO) << "Set " << ITN_DIR << " : " << model_path[ITN_DIR];
        }
      } else {
        LOG(INFO) << "ITN model is not set, use default.";
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

    // hotword file
    std::string hotword_path;
    hotword_path = model_path.at(HOTWORD);
    fst_inc_wts_ = fst_inc_wts.getValue();
    LOG(INFO) << "hotword path: " << hotword_path;
    funasr::ExtractHws(hotword_path, hws_map_);

    bool is_ssl = false;
    if (!s_certfile.empty() && access(s_certfile.c_str(), F_OK) == 0) {
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
    server* server = nullptr;
    wss_server* wss_server = nullptr;
    if (is_ssl) {
      LOG(INFO)<< "SSL is opened!";
      wss_server_.init_asio(&io_server);  // init asio
      wss_server_.set_reuse_addr(
          true);  // reuse address as we create multiple threads

      // list on port for accept
      wss_server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);
      wss_server = &wss_server_;

    } else {
      LOG(INFO)<< "SSL is closed!";
      server_.init_asio(&io_server);  // init asio
      server_.set_reuse_addr(
          true);  // reuse address as we create multiple threads

      // list on port for accept
      server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);
      server = &server_;

    }

    WebSocketServer websocket_srv(
        io_decoder, is_ssl, server, wss_server, s_certfile,
        s_keyfile);  // websocket server for asr engine
    websocket_srv.initAsr(model_path, s_model_thread_num);  // init asr model

    LOG(INFO) << "decoder-thread-num: " << s_decoder_thread_num;
    LOG(INFO) << "io-thread-num: " << s_io_thread_num;
    LOG(INFO) << "model-thread-num: " << s_model_thread_num;
    LOG(INFO) << "asr model init finished. listen on port:" << s_port;

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
    LOG(ERROR) << "Error: " << e.what();
  }

  return 0;
}
