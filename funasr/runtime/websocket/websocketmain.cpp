/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// io server
// Usage:websocketmain  [--model_thread_num <int>] [--decoder_thread_num
//                        <int>] [--io_thread_num <int>] [--port <int>]
//                        [--listen_ip <string>] [--wav-scp <string>]
//                        [--wav-path <string>] [--punc-config <string>]
//                        [--punc-model <string>] --am-config <string>
//                        --am-cmvn <string> --am-model <string>
//                        [--vad-config <string>] [--vad-cmvn <string>]
//                        [--vad-model <string>] [--] [--version] [-h]
#include "websocketsrv.h"

using namespace std;
void GetValue(TCLAP::ValueArg<std::string>& value_arg, string key,
              std::map<std::string, std::string>& model_path) {
  if (value_arg.isSet()) {
    model_path.insert({key, value_arg.getValue()});
    LOG(INFO) << key << " : " << value_arg.getValue();
  }
}
int main(int argc, char* argv[]) {
  try {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("websocketmain", ' ', "1.0");
    TCLAP::ValueArg<std::string> vad_model("", VAD_MODEL_PATH, "vad model path",
                                           false, "", "string");
    TCLAP::ValueArg<std::string> vad_cmvn("", VAD_CMVN_PATH, "vad cmvn path",
                                          false, "", "string");
    TCLAP::ValueArg<std::string> vad_config(
        "", VAD_CONFIG_PATH, "vad config path", false, "", "string");

    TCLAP::ValueArg<std::string> am_model("", AM_MODEL_PATH, "am model path",
                                          true, "", "string");
    TCLAP::ValueArg<std::string> am_cmvn("", AM_CMVN_PATH, "am cmvn path", true,
                                         "", "string");
    TCLAP::ValueArg<std::string> am_config("", AM_CONFIG_PATH, "am config path",
                                           true, "", "string");

    TCLAP::ValueArg<std::string> punc_model(
        "", PUNC_MODEL_PATH, "punc model path", false, "", "string");
    TCLAP::ValueArg<std::string> punc_config(
        "", PUNC_CONFIG_PATH, "punc config path", false, "", "string");

    TCLAP::ValueArg<std::string> wav_path("", WAV_PATH, "wave file path", false,
                                          "", "string");
    TCLAP::ValueArg<std::string> wav_scp("", WAV_SCP, "wave scp path", false,
                                         "", "string");

    TCLAP::ValueArg<std::string> listen_ip("", "listen_ip", "listen_ip", false,
                                           "0.0.0.0", "string");
    TCLAP::ValueArg<int> port("", "port", "port", false, 8889, "int");
    TCLAP::ValueArg<int> io_thread_num("", "io_thread_num", "io_thread_num",
                                       false, 8, "int");
    TCLAP::ValueArg<int> decoder_thread_num(
        "", "decoder_thread_num", "decoder_thread_num", false, 8, "int");
    TCLAP::ValueArg<int> model_thread_num("", "model_thread_num",
                                          "model_thread_num", false, 1, "int");

    cmd.add(vad_model);
    cmd.add(vad_cmvn);
    cmd.add(vad_config);
    cmd.add(am_model);
    cmd.add(am_cmvn);
    cmd.add(am_config);
    cmd.add(punc_model);
    cmd.add(punc_config);
    cmd.add(wav_path);
    cmd.add(wav_scp);
    cmd.add(listen_ip);
    cmd.add(port);
    cmd.add(io_thread_num);
    cmd.add(decoder_thread_num);
    cmd.add(model_thread_num);
    cmd.parse(argc, argv);

    std::map<std::string, std::string> model_path;
    GetValue(vad_model, VAD_MODEL_PATH, model_path);
    GetValue(vad_cmvn, VAD_CMVN_PATH, model_path);
    GetValue(vad_config, VAD_CONFIG_PATH, model_path);
    GetValue(am_model, AM_MODEL_PATH, model_path);
    GetValue(am_cmvn, AM_CMVN_PATH, model_path);
    GetValue(am_config, AM_CONFIG_PATH, model_path);
    GetValue(punc_model, PUNC_MODEL_PATH, model_path);
    GetValue(punc_config, PUNC_CONFIG_PATH, model_path);
    GetValue(wav_path, WAV_PATH, model_path);
    GetValue(wav_scp, WAV_SCP, model_path);


    std::string s_listen_ip = listen_ip.getValue();
    int s_port = port.getValue();
    int s_io_thread_num = io_thread_num.getValue();
    int s_decoder_thread_num = decoder_thread_num.getValue();

    int s_model_thread_num = model_thread_num.getValue();

 
    asio::io_context io_decoder;  // context for decoding

    std::vector<std::thread> decoder_threads;

    auto conn_guard = asio::make_work_guard(
        io_decoder);  // make sure threads can wait in the queue

    // create threads pool
    for (int32_t i = 0; i < s_decoder_thread_num; ++i) {
      decoder_threads.emplace_back([&io_decoder]() { io_decoder.run(); });
    }

    server server_;       // server for websocket
    server_.init_asio();  // init asio
    server_.set_reuse_addr(
        true);  // reuse address as we create multiple threads

    // list on port for accept
    server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);

    WebSocketServer websocket_srv(io_decoder,
                                  &server_);  // websocket server for asr engine
    websocket_srv.initAsr(model_path, s_model_thread_num);  // init asr model
    std::cout << "asr model init finished. listen on port:" << s_port
              << std::endl;

    // Start the ASIO network io_service run loop
    if (s_io_thread_num == 1) {
      server_.run();
    } else {
      typedef websocketpp::lib::shared_ptr<websocketpp::lib::thread> thread_ptr;
      std::vector<thread_ptr> ts;
      // create threads for io network
      for (size_t i = 0; i < s_io_thread_num; i++) {
        ts.push_back(websocketpp::lib::make_shared<websocketpp::lib::thread>(
            &server::run, &server_));
      }
      // wait for theads
      for (size_t i = 0; i < s_io_thread_num; i++) {
        ts[i]->join();
      }
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