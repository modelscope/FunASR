/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// websocket server for asr engine
// take some ideas from https://github.com/k2-fsa/sherpa-onnx
// online-websocket-server-impl.cc, thanks. The websocket server has two threads
// pools, one for handle network data and one for asr decoder.
// now only support offline engine.

#ifndef WEBSOCKETSRV_SERVER_H_
#define WEBSOCKETSRV_SERVER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#define ASIO_STANDALONE 1  // not boost
#include <glog/logging.h>

#include <fstream>
#include <functional>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include "asio.hpp"
#include "com-define.h"
#include "libfunasrapi.h"
#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"
typedef websocketpp::server<websocketpp::config::asio> server;
typedef server::message_ptr message_ptr;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
typedef websocketpp::lib::lock_guard<websocketpp::lib::mutex> scoped_lock;
typedef websocketpp::lib::unique_lock<websocketpp::lib::mutex> unique_lock;

typedef struct {
  std::string msg;
  float snippet_time;
} FUNASR_RECOG_RESULT;

class WebSocketServer {
 public:
  WebSocketServer(asio::io_context& io_decoder, server* server_)
      : io_decoder_(io_decoder), server_(server_) {
    // set message handle
    server_->set_message_handler(
        [this](websocketpp::connection_hdl hdl, message_ptr msg) {
          on_message(hdl, msg);
        });
    // set open handle
    server_->set_open_handler(
        [this](websocketpp::connection_hdl hdl) { on_open(hdl); });
    // set close handle
    server_->set_close_handler(
        [this](websocketpp::connection_hdl hdl) { on_close(hdl); });
    // begin accept
    server_->start_accept();
    // not print log
    server_->clear_access_channels(websocketpp::log::alevel::all);
  }
  void do_decoder(const std::vector<char>& buffer,
                  websocketpp::connection_hdl& hdl);

  void initAsr(std::map<std::string, std::string>& model_path, int thread_num);
  void on_message(websocketpp::connection_hdl hdl, message_ptr msg);
  void on_open(websocketpp::connection_hdl hdl);
  void on_close(websocketpp::connection_hdl hdl);

 private:
  void check_and_clean_connection();
  asio::io_context& io_decoder_;  // threads for asr decoder
  // std::ofstream fout;
  FUNASR_HANDLE asr_hanlde;  // asr engine handle
  bool isonline = false;  // online or offline engine, now only support offline
  server* server_;        // websocket server

  // use map to keep the received samples data from one connection in offline
  // engine. if for online engline, a data struct is needed(TODO)
  std::map<websocketpp::connection_hdl, std::shared_ptr<std::vector<char>>,
           std::owner_less<websocketpp::connection_hdl>>
      sample_map;
  websocketpp::lib::mutex m_lock;  // mutex for sample_map
};

#endif  // WEBSOCKETSRV_SERVER_H_
