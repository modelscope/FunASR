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

#ifndef WEBSOCKET_SERVER_H_
#define WEBSOCKET_SERVER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#define ASIO_STANDALONE 1  // not boost
#include <glog/logging.h>
#include "util/text-utils.h"

#include <fstream>
#include <functional>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/config/asio.hpp>
#include <websocketpp/server.hpp>

#include "asio.hpp"
#include "com-define.h"
#include "funasrruntime.h"
#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"
typedef websocketpp::server<websocketpp::config::asio> server;
typedef websocketpp::server<websocketpp::config::asio_tls> wss_server;
typedef server::message_ptr message_ptr;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

typedef websocketpp::lib::lock_guard<websocketpp::lib::mutex> scoped_lock;
typedef websocketpp::lib::unique_lock<websocketpp::lib::mutex> unique_lock;
typedef websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context>
    context_ptr;

typedef struct {
  std::string msg;
  float snippet_time;
} FUNASR_RECOG_RESULT;

typedef struct {
  nlohmann::json msg;
  std::shared_ptr<std::vector<char>> samples;
  std::shared_ptr<std::vector<std::vector<std::string>>> punc_cache;
  std::shared_ptr<std::vector<std::vector<float>>> hotwords_embedding=nullptr;
  std::shared_ptr<websocketpp::lib::mutex> thread_lock; // lock for each connection
  FUNASR_HANDLE tpass_online_handle=nullptr;
  std::string online_res = "";
  std::string tpass_res = "";
  std::shared_ptr<asio::io_context::strand>  strand_; // for data execute in order
  FUNASR_DEC_HANDLE decoder_handle=nullptr; 
} FUNASR_MESSAGE;

// See https://wiki.mozilla.org/Security/Server_Side_TLS for more details about
// the TLS modes. The code below demonstrates how to implement both the modern
enum tls_mode { MOZILLA_INTERMEDIATE = 1, MOZILLA_MODERN = 2 };
class WebSocketServer {
 public:
  WebSocketServer(asio::io_context& io_decoder, bool is_ssl, server* server,
                  wss_server* wss_server, std::string& s_certfile,
                  std::string& s_keyfile)
      : io_decoder_(io_decoder),
        is_ssl(is_ssl),
        server_(server),
        wss_server_(wss_server) {
    if (is_ssl) {
      std::cout << "certfile path is " << s_certfile << std::endl;
      wss_server->set_tls_init_handler(
          bind<context_ptr>(&WebSocketServer::on_tls_init, this,
                            MOZILLA_INTERMEDIATE, ::_1, s_certfile, s_keyfile));
      wss_server_->set_message_handler(
          [this](websocketpp::connection_hdl hdl, message_ptr msg) {
            on_message(hdl, msg);
          });
      // set open handle
      wss_server_->set_open_handler(
          [this](websocketpp::connection_hdl hdl) { on_open(hdl); });
      // set close handle
      wss_server_->set_close_handler(
          [this](websocketpp::connection_hdl hdl) { on_close(hdl); });
      // begin accept
      wss_server_->start_accept();
      // not print log
      wss_server_->clear_access_channels(websocketpp::log::alevel::all);

    } else {
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
  }
  void do_decoder(std::vector<char>& buffer, websocketpp::connection_hdl& hdl,
                  nlohmann::json& msg,
                  std::vector<std::vector<std::string>>& punc_cache,
                  std::vector<std::vector<float>> &hotwords_embedding,
                  websocketpp::lib::mutex& thread_lock, bool& is_final,
                  std::string wav_name,
                  std::string modetype,
                  bool itn,
                  int audio_fs,
                  std::string wav_format,
                  FUNASR_HANDLE& tpass_online_handle,
                  FUNASR_DEC_HANDLE& decoder_handle,
                  std::string svs_lang,
                  bool sys_itn);

  void initAsr(std::map<std::string, std::string>& model_path, int thread_num);
  void on_message(websocketpp::connection_hdl hdl, message_ptr msg);
  void on_open(websocketpp::connection_hdl hdl);
  void on_close(websocketpp::connection_hdl hdl);
  context_ptr on_tls_init(tls_mode mode, websocketpp::connection_hdl hdl,
                          std::string& s_certfile, std::string& s_keyfile);

 private:
  void check_and_clean_connection();
  asio::io_context& io_decoder_;  // threads for asr decoder
  // std::ofstream fout;
  // FUNASR_HANDLE asr_handle;  // asr engine handle
  FUNASR_HANDLE tpass_handle=nullptr;
  bool isonline = true;  // online or offline engine, now only support offline
  bool is_ssl = true;
  server* server_;          // websocket server
  wss_server* wss_server_;  // websocket server

  // use map to keep the received samples data from one connection in offline
  // engine. if for online engline, a data struct is needed(TODO)

  std::map<websocketpp::connection_hdl, std::shared_ptr<FUNASR_MESSAGE>,
           std::owner_less<websocketpp::connection_hdl>>
      data_map;
  websocketpp::lib::mutex m_lock;  // mutex for sample_map
};

#endif  // WEBSOCKET_SERVER_H_
