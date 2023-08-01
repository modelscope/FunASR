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

#include "websocket-server.h"

#include <thread>
#include <utility>
#include <vector>

context_ptr WebSocketServer::on_tls_init(tls_mode mode,
                                         websocketpp::connection_hdl hdl,
                                         std::string& s_certfile,
                                         std::string& s_keyfile) {
  namespace asio = websocketpp::lib::asio;

  LOG(INFO) << "on_tls_init called with hdl: " << hdl.lock().get();
  LOG(INFO) << "using TLS mode: "
            << (mode == MOZILLA_MODERN ? "Mozilla Modern"
                                       : "Mozilla Intermediate");
                                       
  context_ptr ctx = websocketpp::lib::make_shared<asio::ssl::context>(
      asio::ssl::context::sslv23);

  try {
    if (mode == MOZILLA_MODERN) {
      // Modern disables TLSv1
      ctx->set_options(
          asio::ssl::context::default_workarounds |
          asio::ssl::context::no_sslv2 | asio::ssl::context::no_sslv3 |
          asio::ssl::context::no_tlsv1 | asio::ssl::context::single_dh_use);
    } else {
      ctx->set_options(asio::ssl::context::default_workarounds |
                       asio::ssl::context::no_sslv2 |
                       asio::ssl::context::no_sslv3 |
                       asio::ssl::context::single_dh_use);
    }

    ctx->use_certificate_chain_file(s_certfile);
    ctx->use_private_key_file(s_keyfile, asio::ssl::context::pem);

  } catch (std::exception& e) {
    LOG(INFO) << "Exception: " << e.what();
  }
  return ctx;
}

// feed buffer to asr engine for decoder
void WebSocketServer::do_decoder(const std::vector<char>& buffer,
                                 websocketpp::connection_hdl& hdl,
                                 const nlohmann::json& msg) {
  try {
    int num_samples = buffer.size();  // the size of the buf

    if (!buffer.empty()) {
      // feed data to asr engine
      FUNASR_RESULT Result = FunOfflineInferBuffer(
          asr_hanlde, buffer.data(), buffer.size(), RASR_NONE, NULL, 16000, msg["wav_format"]);

      std::string asr_result =
          ((FUNASR_RECOG_RESULT*)Result)->msg;  // get decode result
      FunASRFreeResult(Result);

      websocketpp::lib::error_code ec;
      nlohmann::json jsonresult;        // result json
      jsonresult["text"] = asr_result;  // put result in 'text'
      jsonresult["mode"] = "offline";

      jsonresult["wav_name"] = msg["wav_name"];

      // send the json to client
      if (is_ssl) {
        wss_server_->send(hdl, jsonresult.dump(),
                          websocketpp::frame::opcode::text, ec);
      } else {
        server_->send(hdl, jsonresult.dump(), websocketpp::frame::opcode::text,
                      ec);
      }

      LOG(INFO) << "buffer.size=" << buffer.size() << ",result json=" << jsonresult.dump();
      if (!isonline) {
        //  close the client if it is not online asr
        // server_->close(hdl, websocketpp::close::status::normal, "DONE", ec);
        // fout.close();
      }
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

void WebSocketServer::on_open(websocketpp::connection_hdl hdl) {
  scoped_lock guard(m_lock);     // for threads safty
  check_and_clean_connection();  // remove closed connection

  std::shared_ptr<FUNASR_MESSAGE> data_msg =
      std::make_shared<FUNASR_MESSAGE>();  // put a new data vector for new
                                           // connection
  data_msg->samples = std::make_shared<std::vector<char>>();
  data_msg->msg = nlohmann::json::parse("{}");
  data_msg->msg["wav_format"] = "pcm";
  data_map.emplace(hdl, data_msg);
  LOG(INFO) << "on_open, active connections: " << data_map.size();
}

void WebSocketServer::on_close(websocketpp::connection_hdl hdl) {
  scoped_lock guard(m_lock);
  data_map.erase(hdl);  // remove data vector when  connection is closed

  LOG(INFO) << "on_close, active connections: " << data_map.size();
}

// remove closed connection
void WebSocketServer::check_and_clean_connection() {
  std::vector<websocketpp::connection_hdl> to_remove;  // remove list
  auto iter = data_map.begin();
  while (iter != data_map.end()) {  // loop to find closed connection
    websocketpp::connection_hdl hdl = iter->first;

    if (is_ssl) {
      wss_server::connection_ptr con = wss_server_->get_con_from_hdl(hdl);
      if (con->get_state() != 1) {  // session::state::open ==1
        to_remove.push_back(hdl);
      }
    } else {
      server::connection_ptr con = server_->get_con_from_hdl(hdl);
      if (con->get_state() != 1) {  // session::state::open ==1
        to_remove.push_back(hdl);
      }
    }

    iter++;
  }
  for (auto hdl : to_remove) {
    data_map.erase(hdl);
    LOG(INFO)<< "remove one connection ";
  }
}
void WebSocketServer::on_message(websocketpp::connection_hdl hdl,
                                 message_ptr msg) {
  unique_lock lock(m_lock);
  // find the sample data vector according to one connection

  std::shared_ptr<FUNASR_MESSAGE> msg_data = nullptr;

  auto it_data = data_map.find(hdl);
  if (it_data != data_map.end()) {
    msg_data = it_data->second;
  }
  std::shared_ptr<std::vector<char>> sample_data_p = msg_data->samples;

  lock.unlock();
  if (sample_data_p == nullptr) {
    LOG(INFO) << "error when fetch sample data vector";
    return;
  }

  const std::string& payload = msg->get_payload();  // get msg type

  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text: {
      nlohmann::json jsonresult = nlohmann::json::parse(payload);
      if (jsonresult["wav_name"] != nullptr) {
        msg_data->msg["wav_name"] = jsonresult["wav_name"];
      }
      if (jsonresult["wav_format"] != nullptr) {
        msg_data->msg["wav_format"] = jsonresult["wav_format"];
      }

      if (jsonresult["is_speaking"] == false ||
          jsonresult["is_finished"] == true) {
        LOG(INFO) << "client done";

        if (isonline) {
          // do_close(ws);
        } else {
          // add padding to the end of the wav data
          // std::vector<short> padding(static_cast<short>(0.3 * 16000));
          // sample_data_p->insert(sample_data_p->end(), padding.data(),
          //                       padding.data() + padding.size());
          // for offline, send all receive data to decoder engine
          asio::post(io_decoder_,
                     std::bind(&WebSocketServer::do_decoder, this,
                               std::move(*(sample_data_p.get())),
                               std::move(hdl), std::move(msg_data->msg)));
        }
      }
      break;
    }
    case websocketpp::frame::opcode::binary: {
      // recived binary data
      const auto* pcm_data = static_cast<const char*>(payload.data());
      int32_t num_samples = payload.size();

      if (isonline) {
        // if online TODO(zhaoming) still not done
        std::vector<char> s(pcm_data, pcm_data + num_samples);
        asio::post(io_decoder_,
                   std::bind(&WebSocketServer::do_decoder, this, std::move(s),
                             std::move(hdl), std::move(msg_data->msg)));
      } else {
        // for offline, we add receive data to end of the sample data vector
        sample_data_p->insert(sample_data_p->end(), pcm_data,
                              pcm_data + num_samples);
      }

      break;
    }
    default:
      break;
  }
}

// init asr model
void WebSocketServer::initAsr(std::map<std::string, std::string>& model_path,
                              int thread_num) {
  try {
    // init model with api

    asr_hanlde = FunOfflineInit(model_path, thread_num);
    LOG(INFO) << "model successfully inited";

  } catch (const std::exception& e) {
    LOG(INFO) << e.what();
  }
}
