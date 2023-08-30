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

#include "websocket-server-2pass.h"

#include <thread>
#include <utility>
#include <vector>
#include <chrono>
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

nlohmann::json handle_result(FUNASR_RESULT result) {
  websocketpp::lib::error_code ec;
  nlohmann::json jsonresult;
  jsonresult["text"] = "";

  std::string tmp_online_msg = FunASRGetResult(result, 0);
  if (tmp_online_msg != "") {
    LOG(INFO) << "online_res :" << tmp_online_msg;
    jsonresult["text"] = tmp_online_msg;
    jsonresult["mode"] = "2pass-online";
  }
  std::string tmp_tpass_msg = FunASRGetTpassResult(result, 0);
  if (tmp_tpass_msg != "") {
    LOG(INFO) << "offline results : " << tmp_tpass_msg;
    jsonresult["text"] = tmp_tpass_msg;
    jsonresult["mode"] = "2pass-offline";
  }

  return jsonresult;
}
// feed buffer to asr engine for decoder
void WebSocketServer::do_decoder(
    std::vector<char>& buffer, websocketpp::connection_hdl& hdl,
    nlohmann::json& msg, std::vector<std::vector<std::string>>& punc_cache,
    websocketpp::lib::mutex& thread_lock, bool& is_final, std::string wav_name,
    FUNASR_HANDLE& tpass_online_handle) {
  // lock for each connection
  scoped_lock guard(thread_lock);
  if(!tpass_online_handle){
	  LOG(INFO) << "tpass_online_handle  is free, return";
	  msg["access_num"]=(int)msg["access_num"]-1;
	  return;
  }
  try {
    FUNASR_RESULT Result = nullptr;
    int asr_mode_ = 2;
    if (msg.contains("mode")) {
      std::string modeltype = msg["mode"];
      if (modeltype == "offline") {
        asr_mode_ = 0;
      } else if (modeltype == "online") {
        asr_mode_ = 1;
      } else if (modeltype == "2pass") {
        asr_mode_ = 2;
      }
    } else {
      // default value
      msg["mode"] = "2pass";
      asr_mode_ = 2;
    }

    // loop to send chunk_size 800*2 data to asr engine.   TODO: chunk_size need
    // get from client
    while (buffer.size() >= 800 * 2 && !msg["is_eof"]) {
      std::vector<char> subvector = {buffer.begin(), buffer.begin() + 800 * 2};
      buffer.erase(buffer.begin(), buffer.begin() + 800 * 2);

      try {
        if (tpass_online_handle) {
          Result = FunTpassInferBuffer(tpass_handle, tpass_online_handle,
                                       subvector.data(), subvector.size(),
                                       punc_cache, false, msg["audio_fs"],
                                       msg["wav_format"], (ASR_TYPE)asr_mode_);

        } else {
          msg["access_num"]=(int)msg["access_num"]-1;
          return;
        }
      } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
        msg["access_num"]=(int)msg["access_num"]-1;
        return;
      }
      if (Result) {
        websocketpp::lib::error_code ec;
        nlohmann::json jsonresult = handle_result(Result);
        jsonresult["wav_name"] = wav_name;
        jsonresult["is_final"] = false;
        if (jsonresult["text"] != "") {
          if (is_ssl) {
            wss_server_->send(hdl, jsonresult.dump(),
                              websocketpp::frame::opcode::text, ec);
          } else {
            server_->send(hdl, jsonresult.dump(),
                          websocketpp::frame::opcode::text, ec);
          }
        }
        FunASRFreeResult(Result);
      }
    }
    if (is_final && !msg["is_eof"]) {
      try {
        if (tpass_online_handle) {
          Result = FunTpassInferBuffer(tpass_handle, tpass_online_handle,
                                       buffer.data(), buffer.size(), punc_cache,
                                       is_final, msg["audio_fs"],
                                       msg["wav_format"], (ASR_TYPE)asr_mode_);
        } else {
          msg["access_num"]=(int)msg["access_num"]-1;	 
          return;
        }
      } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
        msg["access_num"]=(int)msg["access_num"]-1;
        return;
      }
      if(punc_cache.size()>0){
        for (auto& vec : punc_cache) {
          vec.clear();
        }
      }
      if (Result) {
        websocketpp::lib::error_code ec;
        nlohmann::json jsonresult = handle_result(Result);
        jsonresult["wav_name"] = wav_name;
        jsonresult["is_final"] = true;
        if (is_ssl) {
          wss_server_->send(hdl, jsonresult.dump(),
                            websocketpp::frame::opcode::text, ec);
        } else {
          server_->send(hdl, jsonresult.dump(),
                        websocketpp::frame::opcode::text, ec);
        }
        FunASRFreeResult(Result);
      }
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  msg["access_num"]=(int)msg["access_num"]-1;
 
}

void WebSocketServer::on_open(websocketpp::connection_hdl hdl) {
  scoped_lock guard(m_lock);     // for threads safty
  try{
    std::shared_ptr<FUNASR_MESSAGE> data_msg =
        std::make_shared<FUNASR_MESSAGE>();  // put a new data vector for new
                                            // connection
    data_msg->samples = std::make_shared<std::vector<char>>();
    data_msg->thread_lock = std::make_shared<websocketpp::lib::mutex>();  

    data_msg->msg = nlohmann::json::parse("{}");
    data_msg->msg["wav_format"] = "pcm";
    data_msg->msg["audio_fs"] = 16000;
    data_msg->msg["access_num"] = 0; // the number of access for this object, when it is 0, we can free it saftly
    data_msg->msg["is_eof"]=false; // if this connection is closed
    data_msg->punc_cache =
        std::make_shared<std::vector<std::vector<std::string>>>(2);
  	data_msg->strand_ =	std::make_shared<asio::io_context::strand>(io_decoder_);
    // std::vector<int> chunk_size = {5, 10, 5};  //TODO, need get from client
    // FUNASR_HANDLE tpass_online_handle =
    //     FunTpassOnlineInit(tpass_handle, chunk_size);
    // data_msg->tpass_online_handle = tpass_online_handle;

    data_map.emplace(hdl, data_msg);
    // LOG(INFO) << "on_open, active connections: " << data_map.size();
  }catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

void remove_hdl(
    websocketpp::connection_hdl hdl,
    std::map<websocketpp::connection_hdl, std::shared_ptr<FUNASR_MESSAGE>,
             std::owner_less<websocketpp::connection_hdl>>& data_map) {
 
  std::shared_ptr<FUNASR_MESSAGE> data_msg = nullptr;
  auto it_data = data_map.find(hdl);
  if (it_data != data_map.end()) {
    data_msg = it_data->second;
  } else {
    return;
  }
  // scoped_lock guard_decoder(*(data_msg->thread_lock));  //wait for do_decoder
  // finished and avoid access freed tpass_online_handle
  unique_lock guard_decoder(*(data_msg->thread_lock));
  if (data_msg->msg["access_num"]==0 && data_msg->msg["is_eof"]==true) {
    // LOG(INFO) << "----------------FunTpassOnlineUninit----------------------";
    FunTpassOnlineUninit(data_msg->tpass_online_handle);
    data_msg->tpass_online_handle = nullptr;
	  data_map.erase(hdl);
  }
 
  guard_decoder.unlock();
}

void WebSocketServer::on_close(websocketpp::connection_hdl hdl) {
  scoped_lock guard(m_lock);
  std::shared_ptr<FUNASR_MESSAGE> data_msg = nullptr;
  auto it_data = data_map.find(hdl);
  if (it_data != data_map.end()) {
    data_msg = it_data->second;
  } else {
    return;
  }
  unique_lock guard_decoder(*(data_msg->thread_lock));
  data_msg->msg["is_eof"]=true;
  guard_decoder.unlock();
  //LOG(INFO) << "on_close, active connections: " << data_map.size();
}
 
// remove closed connection
void WebSocketServer::check_and_clean_connection() {
  // LOG(INFO)<<"***********begin check_and_clean_connection ****************";
  while(true){
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    // LOG(INFO) << "run check_and_clean_connection" <<", active connections: " << data_map.size();
    std::vector<websocketpp::connection_hdl> to_remove;  // remove list
    auto iter = data_map.begin();
    while (iter != data_map.end()) {  // loop to find closed connection
      websocketpp::connection_hdl hdl = iter->first;
      try{
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
      }
      catch (std::exception const &e)
      {
        // if connection is close, we set is_eof = true
        std::shared_ptr<FUNASR_MESSAGE> data_msg = nullptr;
        auto it_data = data_map.find(hdl);
        if (it_data != data_map.end()) {
          data_msg = it_data->second;
        } else {
            continue;
        }
        unique_lock guard_decoder(*(data_msg->thread_lock));
        data_msg->msg["is_eof"]=true;
        guard_decoder.unlock();
        to_remove.push_back(hdl);
        LOG(INFO)<<"connection is closed: "<<e.what();
        
      }
      iter++;
    }
    for (auto hdl : to_remove) {
      {
        unique_lock lock(m_lock);
        remove_hdl(hdl, data_map);
      }
    }
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
  } else {
    lock.unlock();
    return;
  }

  std::shared_ptr<std::vector<char>> sample_data_p = msg_data->samples;
  std::shared_ptr<std::vector<std::vector<std::string>>> punc_cache_p =
      msg_data->punc_cache;
  std::shared_ptr<websocketpp::lib::mutex> thread_lock_p = msg_data->thread_lock;

  lock.unlock();

  if (sample_data_p == nullptr) {
    LOG(INFO) << "error when fetch sample data vector";
    return;
  }

  const std::string& payload = msg->get_payload();  // get msg type
  unique_lock guard_decoder(*(thread_lock_p)); // mutex for one connection
  switch (msg->get_opcode()) {
    case websocketpp::frame::opcode::text: {
      nlohmann::json jsonresult = nlohmann::json::parse(payload);

      if (jsonresult.contains("wav_name")) {
        msg_data->msg["wav_name"] = jsonresult["wav_name"];
      }
      if (jsonresult.contains("mode")) {
        msg_data->msg["mode"] = jsonresult["mode"];
      }
      if (jsonresult.contains("wav_format")) {
        msg_data->msg["wav_format"] = jsonresult["wav_format"];
      }
      if (jsonresult.contains("audio_fs")) {
        msg_data->msg["audio_fs"] = jsonresult["audio_fs"];
      }
      if (jsonresult.contains("chunk_size")) {
        if (msg_data->tpass_online_handle == NULL) {
          std::vector<int> chunk_size_vec =
              jsonresult["chunk_size"].get<std::vector<int>>();
          FUNASR_HANDLE tpass_online_handle =
              FunTpassOnlineInit(tpass_handle, chunk_size_vec);
          msg_data->tpass_online_handle = tpass_online_handle;
        }
      }
      LOG(INFO) << "jsonresult=" << jsonresult
                << ", msg_data->msg=" << msg_data->msg;
      if (jsonresult["is_speaking"] == false ||
          jsonresult["is_finished"] == true) {
        LOG(INFO) << "client done";

        // if it is in final message, post the sample_data to decode
        try{
		  
          msg_data->strand_->post(
              std::bind(&WebSocketServer::do_decoder, this,
                        std::move(*(sample_data_p.get())), std::move(hdl),
                        std::ref(msg_data->msg), std::ref(*(punc_cache_p.get())),
                        std::ref(*thread_lock_p), std::move(true),
                        msg_data->msg["wav_name"],
                        std::ref(msg_data->tpass_online_handle)));
		      msg_data->msg["access_num"]=(int)(msg_data->msg["access_num"])+1;
        }
        catch (std::exception const &e)
        {
            LOG(ERROR)<<e.what();
        }
      }
      break;
    }
    case websocketpp::frame::opcode::binary: {
      // recived binary data
      const auto* pcm_data = static_cast<const char*>(payload.data());
      int32_t num_samples = payload.size();

      if (isonline) {
        sample_data_p->insert(sample_data_p->end(), pcm_data,
                              pcm_data + num_samples);
        int setpsize =
            800 * 2;  // TODO, need get from client
                      // if sample_data size > setpsize, we post data to decode
        if (sample_data_p->size() > setpsize) {
          int chunksize = floor(sample_data_p->size() / setpsize);
          // make sure the subvector size is an integer multiple of setpsize
          std::vector<char> subvector = {
              sample_data_p->begin(),
              sample_data_p->begin() + chunksize * setpsize};
          // keep remain in sample_data
          sample_data_p->erase(sample_data_p->begin(),
                               sample_data_p->begin() + chunksize * setpsize);

          try{
            // post to decode
            msg_data->strand_->post( 
                      std::bind(&WebSocketServer::do_decoder, this,
                                std::move(subvector), std::move(hdl),
                                std::ref(msg_data->msg),
                                std::ref(*(punc_cache_p.get())),
                                std::ref(*thread_lock_p), std::move(false),
                                msg_data->msg["wav_name"],
                                std::ref(msg_data->tpass_online_handle)));
            msg_data->msg["access_num"]=(int)(msg_data->msg["access_num"])+1;
          }
          catch (std::exception const &e)
          {
              LOG(ERROR)<<e.what();
          }
        }
      } else {
        sample_data_p->insert(sample_data_p->end(), pcm_data,
                              pcm_data + num_samples);
      }
      break;
    }
    default:
      break;
  }
  guard_decoder.unlock();
}

// init asr model
void WebSocketServer::initAsr(std::map<std::string, std::string>& model_path,
                              int thread_num) {
  try {
    tpass_handle = FunTpassInit(model_path, thread_num);
    if (!tpass_handle) {
      LOG(ERROR) << "FunTpassInit init failed";
      exit(-1);
    }
    LOG(INFO) << "initAsr run check_and_clean_connection";
    std::thread clean_thread(&WebSocketServer::check_and_clean_connection,this);  
    clean_thread.detach();
    LOG(INFO) << "initAsr run check_and_clean_connection finished";

  } catch (const std::exception& e) {
    LOG(INFO) << e.what();
  }
}
