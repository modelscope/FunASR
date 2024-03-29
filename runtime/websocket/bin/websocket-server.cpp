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

extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

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
                                 nlohmann::json& msg,
                                 websocketpp::lib::mutex& thread_lock,
                                 std::vector<std::vector<float>> &hotwords_embedding,
                                 std::string wav_name,
                                 bool itn,
                                 int audio_fs,
                                 std::string wav_format,
                                 FUNASR_DEC_HANDLE& decoder_handle) {
  try {
    int num_samples = buffer.size();  // the size of the buf

    if (!buffer.empty() && hotwords_embedding.size() > 0) {
      std::string asr_result="";
      std::string stamp_res="";
      std::string stamp_sents="";
      try{
        FUNASR_RESULT Result = FunOfflineInferBuffer(
            asr_handle, buffer.data(), buffer.size(), RASR_NONE, nullptr, 
            hotwords_embedding, audio_fs, wav_format, itn, decoder_handle);
        if (Result != nullptr){
          asr_result = FunASRGetResult(Result, 0);  // get decode result
          stamp_res = FunASRGetStamp(Result);
          stamp_sents = FunASRGetStampSents(Result);
          FunASRFreeResult(Result);
        } else{
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          LOG(ERROR) << "FUNASR_RESULT is nullptr.";
        }
      }catch (std::exception const& e) {
        LOG(ERROR) << e.what();
      }

      websocketpp::lib::error_code ec;
      nlohmann::json jsonresult;        // result json
      jsonresult["text"] = asr_result;  // put result in 'text'
      jsonresult["mode"] = "offline";
	    jsonresult["is_final"] = false;
      if(stamp_res != ""){
        jsonresult["timestamp"] = stamp_res;
      }
      if(stamp_sents != ""){
        try{
          nlohmann::json json_stamp = nlohmann::json::parse(stamp_sents);
          jsonresult["stamp_sents"] = json_stamp;
        }catch (std::exception const &e)
        {
          LOG(ERROR)<<e.what();
          jsonresult["stamp_sents"] = "";
        }
      }
      jsonresult["wav_name"] = wav_name;

      // send the json to client
      if (is_ssl) {
        wss_server_->send(hdl, jsonresult.dump(),
                          websocketpp::frame::opcode::text, ec);
      } else {
        server_->send(hdl, jsonresult.dump(), websocketpp::frame::opcode::text,
                      ec);
      }

      LOG(INFO) << "buffer.size=" << buffer.size() << ",result json=" << jsonresult.dump();
    }else{
      LOG(INFO) << "Sent empty msg";
      websocketpp::lib::error_code ec;
      nlohmann::json jsonresult;        // result json
      jsonresult["text"] = "";  // put result in 'text'
      jsonresult["mode"] = "offline";
	    jsonresult["is_final"] = false;
      jsonresult["wav_name"] = wav_name;

      // send the json to client
      if (is_ssl) {
        wss_server_->send(hdl, jsonresult.dump(),
                          websocketpp::frame::opcode::text, ec);
      } else {
        server_->send(hdl, jsonresult.dump(), websocketpp::frame::opcode::text,
                      ec);
      }
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  scoped_lock guard(thread_lock);
  msg["access_num"]=(int)msg["access_num"]-1;
}

void WebSocketServer::on_open(websocketpp::connection_hdl hdl) {
  scoped_lock guard(m_lock);     // for threads safty
  std::shared_ptr<FUNASR_MESSAGE> data_msg =
      std::make_shared<FUNASR_MESSAGE>();  // put a new data vector for new
                                           // connection
  data_msg->samples = std::make_shared<std::vector<char>>();
  data_msg->thread_lock = std::make_shared<websocketpp::lib::mutex>();
  data_msg->msg = nlohmann::json::parse("{}");
  data_msg->msg["wav_format"] = "pcm";
  data_msg->msg["wav_name"] = "wav-default-id";
  data_msg->msg["itn"] = true;
  data_msg->msg["audio_fs"] = 16000; // default is 16k
  data_msg->msg["access_num"] = 0; // the number of access for this object, when it is 0, we can free it saftly
  data_msg->msg["is_eof"]=false;
  FUNASR_DEC_HANDLE decoder_handle =
    FunASRWfstDecoderInit(asr_handle, ASR_OFFLINE, global_beam_, lattice_beam_, am_scale_);
  data_msg->decoder_handle = decoder_handle;
  data_map.emplace(hdl, data_msg);
  LOG(INFO) << "on_open, active connections: " << data_map.size();
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

  LOG(INFO) << "on_close, active connections: " << data_map.size();
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
  unique_lock guard_decoder(*(data_msg->thread_lock));
  if (data_msg->msg["access_num"]==0 && data_msg->msg["is_eof"]==true) {
    FunWfstDecoderUnloadHwsRes(data_msg->decoder_handle);
    FunASRWfstDecoderUninit(data_msg->decoder_handle);
    data_msg->decoder_handle = nullptr;
	  data_map.erase(hdl);
    LOG(INFO) << "remove one connection";
  }
  guard_decoder.unlock();
}

void WebSocketServer::check_and_clean_connection() {
  while(true){
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
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
        LOG(INFO)<<"connection is closed.";
        
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
    if(msg_data->msg["is_eof"]){
      lock.unlock();
      return;
    }
  } else{
    lock.unlock();
    return;
  }

  std::shared_ptr<std::vector<char>> sample_data_p = msg_data->samples;
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
      nlohmann::json jsonresult;
      try{
        jsonresult = nlohmann::json::parse(payload);
      }catch (std::exception const &e)
      {
        LOG(ERROR)<<e.what();
        msg_data->msg["is_eof"]=true;
        guard_decoder.unlock();
        return;
      }

      if (jsonresult["wav_name"] != nullptr) {
        msg_data->msg["wav_name"] = jsonresult["wav_name"];
      }
      if (jsonresult["wav_format"] != nullptr) {
        msg_data->msg["wav_format"] = jsonresult["wav_format"];
      }

      // hotwords: fst/nn
      if(msg_data->hotwords_embedding == nullptr){
        std::unordered_map<std::string, int> merged_hws_map;
        std::string nn_hotwords = "";

        if (jsonresult["hotwords"] != nullptr) {
          std::string json_string = jsonresult["hotwords"];
          if (!json_string.empty()){
            nlohmann::json json_fst_hws;
            try{
              json_fst_hws = nlohmann::json::parse(json_string);
              if(json_fst_hws.type() == nlohmann::json::value_t::object){
                // fst
                try{
                  std::unordered_map<std::string, int> client_hws_map = json_fst_hws;
                  merged_hws_map.insert(client_hws_map.begin(), client_hws_map.end());
                } catch (const std::exception& e) {
                  LOG(INFO) << e.what();
                }
              }
            } catch (std::exception const &e)
            {
              LOG(ERROR)<<e.what();
              // nn
              std::string client_nn_hws = jsonresult["hotwords"];
              nn_hotwords += " " + client_nn_hws;
              // LOG(INFO) << "nn hotwords: " << client_nn_hws;
            }
          }
        }
        merged_hws_map.insert(hws_map_.begin(), hws_map_.end());

        // fst
        LOG(INFO) << "hotwords: ";
        for (const auto& pair : merged_hws_map) {
            nn_hotwords += " " + pair.first;
            LOG(INFO) << pair.first << " : " << pair.second;
        }
        FunWfstDecoderLoadHwsRes(msg_data->decoder_handle, fst_inc_wts_, merged_hws_map);

        // nn
        std::vector<std::vector<float>> new_hotwords_embedding= CompileHotwordEmbedding(asr_handle, nn_hotwords);
        msg_data->hotwords_embedding =
            std::make_shared<std::vector<std::vector<float>>>(new_hotwords_embedding);
      }
      if (jsonresult.contains("audio_fs")) {
        msg_data->msg["audio_fs"] = jsonresult["audio_fs"];
      }
      if (jsonresult.contains("itn")) {
        msg_data->msg["itn"] = jsonresult["itn"];
      }
      if ((jsonresult["is_speaking"] == false ||
          jsonresult["is_finished"] == true) && 
          msg_data->msg["is_eof"] != true && 
          msg_data->hotwords_embedding != nullptr) {
        LOG(INFO) << "client done";
        // for offline, send all receive data to decoder engine
        std::vector<std::vector<float>> hotwords_embedding_(*(msg_data->hotwords_embedding));
        asio::post(io_decoder_,
                    std::bind(&WebSocketServer::do_decoder, this,
                              std::move(*(sample_data_p.get())),
                              std::move(hdl), 
                              std::ref(msg_data->msg),
                              std::ref(*thread_lock_p),
                              std::move(hotwords_embedding_),
                              msg_data->msg["wav_name"],
                              msg_data->msg["itn"],
                              msg_data->msg["audio_fs"],
                              msg_data->msg["wav_format"],
                              std::ref(msg_data->decoder_handle)));
        msg_data->msg["access_num"]=(int)(msg_data->msg["access_num"])+1;
      }
      break;
    }
    case websocketpp::frame::opcode::binary: {
      // recived binary data
      const auto* pcm_data = static_cast<const char*>(payload.data());
      int32_t num_samples = payload.size();

      if (isonline) {
        // TODO
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
  guard_decoder.unlock();
}

// init asr model
void WebSocketServer::initAsr(std::map<std::string, std::string>& model_path,
                              int thread_num, bool use_gpu, int batch_size) {
  try {
    // init model with api

    asr_handle = FunOfflineInit(model_path, thread_num, use_gpu, batch_size);
    LOG(INFO) << "model successfully inited";
    
    LOG(INFO) << "initAsr run check_and_clean_connection";
    std::thread clean_thread(&WebSocketServer::check_and_clean_connection,this);  
    clean_thread.detach();
    LOG(INFO) << "initAsr run check_and_clean_connection finished";

  } catch (const std::exception& e) {
    LOG(INFO) << e.what();
  }
}
