/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */

// funasr asr engine

#include "model-decoder.h"

#include <thread>
#include <utility>
#include <vector>

extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

// feed msg to asr engine for decoder
void ModelDecoder::do_decoder(std::shared_ptr<FUNASR_MESSAGE> session_msg) {
  try {
    //   std::this_thread::sleep_for(std::chrono::milliseconds(1000*10));
    if (session_msg->status == 1) return;
    //std::cout << "in do_decoder" << std::endl;
    std::shared_ptr<std::vector<char>> buffer = session_msg->samples;
    int num_samples = buffer->size();  // the size of the buf
    std::string wav_name =session_msg->msg["wav_name"];
    bool itn = session_msg->msg["itn"];
    int audio_fs = session_msg->msg["audio_fs"];;
    std::string wav_format = session_msg->msg["wav_format"];

 

    if (num_samples > 0 && session_msg->hotwords_embedding->size() > 0) {
      std::string asr_result = "";
      std::string stamp_res = "";
      std::string stamp_sents = "";

      try {
        std::vector<std::vector<float>> hotwords_embedding_(
            *(session_msg->hotwords_embedding));
   

        FUNASR_RESULT Result = FunOfflineInferBuffer(
            asr_handle, buffer->data(), buffer->size(), RASR_NONE, nullptr,
            std::move(hotwords_embedding_), audio_fs, wav_format, itn,
            session_msg->decoder_handle);

        if (Result != nullptr) {
          asr_result = FunASRGetResult(Result, 0);  // get decode result
          stamp_res = FunASRGetStamp(Result);
          stamp_sents = FunASRGetStampSents(Result);
          FunASRFreeResult(Result);

        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      } catch (std::exception const &e) {
        std::cout << "error in decoder!!! "<<e.what()  <<std::endl;
      }

      nlohmann::json jsonresult;        // result json
      jsonresult["text"] = asr_result;  // put result in 'text'
      jsonresult["mode"] = "offline";
      jsonresult["is_final"] = false;
      if (stamp_res != "") {
        jsonresult["timestamp"] = stamp_res;
      }
      if (stamp_sents != "") {
        try {
          nlohmann::json json_stamp = nlohmann::json::parse(stamp_sents);
          jsonresult["stamp_sents"] = json_stamp;
        } catch (std::exception const &e) {
          std::cout << "error:" << e.what();
          jsonresult["stamp_sents"] = "";
        }
      }
      jsonresult["wav_name"] = wav_name;

      std::cout << "buffer.size=" << buffer->size()
                << ",result json=" << jsonresult.dump() << std::endl;

      FunWfstDecoderUnloadHwsRes(session_msg->decoder_handle);
      FunASRWfstDecoderUninit(session_msg->decoder_handle);
      session_msg->status = 1;
      session_msg->msg["asr_result"] = jsonresult;
      return;
    } else {
      std::cout << "Sent empty msg";

      nlohmann::json jsonresult;  // result json
      jsonresult["text"] = "";    // put result in 'text'
      jsonresult["mode"] = "offline";
      jsonresult["is_final"] = false;
      jsonresult["wav_name"] = wav_name;
    }

  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

// init asr model
FUNASR_HANDLE ModelDecoder::initAsr(std::map<std::string, std::string> &model_path,
                           int thread_num) {
  try {
    // init model with api

    asr_handle = FunOfflineInit(model_path, thread_num);
    LOG(INFO) << "model successfully inited";

 
    return asr_handle;

  } catch (const std::exception &e) {
    LOG(INFO) << e.what();
    return nullptr;
  }
}
