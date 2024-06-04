/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */

// funasr asr engine

#ifndef MODEL_DECODER_SERVER_H_
#define MODEL_DECODER_SERVER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#define ASIO_STANDALONE 1  // not boost
#include <glog/logging.h>

#include <fstream>
#include <functional>
 

#include "asio.hpp"
#include "asr_sessions.h"
#include "com-define.h"
#include "funasrruntime.h"
#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"
#include "util/text-utils.h"

class ModelDecoder {
 public:
  ModelDecoder(asio::io_context &io_decoder,
               std::map<std::string, std::string> &model_path, int thread_num)
      : io_decoder_(io_decoder) {
    asr_handle = initAsr(model_path, thread_num);
 
  }
  void do_decoder(std::shared_ptr<FUNASR_MESSAGE> session_msg);

  FUNASR_HANDLE initAsr(std::map<std::string, std::string> &model_path, int thread_num);

 
 
  asio::io_context &io_decoder_;  // threads for asr decoder
  FUNASR_HANDLE get_asr_handle()
  {
    return asr_handle;
  }
 private:
 
  FUNASR_HANDLE asr_handle;  // asr engine handle
  bool isonline = false;  // online or offline engine, now only support offline
};

 
#endif  // MODEL_DECODER_SERVER_H_
