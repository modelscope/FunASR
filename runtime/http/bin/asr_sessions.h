/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
// FUNASR_MESSAGE define the needed message between funasr engine and http server
#ifndef HTTP_SERVER2_SESSIONS_HPP
#define HTTP_SERVER2_SESSIONS_HPP
#include "funasrruntime.h"
#include "nlohmann/json.hpp"
#include <atomic>
typedef struct {
  nlohmann::json msg;
  std::shared_ptr<std::vector<char>> samples;
  std::shared_ptr<std::vector<std::vector<float>>> hotwords_embedding=nullptr;
 
  FUNASR_DEC_HANDLE decoder_handle=nullptr;
  std::atomic<int> status;
} FUNASR_MESSAGE;
#endif // HTTP_SERVER2_REQUEST_PARSER_HPP