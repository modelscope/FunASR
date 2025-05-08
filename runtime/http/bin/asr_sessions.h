/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2025 by zhaomingwork@qq.com */
// FUNASR_MESSAGE define the needed message between funasr engine and http server
#ifndef HTTP_SERVER2_SESSIONS_HPP
#define HTTP_SERVER2_SESSIONS_HPP
#include "funasrruntime.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

class Semaphore {
public:
    explicit Semaphore(int count = 0) : count_(count) {}

    
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return count_ > 0; });
        --count_;
    }

 
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_one();
    }

private:
    int count_;
    std::mutex mutex_;
    std::condition_variable cv_;
};
typedef struct {
  nlohmann::json msg;
  std::shared_ptr<std::vector<char>> samples;
  std::shared_ptr<std::vector<std::vector<float>>> hotwords_embedding=nullptr;
 
  FUNASR_DEC_HANDLE decoder_handle=nullptr;
  std::atomic<int> status;
  //std::counting_semaphore<3> sem(0);
  Semaphore sem_resultok; 
} FUNASR_MESSAGE;




#endif // HTTP_SERVER2_REQUEST_PARSER_HPP