// util/kaldi-semaphore.cc

// Copyright 2012  Karel Vesely (Brno University of Technology)
//           2017  Dogan Can (University of Southern California)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#include "base/kaldi-error.h"
#include "util/kaldi-semaphore.h"

namespace kaldi {

Semaphore::Semaphore(int32 count) {
  KALDI_ASSERT(count >= 0);
  count_ = count;
}

Semaphore::~Semaphore() {}

bool Semaphore::TryWait() {
  std::unique_lock<std::mutex> lock(mutex_);
  if(count_) {
      count_--;
      return true;
  }
  return false;
}

void Semaphore::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  while(!count_)
    condition_variable_.wait(lock);
  count_--;
}

void Semaphore::Signal() {
  std::unique_lock<std::mutex> lock(mutex_);
  count_++;
  condition_variable_.notify_one();
}

} // namespace kaldi
