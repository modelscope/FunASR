// base/timer-test.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/timer.h"
#include "base/kaldi-common.h"
#include "base/kaldi-utils.h"


namespace kaldi {

void TimerTest() {
  float time_secs = 0.025 * (rand() % 10);
  std::cout << "target is " << time_secs << "\n";
  Timer timer;
  Sleep(time_secs);
  BaseFloat f = timer.Elapsed();
  std::cout << "time is " << f << std::endl;
  if (fabs(time_secs - f) > 0.05)
    KALDI_ERR << "Timer fail: waited " << f << " seconds instead of "
              <<  time_secs << " secs.";
}
}


int main() {
  for (int i = 0; i < 4; i++)
    kaldi::TimerTest();
}
