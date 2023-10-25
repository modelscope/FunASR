// util/kaldi-thread-test.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
//                 Frantisek Skala
//           2017  University of Southern California (Author: Dogan Can)

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

#include <algorithm>
#include "base/kaldi-common.h"
#include "util/kaldi-thread.h"

namespace kaldi {

// Sums up integers from 0 to max_to_count-1.
class MyThreadClass : public MultiThreadable {
 public:
  MyThreadClass(int32 max_to_count, int32 *i):
      max_to_count_(max_to_count), iptr_(i), private_counter_(0) { }

  // We are defining a copy constructor to ensure that whenever an instance of
  // this class is copied, the default *copy* constructor for MultiThreadable
  // is called instead the default constructor for MultiThreadable.
  MyThreadClass(const MyThreadClass &other):
      MultiThreadable(other),
      max_to_count_(other.max_to_count_), iptr_(other.iptr_),
      private_counter_(0) { }

  void operator() () {
    int32 block_size = (max_to_count_+ (num_threads_-1) ) / num_threads_;
    int32 start = block_size * thread_id_,
        end = std::min(max_to_count_, start + block_size);
    for (int32 j = start; j < end; j++)
      private_counter_ += j;
  }

  ~MyThreadClass() {
    *iptr_ += private_counter_;
  }

 private:
  MyThreadClass() { }  // Disallow empty constructor.
  int32 max_to_count_;
  int32 *iptr_;
  int32 private_counter_;
};


void TestThreads() {
  g_num_threads = 8;
  // run method with temporary threads on 8 threads
  // Note: uncomment following line for the possibility of simple benchmarking
  // for(int i=0; i<100000; i++)
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClass c(max_to_count, &tot);
    RunMultiThreaded(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
  g_num_threads = 1;
  // let's try the same, but with only one thread
  {
    int32 max_to_count = 10000, tot = 0;
    MyThreadClass c(max_to_count, &tot);
    RunMultiThreaded(c);
    KALDI_ASSERT(tot == (10000*(10000-1))/2);
  }
}

class MyTaskClass { // spins for a while, then outputs a pre-given integer.
 public:
  MyTaskClass(int32 i, std::vector<int32> *vec):
      done_(false), i_(i), vec_(vec) { }

  void operator() () {
    int32 spin = 1000000 * Rand() % 100;
    for (int32 i = 0; i < spin; i++);
    done_ = true;
  }
  ~MyTaskClass() {
    KALDI_ASSERT(done_);
    vec_->push_back(i_);
  }

 private:
  bool done_;
  int32 i_;
  std::vector<int32> *vec_;
};


void TestTaskSequencer() {
  TaskSequencerConfig config;
  config.num_threads = 1 + Rand() % 20;
  if (Rand() % 2 == 1 )
    config.num_threads_total = config.num_threads + Rand() % config.num_threads;

  int32 num_tasks = Rand() % 100;

  std::vector<int32> task_output;
  {
    TaskSequencer<MyTaskClass> sequencer(config);
    for (int32 i = 0; i < num_tasks; i++) {
      sequencer.Run(new MyTaskClass(i, &task_output));
    }
  } // and let "sequencer" be destroyed, which waits for the last threads.
  KALDI_ASSERT(task_output.size() == static_cast<size_t>(num_tasks));
  for (int32 i = 0; i < num_tasks; i++)
    KALDI_ASSERT(task_output[i] == i);
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  TestThreads();
  for (int32 i = 0; i < 10; i++)
    TestTaskSequencer();
}
