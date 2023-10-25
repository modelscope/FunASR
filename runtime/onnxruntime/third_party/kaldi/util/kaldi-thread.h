// util/kaldi-thread.h

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

#ifndef KALDI_THREAD_KALDI_THREAD_H_
#define KALDI_THREAD_KALDI_THREAD_H_ 1

#include <thread>
#include <algorithm>
#include "itf/options-itf.h"
#include "util/kaldi-semaphore.h"

// This header provides convenient mechanisms for parallelization.
//
// The class MultiThreader, and the function RunMultiThreaded provide a
// mechanism to run a specified number of jobs in parellel and wait for them
// all to finish. They accept objects of some class C that derives from the
// base class MultiThreadable. C needs to define the operator () that takes
// no arguments. See ExampleClass below.
//
// The class TaskSequencer addresses a different problem typically encountered
// in Kaldi command-line programs that process a sequence of items. The items
// to be processed are coming in. They are all of different sizes, e.g.
// utterances with different numbers of frames. We would like them to be
// processed in parallel to make good use of the threads available but they
// must be output in the same order they came in. Here, we again accept objects
// of some class C with an operator () that takes no arguments. C may also have
// a destructor with side effects (typically some kind of output).
// TaskSequencer is responsible for running the jobs in parallel. It has a
// function Run() that will accept a new object of class C; this will block
// until a thread is free, at which time it will spawn a thread that starts
// running the operator () of the object. When threads are finished running,
// the objects will be deleted. TaskSequencer guarantees that the destructors
// will be called sequentially (not in parallel) and in the same order the
// objects were given to the Run() function, so that it is safe for the
// destructor to have side effects such as outputting data.
// Note: the destructor of TaskSequencer will wait for any remaining jobs that
// are still running and will call the destructors.


namespace kaldi {

extern int32 g_num_threads;  // Maximum number of threads (for programs that
// use threads, which is not many of them, e.g. the SGMM update program does.
// This is 8 by default.  You can change this on the command line, where
// used, with --num-threads.  Programs that think they will use threads
// should register it with their ParseOptions, as something like:
// po.Register("num-threads", &g_num_threads, "Number of threads to use.");

class MultiThreadable {
  // To create a function object that does part of the job, inherit from this
  // class, implement a copy constructor calling the default copy constructor
  // of this base class (so that thread_id_ and num_threads_ are copied to new
  // instances), and finally implement the operator() that does part of the job
  // based on thread_id_ and num_threads_ variables.
  // Note: example implementations are in util/kaldi-thread-test.cc
 public:
  virtual void operator() () = 0;
  // Does the main function of the class
  //  Subclasses have to redefine this
  virtual ~MultiThreadable();
  // Optional destructor.  Note: the destructor of the object passed by the user
  // will also be called, so watch out.

 public:
  // Do not redeclare thread_id_ and num_threads_ in derived classes.
  int32 thread_id_;  // 0 <= thread_id_ < num_threads_
  int32 num_threads_;

 private:
  // Have additional member variables as needed.
};


class ExampleClass: public MultiThreadable {
 public:
  ExampleClass(int32 *foo); // Typically there will be an initializer that
  // takes arguments.

  ExampleClass(const ExampleClass &other); // A copy constructor is also needed;
  // some example classes use the default version of this.

  void operator() () {
    // Does the main function of the class.  This
    // function will typically want to look at the values of the
    // member variables thread_id_ and num_threads_, inherited
    // from MultiThreadable.
  }
  ~ExampleClass() {
    // Optional destructor.  Sometimes useful things happen here,
    // for example summing up of certain quantities.  See code
    // that uses RunMultiThreaded for examples.
  }
 private:
  // Have additional member variables as needed.
};


template<class C>
class MultiThreader {
 public:
  MultiThreader(int32 num_threads, const C &c_in) :
    threads_(std::max<int32>(1, num_threads)),
    cvec_(std::max<int32>(1, num_threads), c_in) {
    if (num_threads == 0) {
      // This is a special case with num_threads == 0, which behaves like with
      // num_threads == 1 but without creating extra threads.  This can be
      // useful in GPU computations where threads cannot be used.
      cvec_[0].thread_id_ = 0;
      cvec_[0].num_threads_ = 1;
      (cvec_[0])();
    } else {
      for (int32 i = 0; i < threads_.size(); i++) {
        cvec_[i].thread_id_ = i;
        cvec_[i].num_threads_ = threads_.size();
        threads_[i] = std::thread(std::ref(cvec_[i]));
      }
    }
  }
  ~MultiThreader() {
    for (size_t i = 0; i < threads_.size(); i++)
      if (threads_[i].joinable())
        threads_[i].join();
  }
 private:
  std::vector<std::thread> threads_;
  std::vector<C> cvec_;
};

/// Here, class C should inherit from MultiThreadable.  Note: if you want to
/// control the number of threads yourself, or need to do something in the main
/// thread of the program while the objects exist, just initialize the
/// MultiThreader<C> object yourself.
template<class C> void RunMultiThreaded(const C &c_in) {
  MultiThreader<C> m(g_num_threads, c_in);
}


struct TaskSequencerConfig {
  int32 num_threads;
  int32 num_threads_total;
  TaskSequencerConfig(): num_threads(1), num_threads_total(0)  { }
  void Register(OptionsItf *opts) {
    opts->Register("num-threads", &num_threads, "Number of actively processing "
                   "threads to run in parallel");
    opts->Register("num-threads-total", &num_threads_total, "Total number of "
                   "threads, including those that are waiting on other threads "
                   "to produce their output.  Controls memory use.  If <= 0, "
                   "defaults to --num-threads plus 20.  Otherwise, must "
                   "be >= num-threads.");
  }
};

// C should have an operator () taking no arguments, that does some kind
// of computation, and a destructor that produces some kind of output (the
// destructors will be run sequentially in the same order Run as called.
template<class C>
class TaskSequencer {
 public:
  TaskSequencer(const TaskSequencerConfig &config):
      num_threads_(config.num_threads),
      threads_avail_(config.num_threads),
      tot_threads_avail_(config.num_threads_total > 0 ? config.num_threads_total :
                         config.num_threads + 20),
      thread_list_(NULL) {
    KALDI_ASSERT((config.num_threads_total <= 0 ||
                  config.num_threads_total >= config.num_threads) &&
                 "num-threads-total, if specified, must be >= num-threads");
  }

  /// This function takes ownership of the pointer "c", and will delete it
  /// in the same sequence as Run was called on the jobs.
  void Run(C *c) {
    // run in main thread
    if (num_threads_ == 0) {
      (*c)();
      delete c;
      return;
    }

    threads_avail_.Wait(); // wait till we have a thread for computation free.
    tot_threads_avail_.Wait(); // this ensures we don't have too many threads
    // waiting on I/O, and consume too much memory.

    // put the new RunTaskArgsList object at head of the singly
    // linked list thread_list_.
    thread_list_ = new RunTaskArgsList(this, c, thread_list_);
    thread_list_->thread = std::thread(TaskSequencer<C>::RunTask,
                                       thread_list_);
  }

  void Wait() { // You call this at the end if it's more convenient
    // than waiting for the destructor.  It waits for all tasks to finish.
    if (thread_list_ != NULL) {
      while (!thread_list_->thread.joinable()) {
        Sleep(1);
      }
      thread_list_->thread.join();
      KALDI_ASSERT(thread_list_->tail == NULL); // thread would not
      // have exited without setting tail to NULL.
      delete thread_list_;
      thread_list_ = NULL;
    }
  }

  /// The destructor waits for the last thread to exit.
  ~TaskSequencer() {
    Wait();
  }
 private:
  struct RunTaskArgsList {
    TaskSequencer *me; // Think of this as a "this" pointer.
    C *c; // Clist element of the task we're expected
    std::thread thread;
    RunTaskArgsList *tail;
    RunTaskArgsList(TaskSequencer *me, C *c, RunTaskArgsList *tail):
        me(me), c(c), tail(tail) {}
  };
  // This static function gets run in the threads that we create.
  static void RunTask(RunTaskArgsList *args) {
    // (1) run the job.
    (*(args->c))(); // call operator () on args->c, which does the computation.
    args->me->threads_avail_.Signal(); // Signal that the compute-intensive
    // part of the thread is done (we want to run no more than
    // config_.num_threads of these.)

    // (2) we want to destroy the object "c" now, by deleting it.  But for
    //     correct sequencing (this is the whole point of this class, it
    //     is intended to ensure the output of the program is in correct order),
    //     we first wait till the previous thread, whose details will be in "tail",
    //     is finished.
    if (args->tail != NULL) {
      while (!args->tail->thread.joinable()){
        Sleep(1);
      }
	    args->tail->thread.join();
    }
    delete args->c; // delete the object "c".  This may cause some output,
    // e.g. to a stream.  We don't need to worry about concurrent access to
    // the output stream, because each thread waits for the previous thread
    // to be done, before doing this.  So there is no risk of concurrent
    // access.
    args->c = NULL;

    if (args->tail != NULL) {
      KALDI_ASSERT(args->tail->tail == NULL); // Because we already
      // did join on args->tail->thread, which means that
      // thread was done, and before it exited, it would have
      // deleted and set to NULL its tail (which is the next line of code).
      delete args->tail;
      args->tail = NULL;
    }
    // At this point we are exiting from the thread.  Signal the
    // "tot_threads_avail_" semaphore which is used to limit the total number of threads that are alive, including
    // not onlhy those that are in active computation in c->operator (), but those
    // that are waiting on I/O or other threads.
    args->me->tot_threads_avail_.Signal();
  }

  int32 num_threads_; // copy of config.num_threads (since Semaphore doesn't store original count)

  Semaphore threads_avail_; // Initialized to the number of threads we are
  // supposed to run with; the function Run() waits on this.

  Semaphore tot_threads_avail_; // We use this semaphore to ensure we don't
  // consume too much memory...
  RunTaskArgsList *thread_list_;

};

} // namespace kaldi

#endif  // KALDI_THREAD_KALDI_THREAD_H_
