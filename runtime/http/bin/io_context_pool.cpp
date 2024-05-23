/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
//
// io_context_pool.cpp
// ~~~~~~~~~~~~~~~~~~~
// copy some codes from  http://www.boost.org/

#include "io_context_pool.hpp"
#include <stdexcept>
#include <thread>

namespace http {
namespace server2 {

io_context_pool::io_context_pool(std::size_t pool_size)
  : next_io_context_(0)
{
  if (pool_size == 0)
    throw std::runtime_error("io_context_pool size is 0");

  // Give all the io_contexts work to do so that their run() functions will not
  // exit until they are explicitly stopped.
  for (std::size_t i = 0; i < pool_size; ++i)
  {
    io_context_ptr io_context(new asio::io_context);
    io_contexts_.push_back(io_context);
    work_.push_back(asio::make_work_guard(*io_context));
  }
}

void io_context_pool::run()
{
  // Create a pool of threads to run all of the io_contexts.
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < io_contexts_.size(); ++i)
    threads.emplace_back([this, i]{ io_contexts_[i]->run(); });

  // Wait for all threads in the pool to exit.
  for (std::size_t i = 0; i < threads.size(); ++i)
    threads[i].join();
}

void io_context_pool::stop()
{
  // Explicitly stop all io_contexts.
  for (std::size_t i = 0; i < io_contexts_.size(); ++i)
    io_contexts_[i]->stop();
}

asio::io_context& io_context_pool::get_io_context()
{
  // Use a round-robin scheme to choose the next io_context to use.
  
  asio::io_context& io_context = *io_contexts_[next_io_context_];
 
  ++next_io_context_;
  if (next_io_context_ == io_contexts_.size())
    next_io_context_ = 0;
  return io_context;
}

} // namespace server2
} // namespace http
