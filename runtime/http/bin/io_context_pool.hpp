/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
//
// io_context_pool.hpp
// ~~~~~~~~~~~~~~~~~~~
// copy some codes from  http://www.boost.org/

#ifndef HTTP_SERVER2_IO_SERVICE_POOL_HPP
#define HTTP_SERVER2_IO_SERVICE_POOL_HPP

#include <asio.hpp>
#include <list>
#include <memory>
#include <vector>

namespace http {
namespace server2 {

/// A pool of io_context objects.
class io_context_pool
{
public:
  /// Construct the io_context pool.
  explicit io_context_pool(std::size_t pool_size);

  /// Run all io_context objects in the pool.
  void run();

  /// Stop all io_context objects in the pool.
  void stop();

  /// Get an io_context to use.
  asio::io_context& get_io_context();

private:
  io_context_pool(const io_context_pool&) = delete;
  io_context_pool& operator=(const io_context_pool&) = delete;

  typedef std::shared_ptr<::asio::io_context> io_context_ptr;
  typedef asio::executor_work_guard<
    asio::io_context::executor_type> io_context_work;

  /// The pool of io_contexts.
  std::vector<io_context_ptr> io_contexts_;

  /// The work that keeps the io_contexts running.
  std::list<io_context_work> work_;

  /// The next io_context to use for a connection.
  std::size_t next_io_context_;
};

} // namespace server2
} // namespace http

#endif // HTTP_SERVER2_IO_SERVICE_POOL_HPP
