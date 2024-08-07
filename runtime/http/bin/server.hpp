/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
//
// server.hpp
// ~~~~~~~~~~
// copy some codes from  http://www.boost.org/

#ifndef HTTP_SERVER2_SERVER_HPP
#define HTTP_SERVER2_SERVER_HPP
#include <asio.hpp>
#include <atomic>
#include <string>

#include "connection.hpp"
#include "funasrruntime.h"
#include "io_context_pool.hpp"
#include "model-decoder.h"
#include "util.h"
namespace http {
namespace server2 {

/// The top-level class of the HTTP server.
class server {
 public:
  server(const server &) = delete;
  server &operator=(const server &) = delete;

  /// Construct the server to listen on the specified TCP address and port, and
  /// serve up files from the given directory.
  explicit server(const std::string &address, const std::string &port,
                  const std::string &doc_root, std::size_t io_context_pool_size,
                  asio::io_context &decoder_context,
                  std::map<std::string, std::string> &model_path,
                  int thread_num);

  /// Run the server's io_context loop.
  void run();

 private:
  /// Perform an asynchronous accept operation.
  void do_accept();

  /// Wait for a request to stop the server.
  void do_await_stop();

  /// The pool of io_context objects used to perform asynchronous operations.
  io_context_pool io_context_pool_;

  asio::io_context &decoder_context;

  /// The signal_set is used to register for process termination notifications.
  asio::signal_set signals_;

  /// Acceptor used to listen for incoming connections.
  asio::ip::tcp::acceptor acceptor_;

 

  std::shared_ptr<ModelDecoder> model_decoder;

  std::atomic<int> atom_id;
  std::mutex m_lock;
};

}  // namespace server2
}  // namespace http

#endif  // HTTP_SERVER2_SERVER_HPP
