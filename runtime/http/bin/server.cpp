/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
//
// server.cpp
// copy some codes from  http://www.boost.org/

#include "server.hpp"

#include <signal.h>

#include <fstream>
#include <iostream>
#include <utility>

#include "util.h"
namespace http {
namespace server2 {

server::server(const std::string &address, const std::string &port,
               const std::string &doc_root, std::size_t io_context_pool_size,
               asio::io_context &decoder_context,
               std::map<std::string, std::string> &model_path, int thread_num)
    : io_context_pool_(io_context_pool_size),
      signals_(io_context_pool_.get_io_context()),
      acceptor_(io_context_pool_.get_io_context()),
      decoder_context(decoder_context) {
  // Register to handle the signals that indicate when the server should exit.
  // It is safe to register for the same signal multiple times in a program,
  // provided all registration for the specified signal is made through Asio.
  try {
    model_decoder =
        std::make_shared<ModelDecoder>(decoder_context, model_path, thread_num);

    LOG(INFO) << "try to listen on port:" << port << std::endl;
    LOG(INFO) << "still not work, pls wait... " << std::endl;
    LOG(INFO) << "if always waiting here, may be port in used, pls change the "
                 "port or kill pre-process!"
              << std::endl;

    atom_id = 0;

    // init model with api

    signals_.add(SIGINT);
    signals_.add(SIGTERM);
#if defined(SIGQUIT)
    signals_.add(SIGQUIT);
#endif  // defined(SIGQUIT)

    do_await_stop();

    // Open the acceptor with the option to reuse the address (i.e.
    // SO_REUSEADDR).
    asio::ip::tcp::resolver resolver(acceptor_.get_executor());
    asio::ip::tcp::endpoint endpoint = *resolver.resolve(address, port).begin();

    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));

    acceptor_.bind(endpoint);

    acceptor_.listen();

    do_accept();
    std::cout << "use curl to test,just as " << std::endl;
    std::cout << "curl -F \"file=@example.wav\" 127.0.0.1:80" << std::endl;

    std::cout << "http post only support offline mode, if you want online "
                 "mode, pls try websocket!"
              << std::endl;
    std::cout << "now succeed listen on port " << address << ":" << port
              << ", can accept data now!!!" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "error:" << e.what();
  }
}

void server::run() { io_context_pool_.run(); }

void server::do_accept() {
  acceptor_.async_accept(
      io_context_pool_.get_io_context(),
      [this](asio::error_code ec, asio::ip::tcp::socket socket) {
        // Check whether the server was stopped by a signal before this
        // completion handler had a chance to run.
        if (!acceptor_.is_open()) {
          return;
        }

        if (!ec) {
          std::lock_guard<std::mutex> lk(m_lock);
          atom_id = atom_id + 1;

          std::make_shared<connection>(std::move(socket), decoder_context,
                                       (atom_id).load(), model_decoder)
              ->start();
        }

        do_accept();
      });
}

void server::do_await_stop() {
  signals_.async_wait([this](asio::error_code /*ec*/, int /*signo*/) {
    io_context_pool_.stop();
  });
}

}  // namespace server2
}  // namespace http
