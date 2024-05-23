/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
//
// copy some codes from  http://www.boost.org/
//

#ifndef HTTP_SERVER2_CONNECTION_HPP
#define HTTP_SERVER2_CONNECTION_HPP

#include <array>
#include <asio.hpp>
#include <atomic>
#include <iostream>
#include <memory>

#include "reply.hpp"

#include <fstream>
 

#include "file_parse.hpp"
#include "model-decoder.h"
 

extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

namespace http {
namespace server2 {

/// Represents a single connection from a client.
class connection : public std::enable_shared_from_this<connection> {
 public:
  connection(const connection &) = delete;
  connection &operator=(const connection &) = delete;
  ~connection() { std::cout << "one connection is close()" << std::endl; };

  /// Construct a connection with the given socket.
  explicit connection(asio::ip::tcp::socket socket,
                      asio::io_context &io_decoder, int connection_id,
                      std::shared_ptr<ModelDecoder> model_decoder);

 
  /// Start the first asynchronous operation for the connection.
  void start();
  std::shared_ptr<FUNASR_MESSAGE> &get_data_msg();
  void write_back(std::string str);

 private:
  /// Perform an asynchronous read operation.
  void do_read();

  /// Perform an asynchronous write operation.
  void do_write();

  void do_decoder();

  void setup_timer();

  /// Socket for the connection.
  asio::ip::tcp::socket socket_;

 

  /// Buffer for incoming data.
  std::array<char, 8192> buffer_;
  /// for time out 
  std::shared_ptr<asio::steady_timer> s_timer;

  

  std::shared_ptr<ModelDecoder> model_decoder;

 

  int connection_id = 0;

  /// The reply to be sent back to the client.
  reply reply_;

  asio::io_context &io_decoder;

 

  std::shared_ptr<FUNASR_MESSAGE> data_msg;
 
  std::mutex m_lock;

 
  std::shared_ptr<asio::io_context::strand> strand_;

  std::shared_ptr<http::server2::file_parser> file_parse;
};

typedef std::shared_ptr<connection> connection_ptr;

}  // namespace server2
}  // namespace http

#endif  // HTTP_SERVER2_CONNECTION_HPP
