/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
// reply.hpp
// ~~~~~~~~~
//
// copy some codes from  http://www.boost.org/

#ifndef HTTP_SERVER2_REPLY_HPP
#define HTTP_SERVER2_REPLY_HPP

#include <asio.hpp>
#include <string>
#include <vector>

#include "header.hpp"

namespace http {
namespace server2 {

/// A reply to be sent to a client.
struct reply {
  /// The status of the reply.
  enum status_type {
    ok = 200,
    created = 201,
    accepted = 202,
    no_content = 204,
    multiple_choices = 300,
    moved_permanently = 301,
    moved_temporarily = 302,
    not_modified = 304,
    bad_request = 400,
    unauthorized = 401,
    forbidden = 403,
    not_found = 404,
    internal_server_error = 500,
    not_implemented = 501,
    bad_gateway = 502,
    service_unavailable = 503
  } status;

  /// The headers to be included in the reply.
  std::vector<header> headers;

  /// The content to be sent in the reply.
  std::string content;

  /// Convert the reply into a vector of buffers. The buffers do not own the
  /// underlying memory blocks, therefore the reply object must remain valid and
  /// not be changed until the write operation has completed.
  std::vector<::asio::const_buffer> to_buffers();

  /// Get a stock reply.
  static reply stock_reply(status_type status);
  static reply stock_reply(std::string jsonresult);
};

}  // namespace server2
}  // namespace http

#endif  // HTTP_SERVER2_REPLY_HPP
