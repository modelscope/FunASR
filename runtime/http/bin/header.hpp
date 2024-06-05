/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
//
// header.hpp
// copy some codes from  http://www.boost.org/

#ifndef HTTP_SERVER2_HEADER_HPP
#define HTTP_SERVER2_HEADER_HPP

#include <string>

namespace http {
namespace server2 {

struct header
{
  std::string name;
  std::string value;
};

} // namespace server2
} // namespace http

#endif // HTTP_SERVER2_HEADER_HPP
