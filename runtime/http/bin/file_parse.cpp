/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */


#include "file_parse.hpp"
 


namespace http {
namespace server2 {

 
file_parser::file_parser(std::shared_ptr<FUNASR_MESSAGE> data_msg)
:data_msg(data_msg)
   
{
	now_state=start;
}

 
 

 

} // namespace server2
} // namespace http
