/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2023-2024 by zhaomingwork@qq.com */
// ~~~~~~~~~~~~~~~~~~


#ifndef HTTP_SERVER2_REQUEST_FILEPARSER_HPP
#define HTTP_SERVER2_REQUEST_FILEPARSER_HPP

#include <iostream>
#include <memory>
#include <tuple>

#include "asr_sessions.h"
namespace http {
namespace server2 {

/// Parser for incoming requests.
class file_parser {
 public:
  /// Construct ready to parse the request method.

  explicit file_parser(std::shared_ptr<FUNASR_MESSAGE> data_msg);

  /// Result of parse.
  enum result_type { start, boundary, filename, data, ok };

  template <typename InputIterator>
  void parse_one_line(InputIterator &is, InputIterator &ie, InputIterator &it) {
    if (is != it) {
      is = it;
    }
    if (*it == '\n') {
      is = std::next(is);
    }

    it = std::find(is, ie, '\n');
    std::string str(is, it);
    // std::cout<<"parse_one_line="<<str<<std::endl;
    // is = it;
    // return it;
  }
  std::string trim_name(std::string raw_string) {
    int pos = raw_string.find('\"');

    if (pos != std::string::npos) {
      raw_string = raw_string.substr(pos + 1);
      pos = raw_string.find('\"');
      raw_string = raw_string.substr(0, pos);
    }
    return raw_string;
  }

  std::string parese_file_ext(std::string file_name) {
    int pos = file_name.rfind('.');
    std::string ext = "";
    if (pos != std::string::npos) ext = file_name.substr(pos + 1);

    return ext;
  }
  template <typename InputIterator>
  int parse_data_content(InputIterator is, InputIterator ie, InputIterator it) {
    int len = std::distance(it + 1, ie);
    if (len <= 0) {
      return 0;
    }
    std::string str(it + 1, ie);

    // check if at the end, "--bonary--" need +4 for "--"
    if (len == bonary.length() + 4)

    {
      std::string str(it + 1, ie);
      // std::cout << "len good=" << str << std::endl;
      if (bonary.length() > 1 && bonary[bonary.length() - 1] == '\n') {
        // remove '\n' in bonary
        bonary = bonary.substr(0, bonary.length() - 2);
      }
      if (bonary.length() > 1 && bonary[bonary.length() - 1] == '\r') {
        // remove '\r' in bonary
        bonary = bonary.substr(0, bonary.length() - 2);
      }

      auto found_boundary = str.find(bonary);

      if (found_boundary == std::string::npos) {
        std::cout << "not found end bonary!=" << found_boundary << std::endl;
        // is=std::next(it);

        return 0;
      }
      // remove the end of data that contains '\n' or '\r'
      int last_sub = 0;
      if (*(it) == '\n') {
        last_sub++;
      }

      /*if (*(it - 1) == '\r') {
        last_sub++;
      }*/
      int lasts_len = std::distance(it, ie);

      data_msg->samples->erase(data_msg->samples->end() - last_sub - lasts_len,
                               data_msg->samples->end());
      std::cout << "one file finished, file size=" << data_msg->samples->size()
                << std::endl;
      return 1;
    }
    // is=std::next(it);
  }
  template <typename InputIterator>
  void parse_bonary_content(InputIterator is, InputIterator ie,
                            InputIterator it) {
    parse_one_line(is, ie, it);
    std::string str;  //(is, it + 1);

    while (it != ie) {
      // std::cout << "now_state == boundary:" << str <<str.length()<<
      // std::endl;
      str = std::string(is, it);
      std::cout << "bonary: str=" << str << std::endl;
      auto found_content = str.find("Content-Disposition:");
      auto found_filename = str.find("filename=");
      if (found_content != std::string::npos &&
          found_filename != std::string::npos) {
        std::string file_name =
            str.substr(found_filename + 9, std::string::npos);
        file_name = trim_name(file_name);

        std::string ext = parese_file_ext(file_name);
        // std::cout << "file_name1=" << file_name <<",ext="<<ext<<std::endl;
        if (file_name.find(".wav") != std::string::npos) {
          std::cout << "set wav_format=pcm, file_name=" << file_name
                    << std::endl;
          data_msg->msg["wav_format"] = "pcm";
        } else {
          std::cout << "set wav_format=" << ext << ", file_name=" << file_name
                    << std::endl;
          data_msg->msg["wav_format"] = ext;
        }
        data_msg->msg["wav_name"] = file_name;
        now_state = data;
      } else {
        auto found_content = str.find("Content-Disposition:");
        auto found_name = str.find("name=");
        if (found_content != std::string::npos &&
            found_name != std::string::npos) {
          std::string name = str.substr(found_name + 5, std::string::npos);
          name = trim_name(name);
          parse_one_line(is, ie, it);
          if (*it == '\n') it++;
          parse_one_line(is, ie, it);
          str = std::string(is, it);
          std::cout << "para: name=" << name << ",value=" << str << std::endl;
        }
      }

      parse_one_line(is, ie, it);
      if (now_state == data && std::distance(is, it) <= 2) {
        break;
      }
      //  str=std::string(is, it + 1);
    }
    // if(*(it-1)=='\r')it++;

    if (now_state == data) {
      if (*it == '\n') it++;
      //   std::cout<<"sample0
      //   size="<<data_msg->samples->size()<<std::endl;
      data_msg->samples->insert(data_msg->samples->end(), it,
                                it + std::distance(it, ie));
      // it=ie;
    }
  }
  template <typename InputIterator>
  result_type parse_file(InputIterator is, InputIterator ie) {
    // std::cout <<
    // "*************************************************************8"  <<
    // std::endl;
    if (now_state == data) {
      data_msg->samples->insert(data_msg->samples->end(), is, ie);
    }
    auto it = is;

    while (it != ie) {
      std::string str(is, it);
      // std::cout << "data_msg->samples=" << data_msg->samples->size() << ","
      // <<str<< std::endl;
      parse_one_line(is, ie, it);
      if (now_state == data) {
        // for data end search

        int ret = parse_data_content(is, ie, it);
        if (ret == 0) continue;
        return ok;
      } else {
        std::string str(is, it + 1);
        // std::cout << "parse:" << str <<std::endl;
        // <<str.length()<<",now_state="<<now_state<<std::endl;

        if (now_state == start) {
          // at start, for "boundary=" header
          // std::cout << "start: str="<<str<<std::endl;

          auto found_boundary = str.find("Content-Length:");
          if (found_boundary != std::string::npos) {
            std::string file_len =
                str.substr(found_boundary + 15, std::string::npos);
            // std::cout << "file_len="<<file_len<<std::endl;
            // std::stoi(file_len)
            data_msg->samples->reserve(std::stoi(file_len));
            std::cout << "reserve size for buffer=" << file_len << std::endl;
            // now_state = boundary;
          }
          found_boundary = str.find("boundary=");
          if (found_boundary != std::string::npos) {
            bonary = str.substr(found_boundary + 9, std::string::npos);
            now_state = boundary;
          }
        } else if (now_state == boundary) {
          // for file header
          auto found_boundary = str.find(bonary);
          if (found_boundary != std::string::npos) {
            parse_bonary_content(is, ie, it);
          }
        }
        // is = it;
      }

      // is = it;
    }

    return now_state;
  }

 private:
  std::shared_ptr<FUNASR_MESSAGE> data_msg;
  result_type now_state;
  std::string bonary = "";
};

}  // namespace server2
}  // namespace http

#endif  // HTTP_SERVER2_REQUEST_FILEPARSER_HPP
