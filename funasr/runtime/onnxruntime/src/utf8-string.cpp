// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//               2023 Jing Du (thuduj12@163.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utf8-string.h"
#include <glog/logging.h>

namespace funasr {
const char* WHITESPACE = " \n\r\t\f\v";

int char_length(char ch) {
  int num_bytes = 1;
  CHECK_LE((ch & 0xF8), 0xF0);
  if ((ch & 0x80) == 0x00) {
    // The first 128 characters (US-ASCII) in UTF-8 format only need one byte.
    num_bytes = 1;
  } else if ((ch & 0xE0) == 0xC0) {
    // The next 1,920 characters need two bytes to encode,
    // which covers the remainder of almost all Latin-script alphabets.
    num_bytes = 2;
  } else if ((ch & 0xF0) == 0xE0) {
    // Three bytes are needed for characters in the rest of
    // the Basic Multilingual Plane, which contains virtually all characters
    // in common use, including most Chinese, Japanese and Korean characters.
    num_bytes = 3;
  } else if ((ch & 0xF8) == 0xF0) {
    // Four bytes are needed for characters in the other planes of Unicode,
    // which include less common CJK characters, various historic scripts,
    // mathematical symbols, and emoji (pictographic symbols).
    num_bytes = 4;
  }
  return num_bytes;
}

int string_length(const std::string& str) {
  int len = 0;
  int num_bytes = 1;
  for (size_t i = 0; i < str.length(); i += num_bytes) {
    num_bytes = char_length(str[i]);
    ++len;
  }
  return len;
}

void string2chars(const std::string& str, std::vector<std::string>* chars) {
  chars->clear();
  int num_bytes = 1;
  for (size_t i = 0; i < str.length(); i += num_bytes) {
    num_bytes = char_length(str[i]);
    chars->push_back(str.substr(i, num_bytes));
  }
}

std::string ltrim(const std::string& str) {
  size_t start = str.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : str.substr(start);
}

std::string rtrim(const std::string& str) {
  size_t end = str.find_last_not_of(WHITESPACE);
  return end == std::string::npos ? "" : str.substr(0, end + 1);
}

std::string trim(const std::string& str) { return rtrim(ltrim(str)); }

void split_string(const std::string& str, const std::string& delim,
                  std::vector<std::string>* output) {
  std::string s = str;
  size_t pos = 0;
  while ((pos = s.find(delim)) != std::string::npos) {
    output->emplace_back(s.substr(0, pos));
    s.erase(0, pos + delim.length());
  }
  output->emplace_back(s);
}

}  // namespace funasr
