// Acknowledgement: this code is adapted from 
// https://github.com/wenet-e2e/WeTextProcessing/blob/master/runtime/utils/string.h
// Retrieved in Aug 2023.

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

#ifndef UTILS_UTF8_STRING_H_
#define UTILS_UTF8_STRING_H_

#include <string>
#include <vector>

namespace funasr {
extern const char* WHITESPACE;

int char_length(char ch);

int string_length(const std::string& str);

void string2chars(const std::string& str, std::vector<std::string>* chars);

std::string ltrim(const std::string& str);

std::string rtrim(const std::string& str);

std::string trim(const std::string& str);

void split_string(const std::string& str, const std::string& delim,
                  std::vector<std::string>* output);

}  // namespace funasr

#endif  // UTILS_UTF8_STRING_H_
