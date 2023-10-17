// Acknowledgement: this code is adapted from 
// https://github.com/wenet-e2e/WeTextProcessing/blob/master/runtime/processor/token_parser.cc
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

#include "itn-token-parser.h"
#include <glog/logging.h>
#include "utf8-string.h"

namespace funasr {
const std::string EOS = "<EOS>";
const std::set<std::string> UTF8_WHITESPACE = {" ", "\t", "\n", "\r",
                                               "\x0b\x0c"};
const std::set<std::string> ASCII_LETTERS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B",
    "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "_"};
const std::unordered_map<std::string, std::vector<std::string>> TN_ORDERS = {
    {"date", {"year", "month", "day"}},
    {"fraction", {"denominator", "numerator"}},
    {"measure", {"denominator", "numerator", "value"}},
    {"money", {"value", "currency"}},
    {"time", {"noon", "hour", "minute", "second"}}};
const std::unordered_map<std::string, std::vector<std::string>> ITN_ORDERS = {
    {"date", {"year", "month", "day"}},
    {"fraction", {"sign", "numerator", "denominator"}},
    {"measure", {"numerator", "denominator", "value"}},
    {"money", {"currency", "value"}},
    {"time", {"hour", "minute", "second", "noon"}}};

TokenParser::TokenParser(ParseType type) {
  if (type == ParseType::kTN) {
    orders = TN_ORDERS;
  } else {
    orders = ITN_ORDERS;
  }
}

void TokenParser::load(const std::string& input) {
  string2chars(input, &text);
  CHECK_GT(text.size(), 0);
  index = 0;
  ch = text[0];
}

bool TokenParser::read() {
  if (index < text.size() - 1) {
    index += 1;
    ch = text[index];
    return true;
  }
  ch = EOS;
  return false;
}

bool TokenParser::parse_ws() {
  bool not_eos = ch != EOS;
  while (not_eos && ch == " ") {
    not_eos = read();
  }
  return not_eos;
}

bool TokenParser::parse_char(const std::string& exp) {
  if (ch == exp) {
    read();
    return true;
  }
  return false;
}

bool TokenParser::parse_chars(const std::string& exp) {
  bool ok = false;
  std::vector<std::string> chars;
  string2chars(exp, &chars);
  for (const auto& x : chars) {
    ok |= parse_char(x);
  }
  return ok;
}

std::string TokenParser::parse_key() {
  CHECK_NE(ch, EOS);
  CHECK_EQ(UTF8_WHITESPACE.count(ch), 0);

  std::string key = "";
  while (ASCII_LETTERS.count(ch) > 0) {
    key += ch;
    read();
  }
  return key;
}

std::string TokenParser::parse_value() {
  CHECK_NE(ch, EOS);
  bool escape = false;

  std::string value = "";
  while (ch != "\"") {
    value += ch;
    escape = ch == "\\" && !escape;
    read();
    if (escape) {
      value += ch;
      read();
    }
  }
  return value;
}

void TokenParser::parse(const std::string& input) {
  load(input);
  while (parse_ws()) {
    std::string name = parse_key();
    parse_chars(" { ");

    Token token(name);
    while (parse_ws()) {
      if (ch == "}") {
        parse_char("}");
        break;
      }
      std::string key = parse_key();
      parse_chars(": \"");
      std::string value = parse_value();
      parse_char("\"");
      token.append(key, value);
    }
    tokens.emplace_back(token);
  }
}

std::string TokenParser::reorder(const std::string& input) {
  parse(input);
  std::string output = "";
  for (auto& token : tokens) {
    output += token.string(orders) + " ";
  }
  return trim(output);
}

}  // namespace funasr
