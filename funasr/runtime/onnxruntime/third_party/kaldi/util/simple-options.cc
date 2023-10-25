// util/simple-options.cc

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "util/simple-options.h"


namespace kaldi {

void SimpleOptions::Register(const std::string &name,
                             bool *value,
                             const std::string &doc) {
  bool_map_[name] = value;
  option_info_list_.push_back(std::make_pair(name, OptionInfo(doc, kBool)));
}

void SimpleOptions::Register(const std::string &name,
                             int32 *value,
                             const std::string &doc) {
  int_map_[name] = value;
  option_info_list_.push_back(std::make_pair(name, OptionInfo(doc, kInt32)));
}

void SimpleOptions::Register(const std::string &name,
                             uint32 *value,
                             const std::string &doc) {
  uint_map_[name] = value;
  option_info_list_.push_back(std::make_pair(name, OptionInfo(doc, kUint32)));
}

void SimpleOptions::Register(const std::string &name,
                             float *value,
                             const std::string &doc) {
  float_map_[name] = value;
  option_info_list_.push_back(std::make_pair(name, OptionInfo(doc, kFloat)));
}

void SimpleOptions::Register(const std::string &name,
                             double *value,
                             const std::string &doc) {
  double_map_[name] = value;
  option_info_list_.push_back(std::make_pair(name, OptionInfo(doc, kDouble)));
}

void SimpleOptions::Register(const std::string &name,
                             std::string *value,
                             const std::string &doc) {
  string_map_[name] = value;
  option_info_list_.push_back(std::make_pair(name, OptionInfo(doc, kString)));
}

template<typename T>
static bool SetOptionImpl(const std::string &key, const T &value,
                          std::map<std::string, T*> &some_map) {
  if (some_map.end() != some_map.find(key)) {
    *(some_map[key]) = value;
    return true;
  }
  return false;
}

bool SimpleOptions::SetOption(const std::string &key, const bool &value) {
  return SetOptionImpl(key, value, bool_map_);
}

bool SimpleOptions::SetOption(const std::string &key, const int32 &value) {
  if (!SetOptionImpl(key, value, int_map_)) {
    if (!SetOptionImpl(key, static_cast<uint32>(value), uint_map_)) {
      return false;
    }
  }
  return true;
}

bool SimpleOptions::SetOption(const std::string &key, const uint32 &value) {
  if (!SetOptionImpl(key, value, uint_map_)) {
    if (!SetOptionImpl(key, static_cast<int32>(value), int_map_)) {
      return false;
    }
  }
  return true;
}

bool SimpleOptions::SetOption(const std::string &key, const float &value) {
  if (!SetOptionImpl(key, value, float_map_)) {
    if (!SetOptionImpl(key, static_cast<double>(value), double_map_)) {
      return false;
    }
  }
  return true;
}

bool SimpleOptions::SetOption(const std::string &key, const double &value) {
  if (!SetOptionImpl(key, value, double_map_)) {
    if (!SetOptionImpl(key, static_cast<float>(value), float_map_)) {
      return false;
    }
  }
  return true;
}

bool SimpleOptions::SetOption(const std::string &key,
                              const std::string &value) {
  return SetOptionImpl(key, value, string_map_);
}

bool SimpleOptions::SetOption(const std::string &key, const char *value) {
  std::string str_value = std::string(value);
  return SetOptionImpl(key, str_value, string_map_);
}


template<typename T>
static bool GetOptionImpl(const std::string &key, T *value,
                          std::map<std::string, T*> &some_map) {
  typename std::map<std::string, T*>::iterator it  = some_map.find(key);
  if (it != some_map.end()) {
    *value = *(it->second);
    return true;
  }
  return false;
}

bool SimpleOptions::GetOption(const std::string &key, bool *value) {
  return GetOptionImpl(key, value, bool_map_);
}

bool SimpleOptions::GetOption(const std::string &key, int32 *value) {
  return GetOptionImpl(key, value, int_map_);
}

bool SimpleOptions::GetOption(const std::string &key, uint32 *value) {
  return GetOptionImpl(key, value, uint_map_);
}

bool SimpleOptions::GetOption(const std::string &key, float *value) {
  return GetOptionImpl(key, value, float_map_);
}

bool SimpleOptions::GetOption(const std::string &key, double *value) {
  return GetOptionImpl(key, value, double_map_);
}

bool SimpleOptions::GetOption(const std::string &key, std::string *value) {
  return GetOptionImpl(key, value, string_map_);
}

std::vector<std::pair<std::string, SimpleOptions::OptionInfo> >
SimpleOptions::GetOptionInfoList() {
  return option_info_list_;
}

bool SimpleOptions::GetOptionType(const std::string &key, OptionType *type) {
  for (std::vector <std::pair<std::string,
      OptionInfo> >::iterator dx = option_info_list_.begin();
      dx != option_info_list_.end(); dx++) {
    std::pair<std::string, SimpleOptions::OptionInfo> info_pair = (*dx);
    if (info_pair.first == key) {
      *type = info_pair.second.type;
      return true;
    }
  }
  return false;
}



}  // namespace kaldi
