// util/simple-options.h

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

#ifndef KALDI_UTIL_SIMPLE_OPTIONS_H_
#define KALDI_UTIL_SIMPLE_OPTIONS_H_

#include <map>
#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {


/// The class SimpleOptions is an implementation of OptionsItf that allows
/// setting and getting option values programmatically, i.e., via getter
/// and setter methods. It doesn't provide any command line parsing
/// functionality.
/// The class ParseOptions should be used for command-line options.
class SimpleOptions : public OptionsItf {
 public:
  SimpleOptions() {
  }

  virtual ~SimpleOptions() {
  }

  // Methods from the interface
  void Register(const std::string &name, bool *ptr, const std::string &doc);
  void Register(const std::string &name, int32 *ptr, const std::string &doc);
  void Register(const std::string &name, uint32 *ptr, const std::string &doc);
  void Register(const std::string &name, float *ptr, const std::string &doc);
  void Register(const std::string &name, double *ptr, const std::string &doc);
  void Register(const std::string &name, std::string *ptr,
                const std::string &doc);

  // set option with the specified key, return true if successful
  bool SetOption(const std::string &key, const bool &value);
  bool SetOption(const std::string &key, const int32 &value);
  bool SetOption(const std::string &key, const uint32 &value);
  bool SetOption(const std::string &key, const float &value);
  bool SetOption(const std::string &key, const double &value);
  bool SetOption(const std::string &key, const std::string &value);
  bool SetOption(const std::string &key, const char* value);

  // get option with the specified key and put to 'value',
  // return true if successful
  bool GetOption(const std::string &key, bool *value);
  bool GetOption(const std::string &key, int32 *value);
  bool GetOption(const std::string &key, uint32 *value);
  bool GetOption(const std::string &key, float *value);
  bool GetOption(const std::string &key, double *value);
  bool GetOption(const std::string &key, std::string *value);

  enum OptionType {
    kBool,
    kInt32,
    kUint32,
    kFloat,
    kDouble,
    kString
  };

  struct OptionInfo {
    OptionInfo(const std::string &doc, OptionType type) :
      doc(doc), type(type) {
    }
    std::string doc;
    OptionType type;
  };

  std::vector<std::pair<std::string, OptionInfo> > GetOptionInfoList();

  /*
   * Puts the type of the option with name 'key' in the argument 'type'.
   * Return true if such option is found, false otherwise.
   */
  bool GetOptionType(const std::string &key, OptionType *type);

 private:

  std::vector<std::pair<std::string, OptionInfo> > option_info_list_;

  // maps for option variables
  std::map<std::string, bool*> bool_map_;
  std::map<std::string, int32*> int_map_;
  std::map<std::string, uint32*> uint_map_;
  std::map<std::string, float*> float_map_;
  std::map<std::string, double*> double_map_;
  std::map<std::string, std::string*> string_map_;
};

}  // namespace kaldi

#endif  // KALDI_UTIL_SIMPLE_OPTIONS_H_
