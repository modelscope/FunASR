// util/simple-io-funcs.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "util/simple-io-funcs.h"
#include "util/text-utils.h"

namespace kaldi {

bool WriteIntegerVectorSimple(std::string wxfilename,
                              const std::vector<int32> &list) {
  kaldi::Output ko;
  // false, false is: text-mode, no Kaldi header.
  if (!ko.Open(wxfilename, false, false)) return false;
  for (size_t i = 0; i < list.size(); i++) ko.Stream() << list[i] << '\n';
  return ko.Close();
}

bool ReadIntegerVectorSimple(std::string rxfilename, std::vector<int32> *list) {
  kaldi::Input ki;
  if (!ki.OpenTextMode(rxfilename)) return false;
  std::istream &is = ki.Stream();
  int32 i;
  list->clear();
  while ( !(is >> i).fail() )
    list->push_back(i);
  is >> std::ws;
  return is.eof();  // should be eof, or junk at end of file.
}

bool WriteIntegerVectorVectorSimple(std::string wxfilename,
                                 const std::vector<std::vector<int32> > &list) {
  kaldi::Output ko;
  // false, false is: text-mode, no Kaldi header.
  if (!ko.Open(wxfilename, false, false)) return false;
  std::ostream &os = ko.Stream();
  for (size_t i = 0; i < list.size(); i++) {
    for (size_t j = 0; j < list[i].size(); j++) {
      os << list[i][j];
      if (j+1 < list[i].size()) os << ' ';
    }
    os << '\n';
  }
  return ko.Close();
}

bool ReadIntegerVectorVectorSimple(std::string rxfilename,
                                   std::vector<std::vector<int32> > *list) {
  kaldi::Input ki;
  if (!ki.OpenTextMode(rxfilename)) return false;
  std::istream &is = ki.Stream();
  list->clear();
  std::string line;
  while (std::getline(is, line)) {
    std::vector<int32> v;
    if (!SplitStringToIntegers(line, " \t\r", true, &v)) {
      list->clear();
      return false;
    }
    list->push_back(v);
  }
  return is.eof();  // if we're not at EOF, something weird happened.
}


}  // end namespace kaldi
