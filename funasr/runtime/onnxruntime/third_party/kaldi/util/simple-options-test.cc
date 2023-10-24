// util/simple-options-test.cc

// Copyright 2013  Tanel Alumae, Tallinn University of Technology

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
#include "util/simple-options.h"

namespace kaldi {


void UnitTestSimpleOptions() {
  std::string str="default_for_str";
  int32 num = 1;
  uint32 unum = 2;
  float realnum = 0.1;
  bool flag = false;
  bool rval;
  SimpleOptions so;

  so.Register("num", &num, "Description of num");
  so.Register("unum", &unum, "Description of unum");
  so.Register("str", &str, "Description of str");
  so.Register("flag", &flag, "Description of flag");
  so.Register("realnum", &realnum, "Description of realnum");

  rval = so.SetOption("num", 42);
  KALDI_ASSERT(rval);
  so.SetOption("unum", static_cast<uint32>(43));
  KALDI_ASSERT(rval);
  rval = so.SetOption("str", (std::string)"foo");
  KALDI_ASSERT(rval);
  rval = so.SetOption("flag", false);
  KALDI_ASSERT(rval);

  KALDI_ASSERT(num == 42);
  KALDI_ASSERT(unum == 43);
  KALDI_ASSERT(str == "foo");
  KALDI_ASSERT(flag == false);

  rval = so.SetOption("str", "foo2");
  KALDI_ASSERT(rval);
  KALDI_ASSERT(str == "foo2");

  // test automatic conversion between int and uint
  rval = so.SetOption("unum", 44);
  KALDI_ASSERT(rval);
  KALDI_ASSERT(unum == 44);

  // test automatic conversion between float and double
  rval = so.SetOption("realnum", static_cast<float>(0.2));
  KALDI_ASSERT(rval);
  KALDI_ASSERT(realnum - 0.2 < 0.000001);
  rval = so.SetOption("realnum", static_cast<double>(0.3));
  KALDI_ASSERT(rval);
  KALDI_ASSERT(realnum - 0.3 < 0.000001);

  SimpleOptions::OptionType type;
  rval = so.GetOptionType("num", &type);
  KALDI_ASSERT(rval);
  KALDI_ASSERT(type == SimpleOptions::kInt32);

  rval = so.GetOptionType("xxxx", &type);
  KALDI_ASSERT(rval == false);
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestSimpleOptions();
  return 0;
}
