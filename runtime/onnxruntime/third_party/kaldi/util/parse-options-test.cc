// util/parse-options-test.cc

// Copyright 2009-2011  Microsoft Corporation
// Copyright 2012-2013  Frantisek Skala;  Arnab Ghoshal
// Copyright 2013       Tanel Alumae

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
#include "util/parse-options.h"

namespace kaldi {

struct DummyOptions {
  int32 my_int;
  bool my_bool;
  std::string my_string;

  DummyOptions() {
    my_int = 0;
    my_bool = true;
    my_string = "default dummy string";
  }

  void Register(ParseOptions *po) {
    po->Register("my-int", &my_int,
                 "An int32 variable in DummyOptions.");
    po->Register("my-bool", &my_bool,
                 "A Boolean varaible in DummyOptions.");
    po->Register("my-str", &my_string,
                 "A string varaible in DummyOptions.");
  }
};

void UnitTestParseOptions() {
  int argc = 7;
  std::string str="default_for_str";
  int32 num = 1;
  uint32 unum = 2;
  const char *argv[7] = { "program_name", "--unum=5", "--num=3", "--i=boo",
    "a", "b", "c" };
  ParseOptions po("my usage msg");
  po.Register("i", &str, "My variable");
  po.Register("num", &num, "My int32 variable");
  po.Register("unum", &unum, "My uint32 variable");
  po.Read(argc, argv);
  KALDI_ASSERT(po.NumArgs() == 3);
  KALDI_ASSERT(po.GetArg(1) == "a");
  KALDI_ASSERT(po.GetArg(2) == "b");
  KALDI_ASSERT(po.GetArg(3) == "c");
  KALDI_ASSERT(unum == 5);
  KALDI_ASSERT(num == 3);
  KALDI_ASSERT(str == "boo");

  ParseOptions po2("my another msg");
  int argc2 = 4;
  const char *argv2[4] = { "program_name", "--i=foo",
    "--to-be-NORMALIZED=test", "c" };
  std::string str2 = "default_for_str2";
  po2.Register("To_Be_Normalized", &str2,
               "My variable (name has to be normalized)");
  po2.Register("i", &str, "My variable");
  po2.Read(argc2, argv2);
  KALDI_ASSERT(po2.NumArgs() == 1);
  KALDI_ASSERT(po2.GetArg(1) == "c");
  KALDI_ASSERT(str2 == "test");
  KALDI_ASSERT(str == "foo");

  ParseOptions po3("now checking options with prefix");
  ParseOptions ro3("prefix", &po3);  // to register with prefix
  ParseOptions so3("prefix2", &ro3);  // to register with prefix, recursively.
  DummyOptions dummy_opts;
  po3.Register("str", &str, "My string variable");
  po3.Register("num", &num, "My int32 variable");
  // Now register with prefix
  ro3.Register("unum", &unum, "My uint32 variable");
  ro3.Register("str", &str2, "My other string variable");
  uint32 unum2 = 0;
  so3.Register("unum", &unum2, "Another uint32 variable");

  int argc3 = 10;
  const char *argv3[10] = {
    "program_name", "--prefix.unum=5", "--num=3",
    "--prefix.str=foo", "--str=bar", "--prefix.my-bool=false",
    "--prefix.my-str=baz", "--prefix.prefix2.unum=42", "a", "b" };

  dummy_opts.Register(&ro3);
  po3.PrintUsage(false);

  po3.Read(argc3, argv3);
  KALDI_ASSERT(po3.NumArgs() == 2);
  KALDI_ASSERT(po3.GetArg(1) == "a");
  KALDI_ASSERT(po3.GetArg(2) == "b");
  KALDI_ASSERT(unum == 5);
  KALDI_ASSERT(unum2 == 42);
  KALDI_ASSERT(num == 3);
  KALDI_ASSERT(str2 == "foo");
  KALDI_ASSERT(str == "bar");
  KALDI_ASSERT(dummy_opts.my_bool == false);
  KALDI_ASSERT(dummy_opts.my_string == "baz");


  try {   // test error with --option=, which is not a valid way to set
          // boolean options.
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option="};
    ParseOptions po4("my usage msg");
    bool val = false;
    po4.Register("option", &val, "My boolean");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }

  { // test that --option sets "option" to true, if bool.
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option"};
    ParseOptions po4("my usage msg");
    bool val = false;
    po4.Register("option", &val, "My boolean");
    po4.Read(argc4, argv4);
    KALDI_ASSERT(val == true);
  }


  try {   // test error with --option, which is not a valid way to set
          // string-valued options.
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option"};
    ParseOptions po4("my usage msg");
    std::string val;
    po4.Register("option", &val, "My string");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }

  { // test that --option= sets "option" to empty, if string.
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option="};
    ParseOptions po4("my usage msg");
    std::string val = "foo";
    po4.Register("option", &val, "My boolean");
    po4.Read(argc4, argv4);
    KALDI_ASSERT(val.empty());
  }

  { // integer options test
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=8"};
    ParseOptions po4("my usage msg");
    int32 val = 32;
    po4.Register("option", &val, "My int");
    po4.Read(argc4, argv4);
    KALDI_ASSERT(val == 8);
  }

  { // float
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=8.5"};
    ParseOptions po4("my usage msg");
    BaseFloat val = 32.0;
    po4.Register("option", &val, "My float");
    po4.Read(argc4, argv4);
    KALDI_ASSERT(val == 8.5);
  }
  { // string options test
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=bar"};
    ParseOptions po4("my usage msg");
    std::string val = "foo";
    po4.Register("option", &val, "My string");
    po4.Read(argc4, argv4);
    KALDI_ASSERT(val == "bar");
  }
  
  try {   // test error with --float=string
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=foo"};
    ParseOptions po4("my usage msg");
    BaseFloat val = 32.0;
    po4.Register("option", &val, "My float");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }


  try {   // test error with --int=string
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=foo"};
    ParseOptions po4("my usage msg");
    int32 val = 32;
    po4.Register("option", &val, "My int");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }

  try {   // test error with --int=int+garbage
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=12xyz"};
    ParseOptions po4("my usage msg");
    int32 val = 32;
    po4.Register("option", &val, "My int");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }

  try {   // test error with --unsigned-int=negative-number.
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=-13"};
    ParseOptions po4("my usage msg");
    uint32 val = 32;
    po4.Register("option", &val, "My int");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected)xxx.";
  }

  try {   // test error with --bool=string
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--option=foo"};
    ParseOptions po4("my usage msg");
    bool val = false;
    po4.Register("option", &val, "My bool");
    po4.Read(argc4, argv4);
    assert(false); // Should not reach this part of code.
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }


  // test error with --=
  try {
    int argc4 = 2;
    const char *argv4[2] = { "program_name", "--=8"};
    int32 num = 0;
    ParseOptions po4("my usage msg");
    po4.Register("num", &num, "My int32 variable");
    po4.Read(argc4, argv4);
    KALDI_ASSERT(num == 0);
  } catch(std::exception e) {
    KALDI_LOG << "Failed to read option (this is expected).";
  }

  // test "--" (no more options)
  int argc4 = 5;
  unum = 2;
  const char *argv4[5] = { "program_name", "--unum=6", "--",  "a", "b" };
  ParseOptions po4("my usage msg");
  po4.Register("unum", &unum, "My uint32 variable");
  po4.Read(argc4, argv4);
  KALDI_ASSERT(po4.NumArgs() == 2);
  KALDI_ASSERT(po4.GetArg(1) == "a");
  KALDI_ASSERT(po4.GetArg(2) == "b");
  KALDI_ASSERT(unum == 6);

  // test obsolete "--" (no more options)
  int argc5 = 3;
  unum = 2;
  const char *argv5[3] = { "program_name", "--unum=7", "--" };
  ParseOptions po5("my usage msg");
  po5.Register("unum", &unum, "My uint32 variable");
  po5.Read(argc5, argv5);
  KALDI_ASSERT(po5.NumArgs() == 0);
  KALDI_ASSERT(unum == 7);

  // test that "--foo=bar" after "--" is interpreted as argument
  int argc6 = 4;
  unum = 2;
  const char *argv6[5] = { "program_name", "--unum=8", "--", "--foo=8" };
  ParseOptions po6("my usage msg");
  po6.Register("unum", &unum, "My uint32 variable");
  po6.Read(argc6, argv6);
  KALDI_ASSERT(po6.NumArgs() == 1);
  KALDI_ASSERT(po6.GetArg(1) == "--foo=8");

  // test that a second registration is ignored
  int argc7 = 2;
  const char *const argv7[] = {"program_name", "--i=8"};
  ParseOptions po7("my usage msg");

  int i7 = 10, k7 = 20;
  po7.Register("i", &i7, "My int32 variable");
  po7.Register("i", &k7, "My int32 variable");

  po7.Read(argc7, argv7);

  KALDI_ASSERT(i7 == 8);
  KALDI_ASSERT(k7 == 20);
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestParseOptions();
  return 0;
}
