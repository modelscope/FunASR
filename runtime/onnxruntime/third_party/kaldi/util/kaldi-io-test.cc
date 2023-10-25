// util/kaldi-io-test.cc

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
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include "base/io-funcs.h"
#include "util/kaldi-io.h"
#include "base/kaldi-math.h"
#include "base/kaldi-utils.h"

namespace kaldi {



void UnitTestClassifyRxfilename() {
  KALDI_ASSERT(ClassifyRxfilename("") == kStandardInput);
  KALDI_ASSERT(ClassifyRxfilename(" ") == kNoInput);
  KALDI_ASSERT(ClassifyRxfilename(" a ") == kNoInput);
  KALDI_ASSERT(ClassifyRxfilename("a ") == kNoInput);
  KALDI_ASSERT(ClassifyRxfilename("a") == kFileInput);
  KALDI_ASSERT(ClassifyRxfilename("-") == kStandardInput);
  KALDI_ASSERT(ClassifyRxfilename("b|") == kPipeInput);
  KALDI_ASSERT(ClassifyRxfilename("|b") == kNoInput);
  KALDI_ASSERT(ClassifyRxfilename("b c|") == kPipeInput);
  KALDI_ASSERT(ClassifyRxfilename(" b c|") == kPipeInput);
  KALDI_ASSERT(ClassifyRxfilename("a b c:123") == kOffsetFileInput);
  KALDI_ASSERT(ClassifyRxfilename("a b c:3") == kOffsetFileInput);
  KALDI_ASSERT(ClassifyRxfilename("a b c:") == kFileInput);
  KALDI_ASSERT(ClassifyRxfilename("a b c/3") == kFileInput);
  KALDI_ASSERT(ClassifyRxfilename("ark,s,cs:a b c") == kNoInput);
  KALDI_ASSERT(ClassifyRxfilename("scp:a b c") == kNoInput);

}


void UnitTestClassifyWxfilename() {
  KALDI_ASSERT(ClassifyWxfilename("") == kStandardOutput);
  KALDI_ASSERT(ClassifyWxfilename(" ") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename(" a ") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("a ") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("a") == kFileOutput);
  KALDI_ASSERT(ClassifyWxfilename("-") == kStandardOutput);
  KALDI_ASSERT(ClassifyWxfilename("b|") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("|b") == kPipeOutput);
  KALDI_ASSERT(ClassifyWxfilename("| b ") == kPipeOutput);
  KALDI_ASSERT(ClassifyWxfilename("b c|") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("a b c:123") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("ark,s,cs:a b c") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("scp:a b c") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("a b c:3") == kNoOutput);
  KALDI_ASSERT(ClassifyWxfilename("a b c:") == kFileOutput);
  KALDI_ASSERT(ClassifyWxfilename("a b c/3") == kFileOutput);
}

void UnitTestIoNew(bool binary) {
  {
    const char *filename = "tmpf";

    Output ko(filename, binary);
    std::ostream &outfile = ko.Stream();
    if (!binary) outfile << "\t";
    int64 i1 = Rand() % 10000;
    WriteBasicType(outfile, binary, i1);
    uint16 i2 = Rand() % 10000;
    WriteBasicType(outfile, binary, i2);
    if (!binary) outfile << "\t";
    char c = Rand();
    WriteBasicType(outfile, binary, c);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::vector<int32> vec1;
    WriteIntegerVector(outfile, binary, vec1);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::vector<uint16> vec2;
    for (size_t i = 0; i < 10; i++) vec2.push_back(Rand()%100 - 10);
    WriteIntegerVector(outfile, binary, vec2);
    if (!binary) outfile << " \n";
    std::vector<char> vec3;
    for (size_t i = 0; i < 10; i++) vec3.push_back(Rand()%100);
    WriteIntegerVector(outfile, binary, vec3);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    const char *token1 = "Hi";
    WriteToken(outfile, binary, token1);
    if (!binary) outfile << " \n";
    std::string token2 = "There.";
    WriteToken(outfile, binary, token2);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::string token3 = "You.";
    WriteToken(outfile, binary, token3);
    if (!binary && Rand()%2 == 0) outfile << " ";
    float f1 = RandUniform();
    WriteBasicType(outfile, binary, f1);
    if (!binary && Rand()%2 == 0) outfile << "\t";
    float f2 = RandUniform();
    WriteBasicType(outfile, binary, f2);
    double d1 = RandUniform();
    WriteBasicType(outfile, binary, d1);
    if (!binary && Rand()%2 == 0) outfile << "\t";
    double d2 = RandUniform();
    WriteBasicType(outfile, binary, d2);
    if (!binary && Rand()%2 == 0) outfile << "\t";
    ko.Close();

    {
      bool binary_in;
      Input ki(filename, &binary_in);
      std::istream &infile = ki.Stream();
      int64 i1_in;
      ReadBasicType(infile, binary_in, &i1_in);
      KALDI_ASSERT(i1_in == i1);
      uint16 i2_in;
      ReadBasicType(infile, binary_in, &i2_in);
      KALDI_ASSERT(i2_in == i2);
      char c_in;
      ReadBasicType(infile, binary_in, &c_in);
      KALDI_ASSERT(c_in == c);
      std::vector<int32> vec1_in;
      ReadIntegerVector(infile, binary_in, &vec1_in);
      KALDI_ASSERT(vec1_in == vec1);
      std::vector<uint16> vec2_in;
      ReadIntegerVector(infile, binary_in, &vec2_in);
      KALDI_ASSERT(vec2_in == vec2);
      std::vector<char> vec3_in;
      ReadIntegerVector(infile, binary_in, &vec3_in);
      KALDI_ASSERT(vec3_in == vec3);
      std::string  token1_in, token2_in;
      KALDI_ASSERT(Peek(infile, binary_in) == static_cast<int>(*token1));
      ReadToken(infile, binary_in, &token1_in);
      KALDI_ASSERT(token1_in == (std::string)token1);
      ReadToken(infile, binary_in, &token2_in);
      KALDI_ASSERT(token2_in == token2);
      if (Rand() % 2 == 0)
        ExpectToken(infile, binary_in, token3.c_str());
      else
        ExpectToken(infile, binary_in, token3);
      float f1_in;  // same type.
      ReadBasicType(infile, binary_in, &f1_in);
      AssertEqual(f1_in, f1);
      double f2_in;  // wrong type.
      ReadBasicType(infile, binary_in, &f2_in);
      AssertEqual(f2_in, f2);
      double d1_in;  // same type.
      ReadBasicType(infile, binary_in, &d1_in);
      AssertEqual(d1_in, d1);
      float d2_in;  // wrong type.
      ReadBasicType(infile, binary_in, &d2_in);
      AssertEqual(d2_in, d2);
      KALDI_ASSERT(Peek(infile, binary_in) == -1);
    }
    unlink(filename);
  }
}

void UnitTestIoPipe(bool binary) {
  // This is as UnitTestIoNew except with different filenames.
  {
#if defined(_MSC_VER) && !defined(KALDI_CYGWIN_COMPAT)
    // self-invocation on Windows that emulates cat(1)
    const char *filename_out = "|kaldi-io-test cat > tmpf.gz",
        *filename_in = "kaldi-io-test cat tmpf.gz|";
#else
    const char *filename_out = "|gzip -c > tmpf.gz",
        *filename_in = "gunzip -c tmpf.gz |";
#endif

    Output ko(filename_out, binary);
    std::ostream &outfile = ko.Stream();
    if (!binary) outfile << "\t";
    int64 i1 = Rand() % 10000;
    WriteBasicType(outfile, binary, i1);
    uint16 i2 = Rand() % 10000;
    WriteBasicType(outfile, binary, i2);
    if (!binary) outfile << "\t";
    char c = Rand();
    WriteBasicType(outfile, binary, c);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::vector<int32> vec1;
    WriteIntegerVector(outfile, binary, vec1);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::vector<uint16> vec2;
    for (size_t i = 0; i < 10; i++) vec2.push_back(Rand()%100 - 10);
    WriteIntegerVector(outfile, binary, vec2);
    if (!binary) outfile << " \n";
    WriteToken(outfile, binary, "<foo>");
    std::vector<char> vec3;
    for (size_t i = 0; i < 10; i++) vec3.push_back(Rand()%100);
    WriteIntegerVector(outfile, binary, vec3);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    const char *token1 = "Hi";
    WriteToken(outfile, binary, token1);
    if (!binary) outfile << " \n";
    std::string token2 = "There.";
    WriteToken(outfile, binary, token2);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::string token3 = "You.";
    WriteToken(outfile, binary, token3);
    if (!binary && Rand()%2 == 0) outfile << " ";
    float f1 = RandUniform();
    WriteBasicType(outfile, binary, f1);
    if (!binary && Rand()%2 == 0) outfile << "\t";
    float f2 = RandUniform();
    WriteBasicType(outfile, binary, f2);
    double d1 = RandUniform();
    WriteBasicType(outfile, binary, d1);
    if (!binary && Rand()%2 == 0) outfile << "\t";
    double d2 = RandUniform();
    WriteBasicType(outfile, binary, d2);
    if (!binary && Rand()%2 == 0) outfile << "\t";
    bool ans = ko.Close();
    KALDI_ASSERT(ans);
#ifndef _MSC_VER
    Sleep(1);  // This test does not work without this sleep:
    // seems to be some kind of file-system latency.
#endif
    {
      bool binary_in;
      Input ki(filename_in, &binary_in);
      std::istream &infile = ki.Stream();
      int64 i1_in;
      ReadBasicType(infile, binary_in, &i1_in);
      KALDI_ASSERT(i1_in == i1);
      uint16 i2_in;
      ReadBasicType(infile, binary_in, &i2_in);
      KALDI_ASSERT(i2_in == i2);
      char c_in;
      ReadBasicType(infile, binary_in, &c_in);
      KALDI_ASSERT(c_in == c);
      std::vector<int32> vec1_in;
      ReadIntegerVector(infile, binary_in, &vec1_in);
      KALDI_ASSERT(vec1_in == vec1);
      std::vector<uint16> vec2_in;
      ReadIntegerVector(infile, binary_in, &vec2_in);
      KALDI_ASSERT(vec2_in == vec2);
      std::vector<char> vec3_in;
      KALDI_ASSERT(PeekToken(infile, binary_in) == static_cast<int>('f'));
      ExpectToken(infile, binary_in, "<foo>");
      ReadIntegerVector(infile, binary_in, &vec3_in);
      KALDI_ASSERT(vec3_in == vec3);
      std::string  token1_in, token2_in;
      KALDI_ASSERT(Peek(infile, binary_in) == static_cast<int>(*token1));
      ReadToken(infile, binary_in, &token1_in);
      KALDI_ASSERT(token1_in == (std::string)token1);
      ReadToken(infile, binary_in, &token2_in);
      KALDI_ASSERT(token2_in == token2);
      if (Rand() % 2 == 0)
        ExpectToken(infile, binary_in, token3.c_str());
      else
        ExpectToken(infile, binary_in, token3);
      float f1_in;  // same type.
      ReadBasicType(infile, binary_in, &f1_in);
      AssertEqual(f1_in, f1);
      double f2_in;  // wrong type.
      ReadBasicType(infile, binary_in, &f2_in);
      AssertEqual(f2_in, f2);
      double d1_in;  // same type.
      ReadBasicType(infile, binary_in, &d1_in);
      AssertEqual(d1_in, d1);
      float d2_in;  // wrong type.
      ReadBasicType(infile, binary_in, &d2_in);
      AssertEqual(d2_in, d2);
      KALDI_ASSERT(Peek(infile, binary_in) == -1);
    }
  }
  unlink("tmpf.txt");
  unlink("tmpf.gz");
}

void UnitTestIoStandard() {
  /*
    Don't do the the following part because it requires
    to pipe from an empty file, for it to not hang.
  {
    Input inp("", NULL);  // standard input.
    KALDI_ASSERT(inp.Stream().get() == -1);
  }
  {
    Input inp("-", NULL);  // standard input.
    KALDI_ASSERT(inp.Stream().get() == -1);
    }*/

  {
    std::cout << "Should see: foo\n";
    Output out("", false);
    out.Stream() << "foo\n";
  }
  {
    std::cout << "Should see: bar\n";
    Output out("-", false);
    out.Stream() << "bar\n";
  }
}

// This is Windows-specific.
void UnitTestNativeFilename() {
#ifdef KALDI_CYGWIN_COMPAT
  extern std::string MapCygwinPath(const std::string &filename);

  KALDI_ASSERT(MapCygwinPath("") == "");
  KALDI_ASSERT(MapCygwinPath(".") == ".");
  KALDI_ASSERT(MapCygwinPath("..") == "..");
  KALDI_ASSERT(MapCygwinPath("/dev/null")[0] != '/');
  KALDI_ASSERT(MapCygwinPath("/tmp")[1] == ':');
  KALDI_ASSERT(MapCygwinPath("/tmp/")[1] == ':');
  KALDI_ASSERT(MapCygwinPath("/tmp/foo")[1] == ':');
  KALDI_ASSERT(MapCygwinPath("/cygdrive/c") == "c:/");
  KALDI_ASSERT(MapCygwinPath("/cygdrive/c/") == "c:/");
  KALDI_ASSERT(MapCygwinPath("/cygdrive/c/foo") == "c:/foo");
#endif
}

}  // end namespace kaldi.

#if defined(_MSC_VER) && !defined(KALDI_CYGWIN_COMPAT)
// Windows has no cat!  There is probably no suitable tool to test popen I/O on
// Windows, so we emulate a lame version of cat(1).
static int TinyCat(int argc, const char** argv) {
  const char* name_in = argc > 0 && strcmp(argv[0], "-") ? argv[0] : NULL;
  int fd_in = name_in ? _open(name_in, _O_RDONLY) : _fileno(stdin);
  if (fd_in < 0)
    return 1;

  int fd_out = _fileno(stdout);
  _setmode(fd_in, _O_BINARY);
  _setmode(fd_out, _O_BINARY);

  char buffer[100];
  int last_read;
  while ((last_read = _read(fd_in, buffer, sizeof(buffer))) > 0)
    _write(fd_out, buffer, last_read);

  if (name_in) _close(fd_in);
  return 0;
}
#endif

int main(int argc, const char** argv) {
  using namespace kaldi;
#if defined(_MSC_VER) && !defined(KALDI_CYGWIN_COMPAT)
  if (argc > 1 && strcmp(argv[1], "cat") == 0)
    return TinyCat(argc - 2, argv + 2);
#endif

  UnitTestNativeFilename();
  UnitTestIoNew(false);
  UnitTestIoNew(true);
  UnitTestIoPipe(true);
  UnitTestIoPipe(false);
  UnitTestIoStandard();
  UnitTestClassifyRxfilename();
  UnitTestClassifyWxfilename();

  KALDI_ASSERT(1);  // just wanted to check that KALDI_ASSERT does not fail
  // for 1.
  return 0;
}
