// base/io-funcs-test.cc

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
#include "base/io-funcs.h"
#include "base/kaldi-math.h"

namespace kaldi {

void UnitTestIo(bool binary) {
  {
    const char *filename = "tmpf";
    std::ofstream outfile(filename, std::ios_base::out | std::ios_base::binary);
    InitKaldiOutputStream(outfile, binary);
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

    int32 size = RandInt(0, 10);
    for (size_t i = 0; i < size; i++) vec3.push_back(Rand()%100);
    WriteIntegerVector(outfile, binary, vec3);
    std::vector<std::pair<int32, int32> > vec4;
    WriteIntegerPairVector(outfile, binary, vec4);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    std::vector<std::pair<uint16, uint16> > vec5;
    for (size_t i = 0; i < size; i++) vec5.push_back(std::make_pair<uint16, uint16>(Rand()%100 - 10, Rand()%100 - 10));
    WriteIntegerPairVector(outfile, binary, vec5);
    if (!binary) outfile << " \n";
    std::vector<std::pair<char, char> > vec6;
    for (size_t i = 0; i < size; i++) vec6.push_back(std::make_pair<char, char>(Rand()%100, Rand()%100));
    WriteIntegerPairVector(outfile, binary, vec6);
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
    outfile.close();

    {
      std::ifstream infile(filename, std::ios_base::in | std::ios_base::binary);
      bool binary_in;
      InitKaldiInputStream(infile, &binary_in);
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
      std::vector<std::pair<int32, int32> > vec4_in;
      ReadIntegerPairVector(infile, binary_in, &vec4_in);
      KALDI_ASSERT(vec4_in == vec4);
      std::vector<std::pair<uint16, uint16> > vec5_in;
      ReadIntegerPairVector(infile, binary_in, &vec5_in);
      KALDI_ASSERT(vec5_in == vec5);
      std::vector<std::pair<char, char> > vec6_in;
      ReadIntegerPairVector(infile, binary_in, &vec6_in);
      KALDI_ASSERT(vec6_in == vec6);
      std::string  token1_in, token2_in;
      KALDI_ASSERT(Peek(infile, binary_in) == static_cast<int>(*token1));
      KALDI_ASSERT(PeekToken(infile, binary_in) == static_cast<int>(*token1));
      // Note:
      // the stuff with skipping over '<' is tested in ../util/kaldi-io-test.cc,
      // since we need to make sure it works with pipes.
      ReadToken(infile, binary_in, &token1_in);
      KALDI_ASSERT(token1_in == std::string(token1));
      ReadToken(infile, binary_in, &token2_in);
      KALDI_ASSERT(token2_in == std::string(token2));
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
      KALDI_ASSERT(PeekToken(infile, binary_in) == -1);
    }
    unlink(filename);
  }
}



}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  for (size_t i = 0; i < 10; i++) {
    UnitTestIo(false);
    UnitTestIo(true);
  }
  KALDI_ASSERT(1);  // just to check that KALDI_ASSERT does not fail for 1.
  return 0;
}

