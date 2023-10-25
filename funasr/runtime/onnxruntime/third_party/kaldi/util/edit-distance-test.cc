// util/edit-distance-test.cc

// Copyright 2009-2011     Microsoft Corporation;  Haihua Xu

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


#include "base/kaldi-common.h"
#include "util/edit-distance.h"

namespace kaldi {

void TestEditDistance() {
  std::vector<int32> a;
  std::vector<int32> b;
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 0);

  a.push_back(1);
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 1);

  b.push_back(1);
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 0);

  b.push_back(2);
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 1);

  a.push_back(2);
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 0);

  a.push_back(3);
  a.push_back(4);
  b.push_back(4);

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 1);

  a.push_back(5);

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 2);

  b.push_back(6);

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 2);

  a.push_back(1);
  b.push_back(1);

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 2);

  b.push_back(10);

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 3);
}


void TestEditDistanceString() {
  std::vector<std::string> a;
  std::vector<std::string> b;
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 0);

  a.push_back("1");
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 1);

  b.push_back("1");
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 0);

  b.push_back("2");
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 1);

  a.push_back("2");
  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 0);

  a.push_back("3");
  a.push_back("4");
  b.push_back("4");

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 1);

  a.push_back("5");

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 2);

  b.push_back("6");

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 2);

  a.push_back("1");
  b.push_back("1");

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 2);

  b.push_back("10");

  KALDI_ASSERT(LevenshteinEditDistance(a, b) == 3);
}



// edit distance calculate
void TestEditDistance2() {
  std::vector<int32>  hyp;
  std::vector<int32>  ref;
  int32 ins, del, sub, total_cost;
  // initialize hypothesis
  hyp.push_back(1);
  hyp.push_back(3);
  hyp.push_back(4);
  hyp.push_back(5);
  // initialize reference
  ref.push_back(2);
  ref.push_back(3);
  ref.push_back(4);
  ref.push_back(5);
  ref.push_back(6);
  ref.push_back(7);
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 3  && ins == 0 && del == 2 && sub == 1);

  std::swap(hyp, ref);
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 3 && ins == 2 && del == 0 && sub == 1);

  hyp.clear();
  ref.clear();
  hyp.push_back(1);
  ref.push_back(1);
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 0 && ins+del+sub == 0);
  hyp.push_back(2);
  ref.push_back(3);
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 1 && ins == 0 && del == 0 && sub == 1);
  // randomized test
  size_t num = 0;
  for (; num < 1000; num ++) {
    int32  hyp_len = Rand()%11;
    int32  ref_len = Rand()%3;
    hyp.resize(hyp_len);
    ref.resize(ref_len);

    int32 index = 0;
    for (; index < hyp_len; index ++)
      hyp[index] = Rand()%4;
    for (index = 0; index < ref_len; index ++)
      ref[index] = Rand()%4;
    // current version
    total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
    // previous version
    int32 total_cost2 = LevenshteinEditDistance(hyp, ref);
    // verify both are the same
    KALDI_ASSERT(total_cost == total_cost2);
    KALDI_ASSERT(ins+del+sub == total_cost);
    KALDI_ASSERT(del-ins == static_cast<int32>(ref.size() -hyp.size()));
  }
  return;
}


// edit distance calculate
void TestEditDistance2String() {
  std::vector<std::string>  hyp;
  std::vector<std::string>  ref;
  int32 ins, del, sub, total_cost;
  // initialize hypothesis
  hyp.push_back("1");
  hyp.push_back("3");
  hyp.push_back("4");
  hyp.push_back("5");
  // initialize reference
  ref.push_back("2");
  ref.push_back("3");
  ref.push_back("4");
  ref.push_back("5");
  ref.push_back("6");
  ref.push_back("7");
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 3  && ins == 0 && del == 2 && sub == 1);

  std::swap(hyp, ref);
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 3 && ins == 2 && del == 0 && sub == 1);

  hyp.clear();
  ref.clear();
  hyp.push_back("1");
  ref.push_back("1");
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 0 && ins+del+sub == 0);
  hyp.push_back("2");
  ref.push_back("3");
  total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
  KALDI_ASSERT(total_cost == 1 && ins == 0 && del == 0 && sub == 1);
  // randomized test
  size_t num = 0;
  for (; num < 1000; num ++) {
    int32  hyp_len = Rand()%11;
    int32  ref_len = Rand()%3;
    hyp.resize(hyp_len);
    ref.resize(ref_len);

    int32 index = 0;
    for (; index < hyp_len; index ++)
      hyp[index] = Rand()%4;
    for (index = 0; index < ref_len; index ++)
      ref[index] = Rand()%4;
    // current version
    total_cost = LevenshteinEditDistance(ref, hyp, &ins, &del, &sub);
    // previous version
    int32 total_cost2 = LevenshteinEditDistance(hyp, ref);
    // verify both are the same
    KALDI_ASSERT(total_cost == total_cost2);
    KALDI_ASSERT(ins+del+sub == total_cost);
    KALDI_ASSERT(del-ins == static_cast<int32>(ref.size() -hyp.size()));
  }
  return;
}


void TestLevenshteinAlignment() {
  for (size_t i = 0; i < 100; i++) {
    size_t a_sz = Rand() % 5, b_sz = Rand() % 5;
    std::vector<int32> a, b;
    for (size_t j = 0; j < a_sz; j++) a.push_back(Rand() % 10);
    for (size_t j = 0; j < b_sz; j++) b.push_back(Rand() % 10);
    int32 eps_sym = -1;
    std::vector<std::pair<int32, int32> > ans;

    int32 e1 = LevenshteinEditDistance(a, b),
        e2 = LevenshteinAlignment(a, b, eps_sym, &ans);
    KALDI_ASSERT(e1 == e2);

    std::vector<int32> a2, b2;
    for (size_t i = 0; i < ans.size(); i++) {
      if (ans[i].first != eps_sym) a2.push_back(ans[i].first);
      if (ans[i].second != eps_sym) b2.push_back(ans[i].second);
    }
    KALDI_ASSERT(a == a2);
    KALDI_ASSERT(b == b2);
  }
}

}  // end namespace kaldi

int main() {
  using namespace kaldi;
  TestEditDistance();
  TestEditDistanceString();
  TestEditDistance2();
  TestEditDistance2String();
  TestLevenshteinAlignment();
  std::cout << "Test OK\n";
}



