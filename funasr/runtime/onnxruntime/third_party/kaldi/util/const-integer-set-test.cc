// util/const-integer-set-test.cc

// Copyright 2009-2011     Microsoft Corporation

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


#include "util/const-integer-set.h"
#include <set>  // for baseline.
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "util/kaldi-io.h"

namespace kaldi {

template<class Int> void TestSetOfNumbers(bool binary) {
  std::set<Int> baseline_set;
  size_t n_in_set = (Rand() % 3) * 50 + (Rand() % 4);  // may be less than this.
  size_t max = (Int) (Rand() % 100) + 1;
  for (size_t i = 0; i < n_in_set; i++) {
    Int to_add((Int) (Rand() % max));
    baseline_set.insert(to_add);
  }

  std::vector<Int> vector_set;
  for (typename std::set<Int>::iterator iter = baseline_set.begin();
       iter!= baseline_set.end();iter++)
    vector_set.push_back(*iter);
  if (vector_set.size() != 0) {
    for (size_t i = 0;i < 10;i++)  // randomize order.
      std::swap(vector_set[Rand()%vector_set.size()],
                vector_set[Rand()%vector_set.size()]);
  }

  ConstIntegerSet<Int> my_set1(baseline_set);

  ConstIntegerSet<Int> my_set2(vector_set);

  ConstIntegerSet<Int> my_set3;
  my_set3.Init(baseline_set);

  ConstIntegerSet<Int> my_set4;
  my_set4.Init(vector_set);

  {
    my_set4.Write(Output("tmpf", binary).Stream(), binary);
  }

  ConstIntegerSet<Int> my_set5;
  {
    bool binary_in;
    Input ki("tmpf", &binary_in);
    my_set5.Read(ki.Stream(), binary_in);
  }


  // if (enable_iterators) {
  size_t sz = baseline_set.size(), sz1 = my_set1.size(), sz2 = my_set2.size(),
      sz3 = my_set3.size(), sz4 = my_set4.size(), sz5 = my_set5.size();
  KALDI_ASSERT(sz == sz1 && sz == sz2 && sz == sz3 && sz == sz4 && sz == sz5);
  // }
  for (size_t i = 0;i < 100;i++) {
    Int some_int;
    if (i%2 == 0 && vector_set.size() != 0)
      some_int = vector_set[Rand()%vector_set.size()];
    else
      some_int = Rand() % max;
    bool in_baseline = (baseline_set.count(some_int) != 0);
    bool in_my_set1 = (my_set1.count(some_int) != 0);
    bool in_my_set2 = (my_set2.count(some_int) != 0);
    bool in_my_set3 = (my_set3.count(some_int) != 0);
    bool in_my_set4 = (my_set4.count(some_int) != 0);
    bool in_my_set5 = (my_set5.count(some_int) != 0);

    if (in_baseline) {
      KALDI_ASSERT(in_my_set1&&in_my_set2&&in_my_set3&&in_my_set4&&in_my_set5);
    } else {
      KALDI_ASSERT(!in_my_set1&&!in_my_set2&&!in_my_set3&&!in_my_set4&&
                   !in_my_set5);
    }
  }

  //  if (enable_iterators) {
  typename std::set<Int>::iterator baseline_iter = baseline_set.begin();
  typename ConstIntegerSet<Int>::iterator my_iter1 = my_set1.begin();
  typename ConstIntegerSet<Int>::iterator my_iter2 = my_set2.begin();
  typename ConstIntegerSet<Int>::iterator my_iter3 = my_set3.begin();
  typename ConstIntegerSet<Int>::iterator my_iter4 = my_set4.begin();
  typename ConstIntegerSet<Int>::iterator my_iter5 = my_set5.begin();
  while (baseline_iter != baseline_set.end()) {
    KALDI_ASSERT(my_iter1 != my_set1.end());
    KALDI_ASSERT(my_iter2 != my_set2.end());
    KALDI_ASSERT(my_iter3 != my_set3.end());
    KALDI_ASSERT(my_iter4 != my_set4.end());
    KALDI_ASSERT(my_iter5 != my_set5.end());
    KALDI_ASSERT(*baseline_iter == *my_iter1);
    KALDI_ASSERT(*baseline_iter == *my_iter2);
    KALDI_ASSERT(*baseline_iter == *my_iter3);
    KALDI_ASSERT(*baseline_iter == *my_iter4);
    KALDI_ASSERT(*baseline_iter == *my_iter5);
    baseline_iter++;
    my_iter1++;
    my_iter2++;
    my_iter3++;
    my_iter4++;
    my_iter5++;
  }
  KALDI_ASSERT(my_iter1 == my_set1.end());
  KALDI_ASSERT(my_iter2 == my_set2.end());
  KALDI_ASSERT(my_iter3 == my_set3.end());
  KALDI_ASSERT(my_iter4 == my_set4.end());
  KALDI_ASSERT(my_iter5 == my_set5.end());
  // }
}

}  // end namespace kaldi



int main() {
  using namespace kaldi;
  for (size_t i = 0;i < 10;i++) {
    TestSetOfNumbers<int>(Rand()%2);
    TestSetOfNumbers<unsigned int>(Rand()%2);
    TestSetOfNumbers<int16>(Rand()%2);
    TestSetOfNumbers<int16>(Rand()%2);
    TestSetOfNumbers<char>(Rand()%2);
    TestSetOfNumbers<unsigned char>(Rand()%2);
  }
  std::cout << "Test OK.\n";
}
