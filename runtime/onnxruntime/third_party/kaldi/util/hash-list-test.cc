// util/hash-list-test.cc

// Copyright 2009-2011     Microsoft Corporation
//                2013     Johns Hopkins University (author: Daniel Povey)

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


#include "util/hash-list.h"
#include <map>  // for baseline.
#include <cstdlib>
#include <iostream>

namespace kaldi {

template<class Int, class T> void TestHashList() {
  typedef typename HashList<Int, T>::Elem Elem;

  HashList<Int, T> hash;
  hash.SetSize(200);  // must be called before use.
  std::map<Int, T> m1;
  for (size_t j = 0; j < 50; j++) {
    Int key = Rand() % 200;
    T val = Rand() % 50;
    m1[key] = val;
    Elem *e = hash.Find(key);
    if (e) e->val = val;
    else  hash.Insert(key, val);
  }


  std::map<Int, T> m2;

  for (int i = 0; i < 100; i++) {
    m2.clear();
    for (typename std::map<Int, T>::const_iterator iter = m1.begin();
        iter != m1.end();
        iter++) {
      m2[iter->first + 1] = iter->second;
    }
    std::swap(m1, m2);

    Elem *h = hash.Clear(), *tmp;

    hash.SetSize(100 + Rand() % 100);  // note, SetSize is relatively cheap
    // operation as long as we are not increasing the size more than it's ever
    // previously been increased to.

    for (; h != NULL; h = tmp) {
      hash.Insert(h->key + 1, h->val);
      tmp = h->tail;
      hash.Delete(h);  // think of this like calling delete.
    }

    // Now make sure h and m2 are the same.
    const Elem *list = hash.GetList();
    size_t count = 0;
    for (; list != NULL; list = list->tail, count++) {
      KALDI_ASSERT(m1[list->key] == list->val);
    }

    for (size_t j = 0; j < 10; j++) {
      Int key = Rand() % 200;
      bool found_m1 = (m1.find(key) != m1.end());
      if (found_m1) m1[key];
      Elem *e = hash.Find(key);
      KALDI_ASSERT((e != NULL) == found_m1);
      if (found_m1)
        KALDI_ASSERT(m1[key] == e->val);
    }

    KALDI_ASSERT(m1.size() == count);
  }
}




}  // end namespace kaldi



int main() {
  using namespace kaldi;
  for (size_t i = 0;i < 3;i++) {
    TestHashList<int, unsigned int>();
    TestHashList<unsigned int, int>();
    TestHashList<int16, int32>();
    TestHashList<int16, int32>();
    TestHashList<char, unsigned char>();
    TestHashList<unsigned char, int>();
  }
  std::cout << "Test OK.\n";
}
