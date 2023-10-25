// fstext/epsilon-property-test.cc

// Copyright 2014    Johns Hopkins University (Author: Daniel Povey)

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


#include "fstext/rand-fst.h"
#include "fstext/epsilon-property.h"


namespace fst {

void TestEnsureEpsilonProperty() {
  
  for (int32 i = 0; i < 10; i++) {
    RandFstOptions opts;
    opts.acyclic = true;
    VectorFst<LogArc> *fst = RandFst<LogArc>(opts);
    VectorFst<LogArc> fst2(*fst); // copy it...
    EnsureEpsilonProperty(&fst2);

    std::vector<char> info;
    ComputeStateInfo(fst2, &info);
    for (size_t i = 0; i < info.size(); i++) {
      char c = info[i];
      assert(!((c & kStateHasEpsilonArcsEntering) != 0 &&
               (c & kStateHasNonEpsilonArcsEntering) != 0));
      assert(!((c & kStateHasEpsilonArcsLeaving) != 0 &&
               (c & kStateHasNonEpsilonArcsLeaving) != 0));
    }
    assert(RandEquivalent(fst2, *fst, 5, 0.01, kaldi::Rand(), 10));    
    delete fst;
  }
}

} // end namespace fst

int main() {
  using namespace fst;
  for (int i = 0; i < 2; i++) {
    TestEnsureEpsilonProperty();
  }
  std::cout << "Test OK\n";
}
