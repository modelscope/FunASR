// fstext/lattice-weight-test.cc

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
#include "base/kaldi-math.h"
#include "fstext/lattice-weight.h"

namespace fst {
using std::vector;
using std::cout;
// these typedefs are the same as in ../lat/kaldi-lattice.h, but
// just used here for testing (doesn't matter if they get out of
// sync).
typedef float BaseFloat;

typedef LatticeWeightTpl<BaseFloat> LatticeWeight;

typedef CompactLatticeWeightTpl<LatticeWeight, int32> CompactLatticeWeight;

typedef CompactLatticeWeightCommonDivisorTpl<LatticeWeight, int32>
  CompactLatticeWeightCommonDivisor;


LatticeWeight RandomLatticeWeight() {
  int tmp = kaldi::Rand() % 4;
  if (tmp == 0) {
    return LatticeWeight::Zero();
  } else if (tmp == 1) {
    return LatticeWeight( 1, 2);  // sometimes return special values..
  } else if (tmp == 2) {
    return LatticeWeight( 2, 1);  // this tests more thoroughly certain properties...
  } else {
    return LatticeWeight( 100 * kaldi::RandGauss(), 100 * kaldi::RandGauss());
  }
}

CompactLatticeWeight RandomCompactLatticeWeight() {
  LatticeWeight w = RandomLatticeWeight();
  if (w == LatticeWeight::Zero()) {
    return CompactLatticeWeight(w, vector<int32>());
  } else {
    int32 len = kaldi::Rand() % 4;
    vector<int32> str;
    for(int32 i = 0; i < len; i++)
      str.push_back(kaldi::Rand() % 10 + 1);
    return CompactLatticeWeight(w, str);
  }
}

void LatticeWeightTest() {
  for(int32 i = 0; i < 100; i++) {
    LatticeWeight l1 = RandomLatticeWeight(), l2 = RandomLatticeWeight();
    LatticeWeight l3 = Plus(l1, l2);
    LatticeWeight l4 = Times(l1, l2);
    BaseFloat f1 = l1.Value1() + l1.Value2(), f2 = l2.Value1() + l2.Value2(), f3 = l3.Value1() + l3.Value2(),
        f4 = l4.Value1() + l4.Value2();
    kaldi::AssertEqual(std::min(f1, f2), f3);
    kaldi::AssertEqual(f1 + f2, f4);

    KALDI_ASSERT(Plus(l3, l3) == l3);
    KALDI_ASSERT(Plus(l1, l2) == Plus(l2, l1)); // commutativity of plus
    KALDI_ASSERT(Times(l1, l2) == Times(l2, l1)); // commutativity of Times (true for this semiring, not always)
    KALDI_ASSERT(Plus(l3, LatticeWeight::Zero()) == l3); // x + 0 = x
    KALDI_ASSERT(Times(l3, LatticeWeight::One()) == l3); // x * 1 = x
    KALDI_ASSERT(Times(l3, LatticeWeight::Zero()) == LatticeWeight::Zero()); // x * 0 = 0

    KALDI_ASSERT(l3.Reverse().Reverse() == l3);

    NaturalLess<LatticeWeight> nl;
    bool a = nl(l1, l2);
    bool b = (Plus(l1, l2) == l1 && l1 != l2);
    KALDI_ASSERT(a == b);

    KALDI_ASSERT(Compare(l1, Plus(l1, l2)) != 1); // so do not have l1 > l1 + l2
    LatticeWeight l5 = RandomLatticeWeight(), l6 = RandomLatticeWeight();
    {
      LatticeWeight wa = Times(Plus(l1, l2), Plus(l5, l6)),
          wb =  Plus(Times(l1, l5), Plus(Times(l1, l6),
                                        Plus(Times(l2, l5), Times(l2, l6))));
      if (!ApproxEqual(wa, wb)) {
        std::cout << "l1 = " << l1 << ", l2 = " << l2
                  << ", l5 = " << l5 << ", l6 = " << l6 << "\n";
        std::cout << "ERROR: " << wa << " != " <<  wb << "\n";
      }
      // KALDI_ASSERT(Times(Plus(l1, l2), Plus(l5, l6))
      // == Plus(Times(l1, l5), Plus(Times(l1,l6),
      // Plus(Times(l2, l5), Times(l2, l6))))); // * distributes over +
    }
    KALDI_ASSERT(l1.Member() && l2.Member() && l3.Member() && l4.Member()
                 && l5.Member() && l6.Member());
    if (l2 != LatticeWeight::Zero())
      KALDI_ASSERT(ApproxEqual(Divide(Times(l1, l2), l2), l1)); // (a*b) / b = a if b != 0
    KALDI_ASSERT(ApproxEqual(l1, l1.Quantize()));

    std::ostringstream s1;
    s1 << l1;
    std::istringstream s2(s1.str());
    s2 >> l2;
    KALDI_ASSERT(ApproxEqual(l1, l2, 0.001));
    std::cout << s1.str() << '\n';
    {
      std::ostringstream s1b;
      l1.Write(s1b);
      std::istringstream s2b(s1b.str());
      l3.Read(s2b);
      KALDI_ASSERT(l1 == l3);
    }
  }
}


void CompactLatticeWeightTest() {
  for(int32 i = 0; i < 100; i++) {
    CompactLatticeWeight l1 = RandomCompactLatticeWeight(), l2 = RandomCompactLatticeWeight();
    CompactLatticeWeight l3 = Plus(l1, l2);
    CompactLatticeWeight l4 = Times(l1, l2);

    KALDI_ASSERT(Plus(l3, l3) == l3);
    KALDI_ASSERT(Plus(l1, l2) == Plus(l2, l1)); // commutativity of plus
    KALDI_ASSERT(Plus(l3, CompactLatticeWeight::Zero()) == l3); // x + 0 = x
    KALDI_ASSERT(Times(l3, CompactLatticeWeight::One()) == l3); // x * 1 = x
    KALDI_ASSERT(Times(l3, CompactLatticeWeight::Zero()) == CompactLatticeWeight::Zero()); // x * 0 = 0
    NaturalLess<CompactLatticeWeight> nl;
    bool a = nl(l1, l2);
    bool b = (Plus(l1, l2) == l1 && l1 != l2);
    KALDI_ASSERT(a == b);

    KALDI_ASSERT(Compare(l1, Plus(l1, l2)) != 1); // so do not have l1 > l1 + l2
    CompactLatticeWeight l5 = RandomCompactLatticeWeight(), l6 = RandomCompactLatticeWeight();
    KALDI_ASSERT(Times(Plus(l1, l2), Plus(l5, l6)) ==
                 Plus(Times(l1, l5), Plus(Times(l1, l6),
                 Plus(Times(l2, l5), Times(l2, l6))))); // * distributes over +
    KALDI_ASSERT(l1.Member() && l2.Member() && l3.Member() && l4.Member()
                 && l5.Member() && l6.Member());
    if (l2 != CompactLatticeWeight::Zero())  {
      KALDI_ASSERT(ApproxEqual(Divide(Times(l1, l2), l2, DIVIDE_RIGHT), l1)); // (a*b) / b = a if b != 0
      KALDI_ASSERT(ApproxEqual(Divide(Times(l2, l1), l2, DIVIDE_LEFT), l1)); // (a*b) / b = a if b != 0
    }
    KALDI_ASSERT(ApproxEqual(l1, l1.Quantize()));

    std::ostringstream s1;
    s1 << l1;
    std::istringstream s2(s1.str());
    s2 >> l2;
    KALDI_ASSERT(ApproxEqual(l1, l2));
    std::cout << s1.str() << '\n';

    {
      std::ostringstream s1b;
      l1.Write(s1b);
      std::istringstream s2b(s1b.str());
      l3.Read(s2b);
      KALDI_ASSERT(l1 == l3);
    }

    CompactLatticeWeightCommonDivisor divisor;
    std::cout << "l5 = " << l5 << '\n';
    std::cout << "l6 = " << l6 << '\n';
    l1 = divisor(l5, l6);
    std::cout << "div = " << l1 << '\n';
    if (l1 != CompactLatticeWeight::Zero()) {
      l2 = Divide(l5, l1, DIVIDE_LEFT);
      l3 = Divide(l6, l1, DIVIDE_LEFT);
      std::cout << "l2 = " << l2 << '\n';
      std::cout << "l3 = " << l3 << '\n';
      l4 = divisor(l2, l3); // make sure l2 is now one.
      std::cout << "l4 = " << l4 << '\n';
      KALDI_ASSERT(ApproxEqual(l4, CompactLatticeWeight::One()));
    } else {
      KALDI_ASSERT(l5 == CompactLatticeWeight::Zero()
                   && l6 == CompactLatticeWeight::Zero());
    }
  }
}


}

int main() {
  fst::LatticeWeightTest();
  fst::CompactLatticeWeightTest();
}

