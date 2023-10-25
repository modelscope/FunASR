// fstext/lattice-utils-test.cc

// Copyright 2011  Microsoft Corporation

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

#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"

namespace fst {

template<class Weight, class Int> void TestConvert(bool invert) {
  typedef ArcTpl<Weight> Arc;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
  for(int i = 0; i < 5; i++) {
    VectorFst<Arc> *fst = RandFst<Arc>();
    std::cout << "FST before converting to compact-arc is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<CompactArc> ofst;
    ConvertLattice<Weight, Int>(*fst, &ofst, invert);

    std::cout << "FST after converting is:\n";
    {
      FstPrinter<CompactArc> fstprinter(ofst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> origfst;
    ConvertLattice<Weight, Int>(ofst, &origfst, invert);
    std::cout << "FST after back conversion is:\n";
    {
      FstPrinter<Arc> fstprinter(origfst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }

    assert(RandEquivalent(*fst, origfst, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
    delete fst;
  }
}

// This tests the ShortestPath algorithm, and by proxy, tests the
// NaturalLess template etc.

template<class Weight, class Int> void TestShortestPath() {
  for (int p = 0; p < 10; p++) {
    typedef ArcTpl<Weight> Arc;
    typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
    for(int i = 0; i < 5; i++) {
      VectorFst<Arc> *fst = RandPairFst<Arc>();
      std::cout << "Testing shortest path\n";
      std::cout << "FST before converting to compact-arc is:\n";
      {
        FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
      VectorFst<CompactArc> cfst;
      ConvertLattice<Weight, Int>(*fst, &cfst, false); // invert == false


      {
        VectorFst<Arc> nbest_fst_1;
        ShortestPath(*fst, &nbest_fst_1, 1);
        VectorFst<Arc> nbest_fst_2;
        ShortestPath(*fst, &nbest_fst_2, 3);
        VectorFst<Arc> nbest_fst_1b;
        ShortestPath(nbest_fst_2, &nbest_fst_1b, 1);


        assert(ApproxEqual(ShortestDistance(nbest_fst_1),
                           ShortestDistance(nbest_fst_1b)));

        // since semiring is idempotent, this should succeed too.
        assert(ApproxEqual(ShortestDistance(*fst),
                           ShortestDistance(nbest_fst_1b)));
      }
      {
        VectorFst<CompactArc> nbest_fst_1;
        ShortestPath(cfst, &nbest_fst_1, 1);
        VectorFst<CompactArc> nbest_fst_2;
        ShortestPath(cfst, &nbest_fst_2, 3);
        VectorFst<CompactArc> nbest_fst_1b;
        ShortestPath(nbest_fst_2, &nbest_fst_1b, 1);

        assert(ApproxEqual(ShortestDistance(nbest_fst_1),
                           ShortestDistance(nbest_fst_1b)));
        // since semiring is idempotent, this should succeed too.
        assert(ApproxEqual(ShortestDistance(cfst),
                           ShortestDistance(nbest_fst_1b)));
      }

      delete fst;
    }
  }
}



template<class Int> void TestConvert2() {
  typedef ArcTpl<LatticeWeightTpl<float> > ArcF;
  typedef ArcTpl<LatticeWeightTpl<double> > ArcD;
  typedef ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, Int> > CArcF;
  typedef ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<double>, Int> > CArcD;

  for(int i = 0; i < 2; i++) {
    {
      VectorFst<ArcF> *fst1 = RandPairFst<ArcF>();
      VectorFst<ArcD> fst2;
      VectorFst<ArcF> fst3;
      ConvertLattice(*fst1, &fst2);
      ConvertLattice(fst2, &fst3);

      assert(RandEquivalent(*fst1, fst3, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }

    {
      VectorFst<ArcF> *fst1 = RandPairFst<ArcF>();
      VectorFst<CArcF> cfst1, cfst3;
      ConvertLattice(*fst1, &cfst1);
      VectorFst<CArcD> cfst2;
      ConvertLattice(cfst1, &cfst2);
      ConvertLattice(cfst2, &cfst3);
      assert(RandEquivalent(cfst1, cfst3, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }

    {
      VectorFst<ArcF> *fst1 = RandPairFst<ArcF>();
      VectorFst<CArcD> cfst1, cfst3;
      ConvertLattice(*fst1, &cfst1);
      VectorFst<CArcF> cfst2;
      ConvertLattice(cfst1, &cfst2);
      ConvertLattice(cfst2, &cfst3);
      assert(RandEquivalent(cfst1, cfst3, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }

    {
      VectorFst<ArcD> *fst1 = RandPairFst<ArcD>();
      VectorFst<CArcD> cfst1, cfst3;
      ConvertLattice(*fst1, &cfst1);
      VectorFst<CArcF> cfst2;
      ConvertLattice(cfst1, &cfst2);
      ConvertLattice(cfst2, &cfst3);
      assert(RandEquivalent(cfst1, cfst3, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }

    {
      VectorFst<ArcD> *fst1 = RandPairFst<ArcD>();
      VectorFst<CArcF> cfst1;
      ConvertLattice(*fst1, &cfst1);
      VectorFst<ArcD> fst2;
      ConvertLattice(cfst1, &fst2);
      assert(RandEquivalent(*fst1, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }

    {
      VectorFst<ArcF> *fst1 = RandPairFst<ArcF>();
      VectorFst<CArcD> cfst1;
      ConvertLattice(*fst1, &cfst1);
      VectorFst<ArcF> fst2;
      ConvertLattice(cfst1, &fst2);
      assert(RandEquivalent(*fst1, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }

    {
      VectorFst<ArcD> *fst1 = RandPairFst<ArcD>();
      VectorFst<CArcF> cfst1;
      ConvertLattice(*fst1, &cfst1);
      VectorFst<ArcD> fst2;
      ConvertLattice(cfst1, &fst2);
      assert(RandEquivalent(*fst1, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
      delete fst1;
    }
  }
}


// use TestConvertPair when the Weight can be constructed from
// a pair of floats.
template<class Weight, class Int> void TestConvertPair(bool invert) {
  typedef ArcTpl<Weight> Arc;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
  for(int i = 0; i < 2; i++) {
    VectorFst<Arc> *fst = RandPairFst<Arc>();
    /*std::cout << "FST before converting to compact-arc is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
      }*/
    VectorFst<CompactArc> ofst;
    ConvertLattice<Weight, Int>(*fst, &ofst, invert);

    /*std::cout << "FST after converting is:\n";
    {
      FstPrinter<CompactArc> fstprinter(ofst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
      }*/
    VectorFst<Arc> origfst;
    ConvertLattice<Weight, Int>(ofst, &origfst, invert);
    /*std::cout << "FST after back conversion is:\n";
    {
      FstPrinter<Arc> fstprinter(origfst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
      }*/

    assert(RandEquivalent(*fst, origfst, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
    delete fst;
  }
}


// use TestConvertPair when the Weight can be constructed from
// a pair of floats.
template<class Weight, class Int> void TestScalePair(bool invert) {
  std::vector<std::vector<double> > scale1 = DefaultLatticeScale(),
      scale2 = DefaultLatticeScale();
  // important that all these numbers exactly representable as floats..
  // exact floating-point comparisons are used in LatticeWeight, and
  // this exactness is being tested here.. this test will fail for
  // other types of number.
  if (kaldi::Rand() % 4 == 0) {
    scale1[0][0] = 2.0;
    scale2[0][0] = 0.5;
    scale1[1][1] = 4.0;
    scale2[1][1] = 0.25;
  } else if (kaldi::Rand() % 3 == 0) {
    // use that [1 0.25; 0 1] [ 1 -0.25; 0 1] is the unit matrix.
    scale1[0][1] = 0.25;
    scale2[0][1] = -0.25;
  } else if (kaldi::Rand() % 2 == 0) {
    scale1[1][0] = 0.25;
    scale2[1][0] = -0.25;
  }


  typedef ArcTpl<Weight> Arc;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;
  for(int i = 0; i < 2; i++) {
    VectorFst<Arc> *fst = RandPairFst<Arc>();
    /*std::cout << "FST before converting to compact-arc is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
      }*/
    VectorFst<CompactArc> ofst;
    ConvertLattice<Weight, Int>(*fst, &ofst, invert);
    ScaleLattice(scale1, &ofst);
    /*std::cout << "FST after converting and scaling is:\n";
    {
      FstPrinter<CompactArc> fstprinter(ofst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
      }*/
    VectorFst<Arc> origfst;
    ConvertLattice<Weight, Int>(ofst, &origfst, invert);
    ScaleLattice(scale2, &origfst);
    /*std::cout << "FST after back conversion and scaling is:\n";
    {
      FstPrinter<Arc> fstprinter(origfst, NULL, NULL, NULL, false, true);
      fstprinter.Print(&std::cout, "standard output");
      }*/
    // If RandEquivalent doesn't work, it could be due to a nasty issue related to the use
    // of exact floating-point comparisons in the Plus function of LatticeWeight.
    if (!RandEquivalent(*fst, origfst, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/)) {
      std::cerr << "Warn, randequivalent returned false.  Checking equivalence another way.\n";
      assert(Equal(*fst, origfst));
    }
    delete fst;
  }
}



} // end namespace fst

int main() {
  using namespace fst;

  typedef ::int64 int64;
  typedef ::uint64 uint64;
  typedef ::int32 int32;
  typedef ::uint32 uint32;

  {
    typedef LatticeWeightTpl<float> LatticeWeight;
    for(int i = 0; i < 2; i++) {
      bool invert = (i % 2);
      TestConvert<TropicalWeight, int32>(invert);
      TestConvertPair<LatticeWeight, int32>(invert);
      TestConvertPair<LatticeWeight, size_t>(invert);
      TestConvertPair<LexicographicWeight<TropicalWeight, TropicalWeight>, size_t>(invert);
      TestScalePair<LatticeWeight, int32>(invert);
      TestScalePair<LatticeWeight, size_t>(invert);
      TestScalePair<LexicographicWeight<TropicalWeight, TropicalWeight>, size_t>(invert);
    }
  }
  {
    typedef LatticeWeightTpl<double> LatticeWeight;
    TestShortestPath<LatticeWeight, int32>();
    TestConvert2<int32>();
    for(int i = 0; i < 2; i++) {
      bool invert = (i % 2);
      TestConvertPair<LatticeWeight, int32>(invert);
      TestConvertPair<LatticeWeight, size_t>(invert);
      TestConvertPair<LexicographicWeight<TropicalWeight, TropicalWeight>, size_t>(invert);
      TestScalePair<LatticeWeight, int32>(invert);
      TestScalePair<LatticeWeight, size_t>(invert);
      TestScalePair<LexicographicWeight<TropicalWeight, TropicalWeight>, size_t>(invert);
    }
  }
  std::cout << "Tests succeeded\n";
}
