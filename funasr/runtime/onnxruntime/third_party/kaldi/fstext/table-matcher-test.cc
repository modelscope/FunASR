// fstext/table-matcher-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "fstext/table-matcher.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"

namespace fst{


// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestTableMatcher(bool connect, bool left) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;


  VectorFst<Arc> *fst1 = RandFst<Arc>();

  VectorFst<Arc> *fst2 = RandFst<Arc>();

  ILabelCompare<Arc> ilabel_comp;
  OLabelCompare<Arc> olabel_comp;

  TableComposeOptions opts;
  if (left) opts.table_match_type = MATCH_OUTPUT;
  else opts.table_match_type = MATCH_INPUT;
  opts.min_table_size = 1 + kaldi::Rand() % 5;
  opts.table_ratio = 0.25 * (kaldi::Rand() % 5);
  opts.connect = connect;

  ArcSort(fst1, olabel_comp);
  ArcSort(fst2, ilabel_comp);

  VectorFst<Arc> composed;

  TableCompose(*fst1, *fst2, &composed, opts);

  if (!connect) Connect(&composed);

  VectorFst<Arc> composed_baseline;

  Compose(*fst1, *fst2, &composed_baseline);


  std::cout << "Connect = "<< (connect?"True\n":"False\n");

  std::cout <<"Table-Composed FST\n";
  {
    FstPrinter<Arc> fstprinter(composed, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  std::cout <<" Baseline-Composed FST\n";
  {
    FstPrinter<Arc> fstprinter(composed_baseline, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  if ( !RandEquivalent(composed, composed_baseline, 3/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 20/*path length-- max?*/)) {
    VectorFst<Arc> diff1;
    Difference(composed, composed_baseline, &diff1);
    std::cout <<" Diff1 (composed - baseline) \n";
    {
      FstPrinter<Arc> fstprinter(diff1, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }


    VectorFst<Arc> diff2;
    Difference(composed_baseline, composed, &diff2);
    std::cout <<" Diff2 (baseline - composed) \n";
    {
      FstPrinter<Arc> fstprinter(diff2, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }

    assert(0);
  }

  delete fst1;
  delete fst2;
}



// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestTableMatcherCacheLeft(bool connect) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;


  VectorFst<Arc> *fst1 = RandFst<Arc>();


  TableComposeOptions opts;
  opts.table_match_type = MATCH_OUTPUT;
  opts.min_table_size = 1 + kaldi::Rand() % 5;
  opts.table_ratio = 0.25 * (kaldi::Rand() % 5);
  opts.connect = connect;

  TableComposeCache<Fst<Arc> > cache(opts);

  for (size_t i = 0; i < 3; i++) {

    VectorFst<Arc> *fst2 = RandFst<Arc>();

    ILabelCompare<Arc> ilabel_comp;
    OLabelCompare<Arc> olabel_comp;


    ArcSort(fst1, olabel_comp);
    ArcSort(fst2, ilabel_comp);

    VectorFst<Arc> composed;

    TableCompose(*fst1, *fst2, &composed, &cache);

    if (!connect) Connect(&composed);

    VectorFst<Arc> composed_baseline;

    Compose(*fst1, *fst2, &composed_baseline);


    std::cout << "Connect = "<< (connect?"True\n":"False\n");


    if ( !RandEquivalent(composed, composed_baseline, 3/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/)) {
      VectorFst<Arc> diff1;
      Difference(composed, composed_baseline, &diff1);
      std::cout <<" Diff1 (composed - baseline) \n";
      {
        FstPrinter<Arc> fstprinter(diff1, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }


      VectorFst<Arc> diff2;
      Difference(composed_baseline, composed, &diff2);
      std::cout <<" Diff2 (baseline - composed) \n";
      {
        FstPrinter<Arc> fstprinter(diff2, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }

      assert(0);
    }
    delete fst2;
  }

  delete fst1;
}


template<class Arc>  void TestTableMatcherCacheRight(bool connect) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;


  VectorFst<Arc> *fst2 = RandFst<Arc>();
  ILabelCompare<Arc> ilabel_comp;
  ArcSort(fst2, ilabel_comp);


  TableComposeOptions opts;
  opts.table_match_type = MATCH_INPUT;
  opts.min_table_size = 1 + kaldi::Rand() % 5;
  opts.table_ratio = 0.25 * (kaldi::Rand() % 5);
  opts.connect = connect;

  TableComposeCache<Fst<Arc> > cache(opts);

  for (size_t i = 0; i < 2; i++) {

    VectorFst<Arc> *fst1 = RandFst<Arc>();


    OLabelCompare<Arc> olabel_comp;


    ArcSort(fst1, olabel_comp);

    VectorFst<Arc> composed;

    TableCompose(*fst1, *fst2, &composed, &cache);

    if (!connect) Connect(&composed);

    VectorFst<Arc> composed_baseline;

    Compose(*fst1, *fst2, &composed_baseline);


    std::cout << "Connect = "<< (connect?"True\n":"False\n");


    if ( !RandEquivalent(composed, composed_baseline, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 20/*path length-- max?*/)) {
      VectorFst<Arc> diff1;
      Difference(composed, composed_baseline, &diff1);
      std::cout <<" Diff1 (composed - baseline) \n";
      {
        FstPrinter<Arc> fstprinter(diff1, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }


      VectorFst<Arc> diff2;
      Difference(composed_baseline, composed, &diff2);
      std::cout <<" Diff2 (baseline - composed) \n";
      {
        FstPrinter<Arc> fstprinter(diff2, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }

      assert(0);
    }
    delete fst1;
  }

  delete fst2;
}


} // namespace fst

int main() {
  using namespace fst;
  for (int i = 0;i < 1;i++) {
    TestTableMatcher<fst::StdArc>(true, true);
    TestTableMatcher<fst::StdArc>(false, true);
    TestTableMatcher<fst::StdArc>(true, false);
    TestTableMatcher<fst::StdArc>(false, false);
    TestTableMatcherCacheLeft<fst::StdArc>(true);
    TestTableMatcherCacheLeft<fst::StdArc>(false);
    TestTableMatcherCacheRight<fst::StdArc>(true);
    TestTableMatcherCacheRight<fst::StdArc>(false);
  }
}
