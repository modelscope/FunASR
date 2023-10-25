// fstext/deterministic-fst-test.cc

// Copyright 2009-2011  Gilles Boulianne

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

#include "fstext/deterministic-fst.h"
#include "fstext/fst-test-utils.h"
#include "util/kaldi-io.h"

#include <sys/stat.h>

namespace fst {
using std::cout;
using std::cerr;
using std::endl;

bool FileExists(std::string strFilename) {
  struct stat stFileInfo;
  bool blnReturn;
  int intStat;

  // Attempt to get the file attributes
  intStat = stat(strFilename.c_str(), &stFileInfo);
  if (intStat == 0) {
    // We were able to get the file attributes
    // so the file obviously exists.
    blnReturn = true;
  } else {
    // We were not able to get the file attributes.
    // This may mean that we don't have permission to
    // access the folder which contains this file. If you
    // need to do that level of checking, lookup the
    // return values of stat which will give you
    // more details on why stat failed.
    blnReturn = false;
  }

  return blnReturn;
}

// Simplify writing
typedef fst::StdArc          StdArc;
typedef fst::StdArc::Label   Label;
typedef fst::StdArc::StateId StateId;
typedef fst::StdVectorFst    StdVectorFst;
typedef fst::StdArc::Weight  Weight;


// something that looks like a language model FST with epsilon backoffs
StdVectorFst* CreateBackoffFst() {
  StdVectorFst *fst = new StdVectorFst();
  fst->AddState();   // state 0
  fst->SetStart(0);
  fst->AddArc(0, StdArc(10, 10, 0.0, 1));

  fst->AddState();    // state 1
  fst->AddArc(1, StdArc(12, 12, 0.0, 4));
  fst->AddArc(1, StdArc(0,0,  0.1,2));  // backoff from 1 to 2

  fst->AddState();    // state 2
  fst->AddArc(2, StdArc(13, 13, 0.2, 4));
  fst->AddArc(2, StdArc(0,0,  0.3,3));  // backoff from 2 to 3

  fst->AddState();     // state 3
  fst->AddArc(3, StdArc(14, 14, 0.4, 4));

  fst->AddState();    // state 4
  fst->AddArc(4, StdArc(15, 15, 0.5, 5));

  fst->AddState();     // state 5
  fst->SetFinal(5, 0.6);

  return fst;
}

// what the resulting DeterministicOnDemand FST should be like
StdVectorFst* CreateResultFst() {
  StdVectorFst *fst = new StdVectorFst();
  fst->AddState();   // state 0
  fst->SetStart(0);
  fst->AddArc(0, StdArc(10, 10, 0.0, 1));

  fst->AddState();    // state 1
  fst->AddArc(1, StdArc(12, 12, 0.0, 4));
  fst->AddArc(1, StdArc(13,13,0.3,4));  // went through 1 backoff
  fst->AddArc(1, StdArc(14,14,0.8,4));  // went through 2 backoffs

  fst->AddState();    // state 2
  fst->AddState();    // state 3

  fst->AddState();    // state 4
  fst->AddArc(4, StdArc(15, 15, 0.5, 5));

  fst->AddState();     // state 5
  fst->SetFinal(5, 0.6);

  return fst;
}

void DeleteTestFst(StdVectorFst *fst) {
  delete fst;
}

// Follow paths from an input fst representing a string
// (poor man's composition)
Weight WalkSinglePath(StdVectorFst *ifst, DeterministicOnDemandFst<StdArc> *dfst) {
  StdArc oarc; // =  new StdArc();
  StateId isrc=ifst->Start();
  StateId dsrc=dfst->Start();
  Weight totalCost = Weight::One();

  while (ifst->Final(isrc) == Weight::Zero()) { // while not final
    fst::ArcIterator<StdVectorFst> aiter(*ifst, isrc);
    const StdArc &iarc = aiter.Value();
    if (dfst->GetArc(dsrc, iarc.olabel, &oarc)) {
      Weight cost = Times(iarc.weight, oarc.weight);
      // cout << "  Matched label "<<iarc.olabel<<" at summed cost "<<cost<<endl;
      totalCost = Times(totalCost, cost);
    } else {
      cout << "  Can't match arc ["<<iarc.ilabel<<","<<iarc.olabel<<","<<iarc.weight<<"] from "<<isrc<<endl;
      exit(1);
    }
    isrc = iarc.nextstate;
    KALDI_LOG << "Setting dsrc = " << oarc.nextstate;
    dsrc = oarc.nextstate;
  }
  totalCost = Times(totalCost, dfst->Final(dsrc));

  cout << "  Total cost: " << totalCost << endl;
  return totalCost;
}


void TestBackoffAndCache() {
  // Build from existing fst
  cout << "Test with single generated backoff FST" << endl;
  StdVectorFst *nfst = CreateBackoffFst();
  StdVectorFst *rfst = CreateResultFst();

  // before using, make sure that it is input sorted
  ArcSort(nfst, StdILabelCompare());
  BackoffDeterministicOnDemandFst<StdArc> dfst1a(*nfst);
  CacheDeterministicOnDemandFst<StdArc> dfst1(&dfst1a);

  // Compare all arcs in dfst1 with expected result
  for (StateIterator<StdVectorFst> riter(*rfst); !riter.Done(); riter.Next()) {
    StateId rsrc = riter.Value();
    // verify that states have same weight (or final status)
    assert(ApproxEqual(rfst->Final(rsrc), dfst1.Final(rsrc)));
    for (ArcIterator<StdVectorFst> aiter(*rfst, rsrc); !aiter.Done(); aiter.Next()) {
      StdArc rarc = aiter.Value();
      StdArc darc;
      if (dfst1.GetArc(rsrc, rarc.ilabel, &darc)) {
        assert(ApproxEqual(rarc.weight, darc.weight, 0.001));
        assert(rarc.ilabel==darc.ilabel);
        assert(rarc.olabel==darc.olabel);
        assert(rarc.nextstate == darc.nextstate);
        cerr << "  Got same arc at state "<<rsrc<<": "<<rarc.ilabel<<" "<<darc.ilabel<<endl;
      } else {
        cerr << "Couldn't find arc "<<rarc.ilabel<<" for state "<<rsrc<<endl;
        exit(1);
      }
    }
  }
  delete nfst;
  delete rfst;
}

void TestCompose() {
  cout << "Test with single generated backoff FST" << endl;
  StdVectorFst *nfst = CreateBackoffFst();
  StdVectorFst *rfst = CreateResultFst();

  StdVectorFst composed_fst;
  Compose(*rfst, *rfst, &composed_fst);

  // before using, make sure that it is input sorted
  ArcSort(nfst, StdILabelCompare());
  BackoffDeterministicOnDemandFst<StdArc> dfst1a(*nfst);
  ComposeDeterministicOnDemandFst<StdArc> dfst1b(&dfst1a, &dfst1a);
  CacheDeterministicOnDemandFst<StdArc> dfst1(&dfst1b);

  typedef StdArc::StateId StateId;
  std::map<StateId, StateId> state_map;
  state_map[composed_fst.Start()] = dfst1.Start();

  VectorFst<StdArc> path_fst;
  ShortestPath(composed_fst, &path_fst);

  BackoffDeterministicOnDemandFst<StdArc> dfst2(composed_fst);

  Weight w1 = WalkSinglePath(&path_fst, &dfst1),
      w2 = WalkSinglePath(&path_fst, &dfst2);
  KALDI_ASSERT(ApproxEqual(w1, w2));

  delete rfst;
  delete nfst;

  { // Mostly checking for compilation errors here.
    LmExampleDeterministicOnDemandFst<StdArc> lm_eg(NULL, 2, 3);
    KALDI_ASSERT(lm_eg.Start() == 0);
    KALDI_ASSERT(lm_eg.Final(0).Value() == 0.5); // I made it this value.
    StdArc arc;
    bool b = lm_eg.GetArc(0, 100, &arc);
    KALDI_ASSERT(b && arc.nextstate == 1 && arc.ilabel == 100 && arc.olabel == 100
                 && arc.weight.Value() == 0.25);
  }
}

}


int main() {
  using namespace fst;
  TestBackoffAndCache();
  TestCompose();
}

