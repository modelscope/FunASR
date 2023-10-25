// lat/kaldi-lattice-test.cc

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


#include "lat/kaldi-lattice.h"
#include "fstext/rand-fst.h"


namespace kaldi {


CompactLattice *RandCompactLattice() {
  Lattice *fst = fst::RandPairFst<LatticeArc>();
  CompactLattice *cfst = new CompactLattice;
  ConvertLattice(*fst, cfst);
  delete fst;
  return cfst;
}

Lattice *RandLattice() {
  Lattice *fst = fst::RandPairFst<LatticeArc>();
  return fst;
}

void TestCompactLatticeTable(bool binary) {
  CompactLatticeWriter writer(binary ? "ark:tmpf" : "ark,t:tmpf");
  int N = 10;
  std::vector<CompactLattice*> lat_vec(N);
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    CompactLattice *fst = RandCompactLattice();
    lat_vec[i] = fst;
    writer.Write(key, *fst);
  }
  writer.Close();

  RandomAccessCompactLatticeReader reader("ark:tmpf");  
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    const CompactLattice &fst = reader.Value(key);
    KALDI_ASSERT(fst::Equal(fst, *(lat_vec[i])));
    delete lat_vec[i];
  }
}

// Write as CompactLattice, read as Lattice.
void TestCompactLatticeTableCross(bool binary) {
  CompactLatticeWriter writer(binary ? "ark:tmpf" : "ark,t:tmpf");
  int N = 10;
  std::vector<CompactLattice*> lat_vec(N);
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    CompactLattice *fst = RandCompactLattice();
    lat_vec[i] = fst;
    writer.Write(key, *fst);
  }
  writer.Close();

  RandomAccessLatticeReader reader("ark:tmpf");  
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    const Lattice &fst = reader.Value(key);
    CompactLattice fst2;
    ConvertLattice(fst, &fst2);
    KALDI_ASSERT(fst::Equal(fst2, *(lat_vec[i])));
    delete lat_vec[i];
  }
}

// Lattice, binary.
void TestLatticeTable(bool binary) {
  LatticeWriter writer(binary ? "ark:tmpf" : "ark,t:tmpf");
  int N = 10;
  std::vector<Lattice*> lat_vec(N);
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    Lattice *fst = RandLattice();
    lat_vec[i] = fst;
    writer.Write(key, *fst);
  }
  writer.Close();

  RandomAccessLatticeReader reader("ark:tmpf");  
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    const Lattice &fst = reader.Value(key);
    KALDI_ASSERT(fst::Equal(fst, *(lat_vec[i])));
    delete lat_vec[i];
  }
}


// Write as Lattice, read as CompactLattice.
void TestLatticeTableCross(bool binary) {
  LatticeWriter writer(binary ? "ark:tmpf" : "ark,t:tmpf");
  int N = 10;
  std::vector<Lattice*> lat_vec(N);
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    Lattice *fst = RandLattice();
    lat_vec[i] = fst;
    writer.Write(key, *fst);
  }
  writer.Close();

  RandomAccessCompactLatticeReader reader("ark:tmpf");  
  for (int i = 0; i < N; i++) {
    char buf[2];
    buf[0] = '0' + i;
    buf[1] = '\0';
    std::string key = "key" + std::string(buf);
    const CompactLattice &fst = reader.Value(key);
    Lattice fst2;
    ConvertLattice(fst, &fst2);
    KALDI_ASSERT(fst::RandEquivalent(fst2, *(lat_vec[i]), 5, 0.01, Rand(), 10));
    delete lat_vec[i];
  }
}



} // end namespace kaldi

int main() {
  using namespace kaldi;
  for (int i = 0; i < 2; i++) {
    bool binary = (i%2 == 0);
    TestCompactLatticeTable(binary);
    TestCompactLatticeTableCross(binary);
    TestLatticeTable(binary);
    TestLatticeTableCross(binary);
  }
  std::cout << "Test OK\n";
  
  unlink("tmpf");
}
