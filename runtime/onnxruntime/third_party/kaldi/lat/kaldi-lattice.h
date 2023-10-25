// lat/kaldi-lattice.h

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


#ifndef KALDI_LAT_KALDI_LATTICE_H_
#define KALDI_LAT_KALDI_LATTICE_H_

#include "fstext/fstext-lib.h"
#include "base/kaldi-common.h"
//#include "util/common-utils.h"


namespace kaldi {
// will import some things above...

typedef fst::LatticeWeightTpl<BaseFloat> LatticeWeight;

// careful: kaldi::int32 is not always the same C type as fst::int32
typedef fst::CompactLatticeWeightTpl<LatticeWeight, int32> CompactLatticeWeight;

typedef fst::CompactLatticeWeightCommonDivisorTpl<LatticeWeight, int32>
  CompactLatticeWeightCommonDivisor;

typedef fst::ArcTpl<LatticeWeight> LatticeArc;

typedef fst::ArcTpl<CompactLatticeWeight> CompactLatticeArc;

typedef fst::VectorFst<LatticeArc> Lattice;

typedef fst::VectorFst<CompactLatticeArc> CompactLattice;

// The following functions for writing and reading lattices in binary or text
// form are provided here in case you need to include lattices in larger,
// Kaldi-type objects with their own Read and Write functions.  Caution: these
// functions return false on stream failure rather than throwing an exception as
// most similar Kaldi functions would do.

bool WriteCompactLattice(std::ostream &os, bool binary,
                         const CompactLattice &clat);
bool WriteLattice(std::ostream &os, bool binary,
                  const Lattice &lat);

// the following function requires that *clat be
// NULL when called.
bool ReadCompactLattice(std::istream &is, bool binary,
                        CompactLattice **clat);
// the following function requires that *lat be
// NULL when called.
bool ReadLattice(std::istream &is, bool binary,
                 Lattice **lat);


class CompactLatticeHolder {
 public:
  typedef CompactLattice T;

  CompactLatticeHolder() { t_ = NULL; }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    // Note: we don't include the binary-mode header when writing
    // this object to disk; this ensures that if we write to single
    // files, the result can be read by OpenFst.
    return WriteCompactLattice(os, binary, t);
  }

  bool Read(std::istream &is);

  static bool IsReadInBinary() { return true; }

  T &Value() {
    KALDI_ASSERT(t_ != NULL && "Called Value() on empty CompactLatticeHolder");
    return *t_;
  }

  void Clear() { delete t_; t_ = NULL; }

  void Swap(CompactLatticeHolder *other) {
    std::swap(t_, other->t_);
  }

  bool ExtractRange(const CompactLatticeHolder &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~CompactLatticeHolder() { Clear(); }
 private:
  T *t_;
};

class LatticeHolder {
 public:
  typedef Lattice T;

  LatticeHolder() { t_ = NULL; }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    // Note: we don't include the binary-mode header when writing
    // this object to disk; this ensures that if we write to single
    // files, the result can be read by OpenFst.
    return WriteLattice(os, binary, t);
  }

  bool Read(std::istream &is);

  static bool IsReadInBinary() { return true; }

  T &Value() {
    KALDI_ASSERT(t_ != NULL && "Called Value() on empty LatticeHolder");
    return *t_;
  }

  void Clear() {  delete t_; t_ = NULL; }

  void Swap(LatticeHolder *other) {
    std::swap(t_, other->t_);
  }

  bool ExtractRange(const LatticeHolder &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~LatticeHolder() { Clear(); }
 private:
  T *t_;
};

//typedef TableWriter<LatticeHolder> LatticeWriter;
//typedef SequentialTableReader<LatticeHolder> SequentialLatticeReader;
//typedef RandomAccessTableReader<LatticeHolder> RandomAccessLatticeReader;
//
//typedef TableWriter<CompactLatticeHolder> CompactLatticeWriter;
//typedef SequentialTableReader<CompactLatticeHolder> SequentialCompactLatticeReader;
//typedef RandomAccessTableReader<CompactLatticeHolder> RandomAccessCompactLatticeReader;


} // namespace kaldi

#endif  // KALDI_LAT_KALDI_LATTICE_H_
