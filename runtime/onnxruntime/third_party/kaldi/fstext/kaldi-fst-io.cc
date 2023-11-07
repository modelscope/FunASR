// fstext/kaldi-fst-io.cc

// Copyright 2009-2011  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)
//                2013  Guoguo Chen

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

#include "fstext/kaldi-fst-io.h"
#include "base/kaldi-error.h"
#include "base/kaldi-math.h"
#include "util/kaldi-io.h"

namespace fst {

VectorFst<StdArc> *ReadFstKaldi(std::string rxfilename) {
  if (rxfilename == "") rxfilename = "-"; // interpret "" as stdin,
  // for compatibility with OpenFst conventions.
  kaldi::Input ki(rxfilename);
  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), rxfilename))
    KALDI_ERR << "Reading FST: error reading FST header from "
              << kaldi::PrintableRxfilename(rxfilename);
  FstReadOptions ropts("<unspecified>", &hdr);
  VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(ki.Stream(), ropts);
  if (!fst)
    KALDI_ERR << "Could not read fst from "
              << kaldi::PrintableRxfilename(rxfilename);
  return fst;
}

// Register const fst to load it automatically. Other types like
// olabel_lookahead or ngram or compact_fst should be registered
// through OpenFst registration API.
static fst::FstRegisterer<VectorFst<StdArc>> VectorFst_StdArc_registerer;
static fst::FstRegisterer<ConstFst<StdArc>> ConstFst_StdArc_registerer;

Fst<StdArc> *ReadFstKaldiGeneric(std::string rxfilename, bool throw_on_err) {
  if (rxfilename == "") rxfilename = "-"; // interpret "" as stdin,
  // for compatibility with OpenFst conventions.
  kaldi::Input ki(rxfilename);
  fst::FstHeader hdr;
  // Read FstHeader which contains the type of FST
  if (!hdr.Read(ki.Stream(), rxfilename)) {
    if(throw_on_err) {
      KALDI_ERR << "Reading FST: error reading FST header from "
                << kaldi::PrintableRxfilename(rxfilename);
    } else {
      KALDI_WARN << "We fail to read FST header from "
                 << kaldi::PrintableRxfilename(rxfilename)
                 << ". A NULL pointer is returned.";
      return NULL;
    }
  }
  // Check the type of Arc
  if (hdr.ArcType() != fst::StdArc::Type()) {
    if(throw_on_err) {
      KALDI_ERR << "FST with arc type " << hdr.ArcType() << " is not supported.";
    } else {
      KALDI_WARN << "Fst with arc type" << hdr.ArcType()
                 << " is not supported. A NULL pointer is returned.";
      return NULL;
    }
  }
  // Read the FST
  FstReadOptions ropts("<unspecified>", &hdr);
  Fst<StdArc> *fst = Fst<StdArc>::Read(ki.Stream(), ropts);
  if (!fst) {
    if(throw_on_err) {
      KALDI_ERR << "Could not read fst from "
               << kaldi::PrintableRxfilename(rxfilename);
    } else {
      KALDI_WARN << "Could not read fst from "
                 << kaldi::PrintableRxfilename(rxfilename)
                 << ". A NULL pointer is returned.";
      return NULL;
    }
  }
  return fst;
}

VectorFst<StdArc> *CastOrConvertToVectorFst(Fst<StdArc> *fst) {
  // This version currently supports ConstFst<StdArc> or VectorFst<StdArc>
  std::string real_type = fst->Type();
  KALDI_ASSERT(real_type == "vector" || real_type == "const");
  if (real_type == "vector") {
    return dynamic_cast<VectorFst<StdArc> *>(fst);
  } else {
    // As the 'fst' can't cast to VectorFst, we create a new
    // VectorFst<StdArc> initialized by 'fst', and delete 'fst'.
    VectorFst<StdArc> *new_fst = new VectorFst<StdArc>(*fst);
    delete fst;
    return new_fst;
  }
}

void ReadFstKaldi(std::string rxfilename, fst::StdVectorFst *ofst) {
  fst::StdVectorFst *fst = ReadFstKaldi(rxfilename);
  *ofst = *fst;
  delete fst;
}

void WriteFstKaldi(const VectorFst<StdArc> &fst,
                   std::string wxfilename) {
  if (wxfilename == "") wxfilename = "-"; // interpret "" as stdout,
  // for compatibility with OpenFst conventions.
  bool write_binary = true, write_header = false;
  kaldi::Output ko(wxfilename, write_binary, write_header);
  FstWriteOptions wopts(kaldi::PrintableWxfilename(wxfilename));
  fst.Write(ko.Stream(), wopts);
}

fst::VectorFst<fst::StdArc> *ReadAndPrepareLmFst(std::string rxfilename) {
  // ReadFstKaldi() will die with exception on failure.
  fst::VectorFst<fst::StdArc> *ans = fst::ReadFstKaldi(rxfilename);
  if (ans->Properties(fst::kAcceptor, true) == 0) {
    // If it's not already an acceptor, project on the output, i.e. copy olabels
    // to ilabels.  Generally the G.fst's on disk will have the disambiguation
    // symbol #0 on the input symbols of the backoff arc, and projection will
    // replace them with epsilons which is what is on the output symbols of
    // those arcs.
    fst::Project(ans, fst::PROJECT_OUTPUT);
  }
  if (ans->Properties(fst::kILabelSorted, true) == 0) {
    // Make sure LM is sorted on ilabel.
    fst::ILabelCompare<fst::StdArc> ilabel_comp;
    fst::ArcSort(ans, ilabel_comp);
  }
  return ans;
}

} // end namespace fst
