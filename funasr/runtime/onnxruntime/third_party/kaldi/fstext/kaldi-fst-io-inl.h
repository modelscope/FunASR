// fstext/kaldi-fst-io-inl.h

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

#ifndef KALDI_FSTEXT_KALDI_FST_IO_INL_H_
#define KALDI_FSTEXT_KALDI_FST_IO_INL_H_

#include "util/text-utils.h"

namespace fst {


template <class Arc>
void WriteFstKaldi(std::ostream &os, bool binary,
                   const VectorFst<Arc> &t) {
  bool ok;
  if (binary) {
    // Binary-mode writing.
    ok = t.Write(os, FstWriteOptions());
  } else {
    // Text-mode output.  Note: we expect that t.InputSymbols() and
    // t.OutputSymbols() would always return NULL.  The corresponding input
    // routine would not work if the FST actually had symbols attached.  Write a
    // newline to start the FST; in a table, the first line of the FST will
    // appear on its own line.
    os << '\n';
    bool acceptor = false, write_one = false;
    FstPrinter<Arc> printer(t, t.InputSymbols(), t.OutputSymbols(),
                            NULL, acceptor, write_one, "\t");
    printer.Print(&os, "<unknown>");
    if (os.fail())
      KALDI_ERR << "Stream failure detected writing FST to stream";
    // Write another newline as a terminating character.  The read routine will
    // detect this [this is a Kaldi mechanism, not something in the original
    // OpenFst code].
    os << '\n';
    ok = os.good();
  }
  if (!ok) {
    KALDI_ERR << "Error writing FST to stream";
  }
}

// Utility function used in ReadFstKaldi
template <class W>
inline bool StrToWeight(const std::string &s, bool allow_zero, W *w) {
  std::istringstream strm(s);
  strm >> *w;
  if (strm.fail() || (!allow_zero && *w == W::Zero())) {
    return false;
  }
  return true;
}

template <class Arc>
void ReadFstKaldi(std::istream &is, bool binary,
                  VectorFst<Arc> *fst) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  if (binary) {
    // We don't have access to the filename here, so write [unknown].
    VectorFst<Arc> *ans =
        VectorFst<Arc>::Read(is, fst::FstReadOptions(std::string("[unknown]")));
    if (ans == NULL) {
      KALDI_ERR << "Error reading FST from stream.";
    }
    *fst = *ans;  // shallow copy.
    delete ans;
  } else {
    // Consume the \r on Windows, the \n that the text-form FST format starts
    // with, and any extra spaces that might have got in there somehow.
    while (std::isspace(is.peek()) && is.peek() != '\n') is.get();
    if (is.peek() == '\n') is.get(); // consume the newline.
    else { // saw spaces but no newline.. this is not expected.
      KALDI_ERR << "Reading FST: unexpected sequence of spaces "
                << " at file position " << is.tellg();
    }
    using std::string;
    using std::vector;
    using kaldi::SplitStringToIntegers;
    using kaldi::ConvertStringToInteger;
    fst->DeleteStates();
    string line;
    size_t nline = 0;
    string separator = FLAGS_fst_field_separator + "\r\n";
    while (std::getline(is, line)) {
      nline++;
      vector<string> col;
      // on Windows we'll write in text and read in binary mode.
      kaldi::SplitStringToVector(line, separator.c_str(), true, &col);
      if (col.size() == 0) break; // Empty line is a signal to stop, in our
      // archive format.
      if (col.size() > 5) {
        KALDI_ERR << "Bad line in FST: " << line;
      }
      StateId s;
      if (!ConvertStringToInteger(col[0], &s)) {
        KALDI_ERR << "Bad line in FST: " << line;
      }
      while (s >= fst->NumStates())
        fst->AddState();
      if (nline == 1) fst->SetStart(s);

      bool ok = true;
      Arc arc;
      Weight w;
      StateId d = s;
      switch (col.size()) {
        case 1:
          fst->SetFinal(s, Weight::One());
          break;
        case 2:
          if (!StrToWeight(col[1], true, &w)) ok = false;
          else fst->SetFinal(s, w);
          break;
        case 3: // 3 columns not ok for Lattice format; it's not an acceptor.
          ok = false;
          break;
        case 4:
          ok = ConvertStringToInteger(col[1], &arc.nextstate) &&
              ConvertStringToInteger(col[2], &arc.ilabel) &&
              ConvertStringToInteger(col[3], &arc.olabel);
          if (ok) {
            d = arc.nextstate;
            arc.weight = Weight::One();
            fst->AddArc(s, arc);
          }
          break;
        case 5:
          ok = ConvertStringToInteger(col[1], &arc.nextstate) &&
              ConvertStringToInteger(col[2], &arc.ilabel) &&
              ConvertStringToInteger(col[3], &arc.olabel) &&
              StrToWeight(col[4], false, &arc.weight);
          if (ok) {
            d = arc.nextstate;
            fst->AddArc(s, arc);
          }
          break;
        default:
          ok = false;
      }
      while (d >= fst->NumStates()) fst->AddState();
      if (!ok)
        KALDI_ERR << "Bad line in FST: " << line;
    }
  }
}




template<class Arc> // static
bool VectorFstTplHolder<Arc>::Write(std::ostream &os, bool binary, const T &t) {
  try {
    WriteFstKaldi(os, binary, t);
    return true;
  } catch (...) {
    return false;
  }
}

template<class Arc> // static
bool VectorFstTplHolder<Arc>::Read(std::istream &is) {
  Clear();
  int c = is.peek();
  if (c == -1) {
    KALDI_WARN << "End of stream detected reading Fst";
    return false;
  } else if (isspace(c)) { // The text form of the FST begins
    // with space (normally, '\n'), so this means it's text (the binary form
    // cannot begin with space because it starts with the FST Type() which is not
    // space).
    try {
      t_ = new VectorFst<Arc>();
      ReadFstKaldi(is, false, t_);
    } catch (...) {
      Clear();
      return false;
    }
  } else {  // reading a binary FST.
    try {
      t_ = new VectorFst<Arc>();
      ReadFstKaldi(is, true, t_);
    } catch (...) {
      Clear();
      return false;
    }
  }
  return true;
}

} // namespace fst.

#endif  // KALDI_FSTEXT_KALDI_FST_IO_INL_H_
