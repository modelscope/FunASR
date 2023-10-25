// lat/kaldi-lattice.cc

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


#include "lat/kaldi-lattice.h"
#include "fst/script/print-impl.h"

namespace kaldi {

/// Converts lattice types if necessary, deleting its input.
template<class OrigWeightType>
CompactLattice* ConvertToCompactLattice(fst::VectorFst<OrigWeightType> *ifst) {
  if (!ifst) return NULL;
  CompactLattice *ofst = new CompactLattice();
  ConvertLattice(*ifst, ofst);
  delete ifst;
  return ofst;
}

// This overrides the template if there is no type conversion going on
// (for efficiency).
template<>
CompactLattice* ConvertToCompactLattice(CompactLattice *ifst) {
  return ifst;
}

/// Converts lattice types if necessary, deleting its input.
template<class OrigWeightType>
Lattice* ConvertToLattice(fst::VectorFst<OrigWeightType> *ifst) {
  if (!ifst) return NULL;
  Lattice *ofst = new Lattice();
  ConvertLattice(*ifst, ofst);
  delete ifst;
  return ofst;
}

// This overrides the template if there is no type conversion going on
// (for efficiency).
template<>
Lattice* ConvertToLattice(Lattice *ifst) {
  return ifst;
}


bool WriteCompactLattice(std::ostream &os, bool binary,
                         const CompactLattice &t) {
  if (binary) {
    fst::FstWriteOptions opts;
    // Leave all the options default.  Normally these lattices wouldn't have any
    // osymbols/isymbols so no point directing it not to write them (who knows what
    // we'd want to if we had them).
    return t.Write(os, opts);
  } else {
    // Text-mode output.  Note: we expect that t.InputSymbols() and
    // t.OutputSymbols() would always return NULL.  The corresponding input
    // routine would not work if the FST actually had symbols attached.
    // Write a newline after the key, so the first line of the FST appears
    // on its own line.
    os << '\n';
    bool acceptor = true, write_one = false;
    fst::FstPrinter<CompactLatticeArc> printer(t, t.InputSymbols(),
                                               t.OutputSymbols(),
                                               NULL, acceptor, write_one, "\t");
    printer.Print(&os, "<unknown>");
    if (os.fail())
      KALDI_WARN << "Stream failure detected.";
    // Write another newline as a terminating character.  The read routine will
    // detect this [this is a Kaldi mechanism, not somethig in the original
    // OpenFst code].
    os << '\n';
    return os.good();
  }
}

/// LatticeReader provides (static) functions for reading both Lattice
/// and CompactLattice, in text form.
class LatticeReader {
  typedef LatticeArc Arc;
  typedef LatticeWeight Weight;
  typedef CompactLatticeArc CArc;
  typedef CompactLatticeWeight CWeight;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
 public:
  // everything is static in this class.

  /** This function reads from the FST text format; it does not know in advance
      whether it's a Lattice or CompactLattice in the stream so it tries to
      read both formats until it becomes clear which is the correct one.
  */
  static std::pair<Lattice*, CompactLattice*> ReadText(
      std::istream &is) {
    typedef std::pair<Lattice*, CompactLattice*> PairT;
    using std::string;
    using std::vector;
    Lattice *fst = new Lattice();
    CompactLattice *cfst = new CompactLattice();
    string line;
    size_t nline = 0;
    string separator = FLAGS_fst_field_separator + "\r\n";
    while (std::getline(is, line)) {
      nline++;
      vector<string> col;
      // on Windows we'll write in text and read in binary mode.
      SplitStringToVector(line, separator.c_str(), true, &col);
      if (col.size() == 0) break; // Empty line is a signal to stop, in our
      // archive format.
      if (col.size() > 5) {
        KALDI_WARN << "Reading lattice: bad line in FST: " << line;
        delete fst;
        delete cfst;
        return PairT(static_cast<Lattice*>(NULL),
                     static_cast<CompactLattice*>(NULL));
      }
      StateId s;
      if (!ConvertStringToInteger(col[0], &s)) {
        KALDI_WARN << "FstCompiler: bad line in FST: " << line;
        delete fst;
        delete cfst;
        return PairT(static_cast<Lattice*>(NULL),
                     static_cast<CompactLattice*>(NULL));
      }
      if (fst)
        while (s >= fst->NumStates())
          fst->AddState();
      if (cfst)
        while (s >= cfst->NumStates())
          cfst->AddState();
      if (nline == 1) {
        if (fst) fst->SetStart(s);
        if (cfst) cfst->SetStart(s);
      }

      if (fst) { // we still have fst; try to read that arc.
        bool ok = true;
        Arc arc;
        Weight w;
        StateId d = s;
        switch (col.size()) {
          case 1 :
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
        while (d >= fst->NumStates())
          fst->AddState();
        if (!ok) {
          delete fst;
          fst = NULL;
        }
      }
      if (cfst) {
        bool ok = true;
        CArc arc;
        CWeight w;
        StateId d = s;
        switch (col.size()) {
          case 1 :
            cfst->SetFinal(s, CWeight::One());
            break;
          case 2:
            if (!StrToCWeight(col[1], true, &w)) ok = false;
            else cfst->SetFinal(s, w);
            break;
          case 3: // compact-lattice is acceptor format: state, next-state, label.
            ok = ConvertStringToInteger(col[1], &arc.nextstate) &&
                ConvertStringToInteger(col[2], &arc.ilabel);
            if (ok) {
              d = arc.nextstate;
              arc.olabel = arc.ilabel;
              arc.weight = CWeight::One();
              cfst->AddArc(s, arc);
            }
            break;
          case 4:
            ok = ConvertStringToInteger(col[1], &arc.nextstate) &&
                ConvertStringToInteger(col[2], &arc.ilabel) &&
                StrToCWeight(col[3], false, &arc.weight);
            if (ok) {
              d = arc.nextstate;
              arc.olabel = arc.ilabel;
              cfst->AddArc(s, arc);
            }
            break;
          case 5: default:
            ok = false;
        }
        while (d >= cfst->NumStates())
          cfst->AddState();
        if (!ok) {
          delete cfst;
          cfst = NULL;
        }
      }
      if (!fst && !cfst) {
        KALDI_WARN << "Bad line in lattice text format: " << line;
        // read until we get an empty line, so at least we
        // have a chance to read the next one (although this might
        // be a bit futile since the calling code will get unhappy
        // about failing to read this one.
        while (std::getline(is, line)) {
          SplitStringToVector(line, separator.c_str(), true, &col);
          if (col.empty()) break;
        }
        return PairT(static_cast<Lattice*>(NULL),
                     static_cast<CompactLattice*>(NULL));
      }
    }
    return PairT(fst, cfst);
  }

  static bool StrToWeight(const std::string &s, bool allow_zero, Weight *w) {
    std::istringstream strm(s);
    strm >> *w;
    if (!strm || (!allow_zero && *w == Weight::Zero())) {
      return false;
    }
    return true;
  }

  static  bool StrToCWeight(const std::string &s, bool allow_zero, CWeight *w) {
    std::istringstream strm(s);
    strm >> *w;
    if (!strm || (!allow_zero && *w == CWeight::Zero())) {
      return false;
    }
    return true;
  }
};


CompactLattice *ReadCompactLatticeText(std::istream &is) {
  std::pair<Lattice*, CompactLattice*> lat_pair = LatticeReader::ReadText(is);
  if (lat_pair.second != NULL) {
    delete lat_pair.first;
    return lat_pair.second;
  } else if (lat_pair.first != NULL) {
    // note: ConvertToCompactLattice frees its input.
    return ConvertToCompactLattice(lat_pair.first);
  } else {
    return NULL;
  }
}


Lattice *ReadLatticeText(std::istream &is) {
  std::pair<Lattice*, CompactLattice*> lat_pair = LatticeReader::ReadText(is);
  if (lat_pair.first != NULL) {
    delete lat_pair.second;
    return lat_pair.first;
  } else if (lat_pair.second != NULL) {
    // note: ConvertToLattice frees its input.
    return ConvertToLattice(lat_pair.second);
  } else {
    return NULL;
  }
}

bool ReadCompactLattice(std::istream &is, bool binary,
                        CompactLattice **clat) {
  KALDI_ASSERT(*clat == NULL);
  if (binary) {
    fst::FstHeader hdr;
    if (!hdr.Read(is, "<unknown>")) {
      KALDI_WARN << "Reading compact lattice: error reading FST header.";
      return false;
    }
    if (hdr.FstType() != "vector") {
      KALDI_WARN << "Reading compact lattice: unsupported FST type: "
                 << hdr.FstType();
      return false;
    }
    fst::FstReadOptions ropts("<unspecified>",
                              &hdr);

    typedef fst::CompactLatticeWeightTpl<fst::LatticeWeightTpl<float>, int32> T1;
    typedef fst::CompactLatticeWeightTpl<fst::LatticeWeightTpl<double>, int32> T2;
    typedef fst::LatticeWeightTpl<float> T3;
    typedef fst::LatticeWeightTpl<double> T4;
    typedef fst::VectorFst<fst::ArcTpl<T1> > F1;
    typedef fst::VectorFst<fst::ArcTpl<T2> > F2;
    typedef fst::VectorFst<fst::ArcTpl<T3> > F3;
    typedef fst::VectorFst<fst::ArcTpl<T4> > F4;

    CompactLattice *ans = NULL;
    if (hdr.ArcType() == T1::Type()) {
      ans = ConvertToCompactLattice(F1::Read(is, ropts));
    } else if (hdr.ArcType() == T2::Type()) {
      ans = ConvertToCompactLattice(F2::Read(is, ropts));
    } else if (hdr.ArcType() == T3::Type()) {
      ans = ConvertToCompactLattice(F3::Read(is, ropts));
    } else if (hdr.ArcType() == T4::Type()) {
      ans = ConvertToCompactLattice(F4::Read(is, ropts));
    } else {
      KALDI_WARN << "FST with arc type " << hdr.ArcType()
                 << " cannot be converted to CompactLattice.\n";
      return false;
    }
    if (ans == NULL) {
      KALDI_WARN << "Error reading compact lattice (after reading header).";
      return false;
    }
    *clat = ans;
    return true;
  } else {
    // The next line would normally consume the \r on Windows, plus any
    // extra spaces that might have got in there somehow.
    while (std::isspace(is.peek()) && is.peek() != '\n') is.get();
    if (is.peek() == '\n') is.get(); // consume the newline.
    else { // saw spaces but no newline.. this is not expected.
      KALDI_WARN << "Reading compact lattice: unexpected sequence of spaces "
                 << " at file position " << is.tellg();
      return false;
    }
    *clat = ReadCompactLatticeText(is); // that routine will warn on error.
    return (*clat != NULL);
  }
}


bool CompactLatticeHolder::Read(std::istream &is) {
  Clear(); // in case anything currently stored.
  int c = is.peek();
  if (c == -1) {
    KALDI_WARN << "End of stream detected reading CompactLattice.";
    return false;
  } else if (isspace(c)) { // The text form of the lattice begins
    // with space (normally, '\n'), so this means it's text (the binary form
    // cannot begin with space because it starts with the FST Type() which is not
    // space).
    return ReadCompactLattice(is, false, &t_);
  } else if (c != 214) { // 214 is first char of FST magic number,
    // on little-endian machines which is all we support (\326 octal)
    KALDI_WARN << "Reading compact lattice: does not appear to be an FST "
               << " [non-space but no magic number detected], file pos is "
               << is.tellg();
    return false;
  } else {
    return ReadCompactLattice(is, true, &t_);
  }
}

bool WriteLattice(std::ostream &os, bool binary, const Lattice &t) {
  if (binary) {
    fst::FstWriteOptions opts;
    // Leave all the options default.  Normally these lattices wouldn't have any
    // osymbols/isymbols so no point directing it not to write them (who knows what
    // we'd want to do if we had them).
    return t.Write(os, opts);
  } else {
    // Text-mode output.  Note: we expect that t.InputSymbols() and
    // t.OutputSymbols() would always return NULL.  The corresponding input
    // routine would not work if the FST actually had symbols attached.
    // Write a newline after the key, so the first line of the FST appears
    // on its own line.
    os << '\n';
    bool acceptor = false, write_one = false;
    fst::FstPrinter<LatticeArc> printer(t, t.InputSymbols(),
                                        t.OutputSymbols(),
                                        NULL, acceptor, write_one, "\t");
    printer.Print(&os, "<unknown>");
    if (os.fail())
      KALDI_WARN << "Stream failure detected.";
    // Write another newline as a terminating character.  The read routine will
    // detect this [this is a Kaldi mechanism, not somethig in the original
    // OpenFst code].
    os << '\n';
    return os.good();
  }
}

bool ReadLattice(std::istream &is, bool binary,
                 Lattice **lat) {
  KALDI_ASSERT(*lat == NULL);
  if (binary) {
    fst::FstHeader hdr;
    if (!hdr.Read(is, "<unknown>")) {
      KALDI_WARN << "Reading lattice: error reading FST header.";
      return false;
    }
    if (hdr.FstType() != "vector") {
      KALDI_WARN << "Reading lattice: unsupported FST type: "
                 << hdr.FstType();
      return false;
    }
    fst::FstReadOptions ropts("<unspecified>",
                              &hdr);

    typedef fst::CompactLatticeWeightTpl<fst::LatticeWeightTpl<float>, int32> T1;
    typedef fst::CompactLatticeWeightTpl<fst::LatticeWeightTpl<double>, int32> T2;
    typedef fst::LatticeWeightTpl<float> T3;
    typedef fst::LatticeWeightTpl<double> T4;
    typedef fst::VectorFst<fst::ArcTpl<T1> > F1;
    typedef fst::VectorFst<fst::ArcTpl<T2> > F2;
    typedef fst::VectorFst<fst::ArcTpl<T3> > F3;
    typedef fst::VectorFst<fst::ArcTpl<T4> > F4;

    Lattice *ans = NULL;
    if (hdr.ArcType() == T1::Type()) {
      ans = ConvertToLattice(F1::Read(is, ropts));
    } else if (hdr.ArcType() == T2::Type()) {
      ans = ConvertToLattice(F2::Read(is, ropts));
    } else if (hdr.ArcType() == T3::Type()) {
      ans = ConvertToLattice(F3::Read(is, ropts));
    } else if (hdr.ArcType() == T4::Type()) {
      ans = ConvertToLattice(F4::Read(is, ropts));
    } else {
      KALDI_WARN << "FST with arc type " << hdr.ArcType()
                 << " cannot be converted to Lattice.\n";
      return false;
    }
    if (ans == NULL) {
      KALDI_WARN << "Error reading lattice (after reading header).";
      return false;
    }
    *lat = ans;
    return true;
  } else {
    // The next line would normally consume the \r on Windows, plus any
    // extra spaces that might have got in there somehow.
    while (std::isspace(is.peek()) && is.peek() != '\n') is.get();
    if (is.peek() == '\n') is.get(); // consume the newline.
    else { // saw spaces but no newline.. this is not expected.
      KALDI_WARN << "Reading compact lattice: unexpected sequence of spaces "
                 << " at file position " << is.tellg();
      return false;
    }
    *lat = ReadLatticeText(is); // that routine will warn on error.
    return (*lat != NULL);
  }
}


/* Since we don't write the binary headers for this type of holder,
   we use a different method to work out whether we're in binary mode.
 */
bool LatticeHolder::Read(std::istream &is) {
  Clear(); // in case anything currently stored.
  int c = is.peek();
  if (c == -1) {
    KALDI_WARN << "End of stream detected reading Lattice.";
    return false;
  } else if (isspace(c)) { // The text form of the lattice begins
    // with space (normally, '\n'), so this means it's text (the binary form
    // cannot begin with space because it starts with the FST Type() which is not
    // space).
    return ReadLattice(is, false, &t_);
  } else if (c != 214) { // 214 is first char of FST magic number,
    // on little-endian machines which is all we support (\326 octal)
    KALDI_WARN << "Reading compact lattice: does not appear to be an FST "
               << " [non-space but no magic number detected], file pos is "
               << is.tellg();
    return false;
  } else {
    return ReadLattice(is, true, &t_);
  }
}



} // end namespace kaldi
