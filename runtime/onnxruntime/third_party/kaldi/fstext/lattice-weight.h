// fstext/lattice-weight.h
// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_FSTEXT_LATTICE_WEIGHT_H_
#define KALDI_FSTEXT_LATTICE_WEIGHT_H_

#include "fst/fstlib.h"
#include "base/kaldi-common.h"

namespace fst {

// Declare weight type for lattice... will import to namespace kaldi.  has two
// members, value1_ and value2_, of type BaseFloat (normally equals float).  It
// is basically the same as the tropical semiring on value1_+value2_, except it
// keeps track of a and b separately.  More precisely, it is equivalent to the
// lexicographic semiring on (value1_+value2_), (value1_-value2_)


template<class FloatType>
class LatticeWeightTpl;

template <class FloatType>
inline std::ostream &operator <<(std::ostream &strm, const LatticeWeightTpl<FloatType> &w);

template <class FloatType>
inline std::istream &operator >>(std::istream &strm, LatticeWeightTpl<FloatType> &w);


template<class FloatType>
class LatticeWeightTpl {
 public:
  typedef FloatType T; // normally float.
  typedef LatticeWeightTpl ReverseWeight;

  inline T Value1() const { return value1_; }

  inline T Value2() const { return value2_; }

  inline void SetValue1(T f) { value1_ = f; }

  inline void SetValue2(T f) { value2_ = f; }

  LatticeWeightTpl(): value1_{}, value2_{} { }

  LatticeWeightTpl(T a, T b): value1_(a), value2_(b) {}

  LatticeWeightTpl(const LatticeWeightTpl &other): value1_(other.value1_), value2_(other.value2_) { }

  LatticeWeightTpl &operator=(const LatticeWeightTpl &w) {
    value1_ = w.value1_;
    value2_ = w.value2_;
    return *this;
  }

  LatticeWeightTpl<FloatType> Reverse() const {
    return *this;
  }

  static const LatticeWeightTpl Zero() {
    return LatticeWeightTpl(std::numeric_limits<T>::infinity(),
                            std::numeric_limits<T>::infinity());
  }

  static const LatticeWeightTpl One() {
    return LatticeWeightTpl(0.0, 0.0);
  }

  static const std::string &Type() {
    static const std::string type = (sizeof(T) == 4 ? "lattice4" : "lattice8") ;
    return type;
  }

  static const LatticeWeightTpl NoWeight() {
    return LatticeWeightTpl(std::numeric_limits<FloatType>::quiet_NaN(),
                            std::numeric_limits<FloatType>::quiet_NaN());
  }

  bool Member() const {
    // value1_ == value1_ tests for NaN.
    // also test for no -inf, and either both or neither
    // must be +inf, and
    if (value1_ != value1_ || value2_ != value2_) return false; // NaN
    if (value1_ == -std::numeric_limits<T>::infinity()  ||
       value2_ == -std::numeric_limits<T>::infinity()) return false; // -infty not allowed
    if (value1_ == std::numeric_limits<T>::infinity() ||
        value2_ == std::numeric_limits<T>::infinity()) {
      if (value1_ != std::numeric_limits<T>::infinity() ||
          value2_ != std::numeric_limits<T>::infinity()) return false; // both must be +infty;
      // this is necessary so that the semiring has only one zero.
    }
    return true;
  }

  LatticeWeightTpl Quantize(float delta = kDelta) const {
    if (value1_ + value2_ == -std::numeric_limits<T>::infinity()) {
      return LatticeWeightTpl(-std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity());
    } else if (value1_ + value2_ == std::numeric_limits<T>::infinity()) {
      return LatticeWeightTpl(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
    } else if (value1_ + value2_ != value1_ + value2_) { // NaN
      return LatticeWeightTpl(value1_ + value2_, value1_ + value2_);
    } else {
      return LatticeWeightTpl(floor(value1_/delta + 0.5F)*delta, floor(value2_/delta + 0.5F) * delta);
    }
  }
  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative |
        kPath | kIdempotent;
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  std::istream &Read(std::istream &strm) {
    // Always read/write as float, even if T is double,
    // so we can use OpenFst-style read/write and still maintain
    // compatibility when compiling with different FloatTypes
    ReadType(strm, &value1_);
    ReadType(strm, &value2_);
    return strm;
  }


  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  std::ostream &Write(std::ostream &strm) const {
    WriteType(strm, value1_);
    WriteType(strm, value2_);
    return strm;
  }

  size_t Hash() const {
    size_t ans;
    union {
      T f;
      size_t s;
    } u;
    u.s = 0;
    u.f = value1_;
    ans = u.s;
    u.f = value2_;
    ans += u.s;
    return ans;
  }

 protected:
  inline static void WriteFloatType(std::ostream &strm, const T &f) {
    if (f == std::numeric_limits<T>::infinity())
      strm << "Infinity";
    else if (f == -std::numeric_limits<T>::infinity())
      strm << "-Infinity";
    else if (f != f)
      strm << "BadNumber";
    else
      strm << f;
  }

  // Internal helper function, used in ReadNoParen.
  inline static void ReadFloatType(std::istream &strm, T &f) {
    std::string s;
    strm >> s;
    if (s == "Infinity") {
      f = std::numeric_limits<T>::infinity();
    } else if (s == "-Infinity") {
      f = -std::numeric_limits<T>::infinity();
    } else if (s == "BadNumber") {
      f = std::numeric_limits<T>::quiet_NaN();
    } else {
      char *p;
      f = strtod(s.c_str(), &p);
      if (p < s.c_str() + s.size())
        strm.clear(std::ios::badbit);
    }
  }

  // Reads LatticeWeight when there are no parentheses around pair terms...
  // currently the only form supported.
  inline std::istream &ReadNoParen(
      std::istream &strm, char separator) {
    int c;
    do {
      c = strm.get();
    } while (isspace(c));

    std::string s1;
    while (c != separator) {
      if (c == EOF) {
        strm.clear(std::ios::badbit);
        return strm;
      }
      s1 += c;
      c = strm.get();
    }
    std::istringstream strm1(s1);
    ReadFloatType(strm1, value1_); // ReadFloatType is class member function
    // read second element
    ReadFloatType(strm, value2_);
    return strm;
  }

  friend std::istream &operator>> <FloatType>(std::istream&, LatticeWeightTpl<FloatType>&);
  friend std::ostream &operator<< <FloatType>(std::ostream&, const LatticeWeightTpl<FloatType>&);

 private:
  T value1_;
  T value2_;


};


/* ScaleTupleWeight is a function defined for LatticeWeightTpl and
   CompactLatticeWeightTpl that mutliplies the pair (value1_, value2_) by a 2x2
   matrix.  Used, for example, in applying acoustic scaling.
 */
template<class FloatType, class ScaleFloatType>
inline LatticeWeightTpl<FloatType> ScaleTupleWeight(
    const LatticeWeightTpl<FloatType> &w,
    const std::vector<std::vector<ScaleFloatType> > &scale) {
  // Without the next special case we'd get NaNs from infinity * 0
  if (w.Value1() == std::numeric_limits<FloatType>::infinity())
    return LatticeWeightTpl<FloatType>::Zero();
  return LatticeWeightTpl<FloatType>(scale[0][0] * w.Value1() + scale[0][1] * w.Value2(),
                                     scale[1][0] * w.Value1() + scale[1][1] * w.Value2());
}

/* For testing purposes and in case it's ever useful, we define a similar
   function to apply to LexicographicWeight and the like, templated on
   TropicalWeight<float> etc.; we use PairWeight which is the base class of
   LexicographicWeight.
*/
template<class FloatType, class ScaleFloatType>
inline PairWeight<TropicalWeightTpl<FloatType>,
                  TropicalWeightTpl<FloatType> > ScaleTupleWeight(
                      const PairWeight<TropicalWeightTpl<FloatType>,
                                       TropicalWeightTpl<FloatType> > &w,
                      const std::vector<std::vector<ScaleFloatType> > &scale) {
  typedef TropicalWeightTpl<FloatType> BaseType;
  typedef PairWeight<BaseType, BaseType> PairType;
  const BaseType zero = BaseType::Zero();
  // Without the next special case we'd get NaNs from infinity * 0
  if (w.Value1() == zero || w.Value2() == zero)
    return PairType(zero, zero);
  FloatType f1 = w.Value1().Value(), f2 = w.Value2().Value();
  return PairType(BaseType(scale[0][0] * f1 + scale[0][1] * f2),
                  BaseType(scale[1][0] * f1 + scale[1][1] * f2));
}



template<class FloatType>
inline bool operator==(const LatticeWeightTpl<FloatType> &wa,
                       const LatticeWeightTpl<FloatType> &wb) {
  // Volatile qualifier thwarts over-aggressive compiler optimizations
  // that lead to problems esp. with NaturalLess().
  volatile FloatType va1 = wa.Value1(), va2 = wa.Value2(),
      vb1 = wb.Value1(), vb2 = wb.Value2();
  return (va1 == vb1 && va2 == vb2);
}

template<class FloatType>
inline bool operator!=(const LatticeWeightTpl<FloatType> &wa,
                       const LatticeWeightTpl<FloatType> &wb) {
  // Volatile qualifier thwarts over-aggressive compiler optimizations
  // that lead to problems esp. with NaturalLess().
  volatile FloatType va1 = wa.Value1(), va2 = wa.Value2(),
      vb1 = wb.Value1(), vb2 = wb.Value2();
  return (va1 != vb1 || va2 != vb2);
}


// We define a Compare function LatticeWeightTpl even though it's
// not required by the semiring standard-- it's just more efficient
// to do it this way rather than using the NaturalLess template.

/// Compare returns -1 if w1 < w2, +1 if w1 > w2, and 0 if w1 == w2.

template<class FloatType>
inline int Compare (const LatticeWeightTpl<FloatType> &w1,
                    const LatticeWeightTpl<FloatType> &w2) {
  FloatType f1 = w1.Value1() + w1.Value2(),
      f2 = w2.Value1() + w2.Value2();
  if (f1 < f2) { return 1; } // having smaller cost means you're larger
  // in the semiring [higher probability]
  else if (f1 > f2) { return -1; }
  // mathematically we should be comparing (w1.value1_-w1.value2_ < w2.value1_-w2.value2_)
  // in the next line, but add w1.value1_+w1.value2_ = w2.value1_+w2.value2_ to both sides and
  // divide by two, and we get the simpler equivalent form w1.value1_ < w2.value1_.
  else if (w1.Value1() < w2.Value1()) { return 1; }
  else if (w1.Value1() > w2.Value1()) { return -1; }
  else { return 0; }
}


template<class FloatType>
inline LatticeWeightTpl<FloatType> Plus(const LatticeWeightTpl<FloatType> &w1,
                                        const LatticeWeightTpl<FloatType> &w2) {
  return (Compare(w1, w2) >= 0 ? w1 : w2);
}


// For efficiency, override the NaturalLess template class.
template<class FloatType>
class NaturalLess<LatticeWeightTpl<FloatType> > {
 public:
  typedef LatticeWeightTpl<FloatType> Weight;

  NaturalLess() {}

  bool operator()(const Weight &w1, const Weight &w2) const {
    // NaturalLess is a negative order (opposite to normal ordering).
    // This operator () corresponds to "<" in the negative order, which
    // corresponds to the ">" in the normal order.
    return (Compare(w1, w2) == 1);
  }
};
template<>
class NaturalLess<LatticeWeightTpl<float> > {
 public:
  typedef LatticeWeightTpl<float> Weight;

  NaturalLess() {}

  bool operator()(const Weight &w1, const Weight &w2) const {
    // NaturalLess is a negative order (opposite to normal ordering).
    // This operator () corresponds to "<" in the negative order, which
    // corresponds to the ">" in the normal order.
    return (Compare(w1, w2) == 1);
  }
};
template<>
class NaturalLess<LatticeWeightTpl<double> > {
 public:
  typedef LatticeWeightTpl<double> Weight;

  NaturalLess() {}

  bool operator()(const Weight &w1, const Weight &w2) const {
    // NaturalLess is a negative order (opposite to normal ordering).
    // This operator () corresponds to "<" in the negative order, which
    // corresponds to the ">" in the normal order.
    return (Compare(w1, w2) == 1);
  }
};

template<class FloatType>
inline LatticeWeightTpl<FloatType> Times(const LatticeWeightTpl<FloatType> &w1,
                                         const LatticeWeightTpl<FloatType> &w2) {
  return LatticeWeightTpl<FloatType>(w1.Value1()+w2.Value1(), w1.Value2()+w2.Value2());
}

// divide w1 by w2 (on left/right/any doesn't matter as
// commutative).
template<class FloatType>
inline LatticeWeightTpl<FloatType> Divide(const LatticeWeightTpl<FloatType> &w1,
                                          const LatticeWeightTpl<FloatType> &w2,
                                          DivideType typ = DIVIDE_ANY) {
  typedef FloatType T;
  T a = w1.Value1() - w2.Value1(), b = w1.Value2() - w2.Value2();
  if (a != a || b != b || a == -std::numeric_limits<T>::infinity()
     || b == -std::numeric_limits<T>::infinity()) {
    KALDI_WARN << "LatticeWeightTpl::Divide, NaN or invalid number produced. "
               << "[dividing by zero?]  Returning zero";
    return LatticeWeightTpl<T>::Zero();
  }
  if (a == std::numeric_limits<T>::infinity() ||
     b == std::numeric_limits<T>::infinity())
    return LatticeWeightTpl<T>::Zero(); // not a valid number if only one is infinite.
  return LatticeWeightTpl<T>(a, b);
}


template<class FloatType>
inline bool ApproxEqual(const LatticeWeightTpl<FloatType> &w1,
                        const LatticeWeightTpl<FloatType> &w2,
                        float delta = kDelta) {
  if (w1.Value1() == w2.Value1() && w1.Value2() == w2.Value2()) return true;  // handles Zero().
  return (fabs((w1.Value1() + w1.Value2()) - (w2.Value1() + w2.Value2())) <= delta);
}

template <class FloatType>
inline std::ostream &operator <<(std::ostream &strm, const LatticeWeightTpl<FloatType> &w) {
  LatticeWeightTpl<FloatType>::WriteFloatType(strm, w.Value1());
  CHECK(FLAGS_fst_weight_separator.size() == 1);
  strm << FLAGS_fst_weight_separator[0]; // comma by default;
  // may or may not be settable from Kaldi programs.
  LatticeWeightTpl<FloatType>::WriteFloatType(strm, w.Value2());
  return strm;
}

template <class FloatType>
inline std::istream &operator >>(std::istream &strm, LatticeWeightTpl<FloatType> &w1) {
  CHECK(FLAGS_fst_weight_separator.size() == 1);
  // separator defaults to ','
  return w1.ReadNoParen(strm, FLAGS_fst_weight_separator[0]);
}



// CompactLattice will be an acceptor (accepting the words/output-symbols),
// with the weights and input-symbol-seqs on the arcs.
// There must be a total order on W.  We assume for the sake of efficiency
// that there is a function
// Compare(W w1, W w2) that returns -1 if w1 < w2, +1 if w1 > w2, and
// zero if w1 == w2, and Plus for type W returns (Compare(w1,w2) >= 0 ? w1 : w2).

template<class WeightType, class IntType>
class CompactLatticeWeightTpl {
 public:
  typedef WeightType W;

  typedef CompactLatticeWeightTpl<WeightType, IntType> ReverseWeight;

  // Plus is like LexicographicWeight on the pair (weight_, string_), but where we
  // use standard lexicographic order on string_ [this is not the same as
  // NaturalLess on the StringWeight equivalent, which does not define a
  // total order].
  // Times, Divide obvious... (support both left & right division..)
  // CommonDivisor would need to be coded separately.

  CompactLatticeWeightTpl() { }

  CompactLatticeWeightTpl(const WeightType &w, const std::vector<IntType> &s):
      weight_(w), string_(s) { }

  CompactLatticeWeightTpl(const CompactLatticeWeightTpl &compactLatticeWeightTpl) = default;

  CompactLatticeWeightTpl &operator=(const CompactLatticeWeightTpl &w) = default;

  const W &Weight() const { return weight_; }

  const std::vector<IntType> &String() const { return string_; }

  void SetWeight(const W &w) { weight_ = w; }

  void SetString(const std::vector<IntType> &s) { string_ = s; }

  static const CompactLatticeWeightTpl<WeightType, IntType> Zero() {
    return CompactLatticeWeightTpl<WeightType, IntType>(
        WeightType::Zero(), std::vector<IntType>());
  }

  static const CompactLatticeWeightTpl<WeightType, IntType> One() {
    return CompactLatticeWeightTpl<WeightType, IntType>(
        WeightType::One(), std::vector<IntType>());
  }

  inline static std::string GetIntSizeString() {
    char buf[2];
    buf[0] = '0' + sizeof(IntType);
    buf[1] = '\0';
    return buf;
  }
  static const std::string &Type() {
    static const std::string type = "compact" + WeightType::Type()
        + GetIntSizeString();
    return type;
  }

  static const CompactLatticeWeightTpl<WeightType, IntType> NoWeight() {
    return CompactLatticeWeightTpl<WeightType, IntType>(
        WeightType::NoWeight(), std::vector<IntType>());
  }


  CompactLatticeWeightTpl<WeightType, IntType> Reverse() const {
    size_t s = string_.size();
    std::vector<IntType> v(s);
    for(size_t i = 0; i < s; i++)
      v[i] = string_[s-i-1];
    return CompactLatticeWeightTpl<WeightType, IntType>(weight_, v);
  }

  bool Member() const {
    // a semiring has only one zero, this is the important property
    // we're trying to maintain here.  So force string_ to be empty if
    // w_ == zero.
    if (!weight_.Member()) return false;
    if (weight_ == WeightType::Zero())
      return string_.empty();
    else
      return true;
  }

  CompactLatticeWeightTpl Quantize(float delta = kDelta) const {
    return CompactLatticeWeightTpl(weight_.Quantize(delta), string_);
  }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kPath | kIdempotent;
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  std::istream &Read(std::istream &strm) {
    weight_.Read(strm);
    if (strm.fail()){ return strm; }
    int32 sz;
    ReadType(strm, &sz);
    if (strm.fail()){ return strm; }
    if (sz < 0) {
      KALDI_WARN << "Negative string size!  Read failure";
      strm.clear(std::ios::badbit);
      return strm;
    }
    string_.resize(sz);
    for(int32 i = 0; i < sz; i++) {
      ReadType(strm, &(string_[i]));
    }
    return strm;
  }

  // This is used in OpenFst for binary I/O.  This is OpenFst-style,
  // not Kaldi-style, I/O.
  std::ostream &Write(std::ostream &strm) const {
    weight_.Write(strm);
    if (strm.fail()){ return strm; }
    int32 sz = static_cast<int32>(string_.size());
    WriteType(strm, sz);
    for(int32 i = 0; i < sz; i++)
      WriteType(strm, string_[i]);
    return strm;
  }
  size_t Hash() const {
    size_t ans = weight_.Hash();
    // any weird numbers here are largish primes
    size_t sz = string_.size(), mult = 6967;
    for(size_t i = 0; i < sz; i++) {
      ans += string_[i] * mult;
      mult *= 7499;
    }
    return ans;
  }
 private:
  W weight_;
  std::vector<IntType> string_;

};

template<class WeightType, class IntType>
inline bool operator==(const CompactLatticeWeightTpl<WeightType, IntType> &w1,
                       const CompactLatticeWeightTpl<WeightType, IntType> &w2) {
  return (w1.Weight() == w2.Weight() && w1.String() == w2.String());
}

template<class WeightType, class IntType>
inline bool operator!=(const CompactLatticeWeightTpl<WeightType, IntType> &w1,
                       const CompactLatticeWeightTpl<WeightType, IntType> &w2) {
  return (w1.Weight() != w2.Weight() || w1.String() != w2.String());
}

template<class WeightType, class IntType>
inline bool ApproxEqual(const CompactLatticeWeightTpl<WeightType, IntType> &w1,
                        const CompactLatticeWeightTpl<WeightType, IntType> &w2,
                        float delta = kDelta) {
  return (ApproxEqual(w1.Weight(), w2.Weight(), delta) && w1.String() == w2.String());
}



// Compare is not part of the standard for weight types, but used internally for
// efficiency.  The comparison here first compares the weight; if this is the
// same, it compares the string.  The comparison on strings is: first compare
// the length, if this is the same, use lexicographical order.  We can't just
// use the lexicographical order because this would destroy the distributive
// property of multiplication over addition, taking into account that addition
// uses Compare.  The string element of "Compare" isn't super-important in
// practical terms; it's only needed to ensure that Plus always give consistent
// answers and is symmetric.  It's essentially for tie-breaking, but we need to
// make sure all the semiring axioms are satisfied otherwise OpenFst might
// break.

template<class WeightType, class IntType>
inline int Compare(const CompactLatticeWeightTpl<WeightType, IntType> &w1,
                   const CompactLatticeWeightTpl<WeightType, IntType> &w2) {
  int c1 = Compare(w1.Weight(), w2.Weight());
  if (c1 != 0) return c1;
  int l1 = w1.String().size(), l2 = w2.String().size();
  // Use opposite order on the string lengths, so that if the costs are the same,
  // the shorter string wins.
  if (l1 > l2) return -1;
  else if (l1 < l2) return 1;
  for(int i = 0; i < l1; i++) {
    if (w1.String()[i] < w2.String()[i]) return -1;
    else if (w1.String()[i] > w2.String()[i]) return 1;
  }
  return 0;
}

// For efficiency, override the NaturalLess template class.
template<class FloatType, class IntType>
class NaturalLess<CompactLatticeWeightTpl<LatticeWeightTpl<FloatType>, IntType> > {
 public:
  typedef CompactLatticeWeightTpl<LatticeWeightTpl<FloatType>, IntType> Weight;

  NaturalLess() {}

  bool operator()(const Weight &w1, const Weight &w2) const {
    // NaturalLess is a negative order (opposite to normal ordering).
    // This operator () corresponds to "<" in the negative order, which
    // corresponds to the ">" in the normal order.
    return (Compare(w1, w2) == 1);
  }
};
template<>
class NaturalLess<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > {
 public:
  typedef CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> Weight;

  NaturalLess() {}

  bool operator()(const Weight &w1, const Weight &w2) const {
    // NaturalLess is a negative order (opposite to normal ordering).
    // This operator () corresponds to "<" in the negative order, which
    // corresponds to the ">" in the normal order.
    return (Compare(w1, w2) == 1);
  }
};
template<>
class NaturalLess<CompactLatticeWeightTpl<LatticeWeightTpl<double>, int32> > {
 public:
  typedef CompactLatticeWeightTpl<LatticeWeightTpl<double>, int32> Weight;

  NaturalLess() {}

  bool operator()(const Weight &w1, const Weight &w2) const {
    // NaturalLess is a negative order (opposite to normal ordering).
    // This operator () corresponds to "<" in the negative order, which
    // corresponds to the ">" in the normal order.
    return (Compare(w1, w2) == 1);
  }
};

// Make sure Compare is defined for TropicalWeight, so everything works
// if we substitute LatticeWeight for TropicalWeight.
inline int Compare(const TropicalWeight &w1,
                   const TropicalWeight &w2) {
  float f1 = w1.Value(), f2 = w2.Value();
  if (f1 == f2) return 0;
  else if (f1 > f2) return -1;
  else return 1;
}



template<class WeightType, class IntType>
inline CompactLatticeWeightTpl<WeightType, IntType> Plus(
    const CompactLatticeWeightTpl<WeightType, IntType> &w1,
    const CompactLatticeWeightTpl<WeightType, IntType> &w2) {
  return (Compare(w1, w2) >= 0 ? w1 : w2);
}

template<class WeightType, class IntType>
inline CompactLatticeWeightTpl<WeightType, IntType> Times(
    const CompactLatticeWeightTpl<WeightType, IntType> &w1,
    const CompactLatticeWeightTpl<WeightType, IntType> &w2) {
  WeightType w = Times(w1.Weight(), w2.Weight());
  if (w == WeightType::Zero()) {
    return CompactLatticeWeightTpl<WeightType, IntType>::Zero();
    // special case to ensure zero is unique
  } else {
    std::vector<IntType> v;
    v.resize(w1.String().size() + w2.String().size());
    typename std::vector<IntType>::iterator iter = v.begin();
    iter = std::copy(w1.String().begin(), w1.String().end(), iter); // returns end of first range.
    std::copy(w2.String().begin(), w2.String().end(), iter);
    return CompactLatticeWeightTpl<WeightType, IntType>(w, v);
  }
}

template<class WeightType, class IntType>
inline CompactLatticeWeightTpl<WeightType, IntType> Divide(const CompactLatticeWeightTpl<WeightType, IntType> &w1,
                                                          const CompactLatticeWeightTpl<WeightType, IntType> &w2,
                                                          DivideType div = DIVIDE_ANY) {
  if (w1.Weight() == WeightType::Zero()) {
    if (w2.Weight() != WeightType::Zero()) {
      return CompactLatticeWeightTpl<WeightType, IntType>::Zero();
    } else {
      KALDI_ERR << "Division by zero [0/0]";
    }
  } else if (w2.Weight() == WeightType::Zero()) {
    KALDI_ERR << "Error: division by zero";
  }
  WeightType w = Divide(w1.Weight(), w2.Weight());

  const std::vector<IntType> v1 = w1.String(), v2 = w2.String();
  if (v2.size() > v1.size()) {
    KALDI_ERR << "Cannot divide, length mismatch";
  }
  typename std::vector<IntType>::const_iterator v1b = v1.begin(),
      v1e = v1.end(), v2b = v2.begin(), v2e = v2.end();
  if (div == DIVIDE_LEFT) {
    if (!std::equal(v2b, v2e, v1b)) { // v2 must be identical to first part of v1.
      KALDI_ERR << "Cannot divide, data mismatch";
    }
    return CompactLatticeWeightTpl<WeightType, IntType>(
        w, std::vector<IntType>(v1b+(v2e-v2b), v1e)); // return last part of v1.
  } else if (div == DIVIDE_RIGHT) {
    if (!std::equal(v2b, v2e, v1e-(v2e-v2b))) { // v2 must be identical to last part of v1.
      KALDI_ERR << "Cannot divide, data mismatch";
    }
    return CompactLatticeWeightTpl<WeightType, IntType>(
        w, std::vector<IntType>(v1b, v1e-(v2e-v2b))); // return first part of v1.

  } else {
    KALDI_ERR << "Cannot divide CompactLatticeWeightTpl with DIVIDE_ANY";
  }
  return CompactLatticeWeightTpl<WeightType,IntType>::Zero(); // keep compiler happy.
}

template <class WeightType, class IntType>
inline std::ostream &operator <<(std::ostream &strm, const CompactLatticeWeightTpl<WeightType, IntType> &w) {
  strm << w.Weight();
  CHECK(FLAGS_fst_weight_separator.size() == 1);
  strm << FLAGS_fst_weight_separator[0]; // comma by default.
  for(size_t i = 0; i < w.String().size(); i++) {
    strm << w.String()[i];
    if (i+1 < w.String().size())
      strm << kStringSeparator; // '_'; defined in string-weight.h in OpenFst code.
  }
  return strm;
}

template <class WeightType, class IntType>
inline std::istream &operator >>(std::istream &strm, CompactLatticeWeightTpl<WeightType, IntType> &w) {
  std::string s;
  strm >> s;
  if (strm.fail()) {
    return strm;
  }
  CHECK(FLAGS_fst_weight_separator.size() == 1);
  size_t pos = s.find_last_of(FLAGS_fst_weight_separator); // normally ","
  if (pos == std::string::npos) {
    strm.clear(std::ios::badbit);
    return strm;
  }
  // get parts of str before and after the separator (default: ',');
  std::string s1(s, 0, pos), s2(s, pos+1);
  std::istringstream strm1(s1);
  WeightType weight;
  strm1 >> weight;
  w.SetWeight(weight);
  if (strm1.fail() || !strm1.eof()) {
    strm.clear(std::ios::badbit);
    return strm;
  }
  // read string part.
  std::vector<IntType> string;
  const char *c = s2.c_str();
  while(*c != '\0') {
    if (*c == kStringSeparator) // '_'
      c++;
    char *c2;
    long int i = strtol(c, &c2, 10);
    if (c2 == c || static_cast<long int>(static_cast<IntType>(i)) != i) {
      strm.clear(std::ios::badbit);
      return strm;
    }
    c = c2;
    string.push_back(static_cast<IntType>(i));
  }
  w.SetString(string);
  return strm;
}

template<class BaseWeightType, class IntType>
class CompactLatticeWeightCommonDivisorTpl {
 public:
  typedef CompactLatticeWeightTpl<BaseWeightType, IntType> Weight;

  Weight operator()(const Weight &w1, const Weight &w2) const {
    // First find longest common prefix of the strings.
    typename std::vector<IntType>::const_iterator s1b = w1.String().begin(),
        s1e = w1.String().end(), s2b = w2.String().begin(), s2e = w2.String().end();
    while (s1b < s1e && s2b < s2e && *s1b == *s2b) {
      s1b++;
      s2b++;
    }
    return Weight(Plus(w1.Weight(), w2.Weight()), std::vector<IntType>(w1.String().begin(), s1b));
  }
};

/** Scales the pair (a, b) of floating-point weights inside a
    CompactLatticeWeight by premultiplying it (viewed as a vector)
    by a 2x2 matrix "scale".
    Assumes there is a ScaleTupleWeight function that applies to "Weight";
    this currently only works if Weight equals LatticeWeightTpl<FloatType>
    for some FloatType.
*/
template<class Weight, class IntType, class ScaleFloatType>
inline CompactLatticeWeightTpl<Weight, IntType> ScaleTupleWeight(
    const CompactLatticeWeightTpl<Weight, IntType> &w,
    const std::vector<std::vector<ScaleFloatType> > &scale) {
  return CompactLatticeWeightTpl<Weight, IntType>(
      Weight(ScaleTupleWeight(w.Weight(), scale)), w.String());
}

/** Define some ConvertLatticeWeight functions that are used in various lattice
    conversions... make them all templates, some with no arguments, since some
    must be templates.*/
template<class Float1, class Float2>
inline void ConvertLatticeWeight(
    const LatticeWeightTpl<Float1> &w_in,
    LatticeWeightTpl<Float2> *w_out) {
  w_out->SetValue1(w_in.Value1());
  w_out->SetValue2(w_in.Value2());
}

template<class Float1, class Float2, class Int>
inline void ConvertLatticeWeight(
    const CompactLatticeWeightTpl<LatticeWeightTpl<Float1>, Int> &w_in,
    CompactLatticeWeightTpl<LatticeWeightTpl<Float2>, Int> *w_out) {
  LatticeWeightTpl<Float2> weight2(w_in.Weight().Value1(),
                                   w_in.Weight().Value2());
  w_out->SetWeight(weight2);
  w_out->SetString(w_in.String());
}

// to convert from Lattice to standard FST
template<class Float1, class Float2>
inline void ConvertLatticeWeight(
    const LatticeWeightTpl<Float1> &w_in,
    TropicalWeightTpl<Float2> *w_out) {
  TropicalWeightTpl<Float2> w1(w_in.Value1());
  TropicalWeightTpl<Float2> w2(w_in.Value2());
  *w_out = Times(w1, w2);
}

template<class Float>
inline double ConvertToCost(const LatticeWeightTpl<Float> &w) {
  return static_cast<double>(w.Value1()) + static_cast<double>(w.Value2());
}

template<class Float, class Int>
inline double ConvertToCost(const CompactLatticeWeightTpl<LatticeWeightTpl<Float>, Int> &w) {
  return static_cast<double>(w.Weight().Value1()) + static_cast<double>(w.Weight().Value2());
}

template<class Float>
inline double ConvertToCost(const TropicalWeightTpl<Float> &w) {
  return w.Value();
}


}  // namespace fst

#endif  // KALDI_FSTEXT_LATTICE_WEIGHT_H_
