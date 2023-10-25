// lat/arctic-weight.h

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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


#ifndef KALDI_LAT_ARCTIC_WEIGHT_H_
#define KALDI_LAT_ARCTIC_WEIGHT_H_

#include "fst/float-weight.h"

namespace fst {

// Arctic semiring: (max, +, inf, 0)
// We define the Arctic semiring T' = (R \cup {-inf, +inf}, max, +, -inf, 0).
// The term "Arctic" came from Keith Kintzley (kintzley@jhu.edu), as opposite
// to the Tropical semiring.
template <class T>
class ArcticWeightTpl : public FloatWeightTpl<T> {
 public:
  using FloatWeightTpl<T>::Value;

  typedef ArcticWeightTpl<T> ReverseWeight;

  ArcticWeightTpl() : FloatWeightTpl<T>() {}

  ArcticWeightTpl(T f) : FloatWeightTpl<T>(f) {}

  ArcticWeightTpl(const ArcticWeightTpl<T> &w) : FloatWeightTpl<T>(w) {}

  static const ArcticWeightTpl<T> Zero() {
    return ArcticWeightTpl<T>(-std::numeric_limits<T>::infinity()); }

  static const ArcticWeightTpl<T> One() {
    return ArcticWeightTpl<T>(0.0F); }

  static const std::string &Type() {
    static const std::string type = std::string("arctic") +
        FloatWeightTpl<T>::GetPrecisionString();
    return type;
  }

  static ArcticWeightTpl<T> NoWeight() {
    return ArcticWeightTpl<T>(std::numeric_limits<T>::infinity());
  }

  bool Member() const {
    // First part fails for IEEE NaN
    return Value() == Value() && Value() != std::numeric_limits<T>::infinity();
  }

  ArcticWeightTpl<T> Quantize(float delta = kDelta) const {
    if (Value() == -std::numeric_limits<T>::infinity() ||
        Value() == std::numeric_limits<T>::infinity() ||
        Value() != Value())
      return *this;
    else
      return ArcticWeightTpl<T>(floor(Value()/delta + 0.5F) * delta);
  }

  ArcticWeightTpl<T> Reverse() const { return *this; }

  static uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative |
        kPath | kIdempotent;
  }
};

// Single precision arctic weight
typedef ArcticWeightTpl<float> ArcticWeight;

template <class T>
inline ArcticWeightTpl<T> Plus(const ArcticWeightTpl<T> &w1,
                                 const ArcticWeightTpl<T> &w2) {
  return w1.Value() > w2.Value() ? w1 : w2;
}

inline ArcticWeightTpl<float> Plus(const ArcticWeightTpl<float> &w1,
                                     const ArcticWeightTpl<float> &w2) {
  return Plus<float>(w1, w2);
}

inline ArcticWeightTpl<double> Plus(const ArcticWeightTpl<double> &w1,
                                      const ArcticWeightTpl<double> &w2) {
  return Plus<double>(w1, w2);
}

template <class T>
inline ArcticWeightTpl<T> Times(const ArcticWeightTpl<T> &w1,
                                  const ArcticWeightTpl<T> &w2) {
  T f1 = w1.Value(), f2 = w2.Value();
  if (f1 == -std::numeric_limits<T>::infinity())
    return w1;
  else if (f2 == -std::numeric_limits<T>::infinity())
    return w2;
  else
    return ArcticWeightTpl<T>(f1 + f2);
}

inline ArcticWeightTpl<float> Times(const ArcticWeightTpl<float> &w1,
                                      const ArcticWeightTpl<float> &w2) {
  return Times<float>(w1, w2);
}

inline ArcticWeightTpl<double> Times(const ArcticWeightTpl<double> &w1,
                                       const ArcticWeightTpl<double> &w2) {
  return Times<double>(w1, w2);
}

template <class T>
inline ArcticWeightTpl<T> Divide(const ArcticWeightTpl<T> &w1,
                                   const ArcticWeightTpl<T> &w2,
                                   DivideType typ = DIVIDE_ANY) {
  T f1 = w1.Value(), f2 = w2.Value();
  if (f2 == -std::numeric_limits<T>::infinity())
    return std::numeric_limits<T>::quiet_NaN();
  else if (f1 == -std::numeric_limits<T>::infinity())
    return -std::numeric_limits<T>::infinity();
  else
    return ArcticWeightTpl<T>(f1 - f2);
}

inline ArcticWeightTpl<float> Divide(const ArcticWeightTpl<float> &w1,
                                       const ArcticWeightTpl<float> &w2,
                                       DivideType typ = DIVIDE_ANY) {
  return Divide<float>(w1, w2, typ);
}

inline ArcticWeightTpl<double> Divide(const ArcticWeightTpl<double> &w1,
                                        const ArcticWeightTpl<double> &w2,
                                        DivideType typ = DIVIDE_ANY) {
  return Divide<double>(w1, w2, typ);
}



} // namespace fst

#endif  // KALDI_LAT_ARCTIC_WEIGHT_H_
