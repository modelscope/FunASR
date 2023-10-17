// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Float weight set and associated semiring operation definitions.

#ifndef FST_FLOAT_WEIGHT_H_
#define FST_FLOAT_WEIGHT_H_

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include <fst/util.h>
#include <fst/weight.h>


namespace fst {

// Numeric limits class.
template <class T>
class FloatLimits {
 public:
  static constexpr T PosInfinity() {
    return std::numeric_limits<T>::infinity();
  }

  static constexpr T NegInfinity() { return -PosInfinity(); }

  static constexpr T NumberBad() { return std::numeric_limits<T>::quiet_NaN(); }
};

// Weight class to be templated on floating-points types.
template <class T = float>
class FloatWeightTpl {
 public:
  using ValueType = T;

  FloatWeightTpl() noexcept {}

  constexpr FloatWeightTpl(T f) : value_(f) {}  // NOLINT

  // TODO(mjansche): Leave implicit once Android NDK r18 is the default.
  FloatWeightTpl(const FloatWeightTpl<T> &) = default;
  FloatWeightTpl(FloatWeightTpl<T> &&) noexcept = default;

  FloatWeightTpl<T> &operator=(const FloatWeightTpl<T> &) = default;
  FloatWeightTpl<T> &operator=(FloatWeightTpl<T> &&) noexcept = default;

  std::istream &Read(std::istream &strm) { return ReadType(strm, &value_); }

  std::ostream &Write(std::ostream &strm) const {
    return WriteType(strm, value_);
  }

  size_t Hash() const {
    size_t hash = 0;
    // Avoid using union, which would be undefined behavior.
    // Use memcpy, similar to bit_cast, but sizes may be different.
    // This should be optimized into a single move instruction by
    // any reasonable compiler.
    std::memcpy(&hash, &value_, std::min(sizeof(hash), sizeof(value_)));
    return hash;
  }

  constexpr const T &Value() const { return value_; }

 protected:
  void SetValue(const T &f) { value_ = f; }

  static constexpr const char *GetPrecisionString() {
    return sizeof(T) == 4 ? ""
        : sizeof(T) == 1 ? "8"
        : sizeof(T) == 2 ? "16"
        : sizeof(T) == 8 ? "64"
        : "unknown";
  }

 private:
  T value_;
};

// Single-precision float weight.
using FloatWeight = FloatWeightTpl<float>;

template <class T>
constexpr bool operator==(const FloatWeightTpl<T> &w1,
                          const FloatWeightTpl<T> &w2) {
#if (defined(__i386__) || defined(__x86_64__)) && !defined(__SSE2_MATH__)
// With i387 instructions, excess precision on a weight in an 80-bit
// register may cause it to compare unequal to that same weight when
// stored to memory.  This breaks =='s reflexivity, in turn breaking
// NaturalLess.
#error "Please compile with -msse -mfpmath=sse, or equivalent."
#endif
  return w1.Value() == w2.Value();
}

// These seemingly unnecessary overloads are actually needed to make
// comparisons like FloatWeightTpl<float> == float compile.  If only the
// templated version exists, the FloatWeightTpl<float>(float) conversion
// won't be found.
constexpr bool operator==(const FloatWeightTpl<float> &w1,
                          const FloatWeightTpl<float> &w2) {
  return operator==<float>(w1, w2);
}

constexpr bool operator==(const FloatWeightTpl<double> &w1,
                          const FloatWeightTpl<double> &w2) {
  return operator==<double>(w1, w2);
}

template <class T>
constexpr bool operator!=(const FloatWeightTpl<T> &w1,
                          const FloatWeightTpl<T> &w2) {
  return !(w1 == w2);
}

constexpr bool operator!=(const FloatWeightTpl<float> &w1,
                          const FloatWeightTpl<float> &w2) {
  return operator!=<float>(w1, w2);
}

constexpr bool operator!=(const FloatWeightTpl<double> &w1,
                          const FloatWeightTpl<double> &w2) {
  return operator!=<double>(w1, w2);
}

template <class T>
constexpr bool FloatApproxEqual(T w1, T w2, float delta = kDelta) {
  return w1 <= w2 + delta && w2 <= w1 + delta;
}

template <class T>
constexpr bool ApproxEqual(const FloatWeightTpl<T> &w1,
                           const FloatWeightTpl<T> &w2, float delta = kDelta) {
  return FloatApproxEqual(w1.Value(), w2.Value(), delta);
}

template <class T>
inline std::ostream &operator<<(std::ostream &strm,
                                const FloatWeightTpl<T> &w) {
  if (w.Value() == FloatLimits<T>::PosInfinity()) {
    return strm << "Infinity";
  } else if (w.Value() == FloatLimits<T>::NegInfinity()) {
    return strm << "-Infinity";
  } else if (w.Value() != w.Value()) {  // Fails for IEEE NaN.
    return strm << "BadNumber";
  } else {
    return strm << w.Value();
  }
}

template <class T>
inline std::istream &operator>>(std::istream &strm, FloatWeightTpl<T> &w) {
  string s;
  strm >> s;
  if (s == "Infinity") {
    w = FloatWeightTpl<T>(FloatLimits<T>::PosInfinity());
  } else if (s == "-Infinity") {
    w = FloatWeightTpl<T>(FloatLimits<T>::NegInfinity());
  } else {
    char *p;
    T f = strtod(s.c_str(), &p);
    if (p < s.c_str() + s.size()) {
      strm.clear(std::ios::badbit);
    } else {
      w = FloatWeightTpl<T>(f);
    }
  }
  return strm;
}

// Tropical semiring: (min, +, inf, 0).
template <class T>
class TropicalWeightTpl : public FloatWeightTpl<T> {
 public:
  using typename FloatWeightTpl<T>::ValueType;
  using FloatWeightTpl<T>::Value;
  using ReverseWeight = TropicalWeightTpl<T>;
  using Limits = FloatLimits<T>;

  TropicalWeightTpl() noexcept : FloatWeightTpl<T>() {}

  constexpr TropicalWeightTpl(T f) : FloatWeightTpl<T>(f) {}

  static constexpr TropicalWeightTpl<T> Zero() { return Limits::PosInfinity(); }

  static constexpr TropicalWeightTpl<T> One() { return 0; }

  static constexpr TropicalWeightTpl<T> NoWeight() {
    return Limits::NumberBad();
  }

  static const string &Type() {
    static const string *const type =
        new string(string("tropical") +
                   FloatWeightTpl<T>::GetPrecisionString());
    return *type;
  }

  constexpr bool Member() const {
    // All floating point values except for NaNs and negative infinity are valid
    // tropical weights.
    //
    // Testing membership of a given value can be done by simply checking that
    // it is strictly greater than negative infinity, which fails for negative
    // infinity itself but also for NaNs. This can usually be accomplished in a
    // single instruction (such as *UCOMI* on x86) without branching logic.
    //
    // An additional wrinkle involves constexpr correctness of floating point
    // comparisons against NaN. GCC is uneven when it comes to which expressions
    // it considers compile-time constants. In particular, current versions of
    // GCC do not always consider (nan < inf) to be a constant expression, but
    // do consider (inf < nan) to be a constant expression. (See
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88173 and
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88683 for details.) In order
    // to allow Member() to be a constexpr function accepted by GCC, we write
    // the comparison here as (-inf < v).
    return Limits::NegInfinity() < Value();
  }

  TropicalWeightTpl<T> Quantize(float delta = kDelta) const {
    if (!Member() || Value() == Limits::PosInfinity()) {
      return *this;
    } else {
      return TropicalWeightTpl<T>(std::floor(Value() / delta + 0.5F) * delta);
    }
  }

  constexpr TropicalWeightTpl<T> Reverse() const { return *this; }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative | kPath | kIdempotent;
  }
};

// Single precision tropical weight.
using TropicalWeight = TropicalWeightTpl<float>;

template <class T>
constexpr TropicalWeightTpl<T> Plus(const TropicalWeightTpl<T> &w1,
                                    const TropicalWeightTpl<T> &w2) {
  return (!w1.Member() || !w2.Member()) ? TropicalWeightTpl<T>::NoWeight()
      : w1.Value() < w2.Value() ? w1 : w2;
}

// See comment at operator==(FloatWeightTpl<float>, FloatWeightTpl<float>)
// for why these overloads are present.
constexpr TropicalWeightTpl<float> Plus(const TropicalWeightTpl<float> &w1,
                                        const TropicalWeightTpl<float> &w2) {
  return Plus<float>(w1, w2);
}

constexpr TropicalWeightTpl<double> Plus(const TropicalWeightTpl<double> &w1,
                                         const TropicalWeightTpl<double> &w2) {
  return Plus<double>(w1, w2);
}

template <class T>
constexpr TropicalWeightTpl<T> Times(const TropicalWeightTpl<T> &w1,
                                     const TropicalWeightTpl<T> &w2) {
  // The following is safe in the context of the Tropical (and Log) semiring
  // for all IEEE floating point values, including infinities and NaNs,
  // because:
  //
  // * If one or both of the floating point Values is NaN and hence not a
  //   Member, the result of addition below is NaN, so the result is not a
  //   Member. This supersedes all other cases, so we only consider non-NaN
  //   values next.
  //
  // * If both Values are finite, there is no issue.
  //
  // * If one of the Values is infinite, or if both are infinities with the
  //   same sign, the result of floating point addition is the same infinity,
  //   so there is no issue.
  //
  // * If both of the Values are infinities with opposite signs, the result of
  //   adding IEEE floating point -inf + inf is NaN and hence not a Member. But
  //   since -inf was not a Member to begin with, returning a non-Member result
  //   is fine as well.
  return TropicalWeightTpl<T>(w1.Value() + w2.Value());
}

constexpr TropicalWeightTpl<float> Times(const TropicalWeightTpl<float> &w1,
                                         const TropicalWeightTpl<float> &w2) {
  return Times<float>(w1, w2);
}

constexpr TropicalWeightTpl<double> Times(const TropicalWeightTpl<double> &w1,
                                          const TropicalWeightTpl<double> &w2) {
  return Times<double>(w1, w2);
}

template <class T>
constexpr TropicalWeightTpl<T> Divide(const TropicalWeightTpl<T> &w1,
                                      const TropicalWeightTpl<T> &w2,
                                      DivideType typ = DIVIDE_ANY) {
  // The following is safe in the context of the Tropical (and Log) semiring
  // for all IEEE floating point values, including infinities and NaNs,
  // because:
  //
  // * If one or both of the floating point Values is NaN and hence not a
  //   Member, the result of subtraction below is NaN, so the result is not a
  //   Member. This supersedes all other cases, so we only consider non-NaN
  //   values next.
  //
  // * If both Values are finite, there is no issue.
  //
  // * If w2.Value() is -inf (and hence w2 is not a Member), the result of ?:
  //   below is NoWeight, which is not a Member.
  //
  //   Whereas in IEEE floating point semantics 0/inf == 0, this does not carry
  //   over to this semiring (since TropicalWeight(-inf) would be the analogue
  //   of floating point inf) and instead Divide(Zero(), TropicalWeight(-inf))
  //   is NoWeight().
  //
  // * If w2.Value() is inf (and hence w2 is Zero), the resulting floating
  //   point value is either NaN (if w1 is Zero or if w1.Value() is NaN) and
  //   hence not a Member, or it is -inf and hence not a Member; either way,
  //   division by Zero results in a non-Member result.
  using Weight = TropicalWeightTpl<T>;
  return w2.Member() ? Weight(w1.Value() - w2.Value()) : Weight::NoWeight();
}

constexpr TropicalWeightTpl<float> Divide(const TropicalWeightTpl<float> &w1,
                                          const TropicalWeightTpl<float> &w2,
                                          DivideType typ = DIVIDE_ANY) {
  return Divide<float>(w1, w2, typ);
}

constexpr TropicalWeightTpl<double> Divide(const TropicalWeightTpl<double> &w1,
                                           const TropicalWeightTpl<double> &w2,
                                           DivideType typ = DIVIDE_ANY) {
  return Divide<double>(w1, w2, typ);
}

// Power(w, n) calculates the n-th power of w with respect to semiring Times.
//
// In the case of the Tropical (and Log) semiring, the exponent n is not
// restricted to be an integer. It can be a floating point value, for example.
//
// In weight.h, a narrower and hence more broadly applicable version of
// Power(w, n) is defined for arbitrary weight types and non-negative integer
// exponents n (of type size_t) and implemented in terms of repeated
// multiplication using Times.
//
// Without further provisions this means that, when an expression such as
//
//   Power(TropicalWeightTpl<float>::One(), static_cast<size_t>(2))
//
// is specified, the overload of Power() is ambiguous. The template function
// below could be instantiated as
//
//   Power<float, size_t>(const TropicalWeightTpl<float> &, size_t)
//
// and the template function defined in weight.h (further specialized below)
// could be instantiated as
//
//   Power<TropicalWeightTpl<float>>(const TropicalWeightTpl<float> &, size_t)
//
// That would lead to two definitions with identical signatures, which results
// in a compilation error. To avoid that, we hide the definition of Power<T, V>
// when V is size_t, so only Power<W> is visible. Power<W> is further
// specialized to Power<TropicalWeightTpl<...>>, and the overloaded definition
// of Power<T, V> is made conditionally available only to that template
// specialization.

template <class T, class V,
          bool Enable = !std::is_same<V, size_t>::value,
          typename std::enable_if<Enable>::type * = nullptr>
constexpr TropicalWeightTpl<T> Power(const TropicalWeightTpl<T> &w, V n) {
  using Weight = TropicalWeightTpl<T>;
  return (!w.Member() || n != n) ? Weight::NoWeight()
      : (n == 0 || w == Weight::One()) ? Weight::One()
      : Weight(w.Value() * n);
}

// Specializes the library-wide template to use the above implementation; rules
// of function template instantiation require this be a full instantiation.

template <>
constexpr TropicalWeightTpl<float> Power<TropicalWeightTpl<float>>(
    const TropicalWeightTpl<float> &weight, size_t n) {
  return Power<float, size_t, true>(weight, n);
}

template <>
constexpr TropicalWeightTpl<double> Power<TropicalWeightTpl<double>>(
    const TropicalWeightTpl<double> &weight, size_t n) {
  return Power<double, size_t, true>(weight, n);
}


// Log semiring: (log(e^-x + e^-y), +, inf, 0).
template <class T>
class LogWeightTpl : public FloatWeightTpl<T> {
 public:
  using typename FloatWeightTpl<T>::ValueType;
  using FloatWeightTpl<T>::Value;
  using ReverseWeight = LogWeightTpl;
  using Limits = FloatLimits<T>;

  LogWeightTpl() noexcept : FloatWeightTpl<T>() {}

  constexpr LogWeightTpl(T f) : FloatWeightTpl<T>(f) {}

  static constexpr LogWeightTpl Zero() { return Limits::PosInfinity(); }

  static constexpr LogWeightTpl One() { return 0; }

  static constexpr LogWeightTpl NoWeight() { return Limits::NumberBad(); }

  static const string &Type() {
    static const string *const type =
        new string(string("log") + FloatWeightTpl<T>::GetPrecisionString());
    return *type;
  }

  constexpr bool Member() const {
    // The comments for TropicalWeightTpl<>::Member() apply here unchanged.
    return Limits::NegInfinity() < Value();
  }

  LogWeightTpl<T> Quantize(float delta = kDelta) const {
    if (!Member() || Value() == Limits::PosInfinity()) {
      return *this;
    } else {
      return LogWeightTpl<T>(std::floor(Value() / delta + 0.5F) * delta);
    }
  }

  constexpr LogWeightTpl<T> Reverse() const { return *this; }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative;
  }
};

// Single-precision log weight.
using LogWeight = LogWeightTpl<float>;

// Double-precision log weight.
using Log64Weight = LogWeightTpl<double>;

namespace internal {

// -log(e^-x + e^-y) = x - LogPosExp(y - x), assuming x >= 0.0.
inline double LogPosExp(double x) {
  DCHECK(!(x < 0));  // NB: NaN values are allowed.
  return log1p(exp(-x));
}

// -log(e^-x - e^-y) = x - LogNegExp(y - x), assuming x > 0.0.
inline double LogNegExp(double x) {
  DCHECK_GT(x, 0);
  return log1p(-exp(-x));
}

// a +_log b = -log(e^-a + e^-b) = KahanLogSum(a, b, ...).
// Kahan compensated summation provides an error bound that is
// independent of the number of addends. Assumes b >= a;
// c is the compensation.
inline double KahanLogSum(double a, double b, double *c) {
  DCHECK_GE(b, a);
  double y = -LogPosExp(b - a) - *c;
  double t = a + y;
  *c = (t - a) - y;
  return t;
}

// a -_log b = -log(e^-a - e^-b) = KahanLogDiff(a, b, ...).
// Kahan compensated summation provides an error bound that is
// independent of the number of addends. Assumes b > a;
// c is the compensation.
inline double KahanLogDiff(double a, double b, double *c) {
  DCHECK_GT(b, a);
  double y = -LogNegExp(b - a) - *c;
  double t = a + y;
  *c = (t - a) - y;
  return t;
}

}  // namespace internal

template <class T>
inline LogWeightTpl<T> Plus(const LogWeightTpl<T> &w1,
                            const LogWeightTpl<T> &w2) {
  using Limits = FloatLimits<T>;
  const T f1 = w1.Value();
  const T f2 = w2.Value();
  if (f1 == Limits::PosInfinity()) {
    return w2;
  } else if (f2 == Limits::PosInfinity()) {
    return w1;
  } else if (f1 > f2) {
    return LogWeightTpl<T>(f2 - internal::LogPosExp(f1 - f2));
  } else {
    return LogWeightTpl<T>(f1 - internal::LogPosExp(f2 - f1));
  }
}

inline LogWeightTpl<float> Plus(const LogWeightTpl<float> &w1,
                                const LogWeightTpl<float> &w2) {
  return Plus<float>(w1, w2);
}

inline LogWeightTpl<double> Plus(const LogWeightTpl<double> &w1,
                                 const LogWeightTpl<double> &w2) {
  return Plus<double>(w1, w2);
}

template <class T>
constexpr LogWeightTpl<T> Times(const LogWeightTpl<T> &w1,
                                const LogWeightTpl<T> &w2) {
  // The comments for Times(Tropical...) above apply here unchanged.
  return LogWeightTpl<T>(w1.Value() + w2.Value());
}

constexpr LogWeightTpl<float> Times(const LogWeightTpl<float> &w1,
                                    const LogWeightTpl<float> &w2) {
  return Times<float>(w1, w2);
}

constexpr LogWeightTpl<double> Times(const LogWeightTpl<double> &w1,
                                     const LogWeightTpl<double> &w2) {
  return Times<double>(w1, w2);
}

template <class T>
constexpr LogWeightTpl<T> Divide(const LogWeightTpl<T> &w1,
                                 const LogWeightTpl<T> &w2,
                                 DivideType typ = DIVIDE_ANY) {
  // The comments for Divide(Tropical...) above apply here unchanged.
  using Weight = LogWeightTpl<T>;
  return w2.Member() ? Weight(w1.Value() - w2.Value()) : Weight::NoWeight();
}

constexpr LogWeightTpl<float> Divide(const LogWeightTpl<float> &w1,
                                     const LogWeightTpl<float> &w2,
                                     DivideType typ = DIVIDE_ANY) {
  return Divide<float>(w1, w2, typ);
}

constexpr LogWeightTpl<double> Divide(const LogWeightTpl<double> &w1,
                                      const LogWeightTpl<double> &w2,
                                      DivideType typ = DIVIDE_ANY) {
  return Divide<double>(w1, w2, typ);
}

// The comments for Power<>(Tropical...) above apply here unchanged.

template <class T, class V,
          bool Enable = !std::is_same<V, size_t>::value,
          typename std::enable_if<Enable>::type * = nullptr>
constexpr LogWeightTpl<T> Power(const LogWeightTpl<T> &w, V n) {
  using Weight = LogWeightTpl<T>;
  return (!w.Member() || n != n) ? Weight::NoWeight()
      : (n == 0 || w == Weight::One()) ? Weight::One()
      : Weight(w.Value() * n);
}

// Specializes the library-wide template to use the above implementation; rules
// of function template instantiation require this be a full instantiation.

template <>
constexpr LogWeightTpl<float> Power<LogWeightTpl<float>>(
    const LogWeightTpl<float> &weight, size_t n) {
  return Power<float, size_t, true>(weight, n);
}

template <>
constexpr LogWeightTpl<double> Power<LogWeightTpl<double>>(
    const LogWeightTpl<double> &weight, size_t n) {
  return Power<double, size_t, true>(weight, n);
}

// Specialization using the Kahan compensated summation.
template <class T>
class Adder<LogWeightTpl<T>> {
 public:
  using Weight = LogWeightTpl<T>;

  explicit Adder(Weight w = Weight::Zero())
      : sum_(w.Value()),
        c_(0.0) { }

  Weight Add(const Weight &w) {
    using Limits = FloatLimits<T>;
    const T f = w.Value();
    if (f == Limits::PosInfinity()) {
      return Sum();
    } else if (sum_ == Limits::PosInfinity()) {
      sum_ = f;
      c_ = 0.0;
    } else if (f > sum_) {
      sum_ = internal::KahanLogSum(sum_, f, &c_);
    } else {
      sum_ = internal::KahanLogSum(f, sum_, &c_);
    }
    return Sum();
  }

  Weight Sum() { return Weight(sum_); }

  void Reset(Weight w = Weight::Zero()) {
    sum_ = w.Value();
    c_ = 0.0;
  }

 private:
  double sum_;
  double c_;   // Kahan compensation.
};

// MinMax semiring: (min, max, inf, -inf).
template <class T>
class MinMaxWeightTpl : public FloatWeightTpl<T> {
 public:
  using typename FloatWeightTpl<T>::ValueType;
  using FloatWeightTpl<T>::Value;
  using ReverseWeight = MinMaxWeightTpl<T>;
  using Limits = FloatLimits<T>;

  MinMaxWeightTpl() noexcept : FloatWeightTpl<T>() {}

  constexpr MinMaxWeightTpl(T f) : FloatWeightTpl<T>(f) {}  // NOLINT

  static constexpr MinMaxWeightTpl Zero() { return Limits::PosInfinity(); }

  static constexpr MinMaxWeightTpl One() { return Limits::NegInfinity(); }

  static constexpr MinMaxWeightTpl NoWeight() { return Limits::NumberBad(); }

  static const string &Type() {
    static const string *const type =
        new string(string("minmax") + FloatWeightTpl<T>::GetPrecisionString());
    return *type;
  }

  // Fails for IEEE NaN.
  constexpr bool Member() const { return Value() == Value(); }

  MinMaxWeightTpl<T> Quantize(float delta = kDelta) const {
    // If one of infinities, or a NaN.
    if (!Member() ||
        Value() == Limits::NegInfinity() || Value() == Limits::PosInfinity()) {
      return *this;
    } else {
      return MinMaxWeightTpl<T>(std::floor(Value() / delta + 0.5F) * delta);
    }
  }

  constexpr MinMaxWeightTpl<T> Reverse() const { return *this; }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative | kIdempotent | kPath;
  }
};

// Single-precision min-max weight.
using MinMaxWeight = MinMaxWeightTpl<float>;

// Min.
template <class T>
constexpr MinMaxWeightTpl<T> Plus(const MinMaxWeightTpl<T> &w1,
                               const MinMaxWeightTpl<T> &w2) {
  return (!w1.Member() || !w2.Member()) ? MinMaxWeightTpl<T>::NoWeight()
      : w1.Value() < w2.Value() ? w1 : w2;
}

constexpr MinMaxWeightTpl<float> Plus(const MinMaxWeightTpl<float> &w1,
                                      const MinMaxWeightTpl<float> &w2) {
  return Plus<float>(w1, w2);
}

constexpr MinMaxWeightTpl<double> Plus(const MinMaxWeightTpl<double> &w1,
                                       const MinMaxWeightTpl<double> &w2) {
  return Plus<double>(w1, w2);
}

// Max.
template <class T>
constexpr MinMaxWeightTpl<T> Times(const MinMaxWeightTpl<T> &w1,
                                   const MinMaxWeightTpl<T> &w2) {
  return (!w1.Member() || !w2.Member()) ? MinMaxWeightTpl<T>::NoWeight()
      : w1.Value() >= w2.Value() ? w1 : w2;
}

constexpr MinMaxWeightTpl<float> Times(const MinMaxWeightTpl<float> &w1,
                                       const MinMaxWeightTpl<float> &w2) {
  return Times<float>(w1, w2);
}

constexpr MinMaxWeightTpl<double> Times(const MinMaxWeightTpl<double> &w1,
                                        const MinMaxWeightTpl<double> &w2) {
  return Times<double>(w1, w2);
}

// Defined only for special cases.
template <class T>
constexpr MinMaxWeightTpl<T> Divide(const MinMaxWeightTpl<T> &w1,
                                    const MinMaxWeightTpl<T> &w2,
                                    DivideType typ = DIVIDE_ANY) {
  return w1.Value() >= w2.Value() ? w1 : MinMaxWeightTpl<T>::NoWeight();
}

constexpr MinMaxWeightTpl<float> Divide(const MinMaxWeightTpl<float> &w1,
                                        const MinMaxWeightTpl<float> &w2,
                                        DivideType typ = DIVIDE_ANY) {
  return Divide<float>(w1, w2, typ);
}

constexpr MinMaxWeightTpl<double> Divide(const MinMaxWeightTpl<double> &w1,
                                         const MinMaxWeightTpl<double> &w2,
                                         DivideType typ = DIVIDE_ANY) {
  return Divide<double>(w1, w2, typ);
}

// Converts to tropical.
template <>
struct WeightConvert<LogWeight, TropicalWeight> {
  constexpr TropicalWeight operator()(const LogWeight &w) const {
    return w.Value();
  }
};

template <>
struct WeightConvert<Log64Weight, TropicalWeight> {
  constexpr TropicalWeight operator()(const Log64Weight &w) const {
    return w.Value();
  }
};

// Converts to log.
template <>
struct WeightConvert<TropicalWeight, LogWeight> {
  constexpr LogWeight operator()(const TropicalWeight &w) const {
    return w.Value();
  }
};

template <>
struct WeightConvert<Log64Weight, LogWeight> {
  constexpr LogWeight operator()(const Log64Weight &w) const {
    return w.Value();
  }
};

// Converts to log64.
template <>
struct WeightConvert<TropicalWeight, Log64Weight> {
  constexpr Log64Weight operator()(const TropicalWeight &w) const {
    return w.Value();
  }
};

template <>
struct WeightConvert<LogWeight, Log64Weight> {
  constexpr Log64Weight operator()(const LogWeight &w) const {
    return w.Value();
  }
};

// This function object returns random integers chosen from [0,
// num_random_weights). The boolean 'allow_zero' determines whether Zero() and
// zero divisors should be returned in the random weight generation. This is
// intended primary for testing.
template <class Weight>
class FloatWeightGenerate {
 public:
  explicit FloatWeightGenerate(
      bool allow_zero = true,
      const size_t num_random_weights = kNumRandomWeights)
      : allow_zero_(allow_zero), num_random_weights_(num_random_weights) {}

  Weight operator()() const {
    const int n = rand() % (num_random_weights_ + allow_zero_);  // NOLINT
    if (allow_zero_ && n == num_random_weights_) return Weight::Zero();
    return Weight(n);
  }

 private:
  // Permits Zero() and zero divisors.
  const bool allow_zero_;
  // Number of alternative random weights.
  const size_t num_random_weights_;
};

template <class T>
class WeightGenerate<TropicalWeightTpl<T>>
    : public FloatWeightGenerate<TropicalWeightTpl<T>> {
 public:
  using Weight = TropicalWeightTpl<T>;
  using Generate = FloatWeightGenerate<Weight>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : Generate(allow_zero, num_random_weights) {}

  Weight operator()() const { return Weight(Generate::operator()()); }
};

template <class T>
class WeightGenerate<LogWeightTpl<T>>
    : public FloatWeightGenerate<LogWeightTpl<T>> {
 public:
  using Weight = LogWeightTpl<T>;
  using Generate = FloatWeightGenerate<Weight>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : Generate(allow_zero, num_random_weights) {}

  Weight operator()() const { return Weight(Generate::operator()()); }
};

// This function object returns random integers chosen from [0,
// num_random_weights). The boolean 'allow_zero' determines whether Zero() and
// zero divisors should be returned in the random weight generation. This is
// intended primary for testing.
template <class T>
class WeightGenerate<MinMaxWeightTpl<T>> {
 public:
  using Weight = MinMaxWeightTpl<T>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : allow_zero_(allow_zero), num_random_weights_(num_random_weights) {}

  Weight operator()() const {
    const int n = (rand() %  // NOLINT
                   (2 * num_random_weights_ + allow_zero_)) -
                  num_random_weights_;
    if (allow_zero_ && n == num_random_weights_) {
      return Weight::Zero();
    } else if (n == -num_random_weights_) {
      return Weight::One();
    } else {
      return Weight(n);
    }
  }

 private:
  // Permits Zero() and zero divisors.
  const bool allow_zero_;
  // Number of alternative random weights.
  const size_t num_random_weights_;
};

}  // namespace fst

#endif  // FST_FLOAT_WEIGHT_H_
