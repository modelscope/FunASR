// util/text-utils.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "util/text-utils.h"
#include <limits>
#include <map>
#include <algorithm>
#include "base/kaldi-common.h"

namespace kaldi {


template<class F>
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out) {
  KALDI_ASSERT(out != NULL);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    F f = 0;
    if (!ConvertStringToReal(split[i], &f))
      return false;
    (*out)[i] = f;
  }
  return true;
}

// Instantiate the template above for float and double.
template
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,
                         std::vector<float> *out);
template
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,
                         std::vector<double> *out);

void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out) {
  size_t start = 0, found = 0, end = full.size();
  out->clear();
  while (found != std::string::npos) {
    found = full.find_first_of(delim, start);
    // start != end condition is for when the delimiter is at the end
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}

void JoinVectorToString(const std::vector<std::string> &vec_in,
                        const char *delim, bool omit_empty_strings,
                        std::string *str_out) {
  std::string tmp_str;
  for (size_t i = 0; i < vec_in.size(); i++) {
    if (!omit_empty_strings || !vec_in[i].empty()) {
      tmp_str.append(vec_in[i]);
      if (i < vec_in.size() - 1)
        if (!omit_empty_strings || !vec_in[i+1].empty())
          tmp_str.append(delim);
    }
  }
  str_out->swap(tmp_str);
}

void Trim(std::string *str) {
  const char *white_chars = " \t\n\r\f\v";

  std::string::size_type pos = str->find_last_not_of(white_chars);
  if (pos != std::string::npos)  {
    str->erase(pos + 1);
    pos = str->find_first_not_of(white_chars);
    if (pos != std::string::npos) str->erase(0, pos);
  } else {
    str->erase(str->begin(), str->end());
  }
}

bool IsToken(const std::string &token) {
  size_t l = token.length();
  if (l == 0) return false;
  for (size_t i = 0; i < l; i++) {
    unsigned char c = token[i];
    if ((!isprint(c) || isspace(c)) && (isascii(c) || c == (unsigned char)255))
      return false;
    // The "&& (isascii(c) || c == 255)" was added so that we won't reject
    // non-ASCII characters such as French characters with accents [except for
    // 255 which is "nbsp", a form of space].
  }
  return true;
}


void SplitStringOnFirstSpace(const std::string &str,
                             std::string *first,
                             std::string *rest) {
  const char *white_chars = " \t\n\r\f\v";
  typedef std::string::size_type I;
  const I npos = std::string::npos;
  I first_nonwhite = str.find_first_not_of(white_chars);
  if (first_nonwhite == npos) {
    first->clear();
    rest->clear();
    return;
  }
  // next_white is first whitespace after first nonwhitespace.
  I next_white = str.find_first_of(white_chars, first_nonwhite);

  if (next_white == npos) {  // no more whitespace...
    *first = std::string(str, first_nonwhite);
    rest->clear();
    return;
  }
  I next_nonwhite = str.find_first_not_of(white_chars, next_white);
  if (next_nonwhite == npos) {
    *first = std::string(str, first_nonwhite, next_white-first_nonwhite);
    rest->clear();
    return;
  }

  I last_nonwhite = str.find_last_not_of(white_chars);
  KALDI_ASSERT(last_nonwhite != npos);  // or coding error.

  *first = std::string(str, first_nonwhite, next_white-first_nonwhite);
  *rest = std::string(str, next_nonwhite, last_nonwhite+1-next_nonwhite);
}

bool IsLine(const std::string &line) {
  if (line.find('\n') != std::string::npos) return false;
  if (line.empty()) return true;
  if (isspace(*(line.begin()))) return false;
  if (isspace(*(line.rbegin()))) return false;
  std::string::const_iterator iter = line.begin(), end = line.end();
  for (; iter != end; iter++)
    if (!isprint(*iter)) return false;
  return true;
}

template <class T>
class NumberIstream{
 public:
  explicit NumberIstream(std::istream &i) : in_(i) {}

  NumberIstream & operator >> (T &x) {
    if (!in_.good()) return *this;
    in_ >> x;
    if (!in_.fail() && RemainderIsOnlySpaces()) return *this;
    return ParseOnFail(&x);
  }

 private:
  std::istream &in_;

  bool RemainderIsOnlySpaces() {
    if (in_.tellg() != std::istream::pos_type(-1)) {
      std::string rem;
      in_ >> rem;

      if (rem.find_first_not_of(' ') != std::string::npos) {
        // there is not only spaces
        return false;
      }
    }

    in_.clear();
    return true;
  }

  NumberIstream & ParseOnFail(T *x) {
    std::string str;
    in_.clear();
    in_.seekg(0);
    // If the stream is broken even before trying
    // to read from it or if there are many tokens,
    // it's pointless to try.
    if (!(in_ >> str) || !RemainderIsOnlySpaces()) {
      in_.setstate(std::ios_base::failbit);
      return *this;
    }

    std::map<std::string, T> inf_nan_map;
    // we'll keep just uppercase values.
    inf_nan_map["INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INF"] = - std::numeric_limits<T>::infinity();
    inf_nan_map["INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["+INFINITY"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-INFINITY"] = - std::numeric_limits<T>::infinity();
    inf_nan_map["NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["+NAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-NAN"] = - std::numeric_limits<T>::quiet_NaN();
    // MSVC
    inf_nan_map["1.#INF"] = std::numeric_limits<T>::infinity();
    inf_nan_map["-1.#INF"] = - std::numeric_limits<T>::infinity();
    inf_nan_map["1.#QNAN"] = std::numeric_limits<T>::quiet_NaN();
    inf_nan_map["-1.#QNAN"] = - std::numeric_limits<T>::quiet_NaN();

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);

    if (inf_nan_map.find(str) != inf_nan_map.end()) {
      *x = inf_nan_map[str];
    } else {
      in_.setstate(std::ios_base::failbit);
    }

    return *this;
  }
};


template <typename T>
bool ConvertStringToReal(const std::string &str,
                         T *out) {
  std::istringstream iss(str);

  NumberIstream<T> i(iss);

  i >> *out;

  if (iss.fail()) {
    // Number conversion failed.
    return false;
  }

  return true;
}

template
bool ConvertStringToReal(const std::string &str,
                         float *out);
template
bool ConvertStringToReal(const std::string &str,
                         double *out);



/*
  This function is a helper function of StringsApproxEqual.  It should be
  thought of as a recursive function-- it was designed that way-- but rather
  than actually recursing (which would cause problems with stack overflow), we
  just set the args and return to the start.

  The 'decimal_places_tolerance' argument is just passed in from outside,
  see the documentation for StringsApproxEqual in text-utils.h to see an
  explanation.  The argument 'places_into_number' provides some information
  about the strings 'a' and 'b' that precedes the current pointers.
  For purposes of this comment, let's define the 'decimal' of a number
  as the part that comes after the decimal point, e.g. in '99.123',
  '123' would be the decimal.  If 'places_into_number' is -1, it means
  we're not currently inside some place like that (i.e. it's not the
  case that we're pointing to the '1' or the '2' or the '3').
  If it's 0, then we'd be pointing to the first place after the decimal,
  '1' in this case.  Note if one of the numbers is shorter than the
  other, like '99.123' versus '99.1234' and 'a' points to the first '3'
  while 'b' points to the second '4', 'places_into_number' referes to the
  shorter of the two, i.e. it would be 2 in this example.


 */
bool StringsApproxEqualInternal(const char *a, const char *b,
                                int32 decimal_places_tolerance,
                                int32 places_into_number) {
start:
  char ca = *a, cb = *b;
  if (ca == cb) {
    if (ca == '\0') {
      return true;
    } else {
      if (places_into_number >= 0) {
        if (isdigit(ca)) {
          places_into_number++;
        } else {
          places_into_number = -1;
        }
      } else {
        if (ca == '.') {
          places_into_number = 0;
        }
      }
      a++;
      b++;
      goto start;
    }
  } else {
    if (places_into_number  >= decimal_places_tolerance &&
        (isdigit(ca) || isdigit(cb))) {
      // we're potentially willing to accept this difference between the
      // strings.
      if (isdigit(ca)) a++;
      if (isdigit(cb)) b++;
      // we'll have advanced at least one of the two strings.
      goto start;
    } else if (places_into_number >= 0 &&
               ((ca == '0' && !isdigit(cb)) || (cb == '0' && !isdigit(ca)))) {
      // this clause is designed to ensure that, for example,
      // "0.1" would count the same as "0.100001".
      if (ca == '0') a++;
      else b++;
      places_into_number++;
      goto start;
    } else {
      return false;
    }
  }

}


bool StringsApproxEqual(const std::string &a,
                        const std::string &b,
                        int32 decimal_places_tolerance) {
  return StringsApproxEqualInternal(a.c_str(), b.c_str(),
                                    decimal_places_tolerance, -1);
}


}  // end namespace kaldi
