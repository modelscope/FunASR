// util/text-utils.h

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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

#ifndef KALDI_UTIL_TEXT_UTILS_H_
#define KALDI_UTIL_TEXT_UTILS_H_

#include <errno.h>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <limits>
#include "base/kaldi-common.h"


namespace kaldi {

/// Split a string using any of the single character delimiters.
/// If omit_empty_strings == true, the output will contain any
/// nonempty strings after splitting on any of the
/// characters in the delimiter.  If omit_empty_strings == false,
/// the output will contain n+1 strings if there are n characters
/// in the set "delim" within the input string.  In this case
/// the empty string is split to a single empty string.
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);

/// Joins the elements of a vector of strings into a single string using
/// "delim" as the delimiter. If omit_empty_strings == true, any empty strings
/// in the vector are skipped. A vector of empty strings results in an empty
/// string on the output.
void JoinVectorToString(const std::vector<std::string> &vec_in,
                        const char *delim, bool omit_empty_strings,
                        std::string *str_out);

/**
  \brief Split a string (e.g. 1:2:3) into a vector of integers.

  \param [in]  delim  String containing a list of characters, any of which
                      is allowed as a delimiter.
  \param [in] omit_empty_strings If true, empty strings between delimiters are
                      allowed and will not produce an output integer; if false,
                      instances of characters in 'delim' that are consecutive or
                      at the start or end of the string would be an error.
                      You'll normally want this to be true if 'delim' consists
                      of spaces, and false otherwise.
  \param [out] out   The output list of integers.
*/
template<class I>
bool SplitStringToIntegers(const std::string &full,
                           const char *delim,
                           bool omit_empty_strings,  // typically false [but
                                                     // should probably be true
                                                     // if "delim" is spaces].
                           std::vector<I> *out) {
  KALDI_ASSERT(out != NULL);
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  if (*(full.c_str()) == '\0') {
    out->clear();
    return true;
  }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty_strings, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    const char *this_str = split[i].c_str();
    char *end = NULL;
    int64 j = 0;
    j = KALDI_STRTOLL(this_str, &end);
    if (end == this_str || *end != '\0') {
      out->clear();
      return false;
    } else {
      I jI = static_cast<I>(j);
      if (static_cast<int64>(jI) != j) {
        // output type cannot fit this integer.
        out->clear();
        return false;
      }
      (*out)[i] = jI;
    }
  }
  return true;
}

// This is defined for F = float and double.
template<class F>
bool SplitStringToFloats(const std::string &full,
                         const char *delim,
                         bool omit_empty_strings,  // typically false
                         std::vector<F> *out);


/// Converts a string into an integer via strtoll and returns false if there was
/// any kind of problem (i.e. the string was not an integer or contained extra
/// non-whitespace junk, or the integer was too large to fit into the type it is
/// being converted into).  Only sets *out if everything was OK and it returns
/// true.
template<class Int>
bool ConvertStringToInteger(const std::string &str,
                            Int *out) {
  KALDI_ASSERT_IS_INTEGER_TYPE(Int);
  const char *this_str = str.c_str();
  char *end = NULL;
  errno = 0;
  int64 i = KALDI_STRTOLL(this_str, &end);
  if (end != this_str)
    while (isspace(*end)) end++;
  if (end == this_str || *end != '\0' || errno != 0)
    return false;
  Int iInt = static_cast<Int>(i);
  if (static_cast<int64>(iInt) != i ||
      (i < 0 && !std::numeric_limits<Int>::is_signed)) {
    return false;
  }
  *out = iInt;
  return true;
}


/// ConvertStringToReal converts a string into either float or double
/// and returns false if there was any kind of problem (i.e. the string
/// was not a floating point number or contained extra non-whitespace junk).
/// Be careful- this function will successfully read inf's or nan's.
template <typename T>
bool ConvertStringToReal(const std::string &str,
                         T *out);

/// Removes the beginning and trailing whitespaces from a string
void Trim(std::string *str);


/// Removes leading and trailing white space from the string, then splits on the
/// first section of whitespace found (if present), putting the part before the
/// whitespace in "first" and the rest in "rest".  If there is no such space,
/// everything that remains after removing leading and trailing whitespace goes
/// in "first".
void SplitStringOnFirstSpace(const std::string &line,
                             std::string *first,
                             std::string *rest);


/// Returns true if "token" is nonempty, and all characters are
/// printable and whitespace-free.
bool IsToken(const std::string &token);


/// Returns true if "line" is free of \n characters and unprintable
/// characters, and does not contain leading or trailing whitespace.
bool IsLine(const std::string &line);



/**
   This function returns true when two text strings are approximately equal, and
   false when they are not.  The definition of 'equal' is normal string
   equality, except that two substrings like "0.31134" and "0.311341" would be
   considered equal.  'decimal_places_tolerance' controls how many digits after
   the '.' have to match up.
   E.g. StringsApproxEqual("hello 0.23 there", "hello 0.24 there", 2) would
   return false because there is a difference in the 2nd decimal, but with
   an argument of 1 it would return true.
 */
bool StringsApproxEqual(const std::string &a,
                        const std::string &b,
                        int32 decimal_places_check = 2);

/**
   This class is responsible for parsing input like
    hi-there xx=yyy a=b c empty= f-oo=Append(bar, sss) ba_z=123 bing='a b c' baz="a b c d='a b' e"
   and giving you access to the fields, in this case

   FirstToken() == "hi-there", and key->value pairs:

   xx->yyy, a->"b c", empty->"", f-oo->"Append(bar, sss)", ba_z->"123",
   bing->"a b c", baz->"a b c d='a b' e"

   The first token is optional, if the line started with a key-value pair then
   FirstValue() will be empty.

   Note: it can parse value fields with space inside them only if they are free of the '='
   character.  If values are going to contain the '=' character, you need to quote them
   with either single or double quotes.

   Key values may contain -_a-zA-Z0-9, but must begin with a-zA-Z_.
 */
class ConfigLine {
 public:
  // Tries to parse the line as a config-file line.  Returns false
  // if it could not for some reason, e.g. parsing failure.  In most cases
  // prints no warnings; the user should do this.  Does not expect comments.
  bool ParseLine(const std::string &line);

  // the GetValue functions are overloaded for various types.  They return true
  // if the key exists with value that can be converted to that type, and false
  // otherwise.  They also mark the key-value pair as having been read.  It is
  // not an error to read values twice.
  bool GetValue(const std::string &key, std::string *value);
  bool GetValue(const std::string &key, BaseFloat *value);
  bool GetValue(const std::string &key, int32 *value);
  // Values may be separated by ":" or by ",".
  bool GetValue(const std::string &key, std::vector<int32> *value);
  bool GetValue(const std::string &key, bool *value);

  bool HasUnusedValues() const;
  /// returns e.g. foo=bar xxx=yyy if foo and xxx were not consumed by one
  /// of the GetValue() functions.
  std::string UnusedValues() const;

  const std::string &FirstToken() const { return first_token_; }

  const std::string WholeLine() { return whole_line_; }
  // use default assignment operator and copy constructor.
 private:
  std::string whole_line_;
  // the first token of the line, e.g. if line is
  // foo-bar baz=bing
  // then first_token_ would be "foo-bar".
  std::string first_token_;

  // data_ maps from key to (value, is-this-value-consumed?).
  std::map<std::string, std::pair<std::string, bool> > data_;

};

/// This function is like ExpectToken but for two tokens, and it will either
/// accept token1 and then token2, or just token2.  This is useful in Read
/// functions where the first token may already have been consumed.
void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                          const std::string &token1,
                          const std::string &token2);


/**
   This function reads in a config file and *appends* its contents to a vector of
   lines; it is responsible for removing comments (anything after '#') and
   stripping out any lines that contain only whitespace after comment removal.
 */
void ReadConfigLines(std::istream &is,
                     std::vector<std::string> *lines);


/**
   This function converts config-lines from a simple sequence of strings
   as output by ReadConfigLines(), into a sequence of first-tokens and
   name-value pairs.  The general format is:
      "command-type bar=baz xx=yyy"
   etc., although there are subtleties as to what exactly is allowed, see
   documentation for class ConfigLine for details.
   This function will die if there was a parsing failure.
 */
void ParseConfigLines(const std::vector<std::string> &lines,
                      std::vector<ConfigLine> *config_lines);


/// Returns true if 'name' would be a valid name for a component or node in a
/// nnet3Nnet.  This is a nonempty string beginning with A-Za-z_, and containing only
/// '-', '_', '.', A-Z, a-z, or 0-9.
bool IsValidName(const std::string &name);

}  // namespace kaldi

#endif  // KALDI_UTIL_TEXT_UTILS_H_
