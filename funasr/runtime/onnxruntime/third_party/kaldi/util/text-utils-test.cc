// util/text-utils-test.cc

// Copyright 2009-2011     Microsoft Corporation
//                2017     Johns Hopkins University (author: Daniel Povey)
//                2015  Vimal Manohar (Johns Hopkins University)

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


#include "base/kaldi-common.h"
#include "util/text-utils.h"

namespace kaldi {

char GetRandChar() {
  return static_cast<char>(32 + Rand() % 95);  // between ' ' and '~'
}

const char *ws_delim = " \t\n\r";
char GetRandDelim() {
  if (Rand() % 2 == 0)
    return static_cast<char>(33 + Rand() % 94);  // between '!' and '~';
  else
    return ws_delim[Rand() % 4];
}


void TestSplitStringToVector() {
  // srand((unsigned int)time(NULL));
  // didn't compile on cygwin.

  {
    std::vector<std::string> str_vec;
    SplitStringToVector("", " ", false, &str_vec);
    KALDI_ASSERT(str_vec.size() == 1);  // If this fails it may just mean
    // that someone changed the
    // semantics of SplitStringToVector in a reasonable way.
    SplitStringToVector("", " ", true, &str_vec);
    KALDI_ASSERT(str_vec.empty());
  }
  for (int j = 0; j < 100; j++) {
    std::vector<std::string> str_vec;
    int sz = Rand() % 73;
    std::string full;
    for (int i = 0; i < sz-1; i++) {
      full.push_back((Rand() % 7 == 0)? GetRandDelim() : GetRandChar());
    }
    std::string delim;
    delim.push_back(GetRandDelim());
    bool omit_empty_strings = (Rand() %2 == 0)? true : false;
    SplitStringToVector(full, delim.c_str(), omit_empty_strings, &str_vec);
    std::string new_full;
    for (size_t i = 0; i < str_vec.size(); i++) {
      if (omit_empty_strings) KALDI_ASSERT(str_vec[i] != "");
      new_full.append(str_vec[i]);
      if (i < str_vec.size() -1) new_full.append(delim);
    }
    std::string new_full2;
    JoinVectorToString(str_vec, delim.c_str(), omit_empty_strings, &new_full2);
    if (omit_empty_strings) {  // sequences of delimiters cannot be matched
      size_t start = full.find_first_not_of(delim),
          end = full.find_last_not_of(delim);
      if (start == std::string::npos) {  // only delimiters
        KALDI_ASSERT(end == std::string::npos);
      } else {
        std::string full_test;
        char last = '\0';
        for (size_t i = start; i <= end; i++) {
          if (full[i] != last || last != *delim.c_str())
            full_test.push_back(full[i]);
          last = full[i];
        }
        if (!full.empty()) {
          KALDI_ASSERT(new_full.compare(full_test) == 0);
          KALDI_ASSERT(new_full2.compare(full_test) == 0);
        }
      }
    } else if (!full.empty()) {
      KALDI_ASSERT(new_full.compare(full) == 0);
      KALDI_ASSERT(new_full2.compare(full) == 0);
    }
  }
}

void TestSplitStringToIntegers() {
  {
    std::vector<int32> v;
    KALDI_ASSERT(SplitStringToIntegers("-1:2:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToIntegers("-1:2:4:", ":", false, &v) == false);
    KALDI_ASSERT(SplitStringToIntegers(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToIntegers("-1\n2\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToIntegers(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    KALDI_ASSERT(SplitStringToIntegers("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<uint32> v;
    KALDI_ASSERT(SplitStringToIntegers("-1:2:4", ":", false, &v) == false);
    // cannot put negative number in uint32.
  }
}



void TestSplitStringToFloats() {
  {
    std::vector<float> v;
    KALDI_ASSERT(SplitStringToFloats("-1:2.5:4", ":", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2.5 && v[2] == 4);
    KALDI_ASSERT(SplitStringToFloats("-1:2.5:4:", ":", false, &v) == false);
    KALDI_ASSERT(SplitStringToFloats(":-1::2:4:", ":", true, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2 && v[2] == 4);
    KALDI_ASSERT(SplitStringToFloats("-1\n2.5\t4", " \n\t\r", false, &v) == true
           && v.size() == 3 && v[0] == -1 && v[1] == 2.5 && v[2] == 4);
    KALDI_ASSERT(SplitStringToFloats(" ", " \n\t\r", true, &v) == true
           && v.size() == 0);
    KALDI_ASSERT(SplitStringToFloats("", " \n\t\r", false, &v) == true
           && v.size() == 0);
  }

  {
    std::vector<double> v;
    KALDI_ASSERT(SplitStringToFloats("-1:2:4", ":", false, &v) == true);
  }
}

void TestConvertStringToInteger() {
  int32 i;
  KALDI_ASSERT(ConvertStringToInteger("12345", &i) && i == 12345);
  KALDI_ASSERT(ConvertStringToInteger("-12345", &i) && i == -12345);
  char j;
  KALDI_ASSERT(!ConvertStringToInteger("-12345", &j));  // too big for char.

  KALDI_ASSERT(ConvertStringToInteger(" -12345 ", &i));  // whitespace accepted

  KALDI_ASSERT(!ConvertStringToInteger("a ", &i));  // non-integers rejected.

  KALDI_ASSERT(ConvertStringToInteger("0", &i) && i == 0);

  uint64 k;
  KALDI_ASSERT(ConvertStringToInteger("12345", &k) && k == 12345);
  KALDI_ASSERT(!ConvertStringToInteger("-12345", &k));  // unsigned,
                                                        // cannot convert.
}

template<class Real>
void TestConvertStringToReal() {
  Real d;
  KALDI_ASSERT(ConvertStringToReal("1", &d) && d == 1.0);
  KALDI_ASSERT(ConvertStringToReal("-1", &d) && d == -1.0);
  KALDI_ASSERT(ConvertStringToReal("-1", &d) && d == -1.0);
  KALDI_ASSERT(ConvertStringToReal(" -1 ", &d) && d == -1.0);
  KALDI_ASSERT(!ConvertStringToReal("-1 x", &d));
  KALDI_ASSERT(!ConvertStringToReal("-1f", &d));
  KALDI_ASSERT(ConvertStringToReal("12345.2", &d) && fabs(d-12345.2) < 1.0);
  KALDI_ASSERT(ConvertStringToReal("1.0e+08", &d) && fabs(d-1.0e+08) < 100.0);

  // it also works for inf or nan.
  KALDI_ASSERT(ConvertStringToReal("inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(" inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("inf ", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(" inf ", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("+inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-inf", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("Inf", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("INF", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("InF", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("infinity", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-infinity", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE inf", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGEinf", &d));
  KALDI_ASSERT(!ConvertStringToReal("infGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("inf_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("inf GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE infinity", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGEinfinity", &d));
  KALDI_ASSERT(!ConvertStringToReal("infinityGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("infinity_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("infinity GARBAGE", &d));
  KALDI_ASSERT(ConvertStringToReal("1.#INF", &d) && d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-1.#INF", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal("-1.#INF  ", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(" -1.#INF ", &d) && d < 0 && d - d != 0);
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE 1.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE1.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("2.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("-2.#INF", &d));
  KALDI_ASSERT(!ConvertStringToReal("1.#INFGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("1.#INF_GARBAGE", &d));

  KALDI_ASSERT(ConvertStringToReal("nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("+nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("-nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("Nan", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("NAN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("NaN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal(" NaN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("NaN ", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal(" NaN ", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("1.#QNAN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("-1.#QNAN", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal("1.#QNAN  ", &d) && d != d);
  KALDI_ASSERT(ConvertStringToReal(" 1.#QNAN ", &d) && d != d);
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE nan", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGEnan", &d));
  KALDI_ASSERT(!ConvertStringToReal("nanGARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("nan_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("nan GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE 1.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("GARBAGE1.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("2.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("-2.#QNAN", &d));
  KALDI_ASSERT(!ConvertStringToReal("-1.#QNAN_GARBAGE", &d));
  KALDI_ASSERT(!ConvertStringToReal("-1.#QNANGARBAGE", &d));
}

template<class Real>
void TestNan() {
  Real d;
  KALDI_ASSERT(ConvertStringToReal(std::to_string(sqrt(-1)), &d) && d != d);
}

template<class Real>
void TestInf() {
  Real d;
  KALDI_ASSERT(ConvertStringToReal(std::to_string(exp(10000)), &d) &&
               d > 0 && d - d != 0);
  KALDI_ASSERT(ConvertStringToReal(std::to_string(-exp(10000)), &d) &&
               d < 0 && d - d != 0);
}


std::string TrimTmp(std::string s) {
  Trim(&s);
  return s;
}

void TestTrim() {
  KALDI_ASSERT(TrimTmp(" a ") == "a");
  KALDI_ASSERT(TrimTmp(" a b  c") == "a b  c");
  KALDI_ASSERT(TrimTmp("") == "");
  KALDI_ASSERT(TrimTmp("X\n") == "X");
  KALDI_ASSERT(TrimTmp("X\n\t") == "X");
  KALDI_ASSERT(TrimTmp("\n\tX") == "X");
}  // end namespace kaldi


void TestSplitStringOnFirstSpace() {
  std::string a, b;
  SplitStringOnFirstSpace("a b", &a, &b);
  KALDI_ASSERT(a == "a" && b == "b");
  SplitStringOnFirstSpace("aa bb", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb");
  SplitStringOnFirstSpace("aa", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "");
  SplitStringOnFirstSpace(" aa \n\t ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "");
  SplitStringOnFirstSpace("  \n\t ", &a, &b);
  KALDI_ASSERT(a == "" && b == "");
  SplitStringOnFirstSpace(" aa   bb \n\t ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb");
  SplitStringOnFirstSpace(" aa   bb cc ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb cc");
  SplitStringOnFirstSpace(" aa   bb  cc ", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb  cc");
  SplitStringOnFirstSpace(" aa   bb  cc", &a, &b);
  KALDI_ASSERT(a == "aa" && b == "bb  cc");
}

void TestIsToken() {
  KALDI_ASSERT(IsToken("a"));
  KALDI_ASSERT(IsToken("ab"));
  KALDI_ASSERT(!IsToken("ab "));
  KALDI_ASSERT(!IsToken(" ab"));
  KALDI_ASSERT(!IsToken("a b"));
  KALDI_ASSERT(IsToken("\231"));  // typical non-ASCII printable character,
                                  // something with an accent.
  KALDI_ASSERT(!IsToken("\377"));  // character 255, which is a form of space.
  KALDI_ASSERT(IsToken("a-b,c,d=ef"));
  KALDI_ASSERT(!IsToken("a\nb"));
  KALDI_ASSERT(!IsToken("a\tb"));
  KALDI_ASSERT(!IsToken("ab\t"));
  KALDI_ASSERT(!IsToken(""));
}

void TestIsLine() {
  KALDI_ASSERT(IsLine("a"));
  KALDI_ASSERT(IsLine("a b"));
  KALDI_ASSERT(!IsLine("a\nb"));
  KALDI_ASSERT(!IsLine("a b "));
  KALDI_ASSERT(!IsLine(" a b"));
}


void TestStringsApproxEqual() {
  // we must test the test.
  KALDI_ASSERT(!StringsApproxEqual("a", "b"));
  KALDI_ASSERT(!StringsApproxEqual("1", "2"));
  KALDI_ASSERT(StringsApproxEqual("1.234", "1.235", 2));
  KALDI_ASSERT(!StringsApproxEqual("1.234", "1.235", 3));
  KALDI_ASSERT(StringsApproxEqual("x 1.234 y", "x 1.2345 y", 3));
  KALDI_ASSERT(!StringsApproxEqual("x 1.234 y", "x 1.2345 y", 4));
  KALDI_ASSERT(StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.49", 1));
  KALDI_ASSERT(!StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.49", 2));
  KALDI_ASSERT(StringsApproxEqual("x 1.234 y 6.41", "x 1.235 y 6.411", 2));
  KALDI_ASSERT(StringsApproxEqual("x 1.0 y", "x 1.0001 y", 3));
  KALDI_ASSERT(!StringsApproxEqual("x 1.0 y", "x 1.0001 y", 4));
}

void UnitTestConfigLineParse() {
  std::string str;
  {
    ConfigLine cfl;
    str = "a-b xx=yyy foo=bar  baz=123 ba=1:2";
    bool status = cfl.ParseLine(str);
    KALDI_ASSERT(status && cfl.FirstToken() == "a-b");

    KALDI_ASSERT(cfl.HasUnusedValues());
    std::string str_value;
    KALDI_ASSERT(cfl.GetValue("xx", &str_value));
    KALDI_ASSERT(str_value == "yyy");
    KALDI_ASSERT(cfl.HasUnusedValues());
    KALDI_ASSERT(cfl.GetValue("foo", &str_value));
    KALDI_ASSERT(str_value == "bar");
    KALDI_ASSERT(cfl.HasUnusedValues());
    KALDI_ASSERT(!cfl.GetValue("xy", &str_value));
    KALDI_ASSERT(cfl.GetValue("baz", &str_value));
    KALDI_ASSERT(str_value == "123");

    std::vector<int32> int_values;
    KALDI_ASSERT(!cfl.GetValue("xx", &int_values));
    KALDI_ASSERT(cfl.GetValue("baz", &int_values));
    KALDI_ASSERT(cfl.HasUnusedValues());
    KALDI_ASSERT(int_values.size() == 1 && int_values[0] == 123);
    KALDI_ASSERT(cfl.GetValue("ba", &int_values));
    KALDI_ASSERT(int_values.size() == 2 && int_values[0] == 1 && int_values[1] == 2);
    KALDI_ASSERT(!cfl.HasUnusedValues());
  }

  {
    ConfigLine cfl;
    str = "a-b baz=x y z pp = qq ab =cd ac= bd";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "a-b baz=x y z pp = qq ab=cd ac=bd";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "foo-bar";
    KALDI_ASSERT(cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "foo-bar a=b c d f=g";
    std::string value;
    KALDI_ASSERT(cfl.ParseLine(str) && cfl.FirstToken() == "foo-bar" &&
                 cfl.GetValue("a", &value)  && value == "b c d" &&
                 cfl.GetValue("f", &value) && value == "g" &&
                 !cfl.HasUnusedValues());
  }
  {
    ConfigLine cfl;
    str = "zzz a=b baz";
    KALDI_ASSERT(cfl.ParseLine(str) && cfl.FirstToken() == "zzz" &&
                 cfl.UnusedValues() == "a=b baz");
  }
  {
    ConfigLine cfl;
    str = "xxx a=b baz ";
    KALDI_ASSERT(cfl.ParseLine(str) && cfl.UnusedValues() == "a=b baz");
  }
  {
    ConfigLine cfl;
    str = "xxx a=b =c";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xxx baz='x y z' pp=qq ab=cd ac=bd";
    KALDI_ASSERT(cfl.ParseLine(str) && cfl.FirstToken() == "xxx");
    std::string str_value;
    KALDI_ASSERT(cfl.GetValue("baz", &str_value));
    KALDI_ASSERT(str_value == "x y z");
    KALDI_ASSERT(cfl.GetValue("pp", &str_value));
    KALDI_ASSERT(str_value == "qq");
    KALDI_ASSERT(cfl.UnusedValues() == "ab=cd ac=bd");
    KALDI_ASSERT(cfl.GetValue("ab", &str_value));
    KALDI_ASSERT(str_value == "cd");
    KALDI_ASSERT(cfl.UnusedValues() == "ac=bd");
    KALDI_ASSERT(cfl.HasUnusedValues());
    KALDI_ASSERT(cfl.GetValue("ac", &str_value));
    KALDI_ASSERT(str_value == "bd");
    KALDI_ASSERT(!cfl.HasUnusedValues());
  }

  {
    ConfigLine cfl;
    str = "x baz= pp = qq flag=t ";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = " x baz= pp=qq flag=t  ";
    KALDI_ASSERT(cfl.ParseLine(str) && cfl.FirstToken() == "x");

    std::string str_value;
    KALDI_ASSERT(cfl.GetValue("baz", &str_value));
    KALDI_ASSERT(str_value == "");
    KALDI_ASSERT(cfl.GetValue("pp", &str_value));
    KALDI_ASSERT(str_value == "qq");
    KALDI_ASSERT(cfl.HasUnusedValues());
    KALDI_ASSERT(cfl.GetValue("flag", &str_value));
    KALDI_ASSERT(str_value == "t");
    KALDI_ASSERT(!cfl.HasUnusedValues());

    bool bool_value = false;
    KALDI_ASSERT(cfl.GetValue("flag", &bool_value));
    KALDI_ASSERT(bool_value);
  }

  {
    ConfigLine cfl;
    str = "xx _baz=a -pp=qq";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xx 0baz=a pp=qq";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xx -baz=a pp=qq";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = "xx _baz'=a pp=qq";
    KALDI_ASSERT(!cfl.ParseLine(str));
  }
  {
    ConfigLine cfl;
    str = " baz=g";
    KALDI_ASSERT(cfl.ParseLine(str) && cfl.FirstToken() == "");
    bool flag;
    KALDI_ASSERT(!cfl.GetValue("baz", &flag));
  }
  {
    ConfigLine cfl;
    str = "xx _baz1=a pp=qq";
    KALDI_ASSERT(cfl.ParseLine(str));

    std::string str_value;
    KALDI_ASSERT(cfl.GetValue("_baz1", &str_value));
  }
}

void UnitTestReadConfig() {
  std::string str = "a-b alpha=aa beta=\"b b\"# String test\n"
      "a-b beta2='b c' beta3=bd # \n"
      "a-b gamma=1:2:3:4  # Int Vector test\n"
      " a-b de1ta=f  # Bool + Integer in key Comment test delta=t  \n"
      "a-b _epsilon=-1  # Int Vector test _epsilon=1 \n"
      "a-b zet-_a=0.15   theta=1.1# Float, -, _ test\n"
      "a-b quoted='a b c' # quoted string\n"
      "a-b quoted2=\"d e 'a b=c' f\" # string quoted with double quotes";

  std::istringstream is(str);
  std::vector<std::string> lines;
  ReadConfigLines(is, &lines);
  KALDI_ASSERT(lines.size() == 8);

  ConfigLine cfl;
  for (size_t i = 0; i < lines.size(); i++) {
    KALDI_ASSERT(cfl.ParseLine(lines[i]) && cfl.FirstToken() == "a-b");
    if (i == 1) {
        KALDI_ASSERT(cfl.GetValue("beta2", &str) && str == "b c");
    }
    if (i == 4) {
      KALDI_ASSERT(cfl.GetValue("_epsilon", &str) && str == "-1");
    }
    if (i == 5) {
      BaseFloat float_val = 0;
      KALDI_ASSERT(cfl.GetValue("zet-_a", &float_val) && ApproxEqual(float_val, 0.15));
    }
    if (i == 6) {
      KALDI_ASSERT(cfl.GetValue("quoted", &str) && str == "a b c");
    }
    if (i == 7) {
      KALDI_ASSERT(cfl.GetValue("quoted2", &str) && str == "d e 'a b=c' f");
    }
  }
}

}  // end namespace kaldi

int main() {
  using namespace kaldi;
  TestSplitStringToVector();
  TestSplitStringToIntegers();
  TestSplitStringToFloats();
  TestConvertStringToInteger();
  TestConvertStringToReal<float>();
  TestConvertStringToReal<double>();
  TestTrim();
  TestSplitStringOnFirstSpace();
  TestIsToken();
  TestIsLine();
  TestStringsApproxEqual();
  TestNan<float>();
  TestNan<double>();
  TestInf<float>();
  TestInf<double>();
  UnitTestConfigLineParse();
  UnitTestReadConfig();
  std::cout << "Test OK\n";
}
