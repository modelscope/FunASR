// Copyright (c) 2003, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <glog/logging.h>
#include <glog/stl_logging.h>

#include <functional>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "config.h"
#include "googletest.h"

using namespace std;

struct user_hash {
  size_t operator()(int x) const { return static_cast<size_t>(x); }
};

static void TestSTLLogging() {
  {
    // Test a sequence.
    vector<int> v;
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    ostringstream ss;
    ss << v;
    EXPECT_EQ(ss.str(), "10 20 30");
    vector<int> copied_v(v);
    CHECK_EQ(v, copied_v);  // This must compile.
  }

  {
    // Test a sorted pair associative container.
    map< int, string > m;
    m[20] = "twenty";
    m[10] = "ten";
    m[30] = "thirty";
    ostringstream ss;
    ss << m;
    EXPECT_EQ(ss.str(), "(10, ten) (20, twenty) (30, thirty)");
    map< int, string > copied_m(m);
    CHECK_EQ(m, copied_m);  // This must compile.
  }

  {
    // Test a long sequence.
    vector<int> v;
    string expected;
    for (int i = 0; i < 100; i++) {
      v.push_back(i);
      if (i > 0) expected += ' ';
      const size_t buf_size = 256;
      char buf[buf_size];
      snprintf(buf, buf_size, "%d", i);
      expected += buf;
    }
    v.push_back(100);
    expected += " ...";
    ostringstream ss;
    ss << v;
    CHECK_EQ(ss.str(), expected.c_str());
  }

  {
    // Test a sorted pair associative container.
    // Use a non-default comparison functor.
    map<int, string, greater<> > m;
    m[20] = "twenty";
    m[10] = "ten";
    m[30] = "thirty";
    ostringstream ss;
    ss << m;
    EXPECT_EQ(ss.str(), "(30, thirty) (20, twenty) (10, ten)");
    map<int, string, greater<> > copied_m(m);
    CHECK_EQ(m, copied_m);  // This must compile.
  }
}

int main(int, char**) {
  TestSTLLogging();
  std::cout << "PASS\n";
  return 0;
}
