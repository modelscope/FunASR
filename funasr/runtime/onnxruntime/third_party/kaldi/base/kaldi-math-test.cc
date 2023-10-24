// base/kaldi-math-test.cc
//
// Copyright 2009-2011  Microsoft Corporation;  Yanmin Qian;  Jan Silovsky

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
#include "base/kaldi-math.h"
#include <limits>
#include "base/timer.h"

namespace kaldi {

template<class I> void UnitTestGcdLcmTpl() {
  for (I a = 1; a < 15; a++) {  // a is min gcd.
    I b = (I)(Rand() % 10);
    I c = (I)(Rand() % 10);
    if (std::numeric_limits<I>::is_signed) {
      if (Rand() % 2 == 0) b = -b;
      if (Rand() % 2 == 0) c = -c;
    }
    if (b == 0 && c == 0) continue;  // gcd not defined for such numbers.
    I g = Gcd(b*a, c*a);
    KALDI_ASSERT(g >= a);
    KALDI_ASSERT((b*a) % g == 0);
    KALDI_ASSERT((c*a) % g == 0);

    // test least common multiple
    if (b <= 0 || c <= 0) continue;  // lcm not defined unless both positive.
    I h = Lcm(b*a, c*a);
    KALDI_ASSERT(h != 0 && (h % (b*a)) == 0 &&
                 (h % (c*a)) == 0);
  }
}

void UnitTestRoundUpToNearestPowerOfTwo() {
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(1) == 1);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(2) == 2);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(3) == 4);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(4) == 4);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(7) == 8);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(8) == 8);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(255) == 256);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(256) == 256);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(257) == 512);
  KALDI_ASSERT(RoundUpToNearestPowerOfTwo(1073700000) == 1073741824);
}

void UnitTestDivideRoundingDown() {
  for (int32 i = 0; i < 100; i++) {
    int32 a = RandInt(-100, 100);
    int32 b = 0;
    while (b == 0)
      b = RandInt(-100, 100);
    KALDI_ASSERT(DivideRoundingDown(a, b) ==
        std::floor(static_cast<double>(a) / static_cast<double>(b)));
  }
}

void UnitTestGcdLcm() {
  UnitTestGcdLcmTpl<int>();
  UnitTestGcdLcmTpl<size_t>();
  UnitTestGcdLcmTpl<int16>();
}

void UnitTestRand() {
  // Testing random-number generation.
  std::cout << "Testing random-number generation.  "
            << "If there is an error this may not terminate.\n";
  std::cout << "If this does not terminate, look more closely.  "
            << "There might be a problem [but might not be]\n";
  for (int i = 1; i < 10; i++) {
    {  // test RandUniform.
      std::cout << "Test RandUniform\n";
      KALDI_ASSERT(RandUniform() >= 0 && RandUniform() <= 1);
      float sum = RandUniform()-0.5;
      for (int j = 0; ; j++) {
        sum += RandUniform()-0.5;
        if (std::abs(sum) < 0.5*sqrt(static_cast<double>(j))) break;
      }
    }
    {  // test RandGauss.
      float sum = RandGauss();
      for (int j = 0; ; j++) {
        sum += RandGauss();
        if (std::abs(sum) < 0.5*sqrt(static_cast<double>(j))) break;
      }
    }
    {  // test RandGauss.
      float sum = RandGauss();
      for (int j = 0; ; j++) {
        float a, b;
        RandGauss2(&a, &b);
        if (i % 2 == 0) sum += a;
        else
          sum += b;
        if (std::abs(sum) < 0.5*sqrt(static_cast<double>(j))) break;
      }
    }
    {  // test poisson_Rand().
      KALDI_ASSERT(RandPoisson(3.0) >= 0);
      KALDI_ASSERT(RandPoisson(0.0) == 0);
      std::cout << "Test RandPoisson\n";
      float lambda = RandUniform() * 3.0;  // between 0 and 3.
      double sum = RandPoisson(lambda) - lambda;  // expected value is zero.
      for (int j = 0; ; j++) {
        sum += RandPoisson(lambda) - lambda;
        if (std::abs(sum) < 0.5*sqrt(static_cast<double>(j))) break;
      }
    }

    { // test WithProb().
      for (int32 i = 0; i < 10; i++) {
        KALDI_ASSERT((WithProb(0.0) == false) && (WithProb(1.0) == true));
      }
      {
        int32 tot = 0, n = 10000;
        BaseFloat p = 0.5;
        for (int32 i = 0; i < n; i++)
          tot += WithProb(p);
        KALDI_ASSERT(tot > (n * p * 0.8) && tot < (n * p * 1.2));
      }
      {
        int32 tot = 0, n = 10000;
        BaseFloat p = 0.25;
        for (int32 i = 0; i < n; i++)
          tot += WithProb(p);
        KALDI_ASSERT(tot > (n * p * 0.8) && tot < (n * p * 1.2));
      }
    }
    {  // test RandInt().
      KALDI_ASSERT(RandInt(0, 3) >= 0 && RandInt(0, 3) <= 3);

      std::cout << "Test RandInt\n";
      int minint = Rand() % 200;
      int maxint = minint + 1 + Rand()  % 20;

      float sum = RandInt(minint, maxint) +  0.5*(minint+maxint);
      for (int j = 0; ; j++) {
        sum += RandInt(minint, maxint) - 0.5*(minint+maxint);
        if (std::abs(static_cast<float>(sum)) <
            0.5*sqrt(static_cast<double>(j))*(maxint-minint)) break;
      }
    }
    { // test RandPrune in basic way.
      KALDI_ASSERT(RandPrune(1.1, 1.0) == 1.1);
      KALDI_ASSERT(RandPrune(0.0, 0.0) == 0.0);
      KALDI_ASSERT(RandPrune(-1.1, 1.0) == -1.1);
      KALDI_ASSERT(RandPrune(0.0, 1.0) == 0.0);
      KALDI_ASSERT(RandPrune(0.5, 1.0) >= 0.0);
      KALDI_ASSERT(RandPrune(-0.5, 1.0) <= 0.0);
      BaseFloat f = RandPrune(-0.5, 1.0);
      KALDI_ASSERT(f == 0.0 || f == -1.0);
      f = RandPrune(0.5, 1.0);
      KALDI_ASSERT(f == 0.0 || f == 1.0);
    }
  }
}

void UnitTestLogAddSub() {
  for (int i = 0; i < 100; i++) {
    double f1 = Rand() % 10000, f2 = Rand() % 20;
    double add1 = Exp(LogAdd(Log(f1), Log(f2)));
    double add2 = Exp(LogAdd(Log(f2), Log(f1)));
    double add = f1 + f2, thresh = add*0.00001;
    KALDI_ASSERT(std::abs(add-add1) < thresh && std::abs(add-add2) < thresh);


    try {
      double f2_check = Exp(LogSub(Log(add), Log(f1))),
             thresh = (f2*0.01)+0.001;
      KALDI_ASSERT(std::abs(f2_check-f2) < thresh);
    } catch(...) {
      KALDI_ASSERT(f2 == 0);  // It will probably crash for f2=0.
    }
  }
}

void UnitTestDefines() {  // Yes, we even unit-test the preprocessor statements.
  KALDI_ASSERT(Exp(kLogZeroFloat) == 0.0);
  KALDI_ASSERT(Exp(kLogZeroDouble) == 0.0);
  BaseFloat den = 0.0;
  KALDI_ASSERT(KALDI_ISNAN(0.0 / den));
  KALDI_ASSERT(!KALDI_ISINF(0.0 / den));
  KALDI_ASSERT(!KALDI_ISFINITE(0.0 / den));
  KALDI_ASSERT(!KALDI_ISNAN(1.0 / den));
  KALDI_ASSERT(KALDI_ISINF(1.0 / den));
  KALDI_ASSERT(!KALDI_ISFINITE(1.0 / den));
  KALDI_ASSERT(KALDI_ISFINITE(0.0));
  KALDI_ASSERT(!KALDI_ISINF(0.0));
  KALDI_ASSERT(!KALDI_ISNAN(0.0));

  std::cout << 1.0+DBL_EPSILON;
  std::cout << 1.0 + 0.5*DBL_EPSILON;
  KALDI_ASSERT(1.0 + DBL_EPSILON != 1.0 && 1.0 + (0.5*DBL_EPSILON) == 1.0
               && "If this test fails, you can probably just comment it out-- "
               "may mean your CPU exceeds expected floating point precision");
  KALDI_ASSERT(1.0f + FLT_EPSILON != 1.0f && 1.0f + (0.5f*FLT_EPSILON) == 1.0f
               && "If this test fails, you can probably just comment it out-- "
               "may mean your CPU exceeds expected floating point precision");
  KALDI_ASSERT(std::abs(sin(M_PI)) < 1.0e-05
               && std::abs(cos(M_PI)+1.0) < 1.0e-05);
  KALDI_ASSERT(std::abs(sin(M_2PI)) < 1.0e-05
               && std::abs(cos(M_2PI)-1.0) < 1.0e-05);
  KALDI_ASSERT(std::abs(sin(Exp(M_LOG_2PI))) < 1.0e-05);
  KALDI_ASSERT(std::abs(cos(Exp(M_LOG_2PI)) - 1.0) < 1.0e-05);
}

void UnitTestAssertFunc() {  // Testing Assert** *functions
  for (int i = 1; i < 100; i++) {
    float f1 = Rand() % 10000 + 1, f2 = Rand() % 20 + 1;
    float tmp1 = f1 * f2;
    float tmp2 = (1/f1 + 1/f2);
    float add = f1 + f2;
    float addeql = tmp1 * tmp2;
    float thresh = 0.00001;
    AssertEqual(add, addeql, thresh);  // test AssertEqual()
  }
}

template<class I> void UnitTestFactorizeTpl() {
  for (int p= 0; p < 100; p++) {
    I m = Rand() % 100000;
    if (m >= 1) {
      std::vector<I> factors;
      Factorize(m, &factors);
      I m2 = 1;
      for (size_t i = 0; i < factors.size(); i++) {
        m2 *= factors[i];
        if (i+1 < factors.size())
          KALDI_ASSERT(factors[i+1] >= factors[i]);  // check sorted.
      }
      KALDI_ASSERT(m2 == m);  // check correctness.
    }
  }
}

void UnitTestFactorize() {
  UnitTestFactorizeTpl<int>();
  UnitTestFactorizeTpl<size_t>();
  UnitTestFactorizeTpl<int16>();
}

void UnitTestApproxEqual() {
  KALDI_ASSERT(ApproxEqual(1.0, 1.00001));
  KALDI_ASSERT(ApproxEqual(1.0, 1.00001, 0.001));
  KALDI_ASSERT(!ApproxEqual(1.0, 1.1));
  KALDI_ASSERT(!ApproxEqual(1.0, 1.01, 0.001));
  KALDI_ASSERT(!ApproxEqual(1.0, 0.0));
  KALDI_ASSERT(ApproxEqual(0.0, 0.0));
  KALDI_ASSERT(!ApproxEqual(0.0, 0.00001));
  KALDI_ASSERT(!ApproxEqual(std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()));
  KALDI_ASSERT(ApproxEqual(std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity()));
  KALDI_ASSERT(ApproxEqual(-std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity()));
  KALDI_ASSERT(!ApproxEqual(-std::numeric_limits<float>::infinity(),
                            0));
  KALDI_ASSERT(!ApproxEqual(-std::numeric_limits<float>::infinity(),
                            1));
}

template<class Real>
void UnitTestExpSpeed() {
  Real sum = 0.0;  // compute the sum to avoid optimizing it away.
  Real time = 0.01;  // how long this should last.
  int block_size = 10;
  int num_ops = 0;
  Timer tim;
  while (tim.Elapsed() < time) {
    for (int i = 0; i < block_size; i++) {
      sum += Exp((Real)i);
    }
    num_ops += block_size;
  }
  KALDI_ASSERT(sum > 0.0);  // make it harder for the compiler to optimize Exp
                            // away, as we have a conditional.
  Real flops = 1.0e-06 * num_ops / tim.Elapsed();
  KALDI_LOG << "Megaflops doing Exp("
            << (sizeof(Real) == 4 ? "float" : "double") << ") is " << flops;
}


template<class Real>
void UnitTestLogSpeed() {
  Real sum = 0.0;  // compute the sum to avoid optimizing it away.
  Real time = 0.01;  // how long this should last.
  int block_size = 10;
  int num_ops = 0;
  Timer tim;
  while (tim.Elapsed() < time) {
    for (int i = 0; i < block_size; i++) {
      sum += Log(static_cast<float>(i + 1));
    }
    num_ops += block_size;
  }
  KALDI_ASSERT(sum > 0.0);  // make it harder for the compiler to optimize Log
                            // away, as we have a conditional.
  Real flops = 1.0e-06 * num_ops / tim.Elapsed();
  KALDI_LOG << "Megaflops doing Log("
            << (sizeof(Real) == 4 ? "float" : "double") << ") is " << flops;
}

}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestApproxEqual();
  UnitTestGcdLcm();
  UnitTestFactorize();
  UnitTestDefines();
  UnitTestLogAddSub();
  UnitTestRand();
  UnitTestAssertFunc();
  UnitTestRoundUpToNearestPowerOfTwo();
  UnitTestDivideRoundingDown();
  UnitTestExpSpeed<float>();
  UnitTestExpSpeed<double>();
  UnitTestLogSpeed<float>();
  UnitTestLogSpeed<double>();
}

