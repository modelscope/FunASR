// base/kaldi-math.cc

// Copyright 2009-2011  Microsoft Corporation;  Yanmin Qian;
//                      Saarland University;  Jan Silovsky

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

#include "base/kaldi-math.h"
#ifndef _MSC_VER
#include <stdlib.h>
#endif
#include <string>
#include <mutex>

namespace kaldi {
// These routines are tested in matrix/matrix-test.cc

int32 RoundUpToNearestPowerOfTwo(int32 n) {
  KALDI_ASSERT(n > 0);
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n+1;
}

static std::mutex _RandMutex;

int Rand(struct RandomState* state) {
#if defined(_MSC_VER) || defined(__CYGWIN__)
  // On Windows and Cygwin, just call Rand()
  return rand();
#else
  if (state) {
    return rand_r(&(state->seed));
  } else {
    std::lock_guard<std::mutex> lock(_RandMutex);
    return rand();
  }
#endif
}

RandomState::RandomState() {
  // we initialize it as Rand() + 27437 instead of just Rand(), because on some
  // systems, e.g. at the very least Mac OSX Yosemite and later, it seems to be
  // the case that rand_r when initialized with rand() will give you the exact
  // same sequence of numbers that rand() will give if you keep calling rand()
  // after that initial call.  This can cause problems with repeated sequences.
  // For example if you initialize two RandomState structs one after the other
  // without calling rand() in between, they would give you the same sequence
  // offset by one (if we didn't have the "+ 27437" in the code).  27437 is just
  // a randomly chosen prime number.
  seed = Rand() + 27437;
}

bool WithProb(BaseFloat prob, struct RandomState* state) {
  KALDI_ASSERT(prob >= 0 && prob <= 1.1);  // prob should be <= 1.0,
  // but we allow slightly larger values that could arise from roundoff in
  // previous calculations.
  KALDI_COMPILE_TIME_ASSERT(RAND_MAX > 128 * 128);
  if (prob == 0) return false;
  else if (prob == 1.0) return true;
  else if (prob * RAND_MAX < 128.0) {
    // prob is very small but nonzero, and the "main algorithm"
    // wouldn't work that well.  So: with probability 1/128, we
    // return WithProb (prob * 128), else return false.
    if (Rand(state) < RAND_MAX / 128) {  // with probability 128...
      // Note: we know that prob * 128.0 < 1.0, because
      // we asserted RAND_MAX > 128 * 128.
      return WithProb(prob * 128.0);
    } else {
      return false;
    }
  } else {
    return (Rand(state) < ((RAND_MAX + static_cast<BaseFloat>(1.0)) * prob));
  }
}

int32 RandInt(int32 min_val, int32 max_val, struct RandomState* state) {
  // This is not exact.
  KALDI_ASSERT(max_val >= min_val);
  if (max_val == min_val) return min_val;

#ifdef _MSC_VER
  // RAND_MAX is quite small on Windows -> may need to handle larger numbers.
  if (RAND_MAX > (max_val-min_val)*8) {
        // *8 to avoid large inaccuracies in probability, from the modulus...
    return min_val +
      ((unsigned int)Rand(state) % (unsigned int)(max_val+1-min_val));
  } else {
    if ((unsigned int)(RAND_MAX*RAND_MAX) >
        (unsigned int)((max_val+1-min_val)*8)) {
        // *8 to avoid inaccuracies in probability, from the modulus...
      return min_val + ( (unsigned int)( (Rand(state)+RAND_MAX*Rand(state)))
                    % (unsigned int)(max_val+1-min_val));
    } else {
      throw std::runtime_error(std::string()
                               +"rand_int failed because we do not support "
                               +"such large random numbers. "
                               +"(Extend this function).");
    }
  }
#else
  return min_val +
      (static_cast<int32>(Rand(state)) % static_cast<int32>(max_val+1-min_val));
#endif
}

// Returns poisson-distributed random number.
// Take care: this takes time proportinal
// to lambda.  Faster algorithms exist but are more complex.
int32 RandPoisson(float lambda, struct RandomState* state) {
  // Knuth's algorithm.
  KALDI_ASSERT(lambda >= 0);
  float L = expf(-lambda), p = 1.0;
  int32 k = 0;
  do {
    k++;
    float u = RandUniform(state);
    p *= u;
  } while (p > L);
  return k-1;
}

void RandGauss2(float *a, float *b, RandomState *state) {
  KALDI_ASSERT(a);
  KALDI_ASSERT(b);
  float u1 = RandUniform(state);
  float u2 = RandUniform(state);
  u1 = sqrtf(-2.0f * logf(u1));
  u2 =  2.0f * M_PI * u2;
  *a = u1 * cosf(u2);
  *b = u1 * sinf(u2);
}

void RandGauss2(double *a, double *b, RandomState *state) {
  KALDI_ASSERT(a);
  KALDI_ASSERT(b);
  float a_float, b_float;
  // Just because we're using doubles doesn't mean we need super-high-quality
  // random numbers, so we just use the floating-point version internally.
  RandGauss2(&a_float, &b_float, state);
  *a = a_float;
  *b = b_float;
}


}  // end namespace kaldi
