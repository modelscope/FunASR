// util/edit-distance-inl.h

// Copyright 2009-2011  Microsoft Corporation;  Haihua Xu;  Yanmin Qian

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

#ifndef KALDI_UTIL_EDIT_DISTANCE_INL_H_
#define KALDI_UTIL_EDIT_DISTANCE_INL_H_
#include <algorithm>
#include <utility>
#include <vector>
#include "util/stl-utils.h"

namespace kaldi {

template<class T>
int32 LevenshteinEditDistance(const std::vector<T> &a,
                              const std::vector<T> &b) {
  // Algorithm:
  //  write A and B for the sequences, with elements a_0 ..
  //  let |A| = M and |B| = N be the lengths, and have
  //  elements a_0 ... a_{M-1} and b_0 ... b_{N-1}.
  //  We are computing the recursion
  //     E(m, n) = min(  E(m-1, n-1) + (1-delta(a_{m-1}, b_{n-1})),
  //                    E(m-1, n) + 1,
  //                    E(m, n-1) + 1).
  //  where E(m, n) is defined for m = 0..M and n = 0..N and out-of-
  //  bounds quantities are considered to be infinity (i.e. the
  //  recursion does not visit them).

  // We do this computation using a vector e of size N+1.
  // The outer iterations range over m = 0..M.

  int M = a.size(), N = b.size();
  std::vector<int32> e(N+1);
  std::vector<int32> e_tmp(N+1);
  // initialize e.
  for (size_t i = 0; i < e.size(); i++)
    e[i] = i;
  for (int32 m = 1; m <= M; m++) {
    // computing E(m, .) from E(m-1, .)
    // handle special case n = 0:
    e_tmp[0] = e[0] + 1;

    for (int32 n = 1; n <= N; n++) {
      int32 term1 = e[n-1] + (a[m-1] == b[n-1] ? 0 : 1);
      int32 term2 = e[n] + 1;
      int32 term3 = e_tmp[n-1] + 1;
      e_tmp[n] = std::min(term1, std::min(term2, term3));
    }
    e = e_tmp;
  }
  return e.back();
}
//
struct error_stats {
  int32 ins_num;
  int32 del_num;
  int32 sub_num;
  int32 total_cost;  // minimum total cost to the current alignment.
};
// Note that both hyp and ref should not contain noise word in
// the following implementation.

template<class T>
int32 LevenshteinEditDistance(const std::vector<T> &ref,
                              const std::vector<T> &hyp,
                              int32 *ins, int32 *del, int32 *sub) {
  // temp sequence to remember error type and stats.
  std::vector<error_stats> e(ref.size()+1);
  std::vector<error_stats> cur_e(ref.size()+1);
  // initialize the first hypothesis aligned to the reference at each
  // position:[hyp_index =0][ref_index]
  for (size_t i =0; i < e.size(); i ++) {
    e[i].ins_num = 0;
    e[i].sub_num = 0;
    e[i].del_num = i;
    e[i].total_cost = i;
  }

  // for other alignments
  for (size_t hyp_index = 1; hyp_index <= hyp.size(); hyp_index ++) {
    cur_e[0] = e[0];
    cur_e[0].ins_num++;
    cur_e[0].total_cost++;
    for (size_t ref_index = 1; ref_index <= ref.size(); ref_index ++) {
     int32 ins_err = e[ref_index].total_cost + 1;
     int32 del_err = cur_e[ref_index-1].total_cost + 1;
     int32 sub_err = e[ref_index-1].total_cost;
      if (hyp[hyp_index-1] != ref[ref_index-1])
       sub_err++;

     if (sub_err < ins_err && sub_err < del_err) {
        cur_e[ref_index] =e[ref_index-1];
        if (hyp[hyp_index-1] != ref[ref_index-1])
          cur_e[ref_index].sub_num++;  // substitution error should be increased
        cur_e[ref_index].total_cost = sub_err;
     } else if (del_err < ins_err) {
        cur_e[ref_index] = cur_e[ref_index-1];
        cur_e[ref_index].total_cost = del_err;
        cur_e[ref_index].del_num++;    // deletion number is increased.
     } else {
        cur_e[ref_index] = e[ref_index];
        cur_e[ref_index].total_cost = ins_err;
        cur_e[ref_index].ins_num++;    // insertion number is increased.
     }
  }
  e = cur_e;  // alternate for the next recursion.
  }
  size_t ref_index = e.size()-1;
  *ins = e[ref_index].ins_num, *del =
    e[ref_index].del_num, *sub = e[ref_index].sub_num;
  return e[ref_index].total_cost;
}

template<class T>
int32 LevenshteinAlignment(const std::vector<T> &a,
                           const std::vector<T> &b,
                           T eps_symbol,
                           std::vector<std::pair<T, T> > *output) {
  // Check inputs:
  {
    KALDI_ASSERT(output != NULL);
    for (size_t i = 0; i < a.size(); i++) KALDI_ASSERT(a[i] != eps_symbol);
    for (size_t i = 0; i < b.size(); i++) KALDI_ASSERT(b[i] != eps_symbol);
  }
  output->clear();
  // This is very memory-inefficiently implemented using a vector of vectors.
  size_t M = a.size(), N = b.size();
  size_t m, n;
  std::vector<std::vector<int32> > e(M+1);
  for (m = 0; m <=M; m++) e[m].resize(N+1);
  for (n = 0; n <= N; n++)
    e[0][n]  = n;
  for (m = 1; m <= M; m++) {
    e[m][0] = e[m-1][0] + 1;
    for (n = 1; n <= N; n++) {
      int32 sub_or_ok = e[m-1][n-1] + (a[m-1] == b[n-1] ? 0 : 1);
      int32 del = e[m-1][n] + 1;  // assumes a == ref, b == hyp.
      int32 ins = e[m][n-1] + 1;
      e[m][n] = std::min(sub_or_ok, std::min(del, ins));
    }
  }
  // get time-reversed output first: trace back.
  m = M;
  n = N;
  while (m != 0 || n != 0) {
    size_t last_m, last_n;
    if (m == 0) {
      last_m = m;
      last_n = n-1;
    } else if (n == 0) {
      last_m = m-1;
      last_n = n;
    } else {
      int32 sub_or_ok = e[m-1][n-1] + (a[m-1] == b[n-1] ? 0 : 1);
      int32 del = e[m-1][n] + 1;  // assumes a == ref, b == hyp.
      int32 ins = e[m][n-1] + 1;
      // choose sub_or_ok if all else equal.
      if (sub_or_ok <= std::min(del, ins)) {
        last_m = m-1;
        last_n = n-1;
      } else {
        if (del <= ins) {  // choose del over ins if equal.
          last_m = m-1;
          last_n = n;
        } else {
          last_m = m;
          last_n = n-1;
        }
      }
    }
    T a_sym, b_sym;
    a_sym = (last_m == m ? eps_symbol : a[last_m]);
    b_sym = (last_n == n ? eps_symbol : b[last_n]);
    output->push_back(std::make_pair(a_sym, b_sym));
    m = last_m;
    n = last_n;
  }
  ReverseVector(output);
  return e[M][N];
}


}  // end namespace kaldi

#endif  // KALDI_UTIL_EDIT_DISTANCE_INL_H_
