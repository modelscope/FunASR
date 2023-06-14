// matrix/matrix-functions-inl.h

// Copyright 2009-2011 Microsoft Corporation
//
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
//
// (*) incorporates, with permission, FFT code from his book
// "Signal Processing with Lapped Transforms", Artech, 1992.



#ifndef KALDI_MATRIX_MATRIX_FUNCTIONS_INL_H_
#define KALDI_MATRIX_MATRIX_FUNCTIONS_INL_H_

namespace kaldi {

//! ComplexMul implements, inline, the complex multiplication b *= a.
template<typename Real> inline void ComplexMul(const Real &a_re, const Real &a_im,
                                            Real *b_re, Real *b_im) {
  Real tmp_re = (*b_re * a_re) - (*b_im * a_im);
  *b_im = *b_re * a_im + *b_im * a_re;
  *b_re = tmp_re;
}

template<typename Real> inline void ComplexAddProduct(const Real &a_re, const Real &a_im,
                                                   const Real &b_re, const Real &b_im,
                                                   Real *c_re, Real *c_im) {
  *c_re += b_re*a_re - b_im*a_im;
  *c_im += b_re*a_im + b_im*a_re;
}


template<typename Real> inline void ComplexImExp(Real x, Real *a_re, Real *a_im) {
  *a_re = std::cos(x);
  *a_im = std::sin(x);
}


} // end namespace kaldi


#endif // KALDI_MATRIX_MATRIX_FUNCTIONS_INL_H_

