// matrix/srfft.cc

// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.

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

// This file includes a modified version of code originally published in Malvar,
// H., "Signal processing with lapped transforms, " Artech House, Inc., 1992.  The
// current copyright holder of the original code, Henrique S. Malvar, has given
// his permission for the release of this modified version under the Apache
// License v2.0.


#include "matrix/srfft.h"
#include "matrix/matrix-functions.h"

namespace kaldi {


template<typename Real>
SplitRadixComplexFft<Real>::SplitRadixComplexFft(MatrixIndexT N) {
  if ( (N & (N-1)) != 0 || N <= 1)
    KALDI_ERR << "SplitRadixComplexFft called with invalid number of points "
              << N;
  N_ = N;
  logn_ = 0;
  while (N > 1) {
    N >>= 1;
    logn_ ++;
  }
  ComputeTables();
}

template <typename Real>
SplitRadixComplexFft<Real>::SplitRadixComplexFft(
    const SplitRadixComplexFft<Real> &other):
    N_(other.N_), logn_(other.logn_) {
  // This code duplicates tables from a previously computed object.
  // Compare with the code in ComputeTables().
  MatrixIndexT lg2 = logn_ >> 1;
  if (logn_ & 1) lg2++;
  MatrixIndexT brseed_size = 1 << lg2;
  brseed_ = new MatrixIndexT[brseed_size];
  std::memcpy(brseed_, other.brseed_, sizeof(MatrixIndexT) * brseed_size);

  if (logn_ < 4) {
    tab_ = NULL;
  } else {
    tab_ = new Real*[logn_ - 3];
    for (MatrixIndexT i = logn_; i >= 4 ; i--) {
      MatrixIndexT m = 1 << i, m2 = m / 2, m4 = m2 / 2;
      MatrixIndexT this_array_size = 6 * (m4 - 2);
      tab_[i-4] = new Real[this_array_size];
      std::memcpy(tab_[i-4], other.tab_[i-4],
                  sizeof(Real) * this_array_size);
    }
  }
}

template<typename Real>
void SplitRadixComplexFft<Real>::ComputeTables() {
  MatrixIndexT    imax, lg2, i, j;
  MatrixIndexT     m, m2, m4, m8, nel, n;
  Real    *cn, *spcn, *smcn, *c3n, *spc3n, *smc3n;
  Real    ang, c, s;

  lg2 = logn_ >> 1;
  if (logn_ & 1) lg2++;
  brseed_ = new MatrixIndexT[1 << lg2];
  brseed_[0] = 0;
  brseed_[1] = 1;
  for (j = 2; j <= lg2; j++) {
    imax = 1 << (j - 1);
    for (i = 0; i < imax; i++) {
      brseed_[i] <<= 1;
      brseed_[i + imax] = brseed_[i] + 1;
    }
  }

  if (logn_ < 4) {
    tab_ = NULL;
  } else {
    tab_ = new Real* [logn_-3];
    for (i = logn_; i>=4 ; i--) {
      /* Compute a few constants */
      m = 1 << i; m2 = m / 2; m4 = m2 / 2; m8 = m4 /2;

      /* Allocate memory for tables */
      nel = m4 - 2;

      tab_[i-4] = new Real[6*nel];

      /* Initialize pointers */
      cn = tab_[i-4]; spcn  = cn + nel;  smcn  = spcn + nel;
      c3n = smcn + nel;  spc3n = c3n + nel; smc3n = spc3n + nel;

      /* Compute tables */
      for (n = 1; n < m4; n++) {
        if (n == m8) continue;
        ang = n * M_2PI / m;
        c = std::cos(ang); s = std::sin(ang);
        *cn++ = c; *spcn++ = - (s + c); *smcn++ = s - c;
        ang = 3 * n * M_2PI / m;
        c = std::cos(ang); s = std::sin(ang);
        *c3n++ = c; *spc3n++ = - (s + c); *smc3n++ = s - c;
      }
    }
  }
}

template<typename Real>
SplitRadixComplexFft<Real>::~SplitRadixComplexFft() {
  delete [] brseed_;
  if (tab_ != NULL) {
    for (MatrixIndexT i = 0; i < logn_-3; i++)
      delete [] tab_[i];
    delete [] tab_;
  }
}

template<typename Real>
void SplitRadixComplexFft<Real>::Compute(Real *xr, Real *xi, bool forward) const {
  if (!forward) {  // reverse real and imaginary parts for complex FFT.
    Real *tmp = xr;
    xr = xi;
    xi = tmp;
  }
  ComputeRecursive(xr, xi, logn_);
  if (logn_ > 1) {
    BitReversePermute(xr, logn_);
    BitReversePermute(xi, logn_);
  }
}

template<typename Real>
void SplitRadixComplexFft<Real>::Compute(Real *x, bool forward,
                                         std::vector<Real> *temp_buffer) const {
  KALDI_ASSERT(temp_buffer != NULL);
  if (temp_buffer->size() != N_)
    temp_buffer->resize(N_);
  Real *temp_ptr = &((*temp_buffer)[0]);
  for (MatrixIndexT i = 0; i < N_; i++) {
    x[i] = x[i * 2];  // put the real part in the first half of x.
    temp_ptr[i] = x[i * 2 + 1];  // put the imaginary part in temp_buffer.
  }
  // copy the imaginary part back to the second half of x.
  memcpy(static_cast<void*>(x + N_),
         static_cast<void*>(temp_ptr),
         sizeof(Real) * N_);

  Compute(x, x + N_, forward);
  // Now change the format back to interleaved.
  memcpy(static_cast<void*>(temp_ptr),
         static_cast<void*>(x + N_),
         sizeof(Real) * N_);
  for (MatrixIndexT i = N_-1; i > 0; i--) {  // don't include 0,
    // in case MatrixIndexT is unsigned, the loop would not terminate.
    // Treat it as a special case.
    x[i*2] = x[i];
    x[i*2 + 1] = temp_ptr[i];
  }
  x[1] = temp_ptr[0];  // special case of i = 0.
}

template<typename Real>
void SplitRadixComplexFft<Real>::Compute(Real *x, bool forward) {
  this->Compute(x, forward, &temp_buffer_);
}

template<typename Real>
void SplitRadixComplexFft<Real>::BitReversePermute(Real *x, MatrixIndexT logn) const {
  MatrixIndexT      i, j, lg2, n;
  MatrixIndexT      off, fj, gno, *brp;
  Real    tmp, *xp, *xq;

  lg2 = logn >> 1;
  n = 1 << lg2;
  if (logn & 1) lg2++;

  /* Unshuffling loop */
  for (off = 1; off < n; off++) {
    fj = n * brseed_[off]; i = off; j = fj;
    tmp = x[i]; x[i] = x[j]; x[j] = tmp;
    xp = &x[i];
    brp = &(brseed_[1]);
    for (gno = 1; gno < brseed_[off]; gno++) {
      xp += n;
      j = fj + *brp++;
      xq = x + j;
      tmp = *xp; *xp = *xq; *xq = tmp;
    }
  }
}


template<typename Real>
void SplitRadixComplexFft<Real>::ComputeRecursive(Real *xr, Real *xi, MatrixIndexT logn) const {

  MatrixIndexT    m, m2, m4, m8, nel, n;
  Real    *xr1, *xr2, *xi1, *xi2;
  Real    *cn, *spcn, *smcn, *c3n, *spc3n, *smc3n;
  Real    tmp1, tmp2;
  Real   sqhalf = M_SQRT1_2;

  /* Check range of logn */
  if (logn < 0)
    KALDI_ERR << "Error: logn is out of bounds in SRFFT";

  /* Compute trivial cases */
  if (logn < 3) {
    if (logn == 2) {  /* length m = 4 */
      xr2  = xr + 2;
      xi2  = xi + 2;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr  = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi  = tmp1;
      xr1  = xr + 1;
      xi1  = xi + 1;
      xr2++;
      xi2++;
      tmp1 = *xr1 + *xr2;
      *xr2 = *xr1 - *xr2;
      *xr1 = tmp1;
      tmp1 = *xi1 + *xi2;
      *xi2 = *xi1 - *xi2;
      *xi1 = tmp1;
      xr2  = xr + 1;
      xi2  = xi + 1;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr  = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi  = tmp1;
      xr1  = xr + 2;
      xi1  = xi + 2;
      xr2  = xr + 3;
      xi2  = xi + 3;
      tmp1 = *xr1 + *xi2;
      tmp2 = *xi1 + *xr2;
      *xi1 = *xi1 - *xr2;
      *xr2 = *xr1 - *xi2;
      *xr1 = tmp1;
      *xi2 = tmp2;
      return;
    }
    else if (logn == 1) {   /* length m = 2 */
      xr2  = xr + 1;
      xi2  = xi + 1;
      tmp1 = *xr + *xr2;
      *xr2 = *xr - *xr2;
      *xr  = tmp1;
      tmp1 = *xi + *xi2;
      *xi2 = *xi - *xi2;
      *xi  = tmp1;
      return;
    }
    else if (logn == 0) return;   /* length m = 1 */
  }

  /* Compute a few constants */
  m = 1 << logn; m2 = m / 2; m4 = m2 / 2; m8 = m4 /2;


  /* Step 1 */
  xr1 = xr; xr2 = xr1 + m2;
  xi1 = xi; xi2 = xi1 + m2;
  for (n = 0; n < m2; n++) {
    tmp1 = *xr1 + *xr2;
    *xr2 = *xr1 - *xr2;
    xr2++;
    *xr1++ = tmp1;
    tmp2 = *xi1 + *xi2;
    *xi2 = *xi1 - *xi2;
    xi2++;
    *xi1++ = tmp2;
  }

  /* Step 2 */
  xr1 = xr + m2; xr2 = xr1 + m4;
  xi1 = xi + m2; xi2 = xi1 + m4;
  for (n = 0; n < m4; n++) {
    tmp1 = *xr1 + *xi2;
    tmp2 = *xi1 + *xr2;
    *xi1 = *xi1 - *xr2;
    xi1++;
    *xr2++ = *xr1 - *xi2;
    *xr1++ = tmp1;
    *xi2++ = tmp2;
    // xr1++; xr2++; xi1++; xi2++;
  }

  /* Steps 3 & 4 */
  xr1 = xr + m2; xr2 = xr1 + m4;
  xi1 = xi + m2; xi2 = xi1 + m4;
  if (logn >= 4) {
    nel = m4 - 2;
    cn  = tab_[logn-4]; spcn  = cn + nel;  smcn  = spcn + nel;
    c3n = smcn + nel;  spc3n = c3n + nel; smc3n = spc3n + nel;
  }
  xr1++; xr2++; xi1++; xi2++;
  // xr1++; xi1++;
  for (n = 1; n < m4; n++) {
    if (n == m8) {
      tmp1 =  sqhalf * (*xr1 + *xi1);
      *xi1 =  sqhalf * (*xi1 - *xr1);
      *xr1 =  tmp1;
      tmp2 =  sqhalf * (*xi2 - *xr2);
      *xi2 = -sqhalf * (*xr2 + *xi2);
      *xr2 =  tmp2;
    } else {
      tmp2 = *cn++ * (*xr1 + *xi1);
      tmp1 = *spcn++ * *xr1 + tmp2;
      *xr1 = *smcn++ * *xi1 + tmp2;
      *xi1 = tmp1;
      tmp2 = *c3n++ * (*xr2 + *xi2);
      tmp1 = *spc3n++ * *xr2 + tmp2;
      *xr2 = *smc3n++ * *xi2 + tmp2;
      *xi2 = tmp1;
    }
    xr1++; xr2++; xi1++; xi2++;
  }

  /* Call ssrec again with half DFT length */
  ComputeRecursive(xr, xi, logn-1);

  /* Call ssrec again twice with one quarter DFT length.
     Constants have to be recomputed, because they are static! */
  // m = 1 << logn; m2 = m / 2;
  ComputeRecursive(xr + m2, xi + m2, logn - 2);
  // m = 1 << logn;
  m4 = 3 * (m / 4);
  ComputeRecursive(xr + m4, xi + m4, logn - 2);
}


template<typename Real>
void SplitRadixRealFft<Real>::Compute(Real *data, bool forward) {
  Compute(data, forward, &this->temp_buffer_);
}


// This code is mostly the same as the RealFft function.  It would be
// possible to replace it with more efficient code from Rico's book.
template<typename Real>
void SplitRadixRealFft<Real>::Compute(Real *data, bool forward,
                                      std::vector<Real> *temp_buffer) const {
  MatrixIndexT N = N_, N2 = N/2;
  KALDI_ASSERT(N%2 == 0);
  if (forward) // call to base class
    SplitRadixComplexFft<Real>::Compute(data, true, temp_buffer);

  Real rootN_re, rootN_im;  // exp(-2pi/N), forward; exp(2pi/N), backward
  int forward_sign = forward ? -1 : 1;
  ComplexImExp(static_cast<Real>(M_2PI/N *forward_sign), &rootN_re, &rootN_im);
  Real kN_re = -forward_sign, kN_im = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
  // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
  for (MatrixIndexT k = 1; 2*k <= N2; k++) {
    ComplexMul(rootN_re, rootN_im, &kN_re, &kN_im);

    Real Ck_re, Ck_im, Dk_re, Dk_im;
    // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
    Ck_re = 0.5 * (data[2*k] + data[N - 2*k]);
    Ck_im = 0.5 * (data[2*k + 1] - data[N - 2*k + 1]);
    // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
    Dk_re = 0.5 * (data[2*k + 1] + data[N - 2*k + 1]);
    // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
    Dk_im =-0.5 * (data[2*k] - data[N - 2*k]);
    // A_k = C_k + 1^(k/N) D_k:
    data[2*k] = Ck_re;  // A_k <-- C_k
    data[2*k+1] = Ck_im;
    // now A_k += D_k 1^(k/N)
    ComplexAddProduct(Dk_re, Dk_im, kN_re, kN_im, &(data[2*k]), &(data[2*k+1]));

    MatrixIndexT kdash = N2 - k;
    if (kdash != k) {
      // Next we handle the index k' = N/2 - k.  This is necessary
      // to do now, to avoid invalidating data that we will later need.
      // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
      // and D_k, so the equations are simple modifications of the above,
      // replacing Ck_im and Dk_im with their negatives.
      data[2*kdash] = Ck_re;  // A_k' <-- C_k'
      data[2*kdash+1] = -Ck_im;
      // now A_k' += D_k' 1^(k'/N)
      // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 * (1^(k/N))^*
      // so it's the same as 1^(k/N) but with the real part negated.
      ComplexAddProduct(Dk_re, -Dk_im, -kN_re, kN_im, &(data[2*kdash]), &(data[2*kdash+1]));
    }
  }

  {  // Now handle k = 0.
    // In simple terms: after the complex fft, data[0] becomes the sum of real
    // parts input[0], input[2]... and data[1] becomes the sum of imaginary
    // pats input[1], input[3]...
    // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
    // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
    Real zeroth = data[0] + data[1],
        n2th = data[0] - data[1];
    data[0] = zeroth;
    data[1] = n2th;
    if (!forward) {
      data[0] /= 2;
      data[1] /= 2;
    }
  }
  if (!forward) {  // call to base class
    SplitRadixComplexFft<Real>::Compute(data, false, temp_buffer);
    for (MatrixIndexT i = 0; i < N; i++)
      data[i] *= 2.0;
    // This is so we get a factor of N increase, rather than N/2 which we would
    // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in dimension N/2.
    // It's for consistency with our normal FFT convensions.
  }
}

template class SplitRadixComplexFft<float>;
template class SplitRadixComplexFft<double>;
template class SplitRadixRealFft<float>;
template class SplitRadixRealFft<double>;


} // end namespace kaldi
