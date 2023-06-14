// matrix/jama-svd.h

// Copyright 2009-2011 Microsoft Corporation

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

// This file consists of a port and modification of materials from
//   JAMA: A Java Matrix Package
// under the following notice: This software is a cooperative product of
// The MathWorks and the National Institute of Standards and Technology (NIST)
// which has been released to the public.  This notice and the original code are
// available at http://math.nist.gov/javanumerics/jama/domain.notice


#ifndef KALDI_MATRIX_JAMA_SVD_H_
#define KALDI_MATRIX_JAMA_SVD_H_ 1


#include "matrix/kaldi-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/cblas-wrappers.h"

namespace kaldi {

#if defined(HAVE_ATLAS) || defined(USE_KALDI_SVD)
// using ATLAS as our math library, which doesn't have SVD -> need
// to implement it.

// This routine is a modified form of jama_svd.h which is part of the TNT distribution.
// (originally comes from JAMA).

/** Singular Value Decomposition.
 * <P>
 * For an m-by-n matrix A with m >= n, the singular value decomposition is
 * an m-by-n orthogonal matrix U, an n-by-n diagonal matrix S, and
 * an n-by-n orthogonal matrix V so that A = U*S*V'.
 * <P>
 * The singular values, sigma[k] = S(k, k), are ordered so that
 * sigma[0] >= sigma[1] >= ... >= sigma[n-1].
 * <P>
 * The singular value decompostion always exists, so the constructor will
 * never fail.  The matrix condition number and the effective numerical
 * rank can be computed from this decomposition.

 * <p>
 *     (Adapted from JAMA, a Java Matrix Library, developed by jointly
 *     by the Mathworks and NIST; see  http://math.nist.gov/javanumerics/jama).
 */


template<typename Real>
bool MatrixBase<Real>::JamaSvd(VectorBase<Real> *s_in,
                               MatrixBase<Real> *U_in,
                               MatrixBase<Real> *V_in) {  //  Destructive!
  KALDI_ASSERT(s_in != NULL && U_in != this && V_in != this);
  int wantu = (U_in != NULL), wantv = (V_in != NULL);
  Matrix<Real> Utmp, Vtmp;
  MatrixBase<Real> &U = (U_in ? *U_in : Utmp), &V = (V_in ? *V_in : Vtmp);
  VectorBase<Real> &s = *s_in;

  int m = num_rows_, n = num_cols_;
  KALDI_ASSERT(m>=n && m != 0 && n != 0);
  if (wantu) KALDI_ASSERT((int)U.num_rows_ == m && (int)U.num_cols_ == n);
  if (wantv) KALDI_ASSERT((int)V.num_rows_ == n && (int)V.num_cols_ == n);
  KALDI_ASSERT((int)s.Dim() == n);  // n<=m so n is min.

  int nu = n;
  U.SetZero();  // make sure all zero.
  Vector<Real> e(n);
  Vector<Real> work(m);
  MatrixBase<Real> &A(*this);
  Real *adata = A.Data(), *workdata = work.Data(), *edata = e.Data(),
      *udata = U.Data(), *vdata = V.Data();
  int astride = static_cast<int>(A.Stride()),
      ustride = static_cast<int>(U.Stride()),
      vstride = static_cast<int>(V.Stride());
  int i = 0, j = 0, k = 0;

  // Reduce A to bidiagonal form, storing the diagonal elements
  // in s and the super-diagonal elements in e.

  int nct = std::min(m-1, n);
  int nrt = std::max(0, std::min(n-2, m));
  for (k = 0; k < std::max(nct, nrt); k++) {
    if (k < nct) {

      // Compute the transformation for the k-th column and
      // place the k-th diagonal in s(k).
      // Compute 2-norm of k-th column without under/overflow.
      s(k) = 0;
      for (i = k; i < m; i++) {
        s(k) = hypot(s(k), A(i, k));
      }
      if (s(k) != 0.0) {
        if (A(k, k) < 0.0) {
          s(k) = -s(k);
        }
        for (i = k; i < m; i++) {
          A(i, k) /= s(k);
        }
        A(k, k) += 1.0;
      }
      s(k) = -s(k);
    }
    for (j = k+1; j < n; j++) {
      if ((k < nct) && (s(k) != 0.0))  {

        // Apply the transformation.

        Real t = cblas_Xdot(m - k, adata + astride*k + k, astride,
                            adata + astride*k + j, astride);
        /*for (i = k; i < m; i++) {
          t += adata[i*astride + k]*adata[i*astride + j];  //   A(i, k)*A(i, j); // 3
          }*/
        t = -t/A(k, k);
        cblas_Xaxpy(m - k, t, adata + k*astride + k, astride,
                    adata + k*astride + j, astride);
        /*for (i = k; i < m; i++) {
          adata[i*astride + j] += t*adata[i*astride + k];  // A(i, j) += t*A(i, k); // 5
          }*/
      }

      // Place the k-th row of A into e for the
      // subsequent calculation of the row transformation.

      e(j) = A(k, j);
    }
    if (wantu & (k < nct)) {

      // Place the transformation in U for subsequent back
      // multiplication.

      for (i = k; i < m; i++) {
        U(i, k) = A(i, k);
      }
    }
    if (k < nrt) {

      // Compute the k-th row transformation and place the
      // k-th super-diagonal in e(k).
      // Compute 2-norm without under/overflow.
      e(k) = 0;
      for (i = k+1; i < n; i++) {
        e(k) = hypot(e(k), e(i));
      }
      if (e(k) != 0.0) {
        if (e(k+1) < 0.0) {
          e(k) = -e(k);
        }
        for (i = k+1; i < n; i++) {
          e(i) /= e(k);
        }
        e(k+1) += 1.0;
      }
      e(k) = -e(k);
      if ((k+1 < m) & (e(k) != 0.0)) {

        // Apply the transformation.

        for (i = k+1; i < m; i++) {
          work(i) = 0.0;
        }
        for (j = k+1; j < n; j++) {
          for (i = k+1; i < m; i++) {
            workdata[i] += edata[j] * adata[i*astride + j];  // work(i) += e(j)*A(i, j); // 5
          }
        }
        for (j = k+1; j < n; j++) {
          Real t(-e(j)/e(k+1));
          cblas_Xaxpy(m - (k+1), t, workdata + (k+1), 1,
                      adata + (k+1)*astride + j, astride);
          /*
          for (i = k+1; i < m; i++) {
            adata[i*astride + j] += t*workdata[i];  // A(i, j) += t*work(i); // 5
            }*/
        }
      }
      if (wantv) {

        // Place the transformation in V for subsequent
        // back multiplication.

        for (i = k+1; i < n; i++) {
          V(i, k) = e(i);
        }
      }
    }
  }

  // Set up the final bidiagonal matrix or order p.

  int p = std::min(n, m+1);
  if (nct < n) {
    s(nct) = A(nct, nct);
  }
  if (m < p) {
    s(p-1) = 0.0;
  }
  if (nrt+1 < p) {
    e(nrt) = A(nrt, p-1);
  }
  e(p-1) = 0.0;

  // If required, generate U.

  if (wantu) {
    for (j = nct; j < nu; j++) {
      for (i = 0; i < m; i++) {
        U(i, j) = 0.0;
      }
      U(j, j) = 1.0;
    }
    for (k = nct-1; k >= 0; k--) {
      if (s(k) != 0.0) {
        for (j = k+1; j < nu; j++) {
          Real t = cblas_Xdot(m - k, udata + k*ustride + k, ustride, udata + k*ustride + j, ustride);
          //for (i = k; i < m; i++) {
          //  t += udata[i*ustride + k]*udata[i*ustride + j];  // t += U(i, k)*U(i, j); // 8
          // }
          t = -t/U(k, k);
          cblas_Xaxpy(m - k, t, udata + ustride*k + k, ustride,
                      udata + k*ustride + j, ustride);
          /*for (i = k; i < m; i++) {
            udata[i*ustride + j] += t*udata[i*ustride + k];  // U(i, j) += t*U(i, k); // 4
            }*/
        }
        for (i = k; i < m; i++ ) {
          U(i, k) = -U(i, k);
        }
        U(k, k) = 1.0 + U(k, k);
        for (i = 0; i < k-1; i++) {
          U(i, k) = 0.0;
        }
      } else {
        for (i = 0; i < m; i++) {
          U(i, k) = 0.0;
        }
        U(k, k) = 1.0;
      }
    }
  }

  // If required, generate V.

  if (wantv) {
    for (k = n-1; k >= 0; k--) {
      if ((k < nrt) & (e(k) != 0.0)) {
        for (j = k+1; j < nu; j++) {
          Real t = cblas_Xdot(n - (k+1), vdata + (k+1)*vstride + k, vstride,
                              vdata + (k+1)*vstride + j, vstride); 
          /*Real t (0.0);
          for (i = k+1; i < n; i++) {
            t += vdata[i*vstride + k]*vdata[i*vstride + j];  // t += V(i, k)*V(i, j); // 7
            }*/
          t = -t/V(k+1, k);
          cblas_Xaxpy(n - (k+1), t, vdata + (k+1)*vstride + k, vstride,
                      vdata + (k+1)*vstride + j, vstride);
          /*for (i = k+1; i < n; i++) {
            vdata[i*vstride + j] += t*vdata[i*vstride + k];  // V(i, j) += t*V(i, k); // 7
            }*/
        }
      }
      for (i = 0; i < n; i++) {
        V(i, k) = 0.0;
      }
      V(k, k) = 1.0;
    }
  }

  // Main iteration loop for the singular values.

  int pp = p-1;
  int iter = 0;
  // note: -52.0 is from Jama code; the -23 is the extension
  // to float, because mantissa length in (double, float)
  // is (52, 23) bits respectively.
  Real eps(pow(2.0, sizeof(Real) == 4 ? -23.0 : -52.0));
  // Note: the -966 was taken from Jama code, but the -120 is a guess
  // of how to extend this to float... the exponent in double goes
  // from -1022 .. 1023, and in float from -126..127.  I'm not sure
  // what the significance of 966 is, so -120 just represents a number
  // that's a bit less negative than -126.  If we get convergence
  // failure in float only, this may mean that we have to make the
  // -120 value less negative.
  Real tiny(pow(2.0, sizeof(Real) == 4 ? -120.0: -966.0 ));
  
  while (p > 0) {
    int k = 0;
    int kase = 0;

    if (iter == 500 || iter == 750) {
      KALDI_WARN << "Svd taking a long time: making convergence criterion less exact.";
      eps = pow(static_cast<Real>(0.8), eps);
      tiny = pow(static_cast<Real>(0.8), tiny);
    }
    if (iter > 1000) {
      KALDI_WARN << "Svd not converging on matrix of size " << m << " by " <<n;
      return false;
    }

    // This section of the program inspects for
    // negligible elements in the s and e arrays.  On
    // completion the variables kase and k are set as follows.

    // kase = 1     if s(p) and e(k-1) are negligible and k < p
    // kase = 2     if s(k) is negligible and k < p
    // kase = 3     if e(k-1) is negligible, k < p, and
    //              s(k), ..., s(p) are not negligible (qr step).
    // kase = 4     if e(p-1) is negligible (convergence).

    for (k = p-2; k >= -1; k--) {
      if (k == -1) {
        break;
      }
      if (std::abs(e(k)) <=
          tiny + eps*(std::abs(s(k)) + std::abs(s(k+1)))) {
        e(k) = 0.0;
        break;
      }
    }
    if (k == p-2) {
      kase = 4;
    } else {
      int ks;
      for (ks = p-1; ks >= k; ks--) {
        if (ks == k) {
          break;
        }
        Real t( (ks != p ? std::abs(e(ks)) : 0.) +
                (ks != k+1 ? std::abs(e(ks-1)) : 0.));
        if (std::abs(s(ks)) <= tiny + eps*t)  {
          s(ks) = 0.0;
          break;
        }
      }
      if (ks == k) {
        kase = 3;
      } else if (ks == p-1) {
        kase = 1;
      } else {
        kase = 2;
        k = ks;
      }
    }
    k++;

    // Perform the task indicated by kase.

    switch (kase) {

      // Deflate negligible s(p).

      case 1: {
        Real f(e(p-2));
        e(p-2) = 0.0;
        for (j = p-2; j >= k; j--) {
          Real t( hypot(s(j), f));
          Real cs(s(j)/t);
          Real sn(f/t);
          s(j) = t;
          if (j != k) {
            f = -sn*e(j-1);
            e(j-1) = cs*e(j-1);
          }
          if (wantv) {
            for (i = 0; i < n; i++) {
              t = cs*V(i, j) + sn*V(i, p-1);
              V(i, p-1) = -sn*V(i, j) + cs*V(i, p-1);
              V(i, j) = t;
            }
          }
        }
      }
        break;

        // Split at negligible s(k).

      case 2: {
        Real f(e(k-1));
        e(k-1) = 0.0;
        for (j = k; j < p; j++) {
          Real t(hypot(s(j), f));
          Real cs( s(j)/t);
          Real sn(f/t);
          s(j) = t;
          f = -sn*e(j);
          e(j) = cs*e(j);
          if (wantu) {
            for (i = 0; i < m; i++) {
              t = cs*U(i, j) + sn*U(i, k-1);
              U(i, k-1) = -sn*U(i, j) + cs*U(i, k-1);
              U(i, j) = t;
            }
          }
        }
      }
        break;

        // Perform one qr step.

      case 3: {

        // Calculate the shift.

        Real scale = std::max(std::max(std::max(std::max(
            std::abs(s(p-1)), std::abs(s(p-2))), std::abs(e(p-2))),
                                       std::abs(s(k))), std::abs(e(k)));
        Real sp = s(p-1)/scale;
        Real spm1 = s(p-2)/scale;
        Real epm1 = e(p-2)/scale;
        Real sk = s(k)/scale;
        Real ek = e(k)/scale;
        Real b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1)/2.0;
        Real c = (sp*epm1)*(sp*epm1);
        Real shift = 0.0;
        if ((b != 0.0) || (c != 0.0)) {
          shift = std::sqrt(b*b + c);
          if (b < 0.0) {
            shift = -shift;
          }
          shift = c/(b + shift);
        }
        Real f = (sk + sp)*(sk - sp) + shift;
        Real g = sk*ek;

        // Chase zeros.

        for (j = k; j < p-1; j++) {
          Real t = hypot(f, g);
          Real cs = f/t;
          Real sn = g/t;
          if (j != k) {
            e(j-1) = t;
          }
          f = cs*s(j) + sn*e(j);
          e(j) = cs*e(j) - sn*s(j);
          g = sn*s(j+1);
          s(j+1) = cs*s(j+1);
          if (wantv) {
            cblas_Xrot(n, vdata + j, vstride, vdata + j+1, vstride, cs, sn);
            /*for (i = 0; i < n; i++) {
              t = cs*vdata[i*vstride + j] + sn*vdata[i*vstride + j+1];  // t = cs*V(i, j) + sn*V(i, j+1);         // 13
              vdata[i*vstride + j+1] = -sn*vdata[i*vstride + j] + cs*vdata[i*vstride + j+1];  // V(i, j+1) = -sn*V(i, j) + cs*V(i, j+1); // 5
              vdata[i*vstride + j] = t;  // V(i, j) = t; // 4
              }*/
          }
          t = hypot(f, g);
          cs = f/t;
          sn = g/t;
          s(j) = t;
          f = cs*e(j) + sn*s(j+1);
          s(j+1) = -sn*e(j) + cs*s(j+1);
          g = sn*e(j+1);
          e(j+1) = cs*e(j+1);
          if (wantu && (j < m-1)) {
            cblas_Xrot(m, udata + j, ustride, udata + j+1, ustride, cs, sn);
            /*for (i = 0; i < m; i++) {
              t = cs*udata[i*ustride + j] + sn*udata[i*ustride + j+1];  // t = cs*U(i, j) + sn*U(i, j+1); // 7
              udata[i*ustride + j+1] = -sn*udata[i*ustride + j] +cs*udata[i*ustride + j+1];  // U(i, j+1) = -sn*U(i, j) + cs*U(i, j+1); // 8
              udata[i*ustride + j] = t;  // U(i, j) = t; // 1
              }*/
          }
        }
        e(p-2) = f;
        iter = iter + 1;
      }
        break;

        // Convergence.

      case 4: {

        // Make the singular values positive.

        if (s(k) <= 0.0) {
          s(k) = (s(k) < 0.0 ? -s(k) : 0.0);
          if (wantv) {
            for (i = 0; i <= pp; i++) {
              V(i, k) = -V(i, k);
            }
          }
        }

        // Order the singular values.

        while (k < pp) {
          if (s(k) >= s(k+1)) {
            break;
          }
          Real t = s(k);
          s(k) = s(k+1);
          s(k+1) = t;
          if (wantv && (k < n-1)) {
            for (i = 0; i < n; i++) {
              t = V(i, k+1); V(i, k+1) = V(i, k); V(i, k) = t;
            }
          }
          if (wantu && (k < m-1)) {
            for (i = 0; i < m; i++) {
              t = U(i, k+1); U(i, k+1) = U(i, k); U(i, k) = t;
            }
          }
          k++;
        }
        iter = 0;
        p--;
      }
        break;
    }
  }
  return true;
}

#endif // defined(HAVE_ATLAS) || defined(USE_KALDI_SVD)

} // namespace kaldi

#endif // KALDI_MATRIX_JAMA_SVD_H_
