// matrix/jama-eig.h

// Copyright 2009-2011 Microsoft Corporation 

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

// This file consists of a port and modification of materials from
//   JAMA: A Java Matrix Package
// under the following notice: This software is a cooperative product of
// The MathWorks and the National Institute of Standards and Technology (NIST)
// which has been released to the public.  This notice and the original code are
// available at http://math.nist.gov/javanumerics/jama/domain.notice



#ifndef KALDI_MATRIX_JAMA_EIG_H_
#define KALDI_MATRIX_JAMA_EIG_H_ 1

#include "matrix/kaldi-matrix.h"

namespace kaldi {

// This class is not to be used externally.  See the Eig function in the Matrix
// class in kaldi-matrix.h.  This is the external interface.

template<typename Real> class EigenvalueDecomposition {
  // This class is based on the EigenvalueDecomposition class from the JAMA
  // library (version 1.0.2).
 public:
  EigenvalueDecomposition(const MatrixBase<Real> &A);

  ~EigenvalueDecomposition();  // free memory.

  void GetV(MatrixBase<Real> *V_out) {  // V is what we call P externally; it's the matrix of
    // eigenvectors.
    KALDI_ASSERT(V_out->NumRows() == static_cast<MatrixIndexT>(n_)
                 && V_out->NumCols() == static_cast<MatrixIndexT>(n_));
    for (int i = 0; i < n_; i++)
      for (int j = 0; j < n_; j++)
        (*V_out)(i, j) = V(i, j);  // V(i, j) is member function.
  }
  void GetRealEigenvalues(VectorBase<Real> *r_out) {
    // returns real part of eigenvalues.
    KALDI_ASSERT(r_out->Dim() == static_cast<MatrixIndexT>(n_));
    for (int i = 0; i < n_; i++)
      (*r_out)(i) = d_[i];
  }
  void GetImagEigenvalues(VectorBase<Real> *i_out) {
    // returns imaginary part of eigenvalues.
    KALDI_ASSERT(i_out->Dim() == static_cast<MatrixIndexT>(n_));
    for (int i = 0; i < n_; i++)
      (*i_out)(i) = e_[i];
  }
 private:

  inline Real &H(int r, int c) { return H_[r*n_ + c]; }
  inline Real &V(int r, int c) { return V_[r*n_ + c]; }

  // complex division
  inline static void cdiv(Real xr, Real xi, Real yr, Real yi, Real *cdivr, Real *cdivi) {
    Real r, d;
    if (std::abs(yr) > std::abs(yi)) {
      r = yi/yr;
      d = yr + r*yi;
      *cdivr = (xr + r*xi)/d;
      *cdivi = (xi - r*xr)/d;
    } else {
      r = yr/yi;
      d = yi + r*yr;
      *cdivr = (r*xr + xi)/d;
      *cdivi = (r*xi - xr)/d;
    }
  }

  // Nonsymmetric reduction from Hessenberg to real Schur form.
  void Hqr2 ();


  int n_;  // matrix dimension.

  Real *d_, *e_;  // real and imaginary parts of eigenvalues.
  Real *V_;  // the eigenvectors (P in our external notation)
  Real *H_;  // the nonsymmetric Hessenberg form.
  Real *ort_;  // working storage for nonsymmetric algorithm.

  // Symmetric Householder reduction to tridiagonal form.
  void Tred2 ();

  // Symmetric tridiagonal QL algorithm.
  void Tql2 ();

  // Nonsymmetric reduction to Hessenberg form.
  void Orthes ();

};

template class EigenvalueDecomposition<float>;  // force instantiation.
template class EigenvalueDecomposition<double>;  // force instantiation.

template<typename Real> void  EigenvalueDecomposition<Real>::Tred2() {
  //  This is derived from the Algol procedures tred2 by
  //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
  //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutine in EISPACK.

  for (int j = 0; j < n_; j++) {
    d_[j] = V(n_-1, j);
  }

  // Householder reduction to tridiagonal form.

  for (int i = n_-1; i > 0; i--) {

    // Scale to avoid under/overflow.

    Real scale = 0.0;
    Real h = 0.0;
    for (int k = 0; k < i; k++) {
      scale = scale + std::abs(d_[k]);
    }
    if (scale == 0.0) {
      e_[i] = d_[i-1];
      for (int j = 0; j < i; j++) {
        d_[j] = V(i-1, j);
        V(i, j) = 0.0;
        V(j, i) = 0.0;
      }
    } else {

      // Generate Householder vector.

      for (int k = 0; k < i; k++) {
        d_[k] /= scale;
        h += d_[k] * d_[k];
      }
      Real f = d_[i-1];
      Real g = std::sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e_[i] = scale * g;
      h = h - f * g;
      d_[i-1] = f - g;
      for (int j = 0; j < i; j++) {
        e_[j] = 0.0;
      }

      // Apply similarity transformation to remaining columns.

      for (int j = 0; j < i; j++) {
        f = d_[j];
        V(j, i) = f;
        g =e_[j] + V(j, j) * f;
        for (int k = j+1; k <= i-1; k++) {
          g += V(k, j) * d_[k];
          e_[k] += V(k, j) * f;
        }
        e_[j] = g;
      }
      f = 0.0;
      for (int j = 0; j < i; j++) {
        e_[j] /= h;
        f += e_[j] * d_[j];
      }
      Real hh = f / (h + h);
      for (int j = 0; j < i; j++) {
        e_[j] -= hh * d_[j];
      }
      for (int j = 0; j < i; j++) {
        f = d_[j];
        g = e_[j];
        for (int k = j; k <= i-1; k++) {
          V(k, j) -= (f * e_[k] + g * d_[k]);
        }
        d_[j] = V(i-1, j);
        V(i, j) = 0.0;
      }
    }
    d_[i] = h;
  }

  // Accumulate transformations.

  for (int i = 0; i < n_-1; i++) {
    V(n_-1, i) = V(i, i);
    V(i, i) = 1.0;
    Real h = d_[i+1];
    if (h != 0.0) {
      for (int k = 0; k <= i; k++) {
        d_[k] = V(k, i+1) / h;
      }
      for (int j = 0; j <= i; j++) {
        Real g = 0.0;
        for (int k = 0; k <= i; k++) {
          g += V(k, i+1) * V(k, j);
        }
        for (int k = 0; k <= i; k++) {
          V(k, j) -= g * d_[k];
        }
      }
    }
    for (int k = 0; k <= i; k++) {
      V(k, i+1) = 0.0;
    }
  }
  for (int j = 0; j < n_; j++) {
    d_[j] = V(n_-1, j);
    V(n_-1, j) = 0.0;
  }
  V(n_-1, n_-1) = 1.0;
   e_[0] = 0.0;
}

template<typename Real> void EigenvalueDecomposition<Real>::Tql2() {
  //  This is derived from the Algol procedures tql2, by
  //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
  //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutine in EISPACK.

  for (int i = 1; i < n_; i++) {
     e_[i-1] = e_[i];
  }
   e_[n_-1] = 0.0;

  Real f = 0.0;
  Real tst1 = 0.0;
  Real eps = std::numeric_limits<Real>::epsilon();
  for (int l = 0; l < n_; l++) {

    // Find small subdiagonal element

    tst1 = std::max(tst1, std::abs(d_[l]) + std::abs(e_[l]));
    int m = l;
    while (m < n_) {
      if (std::abs(e_[m]) <= eps*tst1) {
        break;
      }
      m++;
    }

    // If m == l, d_[l] is an eigenvalue,
    // otherwise, iterate.

    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)

        // Compute implicit shift

        Real g = d_[l];
        Real p = (d_[l+1] - g) / (2.0 *e_[l]);
        Real r = Hypot(p, static_cast<Real>(1.0));  // This is a Kaldi version of hypot that works with templates.
        if (p < 0) {
          r = -r;
        }
        d_[l] =e_[l] / (p + r);
        d_[l+1] =e_[l] * (p + r);
        Real dl1 = d_[l+1];
        Real h = g - d_[l];
        for (int i = l+2; i < n_; i++) {
          d_[i] -= h;
        }
        f = f + h;

        // Implicit QL transformation.

        p = d_[m];
        Real c = 1.0;
        Real c2 = c;
        Real c3 = c;
        Real el1 =e_[l+1];
        Real s = 0.0;
        Real s2 = 0.0;
        for (int i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c *e_[i];
          h = c * p;
          r = Hypot(p, e_[i]);  // This is a Kaldi version of Hypot that works with templates.
          e_[i+1] = s * r;
          s =e_[i] / r;
          c = p / r;
          p = c * d_[i] - s * g;
          d_[i+1] = h + s * (c * g + s * d_[i]);

          // Accumulate transformation.

          for (int k = 0; k < n_; k++) {
            h = V(k, i+1);
            V(k, i+1) = s * V(k, i) + c * h;
            V(k, i) = c * V(k, i) - s * h;
          }
        }
        p = -s * s2 * c3 * el1 *e_[l] / dl1;
        e_[l] = s * p;
        d_[l] = c * p;

        // Check for convergence.

      } while (std::abs(e_[l]) > eps*tst1);
    }
    d_[l] = d_[l] + f;
    e_[l] = 0.0;
  }

  // Sort eigenvalues and corresponding vectors.

  for (int i = 0; i < n_-1; i++) {
    int k = i;
    Real p = d_[i];
    for (int j = i+1; j < n_; j++) {
      if (d_[j] < p) {
        k = j;
        p = d_[j];
      }
    }
    if (k != i) {
      d_[k] = d_[i];
      d_[i] = p;
      for (int j = 0; j < n_; j++) {
        p = V(j, i);
        V(j, i) = V(j, k);
        V(j, k) = p;
      }
    }
  }
}

template<typename Real>
void EigenvalueDecomposition<Real>::Orthes() {

  //  This is derived from the Algol procedures orthes and ortran,
  //  by Martin and Wilkinson, Handbook for Auto. Comp.,
  //  Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutines in EISPACK.

  int low = 0;
  int high = n_-1;

  for (int m = low+1; m <= high-1; m++) {

    // Scale column.

    Real scale = 0.0;
    for (int i = m; i <= high; i++) {
      scale = scale + std::abs(H(i, m-1));
    }
    if (scale != 0.0) {

      // Compute Householder transformation.

      Real h = 0.0;
      for (int i = high; i >= m; i--) {
        ort_[i] = H(i, m-1)/scale;
        h += ort_[i] * ort_[i];
      }
      Real g = std::sqrt(h);
      if (ort_[m] > 0) {
        g = -g;
      }
      h = h - ort_[m] * g;
      ort_[m] = ort_[m] - g;

      // Apply Householder similarity transformation
      // H = (I-u*u'/h)*H*(I-u*u')/h)

      for (int j = m; j < n_; j++) {
        Real f = 0.0;
        for (int i = high; i >= m; i--) {
          f += ort_[i]*H(i, j);
        }
        f = f/h;
        for (int i = m; i <= high; i++) {
          H(i, j) -= f*ort_[i];
        }
      }

      for (int i = 0; i <= high; i++) {
        Real f = 0.0;
        for (int j = high; j >= m; j--) {
          f += ort_[j]*H(i, j);
        }
        f = f/h;
        for (int j = m; j <= high; j++) {
          H(i, j) -= f*ort_[j];
        }
      }
      ort_[m] = scale*ort_[m];
      H(m, m-1) = scale*g;
    }
  }

  // Accumulate transformations (Algol's ortran).

  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < n_; j++) {
      V(i, j) = (i == j ? 1.0 : 0.0);
    }
  }

  for (int m = high-1; m >= low+1; m--) {
    if (H(m, m-1) != 0.0) {
      for (int i = m+1; i <= high; i++) {
        ort_[i] = H(i, m-1);
      }
      for (int j = m; j <= high; j++) {
        Real g = 0.0;
        for (int i = m; i <= high; i++) {
          g += ort_[i] * V(i, j);
        }
        // Double division avoids possible underflow
        g = (g / ort_[m]) / H(m, m-1);
        for (int i = m; i <= high; i++) {
          V(i, j) += g * ort_[i];
        }
      }
    }
  }
}

template<typename Real> void  EigenvalueDecomposition<Real>::Hqr2() {
  //  This is derived from the Algol procedure hqr2,
  //  by Martin and Wilkinson, Handbook for Auto. Comp.,
  //  Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutine in EISPACK.

  int nn = n_;
  int n = nn-1;
  int low = 0;
  int high = nn-1;
  Real eps = std::numeric_limits<Real>::epsilon();
  Real exshift = 0.0;
  Real p = 0, q = 0, r = 0, s = 0, z=0, t, w, x, y;

  // Store roots isolated by balanc and compute matrix norm

  Real norm = 0.0;
  for (int i = 0; i < nn; i++) {
    if (i < low || i > high) {
      d_[i] = H(i, i);
      e_[i] = 0.0;
    }
    for (int j = std::max(i-1, 0); j < nn; j++) {
      norm = norm + std::abs(H(i, j));
    }
  }

  // Outer loop over eigenvalue index

  int iter = 0;
  while (n >= low) {

    // Look for single small sub-diagonal element

    int l = n;
    while (l > low) {
      s = std::abs(H(l-1, l-1)) + std::abs(H(l, l));
      if (s == 0.0) {
        s = norm;
      }
      if (std::abs(H(l, l-1)) < eps * s) {
        break;
      }
      l--;
    }

    // Check for convergence
    // One root found

    if (l == n) {
      H(n, n) = H(n, n) + exshift;
      d_[n] = H(n, n);
      e_[n] = 0.0;
      n--;
      iter = 0;

      // Two roots found

    } else if (l == n-1) {
      w = H(n, n-1) * H(n-1, n);
      p = (H(n-1, n-1) - H(n, n)) / 2.0;
      q = p * p + w;
      z = std::sqrt(std::abs(q));
      H(n, n) = H(n, n) + exshift;
      H(n-1, n-1) = H(n-1, n-1) + exshift;
      x = H(n, n);

      // Real pair

      if (q >= 0) {
        if (p >= 0) {
          z = p + z;
        } else {
          z = p - z;
        }
        d_[n-1] = x + z;
        d_[n] = d_[n-1];
        if (z != 0.0) {
          d_[n] = x - w / z;
        }
        e_[n-1] = 0.0;
        e_[n] = 0.0;
        x = H(n, n-1);
        s = std::abs(x) + std::abs(z);
        p = x / s;
        q = z / s;
        r = std::sqrt(p * p+q * q);
        p = p / r;
        q = q / r;

        // Row modification

        for (int j = n-1; j < nn; j++) {
          z = H(n-1, j);
          H(n-1, j) = q * z + p * H(n, j);
          H(n, j) = q * H(n, j) - p * z;
        }

        // Column modification

        for (int i = 0; i <= n; i++) {
          z = H(i, n-1);
          H(i, n-1) = q * z + p * H(i, n);
          H(i, n) = q * H(i, n) - p * z;
        }

        // Accumulate transformations

        for (int i = low; i <= high; i++) {
          z = V(i, n-1);
          V(i, n-1) = q * z + p * V(i, n);
          V(i, n) = q * V(i, n) - p * z;
        }

        // Complex pair

      } else {
        d_[n-1] = x + p;
        d_[n] = x + p;
        e_[n-1] = z;
        e_[n] = -z;
      }
      n = n - 2;
      iter = 0;

      // No convergence yet

    } else {

      // Form shift

      x = H(n, n);
      y = 0.0;
      w = 0.0;
      if (l < n) {
        y = H(n-1, n-1);
        w = H(n, n-1) * H(n-1, n);
      }

      // Wilkinson's original ad hoc shift

      if (iter == 10) {
        exshift += x;
        for (int i = low; i <= n; i++) {
          H(i, i) -= x;
        }
        s = std::abs(H(n, n-1)) + std::abs(H(n-1, n-2));
        x = y = 0.75 * s;
        w = -0.4375 * s * s;
      }

      // MATLAB's new ad hoc shift

      if (iter == 30) {
        s = (y - x) / 2.0;
        s = s * s + w;
        if (s > 0) {
          s = std::sqrt(s);
          if (y < x) {
            s = -s;
          }
          s = x - w / ((y - x) / 2.0 + s);
          for (int i = low; i <= n; i++) {
            H(i, i) -= s;
          }
          exshift += s;
          x = y = w = 0.964;
        }
      }

      iter = iter + 1;   // (Could check iteration count here.)

      // Look for two consecutive small sub-diagonal elements

      int m = n-2;
      while (m >= l) {
        z = H(m, m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / H(m+1, m) + H(m, m+1);
        q = H(m+1, m+1) - z - r - s;
        r = H(m+2, m+1);
        s = std::abs(p) + std::abs(q) + std::abs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m == l) {
          break;
        }
        if (std::abs(H(m, m-1)) * (std::abs(q) + std::abs(r)) <
            eps * (std::abs(p) * (std::abs(H(m-1, m-1)) + std::abs(z) +
                                  std::abs(H(m+1, m+1))))) {
          break;
        }
        m--;
      }

      for (int i = m+2; i <= n; i++) {
        H(i, i-2) = 0.0;
        if (i > m+2) {
          H(i, i-3) = 0.0;
        }
      }

      // Double QR step involving rows l:n and columns m:n

      for (int k = m; k <= n-1; k++) {
        bool notlast = (k != n-1);
        if (k != m) {
          p = H(k, k-1);
          q = H(k+1, k-1);
          r = (notlast ? H(k+2, k-1) : 0.0);
          x = std::abs(p) + std::abs(q) + std::abs(r);
          if (x != 0.0) {
            p = p / x;
            q = q / x;
            r = r / x;
          }
        }
        if (x == 0.0) {
          break;
        }
        s = std::sqrt(p * p + q * q + r * r);
        if (p < 0) {
          s = -s;
        }
        if (s != 0) {
          if (k != m) {
            H(k, k-1) = -s * x;
          } else if (l != m) {
            H(k, k-1) = -H(k, k-1);
          }
          p = p + s;
          x = p / s;
          y = q / s;
          z = r / s;
          q = q / p;
          r = r / p;

          // Row modification

          for (int j = k; j < nn; j++) {
            p = H(k, j) + q * H(k+1, j);
            if (notlast) {
              p = p + r * H(k+2, j);
              H(k+2, j) = H(k+2, j) - p * z;
            }
            H(k, j) = H(k, j) - p * x;
            H(k+1, j) = H(k+1, j) - p * y;
          }

          // Column modification

          for (int i = 0; i <= std::min(n, k+3); i++) {
            p = x * H(i, k) + y * H(i, k+1);
            if (notlast) {
              p = p + z * H(i, k+2);
              H(i, k+2) = H(i, k+2) - p * r;
            }
            H(i, k) = H(i, k) - p;
            H(i, k+1) = H(i, k+1) - p * q;
          }

          // Accumulate transformations

          for (int i = low; i <= high; i++) {
            p = x * V(i, k) + y * V(i, k+1);
            if (notlast) {
              p = p + z * V(i, k+2);
              V(i, k+2) = V(i, k+2) - p * r;
            }
            V(i, k) = V(i, k) - p;
            V(i, k+1) = V(i, k+1) - p * q;
          }
        }  // (s != 0)
      }  // k loop
    }  // check convergence
  }  // while (n >= low)

  // Backsubstitute to find vectors of upper triangular form

  if (norm == 0.0) {
    return;
  }

  for (n = nn-1; n >= 0; n--) {
    p = d_[n];
    q = e_[n];

    // Real vector

    if (q == 0) {
      int l = n;
      H(n, n) = 1.0;
      for (int i = n-1; i >= 0; i--) {
        w = H(i, i) - p;
        r = 0.0;
        for (int j = l; j <= n; j++) {
          r = r + H(i, j) * H(j, n);
        }
        if (e_[i] < 0.0) {
          z = w;
          s = r;
        } else {
          l = i;
          if (e_[i] == 0.0) {
            if (w != 0.0) {
              H(i, n) = -r / w;
            } else {
              H(i, n) = -r / (eps * norm);
            }

            // Solve real equations

          } else {
            x = H(i, i+1);
            y = H(i+1, i);
            q = (d_[i] - p) * (d_[i] - p) +e_[i] *e_[i];
            t = (x * s - z * r) / q;
            H(i, n) = t;
            if (std::abs(x) > std::abs(z)) {
              H(i+1, n) = (-r - w * t) / x;
            } else {
              H(i+1, n) = (-s - y * t) / z;
            }
          }

          // Overflow control

          t = std::abs(H(i, n));
          if ((eps * t) * t > 1) {
            for (int j = i; j <= n; j++) {
              H(j, n) = H(j, n) / t;
            }
          }
        }
      }

      // Complex vector

    } else if (q < 0) {
      int l = n-1;

      // Last vector component imaginary so matrix is triangular

      if (std::abs(H(n, n-1)) > std::abs(H(n-1, n))) {
        H(n-1, n-1) = q / H(n, n-1);
        H(n-1, n) = -(H(n, n) - p) / H(n, n-1);
      } else {
        Real cdivr, cdivi;
        cdiv(0.0, -H(n-1, n), H(n-1, n-1)-p, q, &cdivr, &cdivi);
        H(n-1, n-1) = cdivr;
        H(n-1, n) = cdivi;
      }
      H(n, n-1) = 0.0;
      H(n, n) = 1.0;
      for (int i = n-2; i >= 0; i--) {
        Real ra, sa, vr, vi;
        ra = 0.0;
        sa = 0.0;
        for (int j = l; j <= n; j++) {
          ra = ra + H(i, j) * H(j, n-1);
          sa = sa + H(i, j) * H(j, n);
        }
        w = H(i, i) - p;

        if (e_[i] < 0.0) {
          z = w;
          r = ra;
          s = sa;
        } else {
          l = i;
          if (e_[i] == 0) {
            Real cdivr, cdivi;
            cdiv(-ra, -sa, w, q, &cdivr, &cdivi);
            H(i, n-1) = cdivr;
            H(i, n) = cdivi;
          } else {
            Real cdivr, cdivi;
            // Solve complex equations

            x = H(i, i+1);
            y = H(i+1, i);
            vr = (d_[i] - p) * (d_[i] - p) +e_[i] *e_[i] - q * q;
            vi = (d_[i] - p) * 2.0 * q;
            if (vr == 0.0 && vi == 0.0) {
              vr = eps * norm * (std::abs(w) + std::abs(q) +
                                 std::abs(x) + std::abs(y) + std::abs(z));
            }
            cdiv(x*r-z*ra+q*sa, x*s-z*sa-q*ra, vr, vi, &cdivr, &cdivi);
            H(i, n-1) = cdivr;
            H(i, n) = cdivi;
            if (std::abs(x) > (std::abs(z) + std::abs(q))) {
              H(i+1, n-1) = (-ra - w * H(i, n-1) + q * H(i, n)) / x;
              H(i+1, n) = (-sa - w * H(i, n) - q * H(i, n-1)) / x;
            } else {
              cdiv(-r-y*H(i, n-1), -s-y*H(i, n), z, q, &cdivr, &cdivi);
              H(i+1, n-1) = cdivr;
              H(i+1, n) = cdivi;
            }
          }

          // Overflow control

          t = std::max(std::abs(H(i, n-1)), std::abs(H(i, n)));
          if ((eps * t) * t > 1) {
            for (int j = i; j <= n; j++) {
              H(j, n-1) = H(j, n-1) / t;
              H(j, n) = H(j, n) / t;
            }
          }
        }
      }
    }
  }

  // Vectors of isolated roots

  for (int i = 0; i < nn; i++) {
    if (i < low || i > high) {
      for (int j = i; j < nn; j++) {
        V(i, j) = H(i, j);
      }
    }
  }

  // Back transformation to get eigenvectors of original matrix

  for (int j = nn-1; j >= low; j--) {
    for (int i = low; i <= high; i++) {
      z = 0.0;
      for (int k = low; k <= std::min(j, high); k++) {
        z = z + V(i, k) * H(k, j);
      }
      V(i, j) = z;
    }
  }
}

template<typename Real>
EigenvalueDecomposition<Real>::EigenvalueDecomposition(const MatrixBase<Real> &A) {
  KALDI_ASSERT(A.NumCols() == A.NumRows() && A.NumCols() >= 1);
  n_ = A.NumRows();
  V_ = new Real[n_*n_];
  d_ = new Real[n_];
  e_ = new Real[n_];
  H_ = NULL;
  ort_ = NULL;
  if (A.IsSymmetric(0.0)) {

    for (int i = 0; i < n_; i++)
      for (int j = 0; j < n_; j++)
        V(i, j) = A(i, j);  // Note that V(i, j) is a member function; A(i, j) is an operator
    // of the matrix A.
    // Tridiagonalize.
    Tred2();

    // Diagonalize.
    Tql2();
  } else {
    H_ = new Real[n_*n_];
    ort_ = new Real[n_];
    for (int i = 0; i < n_; i++)
      for (int j = 0; j < n_; j++)
        H(i, j) = A(i, j);  // as before: H is member function, A(i, j) is operator of matrix.

    // Reduce to Hessenberg form.
    Orthes();

    // Reduce Hessenberg to real Schur form.
    Hqr2();
  }
}

template<typename Real>
EigenvalueDecomposition<Real>::~EigenvalueDecomposition() {
  delete [] d_;
  delete [] e_;
  delete [] V_;
  delete [] H_;
  delete [] ort_;
}

// see function MatrixBase<Real>::Eig in kaldi-matrix.cc


} // namespace kaldi

#endif // KALDI_MATRIX_JAMA_EIG_H_
