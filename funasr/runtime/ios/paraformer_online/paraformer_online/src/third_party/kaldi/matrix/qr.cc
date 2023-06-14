// matrix/qr.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#include <limits>

#include "matrix/sp-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-functions.h"
#include "matrix/cblas-wrappers.h"

// This file contains an implementation of the Symmetric QR Algorithm
// for the symmetric eigenvalue problem.  See Golub and Van Loan,
// 3rd ed., Algorithm 8.3.3.

namespace kaldi {


/* This is from Golub and Van Loan 3rd ed., sec. 5.1.3,
   p210.
   x is the input of dimenson 'dim', v is the output of dimension
   dim, and beta is a scalar. Note: we use zero-based
   not one-based indexing. */
/*
// We are commenting out the function below ("House") because it's not
// needed, but we keep it just to show how we came up with HouseBackward.
template<typename Real>
void House(MatrixIndexT dim, const Real *x, Real *v, Real *beta) {
  KALDI_ASSERT(dim > 0);
  // To avoid overflow, we first compute the max of x_ (or
  // one if that's zero, and we'll replace "x" by x/max(x_i)
  // below.  The householder vector is anyway invariant to
  // the magnitude of x.  We could actually avoid this extra loop
  // over x if we wanted to be a bit smarter, but anyway this
  // doesn't dominate the O(N) performance of the algorithm.
  Real s; // s is a scale on x.
  {
    Real max_x = std::numeric_limits<Real>::min();
    for (MatrixIndexT i = 0; i < dim; i++)
      max_x = std::max(max_x, (x[i] < 0 ? -x[i] : x[i]));
    if (max_x == 0.0) max_x = 1.0;
    s = 1.0 / max_x;
  }
  
  Real sigma = 0.0;
  v[0] = 1.0;
  for (MatrixIndexT i = 1; i < dim; i++) {
    sigma += (x[i]*s) * (x[i]*s);
    v[i] = x[i]*s;
  }
  if (sigma == 0.0) *beta = 0.0;
  else {
    // When we say x1 = x[0], we reference the one-based indexing
    // in Golub and Van Loan.
    Real x1 = x[0] * s, mu = std::sqrt(x1*x1 + sigma);
    if (x1 <= 0) {
      v[0] = x1 - mu;
    } else {
      v[0] = -sigma / (x1 + mu);
      KALDI_ASSERT(KALDI_ISFINITE(v[dim-1]));      
    }
    Real v1 = v[0];
    Real v1sq = v1 * v1;
    *beta = 2 * v1sq / (sigma + v1sq);
    Real inv_v1 = 1.0 / v1;
    if (KALDI_ISINF(inv_v1)) {
      // can happen if v1 is denormal.
      KALDI_ASSERT(v1 == v1 && v1 != 0.0);
      for (MatrixIndexT i = 0; i < dim; i++) v[i] /= v1;
    } else {
      cblas_Xscal(dim, inv_v1, v, 1);
    }
    if (KALDI_ISNAN(inv_v1)) {
      KALDI_ERR << "NaN encountered in HouseBackward";
    }
  }
}
*/

// This is a backward version of the "House" routine above:
// backward because it's the last index, not the first index of
// the vector that is "special".  This is convenient in
// the Tridiagonalize routine that uses reversed indexes for
// compatibility with the packed lower triangular format.
template<typename Real>
void HouseBackward(MatrixIndexT dim, const Real *x, Real *v, Real *beta) {
  KALDI_ASSERT(dim > 0);
  // To avoid overflow, we first compute the max of x_ (or
  // one if that's zero, and we'll replace "x" by x/max(x_i)
  // below.  The householder vector is anyway invariant to
  // the magnitude of x.  We could actually avoid this extra loop
  // over x if we wanted to be a bit smarter, but anyway this
  // doesn't dominate the O(N) performance of the algorithm.
  Real s; // s is a scale on x.
  {
    Real max_x = std::numeric_limits<Real>::min();
    for (MatrixIndexT i = 0; i < dim; i++)
      max_x = std::max(max_x, (x[i] < 0 ? -x[i] : x[i]));
    s = 1.0 / max_x;
  }
  Real sigma = 0.0;
  v[dim-1] = 1.0;
  for (MatrixIndexT i = 0; i + 1  < dim; i++) {
    sigma += (x[i] * s) * (x[i] * s);
    v[i] = x[i] * s;
  }
  KALDI_ASSERT(KALDI_ISFINITE(sigma) &&
               "Tridiagonalizing matrix that is too large or has NaNs.");
  if (sigma == 0.0) *beta = 0.0;
  else {
    Real x1 = x[dim-1] * s, mu = std::sqrt(x1 * x1 + sigma);
    if (x1 <= 0) {
      v[dim-1] = x1 - mu;
    } else {
      v[dim-1] = -sigma / (x1 + mu);
      KALDI_ASSERT(KALDI_ISFINITE(v[dim-1]));
    }
    Real v1 = v[dim-1];
    Real v1sq = v1 * v1;
    *beta = 2 * v1sq / (sigma + v1sq);
    Real inv_v1 = 1.0 / v1;
    if (KALDI_ISINF(inv_v1)) {
      // can happen if v1 is denormal.
      KALDI_ASSERT(v1 == v1 && v1 != 0.0);
      for (MatrixIndexT i = 0; i < dim; i++) v[i] /= v1;
    } else {
      cblas_Xscal(dim, inv_v1, v, 1);
    }
    if (KALDI_ISNAN(inv_v1)) {
      KALDI_ERR << "NaN encountered in HouseBackward";
    }
  }
}


/**
   This routine tridiagonalizes *this.  C.f. Golub and Van Loan 3rd ed., sec.
   8.3.1 (p415).  We reverse the order of the indices as it's more natural
   with packed lower-triangular matrices to do it this way.  There's also
   a shift from one-based to zero-based indexing, so the index
   k is transformed k -> n - k, and a corresponding transpose...
   
   Let the original *this be A.  This algorithms replaces *this with
   a tridiagonal matrix T such that T = Q A Q^T for an orthogonal Q.
   Caution: Q is transposed vs. Golub and Van Loan.
   If Q != NULL it outputs Q. 
*/
template<typename Real>
void SpMatrix<Real>::Tridiagonalize(MatrixBase<Real> *Q) {
  MatrixIndexT n = this->NumRows();
  KALDI_ASSERT(Q == NULL || (Q->NumRows() == n &&
                             Q->NumCols() == n));
  if (Q != NULL) Q->SetUnit();
  Real *data = this->Data();
  Real *qdata = (Q == NULL ? NULL : Q->Data());
  MatrixIndexT qstride = (Q == NULL ? 0 : Q->Stride());
  Vector<Real> tmp_v(n-1), tmp_p(n);
  Real beta, *v = tmp_v.Data(), *p = tmp_p.Data(), *w = p, *x = p;
  for (MatrixIndexT k = n-1; k >= 2; k--) {
    MatrixIndexT ksize = ((k+1)*k)/2;
    // ksize is the packed size of the lower-triangular matrix of size k,
    // which is the size of "all rows previous to this one."
    Real *Arow = data + ksize; // In Golub+Van Loan it was A(k+1:n, k), we
    // have Arow = A(k, 0:k-1).
    HouseBackward(k, Arow, v, &beta); // sets v and beta.
    cblas_Xspmv(k, beta, data, v, 1, 0.0, p, 1); // p = beta * A(0:k-1,0:k-1) v
    Real minus_half_beta_pv = -0.5 * beta * cblas_Xdot(k, p, 1, v, 1);
    cblas_Xaxpy(k, minus_half_beta_pv, v, 1, w, 1); // w = p - (beta p^T v/2) v;
    // this relies on the fact that w and p are the same pointer.
    // We're doing A(k, k-1) = ||Arow||.  It happens that this element
    // is indexed at ksize + k - 1 in the packed lower-triangular format.
    data[ksize + k - 1] = std::sqrt(cblas_Xdot(k, Arow, 1, Arow, 1));
    for (MatrixIndexT i = 0; i + 1 < k; i++)
      data[ksize + i] = 0; // This is not in Golub and Van Loan but is
    // necessary if we're not using parts of A to store the Householder
    // vectors.
    // We're doing A(0:k-1,0:k-1) -= (v w' + w v')
    cblas_Xspr2(k, -1.0, v, 1, w, 1, data);
    if (Q != NULL) { // C.f. Golub, Q is H_1 .. H_n-2... in this
      // case we apply them in the opposite order so it's H_n-1 .. H_1,
      // but also Q is transposed so we really have Q = H_1 .. H_n-1.
      // It's a double negative.    
      // Anyway, we left-multiply Q by each one.  The H_n would each be
      // diag(I + beta v v', I) but we don't ever touch the last dims.
      // We do (in Matlab notation):
      // Q(0:k-1,:) = (I - beta v v') * Q, i.e.:
      // Q(:,0:i-1) += -beta v (v' Q(:,0:k-1)v .. let x = -beta Q(0:k-1,:)^T v.
      cblas_Xgemv(kTrans, k, n, -beta, qdata, qstride, v, 1, 0.0, x, 1);
      // now x = -beta Q(:,0:k-1) v.
      // The next line does: Q(:,0:k-1) += v x'.
      cblas_Xger(k, n, 1.0, v, 1, x, 1, qdata, qstride);
    }
  }
}

// Instantiate these functions, as it wasn't implemented in sp-matrix.cc
// where we instantiated the whole class.
template
void SpMatrix<float>::Tridiagonalize(MatrixBase<float> *Q);
template
void SpMatrix<double>::Tridiagonalize(MatrixBase<double> *Q);

/// Create Givens rotations, as in Golub and Van Loan 3rd ed., page 216.
template<typename Real>
inline void Givens(Real a, Real b, Real *c, Real *s) {
  if (b == 0) {
    *c = 1;
    *s = 0;
  } else {
    if (std::abs(b) > std::abs(a)) {
      Real tau = -a / b;
      *s = 1 / std::sqrt(1 + tau*tau);
      *c = *s * tau;
    } else {
      Real tau = -b / a;
      *c = 1 / std::sqrt(1 + tau*tau);
      *s = *c * tau;
    }
  }
}


// Some internal code for the QR algorithm: one "QR step".
// This is Golub and Van Loan 3rd ed., Algorithm 8.3.2 "Implicit Symmetric QR step
// with Wilkinson shift."  A couple of differences: this code is
// in zero based arithmetic, and we represent Q transposed from
// their Q for memory locality with row-major-indexed matrices.
template <typename Real>
void QrStep(MatrixIndexT n,
            Real *diag,
            Real *off_diag,
            MatrixBase<Real> *Q) {
  KALDI_ASSERT(n >= 2);
  // below, "scale" could be any number; we introduce it to keep the
  // floating point quantities within a good range.
  Real   d = (diag[n-2] - diag[n-1]) / 2.0,
      t = off_diag[n-2],
      inv_scale = std::max(std::max(std::abs(d), std::abs(t)),
                           std::numeric_limits<Real>::min()),
      scale = 1.0 / inv_scale,
      d_scaled = d * scale,
      off_diag_n2_scaled = off_diag[n-2] * scale,
      t2_n_n1_scaled = off_diag_n2_scaled * off_diag_n2_scaled,
      sgn_d = (d > 0.0 ? 1.0 : -1.0),
      mu = diag[n-1] - inv_scale * t2_n_n1_scaled /
      (d_scaled + sgn_d * std::sqrt(d_scaled * d_scaled + t2_n_n1_scaled)),
      x = diag[0] - mu,
      z = off_diag[0];
  KALDI_ASSERT(KALDI_ISFINITE(x));
  Real *Qdata = (Q == NULL ? NULL : Q->Data());
  MatrixIndexT Qstride = (Q == NULL ? 0 : Q->Stride()),
      Qcols = (Q == NULL ? 0 : Q->NumCols());
  for (MatrixIndexT k = 0; k < n-1; k++) {
    Real c, s;
    Givens(x, z, &c, &s);
    // Rotate dimensions k and k+1 with the Givens matrix G, as
    // T <== G^T T G.
    // In 2d, a Givens matrix is [ c s; -s c ].  Forget about
    // the dimension-indexing issues and assume we have a 2x2
    // symmetric matrix [ p q ; q r ]
    // We ask our friends at Wolfram Alpha about
    // { { c, -s}, {s, c} } * { {p, q}, {q, r} } * { { c, s}, {-s, c} }
    // Interpreting the result as [ p', q' ; q', r ]
    //    p' = c (c p - s q) - s (c q - s r)
    //    q' = s (c p - s q) + c (c q - s r)
    //    r' = s (s p + c q) + c (s q + c r)
    Real p = diag[k], q = off_diag[k], r = diag[k+1];
    // p is element k,k; r is element k+1,k+1; q is element k,k+1 or k+1,k.
    // We'll let the compiler optimize this.
    diag[k] = c * (c*p - s*q) - s * (c*q - s*r);
    off_diag[k] = s * (c*p - s*q) + c * (c*q - s*r);
    diag[k+1] = s * (s*p + c*q) + c * (s*q + c*r);

    // We also have some other elements to think of that
    // got rotated in a simpler way: if k>0,
    // then element (k, k-1) and (k+1, k-1) get rotated.  Here,
    // element k+1, k-1 will be present as z; it's the out-of-band
    // element that we remembered from last time.  This is
    // on the left as it's the row indexes that differ, so think of
    // this as being premultiplied by G^T.  In fact we're multiplying
    // T by in some sense the opposite/transpose of the Givens rotation.
    if (k > 0) { // Note, in rotations, going backward, (x,y) -> ((cx - sy), (sx + cy))
      Real &elem_k_km1 = off_diag[k-1],
          elem_kp1_km1 = z; // , tmp = elem_k_km1;
      elem_k_km1 = c*elem_k_km1 - s*elem_kp1_km1;
      // The next line will set elem_kp1_km1 to zero and we'll never access this
      // value, so we comment it out.
      // elem_kp1_km1 = s*tmp + c*elem_kp1_km1;
    }
    if (Q != NULL)
      cblas_Xrot(Qcols, Qdata + k*Qstride, 1,
                 Qdata + (k+1)*Qstride, 1, c, -s);
    if (k < n-2) {
      // Next is the elements (k+2, k) and (k+2, k-1), to be rotated, again
      // backwards.
      Real &elem_kp2_k = z, 
          &elem_kp2_kp1 = off_diag[k+1];
      // Note: elem_kp2_k == z would start off as zero because it's
       // two off the diagonal, and not been touched yet.  Therefore
      // we eliminate it in expressions below, commenting it out.
      // If we didn't do this we should set it to zero first.
      elem_kp2_k =  - s * elem_kp2_kp1; // + c*elem_kp2_k
      elem_kp2_kp1 =  c * elem_kp2_kp1; // + s*elem_kp2_k (original value).
      // The next part is from the algorithm they describe: x = t_{k+1,k}
      x = off_diag[k];
    }
  }
}


// Internal code for the QR algorithm, where the diagonal
// and off-diagonal of the symmetric matrix are represented as
// vectors of length n and n-1.
template <typename Real>
void QrInternal(MatrixIndexT n,
                Real *diag,
                Real *off_diag,
                MatrixBase<Real> *Q) {
  KALDI_ASSERT(Q == NULL || Q->NumCols() == n); // We may
  // later relax the condition that Q->NumCols() == n.

  MatrixIndexT counter = 0, max_iters = 500 + 4*n, // Should never take this many iters.
      large_iters = 100 + 2*n;
  Real epsilon = (pow(2.0, sizeof(Real) == 4 ? -23.0 : -52.0));
  
  for (; counter < max_iters; counter++) { // this takes the place of "until
                                           // q=n"... we'll break out of the
                                           // loop when we converge.
    if (counter == large_iters ||
        (counter > large_iters && (counter - large_iters) % 50 == 0)) {
      KALDI_WARN << "Took " << counter
                 << " iterations in QR (dim is " << n << "), doubling epsilon.";
      SubVector<Real> d(diag, n), o(off_diag, n-1);
      KALDI_WARN << "Diag, off-diag are " << d << " and " << o;
      epsilon *= 2.0;
    }
    for (MatrixIndexT i = 0; i+1 < n; i++) {
      if (std::abs(off_diag[i]) <= epsilon *
          (std::abs(diag[i]) + std::abs(diag[i+1])))
        off_diag[i] = 0.0;
    }
    // The next code works out p, q, and npq which is n - p - q.
    // For the definitions of q and p, see Golub and Van Loan; we 
    // partition the n dims into pieces of size (p, n-p-q, q) where
    // the part of size q is diagonal and the part of size n-p-p is
    // "unreduced", i.e. has no zero off-diagonal elements.
    MatrixIndexT q = 0;
    // Note: below, "n-q < 2" should more clearly be "n-2-q < 0", but that
    // causes problems if MatrixIndexT is unsigned.
    while (q < n && (n-q < 2 || off_diag[n-2-q] == 0.0))
      q++;
    if (q == n) break; // we're done.  It's diagonal.
    KALDI_ASSERT(n - q >= 2);
    MatrixIndexT npq = 2; // Value of n - p - q, where n - p - q must be
    // unreduced.  This is the size of "middle" band of elements.  If q != n,
    // we must have hit a nonzero off-diag element, so the size of this
    // band must be at least two.
    while (npq + q < n && (n-q-npq-1 < 0 || off_diag[n-q-npq-1] != 0.0))
      npq++;
    MatrixIndexT p = n - q - npq;
    { // Checks.
      for (MatrixIndexT i = 0; i+1 < npq; i++)
        KALDI_ASSERT(off_diag[p + i] != 0.0);
      for (MatrixIndexT i = 0; i+1 < q; i++)
        KALDI_ASSERT(off_diag[p + npq - 1 + i] == 0.0);
      if (p > 1) // Something must have stopped npq from growing further..
        KALDI_ASSERT(off_diag[p-1] == 0.0); // so last off-diag elem in
      // group of size p must be zero.
    }

    if (Q != NULL) {
      // Do one QR step on the middle part of Q only.
      // Qpart will be a subset of the rows of Q.
      SubMatrix<Real> Qpart(*Q, p, npq, 0, Q->NumCols());
      QrStep(npq, diag + p, off_diag + p, &Qpart);
    } else {
      QrStep(npq, diag + p, off_diag + p,
             static_cast<MatrixBase<Real>*>(NULL));
    }      
  }
  if (counter == max_iters) {
    KALDI_WARN << "Failure to converge in QR algorithm. "
               << "Exiting with partial output.";
  }
}


/**
   This is the symmetric QR algorithm, from Golub and Van Loan 3rd ed., Algorithm
   8.3.3.  Q is transposed w.r.t. there, though.
*/
template <typename Real>
void SpMatrix<Real>::Qr(MatrixBase<Real> *Q) {
  KALDI_ASSERT(this->IsTridiagonal());
  // We envisage that Q would be square but we don't check for this,
  // as there are situations where you might not want this.
  KALDI_ASSERT(Q == NULL || Q->NumRows() == this->NumRows());
  // Note: the first couple of lines of the algorithm they give would be done
  // outside of this function, by calling Tridiagonalize().

  MatrixIndexT n = this->NumRows();
  Vector<Real> diag(n), off_diag(n-1);
  for (MatrixIndexT i = 0; i < n; i++) {
    diag(i) = (*this)(i, i);
    if (i > 0) off_diag(i-1) = (*this)(i, i-1);
  }
  QrInternal(n, diag.Data(), off_diag.Data(), Q);
  // Now set *this to the value represented by diag and off_diag.
  this->SetZero();
  for (MatrixIndexT i = 0; i < n; i++) {
    (*this)(i, i) = diag(i);
    if (i > 0) (*this)(i, i-1) = off_diag(i-1);
  }
}

template<typename Real>
void SpMatrix<Real>::Eig(VectorBase<Real> *s, MatrixBase<Real> *P) const {
  MatrixIndexT dim = this->NumRows();
  KALDI_ASSERT(s->Dim() == dim);
  KALDI_ASSERT(P == NULL || (P->NumRows() == dim && P->NumCols() == dim));

  SpMatrix<Real> A(*this); // Copy *this, since the tridiagonalization
  // and QR decomposition are destructive.
  // Note: for efficiency of memory access, the tridiagonalization
  // algorithm makes the *rows* of P the eigenvectors, not the columns.
  // We'll transpose P before we exit.
  // Also note: P may be null if you don't want the eigenvectors.  This
  // will make this function more efficient.

  A.Tridiagonalize(P); // Tridiagonalizes.
  A.Qr(P); // Diagonalizes.
  if(P) P->Transpose();
  s->CopyDiagFromPacked(A);
}


template<typename Real>
void SpMatrix<Real>::TopEigs(VectorBase<Real> *s, MatrixBase<Real> *P,
                             MatrixIndexT lanczos_dim) const {
  const SpMatrix<Real> &S(*this); // call this "S" for easy notation.
  MatrixIndexT eig_dim = s->Dim(); // Space of dim we want to retain.
  if (lanczos_dim <= 0)
    lanczos_dim = std::max(eig_dim + 50, eig_dim + eig_dim/2);
  MatrixIndexT dim = this->NumRows();
  if (lanczos_dim >= dim) {
    // There would be no speed advantage in using this method, so just
    // use the regular approach.
    Vector<Real> s_tmp(dim);
    Matrix<Real> P_tmp(dim, dim);
    this->Eig(&s_tmp, &P_tmp);
    SortSvd(&s_tmp, &P_tmp);
    s->CopyFromVec(s_tmp.Range(0, eig_dim));
    P->CopyFromMat(P_tmp.Range(0, dim, 0, eig_dim));
    return;
  }
  KALDI_ASSERT(eig_dim <= dim && eig_dim > 0);
  KALDI_ASSERT(P->NumRows() == dim && P->NumCols() == eig_dim); // each column
  // is one eigenvector.

  Matrix<Real> Q(lanczos_dim, dim); // The rows of Q will be the
  // orthogonal vectors of the Krylov subspace.

  SpMatrix<Real> T(lanczos_dim); // This will be equal to Q S Q^T,
  // i.e. *this projected into the Krylov subspace.  Note: only the
  // diagonal and off-diagonal fo T are nonzero, i.e. it's tridiagonal,
  // but we don't have access to the low-level algorithms that work
  // on that type of matrix (since we want to use ATLAS).  So we just
  // do normal SVD, on a full matrix; it won't typically dominate.

  Q.Row(0).SetRandn();
  Q.Row(0).Scale(1.0 / Q.Row(0).Norm(2));
  for (MatrixIndexT d = 0; d < lanczos_dim; d++) {
    Vector<Real> r(dim);
    r.AddSpVec(1.0, S, Q.Row(d), 0.0);
    // r = S * q_d
    MatrixIndexT counter = 0;
    Real end_prod;
    while (1) { // Normally we'll do this loop only once:
      // we repeat to handle cases where r gets very much smaller
      // and we want to orthogonalize again.
      // We do "full orthogonalization" to preserve stability,
      // even though this is usually a waste of time.
      Real start_prod = VecVec(r, r);
      for (SignedMatrixIndexT e = d; e >= 0; e--) { // e must be signed!
        SubVector<Real> q_e(Q, e);
        Real prod = VecVec(r, q_e);
        if (counter == 0 && static_cast<MatrixIndexT>(e) + 1 >= d) // Keep T tridiagonal, which
          T(d, e) = prod; // mathematically speaking, it is.
        r.AddVec(-prod, q_e); // Subtract component in q_e.
      }
      if (d+1 == lanczos_dim) break;
      end_prod = VecVec(r, r);
      if (end_prod <= 0.1 * start_prod) {
        // also handles case where both are 0.
        // We're not confident any more that it's completely
        // orthogonal to the rest so we want to re-do.
        if (end_prod == 0.0)
          r.SetRandn(); // "Restarting".
        counter++;
        if (counter > 100)
          KALDI_ERR << "Loop detected in Lanczos iteration.";
      } else {
        break;
      }
    }
    if (d+1 != lanczos_dim) {
      // OK, at this point we're satisfied that r is orthogonal
      // to all previous rows.
      KALDI_ASSERT(end_prod != 0.0); // should have looped.
      r.Scale(1.0 / std::sqrt(end_prod)); // make it unit.
      Q.Row(d+1).CopyFromVec(r);
    }
  }

  Matrix<Real> R(lanczos_dim, lanczos_dim);  
  R.SetUnit();
  T.Qr(&R); // Diagonalizes T.
  Vector<Real> s_tmp(lanczos_dim);
  s_tmp.CopyDiagFromSp(T);  

  // Now T = R * diag(s_tmp) * R^T.
  // The next call sorts the elements of s from greatest to least absolute value,
  // and moves around the rows of R in the corresponding way.  This picks out
  // the largest (absolute) eigenvalues.
  SortSvd(&s_tmp, static_cast<Matrix<Real>*>(NULL), &R);
  // Keep only the initial rows of R, those corresponding to greatest (absolute)
  // eigenvalues.
  SubMatrix<Real> Rsub(R, 0, eig_dim, 0, lanczos_dim);
  SubVector<Real> s_sub(s_tmp, 0, eig_dim);
  s->CopyFromVec(s_sub);
      
  // For working out what to do now, just assume the other eigenvalues were
  // zero.  This is just for purposes of knowing how to get the result, and
  // not getting things wrongly transposed.
  // We have T = Rsub^T * diag(s_sub) * Rsub.
  // Now, T = Q S Q^T, with Q orthogonal,  so S = Q^T T Q = Q^T Rsub^T * diag(s) * Rsub * Q.
  // The output is P and we want S = P * diag(s) * P^T, so we need P = Q^T Rsub^T.
  P->AddMatMat(1.0, Q, kTrans, Rsub, kTrans, 0.0);
}


// Instantiate the templates for Eig and TopEig.
template
void SpMatrix<float>::Eig(VectorBase<float>*, MatrixBase<float>*) const;
template
void SpMatrix<double>::Eig(VectorBase<double>*, MatrixBase<double>*) const;

template
void SpMatrix<float>::TopEigs(VectorBase<float>*, MatrixBase<float>*, MatrixIndexT) const;
template
void SpMatrix<double>::TopEigs(VectorBase<double>*, MatrixBase<double>*, MatrixIndexT) const;

// Someone had a problem with the Intel compiler with -O3, with Qr not being
// defined for some strange reason (should automatically happen when
// we instantiate Eig and TopEigs), so we explicitly instantiate it here.
template
void SpMatrix<float>::Qr(MatrixBase<float> *Q);
template
void SpMatrix<double>::Qr(MatrixBase<double> *Q);



}
// namespace kaldi
