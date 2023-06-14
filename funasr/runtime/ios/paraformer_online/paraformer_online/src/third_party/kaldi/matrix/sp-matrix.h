// matrix/sp-matrix.h

// Copyright 2009-2011   Ondrej Glembek;  Microsoft Corporation;  Lukas Burget;
//                       Saarland University;  Ariya Rastrow;  Yanmin Qian;
//                       Jan Silovsky

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
#ifndef KALDI_MATRIX_SP_MATRIX_H_
#define KALDI_MATRIX_SP_MATRIX_H_

#include <algorithm>
#include <vector>

#include "matrix/packed-matrix.h"

namespace kaldi {


/// \addtogroup matrix_group
/// @{
template<typename Real> class SpMatrix;


/**
 * @brief Packed symetric matrix class
*/
template<typename Real>
class SpMatrix : public PackedMatrix<Real> {
  friend class CuSpMatrix<Real>;
 public:
  // so it can use our assignment operator.
  friend class std::vector<Matrix<Real> >;

  SpMatrix(): PackedMatrix<Real>() {}

  /// Copy constructor from CUDA version of SpMatrix
  /// This is defined in ../cudamatrix/cu-sp-matrix.h

  explicit SpMatrix(const CuSpMatrix<Real> &cu);

  explicit SpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
      : PackedMatrix<Real>(r, resize_type) {}

  SpMatrix(const SpMatrix<Real> &orig)
      : PackedMatrix<Real>(orig) {}

  template<typename OtherReal>
  explicit SpMatrix(const SpMatrix<OtherReal> &orig)
      : PackedMatrix<Real>(orig) {}

#ifdef KALDI_PARANOID
  explicit SpMatrix(const MatrixBase<Real> & orig,
                    SpCopyType copy_type = kTakeMeanAndCheck)
      : PackedMatrix<Real>(orig.NumRows(), kUndefined) {
    CopyFromMat(orig, copy_type);
  }
#else
  explicit SpMatrix(const MatrixBase<Real> & orig,
                    SpCopyType copy_type = kTakeMean)
      : PackedMatrix<Real>(orig.NumRows(), kUndefined) {
    CopyFromMat(orig, copy_type);
  }
#endif

  /// Shallow swap.
  void Swap(SpMatrix *other);

  inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    PackedMatrix<Real>::Resize(nRows, resize_type);
  }

  void CopyFromSp(const SpMatrix<Real> &other) {
    PackedMatrix<Real>::CopyFromPacked(other);
  }

  template<typename OtherReal>
  void CopyFromSp(const SpMatrix<OtherReal> &other) {
    PackedMatrix<Real>::CopyFromPacked(other);
  }

#ifdef KALDI_PARANOID
  void CopyFromMat(const MatrixBase<Real> &orig,
                   SpCopyType copy_type = kTakeMeanAndCheck);
#else  // different default arg if non-paranoid mode.
  void CopyFromMat(const MatrixBase<Real> &orig,
                   SpCopyType copy_type = kTakeMean);
#endif

  inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    // if column is less than row, then swap these as matrix is stored
    // as upper-triangular...  only allowed for const matrix object.
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r))
      std::swap(c, r);
    // c<=r now so don't have to check c.
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));
    return *(this->data_ + (r*(r+1)) / 2 + c);
    // Duplicating code from PackedMatrix.h
  }

  inline Real &operator() (MatrixIndexT r, MatrixIndexT c) {
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r))
      std::swap(c, r);
    // c<=r now so don't have to check c.
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));
    return *(this->data_ + (r * (r + 1)) / 2 + c);
    // Duplicating code from PackedMatrix.h
  }

  SpMatrix<Real>& operator=(const SpMatrix<Real> &other) {
    PackedMatrix<Real>::operator=(other);
    return *this;
  }

  using PackedMatrix<Real>::Scale;

  /// matrix inverse.
  /// if inverse_needed = false, will fill matrix with garbage.
  /// (only useful if logdet wanted).
  void Invert(Real *logdet = NULL, Real *det_sign= NULL,
              bool inverse_needed = true);

  // Below routine does inversion in double precision,
  // even for single-precision object.
  void InvertDouble(Real *logdet = NULL, Real *det_sign = NULL,
                    bool inverse_needed = true);

  /// Returns maximum ratio of singular values.
  inline Real Cond() const {
    Matrix<Real> tmp(*this);
    return tmp.Cond();
  }

  /// Takes matrix to a fraction power via Svd.
  /// Will throw exception if matrix is not positive semidefinite
  /// (to within a tolerance)
  void ApplyPow(Real exponent);

  /// This is the version of SVD that we implement for symmetric positive
  /// definite matrices.  This exists for historical reasons; right now its
  /// internal implementation is the same as Eig().  It computes the eigenvalue
  /// decomposition (*this) = P * diag(s) * P^T with P orthogonal.  Will throw
  /// exception if input is not positive semidefinite to within a tolerance.
  void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
                        Real tolerance = 0.001) const;

  /// Solves the symmetric eigenvalue problem: at end we should have (*this) = P
  /// * diag(s) * P^T.  We solve the problem using the symmetric QR method.
  /// P may be NULL.
  /// Implemented in qr.cc.
  /// If you need the eigenvalues sorted, the function SortSvd declared in
  /// kaldi-matrix is suitable.
  void Eig(VectorBase<Real> *s, MatrixBase<Real> *P = NULL) const;

  /// This function gives you, approximately, the largest eigenvalues of the
  /// symmetric matrix and the corresponding eigenvectors.  (largest meaning,
  /// further from zero).  It does this by doing a SVD within the Krylov
  /// subspace generated by this matrix and a random vector.  This is
  /// a form of the Lanczos method with complete reorthogonalization, followed
  /// by SVD within a smaller dimension ("lanczos_dim").
  ///
  /// If *this is m by m, s should be of dimension n and P should be of
  /// dimension m by n, with n <= m.  The *columns* of P are the approximate
  /// eigenvectors; P * diag(s) * P^T would be a low-rank reconstruction of
  /// *this.  The columns of P will be orthogonal, and the elements of s will be
  /// the eigenvalues of *this projected into that subspace, but beyond that
  /// there are no exact guarantees.  (This is because the convergence of this
  /// method is statistical).  Note: it only makes sense to use this
  /// method if you are in very high dimension and n is substantially smaller
  /// than m: for example, if you want the 100 top eigenvalues of a 10k by 10k
  /// matrix.  This function calls Rand() to initialize the lanczos
  /// iterations and also for restarting.
  /// If lanczos_dim is zero, it will default to the greater of:
  /// s->Dim() + 50 or s->Dim() + s->Dim()/2, but not more than this->Dim().
  /// If lanczos_dim == this->Dim(), you might as well just call the function
  /// Eig() since the result will be the same, and Eig() would be faster; the
  /// whole point of this function is to reduce the dimension of the SVD
  /// computation.
  void TopEigs(VectorBase<Real> *s, MatrixBase<Real> *P,
               MatrixIndexT lanczos_dim = 0) const;


  /// Returns the maximum of the absolute values of any of the
  /// eigenvalues.
  Real MaxAbsEig() const;

  void PrintEigs(const char *name) {
    Vector<Real> s((*this).NumRows());
    Matrix<Real> P((*this).NumRows(), (*this).NumCols());
    SymPosSemiDefEig(&s, &P);
    KALDI_LOG << "PrintEigs: " << name << ": " << s;
  }

  bool IsPosDef() const;  // returns true if Cholesky succeeds.
  void AddSp(const Real alpha, const SpMatrix<Real> &Ma) {
    this->AddPacked(alpha, Ma);
  }

  /// Computes log determinant but only for +ve-def matrices
  /// (it uses Cholesky).
  /// If matrix is not +ve-def, it will throw an exception
  /// was LogPDDeterminant()
  Real LogPosDefDet() const;

  Real LogDet(Real *det_sign = NULL) const;

  /// rank-one update, this <-- this + alpha v v'
  template<typename OtherReal>
  void AddVec2(const Real alpha, const VectorBase<OtherReal> &v);

  /// rank-two update, this <-- this + alpha (v w' + w v').
  void AddVecVec(const Real alpha, const VectorBase<Real> &v,
                 const VectorBase<Real> &w);

  /// Does *this = beta * *thi + alpha * diag(v) * S * diag(v)
  void AddVec2Sp(const Real alpha, const VectorBase<Real> &v,
                 const SpMatrix<Real> &S, const Real beta);

  /// diagonal update, this <-- this + diag(v)
  template<typename OtherReal>
  void AddDiagVec(const Real alpha, const VectorBase<OtherReal> &v);

  /// rank-N update:
  /// if (transM == kNoTrans)
  /// (*this) = beta*(*this) + alpha * M * M^T,
  /// or  (if transM == kTrans)
  ///  (*this) = beta*(*this) + alpha * M^T * M
  /// Note: beta used to default to 0.0.
  void AddMat2(const Real alpha, const MatrixBase<Real> &M,
               MatrixTransposeType transM, const Real beta);

  /// Extension of rank-N update:
  /// this <-- beta*this  +  alpha * M * A * M^T.
  /// (*this) and A are allowed to be the same.
  /// If transM == kTrans, then we do it as M^T * A * M.
  void AddMat2Sp(const Real alpha, const MatrixBase<Real> &M,
                 MatrixTransposeType transM, const SpMatrix<Real> &A,
                 const Real beta = 0.0);

  /// This is a version of AddMat2Sp specialized for when M is fairly sparse.
  /// This was required for making the raw-fMLLR code efficient.
  void AddSmat2Sp(const Real alpha, const MatrixBase<Real> &M,
                  MatrixTransposeType transM, const SpMatrix<Real> &A,
                  const Real beta = 0.0);

  /// The following function does:
  /// this <-- beta*this  +  alpha * T * A * T^T.
  /// (*this) and A are allowed to be the same.
  /// If transM == kTrans, then we do it as alpha * T^T * A * T.
  /// Currently it just calls AddMat2Sp, but if needed we
  /// can implement it more efficiently.
  void AddTp2Sp(const Real alpha, const TpMatrix<Real> &T,
                MatrixTransposeType transM, const SpMatrix<Real> &A,
                const Real beta = 0.0);

  /// The following function does:
  /// this <-- beta*this  +  alpha * T * T^T.
  /// (*this) and A are allowed to be the same.
  /// If transM == kTrans, then we do it as alpha * T^T *  T
  /// Currently it just calls AddMat2, but if needed we
  /// can implement it more efficiently.
  void AddTp2(const Real alpha, const TpMatrix<Real> &T,
              MatrixTransposeType transM, const Real beta = 0.0);

  /// Extension of rank-N update:
  /// this <-- beta*this + alpha * M * diag(v) * M^T.
  /// if transM == kTrans, then
  /// this <-- beta*this + alpha * M^T * diag(v) * M.
  void AddMat2Vec(const Real alpha, const MatrixBase<Real> &M,
                  MatrixTransposeType transM, const VectorBase<Real> &v,
                  const Real beta = 0.0);


  ///  Floors this symmetric matrix to the matrix
  /// alpha * Floor, where the matrix Floor is positive
  /// definite.
  /// It is floored in the sense that after flooring,
  ///  x^T (*this) x  >= x^T (alpha*Floor) x.
  /// This is accomplished using an Svd.  It will crash
  /// if Floor is not positive definite. Returns the number of
  /// elements that were floored.
  int ApplyFloor(const SpMatrix<Real> &Floor, Real alpha = 1.0,
                 bool verbose = false);

  /// Floor: Given a positive semidefinite matrix, floors the eigenvalues
  /// to the specified quantity.  A previous version of this function had
  /// a tolerance which is now no longer needed since we have code to
  /// do the symmetric eigenvalue decomposition and no longer use the SVD
  /// code for that purose.
  int ApplyFloor(Real floor);

  bool IsDiagonal(Real cutoff = 1.0e-05) const;
  bool IsUnit(Real cutoff = 1.0e-05) const;
  bool IsZero(Real cutoff = 1.0e-05) const;
  bool IsTridiagonal(Real cutoff = 1.0e-05) const;

  /// sqrt of sum of square elements.
  Real FrobeniusNorm() const;

  /// Returns true if ((*this)-other).FrobeniusNorm() <=
  ///   tol*(*this).FrobeniusNorma()
  bool ApproxEqual(const SpMatrix<Real> &other, float tol = 0.01) const;

  // LimitCond:
  // Limits the condition of symmetric positive semidefinite matrix to
  // a specified value
  // by flooring all eigenvalues to a positive number which is some multiple
  // of the largest one (or zero if there are no positive eigenvalues).
  // Takes the condition number we are willing to accept, and floors
  // eigenvalues to the largest eigenvalue divided by this.
  //  Returns #eigs floored or already equal to the floor.
  // Throws exception if input is not positive definite.
  // returns #floored.
  MatrixIndexT LimitCond(Real maxCond = 1.0e+5, bool invert = false);

  // as LimitCond but all done in double precision. // returns #floored.
  MatrixIndexT LimitCondDouble(Real maxCond = 1.0e+5, bool invert = false) {
    SpMatrix<double> dmat(*this);
    MatrixIndexT ans = dmat.LimitCond(maxCond, invert);
    (*this).CopyFromSp(dmat);
    return ans;
  }
  Real Trace() const;

  /// Tridiagonalize the matrix with an orthogonal transformation.  If
  /// *this starts as S, produce T (and Q, if non-NULL) such that
  /// T = Q A Q^T, i.e. S = Q^T T Q.  Caution: this is the other way
  /// round from most authors (it's more efficient in row-major indexing).
  void Tridiagonalize(MatrixBase<Real> *Q);

  /// The symmetric QR algorithm.  This will mostly be useful in internal code.
  /// Typically, you will call this after Tridiagonalize(), on the same object.
  /// When called, *this (call it A at this point) must be tridiagonal; at exit,
  /// *this will be a diagonal matrix D that is similar to A via orthogonal
  /// transformations.  This algorithm right-multiplies Q by orthogonal
  /// transformations.  It turns *this from a tridiagonal into a diagonal matrix
  /// while maintaining that (Q *this Q^T) has the same value at entry and exit.
  /// At entry Q should probably be either NULL or orthogonal, but we don't check
  /// this.
  void Qr(MatrixBase<Real> *Q);

 private:
 void EigInternal(VectorBase<Real> *s, MatrixBase<Real> *P,
                   Real tolerance, int recurse) const;
};

/// @} end of "addtogroup matrix_group"

/// \addtogroup matrix_funcs_scalar
/// @{


/// Returns tr(A B).
float TraceSpSp(const SpMatrix<float> &A, const SpMatrix<float> &B);
double TraceSpSp(const SpMatrix<double> &A, const SpMatrix<double> &B);


template<typename Real>
inline bool ApproxEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B, Real tol = 0.01) {
  return  A.ApproxEqual(B, tol);
}

template<typename Real>
inline void AssertEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B, Real tol = 0.01) {
  KALDI_ASSERT(ApproxEqual(A, B, tol));
}



/// Returns tr(A B).
template<typename Real, typename OtherReal>
Real TraceSpSp(const SpMatrix<Real> &A, const SpMatrix<OtherReal> &B);



// TraceSpSpLower is the same as Trace(A B) except the lower-diagonal elements
// are counted only once not twice as they should be.  It is useful in certain
// optimizations.
template<typename Real>
Real TraceSpSpLower(const SpMatrix<Real> &A, const SpMatrix<Real> &B);


/// Returns tr(A B).
/// No option to transpose B because would make no difference.
template<typename Real>
Real TraceSpMat(const SpMatrix<Real> &A, const MatrixBase<Real> &B);

/// Returns tr(A B C)
/// (A and C may be transposed as specified by transA and transC).
template<typename Real>
Real TraceMatSpMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                   const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                   MatrixTransposeType transC);

/// Returns tr (A B C D)
/// (A and C may be transposed as specified by transA and transB).
template<typename Real>
Real TraceMatSpMatSp(const MatrixBase<Real> &A, MatrixTransposeType transA,
                     const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                     MatrixTransposeType transC, const SpMatrix<Real> &D);

/** Computes v1^T * M * v2.  Not as efficient as it could be where v1 == v2
 * (but no suitable blas routines available).
 */

/// Returns \f$ v_1^T M v_2 \f$
/// Not as efficient as it could be where v1 == v2.
template<typename Real>
Real VecSpVec(const VectorBase<Real> &v1, const SpMatrix<Real> &M,
               const VectorBase<Real> &v2);


/// @} \addtogroup matrix_funcs_scalar

/// \addtogroup matrix_funcs_misc
/// @{


/// This class describes the options for maximizing various quadratic objective
/// functions.  It's mostly as described in the SGMM paper "the subspace
/// Gaussian mixture model -- a structured model for speech recognition", but
/// the diagonal_precondition option is newly added, to handle problems where
/// different dimensions have very different scaling (we recommend to use the
/// option but it's set false for back compatibility).
struct SolverOptions {
  BaseFloat K; // maximum condition number
  BaseFloat eps;
  std::string name;
  bool optimize_delta;
  bool diagonal_precondition;
  bool print_debug_output;
  explicit SolverOptions(const std::string &name):
      K(1.0e+4), eps(1.0e-40), name(name),
      optimize_delta(true), diagonal_precondition(false),
      print_debug_output(true) { }
  SolverOptions(): K(1.0e+4), eps(1.0e-40), name("[unknown]"),
                   optimize_delta(true), diagonal_precondition(false),
                   print_debug_output(true) { }
  void Check() const;
};


/// Maximizes the auxiliary function
/// \f[    Q(x) = x.g - 0.5 x^T H x     \f]
/// using a numerically stable method. Like a numerically stable version of
/// \f$  x := Q^{-1} g.    \f$
/// Assumes H positive semidefinite.
/// Returns the objective-function change.

template<typename Real>
Real SolveQuadraticProblem(const SpMatrix<Real> &H,
                           const VectorBase<Real> &g,
                           const SolverOptions &opts,
                           VectorBase<Real> *x);



/// Maximizes the auxiliary function :
/// \f[   Q(x) = tr(M^T P Y) - 0.5 tr(P M Q M^T)        \f]
/// Like a numerically stable version of  \f$  M := Y Q^{-1}   \f$.
/// Assumes Q and P positive semidefinite, and matrix dimensions match
/// enough to make expressions meaningful.
/// This is mostly as described in the SGMM paper "the subspace Gaussian mixture
/// model -- a structured model for speech recognition", but the
/// diagonal_precondition option is newly added, to handle problems
/// where different dimensions have very different scaling (we recommend to use
/// the option but it's set false for back compatibility).
template<typename Real>
Real SolveQuadraticMatrixProblem(const SpMatrix<Real> &Q,
                                 const MatrixBase<Real> &Y,
                                 const SpMatrix<Real> &P,
                                 const SolverOptions &opts,
                                 MatrixBase<Real> *M);

/// Maximizes the auxiliary function :
/// \f[   Q(M) =  tr(M^T G) -0.5 tr(P_1 M Q_1 M^T) -0.5 tr(P_2 M Q_2 M^T).   \f]
/// Encountered in matrix update with a prior. We also apply a limit on the
/// condition but it should be less frequently necessary, and can be set larger.
template<typename Real>
Real SolveDoubleQuadraticMatrixProblem(const MatrixBase<Real> &G,
                                       const SpMatrix<Real> &P1,
                                       const SpMatrix<Real> &P2,
                                       const SpMatrix<Real> &Q1,
                                       const SpMatrix<Real> &Q2,
                                       const SolverOptions &opts,
                                       MatrixBase<Real> *M);


/// @} End of "addtogroup matrix_funcs_misc"

}  // namespace kaldi


// Including the implementation (now actually just includes some
// template specializations).
#include "matrix/sp-matrix-inl.h"


#endif  // KALDI_MATRIX_SP_MATRIX_H_
