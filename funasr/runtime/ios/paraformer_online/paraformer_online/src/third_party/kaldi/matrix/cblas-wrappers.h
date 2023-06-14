// matrix/cblas-wrappers.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey);
//                 Haihua Xu; Wei Shi

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
#ifndef KALDI_MATRIX_CBLAS_WRAPPERS_H_
#define KALDI_MATRIX_CBLAS_WRAPPERS_H_ 1


#include <limits>
#include "matrix/sp-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-functions.h"
#include "matrix/kaldi-blas.h"

// Do not include this file directly.  It is to be included
// by .cc files in this directory.

namespace kaldi {


inline void cblas_Xcopy(const int N, const float *X, const int incX, float *Y,
                        const int incY) {
  cblas_scopy(N, X, incX, Y, incY);
}

inline void cblas_Xcopy(const int N, const double *X, const int incX, double *Y,
                        const int incY) {
  cblas_dcopy(N, X, incX, Y, incY);
}


inline float cblas_Xasum(const int N, const float *X, const int incX) {
  return cblas_sasum(N, X, incX);
}

inline double cblas_Xasum(const int N, const double *X, const int incX) {
  return cblas_dasum(N, X, incX);
}

inline void cblas_Xrot(const int N, float *X, const int incX, float *Y,
                       const int incY, const float c, const float s) {
  cblas_srot(N, X, incX, Y, incY, c, s);
}
inline void cblas_Xrot(const int N, double *X, const int incX, double *Y,
                       const int incY, const double c, const double s) {
  cblas_drot(N, X, incX, Y, incY, c, s);
}
inline float cblas_Xdot(const int N, const float *const X,
                        const int incX, const float *const Y,
                        const int incY) {
  return cblas_sdot(N, X, incX, Y, incY);
}
inline double cblas_Xdot(const int N, const double *const X,
                        const int incX, const double *const Y,
                        const int incY) {
  return cblas_ddot(N, X, incX, Y, incY);
}
inline void cblas_Xaxpy(const int N, const float alpha, const float *X,
                        const int incX, float *Y, const int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_Xaxpy(const int N, const double alpha, const double *X,
                        const int incX, double *Y, const int incY) {
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_Xscal(const int N, const float alpha, float *data,
                        const int inc) {
  cblas_sscal(N, alpha, data, inc);
}
inline void cblas_Xscal(const int N, const double alpha, double *data, 
                        const int inc) {
  cblas_dscal(N, alpha, data, inc);
}
inline void cblas_Xspmv(const float alpha, const int num_rows, const float *Mdata,
                        const float *v, const int v_inc,
                        const float beta, float *y, const int y_inc) {
  cblas_sspmv(CblasRowMajor, CblasLower, num_rows, alpha, Mdata, v, v_inc, beta, y, y_inc);
}
inline void cblas_Xspmv(const double alpha, const int num_rows, const double *Mdata,
                        const double *v, const int v_inc,
                        const double beta, double *y, const int y_inc) {
  cblas_dspmv(CblasRowMajor, CblasLower, num_rows, alpha, Mdata, v, v_inc, beta, y, y_inc);
}
inline void cblas_Xtpmv(MatrixTransposeType trans, const float *Mdata,
                        const int num_rows, float *y, const int y_inc) {
  cblas_stpmv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}
inline void cblas_Xtpmv(MatrixTransposeType trans, const double *Mdata,
                        const int num_rows, double *y, const int y_inc) {
  cblas_dtpmv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}


inline void cblas_Xtpsv(MatrixTransposeType trans, const float *Mdata,
                        const int num_rows, float *y, const int y_inc) {
  cblas_stpsv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}
inline void cblas_Xtpsv(MatrixTransposeType trans, const double *Mdata,
                        const int num_rows, double *y, const int y_inc) {
  cblas_dtpsv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}

// x = alpha * M * y + beta * x
inline void cblas_Xspmv(MatrixIndexT dim, float alpha, const float *Mdata,
                        const float *ydata, MatrixIndexT ystride,
                        float beta, float *xdata, MatrixIndexT xstride) {
  cblas_sspmv(CblasRowMajor, CblasLower, dim, alpha, Mdata,
              ydata, ystride, beta, xdata, xstride);
}
inline void cblas_Xspmv(MatrixIndexT dim, double alpha, const double *Mdata,
                        const double *ydata, MatrixIndexT ystride,
                        double beta, double *xdata, MatrixIndexT xstride) {
  cblas_dspmv(CblasRowMajor, CblasLower, dim, alpha, Mdata,
              ydata, ystride, beta, xdata, xstride);
}

// Implements  A += alpha * (x y'  + y x'); A is symmetric matrix.
inline void cblas_Xspr2(MatrixIndexT dim, float alpha, const float *Xdata,
                        MatrixIndexT incX, const float *Ydata, MatrixIndexT incY,
                          float *Adata) {
  cblas_sspr2(CblasRowMajor, CblasLower, dim, alpha, Xdata,
              incX, Ydata, incY, Adata);
}
inline void cblas_Xspr2(MatrixIndexT dim, double alpha, const double *Xdata,
                        MatrixIndexT incX, const double *Ydata, MatrixIndexT incY,
                        double *Adata) {
  cblas_dspr2(CblasRowMajor, CblasLower, dim, alpha, Xdata,
              incX, Ydata, incY, Adata);
}

// Implements  A += alpha * (x x'); A is symmetric matrix.
inline void cblas_Xspr(MatrixIndexT dim, float alpha, const float *Xdata,
                       MatrixIndexT incX, float *Adata) {
  cblas_sspr(CblasRowMajor, CblasLower, dim, alpha, Xdata, incX, Adata);
}
inline void cblas_Xspr(MatrixIndexT dim, double alpha, const double *Xdata,
                       MatrixIndexT incX, double *Adata) {
  cblas_dspr(CblasRowMajor, CblasLower, dim, alpha, Xdata, incX, Adata);
}

// sgemv,dgemv: y = alpha M x + beta y.
inline void cblas_Xgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, float alpha, const float *Mdata,
                        MatrixIndexT stride, const float *xdata,
                        MatrixIndexT incX, float beta, float *ydata, MatrixIndexT incY) {
  cblas_sgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride, xdata, incX, beta, ydata, incY);
}
inline void cblas_Xgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, double alpha, const double *Mdata,
                        MatrixIndexT stride, const double *xdata,
                        MatrixIndexT incX, double beta, double *ydata, MatrixIndexT incY) {
  cblas_dgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride, xdata, incX, beta, ydata, incY);
}

// sgbmv, dgmmv: y = alpha M x +  + beta * y.
inline void cblas_Xgbmv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, MatrixIndexT num_below,
                        MatrixIndexT num_above, float alpha, const float *Mdata,
                        MatrixIndexT stride, const float *xdata,
                        MatrixIndexT incX, float beta, float *ydata, MatrixIndexT incY) {
  cblas_sgbmv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, num_below, num_above, alpha, Mdata, stride, xdata,
              incX, beta, ydata, incY);
}
inline void cblas_Xgbmv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, MatrixIndexT num_below,
                        MatrixIndexT num_above, double alpha, const double *Mdata,
                        MatrixIndexT stride, const double *xdata,
                        MatrixIndexT incX, double beta, double *ydata, MatrixIndexT incY) {
  cblas_dgbmv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, num_below, num_above, alpha, Mdata, stride, xdata,
              incX, beta, ydata, incY);
}


template<typename Real>
inline void Xgemv_sparsevec(MatrixTransposeType trans, MatrixIndexT num_rows,
                            MatrixIndexT num_cols, Real alpha, const Real *Mdata,
                            MatrixIndexT stride, const Real *xdata,
                            MatrixIndexT incX, Real beta, Real *ydata,
                            MatrixIndexT incY) {
  if (trans == kNoTrans) {
    if (beta != 1.0) cblas_Xscal(num_rows, beta, ydata, incY);
    for (MatrixIndexT i = 0; i < num_cols; i++) {
      Real x_i = xdata[i * incX];
      if (x_i == 0.0) continue;
      // Add to ydata, the i'th column of M, times alpha * x_i
      cblas_Xaxpy(num_rows, x_i * alpha, Mdata + i, stride, ydata, incY);
    }    
  } else {
    if (beta != 1.0) cblas_Xscal(num_cols, beta, ydata, incY);
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      Real x_i = xdata[i * incX];
      if (x_i == 0.0) continue;
      // Add to ydata, the i'th row of M, times alpha * x_i
      cblas_Xaxpy(num_cols, x_i * alpha,
                  Mdata + (i * stride), 1, ydata, incY);
    }
  }
}

inline void cblas_Xgemm(const float alpha,
                        MatrixTransposeType transA,
                        const float *Adata,
                        MatrixIndexT a_num_rows, MatrixIndexT a_num_cols, MatrixIndexT a_stride,
                        MatrixTransposeType transB, 
                        const float *Bdata, MatrixIndexT b_stride,
                        const float beta,
                        float *Mdata, 
                        MatrixIndexT num_rows, MatrixIndexT num_cols,MatrixIndexT stride) {
  cblas_sgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA), 
              static_cast<CBLAS_TRANSPOSE>(transB),
              num_rows, num_cols, transA == kNoTrans ? a_num_cols : a_num_rows,
              alpha, Adata, a_stride, Bdata, b_stride,
              beta, Mdata, stride); 
}
inline void cblas_Xgemm(const double alpha,
                        MatrixTransposeType transA,
                        const double *Adata,
                        MatrixIndexT a_num_rows, MatrixIndexT a_num_cols, MatrixIndexT a_stride,
                        MatrixTransposeType transB, 
                        const double *Bdata, MatrixIndexT b_stride,
                        const double beta,
                        double *Mdata, 
                        MatrixIndexT num_rows, MatrixIndexT num_cols,MatrixIndexT stride) {
  cblas_dgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA), 
              static_cast<CBLAS_TRANSPOSE>(transB),
              num_rows, num_cols, transA == kNoTrans ? a_num_cols : a_num_rows,
              alpha, Adata, a_stride, Bdata, b_stride,
              beta, Mdata, stride); 
}


inline void cblas_Xsymm(const float alpha,
                        MatrixIndexT sz,
                        const float *Adata,MatrixIndexT a_stride,
                        const float *Bdata,MatrixIndexT b_stride,
                        const float beta,
                        float *Mdata, MatrixIndexT stride) {
  cblas_ssymm(CblasRowMajor, CblasLeft, CblasLower, sz, sz, alpha, Adata,
              a_stride, Bdata, b_stride, beta, Mdata, stride);
}
inline void cblas_Xsymm(const double alpha,
                        MatrixIndexT sz,
                        const double *Adata,MatrixIndexT a_stride,
                        const double *Bdata,MatrixIndexT b_stride,
                        const double beta,
                        double *Mdata, MatrixIndexT stride) {
  cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, sz, sz, alpha, Adata,
              a_stride, Bdata, b_stride, beta, Mdata, stride);
}
// ger: M += alpha x y^T.
inline void cblas_Xger(MatrixIndexT num_rows, MatrixIndexT num_cols, float alpha,
                       const float *xdata, MatrixIndexT incX, const float *ydata,
                       MatrixIndexT incY, float *Mdata, MatrixIndexT stride) {
  cblas_sger(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
             Mdata, stride);
}
inline void cblas_Xger(MatrixIndexT num_rows, MatrixIndexT num_cols, double alpha,
                       const double *xdata, MatrixIndexT incX, const double *ydata,
                       MatrixIndexT incY, double *Mdata, MatrixIndexT stride) {
  cblas_dger(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
             Mdata, stride);
}

// syrk: symmetric rank-k update.
// if trans==kNoTrans, then C = alpha A A^T + beta C
// else C = alpha A^T A + beta C.
// note: dim_c is dim(C), other_dim_a is the "other" dimension of A, i.e.
// num-cols(A) if kNoTrans, or num-rows(A) if kTrans.
// We only need the row-major and lower-triangular option of this, and this
// is hard-coded.
inline void cblas_Xsyrk (
    const MatrixTransposeType trans, const MatrixIndexT dim_c,
    const MatrixIndexT other_dim_a, const float alpha, const float *A,
    const MatrixIndexT a_stride, const float beta, float *C,
    const MatrixIndexT c_stride) {
  cblas_ssyrk(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              dim_c, other_dim_a, alpha, A, a_stride, beta, C, c_stride);
}

inline void cblas_Xsyrk(
    const MatrixTransposeType trans, const MatrixIndexT dim_c,
    const MatrixIndexT other_dim_a, const double alpha, const double *A,
    const MatrixIndexT a_stride, const double beta, double *C,
    const MatrixIndexT c_stride) {
  cblas_dsyrk(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              dim_c, other_dim_a, alpha, A, a_stride, beta, C, c_stride);
}

/// matrix-vector multiply using a banded matrix; we always call this
/// with b = 1 meaning we're multiplying by a diagonal matrix.  This is used for
/// elementwise multiplication.  We miss some of the arguments out of this
/// wrapper.
inline void cblas_Xsbmv1(
    const MatrixIndexT dim,
    const double *A,
    const double alpha,
    const double *x,
    const double beta,
    double *y) {
  cblas_dsbmv(CblasRowMajor, CblasLower, dim, 0, alpha, A,
              1, x, 1, beta, y, 1);
}

inline void cblas_Xsbmv1(
    const MatrixIndexT dim,
    const float *A,
    const float alpha,
    const float *x,
    const float beta,
    float *y) {
  cblas_ssbmv(CblasRowMajor, CblasLower, dim, 0, alpha, A,
              1, x, 1, beta, y, 1);
}

/// This is not really a wrapper for CBLAS as CBLAS does not have this; in future we could
/// extend this somehow.
inline void mul_elements(
    const MatrixIndexT dim,
    const double *a,
    double *b) { // does b *= a, elementwise.
  double c1, c2, c3, c4;
  MatrixIndexT i;
  for (i = 0; i + 4 <= dim; i += 4) {
    c1 = a[i] * b[i];
    c2 = a[i+1] * b[i+1];
    c3 = a[i+2] * b[i+2];
    c4 = a[i+3] * b[i+3];
    b[i] = c1;
    b[i+1] = c2;
    b[i+2] = c3;
    b[i+3] = c4;
  }
  for (; i < dim; i++)
    b[i] *= a[i];
}

inline void mul_elements(
    const MatrixIndexT dim,
    const float *a,
    float *b) { // does b *= a, elementwise.
  float c1, c2, c3, c4;
  MatrixIndexT i;
  for (i = 0; i + 4 <= dim; i += 4) {
    c1 = a[i] * b[i];
    c2 = a[i+1] * b[i+1];
    c3 = a[i+2] * b[i+2];
    c4 = a[i+3] * b[i+3];
    b[i] = c1;
    b[i+1] = c2;
    b[i+2] = c3;
    b[i+3] = c4;
  }
  for (; i < dim; i++)
    b[i] *= a[i];
}



// add clapack here
#if !defined(HAVE_ATLAS)
inline void clapack_Xtptri(KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *result) {
  stptri_(const_cast<char *>("U"), const_cast<char *>("N"), num_rows, Mdata, result);
}
inline void clapack_Xtptri(KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *result) {
  dtptri_(const_cast<char *>("U"), const_cast<char *>("N"), num_rows, Mdata, result);
}
// 
inline void clapack_Xgetrf2(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols, 
                            float *Mdata, KaldiBlasInt *stride, KaldiBlasInt *pivot, 
                            KaldiBlasInt *result) {
  sgetrf_(num_rows, num_cols, Mdata, stride, pivot, result);
}
inline void clapack_Xgetrf2(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols, 
                            double *Mdata, KaldiBlasInt *stride, KaldiBlasInt *pivot, 
                            KaldiBlasInt *result) {
  dgetrf_(num_rows, num_cols, Mdata, stride, pivot, result);
}

// 
inline void clapack_Xgetri2(KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *stride,
                           KaldiBlasInt *pivot, float *p_work, 
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  sgetri_(num_rows, Mdata, stride, pivot, p_work, l_work, result);
}
inline void clapack_Xgetri2(KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *stride,
                           KaldiBlasInt *pivot, double *p_work, 
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  dgetri_(num_rows, Mdata, stride, pivot, p_work, l_work, result);
}
//
inline void clapack_Xgesvd(char *v, char *u, KaldiBlasInt *num_cols,
                           KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *stride,
                           float *sv, float *Vdata, KaldiBlasInt *vstride,
                           float *Udata, KaldiBlasInt *ustride, float *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  sgesvd_(v, u,
          num_cols, num_rows, Mdata, stride,
          sv, Vdata, vstride, Udata, ustride, 
          p_work, l_work, result); 
}
inline void clapack_Xgesvd(char *v, char *u, KaldiBlasInt *num_cols,
                           KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *stride,
                           double *sv, double *Vdata, KaldiBlasInt *vstride,
                           double *Udata, KaldiBlasInt *ustride, double *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  dgesvd_(v, u,
          num_cols, num_rows, Mdata, stride,
          sv, Vdata, vstride, Udata, ustride,
          p_work, l_work, result); 
}
//
void inline clapack_Xsptri(KaldiBlasInt *num_rows, float *Mdata, 
                           KaldiBlasInt *ipiv, float *work, KaldiBlasInt *result) {
  ssptri_(const_cast<char *>("U"), num_rows, Mdata, ipiv, work, result);
}
void inline clapack_Xsptri(KaldiBlasInt *num_rows, double *Mdata, 
                           KaldiBlasInt *ipiv, double *work, KaldiBlasInt *result) {
  dsptri_(const_cast<char *>("U"), num_rows, Mdata, ipiv, work, result);
}
//
void inline clapack_Xsptrf(KaldiBlasInt *num_rows, float *Mdata,
                           KaldiBlasInt *ipiv, KaldiBlasInt *result) {
  ssptrf_(const_cast<char *>("U"), num_rows, Mdata, ipiv, result);
}
void inline clapack_Xsptrf(KaldiBlasInt *num_rows, double *Mdata,
                           KaldiBlasInt *ipiv, KaldiBlasInt *result) {
  dsptrf_(const_cast<char *>("U"), num_rows, Mdata, ipiv, result);
}
#else
inline void clapack_Xgetrf(MatrixIndexT num_rows, MatrixIndexT num_cols,
                           float *Mdata, MatrixIndexT stride, 
                           int *pivot, int *result) {
  *result = clapack_sgetrf(CblasColMajor, num_rows, num_cols,
                              Mdata, stride, pivot);
}

inline void clapack_Xgetrf(MatrixIndexT num_rows, MatrixIndexT num_cols,
                           double *Mdata, MatrixIndexT stride, 
                           int *pivot, int *result) {
  *result = clapack_dgetrf(CblasColMajor, num_rows, num_cols,
                              Mdata, stride, pivot);
}
//
inline int clapack_Xtrtri(int num_rows, float *Mdata, MatrixIndexT stride) {
  return  clapack_strtri(CblasColMajor, CblasUpper, CblasNonUnit, num_rows,
                              Mdata, stride);
}

inline int clapack_Xtrtri(int num_rows, double *Mdata, MatrixIndexT stride) {
  return  clapack_dtrtri(CblasColMajor, CblasUpper, CblasNonUnit, num_rows,
                              Mdata, stride);
}
//
inline void clapack_Xgetri(MatrixIndexT num_rows, float *Mdata, MatrixIndexT stride,
                      int *pivot, int *result) {
  *result = clapack_sgetri(CblasColMajor, num_rows, Mdata, stride, pivot);
}
inline void clapack_Xgetri(MatrixIndexT num_rows, double *Mdata, MatrixIndexT stride,
                      int *pivot, int *result) {
  *result = clapack_dgetri(CblasColMajor, num_rows, Mdata, stride, pivot);
}
#endif

}
// namespace kaldi

#endif
