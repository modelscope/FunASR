// matrix/optimization.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
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



#ifndef KALDI_MATRIX_OPTIMIZATION_H_
#define KALDI_MATRIX_OPTIMIZATION_H_

#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {


/// @addtogroup matrix_optimization
/// @{

struct LinearCgdOptions {
  int32 max_iters;  //  Maximum number of iters (if >= 0).
  BaseFloat max_error;  // Maximum 2-norm of the residual A x - b (convergence
                        // test)
  // Every time the residual 2-norm decreases by this recompute_residual_factor
  // since the last time it was computed from scratch, recompute it from
  // scratch.  This helps to keep the computed residual accurate even in the
  // presence of roundoff.
  BaseFloat recompute_residual_factor;
  
  LinearCgdOptions(): max_iters(-1),
                      max_error(0.0),
                      recompute_residual_factor(0.01) { }
};
  
/*
  This function uses linear conjugate gradient descent to approximately solve
  the system A x = b.  The value of x at entry corresponds to the initial guess
  of x.  The algorithm continues until the number of iterations equals b.Dim(),
  or until the 2-norm of (A x - b) is <= max_error, or until the number of
  iterations equals max_iter, whichever happens sooner.  It is a requirement
  that A be positive definite.
  It returns the number of iterations that were actually executed (this is
  useful for testing purposes).
*/
template<typename Real>
int32 LinearCgd(const LinearCgdOptions &opts,
                const SpMatrix<Real> &A, const VectorBase<Real> &b,
                VectorBase<Real> *x);






/**
   This is an implementation of L-BFGS.  It pushes responsibility for
   determining when to stop, onto the user.  There is no call-back here:
   everything is done via calls to the class itself (see the example in
   matrix-lib-test.cc).  This does not implement constrained L-BFGS, but it will
   handle constrained problems correctly as long as the function approaches
   +infinity (or -infinity for maximization problems) when it gets close to the
   bound of the constraint.  In these types of problems, you just let the
   function value be +infinity for minimization problems, or -infinity for
   maximization problems, outside these bounds).
*/

struct LbfgsOptions {
  bool minimize; // if true, we're minimizing, else maximizing.
  int m; // m is the number of stored vectors L-BFGS keeps.
  float first_step_learning_rate; // The very first step of L-BFGS is
  // like gradient descent.  If you want to configure the size of that step,
  // you can do it using this variable.
  float first_step_length; // If this variable is >0.0, it overrides
  // first_step_learning_rate; on the first step we choose an approximate
  // Hessian that is the multiple of the identity that would generate this
  // step-length, or 1.0 if the gradient is zero.
  float first_step_impr; // If this variable is >0.0, it overrides
  // first_step_learning_rate; on the first step we choose an approximate
  // Hessian that is the multiple of the identity that would generate this
  // amount of objective function improvement (assuming the "real" objf
  // was linear).
  float c1; // A constant in Armijo rule = Wolfe condition i)
  float c2; // A constant in Wolfe condition ii)
  float d; // An amount > 1.0 (default 2.0) that we initially multiply or
  // divide the step length by, in the line search.
  int max_line_search_iters; // after this many iters we restart L-BFGS.
  int avg_step_length; // number of iters to avg step length over, in
  // RecentStepLength().
  
  LbfgsOptions (bool minimize = true):
      minimize(minimize),
      m(10),
      first_step_learning_rate(1.0),
      first_step_length(0.0),
      first_step_impr(0.0),
      c1(1.0e-04),
      c2(0.9),
      d(2.0),
      max_line_search_iters(50),
      avg_step_length(4) { }
};

template<typename Real>
class OptimizeLbfgs {
 public:
  /// Initializer takes the starting value of x.
  OptimizeLbfgs(const VectorBase<Real> &x,
                const LbfgsOptions &opts);
  
  /// This returns the value of the variable x that has the best objective
  /// function so far, and the corresponding objective function value if
  /// requested.  This would typically be called only at the end.
  const VectorBase<Real>& GetValue(Real *objf_value = NULL) const;
  
  /// This returns the value at which the function wants us
  /// to compute the objective function and gradient.
  const VectorBase<Real>& GetProposedValue() const { return new_x_; }
  
  /// Returns the average magnitude of the last n steps (but not
  /// more than the number we have stored).  Before we have taken
  /// any steps, returns +infinity.  Note: if the most recent
  /// step length was 0, it returns 0, regardless of the other
  /// step lengths.  This makes it suitable as a convergence test
  /// (else we'd generate NaN's).
  Real RecentStepLength() const;
  
  /// The user calls this function to provide the class with the
  /// function and gradient info at the point GetProposedValue().
  /// If this point is outside the constraints you can set function_value
  /// to {+infinity,-infinity} for {minimization,maximization} problems.
  /// In this case the gradient, and also the second derivative (if you call
  /// the second overloaded version of this function) will be ignored.
  void DoStep(Real function_value,
              const VectorBase<Real> &gradient);
  
  /// The user can call this version of DoStep() if it is desired to set some
  /// kind of approximate Hessian on this iteration.  Note: it is a prerequisite
  /// that diag_approx_2nd_deriv must be strictly positive (minimizing), or
  /// negative (maximizing).
  void DoStep(Real function_value,
              const VectorBase<Real> &gradient,
              const VectorBase<Real> &diag_approx_2nd_deriv);
  
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(OptimizeLbfgs);


  // The following variable says what stage of the computation we're at.
  // Refer to Algorithm 7.5 (L-BFGS) of Nodecdal & Wright, "Numerical
  // Optimization", 2nd edition.
  // kBeforeStep means we're about to do
  /// "compute p_k <-- - H_k \delta f_k" (i.e. Algorithm 7.4).
  // kWithinStep means we're at some point within line search; note
  // that line search is iterative so we can stay in this state more
  // than one time on each iteration.
  enum ComputationState {
    kBeforeStep,
    kWithinStep, // This means we're within the step-size computation, and
    // have not yet done the 1st function evaluation.
  };
  
  inline MatrixIndexT Dim() { return x_.Dim(); }
  inline MatrixIndexT M() { return opts_.m; }
  SubVector<Real> Y(MatrixIndexT i) {
    return SubVector<Real>(data_, (i % M()) * 2); // vector y_i
  }
  SubVector<Real> S(MatrixIndexT i) {
    return SubVector<Real>(data_, (i % M()) * 2 + 1); // vector s_i
  }
  // The following are subroutines within DoStep():
  bool AcceptStep(Real function_value,
                  const VectorBase<Real> &gradient);
  void Restart(const VectorBase<Real> &x,
               Real function_value,
               const VectorBase<Real> &gradient);
  void ComputeNewDirection(Real function_value,
                           const VectorBase<Real> &gradient);
  void ComputeHifNeeded(const VectorBase<Real> &gradient);
  void StepSizeIteration(Real function_value,
                         const VectorBase<Real> &gradient);
  void RecordStepLength(Real s);
  
  
  LbfgsOptions opts_;
  SignedMatrixIndexT k_; // Iteration number, starts from zero.  Gets set back to zero
  // when we restart.
  
  ComputationState computation_state_;
  bool H_was_set_; // True if the user specified H_; if false,
  // we'll use a heuristic to estimate it.


  Vector<Real> x_; // current x.
  Vector<Real> new_x_; // the x proposed in the line search.
  Vector<Real> best_x_; // the x with the best objective function so far
                        // (either the same as x_ or something in the current line search.)
  Vector<Real> deriv_; // The most recently evaluated derivative-- at x_k.
  Vector<Real> temp_;
  Real f_; // The function evaluated at x_k.
  Real best_f_; // the best objective function so far.
  Real d_; // a number d > 1.0, but during an iteration we may decrease this, when
  // we switch between armijo and wolfe failures.

  int num_wolfe_i_failures_; // the num times we decreased step size.
  int num_wolfe_ii_failures_; // the num times we increased step size.
  enum { kWolfeI, kWolfeII, kNone } last_failure_type_; // last type of step-search
  // failure on this iter.
  
  Vector<Real> H_; // Current inverse-Hessian estimate.  May be computed by this class itself,
  // or provided by user using 2nd form of SetGradientInfo().
  Matrix<Real> data_; // dimension (m*2) x dim.  Even rows store
  // gradients y_i, odd rows store steps s_i.
  Vector<Real> rho_; // dimension m; rho_(m) = 1/(y_m^T s_m), Eq. 7.17.

  std::vector<Real> step_lengths_; // The step sizes we took on the last
  // (up to m) iterations; these are not stored in a rotating buffer but
  // are shifted by one each time (this is more convenient when we
  // restart, as we keep this info past restarting).
  

};
  
/// @} 


} // end namespace kaldi



#endif

