// matrix/optimization.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)


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

#include <algorithm>

#include "matrix/optimization.h"
#include "matrix/sp-matrix.h"

namespace kaldi {


// Below, N&W refers to Nocedal and Wright, "Numerical Optimization", 2nd Ed.

template<typename Real>
OptimizeLbfgs<Real>::OptimizeLbfgs(const VectorBase<Real> &x,
                                   const LbfgsOptions &opts):
    opts_(opts), k_(0), computation_state_(kBeforeStep), H_was_set_(false) {
  KALDI_ASSERT(opts.m > 0); // dimension.
  MatrixIndexT dim = x.Dim();
  KALDI_ASSERT(dim > 0);
  x_ = x; // this is the value of x_k
  new_x_ = x;  // this is where we'll evaluate the function next.
  deriv_.Resize(dim);
  temp_.Resize(dim);
  data_.Resize(2 * opts.m, dim);
  rho_.Resize(opts.m);
  // Just set f_ to some invalid value, as we haven't yet set it.
  f_ = (opts.minimize ? 1 : -1 ) * std::numeric_limits<Real>::infinity();
  best_f_ = f_;
  best_x_ = x_;
}


template<typename Real>
Real OptimizeLbfgs<Real>::RecentStepLength() const {
  size_t n = step_lengths_.size();
  if (n == 0) return std::numeric_limits<Real>::infinity();
  else {
    if (n >= 2 && step_lengths_[n-1] == 0.0 && step_lengths_[n-2] == 0.0)
      return 0.0; // two zeros in a row means repeated restarts, which is
    // a loop.  Short-circuit this by returning zero.
    Real avg = 0.0;
    for (size_t i = 0; i < n; i++)
      avg += step_lengths_[i] / n;
    return avg;
  }
}

template<typename Real>
void OptimizeLbfgs<Real>::ComputeHifNeeded(const VectorBase<Real> &gradient) {
  if (k_ == 0) {
    if (H_.Dim() == 0) {
      // H was never set up.  Set it up for the first time.
      Real learning_rate;
      if (opts_.first_step_length > 0.0) { // this takes
        // precedence over first_step_learning_rate, if set.
        // We are setting up H for the first time.
        Real gradient_length = gradient.Norm(2.0);
        learning_rate = (gradient_length > 0.0 ?
                         opts_.first_step_length / gradient_length :
                         1.0);
      } else if (opts_.first_step_impr > 0.0) {
        Real gradient_length = gradient.Norm(2.0);
        learning_rate = (gradient_length > 0.0 ?
                  opts_.first_step_impr / (gradient_length * gradient_length) :
                  1.0);
      } else {
        learning_rate = opts_.first_step_learning_rate;
      }
      H_.Resize(x_.Dim());
      KALDI_ASSERT(learning_rate > 0.0);
      H_.Set(opts_.minimize ? learning_rate : -learning_rate);
    }
  } else { // k_ > 0
    if (!H_was_set_) { // The user never specified an approximate
      // diagonal inverse Hessian.
      // Set it using formula 7.20: H_k^{(0)} = \gamma_k I, where
      // \gamma_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
      SubVector<Real> y_km1 = Y(k_-1);
      double gamma_k = VecVec(S(k_-1), y_km1) / VecVec(y_km1, y_km1);
      if (KALDI_ISNAN(gamma_k) || KALDI_ISINF(gamma_k)) {
        KALDI_WARN << "NaN encountered in L-BFGS (already converged?)";
        gamma_k = (opts_.minimize ? 1.0 : -1.0);
      }
      H_.Set(gamma_k);
    }
  }
}  

// This represents the first 2 lines of Algorithm 7.5 (N&W), which
// in fact is mostly a call to Algorithm 7.4.
// Note: this is valid whether we are minimizing or maximizing.
template<typename Real>
void OptimizeLbfgs<Real>::ComputeNewDirection(Real function_value,
                                              const VectorBase<Real> &gradient) {
  KALDI_ASSERT(computation_state_ == kBeforeStep);
  SignedMatrixIndexT m = M(), k = k_;
  ComputeHifNeeded(gradient);
  // The rest of this is computing p_k <-- - H_k \nabla f_k using Algorithm
  // 7.4 of N&W.
  Vector<Real> &q(deriv_), &r(new_x_); // Use deriv_ as a temporary place to put
  // q, and new_x_ as a temporay place to put r.
  // The if-statement below is just to get rid of spurious warnings from
  // valgrind about memcpy source and destination overlap, since sometimes q and
  // gradient are the same variable.
  if (&q != &gradient)
    q.CopyFromVec(gradient); // q <-- \nabla f_k.
  Vector<Real> alpha(m);
  // for i = k - 1, k - 2, ... k - m
  for (SignedMatrixIndexT i = k - 1;
       i >= std::max(k - m, static_cast<SignedMatrixIndexT>(0));
       i--) { 
    alpha(i % m) = rho_(i % m) * VecVec(S(i), q); // \alpha_i <-- \rho_i s_i^T q.
    q.AddVec(-alpha(i % m), Y(i)); // q <-- q - \alpha_i y_i
  }
  r.SetZero();
  r.AddVecVec(1.0, H_, q, 0.0); // r <-- H_k^{(0)} q.
  // for k = k - m, k - m + 1, ... , k - 1
  for (SignedMatrixIndexT i = std::max(k - m, static_cast<SignedMatrixIndexT>(0));
       i < k;
       i++) {
    Real beta = rho_(i % m) * VecVec(Y(i), r); // \beta <-- \rho_i y_i^T r
    r.AddVec(alpha(i % m) - beta, S(i)); // r <-- r + s_i (\alpha_i - \beta)
  }

  { // TEST.  Note, -r will be the direction.
    Real dot = VecVec(gradient, r);
    if ((opts_.minimize && dot < 0) || (!opts_.minimize && dot > 0))
      KALDI_WARN << "Step direction has the wrong sign!  Routine will fail.";
  }
  
  // Now we're out of Alg. 7.4 and back into Alg. 7.5.
  // Alg. 7.4 returned r (using new_x_ as the location), and with \alpha_k = 1
  // as the initial guess, we're setting x_{k+1} = x_k + \alpha_k p_k, with
  // p_k = -r [hence the statement new_x_.Scale(-1.0)]., and \alpha_k = 1.
  // This is the first place we'll get the user to evaluate the function;
  // any backtracking (or acceptance of that step) occurs inside StepSizeIteration.
  // We're still within iteration k; we haven't yet finalized the step size.
  new_x_.Scale(-1.0);
  new_x_.AddVec(1.0, x_);
  if (&deriv_ != &gradient)
    deriv_.CopyFromVec(gradient);
  f_ = function_value;
  d_ = opts_.d;
  num_wolfe_i_failures_ = 0;
  num_wolfe_ii_failures_ = 0;
  last_failure_type_ = kNone;
  computation_state_ = kWithinStep;
}


template<typename Real>
bool OptimizeLbfgs<Real>::AcceptStep(Real function_value,
                                     const VectorBase<Real> &gradient) {
  // Save s_k = x_{k+1} - x_{k}, and y_k = \nabla f_{k+1} - \nabla f_k.
  SubVector<Real> s = S(k_), y = Y(k_);
  s.CopyFromVec(new_x_);
  s.AddVec(-1.0, x_); // s = new_x_ - x_.
  y.CopyFromVec(gradient);
  y.AddVec(-1.0, deriv_); // y = gradient - deriv_.
  
  // Warning: there is a division in the next line.  This could
  // generate inf or nan, but this wouldn't necessarily be an error
  // at this point because for zero step size or derivative we should
  // terminate the iterations.  But this is up to the calling code.
  Real prod = VecVec(y, s);
  rho_(k_ % opts_.m) = 1.0 / prod;
  Real len = s.Norm(2.0);

  if ((opts_.minimize && prod <= 1.0e-20) || (!opts_.minimize && prod >= -1.0e-20)
      || len == 0.0)
    return false; // This will force restart.
  
  KALDI_VLOG(3) << "Accepted step; length was " << len
                << ", prod was " << prod;
  RecordStepLength(len);
  
  // store x_{k+1} and the function value f_{k+1}.
  x_.CopyFromVec(new_x_);
  f_ = function_value;
  k_++;

  return true; // We successfully accepted the step.
}

template<typename Real>
void OptimizeLbfgs<Real>::RecordStepLength(Real s) {
  step_lengths_.push_back(s);
  if (step_lengths_.size() > static_cast<size_t>(opts_.avg_step_length))
    step_lengths_.erase(step_lengths_.begin(), step_lengths_.begin() + 1);
}


template<typename Real>
void OptimizeLbfgs<Real>::Restart(const VectorBase<Real> &x,
                                  Real f,
                                  const VectorBase<Real> &gradient) {
  // Note: we will consider restarting (the transition of x_ -> x)
  // as a step, even if it has zero step size.  This is necessary in
  // order for convergence to be detected.
  {
    Vector<Real> &diff(temp_);
    diff.CopyFromVec(x);
    diff.AddVec(-1.0, x_);
    RecordStepLength(diff.Norm(2.0));
  }
  k_ = 0; // Restart the iterations!  [But note that the Hessian,
  // whatever it was, stays as before.]
  if (&x_ != &x)
    x_.CopyFromVec(x);
  new_x_.CopyFromVec(x);
  f_ = f;
  computation_state_ = kBeforeStep;
  ComputeNewDirection(f, gradient);
}

template<typename Real>
void OptimizeLbfgs<Real>::StepSizeIteration(Real function_value,
                                            const VectorBase<Real> &gradient) {
  KALDI_VLOG(3) << "In step size iteration, function value changed "
                << f_ << " to " << function_value;
  
  // We're in some part of the backtracking, and the user is providing
  // the objective function value and gradient.
  // We're checking two conditions: Wolfe i) [the Armijo rule] and
  // Wolfe ii).
  
  // The Armijo rule (when minimizing) is:
  // f(k_k + \alpha_k p_k) <= f(x_k) + c_1 \alpha_k p_k^T \nabla f(x_k), where
  //  \nabla means the derivative.
  // Below, "temp" is the RHS of this equation, where (\alpha_k p_k) equals
  // (new_x_ - x_); we don't store \alpha or p_k separately, they are implicit
  // as the difference new_x_ - x_.

  // Below, pf is \alpha_k p_k^T \nabla f(x_k).
  Real pf = VecVec(new_x_, deriv_) - VecVec(x_, deriv_);
  Real temp = f_ + opts_.c1 * pf;
  
  bool wolfe_i_ok;
  if (opts_.minimize) wolfe_i_ok = (function_value <= temp);
  else wolfe_i_ok = (function_value >= temp);
  
  // Wolfe condition ii) can be written as:
  //  p_k^T \nabla f(x_k + \alpha_k p_k) >= c_2 p_k^T \nabla f(x_k)
  // p2f equals \alpha_k p_k^T \nabla f(x_k + \alpha_k p_k), where
  // (\alpha_k p_k^T) is (new_x_ - x_).
  // Note that in our version of Wolfe condition (ii) we have an extra
  // factor alpha, which doesn't affect anything.
  Real p2f = VecVec(new_x_, gradient) - VecVec(x_, gradient);
  //eps = (sizeof(Real) == 4 ? 1.0e-05 : 1.0e-10) *
  //(std::abs(p2f) + std::abs(pf));
  bool wolfe_ii_ok;
  if (opts_.minimize) wolfe_ii_ok = (p2f >= opts_.c2 * pf);
  else wolfe_ii_ok = (p2f <= opts_.c2 * pf);

  enum { kDecrease, kNoChange } d_action; // What do do with d_: leave it alone,
  // or take the square root.
  enum { kAccept, kDecreaseStep, kIncreaseStep, kRestart } iteration_action;
  // What we'll do in the overall iteration: accept this value, DecreaseStep
  // (reduce the step size), IncreaseStep (increase the step size), or kRestart
  // (set k back to zero).  Generally when we can't get both conditions to be
  // true with a reasonable period of time, it makes sense to restart, because
  // probably we've almost converged and got into numerical issues; from here
  // we'll just produced NaN's.  Restarting is a safe thing to do and the outer
  // code will quickly detect convergence.

  d_action = kNoChange; // the default.
  
  if (wolfe_i_ok && wolfe_ii_ok) {
    iteration_action = kAccept;
    d_action = kNoChange; // actually doesn't matter, it'll get reset.
  } else if (!wolfe_i_ok) {
    // If wolfe i) [the Armijo rule] failed then we went too far (or are
    // meeting numerical problems).
    if (last_failure_type_ == kWolfeII) { // Last time we failed it was Wolfe ii).
      // When we switch between them we decrease d.
      d_action = kDecrease;
    }
    iteration_action = kDecreaseStep;
    last_failure_type_ = kWolfeI;
    num_wolfe_i_failures_++;
  } else if (!wolfe_ii_ok) {
    // Curvature condition failed -> we did not go far enough.
    if (last_failure_type_ == kWolfeI) // switching between wolfe i and ii failures->
      d_action = kDecrease; // decrease value of d.
    iteration_action = kIncreaseStep;
    last_failure_type_ = kWolfeII;
    num_wolfe_ii_failures_++;
  }

  // Test whether we've been switching too many times betwen wolfe i) and ii)
  // failures, or overall have an excessive number of failures.  We just give up
  // and restart L-BFGS.  Probably we've almost converged.
  if (num_wolfe_i_failures_ + num_wolfe_ii_failures_ >
      opts_.max_line_search_iters) {
    KALDI_VLOG(2) << "Too many steps in line search -> restarting.";
    iteration_action = kRestart;
  }

  if (d_action == kDecrease)
    d_ = std::sqrt(d_);
  
  KALDI_VLOG(3) << "d = " << d_ << ", iter = " << k_ << ", action = "
                << (iteration_action == kAccept ? "accept" :
                    (iteration_action == kDecreaseStep ? "decrease" :
                     (iteration_action == kIncreaseStep ? "increase" :
                      "reject")));
  
  // Note: even if iteration_action != Restart at this point,
  // some code below may set it to Restart.
  if (iteration_action == kAccept) {
    if (AcceptStep(function_value, gradient)) { // If we did
      // not detect a problem while accepting the step..
      computation_state_ = kBeforeStep;
      ComputeNewDirection(function_value, gradient);
    } else {
      KALDI_VLOG(2) << "Restarting L-BFGS computation; problem found while "
                    << "accepting step.";
      iteration_action = kRestart; // We'll have to restart now.
    }
  }
  if (iteration_action == kDecreaseStep || iteration_action == kIncreaseStep) {
    Real scale = (iteration_action == kDecreaseStep ? 1.0 / d_ : d_);
    temp_.CopyFromVec(new_x_);
    new_x_.Scale(scale);
    new_x_.AddVec(1.0 - scale, x_);
    if (new_x_.ApproxEqual(temp_, 0.0)) {
      // Value of new_x_ did not change at all --> we must restart.
      KALDI_VLOG(3) << "Value of x did not change, when taking step; "
                    << "will restart computation.";
      iteration_action = kRestart;
    }
    if (new_x_.ApproxEqual(temp_, 1.0e-08) &&
        std::abs(f_ - function_value) < 1.0e-08 *
        std::abs(f_) && iteration_action == kDecreaseStep) {
      // This is common and due to roundoff.
      KALDI_VLOG(3) << "We appear to be backtracking while we are extremely "
                    << "close to the old value; restarting.";
      iteration_action = kRestart;
    }
        
    if (iteration_action == kDecreaseStep) {
      num_wolfe_i_failures_++;
      last_failure_type_ = kWolfeI;
    } else {
      num_wolfe_ii_failures_++;
      last_failure_type_ = kWolfeII;
    }
  }
  if (iteration_action == kRestart) {
    // We want to restart the computation.  If the objf at new_x_ is
    // better than it was at x_, we'll start at new_x_, else at x_.
    bool use_newx;
    if (opts_.minimize) use_newx = (function_value < f_);
    else use_newx = (function_value > f_);
    KALDI_VLOG(3) << "Restarting computation.";
    if (use_newx) Restart(new_x_, function_value, gradient);
    else Restart(x_, f_, deriv_);
  }
}

template<typename Real>
void OptimizeLbfgs<Real>::DoStep(Real function_value,
                                 const VectorBase<Real> &gradient) {
  if (opts_.minimize ? function_value < best_f_ : function_value > best_f_) {
    best_f_ = function_value;
    best_x_.CopyFromVec(new_x_);
  }
  if (computation_state_ == kBeforeStep)
    ComputeNewDirection(function_value, gradient);
  else // kWithinStep{1,2,3}
    StepSizeIteration(function_value, gradient);
}

template<typename Real>
void OptimizeLbfgs<Real>::DoStep(Real function_value,
                                 const VectorBase<Real> &gradient,
                                 const VectorBase<Real> &diag_approx_2nd_deriv) {
  if (opts_.minimize ? function_value < best_f_ : function_value > best_f_) {
    best_f_ = function_value;
    best_x_.CopyFromVec(new_x_);
  }
  if (opts_.minimize) {
    KALDI_ASSERT(diag_approx_2nd_deriv.Min() > 0.0);
  } else {
    KALDI_ASSERT(diag_approx_2nd_deriv.Max() < 0.0);
  }
  H_was_set_ = true;
  H_.CopyFromVec(diag_approx_2nd_deriv);
  H_.InvertElements();
  DoStep(function_value, gradient);
}

template<typename Real>
const VectorBase<Real>&
OptimizeLbfgs<Real>::GetValue(Real *objf_value) const {
  if (objf_value != NULL) *objf_value = best_f_;
  return best_x_;
}

// to compute the alpha, we are minimizing f(x) =  x^T b - 0.5 x_k^T A x_k  along
// direction p_k... consider alpha
// d/dx of f(x) = b - A x_k = r.

// Notation based on Sec. 5.1 of Nocedal and Wright
// Computation based on Alg. 5.2 of Nocedal and Wright (Pg. 112)
// Notation (replicated for convenience):
//  To solve Ax=b for x
//  k : current iteration
//  x_k : estimate of x (at iteration k)
//  r_k : residual ( r_k \eqdef A x_k - b )
//  \alpha_k : step size
//  p_k : A-conjugate direction
//  \beta_k  : coefficient used in A-conjugate direction computation for next
//  iteration
//  
//  Algo.  LinearCG(A,b,x_0)
//  ========================
//  r_0 = Ax_0 - b
//  p_0 = -r_0
//  k = 0
//
//  while r_k != 0
//    \alpha_k = (r_k^T  r_k) / (p_k^T  A  p_k)
//    x_{k+1} = x_k + \alpha_k  p_k;
//    r_{k+1} = r_k + \alpha_k  A  p_k
//    \beta_{k+1} = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_K}
//    p_{k+1} = -r_{k+1} + \beta_{k+1} p_k
//    k = k + 1
//  end

template<class Real>
int32 LinearCgd(const LinearCgdOptions &opts,
                const SpMatrix<Real> &A,
                const VectorBase<Real> &b,
                VectorBase<Real> *x) {
  // Initialize the variables
  //
  int32 M = A.NumCols();

  Matrix<Real> storage(4, M);
  SubVector<Real> r(storage, 0), p(storage, 1), Ap(storage, 2), x_orig(storage, 3);
  p.CopyFromVec(b);
  p.AddSpVec(-1.0, A, *x, 1.0);  // p_0 = b - A x_0
  r.AddVec(-1.0, p);  // r_0 = - p_0
  x_orig.CopyFromVec(*x);  // in case of failure.
  
  Real r_cur_norm_sq = VecVec(r, r),
      r_initial_norm_sq = r_cur_norm_sq,
      r_recompute_norm_sq = r_cur_norm_sq;

  KALDI_VLOG(5) << "In linear CG: initial norm-square of residual = "
                << r_initial_norm_sq;
  
  KALDI_ASSERT(opts.recompute_residual_factor <= 1.0);
  Real max_error_sq = std::max<Real>(opts.max_error * opts.max_error,
                                     std::numeric_limits<Real>::min()),
      residual_factor = opts.recompute_residual_factor *
                        opts.recompute_residual_factor,
      inv_residual_factor = 1.0 / residual_factor;
  
  // Note: although from a mathematical point of view the method should converge
  // after M iterations, in practice (due to roundoff) it does not always
  // converge to good precision after that many iterations so we let the maximum
  // be M + 5 instead.
  int32 k = 0;
  for (; k < M + 5 && k != opts.max_iters; k++) {
    // Note: we'll break from this loop if we converge sooner due to
    // max_error.
    Ap.AddSpVec(1.0, A, p, 0.0);  // Ap = A p

    // Below is how the code used to look.
    // // next line: \alpha_k = (r_k^T r_k) / (p_k^T A p_k)
    // Real alpha = r_cur_norm_sq / VecVec(p, Ap);
    // 
    // We changed r_cur_norm_sq below to -VecVec(p, r).  Although this is
    // slightly less efficient, it seems to make the algorithm dramatically more
    // robust.  Note that -p^T r is the mathematically more natural quantity to
    // use here, that corresponds to minimizing along that direction... r^T r is
    // recommended in Nocedal and Wright only as a kind of optimization as it is
    // supposed to be the same as -p^T r and we already have it computed.
    Real alpha = -VecVec(p, r) / VecVec(p, Ap);
    
    // next line: x_{k+1} = x_k + \alpha_k p_k;
    x->AddVec(alpha, p);
    // next line: r_{k+1} = r_k + \alpha_k A p_k
    r.AddVec(alpha, Ap);
    Real r_next_norm_sq = VecVec(r, r);
    
    if (r_next_norm_sq < residual_factor * r_recompute_norm_sq ||
        r_next_norm_sq > inv_residual_factor * r_recompute_norm_sq) {
         
      // Recompute the residual from scratch if the residual norm has decreased
      // a lot; this costs an extra matrix-vector multiply, but helps keep the
      // residual accurate.
      // Also do the same if the residual norm has increased a lot since
      // the last time we recomputed... this shouldn't happen often, but
      // it can indicate bad stuff is happening.
      
      // r_{k+1} = A x_{k+1} - b
      r.AddSpVec(1.0, A, *x, 0.0);
      r.AddVec(-1.0, b);
      r_next_norm_sq = VecVec(r, r);
      r_recompute_norm_sq = r_next_norm_sq;

      KALDI_VLOG(5) << "In linear CG: recomputing residual.";
    }
    KALDI_VLOG(5) << "In linear CG: k = " << k
                  << ", r_next_norm_sq = " << r_next_norm_sq;
    // Check if converged.
    if (r_next_norm_sq <= max_error_sq)
      break;
    
    // next line: \beta_{k+1} = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_K}
    Real beta_next = r_next_norm_sq / r_cur_norm_sq;
    // next lines: p_{k+1} = -r_{k+1} + \beta_{k+1} p_k
    Vector<Real> p_old(p);
    p.Scale(beta_next);
    p.AddVec(-1.0, r);
    r_cur_norm_sq = r_next_norm_sq;
  }

  // note: the first element of the && is only there to save compute.
  // the residual r is A x - b, and r_cur_norm_sq and r_initial_norm_sq are
  // of the form r * r, so it's clear that b * b has the right dimension to
  // compare with the residual.
  if (r_cur_norm_sq > r_initial_norm_sq &&
      r_cur_norm_sq > r_initial_norm_sq + 1.0e-10 * VecVec(b, b)) {
    KALDI_WARN << "Doing linear CGD in dimension " << A.NumRows() << ", after " << k
              << " iterations the squared residual has got worse, "
               << r_cur_norm_sq << " > " << r_initial_norm_sq
               << ".  Will do an exact optimization.";
    SolverOptions opts("called-from-linearCGD");
    x->CopyFromVec(x_orig);
    SolveQuadraticProblem(A, b, opts, x);
  }
  return k;
} 
    
// Instantiate the class for float and double.
template
class OptimizeLbfgs<float>;
template
class OptimizeLbfgs<double>;


template
int32 LinearCgd<float>(const LinearCgdOptions &opts,
                      const SpMatrix<float> &A, const VectorBase<float> &b,
                      VectorBase<float> *x);

template
int32 LinearCgd<double>(const LinearCgdOptions &opts,
                        const SpMatrix<double> &A, const VectorBase<double> &b,
                        VectorBase<double> *x);

} // end namespace kaldi
