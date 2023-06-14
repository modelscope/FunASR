// matrix/kaldi-gpsr.cc

// Copyright 2010-2012   Liang Lu,  Arnab Ghoshal

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

// This is an implementation of the GPSR algorithm. See, Figueiredo, Nowak and
// Wright, "Gradient Projection for Sparse Reconstruction: Application to
// Compressed Sensing and Other Inverse Problems," IEEE Journal of Selected
// Topics in Signal Processing, vol. 1, no. 4, pp. 586-597, 2007.
// http://dx.doi.org/10.1109/JSTSP.2007.910281

#include <algorithm>
#include <string>
#include <vector>
using std::vector;

#include "matrix/kaldi-gpsr.h"

namespace kaldi {

/// This calculates the objective function: \f$ c^T z + 0.5 * z^T B z, \f$
/// where z is formed by stacking u and v, and B = [H -H; -H H].
double GpsrObjective(const SpMatrix<double> &H, const Vector<double> &c,
                     const Vector<double> &u, const Vector<double> &v) {
  KALDI_ASSERT(u.Dim() == v.Dim() && u.Dim() > 0);
  KALDI_ASSERT(c.Dim() == 2 * u.Dim());
  KALDI_VLOG(2) << "u dim = " << u.Dim() << ", v dim = " << v.Dim()
                << ", c dim = " << c.Dim();

  MatrixIndexT dim = u.Dim();
  Vector<double> H_x(dim), x(dim);
  // x = u - v, where u_i = (x_i)_+; v_i = (-x_i)_+; and (x)_+ = max{0,x}
  x.CopyFromVec(u);
  x.AddVec(-1.0, v);

  // Calculate c^T z = c^T [u^T v^T]^T
  double objf = VecVec(c.Range(0, dim), u);
  objf += VecVec(c.Range(dim, dim), v);

  // Now, calculate the quadratic term: z^T B z = (u-v)^T H (u-v) = x^T H x
  H_x.AddSpVec(1.0, H, x, 0.0);
  objf += 0.5 * VecVec(x, H_x);
  return objf;
}

/// This calculates the gradient: \f$ c + B z, \f$
/// where z is formed by stacking u and v, and B = [H -H; -H H].
void GpsrGradient(const SpMatrix<double> &H, const Vector<double> &c,
                     const Vector<double> &u, const Vector<double> &v,
                     Vector<double> *grad_u, Vector<double> *grad_v) {
  KALDI_ASSERT(u.Dim() == v.Dim() && u.Dim() > 0);
  KALDI_ASSERT(u.Dim() == grad_u->Dim() && v.Dim() == grad_v->Dim());
  KALDI_ASSERT(c.Dim() == 2 * u.Dim());
  KALDI_VLOG(2) << "u dim = " << u.Dim() << ", v dim = " << v.Dim()
                << ", c dim = " << c.Dim();

  MatrixIndexT dim = u.Dim();
  Vector<double> H_x(dim), x(dim);
  // x = u - v, where u_i = (x_i)_+; v_i = (-x_i)_+; and (x)_+ = max{0,x}
  x.CopyFromVec(u);
  x.AddVec(-1.0, v);
  // To calculate B z = [ H (u-v); -H (u-v) ] = [ H x; -H x ], we only need H x
  H_x.AddSpVec(1.0, H, x, 0.0);
  grad_u->CopyFromVec(c.Range(0, dim));
  grad_u->AddVec(1.0, H_x);
  grad_v->CopyFromVec(c.Range(dim, dim));
  grad_v->AddVec(-1.0, H_x);
}

/// Returns the initial guess of step size in the feasible direction.
/// This is the exact minimizer of the objective function along the feasible
/// direction, which is the negative gradient projected on to the constraint
/// set, or the non-negative orthant, in this case:
/// \f[ \alpha = \frac{g^T g}{g^T B g},  \f]
/// where g is the projected gradient, formed by stacking the projected
/// gradients for the positive & negative parts (u & v); and B = [H -H; -H H].
double GpsrBasicAlpha(const SpMatrix<double> &H, const Vector<double> &u,
                      const Vector<double> &v, const Vector<double> &grad_u,
                      const Vector<double> &grad_v) {
  KALDI_ASSERT(H.NumRows() == grad_u.Dim() && grad_u.Dim() == grad_v.Dim() &&
               grad_u.Dim() > 0);
  KALDI_VLOG(2) << "grad_u dim = " << grad_u.Dim() << ", grad_v dim = "
                << grad_v.Dim() << ", H rows = " << H.NumRows();
  MatrixIndexT dim = grad_u.Dim();

  // Find the projection of the gradient on the nonnegative orthant, or, more
  // precisely, the projection s.t. the next iterate will be in the orthant.
  Vector<double> proj_grad_u(dim);
  Vector<double> proj_grad_v(dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    proj_grad_u(i) = (u(i) > 0 || grad_u(i) < 0)? grad_u(i) : 0;
    proj_grad_v(i) = (v(i) > 0 || grad_v(i) < 0)? grad_v(i) : 0;
  }

  // The numerator: g^T g = g_u^T g_u + g_v^T g_v
  double alpha = VecVec(proj_grad_u, proj_grad_u);
  alpha += VecVec(proj_grad_v, proj_grad_v);

  // The denominator: g^T B g = (g_u - g_v)^T H (g_u - g_v)
  Vector<double> diff_g(proj_grad_u);
  diff_g.AddVec(-1.0, proj_grad_v);
  Vector<double> H_diff_g(dim);
  H_diff_g.AddSpVec(1.0, H, diff_g, 0.0);
  alpha /= (VecVec(diff_g, H_diff_g) + DBL_EPSILON);
  return alpha;
}

/// This calculates the coefficient for the linear term used in the
/// bound-constrained quadratic program: c = \tau 1_{2n} + [-g; g]
void GpsrCalcLinearCoeff(double tau, const Vector<double> &g,
                         Vector<double> *c) {
  KALDI_ASSERT(c->Dim() == 2 * g.Dim() && g.Dim() != 0);
  MatrixIndexT dim = g.Dim();
  c->Set(tau);
  c->Range(0, dim).AddVec(-1.0, g);
  c->Range(dim, dim).AddVec(1.0, g);
}

// This removes the L1 penalty term, and uses conjugate gradient to solve the
// resulting quadratic problem while keeping the zero elements fixed at 0.
double Debias(const GpsrConfig &opts, const SpMatrix<double> &H,
              const Vector<double> &g, Vector<double> *x) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
//  KALDI_ASSERT(H.IsPosDef() &&
//               "Must have positive definite matrix for conjugate gradient.");
  MatrixIndexT dim = x->Dim();

  Vector<double> x_bias(*x);
  Vector<double> nonzero_indices(dim);
  // Initialize the index of non-zero elements in x
  for (MatrixIndexT i = 0; i < dim; i++)
    nonzero_indices(i) = (x_bias(i) == 0)? 0.0 : 1.0;

  Vector<double> residual(dim);
  Vector<double> conj_direction(dim);
  Vector<double> resid_change(dim);
  double alpha_cg;  // CG step size for iterate: x <- x + \alpha p
  double beta_cg;   // CG step size for conj. direction: p <- \beta p - r
  double resid_prod, resid_prod_new;  // inner product of residual vectors

  // Calculate the initial residual: r = H x_0 - g
  residual.AddSpVec(1.0, H, x_bias, 0.0);
  residual.AddVec(-1.0, g);
  residual.MulElements(nonzero_indices);  // only change non-zero elements of x

  conj_direction.CopyFromVec(residual);
  conj_direction.Scale(-1.0);  // Initial conjugate direction p = -r
  resid_prod = VecVec(residual, residual);

  // set the convergence threshold for residual
  double tol_debias = opts.stop_thresh_debias * VecVec(residual, residual);

  for (int32 iter = 0; iter < opts.max_iters_debias; iter++) {
    resid_change.AddSpVec(1.0, H, conj_direction, 0.0);
    resid_change.MulElements(nonzero_indices);  // only change non-zero elements

    alpha_cg = resid_prod / VecVec(conj_direction, resid_change);
    x_bias.AddVec(alpha_cg, conj_direction);
    residual.AddVec(alpha_cg, resid_change);

    resid_prod_new = VecVec(residual, residual);
    beta_cg = resid_prod_new / resid_prod;
    conj_direction.Scale(beta_cg);
    conj_direction.AddVec(-1.0, residual);
    resid_prod = resid_prod_new;

    if (resid_prod < tol_debias) {
      KALDI_VLOG(1) << "iter=" << iter << "\t residual =" << resid_prod
                    << "\t tol_debias=" << tol_debias;
      break;
    }
  }  // end CG iters

  x->CopyFromVec(x_bias);
  return resid_prod;
}

template<>
double GpsrBasic(const GpsrConfig &opts, const SpMatrix<double> &H,
                 const Vector<double> &g, Vector<double> *x,
                 const char *debug_str) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  MatrixIndexT dim = x->Dim();
  if (H.IsZero(0.0)) {
    KALDI_WARN << "Zero quadratic term in GPSR for " << debug_str
               << ": leaving it unchanged.";
    return 0.0;
  }

  // initialize the positive (u) and negative (v) parts of x, s.t. x = u - v
  Vector<double> u(dim, kSetZero);
  Vector<double> v(dim, kSetZero);
  for (MatrixIndexT i = 0; i < dim; i++) {
    if ((*x)(i) > 0) {
      u(i) = (*x)(i);
    } else {
      v(i) = -(*x)(i);
    }
  }

  double tau = opts.gpsr_tau;  // May be modified later.
  Vector<double> c(2*dim);
  GpsrCalcLinearCoeff(tau, g, &c);

  double objf_ori = GpsrObjective(H, c, u, v);  // the obj. function at start
  KALDI_VLOG(2) << "GPSR for " << debug_str << ": tau = " << tau
                << ";\t objf = " << objf_ori;

  Vector<double> grad_u(dim);
  Vector<double> grad_v(dim);
  Vector<double> delta_u(dim);
  Vector<double> delta_v(dim);
  Vector<double> u_new(dim);
  Vector<double> v_new(dim);
  double objf_old, objf_new, num_zeros;
  bool keep_going = true;

  for (int32 iter = 0; keep_going; iter++) {
    objf_old = GpsrObjective(H, c, u, v);
    GpsrGradient(H, c, u, v, &grad_u, &grad_v);
    double alpha = GpsrBasicAlpha(H, u, v, grad_u, grad_v);
    if (alpha < opts.alpha_min) alpha = opts.alpha_min;
    if (alpha > opts.alpha_max) alpha = opts.alpha_max;

    // This is the backtracking line search part:
    for (int32 k = 0; k < opts.max_iters_backtrak; k++) {
      // Calculate the potential new iterate: [z_k - \alpha_k \grad F(z_k)]_+
      u_new.CopyFromVec(u);
      u_new.AddVec(-alpha, grad_u);
      u_new.ApplyFloor(0.0);
      v_new.CopyFromVec(v);
      v_new.AddVec(-alpha, grad_v);
      v_new.ApplyFloor(0.0);

      delta_u.CopyFromVec(u_new);
      delta_v.CopyFromVec(v_new);
      delta_u.AddVec(-1.0, u);
      delta_v.AddVec(-1.0, v);

      double delta_objf_apx = opts.gpsr_mu * (VecVec(grad_u, delta_u) +
                                              VecVec(grad_v, delta_v));
      objf_new = GpsrObjective(H, c, u_new, v_new);
      double delta_objf_real = objf_new - objf_old;

      KALDI_VLOG(2) << "GPSR for " << debug_str << ": iter " << iter
                    << "; tau = " << tau << ";\t objf = " << objf_new
                    << ";\t alpha = " << alpha << ";\t delta_apx = "
                    << delta_objf_apx << ";\t delta_real = " << delta_objf_real;

      if (delta_objf_real < delta_objf_apx + DBL_EPSILON)
        break;
      else
        alpha *= opts.gpsr_beta;

      if (k == opts.max_iters_backtrak - 1) {  // Stop further optimization
        KALDI_WARN << "Backtracking line search did not decrease objective.";
        u_new.CopyFromVec(u);
        u_new.ApplyFloor(0.0);
        v_new.CopyFromVec(v);
        v_new.ApplyFloor(0.0);
        delta_u.SetZero();
        delta_v.SetZero();
      }
    }  // end of backtracking line search

    x->CopyFromVec(u_new);
    x->AddVec(-1.0, v_new);

    num_zeros = 0;
    for (MatrixIndexT i = 0; i < dim; i++)
      if ((*x)(i) == 0)
        num_zeros++;

    // ad hoc way to modify tau, if the solution is too sparse
    if ((num_zeros / static_cast<double>(dim)) > opts.max_sparsity) {
      std::ostringstream msg;
      msg << num_zeros << " out of " << dim << " dimensions set to 0. "
          << "Changing tau from " << tau;
      tau *= opts.tau_reduction;
      GpsrCalcLinearCoeff(tau, g, &c);  // Recalculate c with new tau
      double tmp_objf = GpsrObjective(H, c, u, v);
      msg << " to " << tau << ".\n\tStarting objective function changed from "
          << objf_ori << " to " << tmp_objf << ".";
      KALDI_LOG << "GPSR for " << debug_str << ": " << msg.str();
      iter = 0;
      keep_going = true;
      continue;
    }

    u.CopyFromVec(u_new);
    v.CopyFromVec(v_new);
    double delta = (delta_u.Norm(2.0) + delta_v.Norm(2.0)) / x->Norm(2.0);
    KALDI_VLOG(1) << "GPSR for " << debug_str << ": iter " << iter
                  << ", objf = " << objf_new << ", delta = " << delta;

    keep_going = (iter < opts.max_iters) && (delta > opts.stop_thresh);

    KALDI_VLOG(3) << "GPSR for " << debug_str << ": iter " << iter
                  << ", objf = " << objf_new << ", value = " << x;
  }

  if (num_zeros != 0) {
    KALDI_LOG << "GPSR for " << debug_str << ": number of 0's = " << num_zeros
              << " out of " << dim << " dimensions.";
  }

  if (opts.debias && num_zeros != 0) {
    double residual = Debias(opts, H, g, x);
    KALDI_LOG << "Debiasing: new residual = " << residual;
  }
  return objf_new - objf_ori;
}

template<>
float GpsrBasic(const GpsrConfig &opts, const SpMatrix<float> &H,
                const Vector<float> &g, Vector<float> *x,
                const char *debug_str) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  SpMatrix<double> Hd(H);
  Vector<double> gd(g);
  Vector<double> xd(*x);
  float ans = GpsrBasic(opts, Hd, gd, &xd, debug_str);
  x->CopyFromVec(xd);
  return ans;
}

template<>
double GpsrBB(const GpsrConfig &opts, const SpMatrix<double> &H,
              const Vector<double> &g, Vector<double> *x,
              const char *debug_str) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  MatrixIndexT dim = x->Dim();
  if (H.IsZero(0.0)) {
    KALDI_WARN << "Zero quadratic term in GPSR for " << debug_str
               << ": leaving it unchanged.";
    return 0.0;
  }

  // initialize the positive (u) and negative (v) parts of x, s.t. x = u - v
  Vector<double> u(dim, kSetZero);
  Vector<double> v(dim, kSetZero);
  for (MatrixIndexT i = 0; i < dim; i++) {
    if ((*x)(i) > 0) {
      u(i) = (*x)(i);
    } else {
      v(i) = -(*x)(i);
    }
  }

  double tau = opts.gpsr_tau;  // May be modified later.
  Vector<double> c(2*dim);
  GpsrCalcLinearCoeff(tau, g, &c);

  double objf_ori = GpsrObjective(H, c, u, v);  // the obj. function at start
  KALDI_VLOG(2) << "GPSR for " << debug_str << ": tau = " << tau
                << ";\t objf = " << objf_ori;

  Vector<double> grad_u(dim);
  Vector<double> grad_v(dim);
  Vector<double> delta_u(dim);
  Vector<double> delta_v(dim);
  Vector<double> delta_x(dim);
  Vector<double> H_delta_x(dim);
  Vector<double> u_new(dim);
  Vector<double> v_new(dim);
  double objf_old, objf_new, num_zeros;
  bool keep_going = true;
  double alpha = 1.0;

  for (int32 iter = 0; keep_going; iter++) {
    objf_old = GpsrObjective(H, c, u, v);
    GpsrGradient(H, c, u, v, &grad_u, &grad_v);

    // Calculate the new step: [z_k - \alpha_k \grad F(z_k)]_+ - z_k
    delta_u.CopyFromVec(u);
    delta_u.AddVec(-alpha, grad_u);
    delta_u.ApplyFloor(0.0);
    delta_u.AddVec(-1.0, u);
    delta_v.CopyFromVec(v);
    delta_v.AddVec(-alpha, grad_v);
    delta_v.ApplyFloor(0.0);
    delta_v.AddVec(-1.0, v);

    delta_x.CopyFromVec(delta_u);
    delta_x.AddVec(-1.0, delta_v);
    H_delta_x.AddSpVec(1.0, H, delta_x, 0.0);
    double dx_H_dx = VecVec(delta_x, H_delta_x);

    double lambda = -(VecVec(delta_u, grad_u) + VecVec(delta_v, grad_v))
                / (dx_H_dx + DBL_EPSILON);  // step length
    if (lambda < 0)
      KALDI_WARN << "lambda is less than zero";
    if (lambda > 1.0) lambda = 1.0;

    //update alpha
    alpha = (VecVec(delta_u, delta_u) + VecVec(delta_v, delta_v))
                / (dx_H_dx + DBL_EPSILON);
    if (dx_H_dx <= 0) {
      KALDI_WARN << "nonpositive curvature detected";
      alpha = opts.alpha_max;
    }
    else if (alpha < opts.alpha_min)
      alpha = opts.alpha_min;
    else if (alpha > opts.alpha_max) alpha = opts.alpha_max;

    u_new.CopyFromVec(delta_u);
    u_new.Scale(lambda);
    v_new.CopyFromVec(delta_v);
    v_new.Scale(lambda);
    u_new.AddVec(1.0, u);
    v_new.AddVec(1.0, v);

    objf_new = GpsrObjective(H, c, u_new, v_new);
    double delta_objf = objf_old - objf_new;
    KALDI_VLOG(2) << "GPSR for " << debug_str << ": iter " << iter
                  << "; tau = " << tau << ";\t objf = " << objf_new
                  << ";\t alpha = " << alpha << ";\t delta_real = "
                  << delta_objf;

    u.CopyFromVec(u_new);
    v.CopyFromVec(v_new);
    x->CopyFromVec(u);
    x->AddVec(-1.0, v);

    num_zeros = 0;
    for (MatrixIndexT i = 0; i < dim; i++)
      if ((*x)(i) == 0)
        num_zeros++;

    // ad hoc way to modify tau, if the solution is too sparse
    if ((num_zeros / static_cast<double>(dim)) > opts.max_sparsity) {
      std::ostringstream msg;
      msg << num_zeros << " out of " << dim << " dimensions set to 0. "
          << "Changing tau from " << tau;
      tau *= 0.9;
      GpsrCalcLinearCoeff(tau, g, &c);  // Recalculate c with new tau
      double tmp_objf = GpsrObjective(H, c, u, v);
      msg << " to " << tau << ".\n\tStarting objective function changed from "
          << objf_ori << " to " << tmp_objf << ".";
      KALDI_LOG << "GPSR for " << debug_str << ": " << msg.str();
      iter = 0;
      keep_going = true;
      continue;
    }

    double delta = (delta_u.Norm(2.0) + delta_v.Norm(2.0)) / x->Norm(2.0);
    KALDI_VLOG(1) << "GPSR for " << debug_str << ": iter " << iter
                  << ", objf = " << objf_new << ", delta = " << delta;

    keep_going = (iter < opts.max_iters) && (delta > opts.stop_thresh);

    KALDI_VLOG(3) << "GPSR for " << debug_str << ": iter " << iter
                  << ", objf = " << objf_new << ", value = " << x;
  }

  if (num_zeros != 0) {
    KALDI_LOG << "GPSR for " << debug_str << ": number of 0's = " << num_zeros
              << " out of " << dim << " dimensions.";
  }

  if (opts.debias && num_zeros != 0) {
    double residual = Debias(opts, H, g, x);
    KALDI_LOG << "Debiasing: new residual = " << residual;
  }
  return objf_new - objf_ori;
}

template<>
float GpsrBB(const GpsrConfig &opts, const SpMatrix<float> &H,
             const Vector<float> &g, Vector<float> *x,
             const char *debug_str) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  SpMatrix<double> Hd(H);
  Vector<double> gd(g);
  Vector<double> xd(*x);
  float ans = GpsrBB(opts, Hd, gd, &xd, debug_str);
  x->CopyFromVec(xd);
  return ans;
}

}  // namespace kaldi

