// matrix/kaldi-gpsr.h

// Copyright 2012  Arnab Ghoshal

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

#ifndef KALDI_MATRIX_KALDI_GPSR_H_
#define KALDI_MATRIX_KALDI_GPSR_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/options-itf.h"

namespace kaldi {

/// This is an implementation of the GPSR algorithm. See, Figueiredo, Nowak and
/// Wright, "Gradient Projection for Sparse Reconstruction: Application to
/// Compressed Sensing and Other Inverse Problems," IEEE Journal of Selected
/// Topics in Signal Processing, vol. 1, no. 4, pp. 586-597, 2007.
/// http://dx.doi.org/10.1109/JSTSP.2007.910281

/// The GPSR algorithm, described in Figueiredo, et al., 2007, solves:
/// \f[ \min_x 0.5 * ||y - Ax||_2^2 + \tau ||x||_1, \f]
/// where \f$ x \in R^n, y \in R^k \f$, and \f$ A \in R^{n \times k} \f$.
/// In this implementation, we solve:
/// \f[ \min_x 0.5 * x^T H x - g^T x + \tau ||x||_1, \f]
/// which is the more natural form in which such problems arise in our case.
/// Here, \f$ H = A^T A \in R^{n \times n} \f$ and \f$ g = A^T y \in R^n \f$.


/** \struct GpsrConfig
 *  Configuration variables needed in the GPSR algorithm.
 */
struct GpsrConfig {
  bool use_gpsr_bb;  ///< Use the Barzilai-Borwein gradient projection method

  /// The following options are common to both the basic & Barzilai-Borwein
  /// versions of GPSR
  double stop_thresh;  ///< Stopping threshold
  int32 max_iters;  ///< Maximum number of iterations
  double gpsr_tau;  ///< Regularization scale
  double alpha_min;  ///< Minimum step size in the feasible direction
  double alpha_max;  ///< Maximum step size in the feasible direction
  double max_sparsity;  ///< Maximum percentage of dimensions set to 0
  double tau_reduction;  ///< Multiply tau by this if max_sparsity reached

  /// The following options are for the backtracking line search in basic GPSR.
  /// Step size reduction factor in backtracking line search. 0 < beta < 1
  double gpsr_beta;
  /// Improvement factor in backtracking line search, i.e. the new objective
  /// function must be less than the old one by mu times the gradient in the
  /// direction of the change in x. 0 < mu < 1
  double gpsr_mu;
  int32 max_iters_backtrak;  ///< Max iterations for backtracking line search

  bool debias;  ///< Do debiasing, i.e. unconstrained optimization at the end
  double stop_thresh_debias;  ///< Stopping threshold for debiasing stage
  int32 max_iters_debias;  ///< Maximum number of iterations for debiasing stage

  GpsrConfig() {
    use_gpsr_bb = true;

    stop_thresh = 0.005;
    max_iters = 100;
    gpsr_tau = 10;
    alpha_min = 1.0e-10;
    alpha_max = 1.0e+20;
    max_sparsity = 0.9;
    tau_reduction = 0.8;

    gpsr_beta = 0.5;
    gpsr_mu = 0.1;
    max_iters_backtrak = 50;

    debias = false;
    stop_thresh_debias = 0.001;
    max_iters_debias = 50;
  }

  void Register(OptionsItf *opts);
};

inline void GpsrConfig::Register(OptionsItf *opts) {
  std::string module = "GpsrConfig: ";
  opts->Register("use-gpsr-bb", &use_gpsr_bb, module+
                 "Use the Barzilai-Borwein gradient projection method.");

  opts->Register("stop-thresh", &stop_thresh, module+
                 "Stopping threshold for GPSR.");
  opts->Register("max-iters", &max_iters, module+
                 "Maximum number of iterations of GPSR.");
  opts->Register("gpsr-tau", &gpsr_tau, module+
                 "Regularization scale for GPSR.");
  opts->Register("alpha-min", &alpha_min, module+
                 "Minimum step size in feasible direction.");
  opts->Register("alpha-max", &alpha_max, module+
                 "Maximum step size in feasible direction.");
  opts->Register("max-sparsity", &max_sparsity, module+
                 "Maximum percentage of dimensions set to 0.");
  opts->Register("tau-reduction", &tau_reduction, module+
                 "Multiply tau by this if maximum sparsity is reached.");

  opts->Register("gpsr-beta", &gpsr_beta, module+
                 "Step size reduction factor in backtracking line search (0<beta<1).");
  opts->Register("gpsr-mu", &gpsr_mu, module+
                 "Improvement factor in backtracking line search (0<mu<1).");
  opts->Register("max-iters-backtrack", &max_iters_backtrak, module+
                 "Maximum number of iterations of backtracking line search.");

  opts->Register("debias", &debias, module+
                 "Do final debiasing step.");
  opts->Register("stop-thresh-debias", &stop_thresh_debias, module+
                 "Stopping threshold for debiaisng step.");
  opts->Register("max-iters-debias", &max_iters_debias, module+
                 "Maximum number of iterations of debiasing.");
}

/// Solves a quadratic program in \f$ x \f$, with L_1 regularization:
/// \f[ \min_x 0.5 * x^T H x - g^T x + \tau ||x||_1. \f]
/// This is similar to SolveQuadraticProblem() in sp-matrix.h with an added
/// L_1 term.
template<typename Real>
Real Gpsr(const GpsrConfig &opts, const SpMatrix<Real> &H,
          const Vector<Real> &g, Vector<Real> *x,
          const char *debug_str = "[unknown]") {
  if (opts.use_gpsr_bb)
    return GpsrBB(opts, H, g, x, debug_str);
  else
    return GpsrBasic(opts, H, g, x, debug_str);
}

/// This is the basic GPSR algorithm, where the step size is determined by a
/// backtracking line search. The line search is called "Armijo rule along the
/// projection arc" in Bertsekas, Nonlinear Programming, 2nd ed. page 230.
template<typename Real>
Real GpsrBasic(const GpsrConfig &opts, const SpMatrix<Real> &H,
               const Vector<Real> &g, Vector<Real> *x,
               const char *debug_str = "[unknown]");

/// This is the paper calls the Barzilai-Borwein variant. This is a constrained
/// Netwon's method where the Hessian is approximated by scaled identity matrix
template<typename Real>
Real GpsrBB(const GpsrConfig &opts, const SpMatrix<Real> &H,
            const Vector<Real> &g, Vector<Real> *x,
            const char *debug_str = "[unknown]");


}  // namespace kaldi

#endif  // KALDI_MATRIX_KALDI_GPSR_H_
