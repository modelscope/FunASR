// fstext/push-special.cc

// Copyright 2012      Johns Hopkins University (author: Daniel Povey)
//           2012      Ehsan Variani, Pegah Ghahrmani

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

#include "fstext/push-special.h"
#include "base/kaldi-error.h"
#include "base/kaldi-math.h"

namespace fst {


/*

  This algorithm was briefly described in "COMBINING FORWARD AND BACKWARD SEARCH
  IN DECODING" by Hannemann and Povey, ICASSP 2013,
  http://www.danielpovey.com/files/2013_icassp_pingpong.pdf

  Below is the most relevant excerpt of the LaTeX source.

Real backoff language models represented as WFSTs (\cite{Mohri:08}) will not
exactly sum to one because the backoff structure leads to duplicate paths for
some word sequences.  In fact, such language models cannot be pushed at all in
the general case, because the total weight of the entire WFST may not be finite.
For our language model reversal we need a suitable pushing operation that will
always succeed.

Our solution is to require a modified pushing operation such that each state
``sums to'' the same quantity.
We were able to find an iterative algorithm that does this very efficiently in practice;
it is based on the power method for finding the top eigenvalue of a matrix.
Both for the math and the implementation, we find it more convenient to
use the probability semiring, i.e. we represent the transition-probabilities
as actual probabilities, not negative logs.
Let the transitions be written as a sparse matrix $\mathbf{P}$,
where $p_{ij}$ is the sum of all the probabilities of transitions between state $i$ and state $j$.
As a special case, if $j$ is the initial state,
then $p_{ij}$ is the final-probability of state $i$.
In our method we find the dominant eigenvector $\mathbf{v}$ of the matrix $\mathbf{P}$,
by starting from a random positive vector and iterating with the power method:
each time we let $\mathbf{v} \leftarrow \mathbf{P} \mathbf{v}$
and then renormalize the length of $\mathbf{v}$.
It is convenient to renormalize $\mathbf{v}$ so that $v_I$ is 1,
where $I$ is the initial state of the WFST\footnote{Note: in order to correctly
deal with the case of linear WFSTs, which have different eigenvalues
with the same magnitude but different complex phase,
we modify the iteration to $\mathbf{v} \leftarrow \mathbf{P} \mathbf{v} + 0.1 \mathbf{v}$.}.
This generally converges within several tens of iterations.
At the end we have a vector $\mathbf{v}$ with $v_I = 1$, and a scalar $\lambda > 0$, such that
\begin{equation}
  \lambda \mathbf{v} = \mathbf{P} \mathbf{v} .  \label{eqn:lambdav}
\end{equation}
Suppose we compute a modified transition matrix $\mathbf{P}'$, by letting
\begin{equation}
  p'_{ij} = p_{ij} v_j / v_i .
\end{equation}
Then it is easy to show each row of $\mathbf{P}'$ sums to $\lambda$:
writing one element of Eq.~\ref{eqn:lambdav} as
\begin{equation}
 \lambda v_i = \sum_j p_{ij} v_j,
\end{equation}
it easily follows that $\lambda = \sum_j p'_{ij}$.
We need to perform a similar transformation on the transition-probabilities and
final-probabilities of the WFST; the details are quite obvious, and the
equivalence with the original WFST is easy to show.  Our algorithm is in
practice an order of magnitude faster than the more generic algorithm for
conventional weight-pushing of \cite{Mohri:02}, when applied to cyclic WFSTs.

 */

class PushSpecialClass {
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

 public:
  // Everything happens in the initializer.
  PushSpecialClass(VectorFst<StdArc> *fst,
                   float delta): fst_(fst) {
    num_states_ = fst_->NumStates();
    initial_state_ = fst_->Start();
    occ_.resize(num_states_, 1.0 / sqrt(num_states_)); // unit length

    pred_.resize(num_states_);
    for (StateId s = 0; s < num_states_; s++) {
      for (ArcIterator<VectorFst<StdArc> > aiter(*fst, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        StateId t = arc.nextstate;
        double weight = kaldi::Exp(-arc.weight.Value());
        pred_[t].push_back(std::make_pair(s, weight));
      }
      double final = kaldi::Exp(-fst_->Final(s).Value());
      if (final != 0.0)
        pred_[initial_state_].push_back(std::make_pair(s, final));
    }
    Iterate(delta);
    ModifyFst();
  }
 private:
  double TestAccuracy() { // returns the error (the difference
    // between the min and max weights).
    double min_sum = 0, max_sum = 0;
    for (StateId s = 0; s < num_states_; s++) {
      double sum = 0.0;
      for (ArcIterator<VectorFst<StdArc> > aiter(*fst_, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        StateId t = arc.nextstate;
        sum += kaldi::Exp(-arc.weight.Value()) * occ_[t] / occ_[s];
      }
      sum += kaldi::Exp(-(fst_->Final(s).Value())) * occ_[initial_state_] / occ_[s];
      if (s == 0) {
        min_sum = sum;
        max_sum = sum;
      } else {
        min_sum = std::min(min_sum, sum);
        max_sum = std::max(max_sum, sum);
      }
   }
    KALDI_VLOG(4) << "min,max is " << min_sum << " " << max_sum;
    return kaldi::Log(max_sum / min_sum); // In FST world we'll actually
    // dealing with logs, so the log of the ratio is more suitable
    // to compare with delta (makes testing the algorithm easier).
  }


  void Iterate(float delta) {
    // This is like the power method to find the top eigenvalue of a matrix.
    // We limit it to 200 iters max, just in case something unanticipated
    // happens, but we should exit due to the "delta" thing, usually after
    // several tens of iterations.
    int iter, max_iter = 200;

    for (iter = 0; iter < max_iter; iter++) {
      std::vector<double> new_occ(num_states_);
      // We initialize new_occ to 0.1 * occ.  A simpler algorithm would
      // initialize them to zero, so it's like the pure power method.  This is
      // like the power method on (M + 0.1 I), and we do it this way to avoid a
      // problem we encountered with certain very simple linear FSTs where the
      // eigenvalues of the weight matrix (including negative and imaginary
      // ones) all have the same magnitude.
      for (int i = 0; i < num_states_; i++)
        new_occ[i] = 0.1 * occ_[i];

      for (int i = 0; i < num_states_; i++) {
        std::vector<std::pair<StateId, double> >::const_iterator iter,
            end = pred_[i].end();
        for (iter = pred_[i].begin(); iter != end; ++iter) {
          StateId j = iter->first;
          double p = iter->second;
          new_occ[j] += occ_[i] * p;
        }
      }
      double sumsq = 0.0;
      for (int i = 0; i < num_states_; i++) sumsq += new_occ[i] * new_occ[i];
      lambda_ = std::sqrt(sumsq);
      double inv_lambda = 1.0 / lambda_;
      for (int i = 0; i < num_states_; i++) occ_[i] = new_occ[i] * inv_lambda;
      KALDI_VLOG(4) << "Lambda is " << lambda_;
      if (iter % 5 == 0 && iter > 0 && TestAccuracy() <= delta) {
        KALDI_VLOG(3) << "Weight-pushing converged after " << iter
                      << " iterations.";
        return;
      }
    }
    KALDI_WARN << "push-special: finished " << iter
               << " iterations without converging.  Output will be inaccurate.";
  }


  // Modifies the FST weights and the final-prob to take account of these potentials.
  void ModifyFst() {
    // First get the potentials as negative-logs, like the values
    // in the FST.
    for (StateId s = 0; s < num_states_; s++) {
      occ_[s] = -kaldi::Log(occ_[s]);
      if (KALDI_ISNAN(occ_[s]) || KALDI_ISINF(occ_[s]))
        KALDI_WARN << "NaN or inf found: " << occ_[s];
    }
    for (StateId s = 0; s < num_states_; s++) {
      for (MutableArcIterator<VectorFst<StdArc> > aiter(fst_, s);
           !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        StateId t = arc.nextstate;
        arc.weight = Weight(arc.weight.Value() + occ_[t] - occ_[s]);
        aiter.SetValue(arc);
      }
      fst_->SetFinal(s, Times(fst_->Final(s).Value(),
                              Weight(occ_[initial_state_] - occ_[s])));
    }
  }

 private:
  StateId num_states_;
  StateId initial_state_;
  std::vector<double> occ_; // the top eigenvector of (matrix of weights) transposed.
  double lambda_; // our current estimate of the top eigenvalue.

  std::vector<std::vector<std::pair<StateId, double> > > pred_; // List of transitions
  // into each state.  For the start state, this list consists of the list of
  // states with final-probs, each with their final prob.

  VectorFst<StdArc> *fst_;

};




void PushSpecial(VectorFst<StdArc> *fst, float delta) {
  if (fst->NumStates() > 0)
    PushSpecialClass c(fst, delta); // all the work
  // gets done in the initializer.
}


} // end namespace fst.


/*
  Note: in testing an earlier, simpler
  version of this method (without the 0.1 * old_occ) we had a problem with the following FST.
0    2    3    3    0
1    3    1    4    0.5
2    1    0    0    0.5
3    0.25

 Corresponds to the following matrix [or maybe its transpose, doesn't matter
 probably]

 a=exp(-0.5)
 b=exp(-0.25)
M =   [ 0 1 0 0
       0 0 a 0
       0 0 0 a
       b 0 0 0 ]

eigs(M)
eigs(M)

ans =

  -0.0000 - 0.7316i
  -0.0000 + 0.7316i
   0.7316
  -0.7316

   OK, so the issue appears to be that all the eigenvalues of this matrix
   have the same magnitude.  The solution is to work with the eigenvalues
   of M + alpha I, for some small alpha such as 0.1 (so as not to slow down
   convergence in the normal case).

*/
