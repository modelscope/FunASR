// lat/lattice-functions.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli
//                2013  Cisco Systems (author: Neha Agrawal) [code modified
//                      from original code in ../gmmbin/gmm-rescore-lattice.cc]
//                2014  Guoguo Chen

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


#include "base/kaldi-math.h"
#include "lat/lattice-functions.h"

namespace kaldi {
using std::map;
using std::vector;

//void GetPerFrameAcousticCosts(const Lattice &nbest,
//                              Vector<BaseFloat> *per_frame_loglikes) {
//  using namespace fst;
//  typedef Lattice::Arc::Weight Weight;
//  vector<BaseFloat> loglikes;
//
//  int32 cur_state = nbest.Start();
//  int32 prev_frame = -1;
//  BaseFloat eps_acwt = 0.0;
//  while(1) {
//    Weight w = nbest.Final(cur_state);
//    if (w != Weight::Zero()) {
//      KALDI_ASSERT(nbest.NumArcs(cur_state) == 0);
//      if (per_frame_loglikes != NULL)  {
//        SubVector<BaseFloat> subvec(&(loglikes[0]), loglikes.size());
//        Vector<BaseFloat> vec(subvec);
//        *per_frame_loglikes = vec;
//      }
//      break;
//    } else {
//      KALDI_ASSERT(nbest.NumArcs(cur_state) == 1);
//      fst::ArcIterator<Lattice> iter(nbest, cur_state);
//      const Lattice::Arc &arc = iter.Value();
//      BaseFloat acwt = arc.weight.Value2();
//      if (arc.ilabel != 0) {
//        if (eps_acwt > 0) {
//          acwt += eps_acwt;
//          eps_acwt = 0.0;
//        }
//        loglikes.push_back(acwt);
//        prev_frame++;
//      } else if (acwt == acwt){
//        if (prev_frame > -1) {
//          loglikes[prev_frame] += acwt;
//        } else {
//          eps_acwt += acwt;
//        }
//      }
//      cur_state = arc.nextstate;
//    }
//  }
//}
//
//int32 LatticeStateTimes(const Lattice &lat, vector<int32> *times) {
//  if (!lat.Properties(fst::kTopSorted, true))
//    KALDI_ERR << "Input lattice must be topologically sorted.";
//  KALDI_ASSERT(lat.Start() == 0);
//  int32 num_states = lat.NumStates();
//  times->clear();
//  times->resize(num_states, -1);
//  (*times)[0] = 0;
//  for (int32 state = 0; state < num_states; state++) {
//    int32 cur_time = (*times)[state];
//    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
//        aiter.Next()) {
//      const LatticeArc &arc = aiter.Value();
//
//      if (arc.ilabel != 0) {  // Non-epsilon input label on arc
//        // next time instance
//        if ((*times)[arc.nextstate] == -1) {
//          (*times)[arc.nextstate] = cur_time + 1;
//        } else {
//          KALDI_ASSERT((*times)[arc.nextstate] == cur_time + 1);
//        }
//      } else {  // epsilon input label on arc
//        // Same time instance
//        if ((*times)[arc.nextstate] == -1)
//          (*times)[arc.nextstate] = cur_time;
//        else
//          KALDI_ASSERT((*times)[arc.nextstate] == cur_time);
//      }
//    }
//  }
//  return (*std::max_element(times->begin(), times->end()));
//}
//
//int32 CompactLatticeStateTimes(const CompactLattice &lat,
//                               vector<int32> *times) {
//  if (!lat.Properties(fst::kTopSorted, true))
//    KALDI_ERR << "Input lattice must be topologically sorted.";
//  KALDI_ASSERT(lat.Start() == 0);
//  int32 num_states = lat.NumStates();
//  times->clear();
//  times->resize(num_states, -1);
//  (*times)[0] = 0;
//  int32 utt_len = -1;
//  for (int32 state = 0; state < num_states; state++) {
//    int32 cur_time = (*times)[state];
//    for (fst::ArcIterator<CompactLattice> aiter(lat, state); !aiter.Done();
//        aiter.Next()) {
//      const CompactLatticeArc &arc = aiter.Value();
//      int32 arc_len = static_cast<int32>(arc.weight.String().size());
//      if ((*times)[arc.nextstate] == -1)
//        (*times)[arc.nextstate] = cur_time + arc_len;
//      else
//        KALDI_ASSERT((*times)[arc.nextstate] == cur_time + arc_len);
//    }
//    if (lat.Final(state) != CompactLatticeWeight::Zero()) {
//      int32 this_utt_len = (*times)[state] + lat.Final(state).String().size();
//      if (utt_len == -1) utt_len = this_utt_len;
//      else {
//        if (this_utt_len != utt_len) {
//          KALDI_WARN << "Utterance does not "
//              "seem to have a consistent length.";
//          utt_len = std::max(utt_len, this_utt_len);
//        }
//      }
//    }
//  }
//  if (utt_len == -1) {
//    KALDI_WARN << "Utterance does not have a final-state.";
//    return 0;
//  }
//  return utt_len;
//}
//
//bool ComputeCompactLatticeAlphas(const CompactLattice &clat,
//                                 vector<double> *alpha) {
//  using namespace fst;
//
//  // typedef the arc, weight types
//  typedef CompactLattice::Arc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//
//  //Make sure the lattice is topologically sorted.
//  if (clat.Properties(fst::kTopSorted, true) == 0) {
//    KALDI_WARN << "Input lattice must be topologically sorted.";
//    return false;
//  }
//  if (clat.Start() != 0) {
//    KALDI_WARN << "Input lattice must start from state 0.";
//    return false;
//  }
//
//  int32 num_states = clat.NumStates();
//  (*alpha).resize(0);
//  (*alpha).resize(num_states, kLogZeroDouble);
//
//  // Now propagate alphas forward. Note that we don't acount the weight of the
//  // final state to alpha[final_state] -- we acount it to beta[final_state];
//  (*alpha)[0] = 0.0;
//  for (StateId s = 0; s < num_states; s++) {
//    double this_alpha = (*alpha)[s];
//    for (ArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -(arc.weight.Weight().Value1() +
//                          arc.weight.Weight().Value2());
//      (*alpha)[arc.nextstate] = LogAdd((*alpha)[arc.nextstate],
//                                       this_alpha + arc_like);
//    }
//  }
//
//  return true;
//}
//
//bool ComputeCompactLatticeBetas(const CompactLattice &clat,
//                                vector<double> *beta) {
//  using namespace fst;
//
//  // typedef the arc, weight types
//  typedef CompactLattice::Arc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//
//  // Make sure the lattice is topologically sorted.
//  if (clat.Properties(fst::kTopSorted, true) == 0) {
//    KALDI_WARN << "Input lattice must be topologically sorted.";
//    return false;
//  }
//  if (clat.Start() != 0) {
//    KALDI_WARN << "Input lattice must start from state 0.";
//    return false;
//  }
//
//  int32 num_states = clat.NumStates();
//  (*beta).resize(0);
//  (*beta).resize(num_states, kLogZeroDouble);
//
//  // Now propagate betas backward. Note that beta[final_state] contains the
//  // weight of the final state in the lattice -- compare that with alpha.
//  for (StateId s = num_states-1; s >= 0; s--) {
//    Weight f = clat.Final(s);
//    double this_beta = -(f.Weight().Value1()+f.Weight().Value2());
//    for (ArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -(arc.weight.Weight().Value1() +
//                          arc.weight.Weight().Value2());
//      double arc_beta = (*beta)[arc.nextstate] + arc_like;
//      this_beta = LogAdd(this_beta, arc_beta);
//    }
//    (*beta)[s] = this_beta;
//  }
//
//  return true;
//}

template<class LatType>  // could be Lattice or CompactLattice
bool PruneLattice(BaseFloat beam, LatType *lat) {
  typedef typename LatType::Arc Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;

  KALDI_ASSERT(beam > 0.0);
  if (!lat->Properties(fst::kTopSorted, true)) {
    if (fst::TopSort(lat) == false) {
      KALDI_WARN << "Cycles detected in lattice";
      return false;
    }
  }
  // We assume states before "start" are not reachable, since
  // the lattice is topologically sorted.
  int32 start = lat->Start();
  int32 num_states = lat->NumStates();
  if (num_states == 0) return false;
  std::vector<double> forward_cost(num_states,
                                   std::numeric_limits<double>::infinity());  // viterbi forward.
  forward_cost[start] = 0.0; // lattice can't have cycles so couldn't be
  // less than this.
  double best_final_cost = std::numeric_limits<double>::infinity();
  // Update the forward probs.
  // Thanks to Jing Zheng for finding a bug here.
  for (int32 state = 0; state < num_states; state++) {
    double this_forward_cost = forward_cost[state];
    for (fst::ArcIterator<LatType> aiter(*lat, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc(aiter.Value());
      StateId nextstate = arc.nextstate;
      KALDI_ASSERT(nextstate > state && nextstate < num_states);
      double next_forward_cost = this_forward_cost +
          ConvertToCost(arc.weight);
      if (forward_cost[nextstate] > next_forward_cost)
        forward_cost[nextstate] = next_forward_cost;
    }
    Weight final_weight = lat->Final(state);
    double this_final_cost = this_forward_cost +
        ConvertToCost(final_weight);
    if (this_final_cost < best_final_cost)
      best_final_cost = this_final_cost;
  }
  int32 bad_state = lat->AddState(); // this state is not final.
  double cutoff = best_final_cost + beam;

  // Go backwards updating the backward probs (which share memory with the
  // forward probs), and pruning arcs and deleting final-probs.  We prune arcs
  // by making them point to the non-final state "bad_state".  We'll then use
  // Trim() to remove unnecessary arcs and states.  [this is just easier than
  // doing it ourselves.]
  std::vector<double> &backward_cost(forward_cost);
  for (int32 state = num_states - 1; state >= 0; state--) {
    double this_forward_cost = forward_cost[state];
    double this_backward_cost = ConvertToCost(lat->Final(state));
    if (this_backward_cost + this_forward_cost > cutoff
        && this_backward_cost != std::numeric_limits<double>::infinity())
      lat->SetFinal(state, Weight::Zero());
    for (fst::MutableArcIterator<LatType> aiter(lat, state);
         !aiter.Done();
         aiter.Next()) {
      Arc arc(aiter.Value());
      StateId nextstate = arc.nextstate;
      KALDI_ASSERT(nextstate > state && nextstate < num_states);
      double arc_cost = ConvertToCost(arc.weight),
          arc_backward_cost = arc_cost + backward_cost[nextstate],
          this_fb_cost = this_forward_cost + arc_backward_cost;
      if (arc_backward_cost < this_backward_cost)
        this_backward_cost = arc_backward_cost;
      if (this_fb_cost > cutoff) { // Prune the arc.
        arc.nextstate = bad_state;
        aiter.SetValue(arc);
      }
    }
    backward_cost[state] = this_backward_cost;
  }
  fst::Connect(lat);
  return (lat->NumStates() > 0);
}

// instantiate the template for lattice and CompactLattice.
template bool PruneLattice(BaseFloat beam, Lattice *lat);
template bool PruneLattice(BaseFloat beam, CompactLattice *lat);


//BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *post,
//                                 double *acoustic_like_sum) {
//  // Note, Posterior is defined as follows:  Indexed [frame], then a list
//  // of (transition-id, posterior-probability) pairs.
//  // typedef std::vector<std::vector<std::pair<int32, BaseFloat> > > Posterior;
//  using namespace fst;
//  typedef Lattice::Arc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//
//  if (acoustic_like_sum) *acoustic_like_sum = 0.0;
//
//  // Make sure the lattice is topologically sorted.
//  if (lat.Properties(fst::kTopSorted, true) == 0)
//    KALDI_ERR << "Input lattice must be topologically sorted.";
//  KALDI_ASSERT(lat.Start() == 0);
//
//  int32 num_states = lat.NumStates();
//  vector<int32> state_times;
//  int32 max_time = LatticeStateTimes(lat, &state_times);
//  std::vector<double> alpha(num_states, kLogZeroDouble);
//  std::vector<double> &beta(alpha); // we re-use the same memory for
//  // this, but it's semantically distinct so we name it differently.
//  double tot_forward_prob = kLogZeroDouble;
//
//  post->clear();
//  post->resize(max_time);
//
//  alpha[0] = 0.0;
//  // Propagate alphas forward.
//  for (StateId s = 0; s < num_states; s++) {
//    double this_alpha = alpha[s];
//    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight);
//      alpha[arc.nextstate] = LogAdd(alpha[arc.nextstate], this_alpha + arc_like);
//    }
//    Weight f = lat.Final(s);
//    if (f != Weight::Zero()) {
//      double final_like = this_alpha - (f.Value1() + f.Value2());
//      tot_forward_prob = LogAdd(tot_forward_prob, final_like);
//      KALDI_ASSERT(state_times[s] == max_time &&
//                   "Lattice is inconsistent (final-prob not at max_time)");
//    }
//  }
//  for (StateId s = num_states-1; s >= 0; s--) {
//    Weight f = lat.Final(s);
//    double this_beta = -(f.Value1() + f.Value2());
//    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight),
//          arc_beta = beta[arc.nextstate] + arc_like;
//      this_beta = LogAdd(this_beta, arc_beta);
//      int32 transition_id = arc.ilabel;
//
//      // The following "if" is an optimization to avoid un-needed exp().
//      if (transition_id != 0 || acoustic_like_sum != NULL) {
//        double posterior = Exp(alpha[s] + arc_beta - tot_forward_prob);
//
//        if (transition_id != 0) // Arc has a transition-id on it [not epsilon]
//          (*post)[state_times[s]].push_back(std::make_pair(transition_id,
//                                                           static_cast<kaldi::BaseFloat>(posterior)));
//        if (acoustic_like_sum != NULL)
//          *acoustic_like_sum -= posterior * arc.weight.Value2();
//      }
//    }
//    if (acoustic_like_sum != NULL && f != Weight::Zero()) {
//      double final_logprob = - ConvertToCost(f),
//          posterior = Exp(alpha[s] + final_logprob - tot_forward_prob);
//      *acoustic_like_sum -= posterior * f.Value2();
//    }
//    beta[s] = this_beta;
//  }
//  double tot_backward_prob = beta[0];
//  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8)) {
//    KALDI_WARN << "Total forward probability over lattice = " << tot_forward_prob
//              << ", while total backward probability = " << tot_backward_prob;
//  }
//  // Now combine any posteriors with the same transition-id.
//  for (int32 t = 0; t < max_time; t++)
//    MergePairVectorSumming(&((*post)[t]));
//  return tot_backward_prob;
//}
//
//
//void LatticeActivePhones(const Lattice &lat, const TransitionInformation &trans,
//                         const vector<int32> &silence_phones,
//                         vector< std::set<int32> > *active_phones) {
//  KALDI_ASSERT(IsSortedAndUniq(silence_phones));
//  vector<int32> state_times;
//  int32 num_states = lat.NumStates();
//  int32 max_time = LatticeStateTimes(lat, &state_times);
//  active_phones->clear();
//  active_phones->resize(max_time);
//  for (int32 state = 0; state < num_states; state++) {
//    int32 cur_time = state_times[state];
//    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
//        aiter.Next()) {
//      const LatticeArc &arc = aiter.Value();
//      if (arc.ilabel != 0) {  // Non-epsilon arc
//        int32 phone = trans.TransitionIdToPhone(arc.ilabel);
//        if (!std::binary_search(silence_phones.begin(),
//                                silence_phones.end(), phone))
//          (*active_phones)[cur_time].insert(phone);
//      }
//    }  // end looping over arcs
//  }  // end looping over states
//}
//
//void ConvertLatticeToPhones(const TransitionInformation &trans,
//                            Lattice *lat) {
//  typedef LatticeArc Arc;
//  int32 num_states = lat->NumStates();
//  for (int32 state = 0; state < num_states; state++) {
//    for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
//        aiter.Next()) {
//      Arc arc(aiter.Value());
//      arc.olabel = 0; // remove any word.
//      if ((arc.ilabel != 0) // has a transition-id on input..
//          && (trans.TransitionIdIsStartOfPhone(arc.ilabel))
//          && (!trans.IsSelfLoop(arc.ilabel))) {
//         // && trans.IsFinal(arc.ilabel)) // there is one of these per phone...
//        arc.olabel = trans.TransitionIdToPhone(arc.ilabel);
//      }
//      aiter.SetValue(arc);
//    }  // end looping over arcs
//  }  // end looping over states
//}
//
//
//static inline double LogAddOrMax(bool viterbi, double a, double b) {
//  if (viterbi)
//    return std::max(a, b);
//  else
//    return LogAdd(a, b);
//}
//
//template<typename LatticeType>
//double ComputeLatticeAlphasAndBetas(const LatticeType &lat,
//                                    bool viterbi,
//                                    vector<double> *alpha,
//                                    vector<double> *beta) {
//  typedef typename LatticeType::Arc Arc;
//  typedef typename Arc::Weight Weight;
//  typedef typename Arc::StateId StateId;
//
//  StateId num_states = lat.NumStates();
//  KALDI_ASSERT(lat.Properties(fst::kTopSorted, true) == fst::kTopSorted);
//  KALDI_ASSERT(lat.Start() == 0);
//  alpha->clear();
//  beta->clear();
//  alpha->resize(num_states, kLogZeroDouble);
//  beta->resize(num_states, kLogZeroDouble);
//
//  double tot_forward_prob = kLogZeroDouble;
//  (*alpha)[0] = 0.0;
//  // Propagate alphas forward.
//  for (StateId s = 0; s < num_states; s++) {
//    double this_alpha = (*alpha)[s];
//    for (fst::ArcIterator<LatticeType> aiter(lat, s); !aiter.Done();
//         aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight);
//      (*alpha)[arc.nextstate] = LogAddOrMax(viterbi, (*alpha)[arc.nextstate],
//                                                this_alpha + arc_like);
//    }
//    Weight f = lat.Final(s);
//    if (f != Weight::Zero()) {
//      double final_like = this_alpha - ConvertToCost(f);
//      tot_forward_prob = LogAddOrMax(viterbi, tot_forward_prob, final_like);
//    }
//  }
//  for (StateId s = num_states-1; s >= 0; s--) { // it's guaranteed signed.
//    double this_beta = -ConvertToCost(lat.Final(s));
//    for (fst::ArcIterator<LatticeType> aiter(lat, s); !aiter.Done();
//         aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight),
//          arc_beta = (*beta)[arc.nextstate] + arc_like;
//      this_beta = LogAddOrMax(viterbi, this_beta, arc_beta);
//    }
//    (*beta)[s] = this_beta;
//  }
//  double tot_backward_prob = (*beta)[lat.Start()];
//  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8)) {
//    KALDI_WARN << "Total forward probability over lattice = " << tot_forward_prob
//               << ", while total backward probability = " << tot_backward_prob;
//  }
//  // Split the difference when returning... they should be the same.
//  return 0.5 * (tot_backward_prob + tot_forward_prob);
//}
//
//// instantiate the template for Lattice and CompactLattice
//template
//double ComputeLatticeAlphasAndBetas(const Lattice &lat,
//                                    bool viterbi,
//                                    vector<double> *alpha,
//                                    vector<double> *beta);
//
//template
//double ComputeLatticeAlphasAndBetas(const CompactLattice &lat,
//                                    bool viterbi,
//                                    vector<double> *alpha,
//                                    vector<double> *beta);
//
//
//
///// This is used in CompactLatticeLimitDepth.
//struct LatticeArcRecord {
//  BaseFloat logprob; // logprob <= 0 is the best Viterbi logprob of this arc,
//                     // minus the overall best-cost of the lattice.
//  CompactLatticeArc::StateId state; // state in the lattice.
//  size_t arc; // arc index within the state.
//  bool operator < (const LatticeArcRecord &other) const {
//    return logprob < other.logprob;
//  }
//};
//
//void CompactLatticeLimitDepth(int32 max_depth_per_frame,
//                              CompactLattice *clat) {
//  typedef CompactLatticeArc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//
//  if (clat->Start() == fst::kNoStateId) {
//    KALDI_WARN << "Limiting depth of empty lattice.";
//    return;
//  }
//  if (clat->Properties(fst::kTopSorted, true) == 0) {
//    if (!TopSort(clat))
//      KALDI_ERR << "Topological sorting of lattice failed.";
//  }
//
//  vector<int32> state_times;
//  int32 T = CompactLatticeStateTimes(*clat, &state_times);
//
//  // The alpha and beta quantities here are "viterbi" alphas and beta.
//  std::vector<double> alpha;
//  std::vector<double> beta;
//  bool viterbi = true;
//  double best_prob = ComputeLatticeAlphasAndBetas(*clat, viterbi,
//                                                  &alpha, &beta);
//
//  std::vector<std::vector<LatticeArcRecord> > arc_records(T);
//
//  StateId num_states = clat->NumStates();
//  for (StateId s = 0; s < num_states; s++) {
//    for (fst::ArcIterator<CompactLattice> aiter(*clat, s); !aiter.Done();
//         aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      LatticeArcRecord arc_record;
//      arc_record.state = s;
//      arc_record.arc = aiter.Position();
//      arc_record.logprob =
//          (alpha[s] + beta[arc.nextstate] - ConvertToCost(arc.weight))
//           - best_prob;
//      KALDI_ASSERT(arc_record.logprob < 0.1); // Should be zero or negative.
//      int32 num_frames = arc.weight.String().size(), start_t = state_times[s];
//      for (int32 t = start_t; t < start_t + num_frames; t++) {
//        KALDI_ASSERT(t < T);
//        arc_records[t].push_back(arc_record);
//      }
//    }
//  }
//  StateId dead_state = clat->AddState(); // A non-coaccesible state which we use
//                                         // to remove arcs (make them end
//                                         // there).
//  size_t max_depth = max_depth_per_frame;
//  for (int32 t = 0; t < T; t++) {
//    size_t size = arc_records[t].size();
//    if (size > max_depth) {
//      // we sort from worst to best, so we keep the later-numbered ones,
//      // and delete the lower-numbered ones.
//      size_t cutoff = size - max_depth;
//      std::nth_element(arc_records[t].begin(),
//                       arc_records[t].begin() + cutoff,
//                       arc_records[t].end());
//      for (size_t index = 0; index < cutoff; index++) {
//        LatticeArcRecord record(arc_records[t][index]);
//        fst::MutableArcIterator<CompactLattice> aiter(clat, record.state);
//        aiter.Seek(record.arc);
//        Arc arc = aiter.Value();
//        if (arc.nextstate != dead_state) { // not already killed.
//          arc.nextstate = dead_state;
//          aiter.SetValue(arc);
//        }
//      }
//    }
//  }
//  Connect(clat);
//  TopSortCompactLatticeIfNeeded(clat);
//}
//
//
//void TopSortCompactLatticeIfNeeded(CompactLattice *clat) {
//  if (clat->Properties(fst::kTopSorted, true) == 0) {
//    if (fst::TopSort(clat) == false) {
//      KALDI_ERR << "Topological sorting failed";
//    }
//  }
//}
//
//void TopSortLatticeIfNeeded(Lattice *lat) {
//  if (lat->Properties(fst::kTopSorted, true) == 0) {
//    if (fst::TopSort(lat) == false) {
//      KALDI_ERR << "Topological sorting failed";
//    }
//  }
//}
//
//
///// Returns the depth of the lattice, defined as the average number of
///// arcs crossing any given frame.  Returns 1 for empty lattices.
///// Requires that input is topologically sorted.
//BaseFloat CompactLatticeDepth(const CompactLattice &clat,
//                              int32 *num_frames) {
//  typedef CompactLattice::Arc::StateId StateId;
//  if (clat.Properties(fst::kTopSorted, true) == 0) {
//    KALDI_ERR << "Lattice input to CompactLatticeDepth was not topologically "
//              << "sorted.";
//  }
//  if (clat.Start() == fst::kNoStateId) {
//    *num_frames = 0;
//    return 1.0;
//  }
//  size_t num_arc_frames = 0;
//  int32 t;
//  {
//    vector<int32> state_times;
//    t = CompactLatticeStateTimes(clat, &state_times);
//  }
//  if (num_frames != NULL)
//    *num_frames = t;
//  for (StateId s = 0; s < clat.NumStates(); s++) {
//    for (fst::ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done();
//         aiter.Next()) {
//      const CompactLatticeArc &arc = aiter.Value();
//      num_arc_frames += arc.weight.String().size();
//    }
//    num_arc_frames += clat.Final(s).String().size();
//  }
//  return num_arc_frames / static_cast<BaseFloat>(t);
//}
//
//
//void CompactLatticeDepthPerFrame(const CompactLattice &clat,
//                                 std::vector<int32> *depth_per_frame) {
//  typedef CompactLattice::Arc::StateId StateId;
//  if (clat.Properties(fst::kTopSorted, true) == 0) {
//    KALDI_ERR << "Lattice input to CompactLatticeDepthPerFrame was not "
//              << "topologically sorted.";
//  }
//  if (clat.Start() == fst::kNoStateId) {
//    depth_per_frame->clear();
//    return;
//  }
//  vector<int32> state_times;
//  int32 T = CompactLatticeStateTimes(clat, &state_times);
//
//  depth_per_frame->clear();
//  if (T <= 0) {
//    return;
//  } else {
//    depth_per_frame->resize(T, 0);
//    for (StateId s = 0; s < clat.NumStates(); s++) {
//      int32 start_time = state_times[s];
//      for (fst::ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done();
//           aiter.Next()) {
//        const CompactLatticeArc &arc = aiter.Value();
//        int32 len = arc.weight.String().size();
//        for (int32 t = start_time; t < start_time + len; t++) {
//          KALDI_ASSERT(t < T);
//          (*depth_per_frame)[t]++;
//        }
//      }
//      int32 final_len = clat.Final(s).String().size();
//      for (int32 t = start_time; t < start_time + final_len; t++) {
//        KALDI_ASSERT(t < T);
//        (*depth_per_frame)[t]++;
//      }
//    }
//  }
//}
//
//
//
//void ConvertCompactLatticeToPhones(const TransitionInformation &trans,
//                                   CompactLattice *clat) {
//  typedef CompactLatticeArc Arc;
//  typedef Arc::Weight Weight;
//  int32 num_states = clat->NumStates();
//  for (int32 state = 0; state < num_states; state++) {
//    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
//         !aiter.Done();
//         aiter.Next()) {
//      Arc arc(aiter.Value());
//      std::vector<int32> phone_seq;
//      const std::vector<int32> &tid_seq = arc.weight.String();
//      for (std::vector<int32>::const_iterator iter = tid_seq.begin();
//           iter != tid_seq.end(); ++iter) {
//        if (trans.IsFinal(*iter))// note: there is one of these per phone...
//          phone_seq.push_back(trans.TransitionIdToPhone(*iter));
//      }
//      arc.weight.SetString(phone_seq);
//      aiter.SetValue(arc);
//    } // end looping over arcs
//    Weight f = clat->Final(state);
//    if (f != Weight::Zero()) {
//      std::vector<int32> phone_seq;
//      const std::vector<int32> &tid_seq = f.String();
//      for (std::vector<int32>::const_iterator iter = tid_seq.begin();
//           iter != tid_seq.end(); ++iter) {
//        if (trans.IsFinal(*iter))// note: there is one of these per phone...
//          phone_seq.push_back(trans.TransitionIdToPhone(*iter));
//      }
//      f.SetString(phone_seq);
//      clat->SetFinal(state, f);
//    }
//  }  // end looping over states
//}
//
//bool LatticeBoost(const TransitionInformation &trans,
//                  const std::vector<int32> &alignment,
//                  const std::vector<int32> &silence_phones,
//                  BaseFloat b,
//                  BaseFloat max_silence_error,
//                  Lattice *lat) {
//  TopSortLatticeIfNeeded(lat);
//
//  // get all stored properties (test==false means don't test if not known).
//  uint64 props = lat->Properties(fst::kFstProperties,
//                                 false);
//
//  KALDI_ASSERT(IsSortedAndUniq(silence_phones));
//  KALDI_ASSERT(max_silence_error >= 0.0 && max_silence_error <= 1.0);
//  vector<int32> state_times;
//  int32 num_states = lat->NumStates();
//  int32 num_frames = LatticeStateTimes(*lat, &state_times);
//  KALDI_ASSERT(num_frames == static_cast<int32>(alignment.size()));
//  for (int32 state = 0; state < num_states; state++) {
//    int32 cur_time = state_times[state];
//    for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
//         aiter.Next()) {
//      LatticeArc arc = aiter.Value();
//      if (arc.ilabel != 0) {  // Non-epsilon arc
//        if (arc.ilabel < 0 || arc.ilabel > trans.NumTransitionIds()) {
//          KALDI_WARN << "Lattice has out-of-range transition-ids: "
//                     << "lattice/model mismatch?";
//          return false;
//        }
//        int32 phone = trans.TransitionIdToPhone(arc.ilabel),
//            ref_phone = trans.TransitionIdToPhone(alignment[cur_time]);
//        BaseFloat frame_error;
//        if (phone == ref_phone) {
//          frame_error = 0.0;
//        } else { // an error...
//          if (std::binary_search(silence_phones.begin(), silence_phones.end(), phone))
//            frame_error = max_silence_error;
//          else
//            frame_error = 1.0;
//        }
//        BaseFloat delta_cost = -b * frame_error; // negative cost if
//        // frame is wrong, to boost likelihood of arcs with errors on them.
//        // Add this cost to the graph part.
//        arc.weight.SetValue1(arc.weight.Value1() + delta_cost);
//        aiter.SetValue(arc);
//      }
//    }
//  }
//  // All we changed is the weights, so any properties that were
//  // known before, are still known, except for whether or not the
//  // lattice was weighted.
//  lat->SetProperties(props,
//                     ~(fst::kWeighted|fst::kUnweighted));
//
//  return true;
//}
//
//
//
//BaseFloat LatticeForwardBackwardMpeVariants(
//    const TransitionInformation &trans,
//    const std::vector<int32> &silence_phones,
//    const Lattice &lat,
//    const std::vector<int32> &num_ali,
//    std::string criterion,
//    bool one_silence_class,
//    Posterior *post) {
//  using namespace fst;
//  typedef Lattice::Arc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//
//  KALDI_ASSERT(criterion == "mpfe" || criterion == "smbr");
//  bool is_mpfe = (criterion == "mpfe");
//
//  if (lat.Properties(fst::kTopSorted, true) == 0)
//    KALDI_ERR << "Input lattice must be topologically sorted.";
//  KALDI_ASSERT(lat.Start() == 0);
//
//  int32 num_states = lat.NumStates();
//  vector<int32> state_times;
//  int32 max_time = LatticeStateTimes(lat, &state_times);
//  KALDI_ASSERT(max_time == static_cast<int32>(num_ali.size()));
//  std::vector<double> alpha(num_states, kLogZeroDouble),
//      alpha_smbr(num_states, 0), //forward variable for sMBR
//      beta(num_states, kLogZeroDouble),
//      beta_smbr(num_states, 0); //backward variable for sMBR
//
//  double tot_forward_prob = kLogZeroDouble;
//  double tot_forward_score = 0;
//
//  post->clear();
//  post->resize(max_time);
//
//  alpha[0] = 0.0;
//  // First Pass Forward,
//  for (StateId s = 0; s < num_states; s++) {
//    double this_alpha = alpha[s];
//    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight);
//      alpha[arc.nextstate] = LogAdd(alpha[arc.nextstate], this_alpha + arc_like);
//    }
//    Weight f = lat.Final(s);
//    if (f != Weight::Zero()) {
//      double final_like = this_alpha - (f.Value1() + f.Value2());
//      tot_forward_prob = LogAdd(tot_forward_prob, final_like);
//      KALDI_ASSERT(state_times[s] == max_time &&
//                   "Lattice is inconsistent (final-prob not at max_time)");
//    }
//  }
//  // First Pass Backward,
//  for (StateId s = num_states-1; s >= 0; s--) {
//    Weight f = lat.Final(s);
//    double this_beta = -(f.Value1() + f.Value2());
//    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight),
//          arc_beta = beta[arc.nextstate] + arc_like;
//      this_beta = LogAdd(this_beta, arc_beta);
//    }
//    beta[s] = this_beta;
//  }
//  // First Pass Forward-Backward Check
//  double tot_backward_prob = beta[0];
//  // may loose the condition somehow here 1e-6 (was 1e-8)
//  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-6)) {
//    KALDI_ERR << "Total forward probability over lattice = " << tot_forward_prob
//              << ", while total backward probability = " << tot_backward_prob;
//  }
//
//  alpha_smbr[0] = 0.0;
//  // Second Pass Forward, calculate forward for MPFE/SMBR
//  for (StateId s = 0; s < num_states; s++) {
//    double this_alpha = alpha[s];
//    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight);
//      double frame_acc = 0.0;
//      if (arc.ilabel != 0) {
//        int32 cur_time = state_times[s];
//        int32 phone = trans.TransitionIdToPhone(arc.ilabel),
//            ref_phone = trans.TransitionIdToPhone(num_ali[cur_time]);
//        bool phone_is_sil = std::binary_search(silence_phones.begin(),
//                                               silence_phones.end(),
//                                               phone),
//            ref_phone_is_sil = std::binary_search(silence_phones.begin(),
//                                                  silence_phones.end(),
//                                                  ref_phone),
//            both_sil = phone_is_sil && ref_phone_is_sil;
//        if (!is_mpfe) { // smbr.
//          int32 pdf = trans.TransitionIdToPdf(arc.ilabel),
//              ref_pdf = trans.TransitionIdToPdf(num_ali[cur_time]);
//          if (!one_silence_class)  // old behavior
//            frame_acc = (pdf == ref_pdf && !phone_is_sil) ? 1.0 : 0.0;
//          else
//            frame_acc = (pdf == ref_pdf || both_sil) ? 1.0 : 0.0;
//        } else {
//          if (!one_silence_class)  // old behavior
//            frame_acc = (phone == ref_phone && !phone_is_sil) ? 1.0 : 0.0;
//          else
//            frame_acc = (phone == ref_phone || both_sil) ? 1.0 : 0.0;
//        }
//      }
//      double arc_scale = Exp(alpha[s] + arc_like - alpha[arc.nextstate]);
//      alpha_smbr[arc.nextstate] += arc_scale * (alpha_smbr[s] + frame_acc);
//    }
//    Weight f = lat.Final(s);
//    if (f != Weight::Zero()) {
//      double final_like = this_alpha - (f.Value1() + f.Value2());
//      double arc_scale = Exp(final_like - tot_forward_prob);
//      tot_forward_score += arc_scale * alpha_smbr[s];
//      KALDI_ASSERT(state_times[s] == max_time &&
//                   "Lattice is inconsistent (final-prob not at max_time)");
//    }
//  }
//  // Second Pass Backward, collect Mpe style posteriors
//  for (StateId s = num_states-1; s >= 0; s--) {
//    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_like = -ConvertToCost(arc.weight),
//          arc_beta = beta[arc.nextstate] + arc_like;
//      double frame_acc = 0.0;
//      int32 transition_id = arc.ilabel;
//      if (arc.ilabel != 0) {
//        int32 cur_time = state_times[s];
//        int32 phone = trans.TransitionIdToPhone(arc.ilabel),
//            ref_phone = trans.TransitionIdToPhone(num_ali[cur_time]);
//        bool phone_is_sil = std::binary_search(silence_phones.begin(),
//                                               silence_phones.end(), phone),
//            ref_phone_is_sil = std::binary_search(silence_phones.begin(),
//                                                  silence_phones.end(),
//                                                  ref_phone),
//            both_sil = phone_is_sil && ref_phone_is_sil;
//        if (!is_mpfe) { // smbr.
//          int32 pdf = trans.TransitionIdToPdf(arc.ilabel),
//              ref_pdf = trans.TransitionIdToPdf(num_ali[cur_time]);
//          if (!one_silence_class)  // old behavior
//            frame_acc = (pdf == ref_pdf && !phone_is_sil) ? 1.0 : 0.0;
//          else
//            frame_acc = (pdf == ref_pdf || both_sil) ? 1.0 : 0.0;
//        } else {
//          if (!one_silence_class)  // old behavior
//            frame_acc = (phone == ref_phone && !phone_is_sil) ? 1.0 : 0.0;
//          else
//            frame_acc = (phone == ref_phone || both_sil) ? 1.0 : 0.0;
//        }
//      }
//      double arc_scale = Exp(beta[arc.nextstate] + arc_like - beta[s]);
//      // check arc_scale NAN,
//      // this is to prevent partial paths in Lattices
//      // i.e., paths don't survive to the final state
//      if (KALDI_ISNAN(arc_scale)) arc_scale = 0;
//      beta_smbr[s] += arc_scale * (beta_smbr[arc.nextstate] + frame_acc);
//
//      if (transition_id != 0) { // Arc has a transition-id on it [not epsilon]
//        double posterior = Exp(alpha[s] + arc_beta - tot_forward_prob);
//        double acc_diff = alpha_smbr[s] + frame_acc + beta_smbr[arc.nextstate]
//                               - tot_forward_score;
//        double posterior_smbr = posterior * acc_diff;
//        (*post)[state_times[s]].push_back(std::make_pair(transition_id,
//                                                         static_cast<BaseFloat>(posterior_smbr)));
//      }
//    }
//  }
//
//  //Second Pass Forward Backward check
//  double tot_backward_score = beta_smbr[0];  // Initial state id == 0
//  // may loose the condition somehow here 1e-5/1e-4
//  if (!ApproxEqual(tot_forward_score, tot_backward_score, 1e-4)) {
//    KALDI_ERR << "Total forward score over lattice = " << tot_forward_score
//              << ", while total backward score = " << tot_backward_score;
//  }
//
//  // Output the computed posteriors
//  for (int32 t = 0; t < max_time; t++)
//    MergePairVectorSumming(&((*post)[t]));
//  return tot_forward_score;
//}
//
//bool CompactLatticeToWordAlignment(const CompactLattice &clat,
//                                   std::vector<int32> *words,
//                                   std::vector<int32> *begin_times,
//                                   std::vector<int32> *lengths) {
//  words->clear();
//  begin_times->clear();
//  lengths->clear();
//  typedef CompactLattice::Arc Arc;
//  typedef Arc::Label Label;
//  typedef CompactLattice::StateId StateId;
//  typedef CompactLattice::Weight Weight;
//  using namespace fst;
//  StateId state = clat.Start();
//  int32 cur_time = 0;
//  if (state == kNoStateId) {
//    KALDI_WARN << "Empty lattice.";
//    return false;
//  }
//  while (1) {
//    Weight final = clat.Final(state);
//    size_t num_arcs = clat.NumArcs(state);
//    if (final != Weight::Zero()) {
//      if (num_arcs != 0) {
//        KALDI_WARN << "Lattice is not linear.";
//        return false;
//      }
//      if (! final.String().empty()) {
//        KALDI_WARN << "Lattice has alignments on final-weight: probably "
//            "was not word-aligned (alignments will be approximate)";
//      }
//      return true;
//    } else {
//      if (num_arcs != 1) {
//        KALDI_WARN << "Lattice is not linear: num-arcs = " << num_arcs;
//        return false;
//      }
//      fst::ArcIterator<CompactLattice> aiter(clat, state);
//      const Arc &arc = aiter.Value();
//      Label word_id = arc.ilabel; // Note: ilabel==olabel, since acceptor.
//      // Also note: word_id may be zero; we output it anyway.
//      int32 length = arc.weight.String().size();
//      words->push_back(word_id);
//      begin_times->push_back(cur_time);
//      lengths->push_back(length);
//      cur_time += length;
//      state = arc.nextstate;
//    }
//  }
//}
//
//
//void CompactLatticeShortestPath(const CompactLattice &clat,
//                                CompactLattice *shortest_path) {
//  using namespace fst;
//  if (clat.Properties(fst::kTopSorted, true) == 0) {
//    CompactLattice clat_copy(clat);
//    if (!TopSort(&clat_copy))
//      KALDI_ERR << "Was not able to topologically sort lattice (cycles found?)";
//    CompactLatticeShortestPath(clat_copy, shortest_path);
//    return;
//  }
//  // Now we can assume it's topologically sorted.
//  shortest_path->DeleteStates();
//  if (clat.Start() == kNoStateId) return;
//  typedef CompactLatticeArc Arc;
//  typedef Arc::StateId StateId;
//  typedef CompactLatticeWeight Weight;
//  vector<std::pair<double, StateId> > best_cost_and_pred(clat.NumStates() + 1);
//  StateId superfinal = clat.NumStates();
//  for (StateId s = 0; s <= clat.NumStates(); s++) {
//    best_cost_and_pred[s].first = std::numeric_limits<double>::infinity();
//    best_cost_and_pred[s].second = fst::kNoStateId;
//  }
//  best_cost_and_pred[clat.Start()].first = 0;
//  for (StateId s = 0; s < clat.NumStates(); s++) {
//    double my_cost = best_cost_and_pred[s].first;
//    for (ArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done();
//         aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double arc_cost = ConvertToCost(arc.weight),
//          next_cost = my_cost + arc_cost;
//      if (next_cost < best_cost_and_pred[arc.nextstate].first) {
//        best_cost_and_pred[arc.nextstate].first = next_cost;
//        best_cost_and_pred[arc.nextstate].second = s;
//      }
//    }
//    double final_cost = ConvertToCost(clat.Final(s)),
//        tot_final = my_cost + final_cost;
//    if (tot_final < best_cost_and_pred[superfinal].first) {
//      best_cost_and_pred[superfinal].first = tot_final;
//      best_cost_and_pred[superfinal].second = s;
//    }
//  }
//  std::vector<StateId> states; // states on best path.
//  StateId cur_state = superfinal, start_state = clat.Start();
//  while (cur_state != start_state) {
//    StateId prev_state = best_cost_and_pred[cur_state].second;
//    if (prev_state == kNoStateId) {
//      KALDI_WARN << "Failure in best-path algorithm for lattice (infinite costs?)";
//      return; // return empty best-path.
//    }
//    states.push_back(prev_state);
//    KALDI_ASSERT(cur_state != prev_state && "Lattice with cycles");
//    cur_state = prev_state;
//  }
//  std::reverse(states.begin(), states.end());
//  for (size_t i = 0; i < states.size(); i++)
//    shortest_path->AddState();
//  for (StateId s = 0; static_cast<size_t>(s) < states.size(); s++) {
//    if (s == 0) shortest_path->SetStart(s);
//    if (static_cast<size_t>(s + 1) < states.size()) { // transition to next state.
//      bool have_arc = false;
//      Arc cur_arc;
//      for (ArcIterator<CompactLattice> aiter(clat, states[s]);
//           !aiter.Done();
//           aiter.Next()) {
//        const Arc &arc = aiter.Value();
//        if (arc.nextstate == states[s+1]) {
//          if (!have_arc ||
//              ConvertToCost(arc.weight) < ConvertToCost(cur_arc.weight)) {
//            cur_arc = arc;
//            have_arc = true;
//          }
//        }
//      }
//      KALDI_ASSERT(have_arc && "Code error.");
//      shortest_path->AddArc(s, Arc(cur_arc.ilabel, cur_arc.olabel,
//                                   cur_arc.weight, s+1));
//    } else { // final-prob.
//      shortest_path->SetFinal(s, clat.Final(states[s]));
//    }
//  }
//}
//
//
//void ExpandCompactLattice(const CompactLattice &clat,
//                          double epsilon,
//                          CompactLattice *expand_clat) {
//  using namespace fst;
//  typedef CompactLattice::Arc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//  typedef std::pair<StateId, StateId> StatePair;
//  typedef unordered_map<StatePair, StateId, PairHasher<StateId> > MapType;
//  typedef MapType::iterator IterType;
//
//  if (clat.Start() == kNoStateId) return;
//  // Make sure the input lattice is topologically sorted.
//  if (clat.Properties(kTopSorted, true) == 0) {
//    CompactLattice clat_copy(clat);
//    KALDI_LOG << "Topsort this lattice.";
//    if (!TopSort(&clat_copy))
//      KALDI_ERR << "Was not able to topologically sort lattice (cycles found?)";
//    ExpandCompactLattice(clat_copy, epsilon, expand_clat);
//    return;
//  }
//
//  // Compute backward logprobs betas for the expanded lattice.
//  // Note: the backward logprobs in the original lattice <clat> and the
//  // expanded lattice <expand_clat> are the same.
//  int32 num_states = clat.NumStates();
//  std::vector<double> beta(num_states, kLogZeroDouble);
//  ComputeCompactLatticeBetas(clat, &beta);
//  double tot_backward_logprob = beta[0];
//  std::vector<double> alpha;
//  alpha.push_back(0.0);
//  expand_clat->DeleteStates();
//  MapType state_map; // Map from state pair (orig_state, copy_state) to
//  // copy_state, where orig_state is a state in the original lattice, and
//  // copy_state is its corresponding one in the expanded lattice.
//  unordered_map<StateId, StateId> states; // Map from orig_state to its
//  // copy_state for states with incoming arcs' posteriors <= epsilon.
//  std::queue<StatePair> state_queue;
//
//  // Set start state in the expanded lattice.
//  StateId start_state = expand_clat->AddState();
//  expand_clat->SetStart(start_state);
//  StatePair start_pair(clat.Start(), start_state);
//  state_queue.push(start_pair);
//  std::pair<IterType, bool> result =
//    state_map.insert(std::make_pair(start_pair, start_state));
//  KALDI_ASSERT(result.second == true);
//
//  // Expand <clat> and update forward logprobs alphas in <expand_clat>.
//  while (!state_queue.empty()) {
//    StatePair s = state_queue.front();
//    StateId s1 = s.first,
//            s2 = s.second;
//    state_queue.pop();
//
//    Weight f = clat.Final(s1);
//    if (f != Weight::Zero()) {
//      KALDI_ASSERT(state_map.find(s) != state_map.end());
//      expand_clat->SetFinal(state_map[s], f);
//    }
//
//    for (ArcIterator<CompactLattice> aiter(clat, s1);
//         !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      StateId orig_state = arc.nextstate;
//      double arc_like = -ConvertToCost(arc.weight),
//             this_alpha = alpha[s2] + arc_like,
//             arc_post = Exp(this_alpha + beta[orig_state] -
//                            tot_backward_logprob);
//      // Generate the expanded lattice.
//      StateId copy_state;
//      if (arc_post > epsilon) {
//        copy_state = expand_clat->AddState();
//        StatePair next_pair(orig_state, copy_state);
//        std::pair<IterType, bool> result =
//          state_map.insert(std::make_pair(next_pair, copy_state));
//        KALDI_ASSERT(result.second == true);
//        state_queue.push(next_pair);
//      } else {
//        unordered_map<StateId, StateId>::iterator iter = states.find(orig_state);
//        if (iter == states.end() ) { // The counterpart state of orig_state
//                                   // has not been created in <expand_clat> yet.
//          copy_state = expand_clat->AddState();
//          StatePair next_pair(orig_state, copy_state);
//          std::pair<IterType, bool> result =
//            state_map.insert(std::make_pair(next_pair, copy_state));
//          KALDI_ASSERT(result.second == true);
//          state_queue.push(next_pair);
//          states[orig_state] = copy_state;
//        } else {
//          copy_state = iter->second;
//        }
//      }
//      // Create an arc from state_map[s] to copy_state in the expanded lattice.
//      expand_clat->AddArc(state_map[s], Arc(arc.ilabel, arc.olabel, arc.weight,
//                                            copy_state));
//      // Compute forward logprobs alpha for the expanded lattice.
//      if ((alpha.size() - 1) < copy_state) { // The first time to compute alpha
//                                             // for copy_state in <expand_clat>.
//        alpha.push_back(this_alpha);
//      } else { // Accumulate alpha.
//        alpha[copy_state] = LogAdd(alpha[copy_state], this_alpha);
//      }
//    }
//  } // end while
//}
//
//
//void CompactLatticeBestCostsAndTracebacks(
//    const CompactLattice &clat,
//    CostTraceType *forward_best_cost_and_pred,
//    CostTraceType *backward_best_cost_and_pred) {
//
//  // typedef the arc, weight types
//  typedef CompactLatticeArc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//
//  forward_best_cost_and_pred->clear();
//  backward_best_cost_and_pred->clear();
//  forward_best_cost_and_pred->resize(clat.NumStates());
//  backward_best_cost_and_pred->resize(clat.NumStates());
//  // Initialize the cost and predecessor state for each state.
//  for (StateId s = 0; s < clat.NumStates(); s++) {
//    (*forward_best_cost_and_pred)[s].first =
//                                        std::numeric_limits<double>::infinity();
//    (*backward_best_cost_and_pred)[s].first =
//                                        std::numeric_limits<double>::infinity();
//    (*forward_best_cost_and_pred)[s].second = fst::kNoStateId;
//    (*backward_best_cost_and_pred)[s].second = fst::kNoStateId;
//  }
//
//  StateId start_state = clat.Start();
//  (*forward_best_cost_and_pred)[start_state].first = 0;
//  // Transverse the lattice forwardly to compute the best cost from the start
//  // state to each state and the best predecessor state of each state.
//  for (StateId s = 0; s < clat.NumStates(); s++) {
//    double cur_cost = (*forward_best_cost_and_pred)[s].first;
//    for (fst::ArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double next_cost = cur_cost + ConvertToCost(arc.weight);
//      if (next_cost < (*forward_best_cost_and_pred)[arc.nextstate].first) {
//        (*forward_best_cost_and_pred)[arc.nextstate].first = next_cost;
//        (*forward_best_cost_and_pred)[arc.nextstate].second = s;
//      }
//    }
//  }
//  // Transverse the lattice backwardly to compute the best cost from a final
//  // state to each state and the best predecessor state of each state.
//  for (StateId s = clat.NumStates() - 1; s >= 0; s--) {
//    double this_cost = ConvertToCost(clat.Final(s));
//    for (fst::ArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      double next_cost = (*backward_best_cost_and_pred)[arc.nextstate].first +
//        ConvertToCost(arc.weight);
//      if (next_cost < this_cost) {
//        this_cost = next_cost;
//        (*backward_best_cost_and_pred)[s].second = arc.nextstate;
//      }
//    }
//    (*backward_best_cost_and_pred)[s].first = this_cost;
//  }
//}
//
//
//void AddNnlmScoreToCompactLattice(const MapT &nnlm_scores,
//                                  CompactLattice *clat) {
//  if (clat->Start() == fst::kNoStateId) return;
//  // Make sure the input lattice is topologically sorted.
//  if (clat->Properties(fst::kTopSorted, true) == 0) {
//    KALDI_LOG << "Topsort this lattice.";
//    if (!TopSort(clat))
//      KALDI_ERR << "Was not able to topologically sort lattice (cycles found?)";
//    AddNnlmScoreToCompactLattice(nnlm_scores, clat);
//    return;
//  }
//
//  // typedef the arc, weight types
//  typedef CompactLatticeArc Arc;
//  typedef Arc::Weight Weight;
//  typedef Arc::StateId StateId;
//  typedef std::pair<int32, int32> StatePair;
//
//  int32 num_states = clat->NumStates();
//  unordered_map<StatePair, bool, PairHasher<int32> > final_state_check;
//  for (StateId s = 0; s < num_states; s++) {
//    for (fst::MutableArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done(); aiter.Next()) {
//      Arc arc(aiter.Value());
//      StatePair arc_index = std::make_pair(static_cast<int32>(s),
//                                           static_cast<int32>(arc.nextstate));
//      MapT::const_iterator it = nnlm_scores.find(arc_index);
//      double nnlm_score;
//      if (it != nnlm_scores.end())
//        nnlm_score = it->second;
//      else
//        KALDI_ERR << "Some arc does not have neural language model score.";
//      if (arc.ilabel != 0) { // if there is a word on this arc
//        LatticeWeight weight = arc.weight.Weight();
//        // Add associated neural LM score to each arc.
//        weight.SetValue1(weight.Value1() + nnlm_score);
//        arc.weight.SetWeight(weight);
//        aiter.SetValue(arc);
//      }
//      Weight clat_final = clat->Final(arc.nextstate);
//      StatePair final_pair = std::make_pair(arc.nextstate, arc.nextstate);
//      // Add neural LM scores to each final state only once.
//      if (clat_final != CompactLatticeWeight::Zero() &&
//          final_state_check.find(final_pair) == final_state_check.end()) {
//        MapT::const_iterator final_it = nnlm_scores.find(final_pair);
//        double final_nnlm_score = 0.0;
//        if (final_it != nnlm_scores.end())
//          final_nnlm_score = final_it->second;
//        // Add neural LM scores to the final weight.
//        Weight final_weight(LatticeWeight(clat_final.Weight().Value1() +
//                                          final_nnlm_score,
//                                          clat_final.Weight().Value2()),
//                                          clat_final.String());
//        clat->SetFinal(arc.nextstate, final_weight);
//        final_state_check[final_pair] = true;
//      }
//    } // end looping over arcs
//  } // end looping over states
//}
//
//void AddWordInsPenToCompactLattice(BaseFloat word_ins_penalty,
//                                   CompactLattice *clat) {
//  typedef CompactLatticeArc Arc;
//  int32 num_states = clat->NumStates();
//
//  //scan the lattice
//  for (int32 state = 0; state < num_states; state++) {
//    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
//         !aiter.Done(); aiter.Next()) {
//
//      Arc arc(aiter.Value());
//
//      if (arc.ilabel != 0) { // if there is a word on this arc
//        LatticeWeight weight = arc.weight.Weight();
//        // add word insertion penalty to lattice
//        weight.SetValue1( weight.Value1() + word_ins_penalty);
//        arc.weight.SetWeight(weight);
//        aiter.SetValue(arc);
//      }
//    } // end looping over arcs
//  }  // end looping over states
//}
//
//struct ClatRescoreTuple {
//  ClatRescoreTuple(int32 state, int32 arc, int32 tid):
//      state_id(state), arc_id(arc), tid(tid) { }
//  int32 state_id;
//  int32 arc_id;
//  int32 tid;
//};
//
///** RescoreCompactLatticeInternal is the internal code for both
//    RescoreCompactLattice and RescoreCompatLatticeSpeedup.  For
//    RescoreCompactLattice, "tmodel" will be NULL and speedup_factor will be 1.0.
// */
//bool RescoreCompactLatticeInternal(
//    const TransitionInformation *tmodel,
//    BaseFloat speedup_factor,
//    DecodableInterface *decodable,
//    CompactLattice *clat) {
//  KALDI_ASSERT(speedup_factor >= 1.0);
//  if (clat->NumStates() == 0) {
//    KALDI_WARN << "Rescoring empty lattice";
//    return false;
//  }
//  if (!clat->Properties(fst::kTopSorted, true)) {
//    if (fst::TopSort(clat) == false) {
//      KALDI_WARN << "Cycles detected in lattice.";
//      return false;
//    }
//  }
//  std::vector<int32> state_times;
//  int32 utt_len = kaldi::CompactLatticeStateTimes(*clat, &state_times);
//
//  std::vector<std::vector<ClatRescoreTuple> > time_to_state(utt_len);
//
//  int32 num_states = clat->NumStates();
//  KALDI_ASSERT(num_states == state_times.size());
//  for (size_t state = 0; state < num_states; state++) {
//    KALDI_ASSERT(state_times[state] >= 0);
//    int32 t = state_times[state];
//    int32 arc_id = 0;
//    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
//         !aiter.Done(); aiter.Next(), arc_id++) {
//      CompactLatticeArc arc = aiter.Value();
//      std::vector<int32> arc_string = arc.weight.String();
//
//      for (size_t offset = 0; offset < arc_string.size(); offset++) {
//        if (t < utt_len) { // end state may be past this..
//          int32 tid = arc_string[offset];
//          time_to_state[t+offset].push_back(ClatRescoreTuple(state, arc_id, tid));
//        } else {
//          if (t != utt_len) {
//            KALDI_WARN << "There appears to be lattice/feature mismatch, "
//                       << "aborting.";
//            return false;
//          }
//        }
//      }
//    }
//    if (clat->Final(state) != CompactLatticeWeight::Zero()) {
//      arc_id = -1;
//      std::vector<int32> arc_string = clat->Final(state).String();
//      for (size_t offset = 0; offset < arc_string.size(); offset++) {
//        KALDI_ASSERT(t + offset < utt_len); // already checked in
//        // CompactLatticeStateTimes, so would be code error.
//        time_to_state[t+offset].push_back(
//            ClatRescoreTuple(state, arc_id, arc_string[offset]));
//      }
//    }
//  }
//
//  for (int32 t = 0; t < utt_len; t++) {
//    if ((t < utt_len - 1) && decodable->IsLastFrame(t)) {
//      KALDI_WARN << "Features are too short for lattice: utt-len is "
//                 << utt_len << ", " << t << " is last frame";
//      return false;
//    }
//    // frame_scale is the scale we put on the computed acoustic probs for this
//    // frame.  It will always be 1.0 if tmodel == NULL (i.e. if we are not doing
//    // the "speedup" code).  For frames with multiple pdf-ids it will be one.
//    // For frames with only one pdf-id, it will equal speedup_factor (>=1.0)
//    // with probability 1.0 / speedup_factor, and zero otherwise.  If it is zero,
//    // we can avoid computing the probabilities.
//    BaseFloat frame_scale = 1.0;
//    KALDI_ASSERT(!time_to_state[t].empty());
//    if (tmodel != NULL) {
//      int32 pdf_id = tmodel->TransitionIdToPdf(time_to_state[t][0].tid);
//      bool frame_has_multiple_pdfs = false;
//      for (size_t i = 1; i < time_to_state[t].size(); i++) {
//        if (tmodel->TransitionIdToPdf(time_to_state[t][i].tid) != pdf_id) {
//          frame_has_multiple_pdfs = true;
//          break;
//        }
//      }
//      if (frame_has_multiple_pdfs) {
//        frame_scale = 1.0;
//      } else {
//        if (WithProb(1.0 / speedup_factor)) {
//          frame_scale = speedup_factor;
//        } else {
//          frame_scale = 0.0;
//        }
//      }
//      if (frame_scale == 0.0)
//        continue; // the code below would be pointless.
//    }
//
//    for (size_t i = 0; i < time_to_state[t].size(); i++) {
//      int32 state = time_to_state[t][i].state_id;
//      int32 arc_id = time_to_state[t][i].arc_id;
//      int32 tid = time_to_state[t][i].tid;
//
//      if (arc_id == -1) { // Final state
//        // Access the trans_id
//        CompactLatticeWeight curr_clat_weight = clat->Final(state);
//
//        // Calculate likelihood
//        BaseFloat log_like = decodable->LogLikelihood(t, tid) * frame_scale;
//        // update weight
//        CompactLatticeWeight new_clat_weight = curr_clat_weight;
//        LatticeWeight new_lat_weight = new_clat_weight.Weight();
//        new_lat_weight.SetValue2(-log_like + curr_clat_weight.Weight().Value2());
//        new_clat_weight.SetWeight(new_lat_weight);
//        clat->SetFinal(state, new_clat_weight);
//      } else {
//        fst::MutableArcIterator<CompactLattice> aiter(clat, state);
//
//        aiter.Seek(arc_id);
//        CompactLatticeArc arc = aiter.Value();
//
//        // Calculate likelihood
//        BaseFloat log_like = decodable->LogLikelihood(t, tid) * frame_scale;
//        // update weight
//        LatticeWeight new_weight = arc.weight.Weight();
//        new_weight.SetValue2(-log_like + arc.weight.Weight().Value2());
//        arc.weight.SetWeight(new_weight);
//        aiter.SetValue(arc);
//      }
//    }
//  }
//  return true;
//}
//
//
//bool RescoreCompactLatticeSpeedup(
//    const TransitionInformation &tmodel,
//    BaseFloat speedup_factor,
//    DecodableInterface *decodable,
//    CompactLattice *clat) {
//  return RescoreCompactLatticeInternal(&tmodel, speedup_factor, decodable, clat);
//}
//
//bool RescoreCompactLattice(DecodableInterface *decodable,
//                           CompactLattice *clat) {
//  return RescoreCompactLatticeInternal(NULL, 1.0, decodable, clat);
//}
//
//
//bool RescoreLattice(DecodableInterface *decodable,
//                    Lattice *lat) {
//  if (lat->NumStates() == 0) {
//    KALDI_WARN << "Rescoring empty lattice";
//    return false;
//  }
//  if (!lat->Properties(fst::kTopSorted, true)) {
//    if (fst::TopSort(lat) == false) {
//      KALDI_WARN << "Cycles detected in lattice.";
//      return false;
//    }
//  }
//  std::vector<int32> state_times;
//  int32 utt_len = kaldi::LatticeStateTimes(*lat, &state_times);
//
//  std::vector<std::vector<int32> > time_to_state(utt_len );
//
//  int32 num_states = lat->NumStates();
//  KALDI_ASSERT(num_states == state_times.size());
//  for (size_t state = 0; state < num_states; state++) {
//    int32 t = state_times[state];
//    // Don't check t >= 0 because non-accessible states could have t = -1.
//    KALDI_ASSERT(t <= utt_len);
//    if (t >= 0 && t < utt_len)
//      time_to_state[t].push_back(state);
//  }
//
//  for (int32 t = 0; t < utt_len; t++) {
//    if ((t < utt_len - 1) && decodable->IsLastFrame(t)) {
//      KALDI_WARN << "Features are too short for lattice: utt-len is "
//                 << utt_len << ", " << t << " is last frame";
//      return false;
//    }
//    for (size_t i = 0; i < time_to_state[t].size(); i++) {
//      int32 state = time_to_state[t][i];
//      for (fst::MutableArcIterator<Lattice> aiter(lat, state);
//           !aiter.Done(); aiter.Next()) {
//        LatticeArc arc = aiter.Value();
//        if (arc.ilabel != 0) {
//          int32 trans_id = arc.ilabel; // Note: it doesn't necessarily
//          // have to be a transition-id, just whatever the Decodable
//          // object is expecting, but it's normally a transition-id.
//
//          BaseFloat log_like = decodable->LogLikelihood(t, trans_id);
//          arc.weight.SetValue2(-log_like + arc.weight.Value2());
//          aiter.SetValue(arc);
//        }
//      }
//    }
//  }
//  return true;
//}
//
//
//int32 LongestSentenceLength(const Lattice &lat) {
//  typedef Lattice::Arc Arc;
//  typedef Arc::Label Label;
//  typedef Arc::StateId StateId;
//
//  if (lat.Properties(fst::kTopSorted, true) == 0) {
//    Lattice lat_copy(lat);
//    if (!TopSort(&lat_copy))
//      KALDI_ERR << "Was not able to topologically sort lattice (cycles found?)";
//    return LongestSentenceLength(lat_copy);
//  }
//  std::vector<int32> max_length(lat.NumStates(), 0);
//  int32 lattice_max_length = 0;
//  for (StateId s = 0; s < lat.NumStates(); s++) {
//    int32 this_max_length = max_length[s];
//    for (fst::ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      bool arc_has_word = (arc.olabel != 0);
//      StateId nextstate = arc.nextstate;
//      KALDI_ASSERT(static_cast<size_t>(nextstate) < max_length.size());
//      if (arc_has_word) {
//        // A lattice should ideally not have cycles anyway; a cycle with a word
//        // on is something very bad.
//        KALDI_ASSERT(nextstate > s && "Lattice has cycles with words on.");
//        max_length[nextstate] = std::max(max_length[nextstate],
//                                         this_max_length + 1);
//      } else {
//        max_length[nextstate] = std::max(max_length[nextstate],
//                                         this_max_length);
//      }
//    }
//    if (lat.Final(s) != LatticeWeight::Zero())
//      lattice_max_length = std::max(lattice_max_length, max_length[s]);
//  }
//  return lattice_max_length;
//}
//
//int32 LongestSentenceLength(const CompactLattice &clat) {
//  typedef CompactLattice::Arc Arc;
//  typedef Arc::Label Label;
//  typedef Arc::StateId StateId;
//
//  if (clat.Properties(fst::kTopSorted, true) == 0) {
//    CompactLattice clat_copy(clat);
//    if (!TopSort(&clat_copy))
//      KALDI_ERR << "Was not able to topologically sort lattice (cycles found?)";
//    return LongestSentenceLength(clat_copy);
//  }
//  std::vector<int32> max_length(clat.NumStates(), 0);
//  int32 lattice_max_length = 0;
//  for (StateId s = 0; s < clat.NumStates(); s++) {
//    int32 this_max_length = max_length[s];
//    for (fst::ArcIterator<CompactLattice> aiter(clat, s);
//         !aiter.Done(); aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      bool arc_has_word = (arc.ilabel != 0); // note: olabel == ilabel.
//      // also note: for normal CompactLattice, e.g. as produced by
//      // determinization, all arcs will have nonzero labels, but the user might
//      // decide to remplace some of the labels with zero for some reason, and we
//      // want to support this.
//      StateId nextstate = arc.nextstate;
//      KALDI_ASSERT(static_cast<size_t>(nextstate) < max_length.size());
//      KALDI_ASSERT(nextstate > s && "CompactLattice has cycles");
//      if (arc_has_word)
//        max_length[nextstate] = std::max(max_length[nextstate],
//                                         this_max_length + 1);
//      else
//        max_length[nextstate] = std::max(max_length[nextstate],
//                                         this_max_length);
//    }
//    if (clat.Final(s) != CompactLatticeWeight::Zero())
//      lattice_max_length = std::max(lattice_max_length, max_length[s]);
//  }
//  return lattice_max_length;
//}
//
//void ComposeCompactLatticeDeterministic(
//    const CompactLattice& clat,
//    fst::DeterministicOnDemandFst<fst::StdArc>* det_fst,
//    CompactLattice* composed_clat) {
//  // StdFst::Arc and CompactLatticeArc has the same StateId type.
//  typedef fst::StdArc::StateId StateId;
//  typedef fst::StdArc::Weight Weight1;
//  typedef CompactLatticeArc::Weight Weight2;
//  typedef std::pair<StateId, StateId> StatePair;
//  typedef unordered_map<StatePair, StateId, PairHasher<StateId> > MapType;
//  typedef MapType::iterator IterType;
//
//  // Empties the output FST.
//  KALDI_ASSERT(composed_clat != NULL);
//  composed_clat->DeleteStates();
//
//  MapType state_map;
//  std::queue<StatePair> state_queue;
//
//  // Sets start state in <composed_clat>.
//  StateId start_state = composed_clat->AddState();
//  StatePair start_pair(clat.Start(), det_fst->Start());
//  composed_clat->SetStart(start_state);
//  state_queue.push(start_pair);
//  std::pair<IterType, bool> result =
//      state_map.insert(std::make_pair(start_pair, start_state));
//  KALDI_ASSERT(result.second == true);
//
//  // Starts composition here.
//  while (!state_queue.empty()) {
//    // Gets the first state in the queue.
//    StatePair s = state_queue.front();
//    StateId s1 = s.first;
//    StateId s2 = s.second;
//    state_queue.pop();
//
//
//    Weight2 clat_final = clat.Final(s1);
//    if (clat_final.Weight().Value1() !=
//        std::numeric_limits<BaseFloat>::infinity()) {
//      // Test for whether the final-prob of state s1 was zero.
//      Weight1 det_fst_final = det_fst->Final(s2);
//      if (det_fst_final.Value() !=
//          std::numeric_limits<BaseFloat>::infinity()) {
//        // Test for whether the final-prob of state s2 was zero.  If neither
//        // source-state final prob was zero, then we should create final state
//        // in fst_composed. We compute the product manually since this is more
//        // efficient.
//        Weight2 final_weight(LatticeWeight(clat_final.Weight().Value1() +
//                                           det_fst_final.Value(),
//                                           clat_final.Weight().Value2()),
//                             clat_final.String());
//        // we can assume final_weight is not Zero(), since neither of
//        // the sources was zero.
//        KALDI_ASSERT(state_map.find(s) != state_map.end());
//        composed_clat->SetFinal(state_map[s], final_weight);
//      }
//    }
//
//    // Loops over pair of edges at s1 and s2.
//    for (fst::ArcIterator<CompactLattice> aiter(clat, s1);
//         !aiter.Done(); aiter.Next()) {
//      const CompactLatticeArc& arc1 = aiter.Value();
//      fst::StdArc arc2;
//      StateId next_state1 = arc1.nextstate, next_state2;
//      bool matched = false;
//
//      if (arc1.olabel == 0) {
//        // If the symbol on <arc1> is <epsilon>, we transit to the next state
//        // for <clat>, but keep <det_fst> at the current state.
//        matched = true;
//        next_state2 = s2;
//      } else {
//        // Otherwise try to find the matched arc in <det_fst>.
//        matched = det_fst->GetArc(s2, arc1.olabel, &arc2);
//        if (matched) {
//          next_state2 = arc2.nextstate;
//        }
//      }
//
//      // If matched arc is found in <det_fst>, then we have to add new arcs to
//      // <composed_clat>.
//      if (matched) {
//        StatePair next_state_pair(next_state1, next_state2);
//        IterType siter = state_map.find(next_state_pair);
//        StateId next_state;
//
//        // Adds composed state to <state_map>.
//        if (siter == state_map.end()) {
//          // If the composed state has not been created yet, create it.
//          next_state = composed_clat->AddState();
//          std::pair<const StatePair, StateId> next_state_map(next_state_pair,
//                                                             next_state);
//          std::pair<IterType, bool> result = state_map.insert(next_state_map);
//          KALDI_ASSERT(result.second);
//          state_queue.push(next_state_pair);
//        } else {
//          // If the composed state is already in <state_map>, we can directly
//          // use that.
//          next_state = siter->second;
//        }
//
//        // Adds arc to <composed_clat>.
//        if (arc1.olabel == 0) {
//          composed_clat->AddArc(state_map[s],
//                                CompactLatticeArc(arc1.ilabel, 0,
//                                                  arc1.weight, next_state));
//        } else {
//          Weight2 composed_weight(
//              LatticeWeight(arc1.weight.Weight().Value1() +
//                            arc2.weight.Value(),
//                            arc1.weight.Weight().Value2()),
//              arc1.weight.String());
//          composed_clat->AddArc(state_map[s],
//                                CompactLatticeArc(arc1.ilabel, arc2.olabel,
//                                                  composed_weight, next_state));
//        }
//      }
//    }
//  }
//  fst::Connect(composed_clat);
//}
//
//
//void ComputeAcousticScoresMap(
//    const Lattice &lat,
//    unordered_map<std::pair<int32, int32>, std::pair<BaseFloat, int32>,
//                                        PairHasher<int32> > *acoustic_scores) {
//  // typedef the arc, weight types
//  typedef Lattice::Arc Arc;
//  typedef Arc::Weight LatticeWeight;
//  typedef Arc::StateId StateId;
//
//  acoustic_scores->clear();
//
//  std::vector<int32> state_times;
//  LatticeStateTimes(lat, &state_times);   // Assumes the input is top sorted
//
//  KALDI_ASSERT(lat.Start() == 0);
//
//  for (StateId s = 0; s < lat.NumStates(); s++) {
//    int32 t = state_times[s];
//    for (fst::ArcIterator<Lattice> aiter(lat, s); !aiter.Done();
//          aiter.Next()) {
//      const Arc &arc = aiter.Value();
//      const LatticeWeight &weight = arc.weight;
//
//      int32 tid = arc.ilabel;
//
//      if (tid != 0) {
//        unordered_map<std::pair<int32, int32>, std::pair<BaseFloat, int32>,
//          PairHasher<int32> >::iterator it = acoustic_scores->find(std::make_pair(t, tid));
//        if (it == acoustic_scores->end()) {
//          acoustic_scores->insert(std::make_pair(std::make_pair(t, tid),
//                                          std::make_pair(weight.Value2(), 1)));
//        } else {
//          if (it->second.second == 2
//                && it->second.first / it->second.second != weight.Value2()) {
//            KALDI_VLOG(2) << "Transitions on the same frame have different "
//                          << "acoustic costs for tid " << tid << "; "
//                          << it->second.first / it->second.second
//                          << " vs " << weight.Value2();
//          }
//          it->second.first += weight.Value2();
//          it->second.second++;
//        }
//      } else {
//        // Arcs with epsilon input label (tid) must have 0 acoustic cost
//        KALDI_ASSERT(weight.Value2() == 0);
//      }
//    }
//
//    LatticeWeight f = lat.Final(s);
//    if (f != LatticeWeight::Zero()) {
//      // Final acoustic cost must be 0 as we are reading from
//      // non-determinized, non-compact lattice
//      KALDI_ASSERT(f.Value2() == 0.0);
//    }
//  }
//}
//
//void ReplaceAcousticScoresFromMap(
//    const unordered_map<std::pair<int32, int32>, std::pair<BaseFloat, int32>,
//                                        PairHasher<int32> > &acoustic_scores,
//    Lattice *lat) {
//  // typedef the arc, weight types
//  typedef Lattice::Arc Arc;
//  typedef Arc::Weight LatticeWeight;
//  typedef Arc::StateId StateId;
//
//  TopSortLatticeIfNeeded(lat);
//
//  std::vector<int32> state_times;
//  LatticeStateTimes(*lat, &state_times);
//
//  KALDI_ASSERT(lat->Start() == 0);
//
//  for (StateId s = 0; s < lat->NumStates(); s++) {
//    int32 t = state_times[s];
//    for (fst::MutableArcIterator<Lattice> aiter(lat, s);
//         !aiter.Done(); aiter.Next()) {
//      Arc arc(aiter.Value());
//
//      int32 tid = arc.ilabel;
//      if (tid != 0) {
//        unordered_map<std::pair<int32, int32>, std::pair<BaseFloat, int32>,
//          PairHasher<int32> >::const_iterator it = acoustic_scores.find(std::make_pair(t, tid));
//        if (it == acoustic_scores.end()) {
//          KALDI_ERR << "Could not find tid " << tid << " at time " << t
//                    << " in the acoustic scores map.";
//        } else {
//          arc.weight.SetValue2(it->second.first / it->second.second);
//        }
//      } else {
//        // For epsilon arcs, set acoustic cost to 0.0
//        arc.weight.SetValue2(0.0);
//      }
//      aiter.SetValue(arc);
//    }
//
//    LatticeWeight f = lat->Final(s);
//    if (f != LatticeWeight::Zero()) {
//      // Set final acoustic cost to 0.0
//      f.SetValue2(0.0);
//      lat->SetFinal(s, f);
//    }
//  }
//}

}  // namespace kaldi
