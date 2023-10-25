// fstext/lattice-utils-inl.h

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_FSTEXT_LATTICE_UTILS_INL_H_
#define KALDI_FSTEXT_LATTICE_UTILS_INL_H_
// Do not include this file directly.  It is included by lattice-utils.h


namespace fst {


/* Convert from FST with arc-type Weight, to one with arc-type
   CompactLatticeWeight.  Uses FactorFst to identify chains
   of states which can be turned into a single output arc. */

template<class Weight, class Int>
void ConvertLattice(
    const ExpandedFst<ArcTpl<Weight> > &ifst,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > *ofst,
    bool invert) {
  typedef ArcTpl<Weight> Arc;
  typedef typename Arc::StateId StateId;
  typedef CompactLatticeWeightTpl<Weight, Int> CompactWeight;
  typedef ArcTpl<CompactWeight> CompactArc;

  VectorFst<ArcTpl<Weight> > ffst;
  std::vector<std::vector<Int> > labels;
  if (invert) // normal case: want the ilabels as sequences on the arcs of
    Factor(ifst, &ffst, &labels);  // the output... Factor makes seqs of
                                   // ilabels.
  else {
    VectorFst<ArcTpl<Weight> > invfst(ifst);
    Invert(&invfst);
    Factor(invfst, &ffst,  &labels);
  }

  TopSort(&ffst); // Put the states in ffst in topological order, which is
  // easier on the eye when reading the text-form lattices and corresponds to
  // what we get when we generate the lattices in the decoder.

  ofst->DeleteStates();

  // The states will be numbered exactly the same as the original FST.
  // Add the states to the new FST.
  StateId num_states = ffst.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId news = ofst->AddState();
    assert(news == s);
  }
  ofst->SetStart(ffst.Start());
  for (StateId s = 0; s < num_states; s++) {
    Weight final_weight = ffst.Final(s);
    if (final_weight != Weight::Zero()) {
      CompactWeight final_compact_weight(final_weight, std::vector<Int>());
      ofst->SetFinal(s, final_compact_weight);
    }
    for (ArcIterator<ExpandedFst<Arc> > iter(ffst, s);
         !iter.Done();
         iter.Next()) {
      const Arc &arc = iter.Value();
      KALDI_PARANOID_ASSERT(arc.weight != Weight::Zero());
      // note: zero-weight arcs not allowed anyway so weight should not be zero,
      // but no harm in checking.
      CompactArc compact_arc(arc.olabel, arc.olabel,
                             CompactWeight(arc.weight, labels[arc.ilabel]),
                             arc.nextstate);
      ofst->AddArc(s, compact_arc);
    }
  }
}

template<class Weight, class Int>
void ConvertLattice(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > &ifst,
    MutableFst<ArcTpl<Weight> > *ofst,
    bool invert) {
  typedef ArcTpl<Weight> Arc;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef CompactLatticeWeightTpl<Weight, Int> CompactWeight;
  typedef ArcTpl<CompactWeight> CompactArc;
  ofst->DeleteStates();
  // make the states in the new FST have the same numbers as
  // the original ones, and add chains of states as necessary
  // to encode the string-valued weights.
  StateId num_states = ifst.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId news = ofst->AddState();
    assert(news == s);
  }
  ofst->SetStart(ifst.Start());
  for (StateId s = 0; s < num_states; s++) {
    CompactWeight final_weight = ifst.Final(s);
    if (final_weight != CompactWeight::Zero()) {
      StateId cur_state = s;
      size_t string_length = final_weight.String().size();
      for (size_t n = 0; n < string_length; n++) {
        StateId next_state = ofst->AddState();
        Label ilabel = 0;
        Arc arc(ilabel, final_weight.String()[n],
                (n == 0 ? final_weight.Weight() : Weight::One()),
                next_state);
        if (invert) std::swap(arc.ilabel, arc.olabel);
        ofst->AddArc(cur_state, arc);
        cur_state = next_state;
      }
      ofst->SetFinal(cur_state,
                     string_length > 0 ? Weight::One() : final_weight.Weight());
    }
    for (ArcIterator<ExpandedFst<CompactArc> > iter(ifst, s);
         !iter.Done();
         iter.Next()) {
      const CompactArc &arc = iter.Value();
      size_t string_length = arc.weight.String().size();
      StateId cur_state = s;
      // for all but the last element in the string--
      // add a temporary state.
      for (size_t n = 0 ; n+1 < string_length; n++) {
        StateId next_state = ofst->AddState();
        Label ilabel = (n == 0 ? arc.ilabel : 0),
            olabel = static_cast<Label>(arc.weight.String()[n]);
        Weight weight = (n == 0 ? arc.weight.Weight() : Weight::One());
        Arc new_arc(ilabel, olabel, weight, next_state);
        if (invert) std::swap(new_arc.ilabel, new_arc.olabel);
        ofst->AddArc(cur_state, new_arc);
        cur_state = next_state;
      }
      Label ilabel = (string_length <= 1 ? arc.ilabel : 0),
          olabel = (string_length > 0 ? arc.weight.String()[string_length-1] : 0);
      Weight weight = (string_length <= 1 ? arc.weight.Weight() : Weight::One());
      Arc new_arc(ilabel, olabel, weight, arc.nextstate);
      if (invert) std::swap(new_arc.ilabel, new_arc.olabel);
      ofst->AddArc(cur_state, new_arc);
    }
  }
}

// This function converts lattices between float and double;
// it works for both CompactLatticeWeight and LatticeWeight.
template<class WeightIn, class WeightOut>
void ConvertLattice(
    const ExpandedFst<ArcTpl<WeightIn> > &ifst,
    MutableFst<ArcTpl<WeightOut> > *ofst) {
  typedef ArcTpl<WeightIn> ArcIn;
  typedef ArcTpl<WeightOut> ArcOut;
  typedef typename ArcIn::StateId StateId;
  ofst->DeleteStates();
  // The states will be numbered exactly the same as the original FST.
  // Add the states to the new FST.
  StateId num_states = ifst.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId news = ofst->AddState();
    assert(news == s);
  }
  ofst->SetStart(ifst.Start());
  for (StateId s = 0; s < num_states; s++) {
    WeightIn final_iweight = ifst.Final(s);
    if (final_iweight != WeightIn::Zero()) {
      WeightOut final_oweight;
      ConvertLatticeWeight(final_iweight, &final_oweight);
      ofst->SetFinal(s, final_oweight);
    }
    for (ArcIterator<ExpandedFst<ArcIn> > iter(ifst, s);
         !iter.Done();
         iter.Next()) {
      ArcIn arc = iter.Value();
      KALDI_PARANOID_ASSERT(arc.weight != WeightIn::Zero());
      ArcOut oarc;
      ConvertLatticeWeight(arc.weight, &oarc.weight);
      oarc.ilabel = arc.ilabel;
      oarc.olabel = arc.olabel;
      oarc.nextstate = arc.nextstate;
      ofst->AddArc(s, oarc);
    }
  }
}



template<class Weight, class ScaleFloat>
void ScaleLattice(
    const std::vector<std::vector<ScaleFloat> > &scale,
    MutableFst<ArcTpl<Weight> > *fst) {
  assert(scale.size() == 2 && scale[0].size() == 2 && scale[1].size() == 2);
  if (scale == DefaultLatticeScale()) // nothing to do.
    return;
  typedef ArcTpl<Weight> Arc;
  typedef MutableFst<Arc> Fst;
  typedef typename Arc::StateId StateId;
  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<Fst> aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      arc.weight = Weight(ScaleTupleWeight(arc.weight, scale));
      aiter.SetValue(arc);
    }
    Weight final_weight = fst->Final(s);
    if (final_weight != Weight::Zero())
      fst->SetFinal(s, Weight(ScaleTupleWeight(final_weight, scale)));
  }
}

template<class Weight, class Int>
void RemoveAlignmentsFromCompactLattice(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > *fst) {
  typedef CompactLatticeWeightTpl<Weight, Int> W;
  typedef ArcTpl<W> Arc;
  typedef MutableFst<Arc> Fst;
  typedef typename Arc::StateId StateId;
  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<Fst> aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      arc.weight = W(arc.weight.Weight(), std::vector<Int>());
      aiter.SetValue(arc);
    }
    W final_weight = fst->Final(s);
    if (final_weight != W::Zero())
      fst->SetFinal(s, W(final_weight.Weight(), std::vector<Int>()));
  }
}

template<class Weight, class Int>
bool CompactLatticeHasAlignment(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<Weight, Int> > > &fst) {
  typedef CompactLatticeWeightTpl<Weight, Int> W;
  typedef ArcTpl<W> Arc;
  typedef ExpandedFst<Arc> Fst;
  typedef typename Arc::StateId StateId;
  StateId num_states = fst.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (ArcIterator<Fst> aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (!arc.weight.String().empty()) return true;
    }
    W final_weight = fst.Final(s);
    if (!final_weight.String().empty()) return true;
  }
  return false;
}


template <class Real>
void ConvertFstToLattice(
    const ExpandedFst<ArcTpl<TropicalWeight> > &ifst,
    MutableFst<ArcTpl<LatticeWeightTpl<Real> > > *ofst) {
  int32 num_states_cache = 50000;
  fst::CacheOptions cache_opts(true, num_states_cache);
  fst::MapFstOptions mapfst_opts(cache_opts);
  StdToLatticeMapper<Real> mapper;
  MapFst<StdArc, ArcTpl<LatticeWeightTpl<Real> >,
         StdToLatticeMapper<Real> > map_fst(ifst, mapper, mapfst_opts);
  *ofst = map_fst;
}


}


#endif
