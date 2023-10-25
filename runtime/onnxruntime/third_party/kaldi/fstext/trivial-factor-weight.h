// fstext/trivial-factor-weight.h

// Copyright 2009-2011  Microsoft Corporation

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
//
// This is a modified file from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: allauzen@google.com (Cyril Allauzen)


#ifndef KALDI_FSTEXT_TRIVIAL_FACTOR_WEIGHT_H_
#define KALDI_FSTEXT_TRIVIAL_FACTOR_WEIGHT_H_


// TrivialFactorWeight.h This is an extension to factor-weight.h in the OpenFst
// code.  It is a version of FactorWeight that creates separate states (with
// input epsilons) rather than pushing the factors forward.  This is for
// converting from Gallic FSTs, where you want the result to be a bit more
// trivial with input epsilons inserted where there are multiple output symbols.
// This has the advantage that it always works, for any input (also I just
// prefer this approach).

#include <unordered_map>
using std::unordered_map;

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <fst/cache.h>
#include <fst/test-properties.h>

namespace fst {


template <class Arc>
struct TrivialFactorWeightOptions : CacheOptions {
  typedef typename Arc::Label Label;
  float delta;
  Label extra_ilabel;  // input label of extra arcs
  Label extra_olabel;  // output label of extra arcs

  TrivialFactorWeightOptions(const CacheOptions &opts, float d,
                      Label il = 0, Label ol = 0)
      : CacheOptions(opts), delta(d), extra_ilabel(il), extra_olabel(ol) {}

  explicit TrivialFactorWeightOptions(
      float d, Label il = 0, Label ol = 0)
      : delta(d), extra_ilabel(il), extra_olabel(ol) {}

  TrivialFactorWeightOptions(): delta(kDelta), extra_ilabel(0), extra_olabel(0) {}

};

namespace internal {

// Implementation class for TrivialFactorWeight
template <class A, class F>
class TrivialFactorWeightFstImpl
    : public CacheImpl<A> {
 public:
  using CacheImpl<A>::PushArc;
  using FstImpl<A>::SetType;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::Properties;
  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;

  using CacheBaseImpl< CacheState<A> >::HasStart;
  using CacheBaseImpl< CacheState<A> >::HasFinal;
  using CacheBaseImpl< CacheState<A> >::HasArcs;

  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef F FactorIterator;

  typedef DefaultCacheStore<A> Store;
  typedef typename Store::State State;

  struct Element {
    Element() {}

    Element(StateId s, Weight w) : state(s), weight(w) {}

    StateId state;     // Input state Id
    Weight weight;     // Residual weight
  };

  TrivialFactorWeightFstImpl(const Fst<A> &fst, const TrivialFactorWeightOptions<A> &opts)
      : CacheImpl<A>(opts),
        fst_(fst.Copy()),
        delta_(opts.delta),
        extra_ilabel_(opts.extra_ilabel),
        extra_olabel_(opts.extra_olabel) {
    SetType("factor-weight");
    uint64 props = fst.Properties(kFstProperties, false);
    SetProperties(FactorWeightProperties(props), kCopyProperties);

    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
  }

  TrivialFactorWeightFstImpl(const TrivialFactorWeightFstImpl<A, F> &impl)
      : CacheImpl<A>(impl),
        fst_(impl.fst_->Copy(true)),
        delta_(impl.delta_),
        extra_ilabel_(impl.extra_ilabel_),
        extra_olabel_(impl.extra_olabel_) {
    SetType("factor-weight");
    SetProperties(impl.Properties(), kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
  }

  StateId Start() {
    if (!HasStart()) {
      StateId s = fst_->Start();
      if (s == kNoStateId)
        return kNoStateId;
      StateId start = this->FindState(Element(fst_->Start(), Weight::One()));
      this->SetStart(start);
    }
    return CacheImpl<A>::Start();
  }

  Weight Final(StateId s) {
    if (!HasFinal(s)) {
      const Element &e = elements_[s];
      Weight w;
      if (e.state == kNoStateId) {  // extra state inserted to represent final weights.
        FactorIterator fit(e.weight);
        if (fit.Done()) {  // cannot be factored.
          w = e.weight;  // so it's final
        } else {
          w = Weight::Zero();  // need another transition.
        }
      } else {
        if (e.weight != Weight::One()) {  // Not a real state.
          w = Weight::Zero();
        } else {  // corresponds to a "real" state.
          w = fst_->Final(e.state);
          FactorIterator fit(w);
          if (!fit.Done()) // we would have intermediate states representing this final state.
            w = Weight::Zero();
        }
      }
      this->SetFinal(s, w);
      return w;
    } else {
      return CacheImpl<A>::Final(s);
    }
  }

  size_t NumArcs(StateId s) {
    if (!HasArcs(s))
      Expand(s);
    return CacheImpl<A>::NumArcs(s);
  }

  size_t NumInputEpsilons(StateId s) {
    if (!HasArcs(s))
      Expand(s);
    return CacheImpl<A>::NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) {
    if (!HasArcs(s))
      Expand(s);
    return CacheImpl<A>::NumOutputEpsilons(s);
  }

  void InitArcIterator(StateId s, ArcIteratorData<A> *data) {
    if (!HasArcs(s))
      Expand(s);
    CacheImpl<A>::InitArcIterator(s, data);
  }


  // Find state corresponding to an element. Create new state
  // if element not found.
  StateId FindState(const Element &e) {
    typename ElementMap::iterator eit = element_map_.find(e);
    if (eit != element_map_.end()) {
      return (*eit).second;
    } else {
      StateId s = elements_.size();
      elements_.push_back(e);
      element_map_.insert(std::pair<const Element, StateId>(e, s));
      return s;
    }
  }

  // Computes the outgoing transitions from a state, creating new destination
  // states as needed.
  void Expand(StateId s) {
    CHECK(static_cast<size_t>(s) < elements_.size());
    Element e = elements_[s];
    if (e.weight != Weight::One()) {
      FactorIterator fit(e.weight);
      if (fit.Done()) {  // Cannot be factored-> create a link to dest state directly
        if (e.state != kNoStateId) {
          StateId dest = FindState(Element(e.state, Weight::One()));
          PushArc(s, Arc(extra_ilabel_, extra_olabel_, e.weight, dest));
        } // else we're done.  This is a final state.
      } else {  // Can be factored.
        const std::pair<Weight, Weight> &p = fit.Value();
        StateId dest = FindState(Element(e.state, p.second.Quantize(delta_)));
        PushArc(s, Arc(extra_ilabel_, extra_olabel_, p.first, dest));
      }
    } else {  // Unit weight.  This corresponds to a "real" state.
      CHECK(e.state != kNoStateId);
      for (ArcIterator< Fst<A> > ait(*fst_, e.state);
           !ait.Done();
           ait.Next()) {
        const A &arc = ait.Value();
        FactorIterator fit(arc.weight);
        if (fit.Done()) {  // cannot be factored->just link directly to dest.
          StateId dest = FindState(Element(arc.nextstate, Weight::One()));
          PushArc(s, Arc(arc.ilabel, arc.olabel, arc.weight, dest));
        } else {
          const std::pair<Weight, Weight> &p = fit.Value();
          StateId dest = FindState(Element(arc.nextstate, p.second.Quantize(delta_)));
          PushArc(s, Arc(arc.ilabel, arc.olabel, p.first, dest));
        }
      }
      // See if we have to add arcs for final-states [only if final-weight is factorable].
      Weight final_w = fst_->Final(e.state);
      if (final_w != Weight::Zero()) {
        FactorIterator fit(final_w);
        if (!fit.Done()) {
          const std::pair<Weight, Weight> &p = fit.Value();
          StateId dest = FindState(Element(kNoStateId, p.second.Quantize(delta_)));
          PushArc(s, Arc(extra_ilabel_, extra_olabel_, p.first, dest));
        }
      }
    }
    this->SetArcs(s);
  }

 private:
  // Equality function for Elements, assume weights have been quantized.
  class ElementEqual {
   public:
    bool operator()(const Element &x, const Element &y) const {
      return x.state == y.state && x.weight == y.weight;
    }
  };

  // Hash function for Elements to Fst states.
  class ElementKey {
   public:
    size_t operator()(const Element &x) const {
      return static_cast<size_t>(x.state * kPrime + x.weight.Hash());
    }
   private:
    static const int kPrime = 7853;
  };

  typedef unordered_map<Element, StateId, ElementKey, ElementEqual> ElementMap;

  std::unique_ptr<const Fst<A>> fst_;
  float delta_;
  uint32 mode_;               // factoring arc and/or final weights
  Label extra_ilabel_;        // ilabel of arc created when factoring final w's
  Label extra_olabel_;        // olabel of arc created when factoring final w's
  std::vector<Element> elements_;  // mapping Fst state to Elements
  ElementMap element_map_;    // mapping Elements to Fst state

};

}  // namespace internal

/// TrivialFactorWeightFst takes as template parameter a FactorIterator as
/// defined above. The result of weight factoring is a transducer
/// equivalent to the input whose path weights have been factored
/// according to the FactorIterator. States and transitions will be
/// added as necessary.
/// This algorithm differs from the one implemented in FactorWeightFst
/// in that it does not attempt to push the extra weight forward to the
/// next state: it uses a sequence of "extra" intermediate state, and
/// outputs the remaining weight right away.  This ensures that it will
/// always succeed, even for Gallic representations of FSTs that have cycles
/// with more output than input symbols.

/// Note that the code below was modified from factor-weight.h by just
/// search-and-replacing "FactorWeight" by "TrivialFactorWeight".


template <class A, class F>
class TrivialFactorWeightFst :
    public ImplToFst<internal::TrivialFactorWeightFstImpl<A, F>> {
 public:
  friend class ArcIterator< TrivialFactorWeightFst<A, F> >;
  friend class StateIterator< TrivialFactorWeightFst<A, F> >;

  typedef A Arc;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef DefaultCacheStore<Arc> Store;
  typedef typename Store::State State;
  typedef internal::TrivialFactorWeightFstImpl<A, F> Impl;

  explicit TrivialFactorWeightFst(const Fst<A> &fst)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, TrivialFactorWeightOptions<A>())) {}

  TrivialFactorWeightFst(const Fst<A> &fst,  const TrivialFactorWeightOptions<A> &opts)
      : ImplToFst<Impl>(std::make_shared<Impl>(fst, opts)) {}

  // See Fst<>::Copy() for doc.
  TrivialFactorWeightFst(const TrivialFactorWeightFst<A, F> &fst, bool copy)
      : ImplToFst<Impl>(fst, copy) {}

  // Get a copy of this TrivialFactorWeightFst. See Fst<>::Copy() for further doc.
  TrivialFactorWeightFst<A, F> *Copy(bool copy = false) const override {
    return new TrivialFactorWeightFst<A, F>(*this, copy);
  }

  inline void InitStateIterator(StateIteratorData<A> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<A> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  TrivialFactorWeightFst &operator=(const TrivialFactorWeightFst &fst) = delete;
};


// Specialization for TrivialFactorWeightFst.
template<class A, class F>
class StateIterator< TrivialFactorWeightFst<A, F> >
    : public CacheStateIterator< TrivialFactorWeightFst<A, F> > {
 public:
  explicit StateIterator(const TrivialFactorWeightFst<A, F> &fst)
      : CacheStateIterator< TrivialFactorWeightFst<A, F> >(fst, fst.GetMutableImpl()) {}
};


// Specialization for TrivialFactorWeightFst.
template <class A, class F>
class ArcIterator< TrivialFactorWeightFst<A, F> >
    : public CacheArcIterator< TrivialFactorWeightFst<A, F> > {
 public:
  typedef typename A::StateId StateId;

  ArcIterator(const TrivialFactorWeightFst<A, F> &fst, StateId s)
      : CacheArcIterator< TrivialFactorWeightFst<A, F>>(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) fst.GetMutableImpl()->Expand(s);
  }
};

template <class A, class F>
inline void TrivialFactorWeightFst<A, F>::InitStateIterator(
    StateIteratorData<A> *data) const {
  data->base = new StateIterator< TrivialFactorWeightFst<A, F> >(*this);
}




}  // namespace fst

#endif
