// fstext/table-matcher.h

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

#ifndef KALDI_FSTEXT_TABLE_MATCHER_H_
#define KALDI_FSTEXT_TABLE_MATCHER_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>



namespace fst {


/// TableMatcher is a matcher specialized for the case where the output
/// side of the left FST always has either all-epsilons coming out of
/// a state, or a majority of the symbol table.  Therefore we can
/// either store nothing (for the all-epsilon case) or store a lookup
/// table from Labels to arc offsets.  Since the TableMatcher has to
/// iterate over all arcs in each left-hand state the first time it sees
/// it, this matcher type is not efficient if you compose with
/// something very small on the right-- unless you do it multiple
/// times and keep the matcher around. To do this requires using the
/// most advanced form of ComposeFst in Compose.h, that initializes
/// with ComposeFstImplOptions.

struct TableMatcherOptions {
  float table_ratio;  // we construct the table if it would be at least this full.
  int min_table_size;
  TableMatcherOptions(): table_ratio(0.25), min_table_size(4) { }
};


// Introducing an "impl" class for TableMatcher because
// we need to do a shallow copy of the Matcher for when
// we want to cache tables for multiple compositions.
template<class F, class BackoffMatcher = SortedMatcher<F> >
class TableMatcherImpl : public MatcherBase<typename F::Arc> {
 public:
  typedef F FST;
  typedef typename F::Arc Arc;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef StateId ArcId;  // Use this type to store arc offsets [it's actually size_t
  // in the Seek function of ArcIterator, but StateId should be big enough].
  typedef typename Arc::Weight Weight;


 public:

  TableMatcherImpl(const FST &fst, MatchType match_type,
                   const TableMatcherOptions &opts = TableMatcherOptions()):
      match_type_(match_type),
      fst_(fst.Copy()),
      loop_(match_type == MATCH_INPUT ?
            Arc(kNoLabel, 0, Weight::One(), kNoStateId) :
            Arc(0, kNoLabel, Weight::One(), kNoStateId)),
      aiter_(NULL),
      s_(kNoStateId), opts_(opts),
      backoff_matcher_(fst, match_type)
  {
    assert(opts_.min_table_size > 0);
    if (match_type == MATCH_INPUT)
      assert(fst_->Properties(kILabelSorted, true) == kILabelSorted);
    else if (match_type == MATCH_OUTPUT)
      assert(fst_->Properties(kOLabelSorted, true) == kOLabelSorted);
    else
      assert(0 && "Invalid FST properties");
  }

  virtual const FST &GetFst() const { return *fst_; }

  virtual ~TableMatcherImpl() {
    std::vector<ArcId> *const empty = ((std::vector<ArcId>*)(NULL)) + 1;  // special marker.
    for (size_t i = 0; i < tables_.size(); i++) {
      if (tables_[i] != NULL && tables_[i] != empty)
        delete tables_[i];
    }
    delete aiter_;
    delete fst_;
  }

  virtual MatchType Type(bool test) const {
    return match_type_;
  }

  void SetState(StateId s) {
    if (aiter_) {
      delete aiter_;
      aiter_ = NULL;
    }
    if (match_type_ == MATCH_NONE)
      LOG(FATAL) << "TableMatcher: bad match type";
    s_ = s;
    std::vector<ArcId> *const empty = ((std::vector<ArcId>*)(NULL)) + 1;  // special marker.
    if (static_cast<size_t>(s) >= tables_.size()) {
      assert(s>=0);
      tables_.resize(s+1, NULL);
    }
    std::vector<ArcId>* &this_table_ = tables_[s];  // note: ref to ptr.
    if (this_table_ == empty) {
      backoff_matcher_.SetState(s);
      return;
    } else if (this_table_ == NULL) {  // NULL means has not been set.
      ArcId num_arcs = fst_->NumArcs(s);
      if (num_arcs == 0 || num_arcs < opts_.min_table_size) {
        this_table_ = empty;
        backoff_matcher_.SetState(s);
        return;
      }
      ArcIterator<FST> aiter(*fst_, s);
      aiter.SetFlags(kArcNoCache|(match_type_ == MATCH_OUTPUT?kArcOLabelValue:kArcILabelValue),
                     kArcNoCache|kArcValueFlags);
      // the statement above, says: "Don't cache stuff; and I only need the ilabel/olabel
      // to be computed.
      aiter.Seek(num_arcs - 1);
      Label highest_label = (match_type_ == MATCH_OUTPUT ?
                             aiter.Value().olabel : aiter.Value().ilabel);
      if ((highest_label+1) * opts_.table_ratio > num_arcs) {
        this_table_ = empty;
        backoff_matcher_.SetState(s);
        return;  // table would be too sparse.
      }
      // OK, now we are creating the table.
      this_table_ = new std::vector<ArcId> (highest_label+1, kNoStateId);
      ArcId pos = 0;
      for (aiter.Seek(0); !aiter.Done(); aiter.Next(), pos++) {
        Label label = (match_type_ == MATCH_OUTPUT ?
                       aiter.Value().olabel : aiter.Value().ilabel);
        assert((size_t)label <= (size_t)highest_label);  // also checks >= 0.
        if ((*this_table_)[label] == kNoStateId) (*this_table_)[label] = pos;
        // set this_table_[label] to first position where arc has this
        // label.
      }
    }
    // At this point in the code, this_table_ != NULL and != empty.
    aiter_ = new ArcIterator<FST>(*fst_, s);
    aiter_->SetFlags(kArcNoCache, kArcNoCache);  // don't need to cache arcs as may only
    // need a small subset.
    loop_.nextstate = s;
    // aiter_ = NULL;
    // backoff_matcher_.SetState(s);
  }

  bool Find(Label match_label) {
    if (!aiter_) return backoff_matcher_.Find(match_label);
    else {
      match_label_ = match_label;
      current_loop_ = (match_label == 0);
      // kNoLabel means the implicit loop on the other FST --
      // matches real epsilons but not the self-loop.
      match_label_ = (match_label_ == kNoLabel ? 0 : match_label_);
      if (static_cast<size_t>(match_label_) < tables_[s_]->size() &&
         (*(tables_[s_]))[match_label_] != kNoStateId) {
        aiter_->Seek( (*(tables_[s_]))[match_label_] );  // label exists.
        return true;
      }
      return current_loop_;
    }
  }
  const Arc& Value() const {
    if (aiter_)
      return current_loop_ ? loop_ : aiter_->Value();
    else
      return backoff_matcher_.Value();
  }

  void Next() {
    if (aiter_) {
      if (current_loop_)
        current_loop_ = false;
      else
        aiter_->Next();
    } else
      backoff_matcher_.Next();
  }

  bool Done() const {
    if (aiter_ != NULL) {
      if (current_loop_)
        return false;
      if (aiter_->Done())
        return true;
      Label label = (match_type_ == MATCH_OUTPUT ?
                     aiter_->Value().olabel : aiter_->Value().ilabel);
      return (label != match_label_);
    } else
      return backoff_matcher_.Done();
  }
  const Arc &Value() {
    if (aiter_ != NULL) {
      return (current_loop_ ?  loop_ : aiter_->Value() );
    } else
      return backoff_matcher_.Value();
  }

  virtual TableMatcherImpl<FST> *Copy(bool safe = false) const {
    assert(0);  // shouldn't be called.  This is not a "real" matcher,
    // although we derive from MatcherBase for convenience.
    return NULL;
  }

  virtual uint64 Properties(uint64 props) const { return props; } // simple matcher that does
  // not change its FST, so properties are properties of FST it is applied to

 private:
  virtual void SetState_(StateId s) { SetState(s); }
  virtual bool Find_(Label label) { return Find(label); }
  virtual bool Done_() const { return Done(); }
  virtual const Arc& Value_() const { return Value(); }
  virtual void Next_() { Next(); }

  MatchType match_type_;
  FST *fst_;
  bool current_loop_;
  Label match_label_;
  Arc loop_;
  ArcIterator<FST> *aiter_;
  StateId s_;
  std::vector<std::vector<ArcId> *> tables_;
  TableMatcherOptions opts_;
  BackoffMatcher backoff_matcher_;

};


template<class F, class BackoffMatcher = SortedMatcher<F> >
class TableMatcher : public MatcherBase<typename F::Arc> {
 public:
  typedef F FST;
  typedef typename F::Arc Arc;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef StateId ArcId;  // Use this type to store arc offsets [it's actually size_t
  // in the Seek function of ArcIterator, but StateId should be big enough].
  typedef typename Arc::Weight Weight;
  typedef TableMatcherImpl<F, BackoffMatcher> Impl;

  TableMatcher(const FST &fst, MatchType match_type,
               const TableMatcherOptions &opts = TableMatcherOptions())
      : impl_(std::make_shared<Impl>(fst, match_type, opts)) { }

  TableMatcher(const TableMatcher<FST, BackoffMatcher> &matcher,
               bool safe = false)
      : impl_(matcher.impl_) {
    if (safe == true) {
      LOG(FATAL) << "TableMatcher: Safe copy not supported";
    }
  }

  virtual const FST &GetFst() const { return impl_->GetFst(); }

  virtual MatchType Type(bool test) const { return impl_->Type(test);  }

  void SetState(StateId s) { return impl_->SetState(s); }

  bool Find(Label match_label) { return impl_->Find(match_label); }

  const Arc& Value() const { return impl_->Value(); }

  void Next() { return impl_->Next(); }

  bool Done() const { return impl_->Done(); }

  const Arc &Value() {  return impl_->Value(); }

  virtual TableMatcher<FST, BackoffMatcher> *Copy(bool safe = false) const {
    return new TableMatcher<FST, BackoffMatcher> (*this, safe);
  }

  virtual uint64 Properties(uint64 props) const { return impl_->Properties(props); } // simple matcher that does
  // not change its FST, so properties are properties of FST it is applied to
 private:
  std::shared_ptr<Impl> impl_;

  virtual void SetState_(StateId s) { impl_->SetState(s); }
  virtual bool Find_(Label label) { return impl_->Find(label); }
  virtual bool Done_() const { return impl_->Done(); }
  virtual const Arc& Value_() const { return impl_->Value(); }
  virtual void Next_() { impl_->Next(); }

  TableMatcher &operator=(const TableMatcher &) = delete;
};

struct TableComposeOptions: public TableMatcherOptions {
  bool connect;  // Connect output
  ComposeFilter filter_type;  // Which pre-defined filter to use
  MatchType table_match_type;

  explicit TableComposeOptions(const TableMatcherOptions &mo,
                               bool c = true, ComposeFilter ft = SEQUENCE_FILTER,
                               MatchType tms = MATCH_OUTPUT)
      : TableMatcherOptions(mo), connect(c), filter_type(ft), table_match_type(tms) { }
  TableComposeOptions() : connect(true), filter_type(SEQUENCE_FILTER),
                          table_match_type(MATCH_OUTPUT) { }
};


template<class Arc>
void TableCompose(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                  MutableFst<Arc> *ofst,
                  const TableComposeOptions &opts = TableComposeOptions()) {
  typedef Fst<Arc> F;
  CacheOptions nopts;
  nopts.gc_limit = 0;  // Cache only the last state for fastest copy.
  if (opts.table_match_type == MATCH_OUTPUT) {
    // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
    ComposeFstImplOptions<TableMatcher<F>, SortedMatcher<F> > impl_opts(nopts);
    impl_opts.matcher1 = new TableMatcher<F>(ifst1, MATCH_OUTPUT, opts);
    *ofst = ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  } else {
    assert(opts.table_match_type == MATCH_INPUT) ;
    // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
    ComposeFstImplOptions<SortedMatcher<F>, TableMatcher<F> > impl_opts(nopts);
    impl_opts.matcher2 = new TableMatcher<F>(ifst2, MATCH_INPUT, opts);
    *ofst = ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  }
  if (opts.connect) Connect(ofst);
}


/// TableComposeCache lets us do multiple compositions while caching the same
/// matcher.
template<class F>
struct TableComposeCache {
  TableMatcher<F> *matcher;
  TableComposeOptions opts;
  TableComposeCache(const TableComposeOptions &opts = TableComposeOptions()): matcher (NULL), opts(opts) {}
  ~TableComposeCache() { delete(matcher); }
};

template<class Arc>
void TableCompose(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                  MutableFst<Arc> *ofst,
                  TableComposeCache<Fst<Arc> > *cache) {
  typedef Fst<Arc> F;
  assert(cache != NULL);
  CacheOptions nopts;
  nopts.gc_limit = 0;  // Cache only the last state for fastest copy.
  if (cache->opts.table_match_type == MATCH_OUTPUT) {
    ComposeFstImplOptions<TableMatcher<F>, SortedMatcher<F> > impl_opts(nopts);
    if (cache->matcher == NULL)
      cache->matcher = new TableMatcher<F>(ifst1, MATCH_OUTPUT, cache->opts);
    impl_opts.matcher1 = cache->matcher->Copy();  // not passing "safe": may not
    // be thread-safe-- anway I don't understand this part.
    *ofst = ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  } else {
    assert(cache->opts.table_match_type == MATCH_INPUT) ;
    ComposeFstImplOptions<SortedMatcher<F>, TableMatcher<F> > impl_opts(nopts);
    if (cache->matcher == NULL)
      cache->matcher = new TableMatcher<F>(ifst2, MATCH_INPUT, cache->opts);
    impl_opts.matcher2 = cache->matcher->Copy();
    *ofst = ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  }
  if (cache->opts.connect) Connect(ofst);
}



} // end namespace fst
#endif
