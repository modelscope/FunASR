// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Utilities to convert strings into FSTs.

#ifndef FST_STRING_H_
#define FST_STRING_H_

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fst/flags.h>
#include <fst/log.h>

#include <fst/compact-fst.h>
#include <fst/icu.h>
#include <fst/mutable-fst.h>
#include <fst/util.h>


DECLARE_string(fst_field_separator);

namespace fst {

enum StringTokenType { SYMBOL = 1, BYTE = 2, UTF8 = 3 };

namespace internal {

template <class Label>
bool ConvertSymbolToLabel(const char *str, const SymbolTable *syms,
                          Label unknown_label, bool allow_negative,
                          Label *output) {
  int64 n;
  if (syms) {
    n = syms->Find(str);
    if ((n == -1) && (unknown_label != kNoLabel)) n = unknown_label;
    if (n == -1 || (!allow_negative && n < 0)) {
      LOG(ERROR) << "ConvertSymbolToLabel: Symbol \"" << str
                 << "\" is not mapped to any integer label, symbol table = "
                 << syms->Name();
      return false;
    }
  } else {
    char *p;
    n = strtoll(str, &p, 10);
    if (p < str + strlen(str) || (!allow_negative && n < 0)) {
      LOG(ERROR) << "ConvertSymbolToLabel: Bad label integer "
                 << "= \"" << str << "\"";
      return false;
    }
  }
  *output = n;
  return true;
}

template <class Label>
bool ConvertStringToLabels(const string &str, StringTokenType token_type,
                           const SymbolTable *syms, Label unknown_label,
                           bool allow_negative, std::vector<Label> *labels) {
  labels->clear();
  if (token_type == StringTokenType::BYTE) {
    labels->reserve(str.size());
    return ByteStringToLabels(str, labels);
  } else if (token_type == StringTokenType::UTF8) {
    return UTF8StringToLabels(str, labels);
  } else {
    std::unique_ptr<char[]> c_str(new char[str.size() + 1]);
    str.copy(c_str.get(), str.size());
    c_str[str.size()] = 0;
    std::vector<char *> vec;
    const string separator = "\n" + FLAGS_fst_field_separator;
    SplitString(c_str.get(), separator.c_str(), &vec, true);
    for (const char *c : vec) {
      Label label;
      if (!ConvertSymbolToLabel(c, syms, unknown_label, allow_negative,
                                &label)) {
        return false;
      }
      labels->push_back(label);
    }
  }
  return true;
}

// The sep string is used as a separator between symbols, unless it is nullptr,
// in which case the last character of FLAGS_fst_field_separator is used.
template <class Label>
bool LabelsToSymbolString(const std::vector<Label> &labels, string *str,
                          const SymbolTable &syms,
                          const string *sep = nullptr) {
  std::stringstream ostrm;
  string delim = "";
  for (auto label : labels) {
    ostrm << delim;
    const string &symbol = syms.Find(label);
    if (symbol.empty()) {
      LOG(ERROR) << "LabelToSymbolString: Label " << label
                 << " is not mapped onto any textual symbol in symbol table "
                 << syms.Name();
      return false;
    }
    ostrm << symbol;
    delim = sep != nullptr ? *sep : string(1, FLAGS_fst_field_separator.back());
  }
  *str = ostrm.str();
  return !!ostrm;
}

}  // namespace internal

// TODO(kbg): Move this just before StringPrinter.
// Converts an FST to a vector of output labels. To get input labels, use
// Project or Invert. Returns true on success. Use only with string FSTs; may
// loop for non-string FSTs.
template <class Arc>
bool StringFstToOutputLabels(const Fst<Arc> &fst,
                             std::vector<typename Arc::Label> *labels) {
  labels->clear();
  auto s = fst.Start();
  if (s == kNoStateId) {
    LOG(ERROR) << "StringFstToOutputLabels: Invalid start state";
    return false;
  }
  while (fst.Final(s) == Arc::Weight::Zero()) {
    ArcIterator<Fst<Arc>> aiter(fst, s);
    if (aiter.Done()) {
      LOG(ERROR) << "StringFstToOutputLabels: Does not reach final state";
      return false;
    }
    const auto &arc = aiter.Value();
    labels->push_back(arc.olabel);
    s = arc.nextstate;
    aiter.Next();
    if (!aiter.Done()) {
      LOG(ERROR) << "StringFstToOutputLabels: State has multiple outgoing arcs";
      return false;
    }
  }
  if (fst.NumArcs(s) != 0) {
    LOG(ERROR) << "FstToOutputLabels: Final state has outgoing arc(s)";
    return false;
  }
  return true;
}

// TODO(kbg): Move this just before StringPrinter.
// Converts a list of symbols to a string. Returns true on success. If the token
// type is SYMBOL and sep is provided, it is used to separate textual symbols.
// If the token type is SYMBOL and it is not provided, the last character of
// FLAGS_fst_field_separator is used.
template <class Label>
bool LabelsToString(const std::vector<Label> &labels, string *str,
                    StringTokenType ttype = BYTE,
                    const SymbolTable *syms = nullptr,
                    const string *sep = nullptr) {
  switch (ttype) {
    case StringTokenType::BYTE: {
      return LabelsToByteString(labels, str);
    }
    case StringTokenType::UTF8: {
      return LabelsToUTF8String(labels, str);
    }
    case StringTokenType::SYMBOL: {
      return internal::LabelsToSymbolString(labels, str, *syms, sep);
    }
  }
  return false;
}

// Functor for compiling a string in an FST.
template <class Arc>
class StringCompiler {
 public:
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;

  explicit StringCompiler(StringTokenType token_type,
                          const SymbolTable *syms = nullptr,
                          Label unknown_label = kNoLabel,
                          bool allow_negative = false)
      : token_type_(token_type),
        syms_(syms),
        unknown_label_(unknown_label),
        allow_negative_(allow_negative) {}

  // Compiles string into an FST.
  template <class FST>
  bool operator()(const string &str, FST *fst) const {
    std::vector<Label> labels;
    if (!internal::ConvertStringToLabels(str, token_type_, syms_,
                                         unknown_label_, allow_negative_,
                                         &labels)) {
      return false;
    }
    Compile(labels, fst);
    return true;
  }

  template <class FST>
  bool operator()(const string &str, FST *fst, Weight weight) const {
    std::vector<Label> labels;
    if (!internal::ConvertStringToLabels(str, token_type_, syms_,
                                         unknown_label_, allow_negative_,
                                         &labels)) {
      return false;
    }
    Compile(labels, fst, std::move(weight));
    return true;
  }

 private:
  void Compile(const std::vector<Label> &labels, MutableFst<Arc> *fst,
               Weight weight = Weight::One()) const {
    fst->DeleteStates();
    while (fst->NumStates() <= labels.size()) fst->AddState();
    for (StateId i = 0; i < labels.size(); ++i) {
      fst->AddArc(i, Arc(labels[i], labels[i], Weight::One(), i + 1));
    }
    fst->SetStart(0);
    fst->SetFinal(labels.size(), std::move(weight));
  }

  template <class Unsigned>
  void Compile(const std::vector<Label> &labels,
               CompactStringFst<Arc, Unsigned> *fst) const {
    fst->SetCompactElements(labels.begin(), labels.end());
  }

  template <class Unsigned>
  void Compile(const std::vector<Label> &labels,
               CompactWeightedStringFst<Arc, Unsigned> *fst,
               const Weight &weight = Weight::One()) const {
    std::vector<std::pair<Label, Weight>> compacts;
    compacts.reserve(labels.size() + 1);
    for (StateId i = 0; i < static_cast<StateId>(labels.size()) - 1; ++i) {
      compacts.emplace_back(labels[i], Weight::One());
    }
    compacts.emplace_back(!labels.empty() ? labels.back() : kNoLabel, weight);
    fst->SetCompactElements(compacts.begin(), compacts.end());
  }

  const StringTokenType token_type_;
  const SymbolTable *syms_;    // Symbol table (used when token type is symbol).
  const Label unknown_label_;  // Label for token missing from symbol table.
  const bool allow_negative_;  // Negative labels allowed?

  StringCompiler(const StringCompiler &) = delete;
  StringCompiler &operator=(const StringCompiler &) = delete;
};

// Helpers for StringPrinter.

// Functor for printing a string FST as a string.
template <class Arc>
class StringPrinter {
 public:
  using Label = typename Arc::Label;

  explicit StringPrinter(StringTokenType token_type,
                         const SymbolTable *syms = nullptr)
      : token_type_(token_type), syms_(syms) {}

  // Converts the FST into a string. With SYMBOL token type, sep is used as a
  // separator between symbols, unless it is nullptr, in which case the last
  // character of FLAGS_fst_field_separator is used. Returns true on success.
  bool operator()(const Fst<Arc> &fst, string *str,
                  const string *sep = nullptr) const {
    std::vector<Label> labels;
    return (StringFstToOutputLabels(fst, &labels) &&
            LabelsToString(labels, str, token_type_, syms_, sep));
  }

 private:
  const StringTokenType token_type_;
  const SymbolTable *syms_;

  StringPrinter(const StringPrinter &) = delete;
  StringPrinter &operator=(const StringPrinter &) = delete;
};

}  // namespace fst

#endif  // FST_STRING_H_
