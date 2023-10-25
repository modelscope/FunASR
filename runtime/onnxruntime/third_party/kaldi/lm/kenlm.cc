// Copyright 2020  Jiayu DU

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
#ifdef HAVE_KENLM
#include "lm/kenlm.h"

namespace kaldi {

void KenLm::ComputeSymbolToWordIndexMapping(std::string symbol_table_filename) {
  // count symbol table size
  int num_syms = 0;
  std::string line;
  std::ifstream is(symbol_table_filename);
  while(std::getline(is, line)) {
    if (!line.empty()) num_syms++;
  }

  symid_to_wid_.clear();
  symid_to_wid_.resize(num_syms, 0);

  is.clear();
  is.seekg(0);
  int num_mapped = 0;
  while (std::getline(is, line)) {
    std::vector<std::string> fields;
    SplitStringToVector(line, " ", true, &fields);
    if (fields.size() == 2) {
      std::string sym = fields[0];
      int32 symid = 0;  ConvertStringToInteger(fields[1], &symid);
      // mark special LM word
      if (sym == bos_sym_) {
        bos_symid_ = symid;
      } else if (sym == eos_sym_) {
        eos_symid_ = symid;
      } else if (sym == "<unk>" || sym == "<UNK>") {
        unk_sym_ = sym;
        unk_symid_ = symid;
      }
      // check vocabulary consistency between kaldi and kenlm.
      // note we always handle <unk> & <UNK> as a pair, 
      // so don't worry about the literal mismatch
      // between Kaldi and kenlm arpa (<UNK> vs <unk>)
      WordIndex wid = vocab_->Index(sym.c_str());
      if ((wid == vocab_->Index("<unk>") || wid == vocab_->Index("<UNK>")) 
          && sym != "<unk>" && sym != "<UNK>"
          && sym != "<eps>"
          && sym != "#0") {
        KALDI_ERR << "found mismatched symbol: " << sym
                  << ", this symbol is in Kaldi, but is unseen in KenLm"
                  << ", they should have strictly consistent vocabulary.";
      } else {
        symid_to_wid_[symid] = wid;
        num_mapped += 1;
      }
    }
  }
  KALDI_ASSERT(num_mapped == symid_to_wid_.size());
  KALDI_LOG << "Successfully mapped " << num_mapped 
            << " Kaldi symbols to KenLm words";
}

int KenLm::Load(std::string kenlm_filename,
                std::string symbol_table_filename,
                util::LoadMethod load_method) {
  if (model_ != nullptr) { delete model_; }
  model_ = nullptr;
  vocab_ = nullptr;

  // load KenLm model
  lm::ngram::Config config;
  config.load_method = load_method;
  model_ = lm::ngram::LoadVirtual(kenlm_filename.c_str(), config);
  if (model_ == nullptr) { KALDI_ERR << "Failed to load KenLm model"; }

  // KenLm holds vocabulary internally with ownership,
  // vocab_ here is just for concise reference
  vocab_ = &model_->BaseVocabulary();
  if (vocab_ == nullptr) { KALDI_ERR << "Failed to get vocabulary from KenLm model"; }

  // compute the index mapping from Kaldi symbol to KenLm word
  ComputeSymbolToWordIndexMapping(symbol_table_filename);
  return 0;
}

} // namespace kaldi
#endif