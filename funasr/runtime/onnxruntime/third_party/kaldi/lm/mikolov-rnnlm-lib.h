// lm/mikolov-rnnlm-lib.h

// Copyright       2015  Guoguo Chen  Hainan Xu
//            2010-2012  Tomas Mikolov

// See ../../COPYING for clarification regarding multiple authors
//
// This file is based on version 0.3e of the RNNLM language modeling
// toolkit by Tomas Mikolov.  Changes made by authors other than
// Tomas Mikolov are licensed under the Apache License, the short form
// os which is below.  The original code by Tomas Mikolov is licensed
// under the BSD 3-clause license, whose text is further below.
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
// Original BSD 3-clause license text:
// Copyright (c) 2010-2012 Tomas Mikolov
//
// All rights reserved. Redistribution and use in source and binary forms, with
// or without modification, are permitted provided that the following conditions
// are met: 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following
// disclaimer. 2. Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the
// distribution. 3. Neither name of copyright holders nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission. THIS SOFTWARE IS PROVIDED
// BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef KALDI_LM_MIKOLOV_RNNLM_LIB_H_
#define KALDI_LM_MIKOLOV_RNNLM_LIB_H_

#include <string>
#include <vector>
#include "util/stl-utils.h"

namespace rnnlm {

#define MAX_STRING 100
#define MAX_FILENAME_STRING 300

typedef double real;      //  doubles for NN weights
typedef double direct_t;  //  doubles for ME weights;

struct neuron {
  real ac;    // actual value stored in neuron
  real er;    // error value in neuron, used by learning algorithm
};

struct synapse {
  real weight;  // weight of synapse
};

struct vocab_word {
  int cn;
  char word[MAX_STRING];

  real prob;
  int class_index;
};

const unsigned int PRIMES[] = {108641969, 116049371, 125925907, 133333309,
  145678979, 175308587, 197530793, 234567803, 251851741, 264197411,
  330864029, 399999781,
  407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243,
  656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
  782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127,
  953085797, 985184539, 990122807};
const unsigned int PRIMES_SIZE  =  sizeof(PRIMES) / sizeof(PRIMES[0]);

const int MAX_NGRAM_ORDER = 20;

enum FileTypeEnum {TEXT, BINARY, COMPRESSED};  // COMPRESSED not yet implemented

class CRnnLM {
 protected:
  char train_file[MAX_FILENAME_STRING];
  char valid_file[MAX_FILENAME_STRING];
  char test_file[MAX_FILENAME_STRING];
  char rnnlm_file[MAX_FILENAME_STRING];
  char lmprob_file[MAX_FILENAME_STRING];

  int rand_seed;
  int version;
  int filetype;

  int use_lmprob;
  real gradient_cutoff;

  real dynamic;

  real alpha;
  real starting_alpha;
  int alpha_divide;
  double logp, llogp;
  float min_improvement;
  int iter;
  int vocab_max_size;
  int vocab_size;
  int train_words;
  int train_cur_pos;
  int counter;

  int anti_k;

  real beta;

  int class_size;
  int **class_words;
  int *class_cn;
  int *class_max_cn;
  int old_classes;

  struct vocab_word *vocab;
  void sortVocab();
  int *vocab_hash;
  int vocab_hash_size;

  int layer0_size;
  int layer1_size;
  int layerc_size;
  int layer2_size;

  long long direct_size;
  int direct_order;
  int history[MAX_NGRAM_ORDER];

  int bptt;
  int bptt_block;
  int *bptt_history;
  neuron *bptt_hidden;
  struct synapse *bptt_syn0;

  int gen;

  int independent;

  struct neuron *neu0;    // neurons in input layer
  struct neuron *neu1;    // neurons in hidden layer
  struct neuron *neuc;    // neurons in hidden layer
  struct neuron *neu2;    // neurons in output layer

  struct synapse *syn0;   // weights between input and hidden layer
  struct synapse *syn1;   // weights between hidden and output layer
                          // (or hidden and compression if compression>0)
  struct synapse *sync;   // weights between hidden and compression layer
  direct_t *syn_d;        // direct parameters between input and output layer
                          // (similar to Maximum Entropy model parameters)

  // backup used in training:
  struct neuron *neu0b;
  struct neuron *neu1b;
  struct neuron *neucb;
  struct neuron *neu2b;

  struct synapse *syn0b;
  struct synapse *syn1b;
  struct synapse *syncb;
  direct_t *syn_db;

  // backup used in n-bset rescoring:
  struct neuron *neu1b2;

  unordered_map<std::string, float> unk_penalty;
  std::string unk_sym;

 public:

  int alpha_set, train_file_set;

  CRnnLM();

  ~CRnnLM();

  real random(real min, real max);

  void setRnnLMFile(const std::string &str);
  int getHiddenLayerSize() const { return layer1_size; }
  void setRandSeed(int newSeed);

  int getWordHash(const char *word);
  void readWord(char *word, FILE *fin);
  int searchVocab(const char *word);

  void saveWeights();      // saves current weights and unit activations
  void initNet();
  void goToDelimiter(int delim, FILE *fi);
  void restoreNet();
  void netReset();         // will erase just hidden layer state + bptt history
                           // + maxent history (called at end of sentences in
                           // the independent mode)

  void computeNet(int last_word, int word);
  void copyHiddenLayerToInput();

  void matrixXvector(struct neuron *dest, struct neuron *srcvec,
                     struct synapse *srcmatrix, int matrix_width,
                     int from, int to, int from2, int to2, int type);

  void restoreContextFromVector(const std::vector<float> &context_in);
  void saveContextToVector(std::vector<float> *context_out);

  float computeConditionalLogprob(
      std::string current_word,
      const std::vector<std::string> &history_words,
      const std::vector<float> &context_in,
      std::vector<float> *context_out);

  void setUnkSym(const std::string &unk);
  void setUnkPenalty(const std::string &filename);
  float getUnkPenalty(const std::string &word);
  bool isUnk(const std::string &word);
};

}  // namespace rnnlm

#endif  // KALDI_LM_MIKOLOV_RNNLM_LIB_H_
