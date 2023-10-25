// lm/mikolov-rnnlm-lib.cc

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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lm/mikolov-rnnlm-lib.h"
#include "util/table-types.h"

namespace rnnlm {

///// fast exp() implementation
static union {
  double d;
  struct {
    int j, i;
  } n;
} d2i;
#define EXP_A (1048576 / M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A * (y) + (1072693248 - EXP_C), d2i.d)

CRnnLM::CRnnLM() {
  version = 10;
  filetype = TEXT;

  use_lmprob = 0;
  gradient_cutoff = 15;
  dynamic = 0;

  train_file[0] = 0;
  valid_file[0] = 0;
  test_file[0] = 0;
  rnnlm_file[0] = 0;

  alpha_set = 0;
  train_file_set = 0;

  alpha = 0.1;
  beta = 0.0000001;
  // beta = 0.00000;
  alpha_divide = 0;
  logp = 0;
  llogp = -100000000;
  iter = 0;

  min_improvement = 1.003;

  train_words = 0;
  vocab_max_size = 100;
  vocab_size = 0;
  vocab = (struct vocab_word *)calloc(vocab_max_size,
                                      sizeof(struct vocab_word));

  layer1_size = 30;

  direct_size = 0;
  direct_order = 0;

  bptt = 0;
  bptt_block = 10;
  bptt_history = NULL;
  bptt_hidden = NULL;
  bptt_syn0 = NULL;

  gen = 0;

  independent = 0;

  neu0 = NULL;
  neu1 = NULL;
  neuc = NULL;
  neu2 = NULL;

  syn0 = NULL;
  syn1 = NULL;
  sync = NULL;
  syn_d = NULL;
  syn_db = NULL;
  // backup
  neu0b = NULL;
  neu1b = NULL;
  neucb = NULL;
  neu2b = NULL;

  neu1b2 = NULL;

  syn0b = NULL;
  syn1b = NULL;
  syncb = NULL;

  rand_seed = 1;

  class_size = 100;
  old_classes = 0;

  srand(rand_seed);

  vocab_hash_size = 100000000;
  vocab_hash  =  reinterpret_cast<int *>(calloc(vocab_hash_size, sizeof(int)));
}

CRnnLM::~CRnnLM() {
  int i;

  if (neu0 != NULL) {
    free(neu0);
    free(neu1);
    if (neuc != NULL) free(neuc);
    free(neu2);

    free(syn0);
    free(syn1);
    if (sync != NULL) free(sync);

    if (syn_d != NULL) free(syn_d);

    if (syn_db != NULL) free(syn_db);

    free(neu0b);
    free(neu1b);
    if (neucb != NULL) free(neucb);
    free(neu2b);

    free(neu1b2);

    free(syn0b);
    free(syn1b);
    if (syncb != NULL) free(syncb);

    for (i = 0; i < class_size; i++) {
      free(class_words[i]);
    }
    free(class_max_cn);
    free(class_cn);
    free(class_words);

    free(vocab);
    free(vocab_hash);

    if (bptt_history != NULL) free(bptt_history);
    if (bptt_hidden != NULL) free(bptt_hidden);
    if (bptt_syn0 != NULL) free(bptt_syn0);

    // todo: free bptt variables too
  }
}

real CRnnLM::random(real min, real max) {
  return rand() / (real)RAND_MAX * (max - min) + min;
}

void CRnnLM::setRnnLMFile(const std::string &str) {
  strcpy(rnnlm_file, str.c_str());
}

void CRnnLM::setRandSeed(int newSeed) {
  rand_seed = newSeed;
  srand(rand_seed);
}

void CRnnLM::readWord(char *word, FILE *fin) {
  int a = 0, ch;

  while (!feof(fin)) {
    ch = fgetc(fin);

    if (ch == 13) continue;

    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }

      if (ch == '\n') {
        strcpy(word, const_cast<char *>("</s>"));
        return;
      } else {
        continue;
      }
    }

    word[a] = ch;
    a++;

    if (a >= MAX_STRING) {
      // printf("Too long word found!\n");   //truncate too long words
      a--;
    }
  }
  word[a] = 0;
}

int CRnnLM::getWordHash(const char *word) {
  unsigned int hash, a;

  hash = 0;
  for (a = 0; a < strlen(word); a++) {
    hash = hash * 237 + word[a];
  }
  hash = hash % vocab_hash_size;

  return hash;
}

int CRnnLM::searchVocab(const char *word) {
  int a;
  unsigned int hash;

  hash = getWordHash(word);

  if (vocab_hash[hash] == -1) return -1;
  if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];

  for (a = 0; a < vocab_size; a++) {        // search in vocabulary
    if (!strcmp(word, vocab[a].word)) {
      vocab_hash[hash] = a;
      return a;
    }
  }

  return -1;              // return OOV if not found
}

void CRnnLM::sortVocab() {
  int a, b, max;
  vocab_word swap;

  for (a = 1; a < vocab_size; a++) {
    max = a;
    for (b = a + 1; b < vocab_size; b++) {
      if (vocab[max].cn < vocab[b].cn) max = b;
    }

    swap = vocab[max];
    vocab[max] = vocab[a];
    vocab[a] = swap;
  }
}

void CRnnLM::saveWeights() {      // saves current weights and unit activations
  int a, b;

  for (a = 0; a < layer0_size; a++) {
    neu0b[a].ac = neu0[a].ac;
    neu0b[a].er = neu0[a].er;
  }

  for (a = 0; a < layer1_size; a++) {
    neu1b[a].ac = neu1[a].ac;
    neu1b[a].er = neu1[a].er;
  }

  for (a = 0; a < layerc_size; a++) {
    neucb[a].ac = neuc[a].ac;
    neucb[a].er = neuc[a].er;
  }

  for (a = 0; a < layer2_size; a++) {
    neu2b[a].ac = neu2[a].ac;
    neu2b[a].er = neu2[a].er;
  }

  for (b = 0; b < layer1_size; b++) {
    for (a = 0; a < layer0_size; a++) {
      syn0b[a + b * layer0_size].weight = syn0[a + b * layer0_size].weight;
    }
  }

  if (layerc_size > 0) {
    for (b = 0; b < layerc_size; b++) {
      for (a = 0; a < layer1_size; a++) {
        syn1b[a + b * layer1_size].weight = syn1[a + b * layer1_size].weight;
      }
    }

    for (b = 0; b < layer2_size; b++) {
      for (a = 0; a < layerc_size; a++) {
        syncb[a + b * layerc_size].weight = sync[a + b * layerc_size].weight;
      }
    }
  } else {
    for (b = 0; b < layer2_size; b++) {
      for (a = 0; a < layer1_size; a++) {
        syn1b[a + b * layer1_size].weight = syn1[a + b * layer1_size].weight;
      }
    }
  }

  // for (a = 0; a < direct_size; a++) syn_db[a].weight = syn_d[a].weight;
}

void CRnnLM::initNet() {
  int a, b, cl;

  layer0_size = vocab_size + layer1_size;
  layer2_size = vocab_size + class_size;

  neu0 = (struct neuron *)calloc(layer0_size, sizeof(struct neuron));
  neu1 = (struct neuron *)calloc(layer1_size, sizeof(struct neuron));
  neuc = (struct neuron *)calloc(layerc_size, sizeof(struct neuron));
  neu2 = (struct neuron *)calloc(layer2_size, sizeof(struct neuron));

  syn0 = (struct synapse *)calloc(layer0_size * layer1_size,
                                  sizeof(struct synapse));
  if (layerc_size == 0) {
    syn1 = (struct synapse *)calloc(layer1_size * layer2_size,
                                    sizeof(struct synapse));
  } else {
    syn1 = (struct synapse *)calloc(layer1_size * layerc_size,
                                    sizeof(struct synapse));
    sync = (struct synapse *)calloc(layerc_size * layer2_size,
                                    sizeof(struct synapse));
  }

  if (syn1 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  if (layerc_size > 0)
    if (sync == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }

  syn_d =
    reinterpret_cast<direct_t *>(calloc(static_cast<long long>(direct_size),
                                         sizeof(direct_t)));

  if (syn_d == NULL) {
    printf("Memory allocation for direct"
     " connections failed (requested %lld bytes)\n",
     static_cast<long long>(direct_size) * static_cast<long long>(sizeof(direct_t)));
    exit(1);
  }

  neu0b = (struct neuron *)calloc(layer0_size, sizeof(struct neuron));
  neu1b = (struct neuron *)calloc(layer1_size, sizeof(struct neuron));
  neucb = (struct neuron *)calloc(layerc_size, sizeof(struct neuron));
  neu1b2 = (struct neuron *)calloc(layer1_size, sizeof(struct neuron));
  neu2b = (struct neuron *)calloc(layer2_size, sizeof(struct neuron));

  syn0b = (struct synapse *)calloc(layer0_size * layer1_size,
                                   sizeof(struct synapse));
  // syn1b = (struct synapse *)calloc(layer1_size*layer2_size,
  // sizeof(struct synapse));
  if (layerc_size == 0) {
    syn1b = (struct synapse *)calloc(layer1_size * layer2_size,
                                     sizeof(struct synapse));
  } else {
    syn1b = (struct synapse *)calloc(layer1_size * layerc_size,
                                     sizeof(struct synapse));
    syncb = (struct synapse *)calloc(layerc_size * layer2_size,
                                     sizeof(struct synapse));
  }

  if (syn1b == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  for (a = 0; a < layer0_size; a++) {
    neu0[a].ac = 0;
    neu0[a].er = 0;
  }

  for (a = 0; a < layer1_size; a++) {
    neu1[a].ac = 0;
    neu1[a].er = 0;
  }

  for (a = 0; a < layerc_size; a++) {
    neuc[a].ac = 0;
    neuc[a].er = 0;
  }

  for (a = 0; a < layer2_size; a++) {
    neu2[a].ac = 0;
    neu2[a].er = 0;
  }

  for (b = 0; b < layer1_size; b++) {
    for (a = 0; a < layer0_size; a++) {
      syn0[a + b * layer0_size].weight =
          random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1);
    }
  }

  if (layerc_size > 0) {
    for (b = 0; b < layerc_size; b++) {
      for (a = 0; a < layer1_size; a++) {
        syn1[a + b * layer1_size].weight =
            random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1);
      }
    }

    for (b = 0; b < layer2_size; b++) {
      for (a = 0; a < layerc_size; a++) {
        sync[a + b * layerc_size].weight =
            random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1);
      }
    }
  } else {
    for (b = 0; b < layer2_size; b++) {
      for (a = 0; a < layer1_size; a++) {
        syn1[a + b * layer1_size].weight =
            random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1);
      }
    }
  }

  long long aa;
  for (aa = 0; aa < direct_size; aa++) {
    syn_d[aa] = 0;
  }

  if (bptt > 0) {
    bptt_history = reinterpret_cast<int *>(calloc((bptt + bptt_block + 10),
                                                   sizeof(int)));
    for (a = 0; a < bptt + bptt_block; a++) {
      bptt_history[a] = -1;
    }
    bptt_hidden = reinterpret_cast<neuron *>(calloc(
                        (bptt + bptt_block + 1) * layer1_size, sizeof(neuron)));
    for (a = 0; a < (bptt + bptt_block) * layer1_size; a++) {
      bptt_hidden[a].ac = 0;
      bptt_hidden[a].er = 0;
    }
    bptt_syn0 = (struct synapse *)calloc(layer0_size * layer1_size,
                                         sizeof(struct synapse));
    if (bptt_syn0 == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
  }

  saveWeights();

  double df, dd;
  int i;

  df = 0;
  dd = 0;
  a = 0;
  b = 0;

  if (old_classes) {    // old classes
    for (i = 0; i < vocab_size; i++) {
      b += vocab[i].cn;
    }
    for (i = 0; i < vocab_size; i++) {
      df += vocab[i].cn / static_cast<double>(b);
      if (df > 1) df = 1;
      if (df > (a + 1) / static_cast<double>(class_size)) {
        vocab[i].class_index = a;
        if (a < class_size - 1) a++;
      } else {
        vocab[i].class_index = a;
      }
    }
  } else {      // new classes
    for (i = 0; i < vocab_size; i++) {
      b += vocab[i].cn;
    }
    for (i = 0; i < vocab_size; i++) {
      dd += sqrt(vocab[i].cn / static_cast<double>(b));
    }
    for (i = 0; i < vocab_size; i++) {
      df += sqrt(vocab[i].cn / static_cast<double>(b)) / dd;
      if (df > 1) df = 1;
      if (df > (a + 1) / static_cast<double>(class_size)) {
        vocab[i].class_index = a;
        if (a < class_size - 1) a++;
      } else {
        vocab[i].class_index = a;
      }
    }
  }

  // allocate auxiliary class variables (for faster search when
  // normalizing probability at output layer)

  class_words = reinterpret_cast<int **>(calloc(class_size, sizeof(int *)));
  class_cn = reinterpret_cast<int *>(calloc(class_size, sizeof(int)));
  class_max_cn = reinterpret_cast<int *>(calloc(class_size, sizeof(int)));

  for (i = 0; i < class_size; i++) {
    class_cn[i] = 0;
    class_max_cn[i] = 10;
    class_words[i] = reinterpret_cast<int *>(calloc(class_max_cn[i], sizeof(int)));
  }

  for (i = 0; i < vocab_size; i++) {
    cl = vocab[i].class_index;
    class_words[cl][class_cn[cl]] = i;
    class_cn[cl]++;
    if (class_cn[cl] + 2 >= class_max_cn[cl]) {
      class_max_cn[cl] += 10;
      class_words[cl] = reinterpret_cast<int *>(realloc(class_words[cl],
                                       class_max_cn[cl] * sizeof(int)));
    }
  }
}

void CRnnLM::goToDelimiter(int delim, FILE *fi) {
  int ch = 0;

  while (ch != delim) {
    ch = fgetc(fi);
    if (feof(fi)) {
      printf("Unexpected end of file\n");
      exit(1);
    }
  }
}

void CRnnLM::restoreNet() {   // will read whole network structure
  FILE *fi;
  int a, b, ver, unused_size;
  float fl;
  char str[MAX_STRING];
  double d;

  fi = fopen(rnnlm_file, "rb");
  if (fi == NULL) {
    printf("ERROR: model file '%s' not found!\n", rnnlm_file);
    exit(1);
  }

  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &ver);
  if ((ver == 4) && (version == 5)) {
    /* we will solve this later.. */
  } else {
    if (ver != version) {
      printf("Unknown version of file %s\n", rnnlm_file);
      exit(1);
    }
  }
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &filetype);
  goToDelimiter(':', fi);
  if (train_file_set == 0) {
    unused_size = fscanf(fi, "%s", train_file);
  } else {
    unused_size = fscanf(fi, "%s", str);
  }
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%s", valid_file);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%lf", &llogp);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &iter);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &train_cur_pos);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%lf", &logp);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &anti_k);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &train_words);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &layer0_size);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &layer1_size);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &layerc_size);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &layer2_size);
  if (ver > 5) {
    goToDelimiter(':', fi);
    unused_size = fscanf(fi, "%lld", &direct_size);
  }
  if (ver > 6) {
    goToDelimiter(':', fi);
    unused_size = fscanf(fi, "%d", &direct_order);
  }
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &bptt);
  if (ver > 4) {
    goToDelimiter(':', fi);
    unused_size = fscanf(fi, "%d", &bptt_block);
  } else {
    bptt_block = 10;
  }
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &vocab_size);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &class_size);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &old_classes);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &independent);
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%lf", &d);
  starting_alpha = d;
  goToDelimiter(':', fi);
  if (alpha_set == 0) {
    unused_size = fscanf(fi, "%lf", &d);
    alpha = d;
  } else {
    unused_size = fscanf(fi, "%lf", &d);
  }
  goToDelimiter(':', fi);
  unused_size = fscanf(fi, "%d", &alpha_divide);

  // read normal vocabulary
  if (vocab_max_size < vocab_size) {
    if (vocab != NULL) free(vocab);
    vocab_max_size = vocab_size + 1000;
    // initialize memory for vocabulary
    vocab = (struct vocab_word *)calloc(vocab_max_size,
                                        sizeof(struct vocab_word));
  }
  goToDelimiter(':', fi);
  for (a = 0; a < vocab_size; a++) {
    // unused_size = fscanf(fi, "%d%d%s%d", &b, &vocab[a].cn,
    // vocab[a].word, &vocab[a].class_index);
    unused_size = fscanf(fi, "%d%d", &b, &vocab[a].cn);
    readWord(vocab[a].word, fi);
    unused_size = fscanf(fi, "%d", &vocab[a].class_index);
    // printf("%d  %d  %s  %d\n", b, vocab[a].cn,
    // vocab[a].word, vocab[a].class_index);
  }
  if (neu0 == NULL) initNet();    // memory allocation here

  if (filetype == TEXT) {
    goToDelimiter(':', fi);
    for (a = 0; a < layer1_size; a++) {
      unused_size = fscanf(fi, "%lf", &d);
      neu1[a].ac = d;
    }
  }
  if (filetype == BINARY) {
    fgetc(fi);
    for (a = 0; a < layer1_size; a++) {
      unused_size = fread(&fl, 4, 1, fi);
      neu1[a].ac = fl;
    }
  }
  if (filetype == TEXT) {
    goToDelimiter(':', fi);
    for (b = 0; b < layer1_size; b++) {
      for (a = 0; a < layer0_size; a++) {
        unused_size = fscanf(fi, "%lf", &d);
        syn0[a + b * layer0_size].weight = d;
      }
    }
  }
  if (filetype == BINARY) {
    for (b = 0; b < layer1_size; b++) {
      for (a = 0; a < layer0_size; a++) {
        unused_size = fread(&fl, 4, 1, fi);
        syn0[a + b * layer0_size].weight = fl;
      }
    }
  }
  if (filetype == TEXT) {
    goToDelimiter(':', fi);
    if (layerc_size == 0) {  // no compress layer
      for (b = 0; b < layer2_size; b++) {
        for (a = 0; a < layer1_size; a++) {
          unused_size = fscanf(fi, "%lf", &d);
          syn1[a + b * layer1_size].weight = d;
        }
      }
    } else {        // with compress layer
      for (b = 0; b < layerc_size; b++) {
        for (a = 0; a < layer1_size; a++) {
          unused_size = fscanf(fi, "%lf", &d);
          syn1[a + b * layer1_size].weight = d;
        }
      }

      goToDelimiter(':', fi);

      for (b = 0; b < layer2_size; b++) {
        for (a = 0; a < layerc_size; a++) {
          unused_size = fscanf(fi, "%lf", &d);
          sync[a + b * layerc_size].weight = d;
        }
      }
    }
  }
  if (filetype == BINARY) {
    if (layerc_size == 0) {  // no compress layer
      for (b = 0; b < layer2_size; b++) {
        for (a = 0; a < layer1_size; a++) {
          unused_size = fread(&fl, 4, 1, fi);
          syn1[a + b * layer1_size].weight = fl;
        }
      }
    } else {        // with compress layer
      for (b = 0; b < layerc_size; b++) {
        for (a = 0; a < layer1_size; a++) {
          unused_size = fread(&fl, 4, 1, fi);
          syn1[a + b * layer1_size].weight = fl;
        }
      }

      for (b = 0; b < layer2_size; b++) {
        for (a = 0; a < layerc_size; a++) {
          unused_size = fread(&fl, 4, 1, fi);
          sync[a + b * layerc_size].weight = fl;
        }
      }
    }
  }
  if (filetype == TEXT) {
    goToDelimiter(':', fi);    // direct connections
    long long aa;
    for (aa = 0; aa < direct_size; aa++) {
      unused_size = fscanf(fi, "%lf", &d);
      syn_d[aa] = d;
    }
  }
  if (filetype == BINARY) {
    long long aa;
    for (aa = 0; aa < direct_size; aa++) {
      unused_size = fread(&fl, 4, 1, fi);
      syn_d[aa] = fl;

      /*unused_size = fread(&si, 2, 1, fi);
        fl = si/(float)(4*256);
        syn_d[aa] = fl;*/
    }
  }

  saveWeights();

  // idiom to "use" an unused variable
  (void) unused_size;

  fclose(fi);
}

void CRnnLM::netReset() {  // cleans hidden layer activation + bptt history
  int a, b;

  for (a = 0; a < layer1_size; a++) {
    neu1[a].ac = 1.0;
  }

  copyHiddenLayerToInput();

  if (bptt > 0) {
    for (a = 1; a < bptt + bptt_block; a++) {
      bptt_history[a] = 0;
    }
    for (a = bptt + bptt_block - 1; a > 1; a--) {
      for (b = 0; b < layer1_size; b++) {
        bptt_hidden[a * layer1_size + b].ac = 0;
        bptt_hidden[a * layer1_size + b].er = 0;
      }
    }
  }

  for (a = 0; a < MAX_NGRAM_ORDER; a++) {
    history[a] = 0;
  }
}

void CRnnLM::matrixXvector(struct neuron *dest, struct neuron *srcvec,
                           struct synapse *srcmatrix, int matrix_width,
                           int from, int to, int from2, int to2, int type) {
  int a, b;
  real val1, val2, val3, val4;
  real val5, val6, val7, val8;

  if (type == 0) {    // ac mod
    for (b = 0; b < (to - from) / 8; b++) {
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;

      val5 = 0;
      val6 = 0;
      val7 = 0;
      val8 = 0;

      for (a = from2; a < to2; a++) {
        val1 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 0) * matrix_width].weight;
        val2 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 1) * matrix_width].weight;
        val3 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 2) * matrix_width].weight;
        val4 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 3) * matrix_width].weight;

        val5 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 4) * matrix_width].weight;
        val6 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 5) * matrix_width].weight;
        val7 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 6) * matrix_width].weight;
        val8 += srcvec[a].ac * srcmatrix[a + (b * 8 + from + 7) * matrix_width].weight;
      }
      dest[b * 8 + from + 0].ac += val1;
      dest[b * 8 + from + 1].ac += val2;
      dest[b * 8 + from + 2].ac += val3;
      dest[b * 8 + from + 3].ac += val4;

      dest[b * 8 + from + 4].ac += val5;
      dest[b * 8 + from + 5].ac += val6;
      dest[b * 8 + from + 6].ac += val7;
      dest[b * 8 + from + 7].ac += val8;
    }

    for (b = b * 8; b < to - from; b++) {
      for (a = from2; a < to2; a++) {
        dest[b+from].ac +=
            srcvec[a].ac * srcmatrix[a + (b + from) * matrix_width].weight;
      }
    }
  } else {    // er mod
    for (a = 0; a < (to2 - from2) / 8; a++) {
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;

      val5 = 0;
      val6 = 0;
      val7 = 0;
      val8 = 0;

      for (b = from; b < to; b++) {
        val1 += srcvec[b].er * srcmatrix[a * 8 + from2 + 0 + b * matrix_width].weight;
        val2 += srcvec[b].er * srcmatrix[a * 8 + from2 + 1 + b * matrix_width].weight;
        val3 += srcvec[b].er * srcmatrix[a * 8 + from2 + 2 + b * matrix_width].weight;
        val4 += srcvec[b].er * srcmatrix[a * 8 + from2 + 3 + b * matrix_width].weight;

        val5 += srcvec[b].er * srcmatrix[a * 8 + from2 + 4 + b * matrix_width].weight;
        val6 += srcvec[b].er * srcmatrix[a * 8 + from2 + 5 + b * matrix_width].weight;
        val7 += srcvec[b].er * srcmatrix[a * 8 + from2 + 6 + b * matrix_width].weight;
        val8 += srcvec[b].er * srcmatrix[a * 8 + from2 + 7 + b * matrix_width].weight;
      }
      dest[a * 8 + from2 + 0].er += val1;
      dest[a * 8 + from2 + 1].er += val2;
      dest[a * 8 + from2 + 2].er += val3;
      dest[a * 8 + from2 + 3].er += val4;

      dest[a * 8 + from2 + 4].er += val5;
      dest[a * 8 + from2 + 5].er += val6;
      dest[a * 8 + from2 + 6].er += val7;
      dest[a * 8 + from2 + 7].er += val8;
    }

    for (a = a * 8; a < to2 - from2; a++) {
      for (b = from; b < to; b++) {
        dest[a + from2].er
            += srcvec[b].er * srcmatrix[a + from2 + b * matrix_width].weight;
      }
    }

    if (gradient_cutoff > 0)
      for (a = from2; a < to2; a++) {
        if (dest[a].er > gradient_cutoff) dest[a].er = gradient_cutoff;
        if (dest[a].er < -gradient_cutoff) dest[a].er = -gradient_cutoff;
      }
  }

  // this is normal implementation (about 3x slower):

  /*if (type == 0) {    //ac mod
    for (b = from; b < to; b++) {
    for (a = from2; a < to2; a++) {
    dest[b].ac += srcvec[a].ac * srcmatrix[a+b*matrix_width].weight;
    }
    }
    }
    else     //er mod
    if (type == 1) {
    for (a = from2; a < to2; a++) {
    for (b = from; b < to; b++) {
    dest[a].er += srcvec[b].er * srcmatrix[a+b*matrix_width].weight;
    }
    }
    }*/
}

void CRnnLM::computeNet(int last_word, int word) {
  int a, b, c;
  real val;
  double sum;   // sum is used for normalization: it's better to have larger
                // precision as many numbers are summed together here

  if (last_word != -1) neu0[last_word].ac = 1;

  // propagate 0->1
  for (a = 0; a < layer1_size; a++) {
    neu1[a].ac = 0;
  }
  for (a = 0; a < layerc_size; a++) {
    neuc[a].ac = 0;
  }

  matrixXvector(neu1, neu0, syn0, layer0_size, 0, layer1_size,
                layer0_size - layer1_size, layer0_size, 0);

  for (b = 0; b < layer1_size; b++) {
    a = last_word;
    if (a != -1) neu1[b].ac += neu0[a].ac * syn0[a + b * layer0_size].weight;
  }

  // activate 1      --sigmoid
  for (a = 0; a < layer1_size; a++) {
    if (neu1[a].ac > 50) neu1[a].ac = 50;    // for numerical stability
    if (neu1[a].ac < -50) neu1[a].ac = -50;  // for numerical stability
    val = -neu1[a].ac;
    neu1[a].ac = 1 / (1 + FAST_EXP(val));
  }

  if (layerc_size > 0) {
    matrixXvector(neuc, neu1, syn1, layer1_size,
                  0, layerc_size, 0, layer1_size, 0);
    // activate compression      --sigmoid
    for (a = 0; a < layerc_size; a++) {
      if (neuc[a].ac > 50) neuc[a].ac = 50;    // for numerical stability
      if (neuc[a].ac < -50) neuc[a].ac = -50;  // for numerical stability
      val = -neuc[a].ac;
      neuc[a].ac = 1 / (1 + FAST_EXP(val));
    }
  }

  // 1->2 class
  for (b = vocab_size; b < layer2_size; b++) {
    neu2[b].ac = 0;
  }

  if (layerc_size > 0) {
    matrixXvector(neu2, neuc, sync, layerc_size,
                  vocab_size, layer2_size, 0, layerc_size, 0);
  } else {
    matrixXvector(neu2, neu1, syn1, layer1_size,
                  vocab_size, layer2_size, 0, layer1_size, 0);
  }

  // apply direct connections to classes
  if (direct_size > 0) {
    unsigned long long hash[MAX_NGRAM_ORDER];
    // this will hold pointers to syn_d that contains hash parameters

    for (a = 0; a < direct_order; a++) {
      hash[a] = 0;
    }

    for (a = 0; a < direct_order; a++) {
      b = 0;
      if (a > 0) if (history[a - 1] == -1) break;
      // if OOV was in history, do not use this N-gram feature and higher orders
      hash[a] = PRIMES[0] * PRIMES[1];

      for (b = 1; b <= a; b++) {
        hash[a] += PRIMES[(a * PRIMES[b] + b) % PRIMES_SIZE]
            * static_cast<unsigned long long>(history[b - 1] + 1);
      }
      // update hash value based on words from the history

      hash[a] = hash[a] % (direct_size / 2);
      // make sure that starting hash index is in the first
      // half of syn_d (second part is reserved for history->words features)
    }

    for (a = vocab_size; a < layer2_size; a++) {
      for (b = 0; b < direct_order; b++) {
        if (hash[b]) {
          neu2[a].ac += syn_d[hash[b]];
          // apply current parameter and move to the next one

          hash[b]++;
        } else {
          break;
        }
      }
    }
  }

  // activation 2   --softmax on classes
  sum = 0;
  for (a = vocab_size; a < layer2_size; a++) {
    if (neu2[a].ac > 50) neu2[a].ac = 50;    // for numerical stability
    if (neu2[a].ac < -50) neu2[a].ac = -50;  // for numerical stability
    val = FAST_EXP(neu2[a].ac);
    sum+= val;
    neu2[a].ac = val;
  }
  for (a = vocab_size; a < layer2_size; a++) {
    neu2[a].ac /= sum;
  }
  // output layer activations now sum exactly to 1

  if (gen > 0) return;  // if we generate words, we don't know what current word
                        // is -> only classes are estimated and word is selected
                        // in testGen()


  // 1->2 word
  if (word != -1) {
    for (c = 0; c < class_cn[vocab[word].class_index]; c++) {
      neu2[class_words[vocab[word].class_index][c]].ac = 0;
    }
    if (layerc_size > 0) {
      matrixXvector(neu2, neuc, sync, layerc_size,
                    class_words[vocab[word].class_index][0],
                    class_words[vocab[word].class_index][0]
                    + class_cn[vocab[word].class_index],
                    0, layerc_size, 0);
    } else {
      matrixXvector(neu2, neu1, syn1, layer1_size,
                    class_words[vocab[word].class_index][0],
                    class_words[vocab[word].class_index][0]
                    + class_cn[vocab[word].class_index],
                    0, layer1_size, 0);
    }
  }

  // apply direct connections to words
  if (word != -1) if (direct_size > 0) {
    unsigned long long  hash[MAX_NGRAM_ORDER];

    for (a = 0; a < direct_order; a++) {
      hash[a] = 0;
    }

    for (a = 0; a < direct_order; a++) {
      b = 0;
      if (a > 0) if (history[a - 1] == -1) break;
      hash[a] =
          PRIMES[0] * PRIMES[1] *
          static_cast<unsigned long long>(vocab[word].class_index + 1);

      for (b = 1; b <= a; b++) {
        hash[a] += PRIMES[(a * PRIMES[b] + b) % PRIMES_SIZE]
            * static_cast<unsigned long long>(history[b - 1] + 1);
      }
      hash[a] = (hash[a] % (direct_size / 2)) + (direct_size) / 2;
    }

    for (c = 0; c < class_cn[vocab[word].class_index]; c++) {
      a = class_words[vocab[word].class_index][c];

      for (b = 0; b < direct_order; b++) if (hash[b]) {
        neu2[a].ac += syn_d[hash[b]];
        hash[b]++;
        hash[b] = hash[b] % direct_size;
      } else {
        break;
      }
    }
  }

  // activation 2   --softmax on words
  sum = 0;
  if (word != -1) {
    for (c = 0; c < class_cn[vocab[word].class_index]; c++) {
      a = class_words[vocab[word].class_index][c];
      if (neu2[a].ac > 50) neu2[a].ac = 50;    // for numerical stability
      if (neu2[a].ac < -50) neu2[a].ac = -50;  // for numerical stability
      val = FAST_EXP(neu2[a].ac);
      sum+= val;
      neu2[a].ac = val;
    }
    for (c = 0; c < class_cn[vocab[word].class_index]; c++) {
      neu2[class_words[vocab[word].class_index][c]].ac /= sum;
    }
  }
}

void CRnnLM::copyHiddenLayerToInput() {
  int a;

  for (a = 0; a < layer1_size; a++) {
    neu0[a + layer0_size - layer1_size].ac = neu1[a].ac;
  }
}

void CRnnLM::restoreContextFromVector(const std::vector <float> &context_in) {
  assert(context_in.size() == layer1_size);
  for (int i = 0; i  <  layer1_size; ++i) {
    neu1[i].ac = context_in[i];
  }
}

void CRnnLM::saveContextToVector(std::vector <float> *context_out) {
  assert(context_out != NULL);
  context_out->resize(layer1_size);
  for (int i = 0; i  <  layer1_size; ++i) {
    (*context_out)[i] = neu1[i].ac;
  }
}

float CRnnLM::computeConditionalLogprob(
    std::string current_word,
    const std::vector < std::string >  &history_words,
    const std::vector < float >  &context_in,
    std::vector < float >  *context_out) {
  // We assume the network has been restored.
  netReset();
  restoreContextFromVector(context_in);
  copyHiddenLayerToInput();

  // Maps unk to the unk symbol.
  std::vector <std::string>  history_words_nounk(history_words);
  std::string current_word_nounk = current_word;
  if (isUnk(current_word_nounk)) {
    current_word_nounk = unk_sym;
  }
  for (int i = 0; i < history_words_nounk.size(); ++i) {
    if (isUnk(history_words_nounk[i])) {
      history_words_nounk[i] = unk_sym;
    }
  }

  // Handles history for n-gram features.
  for (int i = 0; i < MAX_NGRAM_ORDER; i++) {
    history[i] = 0;
  }
  for (int i = 0; i < history_words_nounk.size() && i < MAX_NGRAM_ORDER; i++) {
    history[i] = searchVocab(
        history_words_nounk[history_words_nounk.size() - 1 - i].c_str());
  }

  int word = 0, last_word = 0;
  float logprob = 0;
  if (current_word_nounk == unk_sym) {
    logprob += getUnkPenalty(current_word);
  }
  word = searchVocab(current_word_nounk.c_str());
  if (history_words_nounk.size() > 0) {
    last_word = searchVocab(
        history_words_nounk[history_words_nounk.size() - 1].c_str());
  }
  computeNet(last_word, word);

  if (word != -1) {
    logprob +=
        log(neu2[vocab[word].class_index + vocab_size].ac * neu2[word].ac);
  } else {
    logprob += -16.118;
  }

  if (context_out != NULL) {
    saveContextToVector(context_out);
  }

  if (last_word != -1) {
    neu0[last_word].ac = 0;
  }

  return logprob;
}

bool CRnnLM::isUnk(const std::string &word) {
  int word_int = searchVocab(word.c_str());
  if (word_int == -1)
    return true;
  return false;
}

void CRnnLM::setUnkSym(const std::string &unk) {
  unk_sym = unk;
}

float CRnnLM::getUnkPenalty(const std::string &word) {
  unordered_map <std::string, float>::const_iterator iter  =
      unk_penalty.find(word);
  if (iter != unk_penalty.end())
    return iter->second;
  return -16.118;  // Fixed penalty.
}

void CRnnLM::setUnkPenalty(const std::string &filename) {
  if (filename.empty())
    return;
  kaldi::SequentialBaseFloatReader unk_reader(filename);
  for (; !unk_reader.Done(); unk_reader.Next()) {
    std::string key = unk_reader.Key();
    float prob = unk_reader.Value();
    unk_reader.FreeCurrent();
    unk_penalty[key] = log(prob);
  }
}

}  // namespace rnnlm
