#!/usr/bin/env python3
# encoding: utf-8

import sys

# sys.argv[1]: lm dict
# sys.argv[2]: lexicon file
# sys.argv[3]: lexicon file for corpus.dict

lex_dict = {}
with open(sys.argv[2], "r", encoding="utf8") as fin:
    for line in fin:
        words = line.strip().split("\t")
        if len(words) != 2:
            continue
        lex_dict[words[0]] = words[1]

with open(sys.argv[1], "r", encoding="utf8") as fin, open(
    sys.argv[3], "w", encoding="utf8"
) as fout:
    for line in fin:
        word = line.strip()
        if word == "<s>" or word == "</s>":
            continue
        word_lex = ""
        if word in lex_dict:
            word_lex = lex_dict[word]
        else:
            for i in range(len(word)):
                if word[i] in lex_dict:
                    word_lex += " " + lex_dict[word[i]]
                else:
                    word_lex += " <unk>"

        fout.write("{}\t{}\n".format(word, word_lex.strip()))
