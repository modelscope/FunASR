#!/bin/bash 
#

if [ -f path.sh ]; then . path.sh; fi

lm_dir=$1
tgt_lang=$2

arpa_lm=${lm_dir}/lm.arpa
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

# Compose the language model to FST
cat "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   grep -v -i '<unk>' | \
   arpa2fst --read-symbol-table=$tgt_lang/words.txt --keep-symbols=true - | fstprint | \
   fst/eps2disambig.pl | fst/s2eps.pl | fstcompile --isymbols=$tgt_lang/words.txt \
     --osymbols=$tgt_lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $tgt_lang/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $tgt_lang/G.fst 

# Compose the token, lexicon and language-model FST into the final decoding graph
fsttablecompose $tgt_lang/L.fst $tgt_lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $tgt_lang/LG.fst || exit 1;
fsttablecompose $tgt_lang/T.fst $tgt_lang/LG.fst > $tgt_lang/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
rm -r $tgt_lang/LG.fst   # We don't need to keep this intermediate FST

