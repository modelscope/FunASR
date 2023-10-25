#!/bin/bash

## Make sure that srilm is installed

dir=lm
mkdir -p $dir
[ -f path.sh ] && . ./path.sh

# Prepare data, the format of the text should be:
# BAC009S0002W0122 而 对 楼市 成交 抑制 作用 最 大 的 限 购
# BAC009S0002W0123 也 成为 地方 政府 的 眼中 钉 
corpus=lm/text

# generate lm dict
cat $corpus | awk '{for(n=2;n<=NF;n++) print tolower($n); }' | \
   cat - <(echo "<unk>";echo "<s>"; echo "</s>") | \
   sort | uniq -c | sort -nr | awk '{print $2}' > $dir/corpus.dict || exit 1;

# train ngram
cat $corpus | awk '{for(n=2;n<=NF;n++){ printf tolower($n); if(n<NF) printf " "; else print ""; }}' > $dir/train

ngram-count -text $dir/train -order 4 -limit-vocab -vocab $dir/corpus.dict -unk \
  -kndiscount -interpolate -gt1min 1 -gt2min 1 -gt3min 2  -gt4min 2 -lm $dir/lm.arpa


