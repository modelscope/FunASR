#!/bin/bash

# Collect TLG.fst
lm_dir=$1
tgt_dir=$2

tlg=${lm_dir}/lang/TLG.fst

[ ! -f $tlg ] && echo No TLG file $tlg && exit 1;

rm -rf $tgt_dir
mkdir -p $tgt_dir
cp -r $tlg ${tgt_dir}/TLG.fst

# Generate configuration file
wd_file=${lm_dir}/lang/words.txt
cfg_file=${tgt_dir}/config.yaml

[ ! -f $wd_file ] && echo No words list $wd_file && exit 1;

cat $wd_file | awk '{print $1}' | awk '
  BEGIN {
    print "token_list:";
  }
  {
    printf("- \"%s\"\n", $1);
  }
  END {
  }' > $cfg_file || exit 1;

