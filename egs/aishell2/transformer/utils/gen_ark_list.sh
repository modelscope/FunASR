#!/usr/bin/env bash


# Begin configuration section.
nj=32
cmd=./utils/run.pl

echo "$0 $@"

. utils/parse_options.sh || exit 1;

ark_dir=$1
txt_dir=$2
output_dir=$3

[ ! -d ${ark_dir}/ark ] && echo "$0: ark data is required" && exit 1;
[ ! -d ${txt_dir}/txt ] && echo "$0: txt data is required" && exit 1;

for n in $(seq $nj); do
  echo "${ark_dir}/ark/feats.$n.ark ${txt_dir}/txt/text.$n.txt" || exit 1
done > ${output_dir}/ark_txt.scp || exit 1

