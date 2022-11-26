#!/usr/bin/env bash


# Begin configuration section.
nj=4
cmd=./utils/run.pl

echo "$0 $@"

. utils/parse_options.sh || exit 1;

data=$1

[ ! -d ${data}/ark ] && echo "$0: ark data is required" && exit 1;
[ ! -d ${data}/txt ] && echo "$0: txt data is required" && exit 1;

for n in $(seq $nj); do
  echo "$data/ark/feats.$n.ark $data/txt/text.$n" || exit 1
done > $data/ark_txt.scp || exit 1

