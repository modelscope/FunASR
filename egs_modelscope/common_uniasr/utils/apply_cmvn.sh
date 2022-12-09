#!/usr/bin/env bash

. ./path.sh || exit 1;
# Begin configuration section.
nj=32
cmd=./utils/run.pl

echo "$0 $@"

. utils/parse_options.sh || exit 1;

fbankdir=$1
cmvn_file=$2
logdir=$3
output_dir=$4

dump_dir=${output_dir}/ark; mkdir -p ${dump_dir}
mkdir -p ${logdir}

$cmd JOB=1:$nj $logdir/apply_cmvn.JOB.log \
    python utils/apply_cmvn.py -a $fbankdir/ark/feats.JOB.ark \
        -c $cmvn_file -i JOB -o ${dump_dir} \
        || exit 1;

for n in $(seq $nj); do
    cat ${dump_dir}/feats.$n.scp || exit 1
done > ${output_dir}/feats.scp || exit 1

echo "$0: Succeeded apply cmvn"
