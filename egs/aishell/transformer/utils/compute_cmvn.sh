#!/usr/bin/env bash

. ./path.sh || exit 1;
# Begin configuration section.
nj=32
cmd=./utils/run.pl
feats_dim=80

echo "$0 $@"

. utils/parse_options.sh || exit 1;

fbankdir=$1

split_dir=${fbankdir}/cmvn/split_${nj};
#mkdir -p $split_dir
#split_scps=""
#for n in $(seq $nj); do
#    split_scps="$split_scps $split_dir/wav.$n.scp"
#done
#utils/split_scp.pl ${fbankdir}/wav.scp $split_scps || exit 1;
#
#logdir=${fbankdir}/cmvn/log
#$cmd JOB=1:$nj $logdir/cmvn.JOB.log \
#    python utils/compute_cmvn.py \
#      --dim ${feats_dim} \
#      --wav_path $split_dir \
#      --idx JOB

python utils/combine_cmvn_file.py --dim ${feats_dim} --cmvn_dir $split_dir --nj $nj --output_dir ${fbankdir}/cmvn

#echo "$0: Succeeded compute global cmvn"
