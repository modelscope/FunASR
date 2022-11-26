#!/usr/bin/env bash

dev_num_utt=1000

echo "$0 $@"
. utils/parse_options.sh || exit 1;

train_data=$1
out_dir=$2

[ ! -f ${train_data}/wav.scp ] && echo "$0: no such file ${train_data}/wav.scp" && exit 1;
[ ! -f ${train_data}/text ] && echo "$0: no such file ${train_data}/text" && exit 1;

mkdir -p ${out_dir}/train && mkdir -p ${out_dir}/dev

cp ${train_data}/wav.scp ${out_dir}/train/wav.scp.bak
cp ${train_data}/text ${out_dir}/train/text.bak

num_utt=$(wc -l <${out_dir}/train/wav.scp.bak)

utils/shuffle_list.pl --srand 1 ${out_dir}/train/wav.scp.bak > ${out_dir}/train/wav.scp.shuf
head -n ${dev_num_utt} ${out_dir}/train/wav.scp.shuf > ${out_dir}/dev/wav.scp
tail -n $((${num_utt}-${dev_num_utt})) ${out_dir}/train/wav.scp.shuf > ${out_dir}/train/wav.scp

utils/shuffle_list.pl --srand 1 ${out_dir}/train/text.bak > ${out_dir}/train/text.shuf
head -n ${dev_num_utt} ${out_dir}/train/text.shuf > ${out_dir}/dev/text
tail -n $((${num_utt}-${dev_num_utt})) ${out_dir}/train/text.shuf > ${out_dir}/train/text

rm ${out_dir}/train/wav.scp.bak ${out_dir}/train/text.bak
rm ${out_dir}/train/wav.scp.shuf ${out_dir}/train/text.shuf
