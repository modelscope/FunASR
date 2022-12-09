#!/usr/bin/env bash


# Begin configuration section.
nj=32
cmd=utils/run.pl

echo "$0 $@"

. utils/parse_options.sh || exit 1;

# tokenize configuration
text_dir=$1
seg_file=$2
logdir=$3
output_dir=$4

txt_dir=${output_dir}/txt; mkdir -p ${output_dir}/txt
mkdir -p ${logdir}

$cmd JOB=1:$nj $logdir/text_tokenize.JOB.log \
  python utils/text_tokenize.py -t ${text_dir}/txt/text.JOB.txt \
      -s ${seg_file} -i JOB -o ${txt_dir} \
      || exit 1;

# concatenate the text files together.
for n in $(seq $nj); do
  cat ${txt_dir}/text.$n.txt || exit 1
done > ${output_dir}/text || exit 1

for n in $(seq $nj); do
  cat ${txt_dir}/len.$n || exit 1
done > ${output_dir}/text_shape || exit 1

echo "$0: Succeeded text tokenize"
