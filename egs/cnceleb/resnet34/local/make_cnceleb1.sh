#!/usr/bin/env bash
# Copyright 2019  Jiawen Kang
# Apache 2.0.
#
# This script prepares the CN-Celeb1 dataset.
# It creates separate directories for train, eval enroll and eval test.
# It also prepares a trials files, in the eval test directory.

if [  $# != 2 ]; then
    echo "Usage: make_cnceleb1.sh <CN-Celeb1_PATH> <out_dir>"
    echo "E.g.: make_cnceleb1.sh /export/corpora/CN-Celeb1 data"
    exit 1
fi

in_dir=$1
out_dir=$2

# Prepare the cnceleb1 training data
this_out_dir=${out_dir}/cnceleb1_train
mkdir -p $this_out_dir 2>/dev/null
WAVFILE=$this_out_dir/wav.scp
SPKFILE=$this_out_dir/utt2spk
rm $WAVFILE $SPKFILE 2>/dev/null
this_in_dir=${in_dir}/dev

#for spkr_id in `cat $this_in_dir/dev.lst`; do
#  for f in $in_dir/data/$spkr_id/*.wav; do
#    wav_id=$(basename $f | sed s:.wav$::)
#    echo "${spkr_id}-${wav_id} $f" >> $WAVFILE
#    echo "${spkr_id}-${wav_id} ${spkr_id}" >> $SPKFILE
#  done
#done
# utils/fix_data_dir.sh $this_out_dir

# Prepare the evaluation data
for mode in enroll test; do
  this_out_dir=${out_dir}/eval_${mode}
  mkdir -p $this_out_dir 2>/dev/null
  WAVFILE=$this_out_dir/wav.scp
  SPKFILE=$this_out_dir/utt2spk
  rm $WAVFILE $SPKFILE 2>/dev/null
  this_in_dir=${in_dir}/eval/${mode}

  for f in $this_in_dir/*.wav; do
    wav_id=$(basename $f | sed s:.wav$::)
    spkr_id=$(echo ${wav_id} | cut -d "-" -f1)
    echo "${wav_id} $f" >> $WAVFILE
    echo "${wav_id} ${spkr_id}" >> $SPKFILE
  done
  # utils/fix_data_dir.sh $this_out_dir
done

# Prepare test trials
this_out_dir=$out_dir/eval_test/trials
mkdir -p $out_dir/eval_test/trials
this_in_dir=${in_dir}/eval/lists
cat $this_in_dir/trials.lst | sed 's@-enroll@@g' | sed 's@test/@@g' | sed 's@.wav@@g' | \
  awk '{if ($3 == "1")
         {print $1,$2,"target"}
       else
         {print $1,$2,"nontarget"}
       }'> $this_out_dir/trials.lst
