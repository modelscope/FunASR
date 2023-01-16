#!/usr/bin/env bash
# Copyright 2020  Jiawen Kang
# Apache 2.0.
#
# This script prepares the CN-Celeb2 dataset.

if [  $# != 2 ]; then
    echo "Usage: make_cnceleb2.sh <CN-Celeb2_PATH> <out_dir>"
    echo "E.g.: make_cnceleb2.sh /export/corpora/CN-Celeb2 data"
    exit 1
fi

in_dir=$1
out_dir=$2

# Prepare the cnceleb2 training data
this_out_dir=${out_dir}
mkdir -p $this_out_dir 2>/dev/null
WAVFILE=$this_out_dir/wav.scp
SPKFILE=$this_out_dir/utt2spk
rm $WAVFILE $SPKFILE 2>/dev/null
this_in_dir=${in_dir}

for spkr_id in `cat $this_in_dir/spk.lst`; do
  for f in $in_dir/data/$spkr_id/*.wav; do
    wav_id=$(basename $f | sed s:.wav$::)
    echo "${spkr_id}-${wav_id} $f" >> $WAVFILE
    echo "${spkr_id}-${wav_id} ${spkr_id}" >> $SPKFILE
  done
done

# utils/fix_data_dir.sh $this_out_dir