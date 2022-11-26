#!/bin/bash

# Copyright 2017 Xingyu Na
# Apache 2.0

#. ./path.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <audio-path> <text-path> <output-path>"
  echo " $0 /export/a05/xna/data/data_aishell/wav /export/a05/xna/data/data_aishell/transcript data"
  exit 1;
fi

aishell_audio_dir=$1
aishell_text=$2/aishell_transcript_v0.8.txt
output_dir=$3

train_dir=$output_dir/data/local/train
dev_dir=$output_dir/data/local/dev
test_dir=$output_dir/data/local/test
tmp_dir=$output_dir/data/local/tmp

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 141925 ] && \
  echo Warning: expected 141925 data data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

rm -r $tmp_dir

# Transcriptions preparation
for dir in $train_dir $dev_dir $test_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt > $dir/text
done

mkdir -p $output_dir/data/train $output_dir/data/dev $output_dir/data/test

for f in wav.scp text; do
  cp $train_dir/$f $output_dir/data/train/$f || exit 1;
  cp $dev_dir/$f $output_dir/data/dev/$f || exit 1;
  cp $test_dir/$f $output_dir/data/test/$f || exit 1;
done

echo "$0: AISHELL data preparation succeeded"
exit 0;
