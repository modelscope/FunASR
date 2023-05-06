#!/usr/bin/env bash
# Copyright 2010-2013 Microsoft Corporation
#                     Johns Hopkins University (Author: Daniel Povey)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

split_per_spk=true
if [ "$1" == "--per-utt" ]; then
  split_per_spk=false
  shift
fi

if [ $# != 2 ]; then
  echo "Usage: $0 [--per-utt] <data-dir> <num-to-split>"
  echo "E.g.: $0 data/train 50"
  echo "It creates its output in e.g. data/train/split50/{1,2,3,...50}, or if the "
  echo "--per-utt option was given, in e.g. data/train/split50utt/{1,2,3,...50}."
  echo ""
  echo "This script will not split the data-dir if it detects that the output is newer than the input."
  echo "By default it splits per speaker (so each speaker is in only one split dir),"
  echo "but with the --per-utt option it will ignore the speaker information while splitting."
  exit 1
fi

data=$1
numsplit=$2

if ! [ "$numsplit" -gt 0 ]; then
  echo "Invalid num-split argument $numsplit";
  exit 1;
fi

if $split_per_spk; then
  warning_opt=
else
  # suppress warnings from filter_scps.pl about 'some input lines were output
  # to multiple files'.
  warning_opt="--no-warn"
fi

n=0;
feats=""
wavs=""
utt2spks=""
texts=""

nu=`cat $data/utt2spk | wc -l`
nf=`cat $data/feats.scp 2>/dev/null | wc -l`
nt=`cat $data/text 2>/dev/null | wc -l` # take it as zero if no such file
if [ -f $data/feats.scp ] && [ $nu -ne $nf ]; then
  echo "** split_data.sh: warning, #lines is (utt2spk,feats.scp) is ($nu,$nf); you can "
  echo "**  use utils/fix_data_dir.sh $data to fix this."
fi
if [ -f $data/text ] && [ $nu -ne $nt ]; then
  echo "** split_data.sh: warning, #lines is (utt2spk,text) is ($nu,$nt); you can "
  echo "** use utils/fix_data_dir.sh to fix this."
fi


if $split_per_spk; then
  utt2spk_opt="--utt2spk=$data/utt2spk"
  utt=""
else
  utt2spk_opt=
  utt="utt"
fi

s1=$data/split${numsplit}${utt}/1
if [ ! -d $s1 ]; then
  need_to_split=true
else
  need_to_split=false
  for f in utt2spk spk2utt spk2warp feats.scp text wav.scp cmvn.scp spk2gender \
    vad.scp segments reco2file_and_channel utt2lang; do
    if [[ -f $data/$f && ( ! -f $s1/$f || $s1/$f -ot $data/$f ) ]]; then
      need_to_split=true
    fi
  done
fi

if ! $need_to_split; then
  exit 0;
fi

utt2spks=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n/utt2spk; done)

directories=$(for n in `seq $numsplit`; do echo $data/split${numsplit}${utt}/$n; done)

# if this mkdir fails due to argument-list being too long, iterate.
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $numsplit`; do
    mkdir -p $data/split${numsplit}${utt}/$n
  done
fi

# If lockfile is not installed, just don't lock it.  It's not a big deal.
which lockfile >&/dev/null && lockfile -l 60 $data/.split_lock
trap 'rm -f $data/.split_lock' EXIT HUP INT PIPE TERM

utils/split_scp.pl $utt2spk_opt $data/utt2spk $utt2spks || exit 1

for n in `seq $numsplit`; do
  dsn=$data/split${numsplit}${utt}/$n
  utils/utt2spk_to_spk2utt.pl $dsn/utt2spk > $dsn/spk2utt || exit 1;
done

maybe_wav_scp=
if [ ! -f $data/segments ]; then
  maybe_wav_scp=wav.scp  # If there is no segments file, then wav file is
                         # indexed per utt.
fi

# split some things that are indexed by utterance.
for f in feats.scp text vad.scp utt2lang $maybe_wav_scp utt2dur utt2num_frames; do
  if [ -f $data/$f ]; then
    utils/filter_scps.pl JOB=1:$numsplit \
      $data/split${numsplit}${utt}/JOB/utt2spk $data/$f $data/split${numsplit}${utt}/JOB/$f || exit 1;
  fi
done

# split some things that are indexed by speaker
for f in spk2gender spk2warp cmvn.scp; do
  if [ -f $data/$f ]; then
    utils/filter_scps.pl $warning_opt JOB=1:$numsplit \
      $data/split${numsplit}${utt}/JOB/spk2utt $data/$f $data/split${numsplit}${utt}/JOB/$f || exit 1;
  fi
done

if [ -f $data/segments ]; then
  utils/filter_scps.pl JOB=1:$numsplit \
     $data/split${numsplit}${utt}/JOB/utt2spk $data/segments $data/split${numsplit}${utt}/JOB/segments || exit 1
  for n in `seq $numsplit`; do
    dsn=$data/split${numsplit}${utt}/$n
    awk '{print $2;}' $dsn/segments | sort | uniq > $dsn/tmp.reco # recording-ids.
  done
  if [ -f $data/reco2file_and_channel ]; then
    utils/filter_scps.pl $warning_opt JOB=1:$numsplit \
      $data/split${numsplit}${utt}/JOB/tmp.reco $data/reco2file_and_channel \
      $data/split${numsplit}${utt}/JOB/reco2file_and_channel || exit 1
  fi
  if [ -f $data/wav.scp ]; then
    utils/filter_scps.pl $warning_opt JOB=1:$numsplit \
      $data/split${numsplit}${utt}/JOB/tmp.reco $data/wav.scp \
      $data/split${numsplit}${utt}/JOB/wav.scp || exit 1
  fi
  for f in $data/split${numsplit}${utt}/*/tmp.reco; do rm $f; done
fi

exit 0
