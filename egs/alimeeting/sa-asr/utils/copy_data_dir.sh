#!/usr/bin/env bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  feats.scp
#  wav.scp
#  vad.scp
#  spk2utt
#  utt2spk
#  text
#
# It copies to another directory, possibly adding a specified prefix or a suffix
# to the utterance and/or speaker names.  Note, the recording-ids stay the same.
#


# begin configuration section
spk_prefix=
utt_prefix=
spk_suffix=
utt_suffix=
validate_opts=   # should rarely be needed.
# end configuration section

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 --spk-prefix=1- --utt-prefix=1- data/train data/train_1"
  echo "Options"
  echo "   --spk-prefix=<prefix>     # Prefix for speaker ids, default empty"
  echo "   --utt-prefix=<prefix>     # Prefix for utterance ids, default empty"
  echo "   --spk-suffix=<suffix>     # Suffix for speaker ids, default empty"
  echo "   --utt-suffix=<suffix>     # Suffix for utterance ids, default empty"
  exit 1;
fi


export LC_ALL=C

srcdir=$1
destdir=$2

if [ ! -f $srcdir/utt2spk ]; then
  echo "copy_data_dir.sh: no such file $srcdir/utt2spk"
  exit 1;
fi

if [ "$destdir" == "$srcdir" ]; then
  echo "$0: this script requires <srcdir> and <destdir> to be different."
  exit 1
fi

set -e;

mkdir -p $destdir

cat $srcdir/utt2spk | awk -v p=$utt_prefix -v s=$utt_suffix '{printf("%s %s%s%s\n", $1, p, $1, s);}' > $destdir/utt_map
cat $srcdir/spk2utt | awk -v p=$spk_prefix -v s=$spk_suffix '{printf("%s %s%s%s\n", $1, p, $1, s);}' > $destdir/spk_map

if [ ! -f $srcdir/utt2uniq ]; then
  if [[ ! -z $utt_prefix  ||  ! -z $utt_suffix ]]; then
    cat $srcdir/utt2spk | awk -v p=$utt_prefix -v s=$utt_suffix '{printf("%s%s%s %s\n", p, $1, s, $1);}' > $destdir/utt2uniq
  fi
else
  cat $srcdir/utt2uniq | awk -v p=$utt_prefix -v s=$utt_suffix '{printf("%s%s%s %s\n", p, $1, s, $2);}' > $destdir/utt2uniq
fi

cat $srcdir/utt2spk | utils/apply_map.pl -f 1 $destdir/utt_map  | \
  utils/apply_map.pl -f 2 $destdir/spk_map >$destdir/utt2spk

utils/utt2spk_to_spk2utt.pl <$destdir/utt2spk >$destdir/spk2utt

if [ -f $srcdir/feats.scp ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/feats.scp >$destdir/feats.scp
fi

if [ -f $srcdir/vad.scp ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/vad.scp >$destdir/vad.scp
fi

if [ -f $srcdir/segments ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/segments >$destdir/segments
  cp $srcdir/wav.scp $destdir
else # no segments->wav indexed by utt.
  if [ -f $srcdir/wav.scp ]; then
    utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/wav.scp >$destdir/wav.scp
  fi
fi

if [ -f $srcdir/reco2file_and_channel ]; then
  cp $srcdir/reco2file_and_channel $destdir/
fi

if [ -f $srcdir/text ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/text >$destdir/text
fi
if [ -f $srcdir/utt2dur ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/utt2dur >$destdir/utt2dur
fi
if [ -f $srcdir/utt2num_frames ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/utt2num_frames >$destdir/utt2num_frames
fi
if [ -f $srcdir/reco2dur ]; then
  if [ -f $srcdir/segments ]; then
    cp $srcdir/reco2dur $destdir/reco2dur
  else
    utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/reco2dur >$destdir/reco2dur
  fi
fi
if [ -f $srcdir/spk2gender ]; then
  utils/apply_map.pl -f 1 $destdir/spk_map <$srcdir/spk2gender >$destdir/spk2gender
fi
if [ -f $srcdir/cmvn.scp ]; then
  utils/apply_map.pl -f 1 $destdir/spk_map <$srcdir/cmvn.scp >$destdir/cmvn.scp
fi
for f in frame_shift stm glm ctm; do
  if [ -f $srcdir/$f ]; then
    cp $srcdir/$f $destdir
  fi
done

rm $destdir/spk_map $destdir/utt_map

echo "$0: copied data from $srcdir to $destdir"

for f in feats.scp cmvn.scp vad.scp utt2lang utt2uniq utt2dur utt2num_frames text wav.scp reco2file_and_channel frame_shift stm glm ctm; do
  if [ -f $destdir/$f ] && [ ! -f $srcdir/$f ]; then
    echo "$0: file $f exists in dest $destdir but not in src $srcdir.  Moving it to"
    echo " ... $destdir/.backup/$f"
    mkdir -p $destdir/.backup
    mv $destdir/$f $destdir/.backup/
  fi
done


[ ! -f $srcdir/feats.scp ] && validate_opts="$validate_opts --no-feats"
[ ! -f $srcdir/text ] && validate_opts="$validate_opts --no-text"

utils/validate_data_dir.sh $validate_opts $destdir
