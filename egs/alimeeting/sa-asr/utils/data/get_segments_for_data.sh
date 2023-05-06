#!/usr/bin/env bash

# This script operates on a data directory, such as in data/train/,
# and writes new segments to stdout. The file 'segments' maps from
# utterance to time offsets into a recording, with the format:
#   <utterance-id> <recording-id> <segment-begin> <segment-end>
# This script assumes utterance and recording ids are the same (i.e., that
# wav.scp is indexed by utterance), and uses durations from 'utt2dur', 
# created if necessary by get_utt2dur.sh.

. ./path.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <datadir>"
  echo "e.g.:"
  echo " $0 data/train > data/train/segments"
  exit 1
fi

data=$1

if [ ! -s $data/utt2dur ]; then
  utils/data/get_utt2dur.sh $data 1>&2 || exit 1;
fi

# <utt-id> <utt-id> 0 <utt-dur>
awk '{ print $1, $1, 0, $2 }' $data/utt2dur

exit 0
