#!/usr/bin/env bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriSpeech/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

# all utterances are FLAC compressed
if ! which flac >&/dev/null; then
   echo "Please install 'flac' on ALL worker nodes!"
   exit 1
fi

spk_file=$src/../SPEAKERS.TXT

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1
[ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans

for reader_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sort); do
  reader=$(basename $reader_dir)
  if ! [ $reader -eq $reader ]; then  # not integer.
    echo "$0: unexpected subdirectory name $reader"
    exit 1
  fi

  for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    chapter=$(basename $chapter_dir)
    if ! [ "$chapter" -eq "$chapter" ]; then
      echo "$0: unexpected chapter-subdirectory name $chapter"
      exit 1
    fi

    find -L $chapter_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac | \
      awk -v "dir=$chapter_dir" '{printf "%s %s/%s.flac \n", $0, dir, $0}' >>$wav_scp|| exit 1

    chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
    [ ! -f  $chapter_trans ] && echo "$0: expected file $chapter_trans to exist" && exit 1
    cat $chapter_trans >>$trans
  done
done

echo "$0: successfully prepared data in $dst"

exit 0
