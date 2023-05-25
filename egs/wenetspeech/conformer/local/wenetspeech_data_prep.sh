#!/usr/bin/env bash

# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)
#                 Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -o pipefail

stage=1
prefix=
train_subset=L

. ./tools/parse_options.sh || exit 1;

filter_by_id () {
  idlist=$1
  input=$2
  output=$3
  field=1
  if [ $# -eq 4 ]; then
    field=$4
  fi
  cat $input | perl -se '
    open(F, "<$idlist") || die "Could not open id-list file $idlist";
    while(<F>) {
      @A = split;
      @A>=1 || die "Invalid id-list file line $_";
      $seen{$A[0]} = 1;
    }
    while(<>) {
      @A = split;
      @A > 0 || die "Invalid file line $_";
      @A >= $field || die "Invalid file line $_";
      if ($seen{$A[$field-1]}) {
        print $_;
      }
    }' -- -idlist="$idlist" -field="$field" > $output ||\
  (echo "$0: filter_by_id() error: $input" && exit 1) || exit 1;
}

subset_data_dir () {
  utt_list=$1
  src_dir=$2
  dest_dir=$3
  mkdir -p $dest_dir || exit 1;
  # wav.scp text segments utt2dur
  filter_by_id $utt_list $src_dir/utt2dur $dest_dir/utt2dur ||\
    (echo "$0: subset_data_dir() error: $src_dir/utt2dur" && exit 1) || exit 1;
  filter_by_id $utt_list $src_dir/text $dest_dir/text ||\
    (echo "$0: subset_data_dir() error: $src_dir/text" && exit 1) || exit 1;
  filter_by_id $utt_list $src_dir/segments $dest_dir/segments ||\
    (echo "$0: subset_data_dir() error: $src_dir/segments" && exit 1) || exit 1;
  awk '{print $2}' $dest_dir/segments | sort | uniq > $dest_dir/reco
  filter_by_id $dest_dir/reco $src_dir/wav.scp $dest_dir/wav.scp ||\
    (echo "$0: subset_data_dir() error: $src_dir/wav.scp" && exit 1) || exit 1;
  rm -f $dest_dir/reco
}

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <wenetspeech-dataset-dir> <data-dir>"
  echo " e.g.: $0 --train-subset L /disk1/audio_data/wenetspeech/ data/"
  echo ""
  echo "This script takes the WenetSpeech source directory, and prepares the"
  echo "WeNet format data directory."
  echo "  --prefix <prefix>                # Prefix for output data directory."
  echo "  --stage <stage>                  # Processing stage."
  echo "  --train-subset <L|M|S|W>     # Train subset to be created."
  exit 1
fi

wenetspeech_dir=$1
data_dir=$2

declare -A subsets
subsets=(
  [L]="train_l"
  [M]="train_m"
  [S]="train_s"
  [W]="train_w"
  [DEV]="dev"
  [TEST_NET]="test_net"
  [TEST_MEETING]="test_meeting")

prefix=${prefix:+${prefix}_}

corpus_dir=$data_dir/${prefix}corpus/
if [ $stage -le 1 ]; then
  echo "$0: Extract meta into $corpus_dir"
  # Sanity check.
  [ ! -f $wenetspeech_dir/WenetSpeech.json ] &&\
    echo "$0: Please download $wenetspeech_dir/WenetSpeech.json!" && exit 1;
  [ ! -d $wenetspeech_dir/audio ] &&\
    echo "$0: Please download $wenetspeech_dir/audio!" && exit 1;

  [ ! -d $corpus_dir ] && mkdir -p $corpus_dir

  # Files to be created:
  # wav.scp text segments utt2dur
  python3 local/extract_meta.py \
    $wenetspeech_dir/WenetSpeech.json $corpus_dir || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: Split data to train, dev, test_net, and test_meeting"
  [ ! -f $corpus_dir/utt2subsets ] &&\
    echo "$0: No such file $corpus_dir/utt2subsets!" && exit 1;
  for label in $train_subset DEV TEST_NET TEST_MEETING; do
    if [ ! ${subsets[$label]+set} ]; then
      echo "$0: Subset $label is not defined in WenetSpeech.json." && exit 1;
    fi
    subset=${subsets[$label]}
    [ ! -d $data_dir/${prefix}$subset ] && mkdir -p $data_dir/${prefix}$subset
    cat $corpus_dir/utt2subsets | \
       awk -v s=$label '{for (i=2;i<=NF;i++) if($i==s) print $0;}' \
       > $corpus_dir/${prefix}${subset}_utt_list|| exit 1;
    subset_data_dir $corpus_dir/${prefix}${subset}_utt_list \
      $corpus_dir $data_dir/${prefix}$subset || exit 1;
  done
fi

echo "$0: Done"
