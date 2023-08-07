#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script, called by ../run.sh, creates the MUSAN
# data directory. The required dataset is freely available at
#   http://www.openslr.org/17/

set -e
in_dir=$1
data_dir=$2
use_vocals='Y'

mkdir -p local/musan.tmp

echo "Preparing ${data_dir}/musan..."
mkdir -p ${data_dir}/musan
local/make_musan.py ${in_dir} ${data_dir}/musan ${use_vocals}

utils/fix_data_dir.sh ${data_dir}/musan

grep "music" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_music
grep "speech" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_speech
grep "noise" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_noise
utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_music \
  ${data_dir}/musan ${data_dir}/musan_music
utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_speech \
  ${data_dir}/musan ${data_dir}/musan_speech
utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_noise \
  ${data_dir}/musan ${data_dir}/musan_noise

utils/fix_data_dir.sh ${data_dir}/musan_music
utils/fix_data_dir.sh ${data_dir}/musan_speech
utils/fix_data_dir.sh ${data_dir}/musan_noise

rm -rf local/musan.tmp

