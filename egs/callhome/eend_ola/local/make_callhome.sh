#!/bin/bash
# Copyright 2017   David Snyder
# Apache 2.0.
#
# This script prepares the Callhome portion of the NIST SRE 2000
# corpus (LDC2001S97). It is the evaluation dataset used in the
# callhome_diarization recipe.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <callhome-speech> <out-data-dir>"
  echo "e.g.: $0 /mnt/data/LDC2001S97 data/"
  exit 1;
fi

src_dir=$1
data_dir=$2

tmp_dir=$data_dir/callhome/.tmp/
mkdir -p $tmp_dir

# Download some metadata that wasn't provided in the LDC release
if [ ! -d "$tmp_dir/sre2000-key" ]; then
  wget --no-check-certificate -P $tmp_dir/ \
    http://www.openslr.org/resources/10/sre2000-key.tar.gz
  tar -xvf $tmp_dir/sre2000-key.tar.gz -C $tmp_dir/
fi

# The list of 500 recordings
awk '{print $1}' $tmp_dir/sre2000-key/reco2num > $tmp_dir/reco.list

# Create wav.scp file
count=0
missing=0
while read reco; do
  path=$(find $src_dir -name "$reco.sph")
  if [ -z "${path// }" ]; then
    >&2 echo "$0: Missing Sphere file for $reco"
    missing=$((missing+1))
  else
    echo "$reco sph2pipe -f wav -p $path |"
  fi
  count=$((count+1))
done < $tmp_dir/reco.list > $data_dir/callhome/wav.scp

if [ $missing -gt 0 ]; then
  echo "$0: Missing $missing out of $count recordings"
fi

cp $tmp_dir/sre2000-key/segments $data_dir/callhome/
awk '{print $1, $2}' $data_dir/callhome/segments > $data_dir/callhome/utt2spk
utils/utt2spk_to_spk2utt.pl $data_dir/callhome/utt2spk > $data_dir/callhome/spk2utt
cp $tmp_dir/sre2000-key/reco2num $data_dir/callhome/reco2num_spk
cp $tmp_dir/sre2000-key/fullref.rttm $data_dir/callhome/

utils/validate_data_dir.sh --no-text --no-feats $data_dir/callhome
utils/fix_data_dir.sh $data_dir/callhome

utils/copy_data_dir.sh $data_dir/callhome $data_dir/callhome1
utils/copy_data_dir.sh $data_dir/callhome $data_dir/callhome2

utils/shuffle_list.pl $data_dir/callhome/wav.scp | head -n 250 \
  | utils/filter_scp.pl - $data_dir/callhome/wav.scp \
  > $data_dir/callhome1/wav.scp
utils/fix_data_dir.sh $data_dir/callhome1
utils/filter_scp.pl --exclude $data_dir/callhome1/wav.scp \
  $data_dir/callhome/wav.scp > $data_dir/callhome2/wav.scp
utils/fix_data_dir.sh $data_dir/callhome2
utils/filter_scp.pl $data_dir/callhome1/wav.scp $data_dir/callhome/reco2num_spk \
  > $data_dir/callhome1/reco2num_spk
utils/filter_scp.pl $data_dir/callhome2/wav.scp $data_dir/callhome/reco2num_spk \
  > $data_dir/callhome2/reco2num_spk

rm -rf $tmp_dir 2> /dev/null
