#!/usr/bin/env bash

# 2020 @kamo-naoyuki
# This file was copied from Kaldi and 
# I deleted parts related to wav duration 
# because we shouldn't use kaldi's command here
# and we don't need the files actually.

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
#           2014  Tom Ko
#           2018  Emotech LTD (author: Pawel Swietojanski)
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  wav.scp
#  spk2utt
#  utt2spk
#  text
#
# It generates the files which are used for perturbing the speed of the original data.

export LC_ALL=C
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: perturb_data_dir_speed.sh <warping-factor> <srcdir> <destdir>"
    echo "e.g.:"
    echo " $0 0.9 data/train_si284 data/train_si284p"
    exit 1
fi

factor=$1
srcdir=$2
destdir=$3
label="sp"
spk_prefix="${label}${factor}-"
utt_prefix="${label}${factor}-"

#check is sox on the path

! command -v sox &>/dev/null && echo "sox: command not found" && exit 1;

if [[ ! -f ${srcdir}/utt2spk ]]; then
  echo "$0: no such file ${srcdir}/utt2spk"
  exit 1;
fi

if [[ ${destdir} == "${srcdir}" ]]; then
  echo "$0: this script requires <srcdir> and <destdir> to be different."
  exit 1
fi

mkdir -p "${destdir}"

<"${srcdir}"/utt2spk awk -v p="${utt_prefix}" '{printf("%s %s%s\n", $1, p, $1);}' > "${destdir}/utt_map"
<"${srcdir}"/spk2utt awk -v p="${spk_prefix}" '{printf("%s %s%s\n", $1, p, $1);}' > "${destdir}/spk_map"
<"${srcdir}"/wav.scp awk -v p="${spk_prefix}" '{printf("%s %s%s\n", $1, p, $1);}' > "${destdir}/reco_map"
if [[ ! -f ${srcdir}/utt2uniq ]]; then
    <"${srcdir}/utt2spk" awk -v p="${utt_prefix}" '{printf("%s%s %s\n", p, $1, $1);}' > "${destdir}/utt2uniq"
else
    <"${srcdir}/utt2uniq" awk -v p="${utt_prefix}" '{printf("%s%s %s\n", p, $1, $2);}' > "${destdir}/utt2uniq"
fi


<"${srcdir}"/utt2spk utils/apply_map.pl -f 1 "${destdir}"/utt_map | \
  utils/apply_map.pl -f 2 "${destdir}"/spk_map >"${destdir}"/utt2spk

utils/utt2spk_to_spk2utt.pl <"${destdir}"/utt2spk >"${destdir}"/spk2utt

if [[ -f ${srcdir}/segments ]]; then

  utils/apply_map.pl -f 1 "${destdir}"/utt_map <"${srcdir}"/segments | \
      utils/apply_map.pl -f 2 "${destdir}"/reco_map | \
          awk -v factor="${factor}" \
            '{s=$3/factor; e=$4/factor; if (e > s + 0.01) { printf("%s %s %.2f %.2f\n", $1, $2, $3/factor, $4/factor);} }' \
            >"${destdir}"/segments

  utils/apply_map.pl -f 1 "${destdir}"/reco_map <"${srcdir}"/wav.scp | sed 's/| *$/ |/' | \
      # Handle three cases of rxfilenames appropriately; "input piped command", "file offset" and "filename"
      awk -v factor="${factor}" \
          '{wid=$1; $1=""; if ($NF=="|") {print wid $_ " sox -t wav - -t wav - speed " factor " |"}
            else if (match($0, /:[0-9]+$/)) {print wid " wav-copy" $_ " - | sox -t wav - -t wav - speed " factor " |" }
            else  {print wid " sox" $_ " -t wav - speed " factor " |"}}' \
             > "${destdir}"/wav.scp
  if [[ -f ${srcdir}/reco2file_and_channel ]]; then
      utils/apply_map.pl -f 1 "${destdir}"/reco_map \
       <"${srcdir}"/reco2file_and_channel >"${destdir}"/reco2file_and_channel
  fi

else # no segments->wav indexed by utterance.
    if [[ -f ${srcdir}/wav.scp ]]; then
        utils/apply_map.pl -f 1 "${destdir}"/utt_map <"${srcdir}"/wav.scp | sed 's/| *$/ |/' | \
         # Handle three cases of rxfilenames appropriately; "input piped command", "file offset" and "filename"
         awk -v factor="${factor}" \
           '{wid=$1; $1=""; if ($NF=="|") {print wid $_ " sox -t wav - -t wav - speed " factor " |"}
             else if (match($0, /:[0-9]+$/)) {print wid " wav-copy" $_ " - | sox -t wav - -t wav - speed " factor " |" }
             else {print wid " sox" $_ " -t wav - speed " factor " |"}}' \
                 > "${destdir}"/wav.scp
    fi
fi

if [[ -f ${srcdir}/text ]]; then
    utils/apply_map.pl -f 1 "${destdir}"/utt_map <"${srcdir}"/text >"${destdir}"/text
fi
if [[ -f ${srcdir}/spk2gender ]]; then
    utils/apply_map.pl -f 1 "${destdir}"/spk_map <"${srcdir}"/spk2gender >"${destdir}"/spk2gender
fi
if [[ -f ${srcdir}/utt2lang ]]; then
    utils/apply_map.pl -f 1 "${destdir}"/utt_map <"${srcdir}"/utt2lang >"${destdir}"/utt2lang
fi

rm "${destdir}"/spk_map "${destdir}"/utt_map "${destdir}"/reco_map 2>/dev/null
echo "$0: generated speed-perturbed version of data in ${srcdir}, in ${destdir}"

utils/validate_data_dir.sh --no-feats --no-text "${destdir}"
