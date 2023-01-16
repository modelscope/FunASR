#!/usr/bin/env bash

echo "$0 $@"
data_dir=$1

if [ ! -f ${data_dir}/wav.scp ]; then
  echo "$0: wav.scp is not found"
  exit 1;
fi

if [ ! -f ${data_dir}/text ]; then
  echo "$0: text is not found"
  exit 1;
fi



mkdir -p ${data_dir}/.backup

awk '{print $1}' ${data_dir}/wav.scp > ${data_dir}/.backup/wav_id
awk '{print $1}' ${data_dir}/text > ${data_dir}/.backup/text_id

sort ${data_dir}/.backup/wav_id ${data_dir}/.backup/text_id | uniq -d > ${data_dir}/.backup/id

cp ${data_dir}/wav.scp ${data_dir}/.backup/wav.scp
cp ${data_dir}/text ${data_dir}/.backup/text

mv ${data_dir}/wav.scp ${data_dir}/wav.scp.bak
mv ${data_dir}/text ${data_dir}/text.bak

utils/filter_scp.pl -f 1 ${data_dir}/.backup/id ${data_dir}/wav.scp.bak | sort -k1,1 -u > ${data_dir}/wav.scp
utils/filter_scp.pl -f 1 ${data_dir}/.backup/id ${data_dir}/text.bak | sort -k1,1 -u > ${data_dir}/text

rm ${data_dir}/wav.scp.bak
rm ${data_dir}/text.bak
