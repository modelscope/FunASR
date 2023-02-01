#!/usr/bin/env bash

echo "$0 $@"
data_dir=$1

if [ ! -f ${data_dir}/feats.scp ]; then
  echo "$0: feats.scp is not found"
  exit 1;
fi

if [ ! -f ${data_dir}/text ]; then
  echo "$0: text is not found"
  exit 1;
fi

if [ ! -f ${data_dir}/speech_shape ]; then
  echo "$0: feature lengths is not found"
  exit 1;
fi

if [ ! -f ${data_dir}/text_shape ]; then
  echo "$0: text lengths is not found"
  exit 1;
fi

mkdir -p ${data_dir}/.backup

awk '{print $1}' ${data_dir}/feats.scp > ${data_dir}/.backup/wav_id
awk '{print $1}' ${data_dir}/text > ${data_dir}/.backup/text_id

sort ${data_dir}/.backup/wav_id ${data_dir}/.backup/text_id | uniq -d > ${data_dir}/.backup/id

cp ${data_dir}/feats.scp ${data_dir}/.backup/feats.scp
cp ${data_dir}/text ${data_dir}/.backup/text
cp ${data_dir}/speech_shape ${data_dir}/.backup/speech_shape
cp ${data_dir}/text_shape ${data_dir}/.backup/text_shape

mv ${data_dir}/feats.scp ${data_dir}/feats.scp.bak
mv ${data_dir}/text ${data_dir}/text.bak
mv ${data_dir}/speech_shape ${data_dir}/speech_shape.bak
mv ${data_dir}/text_shape ${data_dir}/text_shape.bak

utils/filter_scp.pl -f 1 ${data_dir}/.backup/id ${data_dir}/feats.scp.bak | sort -k1,1 -u > ${data_dir}/feats.scp
utils/filter_scp.pl -f 1 ${data_dir}/.backup/id ${data_dir}/text.bak | sort -k1,1 -u > ${data_dir}/text
utils/filter_scp.pl -f 1 ${data_dir}/.backup/id ${data_dir}/speech_shape.bak | sort -k1,1 -u > ${data_dir}/speech_shape
utils/filter_scp.pl -f 1 ${data_dir}/.backup/id ${data_dir}/text_shape.bak | sort -k1,1 -u > ${data_dir}/text_shape

rm ${data_dir}/feats.scp.bak
rm ${data_dir}/text.bak
rm ${data_dir}/speech_shape.bak
rm ${data_dir}/text_shape.bak

