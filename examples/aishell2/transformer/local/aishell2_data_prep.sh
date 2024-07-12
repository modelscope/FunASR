#!/bin/bash

# Copyright 2017 Xingyu Na
# Apache 2.0

#. ./path.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <train-audio-path> <train-text-path> <dev&test-path> <output-path>"
  echo " $0 /export/a05/xna/data/data_aishell2/wav /export/a05/xna/data/data_aishell2/transcript data"
  exit 1;
fi

# define the audio and text data directories
aishell2_train_audio_dir=$1
aishell2_train_text=$2/trans.txt

aishell2_Android_dev_audio_dir=$3/Android/dev/wav
aishell2_Android_dev_text=$3/Android/dev/trans.txt

aishell2_Android_test_audio_dir=$3/Android/test/wav
aishell2_Android_test_text=$3/Android/test/trans.txt

aishell2_IOS_dev_audio_dir=$3/IOS/dev/wav
aishell2_IOS_dev_text=$3/IOS/dev/trans.txt

aishell2_IOS_test_audio_dir=$3/IOS/test/wav
aishell2_IOS_test_text=$3/IOS/test/trans.txt

aishell2_MIC_dev_audio_dir=$3/MIC/dev/wav
aishell2_MIC_dev_text=$3/MIC/dev/trans.txt

aishell2_MIC_test_audio_dir=$3/MIC/test/wav
aishell2_MIC_test_text=$3/MIC/test/trans.txt

output_dir=$4


# make the output directory and subdirectories
train_dir=$output_dir/data/local/train

dev_dir=$output_dir/data/local/dev
dev_IOS_dir=$dev_dir/IOS
dev_Android_dir=$dev_dir/Android
dev_MIC_dir=$dev_dir/MIC

test_dir=$output_dir/data/local/test
test_IOS_dir=$test_dir/IOS
test_Android_dir=$test_dir/Android
test_MIC_dir=$test_dir/MIC

tmp_dir=$output_dir/data/local/tmp

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir
mkdir -p $tmp_dir
mkdir -p $dev_IOS_dir
mkdir -p $dev_Android_dir
mkdir -p $dev_MIC_dir
mkdir -p $test_IOS_dir
mkdir -p $test_Android_dir
mkdir -p $test_MIC_dir

# data directory check
if [ ! -d $aishell2_train_audio_dir ] || [ ! -f $aishell2_train_text ]; then
  echo "Error: $0.1 requires two directory arguments"
  exit 1;
fi

if [ ! -d $aishell2_Android_dev_audio_dir ] || [ ! -f $aishell2_Android_dev_text ]; then
  echo "Error: $0.2 requires two directory arguments"
  exit 1;
fi

if [ ! -d $aishell2_Android_test_audio_dir ] || [ ! -f $aishell2_Android_test_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

if [ ! -d $aishell2_IOS_dev_audio_dir ] || [ ! -f $aishell2_IOS_dev_text ]; then
  echo "Error: $0.3 requires two directory arguments"
  exit 1;
fi

if [ ! -d $aishell2_IOS_test_audio_dir ] || [ ! -f $aishell2_IOS_test_text ]; then
  echo "Error: $0.4 requires two directory arguments"
  exit 1;
fi

if [ ! -d $aishell2_MIC_dev_audio_dir ] || [ ! -f $aishell2_MIC_dev_text ]; then
  echo "Error: $0.5 requires two directory arguments"
  exit 1;
fi

if [ ! -d $aishell2_MIC_test_audio_dir ] || [ ! -f $aishell2_MIC_test_text ]; then
  echo "Error: $0.6 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $aishell2_train_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
find $aishell2_Android_dev_audio_dir -iname "*.wav" >> $tmp_dir/wav.flist
find $aishell2_Android_test_audio_dir -iname "*.wav" >> $tmp_dir/wav.flist
find $aishell2_IOS_dev_audio_dir -iname "*.wav" >> $tmp_dir/wav.flist
find $aishell2_IOS_test_audio_dir -iname "*.wav" >> $tmp_dir/wav.flist
find $aishell2_MIC_dev_audio_dir -iname "*.wav" >> $tmp_dir/wav.flist
find $aishell2_MIC_test_audio_dir -iname "*.wav" >> $tmp_dir/wav.flist

# statistics
n=`cat $tmp_dir/wav.flist | wc -l`
echo Totall wav files found $n

echo "find wav for $train_dir/wav.flist"
grep -i "data_aishell2/wav/" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
echo "find wav for $dev_IOS_dir/wav.flist"
grep -i "IOS/dev/wav/" $tmp_dir/wav.flist > $dev_IOS_dir/wav.flist || exit 1;
echo "find wav for $dev_Android_dir/wav.flist"
grep -i "Android/dev/wav/" $tmp_dir/wav.flist > $dev_Android_dir/wav.flist || exit 1;
echo "find wav for $dev_MIC_dir/wav.flist"
grep -i "MIC/dev/wav/" $tmp_dir/wav.flist > $dev_MIC_dir/wav.flist || exit 1;
echo "find wav for $test_IOS_dir/wav.flist"
grep -i "IOS/test/wav" $tmp_dir/wav.flist > $test_IOS_dir/wav.flist || exit 1;
echo "find wav for $test_Android_dir/wav.flist"
grep -i "Android/test/wav"/ $tmp_dir/wav.flist > $test_Android_dir/wav.flist || exit 1;
echo "find wav for $test_MIC_dir/wav.flist"
grep -i "MIC/test/wav/" $tmp_dir/wav.flist > $test_MIC_dir/wav.flist || exit 1;

rm -r $tmp_dir

# Transcriptions preparation
dir=$train_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
# 使用 sed 命令从音频文件列表中移除 .wav 扩展名。
# 使用 awk 处理路径，只保留最后的文件名（即 utterance ID），并将结果保存到 $dir/utt.list。
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
# 使用 paste 命令合并 utt.list 和 wav.flist 文件，创建 wav.scp_all 文件，这个文件将话语 ID 与对应的完整音频文件路径关联起来。
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_train_text > $dir/transcripts.txt
# 使用 filter_scp.pl 脚本过滤转录文件，只保留当前目录中存在的话语 ID 的转录。这需要 $aishell2_train_text 文件包含完整的转录。
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
# 更新 utt.list 文件，使其只包含当前数据集中存在的话语标识符。
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
# 再次使用 filter_scp.pl 脚本，这次是用来从 wav.scp_all 过滤出存在于更新的 utt.list 中的条目，确保 wav.scp 文件中的条目是唯一的且与转录相对应。
sort -u $dir/transcripts.txt > $dir/text


dir=$dev_IOS_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_IOS_dev_text > $dir/transcripts.txt
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
sort -u $dir/transcripts.txt > $dir/text


dir=$dev_Android_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_Android_dev_text > $dir/transcripts.txt
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
sort -u $dir/transcripts.txt > $dir/text

dir=$dev_MIC_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_MIC_dev_text > $dir/transcripts.txt
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
sort -u $dir/transcripts.txt > $dir/text

dir=$test_IOS_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_IOS_test_text > $dir/transcripts.txt
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
sort -u $dir/transcripts.txt > $dir/text


dir=$test_Android_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_Android_test_text > $dir/transcripts.txt
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
sort -u $dir/transcripts.txt > $dir/text

dir=$test_MIC_dir
echo Preparing $dir transcriptions
sed -e 's/\.wav//'  $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
utils/filter_scp.pl -f 1 $dir/utt.list $aishell2_MIC_test_text > $dir/transcripts.txt
awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
sort -u $dir/transcripts.txt > $dir/text


mkdir -p $output_dir/data/train $output_dir/data/dev/IOS $output_dir/data/dev/Android $output_dir/data/dev/MIC $output_dir/data/test/IOS $output_dir/data/test/Android $output_dir/data/test/MIC

for f in wav.scp text; do
  cp $train_dir/$f $output_dir/data/train/$f || exit 1;
  cp $dev_IOS_dir/$f $output_dir/data/dev/IOS/$f || exit 1;
  cp $dev_Android_dir/$f $output_dir/data/dev/Android/$f || exit 1;
  cp $dev_MIC_dir/$f $output_dir/data/dev/MIC/$f || exit 1;
  cp $test_IOS_dir/$f $output_dir/data/test/IOS/$f || exit 1;
  cp $test_Android_dir/$f $output_dir/data/test/Android/$f || exit 1;
  cp $test_MIC_dir/$f $output_dir/data/test/MIC/$f || exit 1;
done

echo "$0: aishell2 data preparation succeeded"
exit 0;
