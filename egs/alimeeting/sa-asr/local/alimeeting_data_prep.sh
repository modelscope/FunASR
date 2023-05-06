#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_messge=$(cat << EOF
Usage: $0

Options:
    --no_overlap (bool): Whether to ignore the overlapping utterance in the training set.
    --tgt (string): Which set to process, test or train.
EOF
)

SECONDS=0
tgt=Train #Train or Eval


log "$0 $*"
echo $tgt
. ./utils/parse_options.sh

. ./path.sh

AliMeeting="${PWD}/dataset"

if [ $# -gt 2 ]; then
    log "${help_message}"
    exit 2
fi


if [ ! -d "${AliMeeting}" ]; then
  log "Error: ${AliMeeting} is empty."
  exit 2
fi

# To absolute path
AliMeeting=$(cd ${AliMeeting}; pwd)
echo $AliMeeting
far_raw_dir=${AliMeeting}/${tgt}_Ali_far/
near_raw_dir=${AliMeeting}/${tgt}_Ali_near/

far_dir=data/local/${tgt}_Ali_far
near_dir=data/local/${tgt}_Ali_near
far_single_speaker_dir=data/local/${tgt}_Ali_far_correct_single_speaker
mkdir -p $far_single_speaker_dir

stage=1
stop_stage=4
mkdir -p $far_dir
mkdir -p $near_dir

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
    log "stage 1:process alimeeting near dir"
    
    find -L $near_raw_dir/audio_dir -iname "*.wav" >  $near_dir/wavlist
    awk -F '/' '{print $NF}' $near_dir/wavlist | awk -F '.' '{print $1}' > $near_dir/uttid   
    find -L $near_raw_dir/textgrid_dir  -iname "*.TextGrid" > $near_dir/textgrid.flist
    n1_wav=$(wc -l < $near_dir/wavlist)
    n2_text=$(wc -l < $near_dir/textgrid.flist)
    log  near file found $n1_wav wav and $n2_text text.

    paste $near_dir/uttid $near_dir/wavlist > $near_dir/wav_raw.scp

    # cat $near_dir/wav_raw.scp | awk '{printf("%s sox -t wav  %s -r 16000 -b 16 -c 1 -t wav  - |\n", $1, $2)}'  > $near_dir/wav.scp
    cat $near_dir/wav_raw.scp | awk '{printf("%s sox -t wav  %s -r 16000 -b 16 -t wav  - |\n", $1, $2)}'  > $near_dir/wav.scp
    
    python local/alimeeting_process_textgrid.py --path $near_dir --no-overlap False
    cat $near_dir/text_all | local/text_normalize.pl | local/text_format.pl | sort -u > $near_dir/text
    utils/filter_scp.pl -f 1 $near_dir/text $near_dir/utt2spk_all | sort -u > $near_dir/utt2spk
    #sed -e 's/ [a-z,A-Z,_,0-9,-]\+SPK/ SPK/'  $near_dir/utt2spk_old >$near_dir/tmp1
    #sed -e 's/-[a-z,A-Z,0-9]\+$//' $near_dir/tmp1 | sort -u > $near_dir/utt2spk
    local/utt2spk_to_spk2utt.pl $near_dir/utt2spk > $near_dir/spk2utt
    utils/filter_scp.pl -f 1 $near_dir/text $near_dir/segments_all | sort -u > $near_dir/segments
    sed -e 's/ $//g' $near_dir/text> $near_dir/tmp1
    sed -e 's/！//g' $near_dir/tmp1> $near_dir/tmp2
    sed -e 's/？//g' $near_dir/tmp2> $near_dir/text

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2:process alimeeting far dir"
    
    find -L $far_raw_dir/audio_dir -iname "*.wav" >  $far_dir/wavlist
    awk -F '/' '{print $NF}' $far_dir/wavlist | awk -F '.' '{print $1}' > $far_dir/uttid   
    find -L $far_raw_dir/textgrid_dir  -iname "*.TextGrid" > $far_dir/textgrid.flist
    n1_wav=$(wc -l < $far_dir/wavlist)
    n2_text=$(wc -l < $far_dir/textgrid.flist)
    log  far file found $n1_wav wav and $n2_text text.

    paste $far_dir/uttid $far_dir/wavlist > $far_dir/wav_raw.scp

    cat $far_dir/wav_raw.scp | awk '{printf("%s sox -t wav  %s -r 16000 -b 16 -t wav  - |\n", $1, $2)}'  > $far_dir/wav.scp

    python local/alimeeting_process_overlap_force.py  --path $far_dir \
        --no-overlap false --mars True \
        --overlap_length 0.8 --max_length 7

    cat $far_dir/text_all | local/text_normalize.pl | local/text_format.pl | sort -u > $far_dir/text
    utils/filter_scp.pl -f 1 $far_dir/text $far_dir/utt2spk_all | sort -u > $far_dir/utt2spk
    #sed -e 's/ [a-z,A-Z,_,0-9,-]\+SPK/ SPK/'  $far_dir/utt2spk_old >$far_dir/utt2spk
    
    local/utt2spk_to_spk2utt.pl $far_dir/utt2spk > $far_dir/spk2utt
    utils/filter_scp.pl -f 1 $far_dir/text $far_dir/segments_all | sort -u > $far_dir/segments
    sed -e 's/SRC/$/g' $far_dir/text> $far_dir/tmp1
    sed -e 's/ $//g' $far_dir/tmp1> $far_dir/tmp2
    sed -e 's/！//g' $far_dir/tmp2> $far_dir/tmp3
    sed -e 's/？//g' $far_dir/tmp3> $far_dir/text
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: finali data process"

    local/copy_data_dir.sh $near_dir data/${tgt}_Ali_near
    local/copy_data_dir.sh $far_dir data/${tgt}_Ali_far

    sort $far_dir/utt2spk_all_fifo > data/${tgt}_Ali_far/utt2spk_all_fifo
    sed -i "s/src/$/g" data/${tgt}_Ali_far/utt2spk_all_fifo

    # remove space in text
    for x in ${tgt}_Ali_near ${tgt}_Ali_far; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
        > data/${x}/text
        rm data/${x}/text.org
    done

    log "Successfully finished. [elapsed=${SECONDS}s]"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: process alimeeting far dir (single speaker by oracle time strap)"
    cp -r $far_dir/* $far_single_speaker_dir 
    mv $far_single_speaker_dir/textgrid.flist  $far_single_speaker_dir/textgrid_oldpath
    paste -d " " $far_single_speaker_dir/uttid $far_single_speaker_dir/textgrid_oldpath > $far_single_speaker_dir/textgrid.flist
    python local/process_textgrid_to_single_speaker_wav.py  --path $far_single_speaker_dir
    
    cp $far_single_speaker_dir/utt2spk $far_single_speaker_dir/text    
    local/utt2spk_to_spk2utt.pl $far_single_speaker_dir/utt2spk > $far_single_speaker_dir/spk2utt

    ./local/fix_data_dir.sh $far_single_speaker_dir 
    local/copy_data_dir.sh $far_single_speaker_dir data/${tgt}_Ali_far_single_speaker

    # remove space in text
    for x in ${tgt}_Ali_far_single_speaker; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
        > data/${x}/text
        rm data/${x}/text.org
    done
    log "Successfully finished. [elapsed=${SECONDS}s]"
fi