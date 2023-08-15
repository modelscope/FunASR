#!/usr/bin/env bash

set -e
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# general configuration
stage=1
stop_stage=3
nj=10

log "$0 $*"
. utils/parse_options.sh
. ./path.sh || exit 1
train_cmd=utils/run.pl


data_source_dir=$DATA_SOURCE 
audio_dir=$data_source_dir/audio_dir
output_wpe_dir=$data_source_dir/wpe_audio_dir
output_gss_dir=$data_source_dir/gss_audio_dir
asr_data_path=./data/${DATA_NAME}_wpegss
channel=$1

log "Start Speech Enhancement."

if [ ! -L ./utils ]; then
    ln -s ./pb_chime5/pb_bss
fi

# WPE
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Start WPE."
    for ch in `seq ${channel}`; do
        mkdir -p  ${output_wpe_dir}_${ch}/log/
        # split wav.scp
        find $audio_dir/ -name "*.wav" > ${output_wpe_dir}_${ch}/wav.scp
        arr=""
        for i in `seq ${nj}`; do
            arr="$arr ${output_wpe_dir}_${ch}/log/wav.${i}.scp"
        done
        split_scp.pl ${output_wpe_dir}_${ch}/wav.scp $arr
        # do wpe
        for n in `seq ${nj}`; do
            cat <<-EOF >${output_wpe_dir}_${ch}/log/wpe.${n}.sh
python local/run_wpe.py \
    --wav-scp ${output_wpe_dir}_${ch}/log/wav.${n}.scp \
    --audio-dir ${audio_dir} \
    --output-dir ${output_wpe_dir}_${ch} \
    --ch $ch
EOF
        done
        chmod a+x ${output_wpe_dir}_${ch}/log/wpe.*.sh
        ${train_cmd} JOB=1:${nj} ${output_wpe_dir}_${ch}/log/wpe.JOB.log \
            ${output_wpe_dir}_${ch}/log/wpe.JOB.sh
    done
fi

# GSS
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Start GSS"
    if [ ! -d pb_chime5/ ]; then
        log "Please install pb_chime5 by local/install_pb_chime5.sh"
        exit 1
    fi
    mkdir -p $output_gss_dir/log
    # split wpe.scp
    for i in `seq ${channel}`; do
        find ${output_wpe_dir}_${i}/ -name "*.wav" > $output_gss_dir/tmp${i}
    done
    awk -F '/' '{print($NF)}' $output_gss_dir/tmp1 | cut -d "." -f1 > $output_gss_dir/tmp
    arr=""
    for i in `seq ${channel}`; do
        arr="$arr $output_gss_dir/tmp${i}"
    done
    paste -d " " $output_gss_dir/tmp $arr > $output_gss_dir/wpe.scp
    rm -f $output_gss_dir/tmp*
    arr=""  
    for i in `seq ${nj}`; do
        arr="$arr $output_gss_dir/log/wpe.${i}.scp"
    done
    split_scp.pl $output_gss_dir/wpe.scp $arr

    # do gss
    for n in `seq ${nj}`; do
        cat <<-EOF >${output_gss_dir}/log/gss.${n}.sh
python local/run_gss.py \
    --wav-scp ${output_gss_dir}/log/wpe.${n}.scp \
    --segments $asr_data_path/org/segments \
    --output-dir ${output_gss_dir}
EOF
    done
    chmod a+x ${output_gss_dir}/log/gss.*.sh
    ${train_cmd} JOB=1:${nj} ${output_gss_dir}/log/gss.JOB.log \
        ${output_gss_dir}/log/gss.JOB.sh
fi

# Prepare data for ASR
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Preparing data for ASR"
    find $output_gss_dir -name "*.wav" > $asr_data_path/org/wav_list
    awk -F '/' '{print($NF)}' $asr_data_path/org/wav_list | sed 's/\.wav//g' > $asr_data_path/org/uttid
    paste -d " " $asr_data_path/org/uttid $asr_data_path/org/wav_list > $asr_data_path/org/wav.scp
    bash local/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        --audio-format wav --segments $asr_data_path/org/segments \
        "$asr_data_path/org/wav.scp" "$asr_data_path"
fi

log "End speech enhancement"
