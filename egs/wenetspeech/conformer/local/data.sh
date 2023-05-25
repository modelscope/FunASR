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
SECONDS=0

# general configuration
nj=10
stage=1
stop_stage=100
set=L
data_dir="data"

log "$0 $*"
. utils/parse_options.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ ! -e "${WENETSPEECH}" ]; then
    log "Fill the value of 'WENETSPEECH' of db.sh"
    log "or download the data set follwing the instruction in https://wenet-e2e.github.io/WenetSpeech/"
    exit 1
fi

if [ ! -d "${WENETSPEECH}/audio" ] && [ ! -f "${WENETSPEECH}/WenetSpeech.json" ]; then
    echo "Valid WENETSPEECH data not found in ${WENETSPEECH}."
    echo "Please follow the instruction in https://wenet-e2e.github.io/WenetSpeech/"
    echo "and re-construct the data."
    exit 1
fi

train_set=train_"$(echo "${set}" | tr "[:upper:]" "[:lower:]")"
dev_set=dev
test_sets="test_net test_meeting"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "data preparation"
    mkdir -p ${data_dir}
    abs_data_dir=$(readlink -f ${data_dir})
    log "making Kaldi format data directory in ${abs_data_dir}"
    local/wenetspeech_data_prep.sh \
        --train-subset ${set} \
        --stage 1 \
        ${WENETSPEECH} \
        ${abs_data_dir}

    # prepare utt2spk and spk2utt files
    for x in ${train_set} ${dev_set} ${test_sets}; do
        dir=${data_dir}/${x}
        paste -d " " <(cut -f 1 ${dir}/segments) <(cut -f 1 ${dir}/segments) | \
            sort -u > ${dir}/utt2spk
        utils/utt2spk_to_spk2utt.pl ${dir}/utt2spk > ${dir}/spk2utt
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "process the long term opus audio file, may take about 3 hours"
    for x in ${train_set} ${dev_set} ${test_sets}; do
        log "process audio for ${data_dir}/${x}"
        dir=${data_dir}/${x}
        mkdir -p ${dir}/logs

        nutt=$(<${dir}/segments wc -l)
        nj=$((nj<nutt?nj:nutt))

        split_scps=""
        for n in $(seq ${nj}); do
            split_scps="${split_scps} ${dir}/logs/segments.${n}"
        done
        utils/split_scp.pl ${dir}/segments ${split_scps}

        ${train_cmd} "JOB=1:${nj}" "${dir}/logs/process_audio.JOB.log"\
            python3 local/process_opus.py \
                ${dir}/wav.scp \
                ${dir}/logs/segments.JOB   \
                ${dir}/logs/wav.JOB.scp

        # modify the `wav.scp` file and rename the `segments` file
        # rename the `segments` file to avoid the audio file formatting process in stage 3 of `asr.sh`
        mv ${dir}/wav.scp ${dir}/wav.scp.org
        mv ${dir}/segments ${dir}/segments.org
        for n in $(seq ${nj}); do
            cat ${dir}/logs/wav.${n}.scp || exit 1;
        done | sort -u > ${dir}/wav.scp
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "format text file"
    for x in ${train_set} ${dev_set} ${test_sets}; do
        log "format text for ${data_dir}/${x}"
        dir=${data_dir}/${x}
        mv ${dir}/text ${dir}/text.org
        paste -d " " <(cut -f 1 ${dir}/text.org) \
            <(cut -f 2- ${dir}/text.org | local/text_normalize.pl) | \
            sort -u > ${dir}/text
        utils/fix_data_dir.sh ${dir}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
