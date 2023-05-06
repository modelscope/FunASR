#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0 <in-wav.scp> <out-datadir> [<logdir> [<outdir>]]
e.g.
$0 data/test/wav.scp data/test_format/

Format 'wav.scp': In short words,
changing "kaldi-datadir" to "modified-kaldi-datadir"

The 'wav.scp' format in kaldi is very flexible,
e.g. It can use unix-pipe as describing that wav file,
but it sometime looks confusing and make scripts more complex.
This tools creates actual wav files from 'wav.scp'
and also segments wav files using 'segments'.

Options
  --fs <fs>
  --segments <segments>
  --nj <nj>
  --cmd <cmd>
EOF
)

out_filename=wav.scp
cmd=utils/run.pl
nj=30
fs=none
segments=

ref_channels=
utt2ref_channels=

audio_format=wav
write_utt2num_samples=true

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 2 ] && [ $# -ne 3 ] && [ $# -ne 4 ]; then
    log "${help_message}"
    log "Error: invalid command line arguments"
    exit 1
fi

. ./path.sh  # Setup the environment

scp=$1
if [ ! -f "${scp}" ]; then
    log "${help_message}"
    echo "$0: Error: No such file: ${scp}"
    exit 1
fi
dir=$2


if [ $# -eq 2 ]; then
    logdir=${dir}/logs
    outdir=${dir}/data

elif [ $# -eq 3 ]; then
    logdir=$3
    outdir=${dir}/data

elif [ $# -eq 4 ]; then
    logdir=$3
    outdir=$4
fi


mkdir -p ${logdir}

rm -f "${dir}/${out_filename}"


opts=
if [ -n "${utt2ref_channels}" ]; then
    opts="--utt2ref-channels ${utt2ref_channels} "
elif [ -n "${ref_channels}" ]; then
    opts="--ref-channels ${ref_channels} "
fi


if [ -n "${segments}" ]; then
    log "[info]: using ${segments}"
    nutt=$(<${segments} wc -l)
    nj=$((nj<nutt?nj:nutt))

    split_segments=""
    for n in $(seq ${nj}); do
        split_segments="${split_segments} ${logdir}/segments.${n}"
    done

    utils/split_scp.pl "${segments}" ${split_segments}

    ${cmd} "JOB=1:${nj}" "${logdir}/format_wav_scp.JOB.log" \
        pyscripts/audio/format_wav_scp.py \
            ${opts} \
            --fs ${fs} \
            --audio-format "${audio_format}" \
            "--segment=${logdir}/segments.JOB" \
            "${scp}" "${outdir}/format.JOB"

else
    log "[info]: without segments"
    nutt=$(<${scp} wc -l)
    nj=$((nj<nutt?nj:nutt))

    split_scps=""
    for n in $(seq ${nj}); do
        split_scps="${split_scps} ${logdir}/wav.${n}.scp"
    done

    utils/split_scp.pl "${scp}" ${split_scps}
    ${cmd} "JOB=1:${nj}" "${logdir}/format_wav_scp.JOB.log" \
        pyscripts/audio/format_wav_scp.py \
        ${opts} \
        --fs "${fs}" \
        --audio-format "${audio_format}" \
        "${logdir}/wav.JOB.scp" ${outdir}/format.JOB""
fi

# Workaround for the NFS problem
ls ${outdir}/format.* > /dev/null

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat "${outdir}/format.${n}/wav.scp" || exit 1;
done > "${dir}/${out_filename}" || exit 1

if "${write_utt2num_samples}"; then
    for n in $(seq ${nj}); do
        cat "${outdir}/format.${n}/utt2num_samples" || exit 1;
    done > "${dir}/utt2num_samples"  || exit 1
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
