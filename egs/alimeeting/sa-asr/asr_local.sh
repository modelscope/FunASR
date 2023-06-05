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
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=16                # The number of parallel jobs.
inference_nj=16      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
njob_infer=4
dumpdir=dump2         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.
device=0

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16000             # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Language model related
use_lm=true       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the direcotry path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the direcotry path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
asr_tag=       # Suffix to the result dir for asr model training.
asr_exp=       # Specify the direcotry path for ASR experiment.
               # If this option is specified, asr_tag is ignored.
sa_asr_exp=
asr_stats_dir= # Specify the direcotry path for ASR statistics.
asr_config=    # Config for asr model training.
sa_asr_config=
asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.

# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
sa_asr_inference_tag=
sa_asr_inference_args=

inference_lm=valid.loss.ave.pb        # Language modle path for decoding.
inference_asr_model=valid.acc.ave.pb  # ASR model path for decoding.
                                      # e.g.
                                      # inference_asr_model=train.loss.best.pth
                                      # inference_asr_model=3epoch.pth
                                      # inference_asr_model=valid.acc.best.pth
                                      # inference_asr_model=valid.loss.ave.pth
inference_sa_asr_model=valid.acc_spk.ave.pb
infer_with_pretrained_model=false   # Use pretrained model for decoding
download_sa_asr_model=          # Download the SA-ASR model from ModelScope and use it for decoding.
# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=zh      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.


help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").
    --device         # Which GPUs are use for local training (defalut="${device}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the direcotry path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the direcotry path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ASR model related
    --asr_tag          # Suffix to the result dir for asr model training (default="${asr_tag}").
    --asr_exp          # Specify the direcotry path for ASR experiment.
                       # If this option is specified, asr_tag is ignored (default="${asr_exp}").
    --asr_stats_dir    # Specify the direcotry path for ASR statistics (default="${asr_stats_dir}").
    --asr_config       # Config for asr model training (default="${asr_config}").
    --asr_args         # Arguments for asr model training (default="${asr_args}").
                       # e.g., --asr_args "--max_epoch 10"
                       # Note that it will overwrite args in asr config.
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language modle path for decoding (default="${inference_lm}").
    --inference_asr_model # ASR model path for decoding (default="${inference_asr_model}").
    --infer_with_pretrained_model      # Use pretrained model for decoding (default="${infer_with_pretrained_model}").
    --download_sa_asr_model=          # Download the SA-ASR model from ModelScope and use it for decoding(default="${download_sa_asr_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(python -m funasr.utils.cli_utils $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as ASR for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi

if ${infer_with_pretrained_model}; then
    skip_train=true
fi

# Set tag for naming of model directory
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
    else
        asr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        asr_tag+="_${lang}_${token_type}"
    else
        asr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${asr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${lang}_${token_type}"
    else
        asr_stats_dir="${expdir}/asr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        asr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        asr_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${asr_exp}" ]; then
    asr_exp="${expdir}/asr_${asr_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

if [ -z "${sa_asr_inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        sa_asr_inference_tag="$(basename "${inference_config}" .yaml)"
    else
        sa_asr_inference_tag=sa_asr_inference
    fi
    # Add overwritten arg's info
    if [ -n "${sa_asr_inference_args}" ]; then
        sa_asr_inference_tag+="$(echo "${sa_asr_inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        sa_asr_inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    sa_asr_inference_tag+="_asr_model_$(echo "${inference_sa_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

train_cmd="run.pl"
cuda_cmd="run.pl"
decode_cmd="run.pl"

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."

        ./local/alimeeting_data_prep.sh --tgt Test
        ./local/alimeeting_data_prep.sh --tgt Eval
        ./local/alimeeting_data_prep.sh --tgt Train
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
           for factor in ${speed_perturb_factors}; do
               if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                   local/perturb_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                   _dirs+="data/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="data/${train_set} "
               fi
           done
           local/combine_data.sh "data/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}" "${test_sets}" ; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    if [ "${dset}" = "${test_sets}" ] && [ "${test_sets}" = "Test_Ali_far" ]; then
                        _suf="/org"
                    else
                        _suf=""
                    fi
                fi
                local/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
                
                if [ "${dset}" = "Train_Ali_far" ] || [ "${dset}" = "Eval_Ali_far" ] || [ "${dset}" = "Test_Ali_far" ]; then
                    cp data/"${dset}"/utt2spk_all_fifo "${data_feats}${_suf}/${dset}/"
                fi

                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                local/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        if [ "${test_sets}" = "Test_Ali_far" ]; then
            rm_dset="${train_set} ${valid_set} ${test_sets}"
        else
            rm_dset="${train_set} ${valid_set}"
        fi

        for dset in $rm_dset; do

            # Copy data dir
            local/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                <"${data_feats}/org/${dset}/utt2num_samples" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples"
                <"${data_feats}/org/${dset}/wav.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                    | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                <"${data_feats}/org/${dset}/feats.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            local/fix_data_dir.sh "${data_feats}/${dset}"

            # generate uttid
            cut -d ' ' -f 1 "${data_feats}/${dset}/wav.scp" > "${data_feats}/${dset}/uttid"
            
            if [ "${dset}" = "Train_Ali_far" ] || [ "${dset}" = "Eval_Ali_far" ] || [ "${dset}" = "Test_Ali_far" ]; then
                # filter utt2spk_all_fifo
                python local/filter_utt2spk_all_fifo.py ${data_feats}/${dset}/uttid ${data_feats}/org/${dset} ${data_feats}/${dset}
            fi
        done

        # shellcheck disable=SC2002
        cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Dictionary Preparation"
        mkdir -p data/${lang}_token_list/char/
    
        echo "make a dictionary"
        echo "<blank>" > ${token_list}
        echo "<s>" >> ${token_list}
        echo "</s>" >> ${token_list}
        utils/text2token.py -s 1 -n 1 --space "" ${data_feats}/lm_train.txt | cut -f 2- -d" " | tr " " "\n" \
            | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
        num_token=$(cat ${token_list} | wc -l)
        echo "<unk>" >> ${token_list}
        vocab_size=$(cat ${token_list} | wc -l)
    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Generate speaker settings"
        mkdir -p "profile_log"
        for dset in "${train_set}" "${valid_set}" "${test_sets}"; do
            # generate text_id spk2id
            python local/process_sot_fifo_textchar2spk.py --path ${data_feats}/${dset}
            log "Successfully generate ${data_feats}/${dset}/text_id ${data_feats}/${dset}/spk2id"
            # generate text_id_train for sot
            python local/process_text_id.py ${data_feats}/${dset}
            log "Successfully generate ${data_feats}/${dset}/text_id_train"
            # generate oracle_embedding from single-speaker audio segment
            log "oracle_embedding is being generated in the background, and the log is profile_log/gen_oracle_embedding_${dset}.log"
            python local/gen_oracle_embedding.py "${data_feats}/${dset}" "data/local/${dset}_correct_single_speaker" &> "profile_log/gen_oracle_embedding_${dset}.log"
            log "Successfully generate oracle embedding for ${dset} (${data_feats}/${dset}/oracle_embedding.scp)"
            # generate oracle_profile and cluster_profile from oracle_embedding and cluster_embedding (padding the speaker during training)
            if [ "${dset}" = "${train_set}" ]; then
                python local/gen_oracle_profile_padding.py ${data_feats}/${dset}
                log "Successfully generate oracle profile for ${dset} (${data_feats}/${dset}/oracle_profile_padding.scp)"
            else
                python local/gen_oracle_profile_nopadding.py ${data_feats}/${dset}
                log "Successfully generate oracle profile for ${dset} (${data_feats}/${dset}/oracle_profile_nopadding.scp)"
            fi
            # generate cluster_profile with spectral-cluster directly (for infering and without oracle information)
            if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${test_sets}" ]; then
                log "cluster_profile is being generated in the background, and the log is profile_log/gen_cluster_profile_infer_${dset}.log"
                python local/gen_cluster_profile_infer.py "${data_feats}/${dset}" "data/local/${dset}" 0.996 0.815 &> "profile_log/gen_cluster_profile_infer_${dset}.log"
                log "Successfully generate cluster profile for ${dset} (${data_feats}/${dset}/cluster_profile_infer.scp)"
            fi

            done
    fi

else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if "${use_lm}"; then
        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            log "Stage 7: LM collect stats: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            # 1. Split the key file
            _logdir="${lm_stats_dir}/logdir"
            mkdir -p "${_logdir}"
            # Get the minimum number among ${nj} and the number lines of input files
            _nj=$(min "${nj}" "$(<${data_feats}/lm_train.txt wc -l)" "$(<${lm_dev_text} wc -l)")

            key_file="${data_feats}/lm_train.txt"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/train.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            key_file="${lm_dev_text}"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/dev.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Generate run.sh
            log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
            mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

            # 3. Submit jobs
            log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m funasr.bin.lm_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${data_feats}/lm_train.txt,text,text" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/dev.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    ${_opts} ${lm_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

            # 4. Aggregate shape files
            _opts=
            for i in $(seq "${_nj}"); do
                _opts+="--input_dir ${_logdir}/stats.${i} "
            done
            # shellcheck disable=SC2086
            ${python} -m funasr.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${lm_stats_dir}/train/text_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/train/text_shape.${lm_token_type}"

            <"${lm_stats_dir}/valid/text_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/valid/text_shape.${lm_token_type}"
        fi


        if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
            log "Stage 8: LM Training: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            if [ "${num_splits_lm}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                      --scps "${data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                      --num_splits "${num_splits_lm}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${data_feats}/lm_train.txt,text,text "
                _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
            fi

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

            log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 8 using this script"
            mkdir -p "${lm_exp}"; echo "${run_args} --stage 8 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

            log "LM training started... log: '${lm_exp}/train.log'"
            if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
                # SGE can't include "/" in a job name
                jobname="$(basename ${lm_exp})"
            else
                jobname="${lm_exp}/train.log"
            fi

            mkdir -p ${lm_exp}
            mkdir -p ${lm_exp}/log
            INIT_FILE=${lm_exp}/ddp_init
            if [ -f $INIT_FILE ];then
                rm -f $INIT_FILE
            fi 
            init_method=file://$(readlink -f $INIT_FILE)
            echo "$0: init method is $init_method"
            for ((i = 0; i < $ngpu; ++i)); do
                {
                    # i=0
                    rank=$i
                    local_rank=$i
                    gpu_id=$(echo $device | cut -d',' -f$[$i+1])
                    lm_train.py \
                        --gpu_id $gpu_id \
                        --use_preprocessor true \
                        --bpemodel ${bpemodel} \
                        --token_type ${token_type} \
                        --token_list ${token_list} \
                        --non_linguistic_symbols ${nlsyms_txt} \
                        --cleaner ${cleaner} \
                        --g2p ${g2p} \
                        --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                        --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
                        --resume true \
                        --output_dir ${lm_exp} \
                        --config $lm_config \
                        --ngpu $ngpu \
                        --num_worker_count 1 \
                        --multiprocessing_distributed true \
                        --dist_init_method $init_method \
                        --dist_world_size $ngpu \
                        --dist_rank $rank \
                        --local_rank $local_rank \
                        ${_opts} 1> ${lm_exp}/log/train.log.$i 2>&1
                } &
                done
                wait

        fi


        if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
            log "Stage 9: Calc perplexity: ${lm_test_text}"
            _opts=
            # TODO(kamo): Parallelize?
            log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES=${device}\
            ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
                ${python} -m funasr.bin.lm_calc_perplexity \
                    --ngpu "${ngpu}" \
                    --data_path_and_name_and_type "${lm_test_text},text,text" \
                    --train_config "${lm_exp}"/config.yaml \
                    --model_file "${lm_exp}/${inference_lm}" \
                    --output_dir "${lm_exp}/perplexity_test" \
                    ${_opts}
            log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

        fi

    else
        log "Stage 7-9: Skip lm-related stages: use_lm=${use_lm}"
    fi


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: ASR collect stats: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${asr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_asr_train_dir}/${_scp} wc -l)" "$(<${_asr_valid_dir}/${_scp} wc -l)")

        key_file="${_asr_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_asr_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${asr_stats_dir}/run.sh'. You can resume the process from stage 9 using this script"
        mkdir -p "${asr_stats_dir}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${asr_stats_dir}/run.sh"; chmod +x "${asr_stats_dir}/run.sh"

        # 3. Submit jobs
        log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m funasr.bin.asr_train \
                --collect_stats true \
                --mc true   \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --split_with_space false    \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${asr_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m funasr.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${asr_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${asr_stats_dir}/train/text_shape.${token_type}"

        <"${asr_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${asr_stats_dir}/valid/text_shape.${token_type}"
    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 11: ASR Training: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${asr_config} "
        fi

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_asr_train_dir}/${_scp}" \
                      "${_asr_train_dir}/text" \
                      "${asr_stats_dir}/train/speech_shape" \
                      "${asr_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_asr}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
        fi

        # log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 10 using this script"
        # mkdir -p "${asr_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ASR training started... log: '${asr_exp}/log/train.log'"
        # if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        #     # SGE can't include "/" in a job name
        #     jobname="$(basename ${asr_exp})"
        # else
        #     jobname="${asr_exp}/train.log"
        # fi

        mkdir -p ${asr_exp}
        mkdir -p ${asr_exp}/log
        INIT_FILE=${asr_exp}/ddp_init
        if [ -f $INIT_FILE ];then
            rm -f $INIT_FILE
        fi 
        init_method=file://$(readlink -f $INIT_FILE)
        echo "$0: init method is $init_method"
        for ((i = 0; i < $ngpu; ++i)); do
            {
                # i=0
                rank=$i
                local_rank=$i
                gpu_id=$(echo $device | cut -d',' -f$[$i+1])
                asr_train.py \
                    --mc true   \
                    --gpu_id $gpu_id \
                    --use_preprocessor true \
                    --bpemodel ${bpemodel} \
                    --token_type ${token_type} \
                    --token_list ${token_list} \
                    --split_with_space false    \
                    --non_linguistic_symbols ${nlsyms_txt} \
                    --cleaner ${cleaner} \
                    --g2p ${g2p} \
                    --valid_data_path_and_name_and_type ${_asr_valid_dir}/${_scp},speech,${_type} \
                    --valid_data_path_and_name_and_type ${_asr_valid_dir}/text,text,text \
                    --valid_shape_file ${asr_stats_dir}/valid/speech_shape \
                    --valid_shape_file ${asr_stats_dir}/valid/text_shape.${token_type} \
                    --resume true \
                    --output_dir ${asr_exp} \
                    --config $asr_config \
                    --ngpu $ngpu \
                    --num_worker_count 1 \
                    --multiprocessing_distributed true \
                    --dist_init_method $init_method \
                    --dist_world_size $ngpu \
                    --dist_rank $rank \
                    --local_rank $local_rank \
                    ${_opts} 1> ${asr_exp}/log/train.log.$i 2>&1
            } &
            done
            wait

    fi

    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        _asr_train_dir="${data_feats}/${train_set}"
        _asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 12: SA-ASR Training: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

        _opts=
        if [ -n "${sa_asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${sa_asr_config} "
        fi

        _feats_type="$(<${_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_asr_train_dir}/${_scp}" \
                      "${_asr_train_dir}/text" \
                      "${asr_stats_dir}/train/speech_shape" \
                      "${asr_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_asr}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text_id_train,text_id,text_int "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/oracle_profile_padding.scp,profile,npy "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/oracle_profile_padding.scp,profile,npy "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text_id_train,text_id,text_int "
            _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
        fi

        # log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 10 using this script"
        # mkdir -p "${asr_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "SA-ASR training started... log: '${sa_asr_exp}/log/train.log'"
        # if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        #     # SGE can't include "/" in a job name
        #     jobname="$(basename ${asr_exp})"
        # else
        #     jobname="${asr_exp}/train.log"
        # fi

        mkdir -p ${sa_asr_exp}
        mkdir -p ${sa_asr_exp}/log
        INIT_FILE=${sa_asr_exp}/ddp_init
        
        if [ ! -f "exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth" ]; then
            # download xvector extractor model file
            python local/download_xvector_model.py exp
            log "Successfully download the pretrained xvector extractor to exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth"
        fi
        
        if [ -f $INIT_FILE ];then
            rm -f $INIT_FILE
        fi 
        init_method=file://$(readlink -f $INIT_FILE)
        echo "$0: init method is $init_method"
        for ((i = 0; i < $ngpu; ++i)); do
            {
                # i=0
                rank=$i
                local_rank=$i
                gpu_id=$(echo $device | cut -d',' -f$[$i+1])
                sa_asr_train.py \
                    --gpu_id $gpu_id \
                    --use_preprocessor true \
                    --unused_parameters true \
                    --bpemodel ${bpemodel} \
                    --token_type ${token_type} \
                    --token_list ${token_list} \
                    --max_spk_num 4 \
                    --split_with_space false    \
                    --non_linguistic_symbols ${nlsyms_txt} \
                    --cleaner ${cleaner} \
                    --g2p ${g2p} \
                    --allow_variable_data_keys true \
                    --init_param "${asr_exp}/valid.acc.ave.pb:encoder:asr_encoder"   \
                    --init_param "${asr_exp}/valid.acc.ave.pb:ctc:ctc"   \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.embed:decoder.embed" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.output_layer:decoder.asr_output_layer" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.0.self_attn:decoder.decoder1.self_attn" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.0.src_attn:decoder.decoder3.src_attn" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.0.feed_forward:decoder.decoder3.feed_forward" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.1:decoder.decoder4.0" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.2:decoder.decoder4.1" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.3:decoder.decoder4.2" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.4:decoder.decoder4.3" \
                    --init_param "${asr_exp}/valid.acc.ave.pb:decoder.decoders.5:decoder.decoder4.4" \
                    --init_param "exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth:encoder:spk_encoder"   \
                    --init_param "exp/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/sv.pth:decoder:spk_encoder:decoder.output_dense"   \
                    --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                    --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                    --valid_data_path_and_name_and_type "${_asr_valid_dir}/oracle_profile_nopadding.scp,profile,npy" \
                    --valid_data_path_and_name_and_type "${_asr_valid_dir}/text_id_train,text_id,text_int" \
                    --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
                    --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
                    --resume true \
                    --output_dir ${sa_asr_exp} \
                    --config $sa_asr_config \
                    --ngpu $ngpu \
                    --num_worker_count 1 \
                    --multiprocessing_distributed true \
                    --dist_init_method $init_method \
                    --dist_world_size $ngpu \
                    --dist_rank $rank \
                    --local_rank $local_rank \
                    ${_opts} 1> ${sa_asr_exp}/log/train.log.$i 2>&1
            } &
            done
            wait

    fi

else
    log "Skip the training stages"
fi

if ${infer_with_pretrained_model}; then
    log "Use ${download_sa_asr_model} for decoding and evaluation"
    sa_asr_exp="${expdir}/${download_sa_asr_model}"
    mkdir -p "${sa_asr_exp}"


    python local/download_pretrained_model_from_modelscope.py $download_sa_asr_model ${expdir}
    inference_sa_asr_model="model.pb"
    inference_config=${sa_asr_exp}/decoding.yaml
fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "Stage 13: Decoding SA-ASR (oracle profile): training_dir=${sa_asr_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            inference_nj=$[${ngpu}*${njob_infer}]
            _ngpu=1

        else
            _cmd="${decode_cmd}"
            inference_nj=$inference_nj
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi

        # 2. Generate run.sh
        log "Generate '${sa_asr_exp}/${sa_asr_inference_tag}.oracle/run.sh'. You can resume the process from stage 15 using this script"
        mkdir -p "${sa_asr_exp}/${sa_asr_inference_tag}.oracle"; echo "${run_args} --stage 15 \"\$@\"; exit \$?" > "${sa_asr_exp}/${sa_asr_inference_tag}.oracle/run.sh"; chmod +x "${sa_asr_exp}/${sa_asr_inference_tag}.oracle/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${sa_asr_exp}/${sa_asr_inference_tag}.oracle/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/sa_asr_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                python -m funasr.bin.asr_inference_launch \
                    --batch_size 1 \
                    --mc True   \
                    --nbest 1   \
                    --ngpu "${_ngpu}" \
                    --njob ${njob_infer} \
                    --gpuid_list ${device} \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --data_path_and_name_and_type "${_data}/oracle_profile_nopadding.scp,profile,npy" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --allow_variable_data_keys true \
                    --asr_train_config "${sa_asr_exp}"/config.yaml \
                    --asr_model_file "${sa_asr_exp}"/"${inference_sa_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    --mode sa_asr \
                    ${_opts}


            # 3. Concatenates the output files from each jobs
            for f in token token_int score text text_id; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done
        done
    fi

    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: Scoring SA-ASR (oracle profile)"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${sa_asr_exp}/${sa_asr_inference_tag}.oracle/${dset}"

            sed 's/\$//g' ${_data}/text > ${_data}/text_nosrc
            sed 's/\$//g' ${_dir}/text > ${_dir}/text_nosrc

            python utils/proce_text.py ${_data}/text_nosrc ${_data}/text.proc
            python utils/proce_text.py ${_dir}/text_nosrc ${_dir}/text.proc

            python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
            tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
            cat ${_dir}/text.cer.txt

            python local/process_text_spk_merge.py ${_dir}
            python local/process_text_spk_merge.py ${_data}
            
            python local/compute_cpcer.py ${_data}/text_spk_merge ${_dir}/text_spk_merge ${_dir}/text.cpcer
            tail -n 1 ${_dir}/text.cpcer > ${_dir}/text.cpcer.txt
            cat ${_dir}/text.cpcer.txt
            
        done

    fi

    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        log "Stage 15: Decoding SA-ASR (cluster profile): training_dir=${sa_asr_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            inference_nj=$[${ngpu}*${njob_infer}]
            _ngpu=1

        else
            _cmd="${decode_cmd}"
            inference_nj=$inference_nj
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi

        # 2. Generate run.sh
        log "Generate '${sa_asr_exp}/${sa_asr_inference_tag}.cluster/run.sh'. You can resume the process from stage 17 using this script"
        mkdir -p "${sa_asr_exp}/${sa_asr_inference_tag}.cluster"; echo "${run_args} --stage 17 \"\$@\"; exit \$?" > "${sa_asr_exp}/${sa_asr_inference_tag}.cluster/run.sh"; chmod +x "${sa_asr_exp}/${sa_asr_inference_tag}.cluster/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${sa_asr_exp}/${sa_asr_inference_tag}.cluster/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/sa_asr_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                python -m funasr.bin.asr_inference_launch \
                    --batch_size 1 \
                    --mc True   \
                    --nbest 1   \
                    --ngpu "${_ngpu}" \
                    --njob ${njob_infer} \
                    --gpuid_list ${device} \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --data_path_and_name_and_type "${_data}/cluster_profile_infer.scp,profile,npy" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --allow_variable_data_keys true \
                    --asr_train_config "${sa_asr_exp}"/config.yaml \
                    --asr_model_file "${sa_asr_exp}"/"${inference_sa_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    --mode sa_asr \
                    ${_opts}

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text text_id; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done
        done
    fi

    if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
        log "Stage 16: Scoring SA-ASR (cluster profile)"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${sa_asr_exp}/${sa_asr_inference_tag}.cluster/${dset}"

            sed 's/\$//g' ${_data}/text > ${_data}/text_nosrc
            sed 's/\$//g' ${_dir}/text > ${_dir}/text_nosrc

            python utils/proce_text.py ${_data}/text_nosrc ${_data}/text.proc
            python utils/proce_text.py ${_dir}/text_nosrc ${_dir}/text.proc

            python utils/compute_wer.py ${_data}/text.proc ${_dir}/text.proc ${_dir}/text.cer
            tail -n 3 ${_dir}/text.cer > ${_dir}/text.cer.txt
            cat ${_dir}/text.cer.txt

            python local/process_text_spk_merge.py ${_dir}
            python local/process_text_spk_merge.py ${_data}
            
            python local/compute_cpcer.py ${_data}/text_spk_merge ${_dir}/text_spk_merge ${_dir}/text.cpcer
            tail -n 1 ${_dir}/text.cpcer > ${_dir}/text.cpcer.txt
            cat ${_dir}/text.cpcer.txt
            
        done

    fi

else
    log "Skip the evaluation stages"
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
