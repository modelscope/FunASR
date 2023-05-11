#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ngpu=4
device="0,1,2,3"

stage=12
stop_stage=13


train_set=Train_Ali_far
valid_set=Eval_Ali_far
test_sets="Test_Ali_far"
asr_config=conf/train_asr_conformer.yaml
sa_asr_config=conf/train_sa_asr_conformer.yaml
inference_config=conf/decode_asr_rnn.yaml
infer_with_pretrained_model=true
download_sa_asr_model="damo/speech_saasr_asr-zh-cn-16k-alimeeting"

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false
./asr_local.sh                                         \
    --device ${device}                                 \
    --ngpu ${ngpu}                                     \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --gpu_inference true    \
    --njob_infer 4    \
    --infer_with_pretrained_model ${infer_with_pretrained_model} \
    --download_sa_asr_model $download_sa_asr_model \
    --asr_exp exp/asr_train_multispeaker_conformer_raw_zh_char_data_alimeeting \
    --sa_asr_exp exp/sa_asr_train_conformer_raw_zh_char_data_alimeeting \
    --asr_stats_dir exp/asr_stats_multispeaker_conformer_raw_zh_char_data_alimeeting \
    --lm_exp exp/lm_train_multispeaker_transformer_zh_char_data_alimeeting \
    --lm_stats_dir exp/lm_stats_multispeaker_zh_char_data_alimeeting \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --sa_asr_config "${sa_asr_config}"                 \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --lm_train_text "data/${train_set}/text" "$@"
