#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 && "$1" != -* ]]; then
  exec "$@"
fi

model_root="${FUNASR_DOWNLOAD_MODEL_DIR:-/workspace/models}"
hotword_file="${FUNASR_HOTWORD_FILE:-${model_root}/hotwords.txt}"
mkdir -p "${model_root}"
touch "${hotword_file}"

decoder_threads="${FUNASR_DECODER_THREAD_NUM:-$(nproc)}"
io_threads="${FUNASR_IO_THREAD_NUM:-$(( (decoder_threads + 15) / 16 ))}"

server=/workspace/FunASR/runtime/websocket/build/bin/funasr-wss-server-2pass

exec "${server}" \
  --download-model-dir "${model_root}" \
  --model-dir "${FUNASR_MODEL_DIR:-damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx}" \
  --online-model-dir "${FUNASR_ONLINE_MODEL_DIR:-damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx}" \
  --vad-dir "${FUNASR_VAD_DIR:-damo/speech_fsmn_vad_zh-cn-16k-common-onnx}" \
  --punc-dir "${FUNASR_PUNC_DIR:-damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx}" \
  --itn-dir "${FUNASR_ITN_DIR:-thuduj12/fst_itn_zh}" \
  --lm-dir "${FUNASR_LM_DIR:-damo/speech_ngram_lm_zh-cn-ai-wesp-fst}" \
  --decoder-thread-num "${decoder_threads}" \
  --model-thread-num "${FUNASR_MODEL_THREAD_NUM:-1}" \
  --io-thread-num "${io_threads}" \
  --port "${FUNASR_PORT:-10095}" \
  --certfile "" \
  --keyfile "" \
  --hotword "${hotword_file}" \
  "$@"
