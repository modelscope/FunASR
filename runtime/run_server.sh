
download_model_dir="/workspace/models"
model_dir="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
vad_dir="damo/speech_fsmn_vad_zh-cn-16k-common-onnx"
punc_dir="damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx"
itn_dir="thuduj12/fst_itn_zh"
lm_dir="damo/speech_ngram_lm_zh-cn-ai-wesp-fst"
decoder_thread_num=32
model_thread_num=1
io_thread_num=8
port=10095
certfile="../../../ssl_key/server.crt"
keyfile="../../../ssl_key/server.key"
hotword="../../hotwords.txt"

. ../egs/aishell/transformer/utils/parse_options.sh || exit 1;

cd /workspace/FunASR/runtime/websocket/build/bin
if [ -z "$certfile" ] || [ "$certfile" -eq 0 ]; then
./funasr-wss-server  \
  --download-model-dir "${download_model_dir}" \
  --model-dir "${model_dir}" \
  --vad-dir "${vad_dir}" \
  --punc-dir "${punc_dir}" \
  --itn-dir "${itn_dir}" \
  --lm-dir "${lm_dir}" \
  --decoder-thread-num ${decoder_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --model-thread-num ${model_thread_num} \
  --port ${port} \
  --certfile  "" \
  --keyfile "" \
  --hotword "${hotword}"
else
./funasr-wss-server  \
  --download-model-dir "${download_model_dir}" \
  --model-dir "${model_dir}" \
  --vad-dir "${vad_dir}" \
  --punc-dir "${punc_dir}" \
  --itn-dir "${itn_dir}" \
  --lm-dir "${lm_dir}" \
  --decoder-thread-num ${decoder_thread_num} \
  --model-thread-num ${model_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --port ${port} \
  --certfile  "${certfile}" \
  --keyfile "${keyfile}" \
  --hotword "${hotword}"
fi
