
download_model_dir="/workspace/models"
model_dir="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
online_model_dir="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
vad_dir="damo/speech_fsmn_vad_zh-cn-16k-common-onnx"
punc_dir="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
itn_dir="thuduj12/fst_itn_zh"
port=10095
certfile="../../../ssl_key/server.crt"
keyfile="../../../ssl_key/server.key"
hotword="../../hotwords.txt"
# set decoder_thread_num
decoder_thread_num=$(cat /proc/cpuinfo | grep "processor"|wc -l) || { echo "Get cpuinfo failed. Set decoder_thread_num = 32"; decoder_thread_num=32; }
multiple_io=16
io_thread_num=$(( (decoder_thread_num + multiple_io - 1) / multiple_io ))
model_thread_num=1

. ../egs/aishell/transformer/utils/parse_options.sh || exit 1;

if [ -z "$certfile" ] || [ "$certfile" = "0" ]; then
  certfile=""
  keyfile=""
fi

cd /workspace/FunASR/runtime/websocket/build/bin
./funasr-wss-server-2pass  \
  --download-model-dir "${download_model_dir}" \
  --model-dir "${model_dir}" \
  --online-model-dir "${online_model_dir}" \
  --vad-dir "${vad_dir}" \
  --punc-dir "${punc_dir}" \
  --itn-dir "${itn_dir}" \
  --decoder-thread-num ${decoder_thread_num} \
  --model-thread-num ${model_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --port ${port} \
  --certfile  "${certfile}" \
  --keyfile "${keyfile}" \
  --hotword "${hotword}"

