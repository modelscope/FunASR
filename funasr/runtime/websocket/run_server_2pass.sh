
download_model_dir="../../models"
model_dir="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx"
#model_dir="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
online_model_dir="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
vad_dir="damo/speech_fsmn_vad_zh-cn-16k-common-onnx"
punc_dir="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
itn_dir="damo/fst_itn_zh"
decoder_thread_num=1
io_thread_num=1
port=10095
certfile="../../../ssl_key/server.crt"
keyfile="../../../ssl_key/server.key"

. ./parse_options.sh || exit 1;

cd build_debug/bin

# Since this model is not released by damo, we choose to download it here
itn_real_path=$download_model_dir"/"$itn_dir
if [ ! -d $itn_real_path ]; then 
  git clone https://www.modelscope.cn/thuduj12/fst_itn_zh.git $itn_real_path
fi

./funasr-wss-server-2pass  \
  --download-model-dir ${download_model_dir} \
  --model-dir ${model_dir} \
  --online-model-dir ${online_model_dir}  \
  --quantize false  \
  --vad-dir ${vad_dir} \
  --punc-dir ${punc_dir} \
  --itn-model-dir ${itn_dir}  \
  --decoder-thread-num ${decoder_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --port ${port} \
  --certfile  ${certfile} \
  --keyfile ${keyfile}

