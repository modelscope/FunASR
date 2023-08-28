# workdir is always websocket
build=build  # the build dir

# Build websocket service, with onnxruntime
if [ ! -f $build/bin/funasr-wss-server-2pass ]; then 
  echo "1st time run, we need to build the server, which may take a while(especially when fetch from git)."
  if [ ! -d ffmpeg-N-111383-g20b8688092-linux64-gpl-shared ]; then
    bash ../onnxruntime/third_party/download_ffmpeg.sh
  fi
  if [ ! -d onnxruntime-linux-x64-1.14.0 ]; then 
    bash ../onnxruntime/third_party/download_onnxruntime.sh
  fi

  # we build the server under "build" dir.
  cmake -DONNXRUNTIME_DIR=`pwd`/onnxruntime-linux-x64-1.14.0 \
    -DFFMPEG_DIR=`pwd`/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared \
    -B $build
  cmake --build $build
fi


download_model_dir="models"
# now the hotword model and timestamp model is two different model
# you can choose the following 3 model to use which you want, by comment the others.
model_dir="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"            # offline base model
model_dir="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx" # hotword model
model_dir="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"   # timestamp model
online_model_dir="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
vad_dir="damo/speech_fsmn_vad_zh-cn-16k-common-onnx"
punc_dir="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
itn_dir="damo/fst_itn_zh"
decoder_thread_num=16
io_thread_num=4
port=10095
certfile="../ssl_key/server.crt"
keyfile="../ssl_key/server.key"

. ./parse_options.sh || exit 1;

# Since this model is not released by damo, we choose to download it here
itn_real_path=$download_model_dir"/"$itn_dir
if [ ! -d $itn_real_path ]; then 
  git clone https://www.modelscope.cn/thuduj12/fst_itn_zh.git $itn_real_path
fi

$build/bin/funasr-wss-server-2pass  \
  --download-model-dir ${download_model_dir} \
  --model-dir ${model_dir} \
  --online-model-dir ${online_model_dir}  \
  --quantize true  \
  --vad-dir ${vad_dir} \
  --punc-dir ${punc_dir} \
  --itn-model-dir ${itn_dir}  \
  --decoder-thread-num ${decoder_thread_num} \
  --io-thread-num  ${io_thread_num} \
  --port ${port} \
  --certfile  ${certfile} \
  --keyfile ${keyfile}

