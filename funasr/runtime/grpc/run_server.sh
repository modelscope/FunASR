#!/bin/bash

./build/bin/paraformer-server \
  --port-id 10100 \
  --offline-model-dir /cfs/user/burkliu/data/funasr_models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --online-model-dir /cfs/user/burkliu/data/funasr_models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
  --quantize true \
  --vad-dir /cfs/user/burkliu/data/funasr_models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --vad-quant true \
  --punc-dir /cfs/user/burkliu/data/funasr_models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 \
  --punc-quant true \
  2>&1
