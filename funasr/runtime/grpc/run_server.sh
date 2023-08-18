#!/bin/bash

./build/bin/paraformer-server \
  --port-id 10100 \
  --model-dir models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --online-model-dir models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
  --quantize true \
  --vad-dir models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --vad-quant true \
  --punc-dir models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 \
  --punc-quant true \
  2>&1
