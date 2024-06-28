# GPU Benchmark (libtorch-cpp)

## Configuration
### Data set:
A long audio test set(Non-open source) containing 103 audio files, with durations ranging from 2 to 30 minutes.

## [FSMN-VAD](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/summary) + [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-torchscript/summary) + [CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx/summary) 

```shell
./funasr-onnx-offline-rtf \
    --model-dir    ./damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-torchscript \
    --vad-dir   ./damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
    --punc-dir  ./damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
    --gpu \
    --thread-num 20 \
    --bladedisc true \
    --batch-size 20 \
    --wav-path     ./long_test.scp
```
Node: run in docker, ref to ([docs](./SDK_advanced_guide_offline_gpu_zh.md))

### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz 16core-32processor with avx512_vnni, GPU @ A10

| concurrent-tasks | batch  |   RTF  | Speedup Rate |
|------------------|:------:|:------:|:------------:|
| 1                |   1    | 0.0076 |      130     |
| 1                |   20   | 0.0048 |      208     |
| 5                |   20   | 0.0011 |      850     |
| 10               |   20   | 0.0008 |      1200+   |
| 20               |   20   | 0.0008 |      1200+   |

Node: On CPUs, the single-thread RTF is 0.066, and 32-threads' speedup is 330+

