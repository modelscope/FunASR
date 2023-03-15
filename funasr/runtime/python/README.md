Benchmark [speech_paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) based on Aishell1 test set , the total audio duration is 36108.919 seconds.

(Note: The service has been fully warm up.)

 Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni

| concurrent-tasks | processing time(s) |  RTF   | Speedup Rate |
|:----------------:|:------------------:|:------:|:------------:|
|  1 (onnx fp32)   |        2806        | 0.0777 |     12.9     |
|  1 (onnx int8)   |        1611        | 0.0446 |     22.4     |
|  8 (onnx fp32)   |        538         | 0.0149 |     67.1     |
|  8 (onnx int8)   |        210         | 0.0058 |    172.4     |
|  16 (onnx fp32)  |        288         | 0.0080 |    125.2     |
|  16 (onnx int8)  |        117         | 0.0032 |    309.9     |
|  32 (onnx fp32)  |        167         | 0.0046 |    216.5     |
|  32 (onnx int8)  |        107         | 0.0030 |    338.0     |
|  64 (onnx fp32)  |        158         | 0.0044 |    228.1     |
|  64 (onnx int8)  |         82         | 0.0023 |    442.8     |
|  96 (onnx fp32)  |        151         | 0.0042 |    238.0     |
|  96 (onnx int8)  |         80         | 0.0022 |    452.0     |

