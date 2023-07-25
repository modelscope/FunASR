# CPU Benchmark (ONNX-python)

## Configuration
### Data set:
Aishell1 [test set](https://www.openslr.org/33/) , the total audio duration is 36108.919 seconds.

### Tools
#### Install Requirements
Install ModelScope and FunASR
```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
#pip install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

Install requirements
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR
cd funasr/runtime/python/utils
pip install -r requirements.txt
```

#### Recipe


##### test_rtf
set the model, data path and output_dir
```shell
nohup bash test_rtf.sh &> log.txt &
```

##### test_cer
set the model, data path and output_dir
```shell
nohup bash test_cer.sh &> log.txt &
```

## [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) 

Number of Parameter: 220M 

Storage size: 880MB

Storage size after int8-quant: 237MB

CER: 1.95%

CER after int8-quant: 1.95%

 ### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni

| concurrent-tasks | processing time(s) |   RTF   | Speedup Rate |
|:----------------:|:------------------:|:-------:|:------------:|
|  1 (onnx fp32)   |        2806        | 0.0777  |     12.9     |
|  1 (onnx int8)   |        1611        | 0.0446  |     22.4     |
|  8 (onnx fp32)   |        538         | 0.0149  |     67.1     |
|  8 (onnx int8)   |        210         | 0.0058  |    172.4     |
|  16 (onnx fp32)  |        288         | 0.0080  |    125.2     |
|  16 (onnx int8)  |        117         | 0.0032  |    309.9     |
|  32 (onnx fp32)  |        167         | 0.0046  |    216.5     |
|  32 (onnx int8)  |         86         | 0.0024  |    420.0     |
|  64 (onnx fp32)  |        158         | 0.0044  |    228.1     |
|  64 (onnx int8)  |         82         | 0.0023  |    442.8     |
|  96 (onnx fp32)  |        151         | 0.0042  |    238.0     |
|  96 (onnx int8)  |         80         | 0.0022  |    452.0     |


### Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz   16core-32processor    with avx512_vnni

| concurrent-tasks | processing time(s) |  RTF   | Speedup Rate |
|:----------------:|:------------------:|:------:|:------------:|
|  1 (onnx fp32)   |        2613        | 0.0724 |     13.8     |
|  1 (onnx int8)   |        1321        | 0.0366 |     22.4     |
|  32 (onnx fp32)  |        170         | 0.0047 |    212.7     |
|  32 (onnx int8)  |        89          | 0.0025 |    407.0     |
|  64 (onnx fp32)  |        166         | 0.0046 |    217.1     |
|  64 (onnx int8)  |         87         | 0.0024 |    414.7     |


### Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz    32core-64processor   without avx512_vnni


| concurrent-tasks | processing time(s) |  RTF   | Speedup Rate |
|:----------------:|:------------------:|:------:|:------------:|
|  1 (onnx fp32)   |        2959        | 0.0820 |     12.2     |
|  1 (onnx int8)   |        2814        | 0.0778 |     12.8     |
|  16 (onnx fp32)  |        373         | 0.0103 |     96.9     |
|  16 (onnx int8)  |        331         | 0.0091 |    109.0     |
|  32 (onnx fp32)  |        211         | 0.0058 |    171.4     |
|  32 (onnx int8)  |        181         | 0.0050 |    200.0     |
|  64 (onnx fp32)  |        153         | 0.0042 |    235.9     |
|  64 (onnx int8)  |        103         | 0.0029 |    349.9     |
|  96 (onnx fp32)  |        146         | 0.0041 |    247.0     |
|  96 (onnx int8)  |        108         | 0.0030 |    334.1     |

## [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)

Number of Parameter: 68M 

Storage size: 275MB

Storage size after int8-quant: 81MB

CER: 3.73%

CER after int8-quant: 3.78%

 ### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni

| concurrent-tasks | processing time(s) |  RTF   | Speedup Rate |
|:----------------:|:------------------:|:------:|:------------:|
|  1 (onnx fp32)   |        1173        | 0.0325 |     30.8     |
|  1 (onnx int8)   |        976         | 0.0270 |     37.0     |
|  16 (onnx fp32)  |         91         | 0.0025 |    395.2     |
|  16 (onnx int8)  |         78         | 0.0022 |    463.0     |
|  32 (onnx fp32)  |         60         | 0.0017 |    598.8     |
|  32 (onnx int8)  |         40         | 0.0011 |    892.9     |
|  64 (onnx fp32)  |         55         | 0.0015 |    653.6     |
|  64 (onnx int8)  |         31         | 0.0009 |    1162.8    |
|  96 (onnx fp32)  |         57         | 0.0016 |    632.9     |
|  96 (onnx int8)  |         33         | 0.0009 |    1098.9    |
