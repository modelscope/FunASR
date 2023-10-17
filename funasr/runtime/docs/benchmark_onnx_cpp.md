# CPU Benchmark (ONNX-cpp)

## Configuration
### Data set:
Aishell1 [test set](https://www.openslr.org/33/) , the total audio duration is 36108.919 seconds.

### Tools
#### Install [modelscope and funasr](https://github.com/alibaba-damo-academy/FunASR#installation)

```shell
pip3 install torch torchaudio
pip install -U modelscope
pip install -U funasr
```

#### Export [onnx model](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```

#### Building for Linux/Unix

Download onnxruntime
```shell
# download an appropriate onnxruntime from https://github.com/microsoft/onnxruntime/releases/tag/v1.14.0
# here we get a copy of onnxruntime for linux 64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
```

Install openblas
```shell
sudo apt-get install libopenblas-dev #ubuntu
# sudo yum -y install openblas-devel #centos
```

Build runtime
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/onnxruntime
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
make
```

## [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) 
```shell
./funasr-onnx-offline-rtf \
    --model-dir    ./asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --quantize  true \
    --wav-path     ./aishell1_test.scp  \
    --thread-num 32

Node: '--quantize false' means fp32, otherwise it will be int8 
```

Number of Parameter: 220M 

Storage size: 880MB

Storage size after int8-quant: 237MB

CER: 1.95%

CER after int8-quant: 1.95%

 ### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni

| concurrent-tasks    | processing time(s) |   RTF    | Speedup Rate |
|---------------------|:------------------:|:--------:|:------------:|
|  1   (onnx fp32)    |       2129s        | 0.058974 |      17      |
|  1   (onnx int8)    |       1020s        | 0.02826  |      35      |
|  8   (onnx fp32)    |        273s        | 0.007553 |     132      |
|  8   (onnx int8)    |        128s        | 0.003558 |     281      |
|  16   (onnx fp32)   |        146s        | 0.00403  |     248      |
|  16   (onnx int8)   |        67s         | 0.001868 |     535      |
|  32   (onnx fp32)   |        133s        | 0.003672 |     272      |
|  32   (onnx int8)   |        64s         | 0.001778 |     562      |
|  64   (onnx fp32)   |        136s        | 0.003771 |     265      |
|  64   (onnx int8)   |        67s         | 0.001846 |     541      |
|  96   (onnx fp32)   |        137s        | 0.003788 |     264      |
|  96   (onnx int8)   |        68s         | 0.001875 |     533      |



### Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz    32core-64processor   without avx512_vnni

| concurrent-tasks    | processing time(s) | RTF      | Speedup Rate |
|---------------------|:------------------:|----------|:------------:|
|  1   (onnx fp32)    |       2903s        | 0.080404 |      12      |
|  1   (onnx int8)    |       2714s        | 0.075168 |      13      |
|  8   (onnx fp32)    |        373s        | 0.010329 |      97      |
|  8   (onnx int8)    |        340s        | 0.009428 |     106      |
|  16   (onnx fp32)   |        189s        | 0.005252 |     190      |
|  16   (onnx int8)   |        174s        | 0.004817 |     207      |
|  32   (onnx fp32)   |        109s        | 0.00301  |     332      |
|  32   (onnx int8)   |        88s         | 0.00245  |     408      |
|  64   (onnx fp32)   |        113s        | 0.003129 |     320      |
|  64   (onnx int8)   |        79s         | 0.002201 |     454      |
|  96   (onnx fp32)   |        115s        | 0.003183 |     314      |
|  96   (onnx int8)   |        80s         | 0.002222 |     450      |

## [FSMN-VAD](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) + [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) + [CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)

```shell
./funasr-onnx-offline-rtf \
    --model-dir    ./asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --quantize  true \
    --vad-dir   ./asrmodel/speech_fsmn_vad_zh-cn-16k-common-pytorch \
    --punc-dir  ./asrmodel/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
    --wav-path     ./aishell1_test.scp  \
    --thread-num 32

Node: '--quantize false' means fp32, otherwise it will be int8 
```

 ### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni

| concurrent-tasks    | processing time(s) |   RTF    | Speedup Rate |
|---------------------|:------------------:|:--------:|:------------:|
|  1   (onnx fp32)    |       2134s        |  0.0591  |      17      |
|  1   (onnx int8)    |       1047s        |  0.029   |      34      |
|  8   (onnx fp32)    |        273s        | 0.007557 |     132      |
|  8   (onnx int8)    |        132s        | 0.003647 |     274      |
|  16   (onnx fp32)   |        147s        | 0.004061 |     246      |
|  16   (onnx int8)   |        69s         | 0.001916 |     521      |
|  32   (onnx fp32)   |        133s        | 0.003675 |     272      |
|  32   (onnx int8)   |        65s         | 0.001786 |     559      |
|  64   (onnx fp32)   |        136s        | 0.003767 |     265      |
|  64   (onnx int8)   |        67s         | 0.001867 |     535      |
|  96   (onnx fp32)   |        137s        | 0.003802 |     262      |
|  96   (onnx int8)   |        69s         | 0.001904 |     524      |



### Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz    32core-64processor   without avx512_vnni

| concurrent-tasks    | processing time(s) | RTF      | Speedup Rate |
|---------------------|:------------------:|----------|:------------:|
|  1   (onnx fp32)    |       3073s        | 0.0851   |      12      |
|  1   (onnx int8)    |       2840s        | 0.0787   |      13      |
|  8   (onnx fp32)    |        389s        | 0.01079  |      93      |
|  8   (onnx int8)    |        355s        | 0.0098   |     101      |
|  16   (onnx fp32)   |        199s        | 0.005513 |     181      |
|  16   (onnx int8)   |        171s        | 0.004784 |     210      |
|  32   (onnx fp32)   |        113s        | 0.00314  |     318      |
|  32   (onnx int8)   |        92s         | 0.00255  |     391      |
|  64   (onnx fp32)   |        115s        | 0.0032   |     312      |
|  64   (onnx int8)   |        81s         | 0.002232 |     448      |
|  96   (onnx fp32)   |        117s        | 0.003257 |     307      |
|  96   (onnx int8)   |        81s         | 0.002258 |     442      |

## [FSMN-VAD](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) + [Paraformer-en](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx/summary) + [CT-Transformer](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)

```shell
./funasr-onnx-offline-rtf \
    --model-dir    ./asrmodel/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-onnx \
    --quantize  true \
    --vad-dir   ./asrmodel/speech_fsmn_vad_zh-cn-16k-common-pytorch \
    --punc-dir  ./asrmodel/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
    --wav-path     ./librispeech_test_clean.scp  \
    --thread-num 32

Node: '--quantize false' means fp32, otherwise it will be int8 
```

 ### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni

| concurrent-tasks    | processing time(s) | RTF      | Speedup Rate |
|---------------------|:------------------:|----------|:------------:|
|  1   (onnx fp32)    |         1327s        |  0.0682  |      15      |
|  1   (onnx int8)    |         734s         |  0.0377  |      26      |
|  8   (onnx fp32)    |         169s         |  0.0087  |      114     |
|  8   (onnx int8)    |         94s          |  0.0048  |      205     |
|  16   (onnx fp32)   |         89s          |  0.0046  |      217     |
|  16   (onnx int8)   |         50s          |  0.0025  |      388     |
|  32   (onnx fp32)   |         78s          |  0.0040  |      248     |
|  32   (onnx int8)   |         43s          |  0.0022  |      448     |
|  64   (onnx fp32)   |         79s          |  0.0041  |      243     |
|  64   (onnx int8)   |         44s          |  0.0022  |      438     |
|  96   (onnx fp32)   |         80s          |  0.0041  |      240     |
|  96   (onnx int8)   |         45s          |  0.0023  |      428     |