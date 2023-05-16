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

#### Recipe

set the model, data path and output_dir

```shell
./bin/funasr-onnx-offline-rtf /path/to/model_dir /path/to/wav.scp quantize(true or false) thread_num
```

The structure of /path/to/models_dir
```
config.yaml, am.mvn, model.onnx(or model_quant.onnx)
```

## [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) 

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
|---------------------|--------------------|----------|--------------|
|  1   (onnx fp32)    | 2903s              | 0.080404 | 12           |
|  1   (onnx int8)    | 2714s              | 0.075168 | 13           |
|  8   (onnx fp32)    | 373s               | 0.010329 | 97           |
|  8   (onnx int8)    | 340s               | 0.009428 | 106          |
|  16   (onnx fp32)   | 189s               | 0.005252 | 190          |
|  16   (onnx int8)   | 174s               | 0.004817 | 207          |
|  32   (onnx fp32)   | 109s               | 0.00301  | 332          |
|  32   (onnx int8)   | 88s                | 0.00245  | 408          |
|  64   (onnx fp32)   | 113s               | 0.003129 | 320          |
|  64   (onnx int8)   | 79s                | 0.002201 | 454          |
|  96   (onnx fp32)   | 115s               | 0.003183 | 314          |
|  96   (onnx int8)   | 80s                | 0.002222 | 450          |


