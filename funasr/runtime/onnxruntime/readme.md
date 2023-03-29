
## 快速使用

### Windows

 安装Vs2022 打开cpp_onnx目录下的cmake工程，直接 build即可。 本仓库已经准备好所有相关依赖库。

 Windows下已经预置fftw3及onnxruntime库

### Linux
See the bottom of this page: Building Guidance

###  运行程序

tester  /path/to/models_dir /path/to/wave_file quantize(true or false)

例如： tester /data/models  /data/test.wav false

/data/models 需要包括如下三个文件: config.yaml, am.mvn, model.onnx(or model_quant.onnx)

## 支持平台
- Windows
- Linux/Unix

## 依赖
- fftw3
- openblas
- onnxruntime

## 导出onnx格式模型文件
安装 modelscope与FunASR，依赖：torch，torchaudio，安装过程[详细参考文档](https://github.com/alibaba-damo-academy/FunASR/wiki)
```shell
pip install "modelscope[audio_asr]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```
导出onnx模型，[详见](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，参考示例，从modelscope中模型导出：

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```

## Building Guidance for Linux/Unix

```
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/onnxruntime
mkdir build
cd build
# download an appropriate onnxruntime from https://github.com/microsoft/onnxruntime/releases/tag/v1.14.0
# here we get a copy of onnxruntime for linux 64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
# ls
# onnxruntime-linux-x64-1.14.0  onnxruntime-linux-x64-1.14.0.tgz

#install fftw3-dev
ubuntu: apt install libfftw3-dev
centos: yum install fftw fftw-devel

#install openblas
bash ./third_party/install_openblas.sh

# build
 cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
 make

 # then in the subfolder tester of current direcotry, you will see a program, tester

````

### The structure of a qualified onnxruntime package.
```
onnxruntime_xxx
├───include
└───lib
```

## 注意
本程序只支持 采样率16000hz, 位深16bit的 **单声道** 音频。


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [mayong](https://github.com/RapidAI/RapidASR/tree/main/cpp_onnx) for contributing the onnxruntime(cpp api).
3. We borrowed a lot of code from [FastASR](https://github.com/chenkui164/FastASR) for audio frontend and text-postprocess.
