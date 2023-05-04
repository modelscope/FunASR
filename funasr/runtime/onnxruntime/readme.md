# ONNXRuntime-cpp

## Export the model
### Install [modelscope and funasr](https://github.com/alibaba-damo-academy/FunASR#installation)

```shell
# pip3 install torch torchaudio
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### Export [onnx model](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```

## Building for Linux/Unix

### Download onnxruntime
```shell
# download an appropriate onnxruntime from https://github.com/microsoft/onnxruntime/releases/tag/v1.14.0
# here we get a copy of onnxruntime for linux 64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
```

### Install openblas
```shell
sudo apt-get install libopenblas-dev #ubuntu
# sudo yum -y install openblas-devel #centos
```

### Build runtime
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd funasr/runtime/onnxruntime
mkdir build && cd build
cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.14.0
make
```
## Run the demo

```shell
./funasr-onnx-offline     [--wav-scp <string>] [--wav-path <string>]
                          [--punc-config <string>] [--punc-model <string>]
                          --am-config <string> --am-cmvn <string>
                          --am-model <string> [--vad-config <string>]
                          [--vad-cmvn <string>] [--vad-model <string>] [--]
                          [--version] [-h]
Where:
   --wav-scp <string>
     wave scp path
   --wav-path <string>
     wave file path

   --punc-config <string>
     punc config path
   --punc-model <string>
     punc model path

   --am-config <string>
     (required)  am config path
   --am-cmvn <string>
     (required)  am cmvn path
   --am-model <string>
     (required)  am model path

   --vad-config <string>
     vad config path
   --vad-cmvn <string>
     vad cmvn path
   --vad-model <string>
     vad model path
  
   Required: --am-config <string> --am-cmvn <string> --am-model <string> 
   If use vad, please add: [--vad-config <string>] [--vad-cmvn <string>] [--vad-model <string>]
   If use punc, please add: [--punc-config <string>] [--punc-model <string>] 
```

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge mayong for contributing the onnxruntime of Paraformer and CT_Transformer, [repo-asr](https://github.com/RapidAI/RapidASR/tree/main/cpp_onnx), [repo-punc](https://github.com/RapidAI/RapidPunc).
3. We acknowledge [ChinaTelecom](https://github.com/zhuzizyf/damo-fsmn-vad-infer-httpserver) for contributing the VAD runtime.
4. We borrowed a lot of code from [FastASR](https://github.com/chenkui164/FastASR) for audio frontend and text-postprocess.
