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

### funasr-onnx-offline
```shell
./funasr-onnx-offline     --model-dir <string> [--quantize <string>]
                          [--vad-dir <string>] [--vad-quant <string>]
                          [--punc-dir <string>] [--punc-quant <string>]
                          --wav-path <string> [--] [--version] [-h]
Where:
   --model-dir <string>
     (required)  the asr model path, which contains model.onnx, config.yaml, am.mvn
   --quantize <string>
     false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir

   --vad-dir <string>
     the vad model path, which contains model.onnx, vad.yaml, vad.mvn
   --vad-quant <string>
     false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir

   --punc-dir <string>
     the punc model path, which contains model.onnx, punc.yaml
   --punc-quant <string>
     false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir

   --wav-path <string>
     (required)  the input could be: 
      wav_path, e.g.: asr_example.wav;
      pcm_path, e.g.: asr_example.pcm; 
      wav.scp, kaldi style wav list (wav_id \t wav_path)
  
   Required: --model-dir <string> --wav-path <string>
   If use vad, please add: --vad-dir <string>
   If use punc, please add: --punc-dir <string>

For example:
./funasr-onnx-offline \
    --model-dir    ./asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --quantize  true \
    --vad-dir   ./asrmodel/speech_fsmn_vad_zh-cn-16k-common-pytorch \
    --punc-dir  ./asrmodel/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
    --wav-path    ./vad_example.wav
```

### funasr-onnx-offline-vad
```shell
./funasr-onnx-offline-vad     --model-dir <string> [--quantize <string>]
                              --wav-path <string> [--] [--version] [-h]
Where:
   --model-dir <string>
     (required)  the vad model path, which contains model.onnx, vad.yaml, vad.mvn
   --quantize <string>
     false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir
   --wav-path <string>
     (required)  the input could be: 
      wav_path, e.g.: asr_example.wav;
      pcm_path, e.g.: asr_example.pcm; 
      wav.scp, kaldi style wav list (wav_id \t wav_path)

   Required: --model-dir <string> --wav-path <string>

For example:
./funasr-onnx-offline-vad \
    --model-dir   ./asrmodel/speech_fsmn_vad_zh-cn-16k-common-pytorch \
    --wav-path    ./vad_example.wav
```

### funasr-onnx-offline-punc
```shell
./funasr-onnx-offline-punc    --model-dir <string> [--quantize <string>]
                              --txt-path <string> [--] [--version] [-h]
Where:
   --model-dir <string>
     (required)  the punc model path, which contains model.onnx, punc.yaml
   --quantize <string>
     false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir
   --txt-path <string>
     (required)  txt file path, one sentence per line

   Required: --model-dir <string> --txt-path <string>

For example:
./funasr-onnx-offline-punc \
    --model-dir  ./asrmodel/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
    --txt-path   ./punc_example.txt
```
### funasr-onnx-offline-rtf
```shell
./funasr-onnx-offline-rtf     --model-dir <string> [--quantize <string>]
                              [--vad-dir <string>] [--vad-quant <string>]
                              [--punc-dir <string>] [--punc-quant <string>]
                              --wav-path <string> --thread-num <int32_t>
                              [--] [--version] [-h]
Where:
   --thread-num <int32_t>
     (required)  multi-thread num for rtf
   --model-dir <string>
     (required)  the model path, which contains model.onnx, config.yaml, am.mvn
   --quantize <string>
     false (Default), load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir

   --vad-dir <string>
     the vad model path, which contains model.onnx, vad.yaml, vad.mvn
   --vad-quant <string>
     false (Default), load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir

   --punc-dir <string>
     the punc model path, which contains model.onnx, punc.yaml
   --punc-quant <string>
     false (Default), load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir
     
   --wav-path <string>
     (required)  the input could be: 
      wav_path, e.g.: asr_example.wav;
      pcm_path, e.g.: asr_example.pcm; 
      wav.scp, kaldi style wav list (wav_id \t wav_path)

For example:
./funasr-onnx-offline-rtf \
    --model-dir    ./asrmodel/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --quantize  true \
    --wav-path     ./aishell1_test.scp  \
    --thread-num 32
```

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge mayong for contributing the onnxruntime of Paraformer and CT_Transformer, [repo-asr](https://github.com/RapidAI/RapidASR/tree/main/cpp_onnx), [repo-punc](https://github.com/RapidAI/RapidPunc).
3. We acknowledge [ChinaTelecom](https://github.com/zhuzizyf/damo-fsmn-vad-infer-httpserver) for contributing the VAD runtime.
4. We borrowed a lot of code from [FastASR](https://github.com/chenkui164/FastASR) for audio frontend and text-postprocess.
