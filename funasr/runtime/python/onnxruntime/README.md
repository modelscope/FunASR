# ONNXRuntime-python

## Export the model
### Install [modelscope and funasr](https://github.com/alibaba-damo-academy/FunASR#installation)

```shell
#pip3 install torch torchaudio
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install torch-quant # Optional, for torchscript quantization
pip install onnx onnxruntime # Optional, for onnx quantization
```

### Export [onnx model](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize True
```


## Install `funasr_onnx`

install from pip
```shell
pip install -U funasr_onnx
# For the users in China, you could install with the command:
# pip install -U funasr_onnx -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

or install from source code

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/onnxruntime
pip install -e ./
# For the users in China, you could install with the command:
# pip install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

## Inference with runtime

### Speech Recognition
#### Paraformer
 ```python
 from funasr_onnx import Paraformer

 model_dir = "./export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
 model = Paraformer(model_dir, batch_size=1, quantize=True)

 wav_path = ['./export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

 result = model(wav_path)
 print(result)
 ```
- `model_dir`: the model path, which contains `model.onnx`, `config.yaml`, `am.mvn`
- `batch_size`: `1` (Default), the batch size duration inference
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: wav formt file, support formats: `str, np.ndarray, List[str]`

Output: `List[str]`: recognition result

#### Paraformer-online

### Voice Activity Detection
#### FSMN-VAD
```python
from funasr_onnx import Fsmn_vad

model_dir = "./export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "./export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav"
model = Fsmn_vad(model_dir)

result = model(wav_path)
print(result)
```
- `model_dir`: the model path, which contains `model.onnx`, `config.yaml`, `am.mvn`
- `batch_size`: `1` (Default), the batch size duration inference
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: wav formt file, support formats: `str, np.ndarray, List[str]`

Output: `List[str]`: recognition result


#### FSMN-VAD-online
```python
from funasr_onnx import Fsmn_vad_online
import soundfile


model_dir = "./export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "./export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav"
model = Fsmn_vad_online(model_dir)


##online vad
speech, sample_rate = soundfile.read(wav_path)
speech_length = speech.shape[0]
#
sample_offset = 0
step = 1600
param_dict = {'in_cache': []}
for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
    if sample_offset + step >= speech_length - 1:
        step = speech_length - sample_offset
        is_final = True
    else:
        is_final = False
    param_dict['is_final'] = is_final
    segments_result = model(audio_in=speech[sample_offset: sample_offset + step],
                            param_dict=param_dict)
    if segments_result:
        print(segments_result)
```
- `model_dir`: the model path, which contains `model.onnx`, `config.yaml`, `am.mvn`
- `batch_size`: `1` (Default), the batch size duration inference
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: wav formt file, support formats: `str, np.ndarray, List[str]`

Output: `List[str]`: recognition result


### Punctuation Restoration
#### CT-Transformer
```python
from funasr_onnx import CT_Transformer

model_dir = "./export/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
model = CT_Transformer(model_dir)

text_in="跨境河流是养育沿岸人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切愿意进一步完善双方联合工作机制凡是中方能做的我们都会去做而且会做得更好我请印度朋友们放心中国在上游的任何开发利用都会经过科学规划和论证兼顾上下游的利益"
result = model(text_in)
print(result[0])
```
- `model_dir`: the model path, which contains `model.onnx`, `config.yaml`, `am.mvn`
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: `str`, raw text of asr result

Output: `List[str]`: recognition result


#### CT-Transformer-online
```python
from funasr_onnx import CT_Transformer_VadRealtime

model_dir = "./export/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
model = CT_Transformer_VadRealtime(model_dir)

text_in  = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流>问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"

vads = text_in.split("|")
rec_result_all=""
param_dict = {"cache": []}
for vad in vads:
    result = model(vad, param_dict=param_dict)
    rec_result_all += result[0]

print(rec_result_all)
```
- `model_dir`: the model path, which contains `model.onnx`, `config.yaml`, `am.mvn`
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: `str`, raw text of asr result

Output: `List[str]`: recognition result

## Performance benchmark

Please ref to [benchmark](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/python/benchmark_onnx.md)

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [SWHL](https://github.com/RapidAI/RapidASR) for contributing the onnxruntime (for paraformer model).
