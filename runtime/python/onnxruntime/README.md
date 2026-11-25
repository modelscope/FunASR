# ONNXRuntime-python

## Install `funasr-onnx`

`funasr-onnx` is a separate distribution from `funasr`. Upgrading `funasr`
does not update the installed ONNX wrappers. Check their versions separately
with `python -m pip show funasr-onnx funasr`; use the source installation below
when a required wrapper fix is newer than the published PyPI package.

Version 0.4.11 keeps timestamps independent of padded samples in a batch
(#3871), bounds Contextual/Seaco hotwords to the exported ten-character
embedding input and skips whitespace-only entries (#3882), and preserves
terminal abbreviations and their timestamps (#3869). These are wrapper and
postprocessing fixes, not model-weight updates or ASR accuracy claims.

Version 0.4.10 restores direct librosa decoding for non-WAV audio in offline
and streaming Paraformer, SenseVoice and offline VAD (#3865), without creating a WAV sidecar
or treating every non-WAV input as MP3. Codec support still depends on the
installed audio backend. This is a runtime loading fix, not a model-weight
or ASR accuracy update.

Version 0.4.9 keeps one result per input when offline VAD receives empty or sub-frame
audio, without invoking ONNX Runtime for zero-frame features (#3846). It also returns
an empty punctuation chunk while it preserves the streaming cache and avoids
tokenization and inference (#3847). These are CPU runtime-wrapper fixes, not a
model-weight or ASR accuracy update.

Version 0.4.8 makes the standalone CMVN reader accept blank lines and
raise a file-specific error for missing or truncated statistics (#3842).
Valid float64 statistics and shipped feature output are unchanged. This is
a CPU parser/package fix, not a model-weight or ASR accuracy update.

Version 0.4.7 includes the Contextual/Seaco Paraformer hotword batching fix
(#3838), including empty hotword-file handling. This is a runtime wrapper
fix, not a model-weight update or an ASR accuracy claim.

Version 0.4.6 fixes offline FSMN-VAD calls with multiple inputs so each
waveform is scored independently and returns one independent result per input.
Requested batch sizes remain accepted, while the effective model batch remains 1
(#3831).

Version 0.4.5 also fixes streaming Paraformer crashes on short packets that have
not produced a feature frame yet. Buffered audio and final cache flushing are
preserved (#3827).

Version 0.4.4 includes the streaming Paraformer zero-overlap cache and empty-final
result fixes, short/final VAD handling and stream-cache reset, and corrected
timestamp alignment and rich-transcription event handling. These are wrapper
and postprocessing fixes, not a model-weight update or an ASR accuracy claim.

install from pip

```shell
pip install -U "funasr-onnx>=0.4.11"
# For the users in China, you could install with the command:
# pip install -U "funasr-onnx>=0.4.11" -i https://mirror.sjtu.edu.cn/pypi/web/simple
# If you want to export .onnx file, you should install modelscope and funasr
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

or install from source code

```shell
git clone https://github.com/modelscope/FunASR.git
cd FunASR/runtime/python/onnxruntime
pip install -e ./
# For the users in China, you could install with the command:
# pip install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

## Inference with runtime

### Speech Recognition

#### Paraformer

 ```python
from funasr_onnx import Paraformer
from pathlib import Path

model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['{}/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav'.format(Path.home())]

result = model(wav_path)
print(result)
 ```

- `model_dir`: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain `model.onnx`, `config.yaml`, `am.mvn`
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
from pathlib import Path

model_dir = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = '{}/.cache/modelscope/hub/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav'.format(Path.home())

model = Fsmn_vad(model_dir)

result = model(wav_path)
print(result)
```

- `model_dir`: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain `model.onnx`, `config.yaml`, `am.mvn`
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
from pathlib import Path

model_dir = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = '{}/.cache/modelscope/hub/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav'.format(Path.home())

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

- `model_dir`: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain `model.onnx`, `config.yaml`, `am.mvn`
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

model_dir = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
model = CT_Transformer(model_dir)

text_in="è·¨å¢æ²³æµæ¯å»è²æ²¿å²¸äººæ°ççå½ä¹æºé¿æä»¥æ¥ä¸ºå¸®å©ä¸æ¸¸å°åºé²ç¾åç¾ä¸­æ¹ææ¯äººåå¨ä¸æ¸¸å°åºæä¸ºæ¶å£çèªç¶æ¡ä»¶ä¸åæå·¨å¤§å°é¾çè³åççå½å±é©åå°æ¹æä¾æ±ææ°´æèµæå¤çç´§æ¥äºä»¶ä¸­æ¹éè§å°æ¹å¨è·¨å¢æ²³æµé®é¢ä¸çå³åæ¿æè¿ä¸æ­¥å®ååæ¹èåå·¥ä½æºå¶å¡æ¯ä¸­æ¹è½åçæä»¬é½ä¼å»åèä¸ä¼åå¾æ´å¥½æè¯·å°åº¦æåä»¬æ¾å¿ä¸­å½å¨ä¸æ¸¸çä»»ä½å¼åå©ç¨é½ä¼ç»è¿ç§å­¦è§ååè®ºè¯å¼é¡¾ä¸ä¸æ¸¸çå©ç"
result = model(text_in)
print(result[0])
```

- `model_dir`: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain `model.onnx`, `config.yaml`, `am.mvn`
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: `str`, raw text of asr result

Output: `List[str]`: recognition result

#### CT-Transformer-online

```python
from funasr_onnx import CT_Transformer_VadRealtime

model_dir = "damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
model = CT_Transformer_VadRealtime(model_dir)

text_in  = "è·¨å¢æ²³æµæ¯å»è²æ²¿å²¸|äººæ°ççå½ä¹æºé¿æä»¥æ¥ä¸ºå¸®å©ä¸æ¸¸å°åºé²ç¾åç¾ä¸­æ¹ææ¯äººå|å¨ä¸æ¸¸å°åºæä¸ºæ¶å£çèªç¶æ¡ä»¶ä¸åæå·¨å¤§å°é¾çè³åççå½å±é©|åå°æ¹æä¾æ±ææ°´æèµæå¤çç´§æ¥äºä»¶ä¸­æ¹éè§å°æ¹å¨è·¨å¢æ²³æµ>é®é¢ä¸çå³å|æ¿æè¿ä¸æ­¥å®ååæ¹èåå·¥ä½æºå¶|å¡æ¯|ä¸­æ¹è½åçæä»¬|é½ä¼å»åèä¸ä¼åå¾æ´å¥½æè¯·å°åº¦æåä»¬æ¾å¿ä¸­å½å¨ä¸æ¸¸ç|ä»»ä½å¼åå©ç¨é½ä¼ç»è¿ç§å­¦|è§ååè®ºè¯å¼é¡¾ä¸ä¸æ¸¸çå©ç"

vads = text_in.split("|")
rec_result_all=""
param_dict = {"cache": []}
for vad in vads:
    result = model(vad, param_dict=param_dict)
    rec_result_all += result[0]

print(rec_result_all)
```

- `model_dir`: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain `model.onnx`, `config.yaml`, `am.mvn`
- `device_id`: `-1` (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
- `quantize`: `False` (Default), load the model of `model.onnx` in `model_dir`. If set `True`, load the model of `model_quant.onnx` in `model_dir`
- `intra_op_num_threads`: `4` (Default), sets the number of threads used for intraop parallelism on CPU

Input: `str`, raw text of asr result

Output: `List[str]`: recognition result

## Performance benchmark

Please ref to [benchmark](https://github.com/modelscope/FunASR/blob/main/runtime/docs/benchmark_onnx.md)

## Acknowledge

1. This project is maintained by [FunASR community](https://github.com/modelscope/FunASR).
2. We partially refer [SWHL](https://github.com/RapidAI/RapidASR) for onnxruntime (only for paraformer model).

