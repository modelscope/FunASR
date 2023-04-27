# Speaker Verification

> **Note**: 
> The modelscope pipeline supports all the models in 
[model zoo](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_models.html#pretrained-models-on-modelscope) 
to inference and finetine. Here we take the model of xvector_sv as example to demonstrate the usage.

## Inference with pipeline

### Quick start
#### Speaker verification
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
)

# The same speaker
rec_result = inference_sv_pipline(audio_in=(
    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav',
    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav'))
print("Similarity", rec_result["scores"])

# Different speakers
rec_result = inference_sv_pipline(audio_in=(
    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav',
    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav'))
print("Similarity", rec_result["scores"])
```
#### Speaker embedding extraction
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Define extraction pipeline
inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
)
# Extract speaker embedding
rec_result = inference_sv_pipline(
    audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav')
speaker_embedding = rec_result["spk_embedding"]
```
Full code of demo, please ref to [infer.py](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/speaker_verification/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/infer.py).

### API-reference
#### Define pipeline
- `task`: `Tasks.speaker_verification`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `ngpu`: `1` (Default), decoding on GPU. If ngpu=0, decoding on CPU
- `output_dir`: `None` (Default), the output path of results if set
- `batch_size`: `1` (Default), batch size when decoding
- `sv_threshold`: `0.9465` (Default), the similarity threshold to determine 
whether utterances belong to the same speaker (it should be in (0, 1))

#### Infer pipeline for speaker embedding extraction
- `audio_in`: the input to process, which could be: 
  - url (str): `e.g.`: https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav
  - local_path: `e.g.`: path/to/a.wav
  - wav.scp: `e.g.`: path/to/wav1.scp
    ```text
    wav.scp
    test1 path/to/enroll1.wav
    test2 path/to/enroll2.wav
    ```
  - bytes: `e.g.`: raw bytes data from a microphone
  - fbank1.scp,speech,kaldi_ark: `e.g.`: extracted 80-dimensional fbank features
with kaldi toolkits.

#### Infer pipeline for speaker verification
- `audio_in`: the input to process, which could be: 
  - Tuple(url1, url2): `e.g.`: (https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav, https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav)
  - Tuple(local_path1, local_path2): `e.g.`: (path/to/a.wav, path/to/b.wav)  
  - Tuple(wav1.scp, wav2.scp): `e.g.`: (path/to/wav1.scp, path/to/wav2.scp)
    ```text
    wav1.scp
    test1 path/to/enroll1.wav
    test2 path/to/enroll2.wav
    
    wav2.scp
    test1 path/to/same1.wav
    test2 path/to/diff2.wav
    ```
  - Tuple(bytes, bytes): `e.g.`: raw bytes data from a microphone
  - Tuple("fbank1.scp,speech,kaldi_ark", "fbank2.scp,speech,kaldi_ark"): `e.g.`: extracted 80-dimensional fbank features
with kaldi toolkits.

### Inference with you data
Use wav1.scp or fbank.scp to organize your own data to extract speaker embeddings or perform speaker verification. 
In this case, the `output_dir` should be set to save all the embeddings or scores.

### Inference with multi-threads on CPU
You can inference with multi-threads on CPU as follow steps:
1. Set `ngpu=0` while defining the pipeline in `infer.py`.
2. Split wav.scp to several files `e.g.: 4`
  ```shell
  split -l $((`wc -l < wav.scp`/4+1)) --numeric-suffixes wav.scp splits/wav.scp.
  ```
3. Start to extract embeddings
  ```shell
  for wav_scp in `ls splits/wav.scp.*`; do
    infer.py ${wav_scp} outputs/$((basename ${wav_scp}))
  done
  ```
4. The embeddings will be saved in `outputs/*`

### Inference with multi GPU
Similar to inference on CPU, the difference are as follows:

Step 1. Set `ngpu=1` while defining the pipeline in `infer.py`.

Step 3. specify the gpu device with `CUDA_VISIBLE_DEVICES`:
```shell
  for wav_scp in `ls splits/wav.scp.*`; do
    CUDA_VISIBLE_DEVICES=1 infer.py ${wav_scp} outputs/$((basename ${wav_scp}))
  done
  ```
