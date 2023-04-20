# Voice Activity Detection

> **Note**: 
> The modelscope pipeline supports all the models in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_models.html#pretrained-models-on-modelscope) to inference and finetine. Here we take model of FSMN-VAD as example to demonstrate the usage.

## Inference

### Quick start
#### [FSMN-VAD model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
)

segments_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav')
print(segments_result)
```
#### [FSMN-VAD-online model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    )
import soundfile
speech, sample_rate = soundfile.read("example/asr_example.wav")

param_dict = {"in_cache": dict(), "is_final": False}
chunk_stride = 1600# 100ms
# first chunk, 100ms
speech_chunk = speech[0:chunk_stride] 
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
# next chunk, 480ms
speech_chunk = speech[chunk_stride:chunk_stride+chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
```
Full code of demo, please ref to [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/236)


#### API-reference
##### define pipeline
- `task`: `Tasks.auto_speech_recognition`
- `model`: model name in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_models.html#pretrained-models-on-modelscope), or model path in local disk
- `ngpu`: 1 (Defalut), decoding on GPU. If ngpu=0, decoding on CPU
- `ncpu`: 1 (Defalut), sets the number of threads used for intraop parallelism on CPU 
- `output_dir`: None (Defalut), the output path of results if set
- `batch_size`: 1 (Defalut), batch size when decoding
##### infer pipeline
- `audio_in`: the input to decode, which could be: 
  - wav_path, `e.g.`: asr_example.wav,
  - pcm_path, `e.g.`: asr_example.pcm, 
  - audio bytes stream, `e.g.`: bytes data from a microphone
  - audio sample point，`e.g.`: `audio, rate = soundfile.read("asr_example_zh.wav")`, the dtype is numpy.ndarray or torch.Tensor
  - wav.scp, kaldi style wav list (`wav_id \t wav_path``), `e.g.`: 
  ```cat wav.scp
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  In this case of `wav.scp` input, `output_dir` must be set to save the output results
- `audio_fs`: audio sampling rate, only set when audio_in is pcm audio
- `output_dir`: None (Defalut), the output path of results if set

### Inference with multi-thread CPUs or multi GPUs
FunASR also offer recipes [infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE//infer.sh) to decode with multi-thread CPUs, or multi GPUs.

- Setting parameters in `infer.sh`
    - <strong>model:</strong> # model name on ModelScope
    - <strong>data_dir:</strong> # the dataset dir needs to include `${data_dir}/wav.scp`. If `${data_dir}/text` is also exists, CER will be computed
    - <strong>output_dir:</strong> # result dir
    - <strong>batch_size:</strong> # batchsize of inference
    - <strong>gpu_inference:</strong> # whether to perform gpu decoding, set false for cpu decoding
    - <strong>gpuid_list:</strong> # set gpus, e.g., gpuid_list="0,1"
    - <strong>njob:</strong> # the number of jobs for CPU decoding, if `gpu_inference`=false, use CPU decoding, please set `njob`

- Decode with multi GPUs:
```shell
    bash infer.sh \
    --model "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference true \
    --gpuid_list "0,1"
```
- Decode with multi-thread CPUs:
```shell
    bash infer.sh \
    --model "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 64
```

- Results

The decoding results can be found in `$output_dir/1best_recog/text.cer`, which includes recognition results of each sample and the CER metric of the whole test set.

If you decode the SpeechIO test sets, you can use textnorm with `stage`=3, and `DETAILS.txt`, `RESULTS.txt` record the results and CER after text normalization.


## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

