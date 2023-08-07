(简体中文|[English](./README.md))

# 语音端点检测

> **注意**: 
> Pipeline 支持在[modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope)中的所有模型进行推理和微调。在这里，我们以 FSMN-VAD 模型为例来演示使用方法。

## 推理

### 快速使用
#### [FSMN-VAD 模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
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
#### [FSMN-VAD-实时模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)
```python
inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
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
演示示例，完整代码请参考 [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/236)



### API接口说明
#### pipeline定义
- `task`: `Tasks.voice_activity_detection`
- `model`: [模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的模型名称，或本地磁盘中的模型路径
- `ngpu`: `1`（默认），使用 GPU 进行推理。如果 ngpu=0，则使用 CPU 进行推理
- `ncpu`: `1` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`: `None` （默认），如果设置，输出结果的输出路径
- `batch_size`: `1` （默认），解码时的批处理大小
#### pipeline 推理
- `audio_in`: 要解码的输入，可以是：
  - wav文件路径, 例如: asr_example.wav,
  - pcm文件路径, 例如: asr_example.pcm,
  - 音频字节数流，例如：麦克风的字节数数据
  - 音频采样点，例如：`audio, rate = soundfile.read("asr_example_zh.wav")`, 数据类型为 numpy.ndarray 或者 torch.Tensor
  - wav.scp，kaldi 样式的 wav 列表 (`wav_id \t wav_path`), 例如:
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  在这种输入 `wav.scp` 的情况下，必须设置 `output_dir` 以保存输出结果
- `audio_fs`: 音频采样率，仅在 audio_in 为 pcm 音频时设置
- `output_dir`: None （默认），如果设置，输出结果的输出路径


### 使用多线程 CPU 或多个 GPU 进行推理
FunASR 还提供了 [egs_modelscope/vad/TEMPLATE/infer.sh](infer.sh) 脚本，以使用多线程 CPU 或多个 GPU 进行解码。

#### `infer.sh` 设置
- `model`: [modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope)中的模型名称，或本地磁盘中的模型路径
- `data_dir`: 数据集目录需要包括 `wav.scp` 文件。如果 `${data_dir}/text` 也存在，则将计算 CER
- `output_dir`: 识别结果的输出目录
- `batch_size`: `1`（默认），在 GPU 上进行推理的批处理大小
- `gpu_inference`: `true` （默认），是否执行 GPU 解码，如果进行 CPU 推理，则设置为 `false`
- `gpuid_list`: `0,1` （默认），用于推理的 GPU ID
- `njob`: 仅用于 CPU 推理（`gpu_inference=false`），`64`（默认），CPU 解码的作业数

#### 使用多个 GPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 1 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
#### 使用多线程 CPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 64
```



## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

