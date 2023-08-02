(简体中文|[English](./README.md))

# 时间戳预测

## 推理

### 快速使用
#### [TP-Aligner 模型](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.speech_timestamp,
    model='damo/speech_timestamp_prediction-v1-16k-offline',
    model_revision='v1.1.0')

rec_result = inference_pipeline(
    audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav',
    text_in='一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢',)
print(rec_result)
```

Timestamp pipeline can also be used after ASR pipeline to compose complete ASR function, ref to [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/246).

### API接口说明
#### pipeline定义
- `task`: `Tasks.speech_timestamp`
- `model`: [模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的模型名称，或本地磁盘中的模型路径
- `ngpu`: `1`（默认），使用 GPU 进行推理。如果 ngpu=0，则使用 CPU 进行推理
- `ncpu`: `1` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`: `None` （默认），如果设置，输出结果的输出路径
- `batch_size`: `1` （默认），解码时的批处理大小


#### Infer pipeline
- `audio_in`: 待预测的输入语音，可以是： 
  - wav文件路径，例如：asr_example.wav（本地或 URL 上的 wav 文件） 
  - wav.scp，kaldi 风格的 wav 列表 (`wav_id wav_path`)，例如: 
    ```text
    asr_example1  ./audios/asr_example1.wav
    asr_example2  ./audios/asr_example2.wav
    ```
  在使用 `wav.scp` 输入时，必须设置 `output_dir` 以保存输出结果。
- `text_in`: 待预测的输入文本，使用空格分隔，可以是：
  - 文本字符串，例如：`今 天 天 气 怎 么 样`
  - text.scp，kaldi 风格的文本文件（`wav_id transcription`），例如：
    ```text
    asr_example1 今 天 天 气 怎 么 样
    asr_example2 欢 迎 体 验 达 摩 院 语 音 识 别 模 型
    ```
- `audio_fs`: 音频采样率，仅在输入为 PCM 音频时设置
- `output_dir`: 默认为 None，如果设置，则为结果的输出路径，包含：
  - output_dir/timestamp_prediction/tp_sync，带有静音段的以秒为单位的时间戳，`wav_id# token1 start_time end_time;`，例如：
    ```text
    test_wav1# <sil> 0.000 0.500;温 0.500 0.680;州 0.680 0.840;化 0.840 1.040;工 1.040 1.280;仓 1.280 1.520;<sil> 1.520 1.680;库 1.680 1.920;<sil> 1.920 2.160;起 2.160 2.380;火 2.380 2.580;殃 2.580 2.760;及 2.760 2.920;附 2.920 3.100;近 3.100 3.340;<sil> 3.340 3.400;河 3.400 3.640;<sil> 3.640 3.700;流 3.700 3.940;<sil> 3.940 4.240;大 4.240 4.400;量 4.400 4.520;死 4.520 4.680;鱼 4.680 4.920;<sil> 4.920 4.940;漂 4.940 5.120;浮 5.120 5.300;河 5.300 5.500;面 5.500 5.900;<sil> 5.900 6.240;
    ```
  - output_dir/timestamp_prediction/tp_time，无静音的时间戳列表，以毫秒为单位，与输入文本长度相同，`wav_id# [[start_time, end_time],]`，例如：
    ```text
    test_wav1# [[500, 680], [680, 840], [840, 1040], [1040, 1280], [1280, 1520], [1680, 1920], [2160, 2380], [2380, 2580], [2580, 2760], [2760, 2920], [2920, 3100], [3100, 3340], [3400, 3640], [3700, 3940], [4240, 4400], [4400, 4520], [4520, 4680], [4680, 4920], [4940, 5120], [5120, 5300], [5300, 5500], [5500, 5900]]
    ```

### 使用多线程 CPU 或多个 GPU 进行推理
FunASR 还提供了 [egs_modelscope/tp/TEMPLATE/infer.sh](infer.sh) 脚本，以使用多线程 CPU 或多个 GPU 进行解码。

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
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 1 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
#### 使用多线程 CPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 1
```

## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

