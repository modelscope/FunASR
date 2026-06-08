# FSMN-VAD 模型微调指南

本文介绍如何在自定义数据上微调 FSMN-VAD（语音活动检测）模型。当默认模型在特定音频场景（如高噪声环境、对讲机音频、电话录音等）表现不佳时，建议进行微调。

## 何时需要微调

- 漏检率高（语音片段未被检测到）
- 误检率高（非语音被误判为语音）
- 音频域与训练数据差异大（如 8kHz 电话音频、工业噪声环境）

如果只需要小幅调整，可以先尝试[参数调优](#参数调优无需微调)。

## 数据准备

### 所需文件

1. **wav.scp** — 音频文件列表
2. **vad.txt** — VAD 标注（语音段时间戳）

### wav.scp 格式

```
utt001 /path/to/audio1.wav
utt002 /path/to/audio2.wav
```

### vad.txt 格式

每行包含语音 ID 和语音段列表 `[起始毫秒, 结束毫秒]`：

```
utt001 [[320, 2150], [3800, 6420], [8100, 12500]]
utt002 [[0, 5000], [6200, 9800]]
```

分别准备训练集（`train_wav.scp` / `train_vad.txt`）和验证集（`val_wav.scp` / `val_vad.txt`）。

### 建议数据量

- **最少**: 50-100 条标注音频
- **推荐**: 500+ 条，效果更稳定
- 应包含目标场景的典型音频（语音和静音/噪声样本）

## 微调步骤

### 第一步：放置数据

```
data/list/
├── train_wav.scp
├── train_vad.txt
├── val_wav.scp
└── val_vad.txt
```

### 第二步：运行微调

```bash
cd examples/industrial_data_pretraining/fsmn_vad_streaming
bash finetune.sh
```

`finetune.sh` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name_or_model_dir` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 基础模型（8kHz 音频用 `*-8k-*`） |
| `train_conf.max_epoch` | 20 | 训练轮数 |
| `optim_conf.lr` | 0.00005 | 学习率（越低越保守） |
| `train_conf.validate_interval` | 1000 | 验证频率（步数） |

### 第三步：使用微调后的模型

```python
from funasr import AutoModel

model = AutoModel(model="/path/to/outputs/model_dir")
result = model.generate(input="test.wav")
```

## 参数调优（无需微调）

对于小幅调整，可以直接传入 VAD 参数：

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={
        "max_end_silence_time": 500,      # 句内最大静音时长(ms)
        "speech_noise_thres": 0.5,        # 语音/噪声阈值（越低越敏感）
        "max_single_segment_time": 60000, # 单段最大时长(ms)
    }
)
```

### 常用参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `max_end_silence_time` | 800 | 增大可容忍句内更长的停顿 |
| `speech_noise_thres` | 0.6 | 减小可检测更多语音（可能增加误检） |
| `max_single_segment_time` | 60000 | 单个语音段的最大时长 |
| `speech_2_noise_ratio` | 1.0 | 减小可提高对语音的敏感度 |
| `max_start_silence_time` | 3000 | 起始段最大静音时长 |
