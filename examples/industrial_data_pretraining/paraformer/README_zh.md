(简体中文|[English](./README.md))

# 语音识别

> **注意**:
> pipeline 支持 [modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的所有模型进行推理和微调。这里我们以典型模型作为示例来演示使用方法。

## 推理

### 快速使用
#### [Paraformer 模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
```python
from funasr import AutoModel

model = AutoModel(model="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

res = model(input="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav")
print(res)
```

### API接口说明
#### AutoModel 定义
- `model`: [模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的模型名称，或本地磁盘中的模型路径
- `device`: `cuda`（默认），使用 GPU 进行推理。如果为`cpu`，则使用 CPU 进行推理
- `ncpu`: `None` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`: `None` （默认），如果设置，输出结果的输出路径
- `batch_size`: `1` （默认），解码时的批处理大小
#### AutoModel 推理
- `input`: 要解码的输入，可以是：
  - wav文件路径, 例如: asr_example.wav
  - pcm文件路径, 例如: asr_example.pcm，此时需要指定音频采样率fs（默认为16000）
  - 音频字节数流，例如：麦克风的字节数数据
  - wav.scp，kaldi 样式的 wav 列表 (`wav_id \t wav_path`), 例如:
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  在这种输入 `wav.scp` 的情况下，必须设置 `output_dir` 以保存输出结果
  - 音频采样点，例如：`audio, rate = soundfile.read("asr_example_zh.wav")`, 数据类型为 numpy.ndarray。支持batch输入，类型为list：
  ```[audio_sample1, audio_sample2, ..., audio_sampleN]```
  - fbank输入，支持组batch。shape为[batch, frames, dim]，类型为torch.Tensor，例如
- `output_dir`: None （默认），如果设置，输出结果的输出路径


## 微调

#### 准备数据

`train_text.txt`

左边为数据唯一ID，需与`train_wav.scp`中的`ID`一一对应
右边为音频文件标注文本

```bash
ID0012W0013 当客户风险承受能力评估依据发生变化时
ID0012W0014 杨涛不得不将工厂关掉
```


`train_wav.scp`

左边为数据唯一ID，需与`train_text.txt`中的`ID`一一对应
右边为音频文件的绝对路径

```bash
ID0012W0013 /Users/zhifu/funasr_github/test_local/aishell2_dev_ios/wav/D0012/ID0012W0013.wav
ID0012W0014 /Users/zhifu/funasr_github/test_local/aishell2_dev_ios/wav/D0012/ID0012W0014.wav
```

#### 训练

```bash
cd examples/industrial_data_pretraining/paraformer
sh finetune_from_local.sh
```

**查看训练日志**

```bash
tensorboard --logdir /xxxx/FunASR/examples/industrial_data_pretraining/paraformer/outputs/log/tensorboard
```


## 导出onnx

```python
from funasr import AutoModel
wav_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"

model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  model_revision="v2.0.4")

res = model.export(input=wav_file, type="onnx", quantize=False)
print(res)
```