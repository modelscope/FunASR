# 微调

「简体中文」|「[English](finetune.md)」

## 安装训练环境

```
pip install funasr>=1.3.0
```

## 数据准备

数据格式需要包括如下几个字段：

```
head -n1 data/train_example.jsonl | jq

{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "语音转写：<|startofspeech|>!https://modelscope.cn/datasets/FunAudioLLM/funasr-demo/resolve/master/audios/IT0011W0002.wav<|endofspeech|>"
    },
    {
      "role": "assistant",
      "content": "几点了？"
    }
  ],
  "speech_length": 145,
  "text_length": 3
}
```

详细可以参考：`data/train_example.jsonl`

数据准备细节介绍：

- system 的 content 固定为 `You are a helpful assistant.`
- user 的 content 包含了 prompt 和音频文件的路径（位于 `<|startofspeech|>!` 和 `<|endofspeech|>`之间）
  - prompt 默认为`语音转写：`和`Speech transcription: `
  - 可以结合对应的语种为`语音转写成英文：`和`Transcribe speech into Chinese: `
  - 当音频文件对应的文本标注不含阿拉伯数字或者标点符号时，可以使用`语音转写，不进行文本规整：`和 `Speech transcription without text normalization: `
- assistant 的 content 对应音频文件对应的文本标注
- speech_length：音频文件的 fbank 帧数（一帧 10ms）
- text_length：音频文件标注文本的 token 数 (用 `Qwen/Qwen3-0.6B` 编码)

我们提供了数据格式转换工具 `scp2jsonl.py`，可以将常见的语音识别训练数据格式 wav scp 和 transcription 转成 ChatML 格式。

`train_wav.scp`

左边为数据唯一 ID，需与 `train_text.txt` 中的 ID 一一对应 右边为音频文件的路径，格式如下

```
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
```

`train_text.txt`

左边为数据唯一 ID，需与 `train_wav.scp` 中的 ID 一一对应 右边为音频文件标注文本，格式如下：

```
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
```

```
python tools/scp2jsonl.py \
  ++scp_file=data/train_wav.scp \
  ++transcript_file=data/train_text.txt \
  ++jsonl_file=data/train_example.jsonl
```

## 启动训练

修改 `finetune.sh` 中的 `audio_encoder_conf.freeze`, `audio_adaptor_conf.freeze` 和 `llm_conf.freeze`。

将需要微调的模块 `freeze` 设置成 `false`（默认只微调 llm）。

更多参数细节参考：[SenseVoice 模型训练与测试](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E6%B5%8B%E8%AF%95)

```
bash finetune.sh
```

### 推荐配置

- 训练数据少于 1000 小时，建议微调 audio_adaptor
- 训练数据少于 5000 小时，建议微调 audio_encoder 和 audio_adaptor
- 训练数据大于 10000 小时，建议全量参数微调

## 模型评测

当模型微调结束后，可以使用 decode.py 脚本对模型进行解码：

```
python decode.py \
  ++model_dir=/path/to/finetuned \
  ++scp_file=data/val_wav.scp \
  ++output_file=output.txt
```

解码结束后，需要对标注和识别结果做文本逆归一化，然后计算 WER：

```
python tools/whisper_mix_normalize.py data/val_text.txt data/val_norm.txt
python tools/whisper_mix_normalize.py output.txt output_norm.txt
compute-wer data/val_norm.txt output_norm.txt cer.txt
tail -n8 cer.txt
```
