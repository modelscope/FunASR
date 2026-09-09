# 用 Transformers 转写 Fun-ASR-Nano

简体中文 | [English](./transformers_native.md)

用 `AutoProcessor` 和 `AutoModelForSpeechSeq2Seq` 转写中、英、日语音。
**正式发布的 Transformers 5.17.0 已包含 Fun-ASR-Nano 原生支持。**
不需要克隆源码、安装 FunASR 工具库，也不需要执行模型仓库的远程 Python 代码。

[在线 Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano) ·
[打开 Notebook](https://colab.research.google.com/github/QwenAudio/Fun-ASR/blob/main/examples/colab/fun_asr_nano_transformers.ipynb) ·
[可运行示例](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)

## 先得到第一条转写

使用独立环境。下面是已验证的 Linux x86-64、Python 3.12 **CPU** 安装路径，
不要直接升级已有的 FunASR 或 vLLM 服务环境。torch 与 torchaudio 必须匹配，
因为原生特征提取器使用 `torchaudio.compliance.kaldi.fbank`。

```bash
python3.12 -m venv .venv-funasr-native
. .venv-funasr-native/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' 'torchaudio==2.10.0+cpu'
python -m pip install 'transformers==5.17.0' 'numpy==1.26.4' \
  'librosa==0.11.0' 'soundfile==0.13.1' \
  'huggingface-hub==1.30.0' 'tokenizers==0.23.2'
python -m pip check
```

运行以下 Python 示例。首次加载需要下载约 1.66 GB 模型权重，然后转写一段
固定版本的官方英文样例。明确使用 CPU float32，不要求 GPU。

<!-- native-example: transcribe -->
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

torch.set_num_threads(4)
model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
audio = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/272c57b82523ada6fd87095e955f8e29100979ab/example/en.mp3"

processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False,
    dtype=torch.float32,
).to("cpu").eval()
inputs = processor.apply_transcription_request(
    audio=audio, language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
        "text_kwargs": {"padding": True},
    },
)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
new_tokens = generated[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

模型输出同时包含输入提示和生成结果，因此解码前需要去掉完整输入张量宽度。
这段音频实测的原始输出为：
“The tribal chieftain called for the boy, and presented him with fifty pieces of gold.”
这是功能示例，不是质量保证。

## 换成自己的录音，再做批处理

[命令行示例](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)
补齐了输入校验、采样率转换和生成完成检查。克隆 **QwenAudio/Fun-ASR 代码仓库**
（不是 HF 权重仓库），按该目录说明安装独立依赖，再在代码仓库根目录运行：

```bash
python examples/transformers/transcribe.py recording.wav --language zh --keywords 开放时间
python examples/transformers/transcribe.py chinese.wav english.wav --language zh en
```

示例将多声道取均值为单声道，再用 `soxr_hq` 显式重采样到 16 kHz，
不会覆盖原始录音。接受 1-4 个文件，总音频时长最多 60 秒；过长、空输入
和非有限采样值会被拒绝，不会静默截断。

在自己的应用中，可向 `processor.apply_transcription_request` 传入波形、
本地路径或支持的 URL。原始数组必须使用真实采样率，不能把 48 kHz 数组标成
16 kHz。语言使用 `zh`、`en` 或 `ja`。批量时传入音频列表、对应语言列表，
并设置 `text_kwargs={"padding": True}`，解码后保留输入顺序。

`keywords` 是词汇提示，`prompt` 是上下文，不是强制词典，也不是工具库的
`hotword` 参数。中文样例加入 `开放时间` 后，实测仍输出 `开饭时间`；
需要用代表性录音和人工参考文本评估质量。

上面的短样例最多生成 128 个新 token。没有 EOS 可能意味着截断，单纯提高
上限不等于长录音已完整覆盖。CLI 输出 `reached_eos`，缺少 EOS 或文本为空时
以非零状态结束；出现 EOS 也不代表每个字都识别正确。

## 从示例走向服务

| 需求 | checkpoint 与接口 |
| --- | --- |
| Python 原生 Transformers | `FunAudioLLM/Fun-ASR-Nano-2512-hf`，`AutoProcessor` + `AutoModelForSpeechSeq2Seq` |
| FunASR 流水线和 HTTP 适配器 | `FunAudioLLM/Fun-ASR-Nano-2512`，`funasr.AutoModel`；[部署矩阵](./deployment_matrix_zh.md) |
| 原生 vLLM | `FunAudioLLM/Fun-ASR-Nano-2512-vllm`；[原生 vLLM 指南](./vllm_official_native_validation_zh.md) |
| C++ / 边缘部署 | 转换后的 GGUF；[llama.cpp](../runtime/llama.cpp/README.md) |

这些是不同格式，不能只替换模型 ID。Transformers 原生支持不自动提供 HTTP
服务、队列、流式协议、字级时间戳或说话人身份。该原生导出不包含 CTC 时间戳
分支。基础 Nano 支持中、英、日及中文方言、口音；**31 语言 MLT 是独立 checkpoint**。

需要转写、匿名说话人和片段时间戳时，查看第三方 OpenMOSS
[MOSS 指南](./moss_transcribe_diarize_zh.md)；其他任务见
[Model Zoo](../model_zoo/readme_zh.md)。GPU dtype、attention 后端、显存、
并发和质量都要在实际目标硬件上另行验证。

## 不加载权重的排查方法

加载失败时，先检查当前解释器、`transformers.__version__`、`-hf` 模型 ID
和 revision。5.16.1 不包含此模型，请在新环境升级，不要改动无关运行服务。
原生类位于 `transformers.models.fun_asr_nano`。

下面是可选的合成静音预处理检查，只下载 processor，不加载模型权重：

<!-- native-example: processor -->
```python
import numpy as np
from transformers import AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
inputs = processor.apply_transcription_request(
    audio=np.zeros(16000, dtype=np.float32), language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
    },
)
print({name: tuple(value.shape) for name, value in inputs.items()})
```

应用层先拒绝空录音和空批次。预处理成功**不代表准确率或容量验证**。

## 复现记录与来源

**2026-09-09** 使用实际发布的 5.17.0 wheel 验证：Python 3.12.3，
torch/torchaudio 2.10.0+cpu、float32、四个 Torch 线程，CPU 为
Intel Xeon Platinum 8480+。官方英文 URL、中文波形、中文热词和中英 padding
批次均返回非空文本，并在 128 token 之前出现 EOS。两段公开短样例的功能
检查不等于 CER/WER、GPU 兼容性或服务容量评测。

公开原生 revision 为 `d93b302ee7fd505e1b3576120fc142fc6f7820e1`，
解析到 `FunAsrNanoForConditionalGeneration`，使用
`trust_remote_code=False`。`model.safetensors` 大小 1,659,773,320 字节，
SHA256：`1bbb6dcc5d8b75084a399d48c4d4b0f3aa1d3f09f2616ae40ed2c4fba03d89c9`。
音频来自原始模型仓库的
[固定样例目录](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/272c57b82523ada6fd87095e955f8e29100979ab/example)，
不是虚构的 `-hf/example` 路径。

[PR #46180](https://github.com/huggingface/transformers/pull/46180) 的合并提交为
`fc501343edfccdc840eb8594a6cafa8185c2de53`，早于正式包发布。旧版源码安装说明
对应那个时间窗口；现在新安装应使用上面的正式包。
参阅 [5.17.0 官方模型文档](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/fun_asr_nano)
与 [原生 model card](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf)。

评估应用时保留版本、输入摘要与原始结果，不要向公开 issue 上传私人录音或凭证。
