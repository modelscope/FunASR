# 用原生 Transformers 接入 Fun-ASR-Nano

简体中文 | [English](./transformers_native.md)

如果 Python 应用已经使用 Hugging Face 的处理器和生成接口，可以选择这条路径。
它通过 Transformers 原生类运行 Fun-ASR-Nano，不依赖 FunASR 工具包或 checkpoint
中的远程 Python 代码。需要 HTTP 服务时先看[部署矩阵](./deployment_matrix_zh.md)：
本指南不会自动提供 OpenAI 接口、请求队列或实时服务器。

## 合入主线，不等于已经发布

**2026-09-09**，[Transformers PR #46180](https://github.com/huggingface/transformers/pull/46180)
合入提交 `fc501343edfccdc840eb8594a6cafa8185c2de53`。
当天核验时，最新稳定版 **5.16.1 不包含这个模型**，其 tag 和实际 wheel 均已检查。
源码构建显示 `5.17.0.dev0`，不代表下一稳定版的版本或发布日期承诺。
请固定下方经过验证的源码提交；改用后续稳定版前，实际确认安装包包含 `fun_asr_nano`。

## 模型格式和接口一起选

| 路径 | Checkpoint | 入口 |
| --- | --- | --- |
| FunASR 工具包 | `FunAudioLLM/Fun-ASR-Nano-2512` | `funasr.AutoModel`；拆分引擎另见对应文档 |
| 原生 Transformers | `FunAudioLLM/Fun-ASR-Nano-2512-hf` | `AutoProcessor` + `AutoModelForSpeechSeq2Seq` |
| 原生 vLLM | `FunAudioLLM/Fun-ASR-Nano-2512-vllm` | [官方原生 vLLM 指南](./vllm_official_native_validation_zh.md) |
| 原生 C++ | 对应运行时转换后的 GGUF 文件 | [llama.cpp 指南](../runtime/llama.cpp/README.md) |

这些产物及 API 契约不同。改模型 ID 的名字或传 `backend="vllm"` 不会转换权重；
原生 Transformers 支持不意味着 `-hf` 可以交给 vLLM、GGUF 运行时或 FunASR HTTP 服务。

官方[原生 HF checkpoint](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf/tree/d93b302ee7fd505e1b3576120fc142fc6f7820e1)
在 revision `d93b302ee7fd505e1b3576120fc142fc6f7820e1` 公开且无需申请访问。
配置映射到 `FunAsrNanoForConditionalGeneration`，特征提取器嵌在
`processor_config.json` 中。此快照不需要另一个 `preprocessor_config.json`。

## 准备独立的 CPU 环境

使用新目录创建虚拟环境，不要直接升级正在运行的 FunASR/vLLM 服务环境。
下面对应 **Linux x86-64、Python 3.12、仅 CPU**，不是 CUDA 或 macOS 安装方案。

```bash
python3.12 -m venv .venv-funasr-native
. .venv-funasr-native/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' 'torchaudio==2.10.0+cpu'
python -m pip install 'numpy==1.26.4' 'librosa==0.11.0' 'soundfile==0.13.1' \
  'huggingface-hub==1.30.0' 'tokenizers==0.23.2' \
  'https://github.com/huggingface/transformers/archive/fc501343edfccdc840eb8594a6cafa8185c2de53.zip'
python -m pip check
```

验证环境安装的是从上述精确提交构建的 wheel。原生特征提取器调用
`torchaudio.compliance.kaldi.fbank`，因此需要匹配的 **torch 和 torchaudio**。
FunASR 工具包的可选音频依赖策略不能套用到这里。旧环境曾被过旧的 Hub/tokenizers
依赖挡住；不要关闭依赖检查或混搭版本来强行导入。
这是经过验证的环境，不是所有传递依赖的完整锁文件；自己的运行还应记录
`python -m pip freeze`、平台信息和模型 revision。

## 先验证预处理，不加载权重

这一步仅下载公开的配置、tokenizer 和聊天模板，**不下载模型权重**。
一秒合成静音用来检查入口，不是识别质量测试。

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
    audio=np.zeros(16000, dtype=np.float32),
    language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
    },
)
print({name: tuple(value.shape) for name, value in inputs.items()})
```

验证环境中文字张量形状是 `(1, 42)`，音频特征是 `(1, 17, 560)`，
特征 mask 是 `(1, 17)`；17 个音频占位 token 与 17 帧有效特征对应。
这些形状随输入时长和模板变化，不应在真实应用里写死。

## 转写一份录音

在工作目录准备非空、**单声道 16 kHz WAV** 文件 `audio.wav`。
代码明确拒绝其他采样率或声道布局，而不是把任意采样直接标成 16 kHz。
重采样和混音请使用明确的策略，在这一步之前完成，并保留原录音便于回听。

首次加载模型会下载权重，已有固定 revision 缓存时复用缓存。
其内存和耗时明显高于仅预处理。下面显式使用 CPU float32；
GPU 放置、混合精度、attention 后端和服务并发需要分别验证。

<!-- native-example: transcribe -->
```python
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
audio, sample_rate = sf.read("audio.wav", dtype="float32")
if sample_rate != 16000 or audio.ndim != 1 or audio.size == 0:
    raise ValueError("Use a non-empty mono 16 kHz WAV file")

processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False,
    dtype=torch.float32,
).to("cpu").eval()
inputs = processor.apply_transcription_request(
    audio=audio, language="zh",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": sample_rate},
    },
)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
new_tokens = generated[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

`generated` 包含输入 prompt。解码前去掉 `inputs.input_ids.shape[1]`，
避免把提示模板当作识别结果。`max_new_tokens=128` 限定这份短文件示例的生成上限；
触顶可能截断输出，调大也不保证整段覆盖。对应语言可改为 `language="en"` 或
`language="ja"`；这里是转写语言指令，不是翻译能力承诺。

## 上下文、热词和批处理

`apply_transcription_request` 使用 `prompt` 传上下文、`keywords` 传热词，
不是工具包的 `hotword` 参数或 vLLM HTTP 字段。源码 API 接受全批统一语言，
或与批大小一致的语言列表；逐录音的 prompt 和嵌套关键词列表也必须与批大小匹配。
结果应保持输入顺序。

模板接收了关键词，不等于识别更准确。应在获授权且有代表性的录音中检查专名、
数字和生僻词。把单文件示例变成生产批处理之前，应一起测试 padding、内存和生成文本，
而不是只检查请求张量的形状。

## 验证范围与边界

本指南的单录音代码还实际运行了官方中文样本，随后测试了英文单录音、中文关键词请求及
中英混合批量。各请求均返回非空文本，并在 128 token 上限之前生成 EOS。
这只是两份公开短样本的功能检查，不是 CER/WER 或容量评测。
CPU 为 Intel Xeon Platinum 8480+，float32、Torch 四线程；torch/torchaudio
2.10.0+cpu、librosa 0.11.0。模型来自先前下载的官方缓存，本轮重新检查了 SHA 和文件大小。

样本来自原始模型的[固定 example 目录](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/272c57b82523ada6fd87095e955f8e29100979ab/example)，
不是假定存在的 -hf 示例目录。中文源文件是单声道 48 kHz MP3，先用 librosa 默认的
soxr_hq 明确重采样，再写成 float32 16 kHz WAV，运行未改写的指南代码。
原始文本为 `开饭时间早上九点至下午五点。`；传入候选关键词 `开放时间` 也没有强制产生该词形。
热词是提示，不是强制词表约束。

合成处理器测试覆盖了语言别名、上下文/关键词列表和不同长度的批量输入。
空列表仍会在上游预期校验前抛 `IndexError`，应用应提前拒绝空批量；指南已显式拒绝空波形，
不经过这一路径。成功用例不能抹去这个限制。最初的重采样也暴露出 librosa 0.10.1
依赖已移除的 `pkg_resources`；最终环境使用 0.11.0 并重新验证。

2026-09-09 的独立 CPU 验证通过依赖检查、原生配置/类解析和固定官方 revision 的合成预处理。
已安装的配置、模型、处理器、特征提取器源码逐字节匹配上游合并提交。
预处理通过**不代表准确率或容量结论**，也不证明字级时间戳、说话人分离、实时行为、
GPU 兼容性或与 vLLM 的性能一致。

需要匿名说话人与段级时间戳时，参考第三方 OpenMOSS 的
[MOSS 指南](./moss_transcribe_diarize_zh.md)；其他路径见
[Model Zoo](../model_zoo/readme_zh.md)。不能从“返回了文本”推导时间戳或身份识别能力。
应用验收应保留输入散列、固定版本、原始输出和获授权的人工回听记录；
不要把私有录音或凭据附到公开 issue 中。

## 来源与下一步

- [合并后的实现与官方用法](https://github.com/huggingface/transformers/blob/fc501343edfccdc840eb8594a6cafa8185c2de53/docs/source/en/model_doc/fun_asr_nano.md)。
- [原生处理器](https://github.com/huggingface/transformers/blob/fc501343edfccdc840eb8594a6cafa8185c2de53/src/transformers/models/fun_asr_nano/processing_fun_asr_nano.py)与[音频特征提取器](https://github.com/huggingface/transformers/blob/fc501343edfccdc840eb8594a6cafa8185c2de53/src/transformers/models/fun_asr_nano/feature_extraction_fun_asr_nano.py)。
- [为什么 checkpoint 后缀重要](https://www.funasr.com/blog/fun-asr-nano-transformers.html)：面向应用的专题解读。
- [Fun-ASR 模型项目](https://github.com/QwenAudio/Fun-ASR)与[FunASR 工具包](https://github.com/modelscope/FunASR)。
