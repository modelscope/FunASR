# FunASR 实测：比 Whisper 快 30 倍的开源语音识别，还自带说话人分离

> 本文适合正在使用 OpenAI Whisper 做语音转写、但遇到速度慢或中文效果差问题的开发者。

## 背景

OpenAI Whisper 是目前最流行的开源 ASR 模型，但在生产环境中经常遇到几个痛点：

1. **速度慢** — Whisper-large-v3 在 A100 上只有 13x 实时，处理 1 小时音频需要 ~5 分钟
2. **中文方言差** — 对粤语、四川话、上海话等方言识别率很低
3. **缺少说话人分离** — 需要额外集成 pyannote，增加复杂度
4. **没有流式能力** — 无法做实时字幕

这些正是 [FunASR](https://github.com/modelscope/FunASR) 解决的问题。

## 性能对比

测试条件：184 个中文会议录音，总计 192 分钟，NVIDIA A100。

| 模型 | GPU 速度 | CPU 速度 | 说话人 | 流式 |
|------|---------|---------|--------|------|
| **FunASR SenseVoice-Small** | **170x** 实时 | **17x** 实时 | ✅ | ❌ |
| **FunASR Fun-ASR-Nano (vLLM)** | **340x** 实时 | — | ✅ | ✅ |
| Whisper-large-v3-turbo | 46x 实时 | ❌ | ❌ | ❌ |
| Whisper-large-v3 | 13x 实时 | ❌ | ❌ | ❌ |

**关键发现：FunASR 在 CPU 上比 Whisper 在 GPU 上还快。**

## 最大区别：一站式 vs 拼装

用 Whisper 搭建一个完整的转写系统，你需要：

```
Whisper (ASR) + pyannote (说话人) + silero-vad (VAD) + 
deepmultilingualpunctuation (标点) + 自己写逻辑拼接
```

用 FunASR，**一次调用搞定**：

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
)
result = model.generate(input="meeting.wav")
# 输出：带说话人、时间戳、标点的结构化结果
```

## OpenAI 兼容 API：零改动迁移

如果你已经在用 OpenAI 的 Whisper API，迁移到 FunASR 不需要改一行客户端代码：

```bash
# 一行启动本地 ASR 服务
pip install torch torchaudio
pip install funasr vllm fastapi uvicorn python-multipart
funasr-server --device cuda
```

```python
# 客户端代码完全不变
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
result = client.audio.transcriptions.create(
    model="fun-asr-nano",
    file=open("meeting.wav", "rb")
)
```

## 中文方言支持

这是 Whisper 生态完全没有的：

- **7 大方言**：吴语、粤语、闽语、客家话、赣语、湘语、晋语
- **26 种地域口音**：河南、四川、广东、湖北、云南等
- **歌词识别**：音乐背景下的人声转写

## 什么时候选 FunASR？

| 场景 | 推荐 |
|------|------|
| 中文会议转写 | FunASR（方言+说话人） |
| 实时字幕/直播 | FunASR（流式 WebSocket） |
| 批量文件处理 | FunASR（vLLM 340x 加速） |
| 私有化部署 | FunASR（MIT 开源，无费用） |
| 纯英文、少量使用 | Whisper 也可以 |

## 开始使用

```bash
pip install torch torchaudio
pip install funasr
```

```python
from funasr import AutoModel
model = AutoModel(model="iic/SenseVoiceSmall", device="cuda")
result = model.generate(input="audio.wav")
print(result[0]["text"])
```

更多信息：
- GitHub：https://github.com/modelscope/FunASR（16K+ stars）
- 官网：https://www.funasr.com
- 完整评测：https://modelscope.github.io/FunASR/benchmark.html
- 从 Whisper 迁移指南：https://github.com/modelscope/FunASR/blob/main/docs/migration_from_whisper.md
