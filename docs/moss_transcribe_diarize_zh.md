# 在 FunASR 生态中部署 MOSS-Transcribe-Diarize

[English](./moss_transcribe_diarize.md)

本文把第三方
[OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
模型接入 FunASR 部署生态。模型由 OpenMOSS 以 Apache-2.0 发布，不是 FunASR
自有模型，也没有注册到 FunASR `AutoModel`。

MOSS-Transcribe-Diarize 会联合生成转写、时间戳和 `[S01]` 等说话人标签，应用侧
不必再拼接外部 VAD、ASR 和 diarization 管线。这里描述的是部署形态，不表示模型
内部没有分块或分段。

## 固定上游版本

本文按以下不可变版本验证：

- 源码：`OpenMOSS/MOSS-Transcribe-Diarize@cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`
- 模型：`OpenMOSS-Team/MOSS-Transcribe-Diarize@e8681d68e7042738ffca8ac8212bc8fcb1131ab8`
- 许可证：固定上游源码和模型元数据中的 Apache-2.0
- vLLM nightly 索引：`68b4a1d582818e67adc903bf1b8fc5a5447da2fa`

三者都要固定。模型依赖 `trust_remote_code`，生产服务不要执行浮动的模型 revision。

## 选择服务后端

| 路径 | 环境 | 响应契约 | 适用场景 |
|---|---|---|---|
| vLLM | CUDA 12（`cu129`）或 CUDA 13（`cu130`） | `response_format=json` 返回原始 `[start][Sxx]text[end]` 文本 | 已运行 vLLM，或需要其调度能力 |
| SGLang Omni | 当前上游指南为 CUDA 13 | `response_format=verbose_json` 返回解析后的 segments | API 需要直接返回结构化说话人分段 |
| Transformers | PyTorch 进程 | Python 对象和原始标签文本 | 评测、排障或自定义预处理 |

两个 HTTP 后端都使用 `/v1/audio/transcriptions`，但官方文档中的响应格式不能混为
一谈。没有对 exact vLLM revision 做验证时，不要承诺 vLLM 返回 `segments`。

## vLLM

创建隔离环境并安装上游固定的 nightly build：

```bash
uv venv --python 3.12 .venv-moss
uv pip install --python .venv-moss/bin/python -U 'vllm[audio]' \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129
```

必须安装 `audio` extra。只安装 plain `vllm` 时服务可以启动，但因为没有音频解码器，
请求会返回 HTTP 400（`Invalid or unsupported audio file`）。

使用不可变模型 revision 启动：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-moss/bin/vllm serve \
  OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision e8681d68e7042738ffca8ac8212bc8fcb1131ab8 \
  --served-model-name moss-transcribe-diarize \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8898
```

提交音频并验证说话人标签文本：

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=moss-transcribe-diarize \
  -F response_format=json \
  -F temperature=0 \
  | tee moss-transcription.json

python - <<'PY'
import json

with open("moss-transcription.json", encoding="utf-8") as stream:
    payload = json.load(stream)
text = payload.get("text", "")
assert text.strip(), payload
assert "[S01]" in text, text
print(text)
PY
```

长音频需要按业务最长会议验证 `max_completion_tokens`。增大上限也会影响显存和尾延迟。

### 已复现的 vLLM 契约

FunASR 生态验证使用 vLLM `0.23.1rc1.dev949+g68b4a1d58`、Torch
`2.11.0+cu129` 和一张 H100 80GB：

- 仓库内 6.000 秒样例
  （`ea03e1f473ad1618a03da3327a545369cb8f6f06cb0f4115535e5a866167d47e`）
  返回 HTTP 200 和 `[0.96][S01]... [5.94]`；
- A + 0.8 秒静音 + B + 0.8 秒静音 + A 的拼接样例
  （`dbb32bcfed2e8226bedf64248a9f4a44685b293a4696d18fb4cfa701b04db912`）
  返回 HTTP 200、`S01 -> S02 -> S01`，时间戳覆盖到 19.08 秒。

这证明固定版本对这些输入满足 API 和说话人返回契约，不是 diarization 精度、重叠语音、
吞吐或生产容量结论。

## SGLang Omni

按固定上游 SGLang Omni 安装指南准备 CUDA 13 环境，然后下载不可变模型快照并从本地目录启动：

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision e8681d68e7042738ffca8ac8212bc8fcb1131ab8 \
  --local-dir .models/moss-transcribe-diarize

sgl-omni serve \
  --model-path .models/moss-transcribe-diarize \
  --port 8898 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80
```

请求结构化分段：

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F response_format=verbose_json
```

接入字幕、会议纪要或分析系统前，验证每个 segment 都包含起止时间、文本和预期的 speaker 字段。

## 上线前验证

不要只用短单人音频，应使用真实多人长音频：

- 检查长轮次及说话人再次出现时的身份一致性；
- 覆盖重叠语音、串音、音乐、噪声和长静音；
- 验证时间戳单调递增并覆盖完整音频；
- 记录 GPU、CUDA、Torch、后端 commit、模型 revision、音频时长、生成上限、
  墙钟延迟和峰值显存；
- 在 API 网关完成认证、TLS、请求限制和数据留存策略，模型 worker 只绑定内网。

向 FunASR 报告问题时，请明确这是 OpenMOSS 第三方路径，并附上后端与上游 exact
revision。模型架构或权重问题应反馈给上游；部署中心的契约或文档问题可以在 FunASR 提交。
