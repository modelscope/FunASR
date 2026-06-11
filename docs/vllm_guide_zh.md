# FunASR vLLM 推理引擎指南

---

## Benchmark

**测试集**：184 文件，11541 秒，Fun-ASR-Nano / GLM-ASR-Nano。

| 模型 | 引擎 | VAD | RTFx | CER | 备注 |
|------|------|-----|------|-----|------|
| Fun-ASR-Nano | PyTorch | dynamic | 21 | 8.06% | 基准 |
| Fun-ASR-Nano | **vLLM batch** | dynamic | **340** | **8.20%** | 16x 加速 |
| Fun-ASR-Nano | **离线服务 (no SPK)** | dynamic | **102** | 8.14% | |
| Fun-ASR-Nano | **离线服务 (+SPK)** | dynamic | **46** | 8.19% | SPK 默认关闭 |
| GLM-ASR-Nano | **vLLM batch** | fixed | **265** | 12.93% | 不支持长音频推理 |

> vLLM 与 PyTorch CER 完全一致（差 < 0.2%），速度提升 16-340x。

---

## 目录

1. [安装与环境](#1-安装与环境)
2. [vLLM 推理引擎架构](#2-vllm-推理引擎架构)
3. [离线 SDK 推理](#3-离线-sdk-推理)
4. [流式 SDK 推理](#4-流式-sdk-推理)
5. [离线语音识别服务](#5-离线语音识别服务)
6. [流式语音识别服务](#6-流式语音识别服务)
7. [动态 VAD](#7-动态-vad)
8. [API 参考](#8-api-参考)
9. [FAQ](#9-faq)

---

## 1. 安装与环境

```bash
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart

cd /path/to/FunASR && pip install -e .
```

**硬件**：GPU ≥ 8GB VRAM，CUDA ≥ 11.8。推荐 16GB+。

请根据 NVIDIA 驱动和 `nvidia-smi` 显示的 CUDA 版本选择匹配的
PyTorch/torchaudio/vLLM 组合，不要无条件保留 pip 拉到的最新 wheel。若
vLLM 或 PyTorch wheel 依赖的 CUDA runtime 高于当前驱动支持范围，可能在
FunASR 启动前就报 `The NVIDIA driver on your system is too old`。遇到该错误时，
优先重装与当前驱动/CUDA 匹配的 PyTorch、torchaudio、vLLM wheel，或先升级
NVIDIA 驱动。

---

## 2. vLLM 推理引擎架构

### 整体架构

FunASR 的 vLLM 集成将 ASR 模型拆分为两部分独立运行：

```
┌──────────────────────────────────────────────────────────────┐
│                    FunASR + vLLM 推理架构                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────── PyTorch (单 GPU) ───────────────┐          │
│  │                                                │          │
│  │  Audio ──→ Frontend ──→ Audio Encoder ──→ Adaptor         │
│  │            (fbank)      (SenseVoice/     (Transformer/    │
│  │                          Whisper)         MLP)            │
│  │                              │                            │
│  │                              ▼                            │
│  │                     Audio Embeddings                      │
│  │                              │                            │
│  │  Text Prompt ──→ Tokenize ──→ Embed                      │
│  │  (system/user/                  │                         │
│  │   hotwords/language)            │                         │
│  │                                 ▼                         │
│  │                          [Concat Embeddings]              │
│  └─────────────────────────────────┼─────────────┘          │
│                                    │                         │
│                                    ▼ EmbedsPrompt            │
│  ┌─────────────── vLLM Engine ────────────────────┐          │
│  │                                                │          │
│  │   PagedAttention + Continuous Batching         │          │
│  │   KV Cache 管理 + CUDA Graph                   │          │
│  │   Tensor Parallel (多卡)                       │          │
│  │                                                │          │
│  │   Qwen3-0.6B / Llama-2B (LLM 解码)            │          │
│  │                                                │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│                Generated Text                                │
│                       │                                      │
│  ┌────────────────────┼──────────────────────────┐           │
│  │  (可选) CTC Decoder ──→ Forced Alignment      │           │
│  │           ──→ 字级别时间戳                     │           │
│  └───────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

### 为什么用 vLLM？

| 特性 | PyTorch generate() | vLLM |
|------|-------------------|------|
| KV Cache 管理 | 固定分配，浪费显存 | PagedAttention，按需分配 |
| 批处理 | 需手动 padding | Continuous Batching，自动调度 |
| CUDA 优化 | 无 | CUDA Graph + 算子融合 |
| 多卡并行 | 手动实现 | Tensor Parallel 一行配置 |
| 吞吐量 | RTFx ~20 | **RTFx 340+** |

### 支持模型

| 模型 | LLM 部分 | audio encoder | vLLM 加速 |
|------|---------|---------------|-----------|
| **Fun-ASR-Nano** | Qwen3-0.6B | SenseVoice | ✓ 21.7x |
| **GLM-ASR-Nano** | Llama-2B | Whisper-like | ✓ 7.6x |
| LLMASR | Qwen/Vicuna | Whisper | ✓ |
| Paraformer | 无 LLM | — | ✗ 非自回归 |
| SenseVoice | 无 LLM | — | ✗ encoder-decoder |

### 关键实现细节

1. **权重分离**：从 `model.pt` 提取 LLM 权重，转为 HuggingFace 格式供 vLLM 加载
2. **EmbedsPrompt**：音频 embedding + 文本 embedding 拼接后作为 prompt embedding 送入 vLLM
3. **use_low_frame_rate**：Fun-ASR-Nano 的 adaptor 输出需按公式截断到正确 token 数（一致性关键）
4. **batch encode**：多条音频通过 `extract_fbank` → `audio_encoder` → `audio_adaptor` 一次前向
5. **CTC 时间戳**：保留 encoder_out，生成文本后做 forced alignment 得到字级别时间

---


## 3. 离线 SDK 推理

适用于大规模音频转写、离线批量处理。vLLM 的批处理能力在此场景优势最大。

### 设计原理

离线 SDK 推理将 ASR 流水线拆分为两阶段独立执行：

```
┌─────────────────────────────────────────────────────────────────────┐
│                  阶段 1: 音频编码（PyTorch, 单 GPU）                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  音频文件列表 ──→ 分组（每 8 条）──→ Frontend(Fbank)                 │
│       │                                     │                       │
│       │                                     ▼                       │
│       │                            SenseVoice Encoder               │
│       │                                     │                       │
│       │                                     ▼                       │
│       │                            Audio Adaptor                    │
│       │                            (dim 转换 + low_frame_rate 截断)  │
│       │                                     │                       │
│       └─── 共享文本 prompt 预编码 ─────┐      ▼                       │
│            (system/hotwords/language)  │  audio_embeds               │
│                     │                 │      │                       │
│                     ▼                 │      ▼                       │
│                prefix_emb ──→ [concat: prefix | audio | suffix]      │
│                                              │                       │
│                                              ▼                       │
│                                     EmbedsPrompt（N 条）             │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              阶段 2: LLM 解码（vLLM, 多 GPU Tensor Parallel）         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EmbedsPrompt × N ──→ vLLM Continuous Batching                      │
│                        (PagedAttention + CUDA Graph)                 │
│                              │                                      │
│                              ▼                                      │
│                     Generated token_ids × N                          │
│                              │                                      │
│                              ▼                                      │
│                     Decode + 后处理（去特殊标记、清洗）               │
│                              │                                      │
│                              ▼                                      │
│                     (可选) CTC Forced Alignment → 字级别时间戳       │
└─────────────────────────────────────────────────────────────────────┘
```

**关键设计决策：**

1. **权重分离**：首次运行时从 `model.pt` 提取 `llm.*` 前缀的权重，保存为 HuggingFace safetensors 格式供 vLLM 加载（缓存到 `Qwen3-0.6B-vllm/` 目录）
2. **Embedding 拼接**：文本 prompt 通过 LLM 的 `embed_tokens` 层编码为 embedding，与音频 adaptor 输出在序列维度拼接：`[prefix_emb | audio_emb | suffix_emb]`，以 `EmbedsPrompt` 形式送入 vLLM
3. **Low Frame Rate 截断**：adaptor 输出需按公式 `fake_token_len = ((((fbank_len - 3 + 2) // 2 - 3 + 2) // 2) - 1) // 2 + 1` 截断到正确长度，确保与 PyTorch 训练时一致
4. **批量音频编码**：多条音频按 batch_size=8 分组通过 encoder + adaptor 前向，减少 GPU kernel launch 开销
5. **文本 prompt 共享**：同一批次内 hotwords/language 相同时，prefix_emb 和 suffix_emb 只计算一次
6. **CTC 时间戳**：保留 encoder_out，LLM 生成文本后做 forced alignment 得到字级别时间

**为什么比 PyTorch generate() 快？**

| 维度 | PyTorch | vLLM |
|------|---------|------|
| KV Cache | 固定预分配（浪费显存） | PagedAttention 按需分配 |
| 批处理 | 需手动 padding 对齐 | Continuous Batching 自动调度 |
| CUDA | 逐 sample 串行 | CUDA Graph + 算子融合 |
| 多卡 | 需手动实现 | Tensor Parallel 一行配置 |
| 结果 | RTFx ~20 | **RTFx 340+**（16倍加速） |

### 通用接口（推荐）

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",                    # 或 "hf"
    tensor_parallel_size=2,      # 多卡并行
    gpu_memory_utilization=0.8,
)

results = model.generate(
    ["audio1.wav", "audio2.wav"],
    language="中文",
    hotwords=["张三", "北京"],
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
```

### 直接接口

```python
from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

engine = FunASRNanoVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=4,
)

results = engine.generate(
    inputs="wav.scp",  # 支持 scp/jsonl/文件列表
    hotwords=["开放时间"],
    language="中文",
    max_new_tokens=512,
)
```

### 命令行

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# 单文件
python demo_vllm.py --input audio.wav --language 中文

# 批量 + 多卡
python demo_vllm.py --input wav.scp --tensor-parallel-size 4 --batch-size 32

# 带热词 + 保存结果
python demo_vllm.py --input audio.wav --hotwords 张三 北京 --output results.jsonl
```

---

## 4. 流式 SDK 推理

将音频按 720ms chunk 逐步处理，输出逐步稳定的识别结果。适用于 SDK 集成实时字幕场景。

### 设计原理

```
音频流（720ms chunks）
    │ 累积重编码（每个 chunk 包含从头到当前的全部音频）
    ▼
┌──────────────────────┐
│ Stage 1: 前 10 chunk  │  ← 无 prev_text，批量生成
│ 找到稳定输出          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Stage 2: 后续 chunk   │  ← 用稳定输出作 prev_text
└──────────┬───────────┘
           ▼
每个 chunk: [fixed 区域（确认）] + [8字 unfixed（可能变）]
```

### 用法

```python
from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

engine = FunASRNanoStreamingVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    chunk_ms=720,
    rollback_chars=8,
)

for result in engine.streaming_generate("audio.wav", language="中文"):
    if result["is_final"]:
        print(f"最终: {result['text']}")
    else:
        print(f"[{result['audio_duration_ms']:.0f}ms] 确认: {result['fixed_text']}")
```

### 输出特性

| 累积音频 | 输出质量 |
|---------|---------|
| < 1.5s | 空或噪声 |
| 1.5-3.0s | 部分正确 |
| > 3.0s | 准确输出 |

> Note: `repetition_penalty=1.3` 内部硬编码，防止短 chunk 重复退化。

---

## 5. 离线语音识别服务

### 3.1 服务架构

```
客户端                                  serve_vllm.py
  │                                        │
  │── HTTP/OpenAI/WebSocket ──────────────→│
  │                                        │
  │                                   ┌────┴────────────────────────┐
  │                                   │ 1. 接收完整音频文件          │
  │                                   │ 2. 动态 VAD 分段（≤60s/段） │
  │                                   │ 3. vLLM batch 推理所有段    │
  │                                   │ 4. CTC 时间戳（逐字）       │
  │                                   │ 5. 说话人分离（可选）        │
  │                                   └────┬────────────────────────┘
  │                                        │
  │←── JSON 结果 ─────────────────────────│
```

**特点**：
- 音频完整到达后处理，适合文件转写
- 动态 VAD 保留长段（≤60s），减少边界切割损失
- batch 推理所有 VAD 段，吞吐量高
- 自动输出字级别时间戳
- SPK 说话人分离默认关闭，客户端可开启

### 3.2 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py \
    --port 8899 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --gpu-memory-utilization 0.5
```

### 3.3 协议一：HTTP REST — `POST /asr`

功能最全的接口，支持 SPK、时间戳、热词。

**请求**：`multipart/form-data`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | 必填 | 音频文件（wav/mp3/flac） |
| `language` | string | None | 语种（"中文"/"English"/...），None 为自动 |
| `hotwords` | string | "" | 热词，逗号分隔 |
| `spk` | bool | false | 是否开启说话人分离 |
| `timestamp` | bool | true | 是否输出字级别时间戳 |

**响应**：

```json
{
    "text": "完整识别文本",
    "segments": [
        {
            "text": "段文本",
            "start": 1.7,
            "end": 14.8,
            "speaker": "SPK0",
            "words": [
                {"word": "砸", "start": 2.02, "end": 2.08},
                {"word": "了", "start": 2.26, "end": 2.32}
            ]
        }
    ],
    "duration": 227.4,
    "processing_time": 3.422,
    "rtf": 0.015
}
```

**客户端示例**：

```bash
# cURL
curl -X POST http://localhost:8899/asr \
    -F "file=@meeting.wav" -F "language=中文" -F "spk=true"
```

```python
# Python requests
import requests
resp = requests.post("http://localhost:8899/asr",
    files={"file": open("audio.wav", "rb")},
    data={"language": "中文", "spk": "true"})
result = resp.json()
```

```javascript
// JavaScript fetch
const form = new FormData();
form.append("file", audioBlob, "audio.wav");
form.append("language", "中文");
form.append("spk", "true");
const resp = await fetch("http://localhost:8899/asr", { method: "POST", body: form });
const result = await resp.json();
```

### 3.4 协议二：OpenAI Whisper 兼容 — `POST /v1/audio/transcriptions`

兼容 OpenAI Whisper API 标准，可直接用 OpenAI SDK 接入。

**请求**：`multipart/form-data`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | 必填 | 音频文件 |
| `model` | string | "fun-asr-nano" | 模型名（兼容字段） |
| `language` | string | None | 语种 |
| `response_format` | string | "json" | "json" / "text" / "verbose_json" |
| `timestamp_granularities` | string | "word" | "word" / "segment" |
| `spk` | bool | false | 说话人分离（FunASR 扩展字段） |

**响应**（`verbose_json`）：

```json
{
    "task": "transcribe",
    "language": "zh",
    "duration": 5.17,
    "text": "我一直没有照顾孩子，但是我想要抚养权。",
    "segments": [
        {
            "id": 0, "start": 0.0, "end": 5.15,
            "text": "我一直没有照顾孩子，但是我想要抚养权。",
            "words": [{"word": "我", "start": 0.42, "end": 0.48}, ...]
        }
    ]
}
```

**客户端示例**：

```python
# OpenAI SDK（推荐）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8899/v1", api_key="none")
result = client.audio.transcriptions.create(
    model="fun-asr-nano",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
)
print(result.text)
```

```bash
# cURL
curl -X POST http://localhost:8899/v1/audio/transcriptions \
    -F "file=@audio.wav" -F "model=fun-asr-nano" -F "response_format=verbose_json"
```

### 3.5 协议三：WebSocket — `ws://host:port/ws`


离线服务的 WebSocket 接口，发送完整音频后获取结果。STOP 时自动进行说话人聚类，结果中包含 `spk` 字段。

**客户端 → 服务端**：

| 消息 | 说明 |
|------|------|
| `"START"` | 开始会话 |
| `"LANGUAGE:中文"` | 设置语种（可选） |
| `"HOTWORDS:词1,词2"` | 设置热词（可选） |
| `[binary]` | PCM16 16kHz mono 音频数据 |
| `"STOP"` | 结束，请求识别结果 |

**服务端 → 客户端**：

```json
{"event": "started"}
{"event": "language_set", "language": "中文"}
{"sentences": [{"text":"...","start":..,"end":..}], "is_final": true, "duration_ms": 5170}
{"event": "stopped"}
```

**客户端示例**：

```python
import asyncio, websockets, json, numpy as np, soundfile as sf

async def offline_ws(audio_path):
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16)

    async with websockets.connect("ws://localhost:8899/ws") as ws:
        await ws.send("START")
        await ws.recv()
        await ws.send("LANGUAGE:中文")
        await ws.recv()

        # 发送完整音频
        await ws.send(pcm.tobytes())
        await ws.send("STOP")

        # 接收结果
        async for msg in ws:
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"[{s['start']/1000:.1f}s] {s['text']}")
                break

asyncio.run(offline_ws("audio.wav"))
```

---

## 6. 流式语音识别服务

### 4.1 服务架构

```
客户端（麦克风/音频流）              serve_realtime_ws.py
  │                                      │
  │── WebSocket PCM16 16kHz ────────────→│
  │   (每帧 ~100ms，持续发送)             │
  │                                      │
  │                                 ┌────┴─────────────────────────┐
  │                                 │ 实时循环：                     │
  │                                 │  ├─ 动态 VAD（60ms chunk）    │
  │                                 │  ├─ 检测到端点 → vLLM 解码    │
  │                                 │  ├─ 未结束 → partial 预览     │
  │                                 │  └─ 说话人流式分配             │
  │                                 └────┬─────────────────────────┘
  │                                      │
  │←── JSON 实时推送 ───────────────────│
```

**特点**：
- 音频逐帧到达，边收边处理
- 基于 VAD 端点自然分句
- 确认段文字锁定不变，partial 实时更新
- 流式说话人分配 + STOP 时全局重聚类
- 首字延迟 ~480ms

### 4.2 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --hotword-file 热词列表
```

### 4.3 WebSocket 协议

**连接**：`ws://host:10095`

**客户端 → 服务端**：

| 消息 | 格式 | 说明 |
|------|------|------|
| 开始 | `"START"` | 初始化 session |
| 热词 | `"HOTWORDS:词1,词2"` | 可选 |
| 语种 | `"LANGUAGE:中文"` | 可选 |
| 音频 | `binary` | PCM16 16kHz mono |
| 结束 | `"STOP"` | 最终解码 + SPK 重聚类 |

**服务端 → 客户端**：

```json
{"event": "started"}
{"sentences": [{"text":"你好","start":300,"end":1200,"spk":0}], "partial": "世界", "is_final": false}
{"sentences": [...], "is_final": true}
{"event": "stopped"}
```

**字段**：`sentences[]` = 已锁定，`partial` = 正在说（会变），`is_final` = STOP 后为 true。

**时序**：
```
Client              Server
  │── START ───────→│
  │←─ started ──────│
  │── [audio] ─────→│
  │←─ {partial} ────│
  │── [audio] ─────→│
  │←─ {sentences+partial} ─│  (VAD 切了一句)
  │── STOP ────────→│
  │←─ {is_final:true} ────│
  │←─ stopped ─────│
```

### 4.4 客户端调用

**Python CLI**：
```bash
python client_python.py --server ws://localhost:10095 --mic
python client_python.py --server ws://localhost:10095 --file audio.wav
```

**浏览器**：打开 `client_mic.html`

**自定义 Python**：
```python
import asyncio, websockets, numpy as np, json

async def stream(audio_path):
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16)

    async with websockets.connect("ws://localhost:10095") as ws:
        await ws.send("START")
        await ws.recv()

        for i in range(0, len(pcm), 1600):
            await ws.send(pcm[i:i+1600].tobytes())
            await asyncio.sleep(0.05)

        await ws.send("STOP")
        async for msg in ws:
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"[{s['start']/1000:.1f}s] {s['text']}")
                break

asyncio.run(stream("audio.wav"))
```

---

## 7. 动态 VAD

fsmn-vad 默认启用动态静音阈值。离线和流式使用不同配置。

| 累积时长 | 离线（保留长段 ≤60s） | 流式（平衡延迟） |
|---------|-------------------|----------------|
| ≤ 5s | 2000ms | 2000ms |
| 5-10s | 2000ms | 1500ms |
| 10-15s | 1000ms | 1000ms |
| 15-20s | 1000ms | 800ms |
| 20-30s | 800ms | 800ms |
| 30-45s | 600ms | 400ms |
| 45-60s | 200-400ms | 100ms |
| > 60s | 100ms | 100ms |

离线倾向保留长段减少边界损失；流式更快收紧以降低延迟。

### 自定义

```python
model.generate(input="audio.wav", silence_schedule=[(5000,1500), (20000,800), (float('inf'),300)])
```

> GLM-ASR 不支持长段，使用时传 `dynamic_silence=False`。

---

## 8. API 参考

| 参数 | AutoModelVLLM | serve_vllm.py | serve_realtime_ws.py |
|------|--------------|---------------|---------------------|
| model | ✓ | --model | --model |
| gpu_memory_utilization | ✓ | --gpu-memory-utilization | --gpu-memory-utilization |
| tensor_parallel_size | ✓ | — | --tensor-parallel-size |
| max_model_len | ✓ | --max-model-len | --max-model-len |
| language | generate() 参数 | API 参数 | --language / LANGUAGE: |
| hotwords | generate() 参数 | API 参数 | --hotword-file / HOTWORDS: |

---

## 9. FAQ

**Q: 离线还是流式？**
完整文件 → 离线（高吞吐）。麦克风/直播 → 流式（低延迟）。

**Q: GLM-ASR 用动态 VAD？**
不支持长段推理，用 `dynamic_silence=False`。

**Q: SPK 性能影响？**
RTFx 102 → 46。CER 不变。默认关闭。

**Q: 二次开发入口？**
离线：`serve_vllm.process_audio()` / `FunASRNanoVLLM.generate()`
流式：`serve_realtime_ws.RealtimeASRSession`

**Q: 首次慢？**
vLLM 初始化 60-90s，之后即时。
