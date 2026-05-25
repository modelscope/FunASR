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
3. [离线语音识别服务](#3-离线语音识别服务)
4. [流式语音识别服务](#4-流式语音识别服务)
5. [动态 VAD](#5-动态-vad)
6. [API 参考](#6-api-参考)
7. [FAQ](#7-faq)

---

## 1. 安装与环境

```bash
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart

cd /path/to/FunASR && pip install -e .
```

**硬件**：GPU ≥ 8GB VRAM，CUDA ≥ 11.8。推荐 16GB+。

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

## 3. 离线语音识别服务

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

### 3.3 服务协议

#### HTTP REST — `POST /asr`

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

#### OpenAI Whisper 兼容 — `POST /v1/audio/transcriptions`

**请求**：`multipart/form-data`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | 必填 | 音频文件 |
| `model` | string | "fun-asr-nano" | 模型名 |
| `language` | string | None | 语种 |
| `response_format` | string | "json" | "json" / "text" / "verbose_json" |
| `timestamp_granularities` | string | "word" | "word" / "segment" |

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

### 3.4 客户端调用示例

**Python**：
```python
import requests

resp = requests.post("http://localhost:8899/asr",
    files={"file": open("audio.wav", "rb")},
    data={"language": "中文", "spk": "true"})
result = resp.json()
print(result["text"])
for seg in result["segments"]:
    print(f"  [{seg['start']:.1f}-{seg['end']:.1f}s] {seg.get('speaker','')}: {seg['text']}")
```

**OpenAI SDK**：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8899/v1", api_key="none")
result = client.audio.transcriptions.create(model="fun-asr-nano", file=open("audio.wav", "rb"))
print(result.text)
```

**cURL**：
```bash
curl -X POST http://localhost:8899/asr \
    -F "file=@meeting.wav" -F "language=中文" -F "spk=true"
```

**JavaScript**：
```javascript
const form = new FormData();
form.append("file", audioBlob, "audio.wav");
form.append("language", "中文");
const resp = await fetch("http://localhost:8899/asr", { method: "POST", body: form });
const result = await resp.json();
```

---

## 4. 流式语音识别服务

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

## 5. 动态 VAD

fsmn-vad 默认启用动态静音阈值。离线和流式使用不同配置。

### 离线配置（保留长段 ≤60s）

| 累积时长 | 静音阈值 |
|---------|---------|
| ≤ 10s | 2000ms |
| 10-20s | 1000ms |
| 20-30s | 800ms |
| 30-40s | 600ms |
| 40-50s | 400ms |
| 50-60s | 200ms |
| > 60s | 100ms |

### 流式配置（平衡延迟）

| 累积时长 | 静音阈值 |
|---------|---------|
| ≤ 5s | 2000ms |
| 5-10s | 1500ms |
| 10-15s | 1000ms |
| 15-30s | 800ms |
| 30-45s | 400ms |
| > 45s | 100ms |

### 自定义

```python
model.generate(input="audio.wav", silence_schedule=[(5000,1500), (20000,800), (float('inf'),300)])
```

> GLM-ASR 不支持长段，使用时传 `dynamic_silence=False`。

---

## 6. API 参考

| 参数 | AutoModelVLLM | serve_vllm.py | serve_realtime_ws.py |
|------|--------------|---------------|---------------------|
| model | ✓ | --model | --model |
| gpu_memory_utilization | ✓ | --gpu-memory-utilization | --gpu-memory-utilization |
| tensor_parallel_size | ✓ | — | --tensor-parallel-size |
| max_model_len | ✓ | --max-model-len | --max-model-len |
| language | generate() 参数 | API 参数 | --language / LANGUAGE: |
| hotwords | generate() 参数 | API 参数 | --hotword-file / HOTWORDS: |

---

## 7. FAQ

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
