# FunASR vLLM 推理引擎指南

---

## 目录

1. [概述](#1-概述)
2. [安装与环境](#2-安装与环境)
3. [离线语音识别服务](#3-离线语音识别服务)
4. [流式语音识别服务](#4-流式语音识别服务)
5. [动态 VAD](#5-动态-vad)
6. [Benchmark](#6-benchmark)
7. [API 参考](#7-api-参考)
8. [FAQ](#8-faq)

---

## 1. 概述

FunASR 集成 vLLM 推理引擎，加速 LLM-based ASR 模型的自回归解码。

### 支持模型

| 模型 | 说明 | vLLM 加速 |
|------|------|-----------|
| **Fun-ASR-Nano** | Qwen3-0.6B，31语言+方言 | RTFx 371 |
| **GLM-ASR-Nano** | Llama 2B，17语言 | RTFx 265 |
| Paraformer / SenseVoice | 非自回归模型 | ✗ 不适用 |

### 两种服务

| 服务 | 入口 | 协议 | 场景 |
|------|------|------|------|
| [离线语音识别服务](#3-离线语音识别服务) | `serve_vllm.py` | HTTP REST / OpenAI API / WebSocket | 文件转写、批量处理、API 集成 |
| [流式语音识别服务](#4-流式语音识别服务) | `serve_realtime_ws.py` | WebSocket | 麦克风实时识别、直播字幕 |

---

## 2. 安装与环境

```bash
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart

cd /path/to/FunASR && pip install -e .
```

**硬件**：GPU ≥ 8GB VRAM，CUDA ≥ 11.8。

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
- SPK 说话人分离可选（默认关闭）

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
| `model` | string | "fun-asr-nano" | 模型名（兼容字段） |
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
            "id": 0,
            "start": 0.0,
            "end": 5.15,
            "text": "我一直没有照顾孩子，但是我想要抚养权。",
            "words": [{"word": "我", "start": 0.42, "end": 0.48}, ...]
        }
    ]
}
```

#### WebSocket — `ws://host:port/ws`

非流式 WebSocket：客户端发送完整音频后获取结果。

**客户端 → 服务端**：
```
"START"                    开始会话
"LANGUAGE:中文"            设置语种（可选）
"HOTWORDS:词1,词2"         设置热词（可选）
[binary bytes]             完整音频 PCM16 16kHz mono
"STOP"                     请求识别结果
```

**服务端 → 客户端**：
```json
{"event": "started"}
{"event": "language_set", "language": "中文"}
{"sentences": [...], "is_final": true, "duration_ms": 5170}
{"event": "stopped"}
```

### 3.4 客户端调用示例

**Python**：
```python
import requests

# HTTP REST
resp = requests.post("http://localhost:8899/asr",
    files={"file": open("audio.wav", "rb")},
    data={"language": "中文", "spk": "true"})
print(resp.json()["text"])

# OpenAI 兼容（可直接用 openai SDK）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8899/v1", api_key="none")
result = client.audio.transcriptions.create(model="fun-asr-nano", file=open("audio.wav", "rb"))
print(result.text)
```

**cURL**：
```bash
curl -X POST http://localhost:8899/asr \
    -F "file=@meeting.wav" -F "language=中文" -F "spk=true" -F "timestamp=true"
```

**JavaScript**：
```javascript
const form = new FormData();
form.append("file", audioBlob, "audio.wav");
form.append("language", "中文");
const resp = await fetch("http://localhost:8899/asr", { method: "POST", body: form });
const result = await resp.json();
console.log(result.text, result.segments);
```

---

## 4. 流式语音识别服务

### 4.1 服务架构

```
客户端（麦克风/音频流）              serve_realtime_ws.py
  │                                      │
  │── WebSocket PCM16 16kHz ────────────→│
  │   (每 100ms 一帧，持续发送)           │
  │                                      │
  │                                 ┌────┴─────────────────────────┐
  │                                 │ 实时处理循环：                 │
  │                                 │  ├─ 动态 VAD（60ms chunk）    │
  │                                 │  ├─ 检测到端点 → vLLM 解码段  │
  │                                 │  ├─ 未结束 → partial 预览     │
  │                                 │  └─ 说话人流式分配             │
  │                                 └────┬─────────────────────────┘
  │                                      │
  │←── JSON 实时推送（每次有更新） ───────│
  │    {sentences:[], partial:"正在说..."} │
```

**特点**：
- 音频逐帧到达，边收边处理
- 基于 VAD 端点自然分句（非固定 chunk）
- 确认段文字锁定不变，partial 实时更新
- 流式说话人分配 + STOP 时全局重聚类
- 首字延迟 ~480ms

### 4.2 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 \
    --language 中文 \
    --hotword-file 热词列表
```

### 4.3 服务协议（WebSocket）

**连接**：`ws://host:10095`

**客户端 → 服务端**：

| 消息类型 | 格式 | 说明 |
|---------|------|------|
| 开始 | `"START"` | 初始化/重置 session |
| 热词 | `"HOTWORDS:词1,词2"` | 可选，逗号分隔 |
| 语种 | `"LANGUAGE:中文"` | 可选 |
| 音频 | `binary` | PCM16 16kHz 单声道，任意帧长 |
| 结束 | `"STOP"` | 触发最终解码 + 说话人重聚类 |

**服务端 → 客户端**：

| 消息类型 | 说明 |
|---------|------|
| `{"event": "started"}` | 会话开始确认 |
| `{"event": "hotwords_set", "hotwords": [...]}` | 热词设置确认 |
| `{"event": "language_set", "language": "..."}` | 语种设置确认 |
| `{"sentences": [...], "partial": "...", "is_final": false}` | 实时结果推送 |
| `{"sentences": [...], "is_final": true}` | 最终结果（STOP 后） |
| `{"event": "stopped"}` | 会话结束 |

**实时结果字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sentences` | array | 已确认句子 `[{"text", "start", "end", "spk"}]` |
| `partial` | string | 当前在说的文字（未确认，会变） |
| `partial_start_ms` | int | partial 起始时间 |
| `duration_ms` | int | 已接收音频总时长 |
| `is_final` | bool | STOP 后为 true |

**交互时序**：

```
Client                          Server
  │── "START" ─────────────────→│
  │←── {"event":"started"} ─────│
  │── "LANGUAGE:中文" ─────────→│
  │←── {"event":"language_set"} │
  │── [audio 100ms] ──────────→│
  │── [audio 100ms] ──────────→│
  │←── {sentences:[], partial:"你好"} ──│  (VAD 未切，partial 更新)
  │── [audio 100ms] ──────────→│
  │←── {sentences:[{text:"你好世界"}], partial:""} ──│  (VAD 切了，锁定)
  │── [audio 100ms] ──────────→│
  │←── {sentences:[...], partial:"今天"} ──│
  │── "STOP" ──────────────────→│
  │←── {sentences:[..., 最后一句], is_final:true} ──│
  │←── {"event":"stopped"} ────│
```

### 4.4 客户端调用示例

**Python（麦克风）**：
```python
python client_python.py --server ws://localhost:10095 --mic
```

**Python（文件流式发送）**：
```python
python client_python.py --server ws://localhost:10095 --file audio.wav --hotwords "张三,北京"
```

**浏览器**：直接打开 `client_mic.html`，支持麦克风实时录音、文件上传、热词文件加载。

**自定义 Python 客户端**：
```python
import asyncio, websockets, numpy as np

async def stream_asr(audio_path, server="ws://localhost:10095"):
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    audio_int16 = (audio * 32768).astype(np.int16)

    async with websockets.connect(server, ping_interval=None) as ws:
        await ws.send("START")
        await ws.recv()  # started

        # 流式发送
        chunk_size = 1600  # 100ms @ 16kHz
        for i in range(0, len(audio_int16), chunk_size):
            await ws.send(audio_int16[i:i+chunk_size].tobytes())
            await asyncio.sleep(0.05)

            # 接收中间结果（非阻塞）
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                data = json.loads(msg)
                if data.get("partial"):
                    print(f"\r  {data['partial']}", end="")
            except asyncio.TimeoutError:
                pass

        await ws.send("STOP")
        # 接收最终结果
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get("is_final"):
                for s in data["sentences"]:
                    print(f"\n[{s['start']/1000:.1f}-{s['end']/1000:.1f}s] SPK{s.get('spk','?')}: {s['text']}")
                break

asyncio.run(stream_asr("meeting.wav"))
```

**JavaScript（浏览器 WebSocket）**：
```javascript
const ws = new WebSocket("ws://localhost:10095");
ws.onopen = () => {
    ws.send("START");
    ws.send("LANGUAGE:中文");
    // 开始发送麦克风 PCM16 数据...
};
ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.sentences) {
        // 更新 UI：显示 sentences + partial
    }
};
// 停止时
ws.send("STOP");
```

---

## 5. 动态 VAD

fsmn-vad 默认启用动态静音阈值，根据语音段累积时长自动调整。

### 离线与流式的配置差异

**离线**（`serve_vllm.py` / `AutoModel.generate`）：

| 累积时长 | 静音阈值 | 策略 |
|---------|---------|------|
| ≤ 10s | 2000ms | 充分等待 |
| 10-20s | 1000ms | |
| 20-30s | 800ms | |
| 30-40s | 600ms | |
| 40-50s | 400ms | |
| 50-60s | 200ms | |
| > 60s | 100ms | 强制切 |

目标：保留长段（≤60s），减少边界损失，提升 CER。

**流式**（`serve_realtime_ws.py` / `DynamicStreamingVAD`）：

| 累积时长 | 静音阈值 | 策略 |
|---------|---------|------|
| ≤ 5s | 2000ms | 等用户说完 |
| 5-10s | 1500ms | |
| 10-15s | 1000ms | |
| 15-30s | 800ms | |
| 30-45s | 400ms | 快速切分 |
| > 45s | 100ms | 强制切 |

目标：平衡延迟和完整性，尽快给出确认结果。

### 自定义

```python
# 离线调用时自定义
model.generate(input="audio.wav", silence_schedule=[
    (5000, 1500), (20000, 800), (float('inf'), 300)
])

# 流式自定义
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD
vad = DynamicStreamingVAD(vad_model, silence_schedule=[...])
```

> **注意**：GLM-ASR 不支持长段推理（>15s 质量下降），使用 GLM-ASR 时应传 `dynamic_silence=False` 回退到固定阈值。

---

## 6. Benchmark

**测试集**：184 文件，11541 秒。

| 模型 | 引擎 | VAD | RTFx | CER | 备注 |
|------|------|-----|------|-----|------|
| Fun-ASR-Nano | PyTorch | dynamic | 21 | 8.06% | 基准 |
| Fun-ASR-Nano | **vLLM batch** | dynamic | **340** | **8.20%** | 16x 加速 |
| Fun-ASR-Nano | **离线服务 (no SPK)** | dynamic | **102** | 8.14% | |
| Fun-ASR-Nano | **离线服务 (+SPK)** | dynamic | **46** | 8.19% | SPK 默认关闭 |
| GLM-ASR-Nano | **vLLM batch** | fixed | **265** | 12.93% | 不支持长音频推理 |

**复现**：
```bash
python benchmark_vllm.py --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --audio-dir /path/to/audio --label-json /path/to/labels.json
```

---

## 7. API 参考

### AutoModelVLLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | — | 模型名或路径 |
| `hub` | `"ms"` | `"ms"` / `"hf"` |
| `tensor_parallel_size` | `1` | GPU 并行数 |
| `gpu_memory_utilization` | `0.5` | KV Cache 显存比例 |
| `max_model_len` | `4096` | 最大序列长度 |

### serve_vllm.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 8000 | 服务端口 |
| `--model` | Fun-ASR-Nano-2512 | ASR 模型 |
| `--gpu-memory-utilization` | 0.5 | 显存比例 |

### serve_realtime_ws.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 10095 | WebSocket 端口 |
| `--decode-interval` | 0.48 | partial 更新间隔（秒） |
| `--hotword-file` | 热词列表 | 热词文件路径 |
| `--language` | None | 默认语种 |

---

## 8. FAQ

**Q: 离线服务和流式服务怎么选？**
- 有完整音频文件 → 离线服务（更高吞吐，支持 SPK）
- 麦克风/直播实时 → 流式服务（低延迟，逐句输出）

**Q: GLM-ASR 能用动态 VAD 吗？**
不支持。GLM-ASR 长音频段质量严重下降，必须用固定 VAD。

**Q: SPK 影响性能吗？**
RTFx 从 102 → 46（说话人聚类开销）。CER 不受影响。默认关闭。

**Q: 如何二次开发？**
- 离线：直接调用 `process_audio()` 函数，传入 numpy 音频数组
- 流式：参考 `serve_realtime_ws.py` 的 `RealtimeASRSession` 类
- vLLM 引擎：直接用 `FunASRNanoVLLM.generate(inputs=[...])` 或 `AutoModelVLLM`

**Q: 首次启动慢？**
vLLM KV Cache 初始化约 60-90 秒，之后推理即时响应。
