# FunASR vLLM 推理引擎指南

---

## Benchmark

**测试集**：184 文件，11541 秒，Fun-ASR-Nano / GLM-ASR-Nano。RTFx 定义、计时口径和可复现字段请见 [Benchmark RTF and Reproducibility Notes](./benchmark/rtf_reproducibility.md)。

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

先安装 vLLM,按 NVIDIA 驱动的 CUDA 版本选对应版本;vLLM 会自动钉定并安装匹配的 torch / torchaudio / torchvision 三件套,所以不要自己装 torch/torchaudio——三者 ABI 锁死,必须是互相编译匹配的同一组(如 torch 2.10.0 ↔ torchaudio 2.10.0 ↔ torchvision 0.25.0),只能随 vLLM 一起来。

```bash
# 1) 先装 vLLM。按 `nvidia-smi` 显示的 CUDA 版本(驱动支持的最高 CUDA,不是 runtime CUDA)选版本,
#    vLLM 会带来匹配的 torch/torchaudio/torchvision。
#    驱动 CUDA 12.x  -> pip install vllm==0.19.1   (附带 torch 2.10 / cu128)
#    驱动 CUDA >= 13 -> pip install vllm           (最新版;附带 torch 2.11 / cu130)

pip install "vllm==0.19.1"   # 按你的驱动 CUDA 调整;见下方说明

# 2) 再装 FunASR 与其余依赖。
pip install funasr>=1.3.0
pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart

cd /path/to/FunASR && pip install -e .
```

**硬件**：GPU ≥ 8GB VRAM，CUDA ≥ 11.8。推荐 16GB+。

为什么不要单独执行 `pip install torch torchaudio` ? torch/torchaudio/torchvision 的版本由 vLLM 版本决定—— 每个大版本会一起升级(见 vLLM 的 [requirements/cuda.txt](https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt))。手动安装会拉到最新 wheel,可能是为比你驱动更新的 CUDA runtime 编译的;PyTorch 会在 CUDA 初始化阶段、FunASR 启动前就报 The NVIDIA driver on your system is too old。让 vLLM 统一钉定这三件套即可避免。若仍遇到该错误,请安装其 CUDA 构建与 nvidia-smi 显示的 CUDA 匹配的 vLLM 版本(如 CUDA 12.x 用 vllm==0.19.1),或先升级 NVIDIA 驱动。

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
│  │  Text Prompt ──→ Tokenize ──→ Embed                       │
│  │  (system/user/                  │                         │
│  │   hotwords/language)            │                         │
│  │                                 ▼                         │
│  │                          [Concat Embeddings]              │
│  └─────────────────────────────────┼─────────────┘           │
│                                    │                         │
│                                    ▼ EmbedsPrompt            │
│  ┌─────────────── vLLM Engine ────────────────────┐          │
│  │                                                │          │
│  │   PagedAttention + Continuous Batching         │          │
│  │   KV Cache 管理 + CUDA Graph                   │          │
│  │   Tensor Parallel (多卡)                       │          │
│  │                                                │          │
│  │   Qwen3-0.6B / Llama-2B (LLM 解码)              │          │
│  │                                                │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│                Generated Text                                │
│                       │                                      │
│  ┌────────────────────┼──────────────────────────┐           │
│  │  (可选) CTC Decoder ──→ Forced Alignment      │            │
│  │           ──→ 字级别时间戳                     │            │
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
2. **EmbedsPrompt**：直接把**已算好的 embedding 向量**（而非通常的 token ID）作为 prompt 送入 vLLM（开关 `enable_prompt_embeds=True`）。Fun-ASR-Nano 必须用它，因为音频经 adaptor 得到的是连续向量、不是 token，需把音频 embedding 与文本 embedding 在序列维拼接后整体送入 vLLM
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
│  音频文件列表 ──→ 分组（每 8 条）──→ Frontend(Fbank)                    │
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
│                        (PagedAttention + CUDA Graph)                │
│                              │                                      │
│                              ▼                                      │
│                     Generated token_ids × N                         │
│                              │                                      │
│                              ▼                                      │
│                     Decode + 后处理（去特殊标记、清洗）                 │
│                              │                                      │
│                              ▼                                      │
│                     (可选) CTC Forced Alignment → 字级别时间戳         │
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
│ Stage 1: 前 10 chunk │  ← 无 prev_text，批量生成
│ 找到稳定输出           │
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

**注意：EmbedsPrompt 下不能用 `repetition_penalty`。** 此时 prompt 是 embedding 向量、没有对应的 token ID，而 `repetition_penalty` 要靠 prompt 的 token ID 在 logits 上给已出现的词降分；用在 EmbedsPrompt 上会**索引越界、触发 CUDA device-side assert**。

### 生产 API 稳定性清单

把 `AutoModelVLLM` 封装成长驻 API 服务时，请隔离每次请求的状态，并固定安全的解码默认值：

```python
common = dict(
    language="auto",
    temperature=0.0,
    repetition_penalty=1.0,
    max_new_tokens=200,
)

for _ in range(2):
    results = model.generate(["vad_segment_01.wav", "vad_segment_02.wav"], **common)
    print([r["text"] for r in results])
```

如果同一个音频第一次请求正常、第二次请求开始重复：

1. 先把 API 层拿掉，用相同 VAD 分段跑上面的最小脚本。
2. 如果最小脚本稳定，优先检查 API 封装是否复用了请求级变量、上一轮 VAD 分段列表、上一轮 `results` 或累积文本。
3. 如果最小脚本也重复，再记录完整的 `funasr`、`vllm`、`torch` 版本，以及第一次和第二次输出文本，再调整其它解码参数。

不要通过调大 `repetition_penalty` 来压制 Fun-ASR-Nano vLLM 重复输出；prompt-embeds 路径应保持中性值 `1.0`。

### 输出特性

| 累积音频 | 输出质量 |
|---------|---------|
| < 1.5s | 空或噪声 |
| 1.5-3.0s | 部分正确 |
| > 3.0s | 准确输出 |


---

## 5. 离线语音识别服务

### 5.1 服务架构

```
客户端                                  serve_vllm.py
  │                                        │
  │── HTTP/OpenAI/WebSocket ──────────────→│
  │                                        │
  │                                   ┌────┴────────────────────────┐
  │                                   │ 1. 接收完整音频文件            │
  │                                   │ 2. 动态 VAD 分段（≤60s/段）    │
  │                                   │ 3. vLLM batch 推理所有段      │
  │                                   │ 4. CTC 时间戳（逐字）          │
  │                                   │ 5. 说话人分离（可选）          │
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

### 5.2 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py \
    --port 8899 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --gpu-memory-utilization 0.5
```

> **关于 `CUDA_VISIBLE_DEVICES`**：这是[vllm的一个环境变量](https://docs.vllm.ai/en/v0.4.3/serving/env_vars.html) ，示例中的 `=0` 只是"用第 0 张卡"的示例值，**不是固定写法**，它选择本进程可见的 GPU（编号同 `nvidia-smi`），单卡机器也不需要设置。
> 
> **单卡多实例**：0.6B / 1.7B 这类小模型一张卡可起多个实例，多进程可都指向同一张卡（如都 `=0`）+ MPS 共享；分卡则进程 A `=0`、B `=1`（见 §6.7）。

### 5.3 协议一：HTTP REST — `POST /asr`

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

### 5.4 协议二：OpenAI Whisper 兼容 — `POST /v1/audio/transcriptions`

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

### 5.5 协议三：WebSocket — `ws://host:port/ws`


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

### 6.1 服务架构

```
客户端（麦克风/音频流）              serve_realtime_ws.py
  │                                      │
  │── WebSocket PCM16 16kHz ────────────→│
  │   (每帧 ~100ms，持续发送)             │
  │                                      │
  │                                 ┌────┴─────────────────────────┐
  │                                 │ 实时循环：                     │
  │                                 │  ├─ 动态 VAD（60ms chunk）    │
  │                                 │  ├─ 检测到端点 → vLLM 解码     │
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
- 可选流式说话人分配（`--enable-spk`）+ STOP 时全局重聚类
- 首字延迟 ~480ms

### 6.2 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --hotword-file 热词列表
```

多客户端或长时间连续语音场景，建议先限制 partial 预览窗口并适当降低刷新频率：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 \
    --partial-window-sec 8 --decode-interval 0.8
```

说话人分离默认关闭；只有确实需要 `spk` 字段时再加 `--enable-spk`。

如果麦克风长连接经过 Docker、nginx 或云负载均衡，建议保持 WebSocket
ping/pong 开启，并把 timeout 调到能覆盖短暂网络抖动：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 \
    --ws-ping-interval 20 --ws-ping-timeout 60
```

只有在外部网关已经统一负责 keepalive / reconnect 策略时，才考虑设置
`--ws-ping-interval 0` 关闭服务端 ping。

长会话排障，尤其是启用 `--enable-spk` 时，可以打开周期性 session 状态日志：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文 --enable-spk \
    --log-session-stats-interval 30
```

服务端会每 30 秒输出一行 `Session stats:`。提交 issue 时请连同最后几行
`Session stats`、末段 RTF、进程 RSS、GPU memory 和断线前后的服务端日志一起提供。

### 6.3 WebSocket 协议

**连接**：`ws://host:10095`

**客户端 → 服务端**：

| 消息 | 格式 | 说明 |
|------|------|------|
| 开始 | `"START"` | 初始化 session |
| 热词 | `"HOTWORDS:词1,词2"` | 可选 |
| 语种 | `"LANGUAGE:中文"` | 可选 |
| 音频 | `binary` | PCM16 16kHz mono |
| 结束 | `"STOP"` | 最终解码；启用 `--enable-spk` 时会做 SPK 重聚类 |

**服务端 → 客户端**：

```json
{"event": "started"}
{"sentences": [{"text":"你好","start":300,"end":1200}], "partial": "世界", "is_final": false}
{"sentences": [...], "is_final": true}
{"event": "stopped"}
```

**字段**：`sentences[]` = 已锁定句子，`partial` = 当前正在说的临时文本（可能变化），`partial_start_ms` = 当前 `partial` 对应音频窗口的起点，`is_final` = STOP 后为 true。启用 `--enable-spk` 后，`sentences[]` 会包含 `spk`。

**时序**：
```
Client              Server
  │── START ───────→│
  │←─ started ──────│
  │── [audio] ─────→│
  │←─ {partial} ────│    # partial 的原理是注意事项见 6.5
  │── [audio] ─────→│
  │←─ {sentences+partial} ─│  (VAD 切了一句)
  │── STOP ────────→│
  │←─ {is_final:true} ────│
  │←─ stopped ─────│
```

### 6.4 客户端调用

**Python CLI**：
```bash
python client_python.py --server ws://localhost:10095 --mic
python client_python.py --server ws://localhost:10095 --file audio.wav
```

**实时压测**：
```bash
python examples/industrial_data_pretraining/fun_asr_nano/realtime_ws_benchmark.py \
    audio_16k_mono_pcm16.wav --server ws://localhost:10095 --clients 4 \
    --output-jsonl realtime_ws_4c.jsonl
```

指标定义和报告字段见 [Realtime WebSocket Benchmark](./benchmark/realtime_ws_benchmark.md)。

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

### 6.5 partial 预览机制与长句特性

**partial 是什么、怎么产生的**
流式服务在用户说话过程中会周期性地（`serve_realtime_ws.py` 默认 `decode_interval≈0.48s`）对"当前这句话从句首到现在"的音频解码一次，输出**临时文字**（即协议里的 `partial` 字段，可被后续刷新覆盖），直到 VAD 判定句尾才锁定进 `sentences`。这让用户边说边看到字。

> 注：`serve_vllm.py`（§5）的 `/ws` **没有 partial**、只在句尾返回；要实时预览请用 `serve_realtime_ws.py`。

**前端渲染原则**
`partial` 只能当作“可替换预览”，不要把连续两次 `partial` 直接追加到最终文本里。推荐把已锁定文本和临时预览分开：

```js
const committed = data.sentences.map((s) => s.text).join("");
const preview = data.partial || "";
render(committed + preview);
```

如果启用了 `--partial-window-sec`，`partial_start_ms` 可能随着窗口向前滑动；这时 `partial` 只描述当前受限窗口内的临时识别结果。前端应每次替换 preview 区域，只把 VAD 已锁定的 `sentences` 或最终 `is_final=true` 结果追加到正式转写区。

**原理：为什么每次 partial 都从句首整段重编**
Fun-ASR-Nano 的声学编码器（SenseVoice）是**全上下文、非流式**编码器——每一帧的表示都依赖整段音频的前后文。当这句话又往下说了一截、音频变长时，先前那些帧的上下文随之改变，**之前算出的编码不再成立**，因此无法像流式 / 因果编码器那样"缓存历史、只算新增帧"，只能把"句首→当前"的整段重新过一遍编码器。

**由此带来的特性：长句的 partial 会越来越慢（O(L²)）**
正因每次都从句首重编，一句话越长，单次 partial 要编的音频越长、刷新次数也越多——**总编码量随句长二次增长**。实测一句约 29s 的连续发言会被完整重编十余次，单次 encoder 耗时从几十毫秒爬到数百毫秒。（§4 SDK 流式"每个 chunk 包含从头到当前的全部音频"是同一机制，长文件同理。）

**使用建议**
- 正常对话语音有自然停顿，VAD 会把它切成一句句较短的语音，每句 partial 的开销自然受限，**通常无需关注**。
- 只有**超长、不停顿的连续语音**（如长篇朗读）会让单句不断变长、partial 预览逐渐变慢。`serve_realtime_ws.py` 默认用 `--partial-window-sec 15` 限制临时预览窗口；多客户端或连续独白压测时可降到 `8-10`，并把 `--decode-interval` 提高到 `0.8-1.0`。这只影响临时 `partial`，VAD 锁定句和 STOP 最终结果仍走完整音频。

### 6.6 说话人分离（SPK）的代价与开关

`serve_realtime_ws.py` 默认**不加载** SPK 模型。只有启动时显式加 `--enable-spk`，才会加载 `--spk-model`（默认 `iic/speech_eres2netv2_sv_zh-cn_16k-common`）并在流式中对每个 VAD 完成句调用一次说话人分配。需要注意：

- **Fun-ASR-Nano 上 SPK 效果有限**（见 #2944），多数实时 ASR 场景并不需要说话人分离。
- **流式 SPK 代价高且随会话变长**：每句对**全部历史 embedding** 做一次全量重聚类（**O(N²)**，会话越长每句越贵），且**同步阻塞事件循环**；而会话结束时还会**全量重聚一遍**，流式期间每句的聚类结果会被最终结果覆盖——对最终输出而言属于重复计算。长会话 + 高并发下尤其明显。
- **建议**：多客户端实时转写优先保持默认关闭；确需 diarization 时再加 `--enable-spk`，并以 STOP 后的最终 `spk` 标签为准。
- **长会话诊断**：如果 session 仍然逐渐变慢或断开，请用 `--log-session-stats-interval 30` 复测，并观察 `audio_buffer_samples`、`locked_sentences`、`speaker_history_chunks`、`speaker_history_embeddings` 和 `speaker_centers` 是否保持有界。如果这些计数都接近上限但 RTF 仍持续升高，剩余瓶颈更可能在模型推理、返回 payload 大小或环境调度，而不是 session 状态继续泄漏。

### 6.7 生产并发与多进程部署

`serve_realtime_ws.py` 是**单 asyncio 事件循环**服务：`decode()`（定时 partial）与 `add_audio()`（VAD 句尾触发解码）都**同步阻塞**整个事件循环——任一路在解码时，其余连接全部暂停收发。因此：

- **单进程并发墙来自事件循环串行，不是 GPU 算力**。
- **目前扩展可行方案 = 单卡多个独立进程 + CUDA MPS + nginx 轮询**：每个进程有独立的 GIL 与 CUDA 上下文，绕开单循环串行；MPS 让多进程真正并发共享 GPU、填满空闲算力；nginx 在多个 WebSocket 后端间轮询。超过单卡余量后，再横向加卡（每卡一实例 + 负载均衡）。
- **小并发实时流不一定比多个 PyTorch 进程更适合 vLLM**。vLLM 的优势主要来自批处理和 LLM token decode 调度；而当前实时 WebSocket 路径会把多路小请求通过单事件循环同步送进解码，无法自然形成大 batch，所以可能显存占用更高但 GPU 利用率仍不高。对于少量连续麦克风流，多个轻量 PyTorch 进程有时更容易在一张卡上排布；使用 vLLM 时请按真实话务压测，先配合较低的 `--gpu-memory-utilization` 和多进程服务，而不是假设一个 vLLM 进程就应该承载所有连接。
- **可持续并发没有通用的"支持 N 路"数字**：决定上限的不是在线连接数，而是**同一时刻有多少路正在"说话"**——每路只要在说，就每约 1 秒触发一次 partial 解码，全部串行在那个单事件循环上。它主要随两点变化：**① 停顿 / 静音占比**——真实一问一答中用户大半时间在听、不出声，同时解码的路数远少于在线连接数；连续独白则几乎每路都在持续解码，负载高得多。**② 句长**——句子越长，单次 partial 的编码越贵（见 6.5 的 O(L²)），同样路数下负载更高。因此同一套"单卡 L20 + 多进程 + MPS"，在接近真实 turn-taking 的负载下可稳定支撑数十路，而在长句、连续不停顿的负载下会显著更低。**任何"支持 X 路"的数字都只在它被测出来的那种话务下成立**——请按自己的真实话务（句长、停顿、是否连续说话）压测确定，别把别处测出的某个并发数当成自己的规格。

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

**Q: vLLM 输出连续标点（例如 `!!!!!!!!`），但 PyTorch/HF generate 正常，应该先查什么？**
这通常说明音频 frontend 和 checkpoint 本身能工作，但 vLLM prompt-embedding
路径或解码参数和 upstream runner 不一致。改模型前先检查这些项：

- 传给 vLLM 的 prompt embeddings 要显式转成 float32：
  `EmbedsPrompt(prompt_embeds=input_embeds.float())`。
- 使用 ASR 更合适的确定性解码。Fun-ASR-Nano vLLM 路径默认使用
  `temperature=0.0`、`top_p=1.0` 和 `skip_special_tokens=True`。在
  prompt-embeds 模式下，`repetition_penalty` 保持中性的 `1.0`，除非你走的是
  token prompt 路径；FunASR 的 vLLM helper 会把其他值归一化，避免 vLLM CUDA scatter
  错误。
- 确认 `model_dir` 和 `vllm_model_dir` 是匹配的一组 Fun-ASR-Nano 模型。如果清空
  `vllm_model_dir` 后同一音频走 HF generate 正常，就继续排查 vLLM 路径，而不是音频文件。
- 对一个失败样本记录 vLLM `finish_reason`、生成 token ids、prompt embedding dtype
  和 shape。连续标点且 `finish_reason="length"` 时，通常更像解码/prompt 不匹配，而不是
  VAD 或音频读取问题。
