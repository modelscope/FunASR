# FunASR vLLM 推理引擎指南

FunASR 集成 vLLM 高吞吐推理引擎，用于加速 LLM-based ASR 模型的自回归解码。支持离线批量推理、SDK 流式推理、生产级 WebSocket 实时服务三种模式。

---

## 目录

1. [概述](#1-概述)
2. [安装与环境](#2-安装与环境)
3. [离线批量推理](#3-离线批量推理)
4. [流式 SDK 推理](#4-流式-sdk-推理)
5. [实时 WebSocket 服务](#5-实时-websocket-服务)
6. [性能对比](#6-性能对比)
7. [API 参考](#7-api-参考)
8. [FAQ](#8-faq)

---

## 1. 概述

### 适用模型

| 模型 | vLLM 支持 | 说明 |
|------|-----------|------|
| **FunASRNano** | ✓ | Qwen3-0.6B LLM，支持离线和流式 |
| **LLMASR** | ✓ | Whisper + Qwen/Vicuna/LLaMA |
| **GLMASR** | ✓ | GLM-ASR-Nano (RTFx 263, CER 12.92%) |
| **QwenAudioWarp** | ✓ | Qwen-Audio |
| Paraformer | ✗ | 非自回归模型（CIF predictor），无 LLM 解码 |
| SenseVoice | ✗ | Whisper-like encoder-decoder，非 LLM |
| Conformer/Transformer | ✗ | CTC/attention 解码，非 LLM |

### 架构

```
┌────────────────────────────────────────────────────────┐
│                   FunASR + vLLM                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Audio ──→ Frontend ──→ Audio Encoder ──→ Adaptor      │
│            (fbank)      (SenseVoice)     (Transformer) │
│                                │                       │
│                                ▼                       │
│                       Audio Embeddings                  │
│                                │                       │
│  Text Prompt ──→ Tokenize ──→ Embed ──→ [Concat]      │
│  (hotwords/language/itn)                    │          │
│                                             ▼          │
│                                    ┌──────────────┐    │
│                                    │  vLLM Engine │    │
│                                    │  (Qwen3 etc) │    │
│                                    │  TP 并行加速  │    │
│                                    └──────┬───────┘    │
│                                           ▼            │
│                                    Generated Text      │
│                                                        │
│  (可选) CTC Forced Alignment ──→ 字级别时间戳          │
└────────────────────────────────────────────────────────┘
```

### 三种使用模式

| 模式 | 入口 | 适用场景 |
|------|------|---------|
| [离线批量推理](#3-离线批量推理) | `AutoModelVLLM` / `FunASRNanoVLLM` | 大规模转写、批量处理 |
| [流式 SDK 推理](#4-流式-sdk-推理) | `FunASRNanoStreamingVLLM` | 实时字幕展示（SDK 集成） |
| [WebSocket 实时服务](#5-实时-websocket-服务) | `serve_realtime_ws.py` | 生产部署、多端接入 |

---

## 2. 安装与环境

### 依赖

```bash
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors tiktoken websockets regex

# FunASR 开发模式安装（需要 auto_model_vllm）
cd /path/to/FunASR && pip install -e .
```

### 硬件要求

| 配置 | 最低要求 | 推荐 |
|------|---------|------|
| GPU 显存 | 8GB | 16GB+ |
| CUDA | 11.8 | 12.0+ |
| GPU 数量 | 1 | 2+（tensor parallel） |

---

## 3. 离线批量推理

适用于大规模音频转写、离线批量处理。vLLM 的批处理能力在此场景优势最大。

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

## 5. 实时 WebSocket 服务

生产级实时语音识别服务，集成 VAD 分句 + vLLM 推理 + 说话人分离 + 热词。

### 推理逻辑

**核心设计：基于 VAD 端点逐段解码（非固定 chunk）**

```
音频流 ──→ StreamingVAD (60ms) ──→ 检测到端点 ──→ vLLM 解码整段
                                │                      │
                                │                      ▼
                                │             locked_sentences（锁定）
                                │                      │
                                │                      ▼
                                │             SPK assign（说话人分配）
                                │
                                └──→ 未结束 ──→ vLLM partial decode（每0.48s）
                                                      │
                                                      ▼
                                               partial text（预览，会覆盖）
```

**两条推理路径：**

| 路径 | 触发条件 | 输出 |
|------|---------|------|
| 确认段解码 | VAD 检测到静音端点 | 锁定到 sentences，永不改变 |
| Partial 预览 | 每 0.48s + 新音频 ≥ 960ms | 临时文字，随时覆盖 |

**动态 VAD 阈值：**

| 累积时长 | 静音阈值 | 效果 |
|---------|---------|------|
| ≤ 5s | 2.0s | 短句不切碎 |
| 5-10s | 1.5s | 正常分句 |
| 10-15s | 1.0s | 开始收紧 |
| 15-30s | 0.8s | 较快切分 |
| 30-45s | 0.4s | 防止过长 |
| > 45s | 0.1s | 强制切分 |

**STOP 最终处理：**
1. 剩余音频喂给 VAD（is_final=True）
2. 强制结束当前在说话的段
3. 全局 SPK 重聚类（修正说话人 ID）
4. 返回 `is_final: true`

### 部署

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# 单卡
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095 --language 中文

# 多卡
CUDA_VISIBLE_DEVICES=0,1 python serve_realtime_ws.py \
    --port 10095 --tensor-parallel-size 2 --language 中文

# 完整参数
python serve_realtime_ws.py \
    --port 10095 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --hub ms \
    --device cuda:0 \
    --decode-interval 0.48 \
    --hotword-file 热词列表 \
    --language 中文 \
    --dtype bf16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048
```

### 客户端

| 客户端 | 用法 |
|--------|------|
| 浏览器 | 打开 `client_mic.html`（麦克风/文件/热词/说话人） |
| Python CLI | `python client_python.py --server ws://localhost:10095 --mic` |
| 测试脚本 | `python client_test.py --server ws://localhost:10095 --file audio.wav` |

远程访问需 SSH 端口转发：`ssh -L 10095:localhost:10095 <server>`

### WebSocket 协议

**客户端 → 服务端：**

| 消息 | 格式 | 说明 |
|------|------|------|
| 开始会话 | `"START"` | 初始化 session |
| 设置热词 | `"HOTWORDS:词1,词2"` | 可选 |
| 设置语种 | `"LANGUAGE:中文"` | 可选 |
| 音频数据 | `bytes` | PCM16, 16kHz, mono |
| 结束会话 | `"STOP"` | 触发最终解码 |

**服务端 → 客户端：**

```json
// 事件
{"event": "started"}
{"event": "hotwords_set", "hotwords": ["词1", "词2"]}
{"event": "language_set", "language": "中文"}
{"event": "stopped"}

// 实时结果
{
    "sentences": [{"text": "已确认", "start": 1700, "end": 5500, "spk": 0}],
    "partial": "正在说...",
    "partial_start_ms": 5800,
    "duration_ms": 7200,
    "is_final": false
}

// 最终结果
{
    "sentences": [{"text": "...", "start": ..., "end": ..., "spk": ...}, ...],
    "partial": "",
    "partial_start_ms": 0,
    "duration_ms": 10000,
    "is_final": true
}
```

**交互时序：**

```
Client                          Server
  │── "START" ─────────────────→│
  │←─ {"event":"started"} ──────│
  │── "HOTWORDS:张三,北京" ────→│
  │←─ {"event":"hotwords_set"} ─│
  │── "LANGUAGE:中文" ─────────→│
  │←─ {"event":"language_set"} ─│
  │── [audio bytes] ───────────→│
  │←─ {sentences,partial} ──────│  (实时推送)
  │── [audio bytes] ───────────→│
  │←─ {sentences,partial} ──────│
  │── "STOP" ──────────────────→│
  │←─ {sentences,is_final:true} │  (最终结果)
  │←─ {"event":"stopped"} ──────│
```

### 热词文件

默认文件名 `热词列表`（`--hotword-file` 指定），一行一个词：

```
张三
李四
北京大学
```

也可通过 WebSocket 动态设置：`HOTWORDS:词1,词2,词3`

---

## 6. 性能对比

### Benchmark 结果（184 files, 11541s audio, 1947 VAD segments）

| 方法 | 耗时 | RTFx | CER |
|------|------|------|-----|
| PyTorch native | 589s | 19.6 | 8.94% |
| **Our vLLM (batch)** | **29.3s** | **393.9** | **8.91%** |
| yuekaizhang vLLM | 42.7s | 273.0 | 17.07% |

- **加速比**: 20.7x (vs PyTorch)
- **CER 一致性**: 8.91% vs 8.94%（差异 < 0.05%，完全对齐）
- **vs 第三方实现**: 比 yuekaizhang/Fun-ASR-vllm 快 44%，CER 优 8%

### 关键优化

| 优化项 | 效果 |
|--------|------|
| `use_low_frame_rate` token 截断 | CER 19.68% → 8.91% |
| Batch audio encode (groups of 8) | 音频编码加速 ~8x |
| vLLM batch generate (all prompts) | LLM 解码加速 ~20x |
| 去掉 `<think>` tokens | 减少无效生成步骤 |

### GLM-ASR-Nano

| 方法 | RTFx | CER | 加速 |
|------|------|-----|------|
| PyTorch native | 34.4 | 12.94% | 基准 |
| **vLLM (ours)** | **263.2** | **12.92%** | **7.6x** |

```python
from funasr.models.glm_asr.inference_vllm import GLMASRVLLMEngine

engine = GLMASRVLLMEngine.from_pretrained(
    model="zai-org/GLM-ASR-Nano-2512",
    hub="ms",
    gpu_memory_utilization=0.4,
    max_model_len=4096,
)
results = engine.generate(inputs=["audio.wav"])
print(results[0]["text"])
```

### WebSocket 实时服务

| 指标 | 数值 |
|------|------|
| RTF | < 0.08 |
| 首字延迟 | ~480ms |
| 30s 音频总耗时 | ~2.3s |
| 并发 | 多 WebSocket 连接 |

### 复现 Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/benchmark_vllm.py \
    --audio-dir /path/to/benchmark_audio \
    --label-json /path/to/benchmark_testset.json
```

---

## 7. API 参考

### AutoModelVLLM

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM
```

通用入口，自动检测模型类型并选择对应的 vLLM 实现。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | - | 模型名（hub）或本地路径 |
| `hub` | `"ms"` | `"ms"` / `"hf"` |
| `device` | `"cuda:0"` | 音频编码器设备 |
| `dtype` | `"bf16"` | 精度 |
| `tensor_parallel_size` | `1` | vLLM GPU 并行数 |
| `gpu_memory_utilization` | `0.8` | KV Cache 显存比例 |
| `max_model_len` | `4096` | 最大序列长度 |

### AutoModelVLLM.generate()

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `inputs` | - | 音频路径/列表/numpy/tensor |
| `language` | `None` | 语种提示 |
| `hotwords` | `None` | 热词列表 |
| `itn` | `True` | 逆文本正则化 |
| `max_new_tokens` | `512` | 最大生成 token |
| `temperature` | `0.0` | 采样温度 |
| `repetition_penalty` | `1.0` | 重复惩罚 |

**返回**: `[{"key": str, "text": str, "timestamps": [...]}]`

### FunASRNanoStreamingVLLM.from_pretrained()

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `"FunAudioLLM/Fun-ASR-Nano-2512"` | 模型名或路径 |
| `hub` | `"ms"` | 模型来源 |
| `device` | `"cuda:0"` | 设备 |
| `dtype` | `"bf16"` | 精度 |
| `tensor_parallel_size` | `1` | GPU 并行数 |
| `gpu_memory_utilization` | `0.8` | 显存比例 |
| `max_model_len` | `2048` | 序列长度 |
| `chunk_ms` | `720` | chunk 时长 |
| `rollback_chars` | `8` | 回退字符数 |

### streaming_generate()

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `audio_input` | - | 音频路径/数据 |
| `chunk_ms` | `720` | chunk 大小 |
| `rollback_chars` | `8` | 回退字符 |
| `hotwords` | `None` | 热词 |
| `language` | `None` | 语种 |
| `max_new_tokens` | `200` | 每 chunk 最大 token |
| `temperature` | `0.0` | 采样温度 |

**Yields**: `{"text", "fixed_text", "is_final", "chunk_idx", "audio_duration_ms"}`

> `repetition_penalty=1.3` 内部硬编码。

### serve_realtime_ws.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `10095` | WebSocket 端口 |
| `--model` | `FunAudioLLM/Fun-ASR-Nano-2512` | 模型 |
| `--hub` | `ms` | 来源 |
| `--device` | `cuda:0` | 设备 |
| `--decode-interval` | `0.48` | partial 解码间隔（秒） |
| `--hotword-file` | `热词列表` | 热词文件 |
| `--language` | `None` | 语种 |
| `--dtype` | `bf16` | 精度 |
| `--tensor-parallel-size` | `1` | GPU 并行 |
| `--gpu-memory-utilization` | `0.8` | 显存比例 |
| `--max-model-len` | `2048` | 序列长度 |

---

## 8. FAQ

### Q: 首次启动为什么慢？
vLLM 需要初始化 KV Cache 和 CUDA Graph warmup，约 60-90 秒。后续推理即时响应。

### Q: CUDA OOM 怎么办？
- 减小 `gpu_memory_utilization`（如 0.6）
- 增加 `tensor_parallel_size` 分摊到多卡
- 减小 `max_model_len`

### Q: Paraformer 能用 vLLM 吗？
不能。Paraformer 是非自回归模型（CIF predictor），所有 token 并行生成，不使用 KV-cache。vLLM 只加速自回归 LLM 解码。

### Q: WebSocket 服务和 streaming_generate 有什么区别？

| | WebSocket 服务 | streaming_generate |
|---|---|---|
| 分句 | VAD 自然端点 | 固定 720ms chunk |
| 推理 | 每个 VAD 段整体解码 | 累积重编码全部音频 |
| 准确率 | 更高 | 前 3s 较低 |
| 场景 | 生产部署 | SDK 集成 |

### Q: 流式推理前几秒输出为空？
正常。模型需要 ~3 秒累积音频才能产生有意义输出，这是模型特性而非 vLLM 限制。

### Q: 支持哪些音频格式？
wav、mp3、flac 等主流格式，采样率自动转为 16kHz。

### Q: 浏览器无法使用麦克风？
Chrome 要求 HTTPS 或 localhost。远程服务器用 SSH 端口转发：`ssh -L 10095:localhost:10095 <server>`

### Q: 多个并发连接会互相影响吗？
不会。每个 WebSocket 连接有独立的 session（VAD/ASR 状态隔离）。vLLM 内部会自动调度。

---

## 附录：DynamicStreamingVAD

`funasr.models.fsmn_vad_streaming.dynamic_vad.DynamicStreamingVAD` 是通用的动态阈值流式 VAD 封装，
在 fsmn-vad 基础上根据当前语音段的累积时长动态调整静音切分阈值。

### 设计动机

fsmn-vad 默认使用固定静音阈值（800ms）。实际场景中：
- 短句（如"好的"）需要等更长的静音才切，否则会把一句话切碎
- 长段（如会议发言 30s+）需要更快切分，否则 ASR 输入过长导致质量下降

### 用法

```python
from funasr import AutoModel
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

vad_model = AutoModel(model="fsmn-vad", device="cuda:0")

# 使用默认阈值配置
vad = DynamicStreamingVAD(vad_model)

# 或自定义阈值
vad = DynamicStreamingVAD(
    vad_model,
    silence_schedule=[
        (3000, 1500),       # 累积 <=3s: 等 1.5s 静音
        (10000, 800),       # 累积 3-10s: 等 0.8s
        (float('inf'), 300), # 累积 >10s: 等 0.3s
    ],
    speech_noise_thres=0.5,
)
```

#### 流式调用

```python
import torch

for audio_chunk in audio_stream:  # 实时音频流
    segments = vad.feed(torch.from_numpy(audio_chunk).float())
    for seg in segments:
        print(f"Speech: {seg[0]}-{seg[1]}ms")

# 结束时
final_segments = vad.finalize()
```

#### 非流式调用

```python
segments = vad.process(full_audio_tensor)
for seg in segments:
    print(f"Speech: {seg[0]}-{seg[1]}ms")
```

### 默认阈值配置

| 累积时长 | 静音阈值 | 说明 |
|---------|---------|------|
| ≤ 5s | 2.0s | 短句不切碎 |
| 5-10s | 1.5s | 正常分句 |
| 10-15s | 1.0s | 开始收紧 |
| 15-30s | 0.8s | 较快切分 |
| 30-45s | 0.4s | 防止过长 |
| > 45s | 0.1s | 强制切分 |

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vad_model` | - | FunASR AutoModel 加载的 fsmn-vad 实例 |
| `chunk_size_ms` | 60 | VAD 内部处理 chunk 大小 |
| `speech_noise_thres` | 0.5 | 语音/噪声判别阈值 |
| `speech_to_sil_thres_ms` | 150 | 语音转静音基础时间 |
| `silence_schedule` | 见上表 | 动态阈值配置 `[(上限ms, 静音ms), ...]` |
| `sample_rate` | 16000 | 采样率 |

### 属性

| 属性 | 说明 |
|------|------|
| `vad.is_speaking` | 当前是否在语音状态中 |
| `vad.current_duration_ms` | 当前段已累积时长 |
| `vad.current_threshold_ms` | 当前使用的静音阈值 |
