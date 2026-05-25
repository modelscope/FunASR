# FunASR vLLM 推理引擎指南

---

## 目录

1. [概述](#1-概述)
2. [安装与环境](#2-安装与环境)
3. [离线推理](#3-离线推理)
4. [实时流式服务](#4-实时流式服务)
5. [动态 VAD](#5-动态-vad)
6. [Benchmark](#6-benchmark)
7. [API 参考](#7-api-参考)
8. [FAQ](#8-faq)

---

## 1. 概述

FunASR 集成 vLLM 推理引擎，加速 LLM-based ASR 模型的自回归解码（PagedAttention + 连续批处理）。

### 支持模型

| 模型 | 说明 | vLLM 加速 |
|------|------|-----------|
| **Fun-ASR-Nano** | Qwen3-0.6B，31语言+方言 | RTFx 371，CER 8.88% |
| **GLM-ASR-Nano** | Llama 2B，17语言 | RTFx 265，CER 12.93% |
| LLMASR | Whisper + Qwen/Vicuna | ✓ |
| Paraformer / SenseVoice | 非自回归模型 | ✗ 不适用 |

### 架构

```
Audio ──→ Frontend ──→ Encoder ──→ Adaptor ──→ Audio Embeddings
                                                     │
Text Prompt ──→ Tokenize ──→ Embed ──────────→ [Concat] ──→ vLLM LLM ──→ Text
                                                                          │
                                              CTC Forced Alignment ──→ Timestamps
```

音频编码在 PyTorch 中完成，LLM 解码由 vLLM 加速。两者通过 `EmbedsPrompt` 连接。

---

## 2. 安装与环境

```bash
pip install funasr>=1.3.0
pip install vllm>=0.12.0
pip install safetensors tiktoken websockets regex fastapi uvicorn

cd /path/to/FunASR && pip install -e .
```

**硬件要求**：GPU ≥ 8GB VRAM，CUDA ≥ 11.8。推荐 16GB+。

---

## 3. 离线推理

适用于批量转写、文件处理。音频先经动态 VAD 切分为段，再 batch 送入 vLLM 推理。

### 使用方式

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
)

results = model.generate(
    ["audio1.wav", "audio2.wav"],
    language="中文",
    hotwords=["张三", "北京"],
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
    if "timestamps" in r:
        for ts in r["timestamps"]:
            print(f"  {ts['token']} [{ts['start_time']:.2f}-{ts['end_time']:.2f}s]")
```

**输出包含**：识别文本 + 字级别时间戳（CTC forced alignment，自动输出）。

### GLM-ASR

```python
from funasr.models.glm_asr.inference_vllm import GLMASRVLLMEngine

engine = GLMASRVLLMEngine.from_pretrained("zai-org/GLM-ASR-Nano-2512")
results = engine.generate(inputs=["audio.wav"])
```

### 离线动态 VAD 配置

离线场景下，动态 VAD 倾向保留较长段（≤60s），减少边界切割损失：

| 累积语音时长 | 静音阈值 | 说明 |
|------------|---------|------|
| ≤ 10s | 2000ms | 充分等待，短句不切碎 |
| 10-20s | 1000ms | 逐步收紧 |
| 20-30s | 800ms | |
| 30-40s | 600ms | |
| 40-50s | 400ms | |
| 50-60s | 200ms | 接近上限 |
| > 60s | 100ms | 强制切分 |

> GLM-ASR 不支持长音频段推理（>15s 质量下降），应配合传统 VAD（固定阈值 800ms）使用。

### 命令行

```bash
python benchmark_vllm.py \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --audio-dir /path/to/audio \
    --label-json /path/to/labels.json
```

---

## 4. 实时流式服务

两种服务模式：WebSocket 流式（逐帧推送）和 HTTP 非流式（文件上传）。

### 4.1 WebSocket 流式服务（serve_realtime_ws.py）

音频实时到达，VAD 检测端点后逐段解码，partial 预览持续更新。

```bash
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095 --language 中文
```

**流式动态 VAD 配置**（与离线不同，更积极切分以降低延迟）：

| 累积语音时长 | 静音阈值 | 说明 |
|------------|---------|------|
| ≤ 5s | 2000ms | 等用户说完一句 |
| 5-10s | 1500ms | |
| 10-15s | 1000ms | |
| 15-30s | 800ms | |
| 30-45s | 400ms | 快速切分 |
| > 45s | 100ms | 强制切分 |

**协议**：

```
Client → Server:  "START" | "HOTWORDS:词1,词2" | "LANGUAGE:中文" | [bytes] | "STOP"
Server → Client:  {"sentences":[...], "partial":"...", "is_final":false}
```

### 4.2 HTTP/OpenAI 统一服务（serve_vllm.py）

集成 VAD + vLLM + SPK + 时间戳，提供三种 API：

```bash
CUDA_VISIBLE_DEVICES=0 python serve_vllm.py --port 8899
```

| 接口 | 路径 | 功能 |
|------|------|------|
| HTTP REST | `POST /asr` | 文件上传 → text + timestamps + speaker |
| OpenAI API | `POST /v1/audio/transcriptions` | Whisper 兼容 |
| WebSocket | `ws://host:port/ws` | 流式音频 |

**示例**：

```bash
# 基本识别
curl -X POST http://localhost:8899/asr -F "file=@audio.wav" -F "language=中文"

# 开启说话人分离（默认关闭）
curl -X POST http://localhost:8899/asr -F "file=@audio.wav" -F "spk=true"

# OpenAI 兼容（含字级别时间戳）
curl -X POST http://localhost:8899/v1/audio/transcriptions \
    -F "file=@audio.wav" -F "model=fun-asr-nano" -F "response_format=verbose_json"
```

> SPK 默认关闭。开启后 RTFx 从 102 降至 46（聚类开销），CER 不受影响。

---

## 5. 动态 VAD

FunASR 的 fsmn-vad 默认启用动态静音阈值：根据当前语音段累积时长自动调整切分灵敏度。

### 离线 vs 流式的策略差异

| 场景 | 策略 | 目标 |
|------|------|------|
| **离线** | 宽松阈值（2000→100ms） | 保留长段，减少边界损失 |
| **流式** | 中等阈值（2000→100ms，但更快收紧） | 平衡延迟和完整性 |

两者都是动态的，区别在于收紧速度。离线场景音频已完整到达，容忍更长段；流式需要更快给出结果。

### 自定义配置

```python
# 通过 AutoModel 调用时传入
model.generate(input="audio.wav", silence_schedule=[
    (5000, 1500),
    (15000, 800),
    (30000, 400),
    (float('inf'), 200),
])
```

### DynamicStreamingVAD（流式 wrapper）

```python
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

vad = DynamicStreamingVAD(vad_model)
for chunk in audio_stream:
    segments = vad.feed(chunk)
    for seg in segments:
        print(f"Speech: {seg[0]}-{seg[1]}ms")
final = vad.finalize()
```

**属性**：`vad.is_speaking`、`vad.current_duration_ms`、`vad.current_threshold_ms`

---

## 6. Benchmark

**测试集**：184 文件，11541 秒音频，1947 VAD 段（固定阈值）/ 1120 段（动态阈值）。

| 模型 | 引擎 | VAD | RTFx | CER | 备注 |
|------|------|-----|------|-----|------|
| Fun-ASR-Nano | PyTorch | fixed | 17 | 8.90% | 基准 |
| Fun-ASR-Nano | PyTorch | dynamic | 21 | 8.06% | |
| Fun-ASR-Nano | **vLLM batch** | fixed | **371** | 8.88% | 21.7x |
| Fun-ASR-Nano | **vLLM batch** | dynamic | **340** | **8.20%** | 19.9x |
| Fun-ASR-Nano | **Service (no SPK)** | dynamic | **102** | 8.14% | |
| Fun-ASR-Nano | **Service (+SPK)** | dynamic | **46** | 8.19% | SPK 默认关闭 |
| Fun-ASR-Nano | yuekaizhang vLLM | fixed | 273 | 17.07% | 第三方 |
| GLM-ASR-Nano | **vLLM batch** | fixed | **265** | 12.93% | 7.6x |
| GLM-ASR-Nano | vLLM batch | dynamic | — | — | 不支持长音频推理 |

**复现**：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_vllm.py \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --audio-dir /path/to/benchmark_audio \
    --label-json /path/to/benchmark_testset.json
```

---

## 7. API 参考

### AutoModelVLLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | — | 模型名或路径 |
| `hub` | `"ms"` | `"ms"` / `"hf"` |
| `tensor_parallel_size` | `1` | GPU 并行数 |
| `gpu_memory_utilization` | `0.8` | KV Cache 显存比例 |
| `max_model_len` | `4096` | 最大序列长度 |

### generate()

| 参数 | 说明 |
|------|------|
| `inputs` | 音频路径列表 / numpy / tensor |
| `language` | 语种（"中文"/"English"/...） |
| `hotwords` | 热词列表 |
| `max_new_tokens` | 最大生成 token（默认 500） |

**返回**：`[{"key": str, "text": str, "timestamps": [...]}]`

### serve_vllm.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 8000 | 服务端口 |
| `--model` | Fun-ASR-Nano-2512 | 模型 |
| `--gpu-memory-utilization` | 0.5 | 显存比例 |
| `--max-model-len` | 4096 | 序列长度 |

### serve_realtime_ws.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 10095 | WebSocket 端口 |
| `--decode-interval` | 0.48 | partial 解码间隔 |
| `--hotword-file` | 热词列表 | 一行一词 |
| `--language` | None | 语种 |

---

## 8. FAQ

**Q: vLLM 首次启动慢？**
KV Cache 初始化 + CUDA Graph warmup 约 60-90 秒，之后即时推理。

**Q: CUDA OOM？**
减小 `gpu_memory_utilization`，增加 `tensor_parallel_size`，或减小 `max_model_len`。

**Q: GLM-ASR 能用动态 VAD 吗？**
不建议。GLM-ASR 不支持长音频段推理（>15s 质量严重下降），应使用固定阈值 VAD。

**Q: SPK 对性能的影响？**
RTFx 从 102 降至 46（说话人聚类开销），CER 不受影响。默认关闭。

**Q: 流式和离线 VAD 为什么配置不同？**
离线音频已完整，可保留更长段减少边界损失；流式需要快速响应，阈值收紧更快。

**Q: 浏览器麦克风不可用？**
Chrome 要求 HTTPS 或 localhost。远程用 `ssh -L 10095:localhost:10095 <server>`。
