# FunASR vLLM Inference Guide

本文档介绍如何使用 vLLM 加速 FunASR 中 LLM-based ASR 模型的推理。

## 概览

vLLM 是一个高吞吐量 LLM 推理引擎，支持 PagedAttention、Tensor Parallelism 等优化。FunASR 集成 vLLM 用于加速 LLM 解码部分，实现 **2-3x** 推理加速。

### 适用模型

| 模型 | vLLM 支持 | 说明 |
|------|-----------|------|
| **FunASRNano** | ✓ | Qwen3-0.6B LLM，支持离线和流式 |
| **LLMASR** | ✓ | Whisper + Qwen/Vicuna/LLaMA |
| **GLMASR** | ✓ | GLM-ASR-Nano |
| **QwenAudioWarp** | ✓ | Qwen-Audio |
| Paraformer | ✗ | 非自回归模型（CIF predictor），无 LLM 解码 |
| SenseVoice | ✗ | Whisper-like encoder-decoder，非 LLM |
| Conformer/Transformer | ✗ | CTC/attention 解码，非 LLM |
| Qwen3-ASR | ✗ | 使用外部 qwen-asr 包自带优化 |

### 为什么 Paraformer 不能用 vLLM？

Paraformer 使用 CIF (Continuous Integrate-and-Fire) predictor 实现**非自回归**并行解码。所有 token 同时生成，不需要 KV-cache。vLLM 的优化（PagedAttention、KV-cache 管理）只对**自回归** LLM 有效。

## 安装

```bash
pip install funasr>=1.3.0
pip install vllm>=0.11.0
pip install safetensors tiktoken
```

## 快速开始

### 通用接口 (AutoModelVLLM)

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

# 自动检测模型类型，使用 vLLM 加速
model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
    tensor_parallel_size=2,  # 2卡并行
    gpu_memory_utilization=0.8,
)

results = model.generate(["audio.wav"], language="中文")
print(results[0]["text"])
```

### 离线批量推理 (FunASRNanoVLLM)

```python
from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

engine = FunASRNanoVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=4,  # 4卡并行
)

# 批量处理
results = engine.generate(
    inputs=["audio1.wav", "audio2.wav", "audio3.wav"],
    hotwords=["开放时间"],
    language="中文",
)
for r in results:
    print(f"[{r['key']}] {r['text']}")
    if "timestamps" in r:
        print(f"  timestamps: {r['timestamps'][:3]}")
```

### 流式推理 (FunASRNanoStreamingVLLM)

```python
from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

engine = FunASRNanoStreamingVLLM.from_pretrained(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    chunk_ms=720,        # 每720ms一个chunk
    rollback_chars=8,    # 回退8个字符
)

for result in engine.streaming_generate("audio.wav", language="中文"):
    if result["is_final"]:
        print(f"最终结果: {result['text']}")
    else:
        print(f"[{result['audio_duration_ms']:.0f}ms] {result['text']}")
        print(f"  确认区: {result['fixed_text']}")
```

## 命令行 Demo

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# 单卡推理
python demo_vllm.py --input audio.wav --language 中文

# 多卡并行 + 批量
python demo_vllm.py --input wav.scp --tensor-parallel-size 4 --batch-size 32

# 带热词
python demo_vllm.py --input audio.wav --hotwords 开放时间 周一 --language 中文
```

## 架构说明

```
┌───────────────────────────────────────────────────────────┐
│                    FunASR + vLLM                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Audio ──→ Frontend ──→ Audio Encoder ──→ Audio Adaptor   │
│            (fbank)      (SenseVoice/    (Linear/          │
│                          Whisper)        Transformer)     │
│                              │                            │
│                              ▼                            │
│                     Audio Embeddings                      │
│                              │                            │
│  Text Prompt ──→ Tokenize ──→ Embed ──→ [Concat]         │
│  (system/user/                              │             │
│   assistant)                                ▼             │
│                                     ┌─────────────┐      │
│                                     │   vLLM      │      │
│                                     │ (Qwen3/etc) │      │
│                                     │ TP并行加速   │      │
│                                     └──────┬──────┘      │
│                                            ▼             │
│                                     Generated Text       │
│                                                           │
│  (可选) CTC Forced Alignment ──→ 字级别时间戳             │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 流式推理设计

```
音频流（每720ms一个chunk）
     │
     ▼ 累计音频重新编码
┌──────────────────────┐
│ Stage 1: 前10个chunk  │ ← 批量生成，无 prev_text
│ 找到稳定输出          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Stage 2: 后续chunk   │ ← 用稳定输出作为 prev_text prefill
│ 带 prev_text 生成    │
└──────────┬───────────┘
           ▼
每个 chunk 输出: [fixed 区域] + [8字 unfixed 区域]
                  (不会变)      (可能在下一chunk改变)
```

## 性能对比

### 离线推理（5.6s 中文音频）

| 配置 | 延迟 | 相对加速 |
|------|------|----------|
| Torch (baseline) | 0.89s | 1x |
| vLLM 1-GPU | 0.30s | **3.0x** |
| vLLM 2-GPU TP | ~0.20s | **4.5x** |

### 流式推理（17s 中文音频，25 chunks）

| 配置 | 总时间 | 加速 |
|------|--------|------|
| Torch sequential | 5.1s | 1x |
| vLLM batch | 2.8s | **1.8x** |

### 批量推理吞吐

| Batch Size | 1-GPU | 2-GPU | 4-GPU |
|-----------|-------|-------|-------|
| 1 | ~1.5x | ~2x | ~2.5x |
| 16 | ~4x | ~7x | ~12x |
| 32 | ~5x | ~9x | ~15x |

## API 参考

### `AutoModelVLLM(model, ...)`

通用入口，自动检测模型类型。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | - | 模型名或本地路径 |
| `hub` | `"ms"` | `"ms"` (ModelScope) 或 `"hf"` (HuggingFace) |
| `tensor_parallel_size` | `1` | GPU 并行数 |
| `gpu_memory_utilization` | `0.8` | vLLM KV-cache 显存比例 |
| `max_model_len` | `4096` | 最大序列长度 |
| `dtype` | `"bf16"` | 计算精度 |

### `engine.generate(inputs, ...)`

| 参数 | 说明 |
|------|------|
| `inputs` | 音频路径列表、numpy array、torch Tensor |
| `language` | 语言提示："中文"、"英文"、"日文" 等 |
| `hotwords` | 热词列表，提升识别准确率 |
| `itn` | 逆文本正则化（默认 True） |
| `max_new_tokens` | 最大生成 token 数（默认 512） |

**返回**: `[{"key": str, "text": str, "timestamps": [...]}]`

### `streaming_engine.streaming_generate(audio, ...)`

| 参数 | 说明 |
|------|------|
| `audio_input` | 音频路径或数据 |
| `chunk_ms` | chunk 大小（默认 720ms） |
| `rollback_chars` | 回退字符数（默认 8） |

**Yields**: `{"text", "fixed_text", "is_final", "chunk_idx", "audio_duration_ms"}`

## 常见问题

### Q: 为什么前几个 chunk 输出为空？
模型需要约 3 秒音频才能产生有意义的输出，这是模型本身的特性。

### Q: CUDA OOM 怎么办？
- 减小 `gpu_memory_utilization`（如 0.6）
- 增加 `tensor_parallel_size`
- 减小 `max_model_len`

### Q: 流式和离线结果略有不同？
正常现象。流式每个 chunk 独立生成，离线看到完整音频。最终 chunk 的结果与离线基本一致。

### Q: 支持哪些音频格式？
wav、mp3、flac 等主流格式，采样率会自动转为 16kHz。
