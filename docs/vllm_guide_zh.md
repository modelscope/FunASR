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

> 表中 Fun-ASR-Nano 的 batch 吞吐量比值为 `340 / 21 = 16.2`，前提是计时范围相同。`RTFx 340` 表示实时倍率，不是相对 PyTorch 加速 340 倍。CER 从 `8.06%` 变为 `8.20%`，相差 0.14 个百分点，并非完全相同。以上历史测量不保证其他硬件、话务和配置的性能或精度。

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

本指南的 SDK、离线与 WebSocket 服务使用 **FunASR 拆分引擎**；不要把它的环境与下文原生 `vllm serve` 的验证环境混用。先选定 vLLM 版本和对应的 GPU wheel，再在独立虚拟环境中安装。`nvidia-smi` 的 CUDA 数字是驱动支持的上限，不是已安装的 CUDA runtime，也不能仅凭“12.x / 13.x”判断 wheel 是否兼容。

下面以拆分引擎使用过的 `vllm==0.19.1` 和固定 FunASR 源码为起点。它固定了两个项目的版本，但**不是完整依赖锁文件，也不是所有 GPU 的干净安装验收**。GPU 构建与驱动要求请核对该版本的 [vLLM 安装文档](https://github.com/vllm-project/vllm/blob/v0.19.1/docs/getting_started/installation/gpu.md)。

```bash
python3.12 -m venv .venv-funasr-vllm
source .venv-funasr-vllm/bin/activate
python -m pip install "vllm==0.19.1"

# 服务脚本来自源码；在新的目录中安装，不覆盖已有工作区。
git clone https://github.com/modelscope/FunASR.git FunASR-vllm
cd FunASR-vllm
git checkout --detach e42443f55971d0c804dcf2973fdd2e6e09bd5611
python -m pip install -e .
python -m pip install safetensors tiktoken websockets regex fastapi uvicorn python-multipart
python -m pip check
python -m pip freeze > environment.txt
```

保存 GPU/驱动、Python、源码提交、模型 revision 和 `environment.txt`，再执行单请求、真实 WebSocket 及目标并发验收。`pip check` 只能检查声明的依赖关系，不证明 CUDA、音频算子或端到端服务可用。单纯安装 PyPI 包不会提供本指南引用的仓库服务脚本。

### 开始前先选定模型路径

目前有两条不同的 Fun-ASR-Nano vLLM 集成路径，checkpoint 与调用接口不能混用：

#### A. FunASR 拆分引擎路径（本指南）

请从 [ModelScope](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)
或 [Hugging Face](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
下载官方 `FunAudioLLM/Fun-ASR-Nano-2512`。当前 Hugging Face 的 `model.pt`
CTC 权重不完整，因此仍可用于转写，但 FunASR 会禁用受影响的 CTC 路径，避免
返回不可靠的时间戳或说话人分离结果。部署需要时间戳或说话人分离时，请使用
ModelScope checkpoint；权重发布修复进度见 [#3496](https://github.com/modelscope/FunASR/issues/3496)。
`Qwen3-0.6B/` 子目录按设计只保存 LLM 配置与 tokenizer，不是可以单独加载的权重目录。

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

# 二选一；默认使用 ModelScope。
model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
)
# model = AutoModelVLLM(
#     model="FunAudioLLM/Fun-ASR-Nano-2512",
#     hub="hf",
# )
```

首次加载时，FunASR 会自动调用 `prepare_vllm_model_dir()`：从
`Qwen3-0.6B/` 复制配置与 tokenizer，从根目录 `model.pt` 提取 `llm.*`
权重，并生成 `Qwen3-0.6B-vllm/model.safetensors`。不要把 `model` 指向
`Qwen3-0.6B/`，也不要直接让 vLLM 加载这个只有配置的子目录。

#### B. vLLM 原生转写路径

原生 `FunASRForConditionalGeneration` 使用完整 native checkpoint，不是路径 A 的 `model.pt` 拆分布局。官方维护的 native checkpoint 是
[`FunAudioLLM/Fun-ASR-Nano-2512-vllm`](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-vllm)。
2026-09-07 审计时，v0.28.0 的模型注册表仍将
[`allendou/Fun-ASR-Nano-2512-vllm`](https://huggingface.co/allendou/Fun-ASR-Nano-2512-vllm)
列为原生 `FunASRForConditionalGeneration` 架构示例；后者是托管在官方
FunAudioLLM 组织之外的社区转换完整 checkpoint。main 与发布版不一定使用同一引用，
下方官方验证记录区分了当时核查的版本，不能把不同版本的模型列表混为一谈。
只有明确选择 vLLM 原生转写
接口时，才应使用这两种 native checkpoint；不要用它们替换下文 FunASR
`AutoModelVLLM` 示例中的官方 checkpoint，也不要传给
[serve_realtime_ws.py](../examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py)；这些服务预期
`model.pt`、`config.yaml` 与 `Qwen3-0.6B/`，不能用 native layout 替代。

两种 native 路径都通过 `vllm serve` 提供非实时的请求/响应式转写
`/v1/audio/transcriptions`，不会注册 `/v1/realtime`。vLLM 只为声明
realtime 任务的模型注册该 WebSocket 端点，`FunASRForConditionalGeneration`
当前不在它的 [Realtime Transcription 表](https://docs.vllm.ai/en/latest/models/supported_models/#realtime-transcription)
中。需要实时流式识别时，请使用 FunASR 流式 SDK 推理或流式 ASR 服务；路径 A 的
`AutoModelVLLM` 示例同样属于离线推理。

原生 HTTP API 不会自动获得 FunASR WebSocket 服务的 VAD、partial 预览、会话状态或 SPK 处理。WebSocket 握手的拒绝状态不是稳定的 API 契约，也不是模型支持实时转写的探测方法。

官方权重的固定 revision、启动参数和实际输出见 [原生转写验证记录](./vllm_official_native_validation_zh.md)。该记录仅覆盖既有 H100 环境中的 vLLM 0.27.1 文件转写，不是干净安装、精度、容量、长音频、实时流式或说话人分离验收；不要把它的环境替换到上面的 0.19.1 拆分引擎命令中。

**硬件**：以所选 vLLM GPU 构建的要求为准。显存需求还取决于模型、精度、KV cache、batch 和会话数；本指南不承诺统一的最低显存或并发容量。

不要在已验证环境中单独升级 `torch` 或 `torchaudio`。依赖约束随 vLLM 版本变化，不能概括成“始终自动安装相同版本的三件套”：例如 [0.19.1 的发布元数据](https://pypi.org/pypi/vllm/0.19.1/json)声明 `torch==2.10.0`、`torchaudio==2.10.0`、`torchvision==0.25.0`；[0.27.1 的发布元数据](https://pypi.org/pypi/vllm/0.27.1/json)声明 `torch==2.13.0`、`torchaudio==2.11.0`、`torchvision==0.28.0`。这些只是声明约束，不是安装成功或 ABI 兼容的证明。升级时新建环境并重新验收；遇到驱动过旧错误，应检查实际 wheel 的 CUDA 构建与驱动要求，而不是盲目安装最新版。

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
| 已报告的 batch 吞吐量 | RTFx 21 | RTFx 340；范围见 Benchmark |

### 支持模型

| 模型 | LLM 部分 | audio encoder | 集成路径 |
|------|---------|---------------|-----------|
| **Fun-ASR-Nano** | Qwen3-0.6B | SenseVoice | 专用 split engine |
| **GLM-ASR-Nano** | Llama-2B | Whisper-like | 专用 split engine |
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
| 已报告的 batch 结果 | RTFx 21 | RTFx 340；需匹配硬件和计时范围 |

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
    inputs=["audio1.wav", "audio2.wav"],  # 已存在的音频文件，不是清单文件
    hotwords=["开放时间"],
    language="中文",
    max_new_tokens=512,
)
```

直接引擎接收音频路径、音频路径列表或 16 kHz 波形数组/tensor，不展开 SCP/JSONL 清单。清单使用下方 [demo_vllm.py](../examples/industrial_data_pretraining/fun_asr_nano/demo_vllm.py)：SCP 每行为路径或 `key path`，JSONL 每行对象的 `source` 字段为音频路径。相对路径按进程工作目录解析。

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

以下数值用于说明结构，不是实测结果。HTTP 时间戳和时长单位为秒；可选 `words`、`speaker` 字段取决于相应处理路径，见[服务与序列化实现](../examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py)。

```json
{
    "text": "你好",
    "segments": [
        {
            "text": "你好",
            "start": 0.3,
            "end": 1.2,
            "speaker": "SPK0",
            "words": [
                {"word": "你", "start": 0.3, "end": 0.6},
                {"word": "好", "start": 0.6, "end": 1.2}
            ]
        }
    ],
    "duration": 2.0,
    "processing_time": 0.1,
    "rtf": 0.05
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

以下为结构完整的示例响应；时间戳单位为秒，不是识别质量保证。

```json
{
    "task": "transcribe",
    "language": "zh",
    "duration": 2.0,
    "text": "你好",
    "segments": [
        {
            "id": 0, "start": 0.3, "end": 1.2,
            "text": "你好",
            "words": [
                {"word": "你", "start": 0.3, "end": 0.6},
                {"word": "好", "start": 0.6, "end": 1.2}
            ]
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

下面每行是独立的 JSON WebSocket 消息，不是一个 JSON 文档。数值为示例，WebSocket 偏移单位为毫秒。

```text
{"event": "started"}
{"event": "language_set", "language": "中文"}
{"sentences": [{"text": "你好", "start": 300, "end": 1200}], "is_final": true, "duration_ms": 2000}
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
- partial 解码间隔是调度参数，不是首字延迟保证；需按实际话务测量端到端延迟

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

服务端默认每 20 秒发送一次 WebSocket ping，但不因 ping 超时主动断开连接。
在长音频高并发下，模型推理和 VAD 处理会延迟控制帧，即使连接仍然健康；
固定 timeout 因此可能在计算或排队反压期间误杀正常会话。

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
    --port 10095 --language 中文
```

只有在按生产流量测得最坏推理和排队延迟后，才设置正数
`--ws-ping-timeout`；该值应高于实测延迟，并与网关 idle timeout 策略配合。
`websockets` 库的接收高水位 `max_queue` 在队列满时可能暂停 socket 读取；
即使推理已经移出事件循环，后续 Ping 帧也会因此延迟处理。
增大高水位只是推后过载触发点，不是吞吐修复。只有外部网关
已经统一负责 keepalive / reconnect 策略时，才设置 `--ws-ping-interval 0`
关闭服务端 ping。

源码服务将消息接收与会话推理解耦，应用层 FIFO 默认限制为
`--ws-receive-max-messages 128` 和 `--ws-receive-max-bytes 16777216`，
两项都必须为正数。限制分别统计排队消息数和载荷字节数（文本命令按 UTF-8
字节计），并非进程总内存上限；协议缓冲、正在处理的消息与会话音频还会占用内存。
超限时连接以 **1013** 关闭，不会作为成功的 final 返回。没有应用层恢复策略时，
不要自动重放可能已经部分处理的会话。

音频与命令保持原顺序。当队列中的下一条消息仍是音频时，会推迟一次本已到期的
临时预览解码，先消费积压音频；不会丢弃或合并音频帧。VAD 完整段解码和
`COMMIT`/`STOP` 最终解码仍处理完整输入。负载下的预览频率、上下文与回退观测可能
变化，因此不保证转写文本逐字一致或硬件吞吐提升。使用 `--log-decode-profile`
可同时记录 `peak_messages`、`peak_bytes`、`skipped_partials` 与原有 engine profile。

以上源码改动尚未进入已发布的 `funasr==1.4.15`。合成解码器的传输测试不能代替
L20 或生产流量验收，详见[压测契约](benchmark/realtime_ws_benchmark.md)。

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

示例消息序列，每个 JSON 对象对应一条 WebSocket 消息。偏移单位为毫秒，不是延迟实测记录。

```text
{"event": "started"}
{"sentences": [{"text": "你好", "start": 300, "end": 1200}], "partial": "世界", "is_final": false}
{"sentences": [{"text": "你好", "start": 300, "end": 1200}, {"text": "世界", "start": 1500, "end": 2200}], "is_final": true}
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
- 只有**超长、不停顿的连续语音**（如长篇朗读）会让单句不断变长、partial 预览逐渐变慢。`serve_realtime_ws.py` 默认用 `--partial-window-sec 8` 限制临时预览窗口；多客户端连续独白负载应先从该值开始，并只在实测有余量时再提高窗口。它只影响临时 `partial`，VAD 锁定句和 STOP 最终结果仍走完整音频。可先参考 §6.7 的 L20 实测起点。

### 6.6 说话人分离（SPK）的代价与开关

`serve_realtime_ws.py` 默认**不加载** SPK 模型。只有启动时显式加 `--enable-spk`，才会加载 `--spk-model`（默认 `iic/speech_eres2netv2_sv_zh-cn_16k-common`）并在流式中对每个 VAD 完成句调用一次说话人分配。需要注意：

- **Fun-ASR-Nano 上 SPK 效果有限**（见 #2944），多数实时 ASR 场景并不需要说话人分离。
- **流式 SPK 代价高且随会话变长**：每句会在该 session 的 worker 中对**全部历史 embedding** 做一次全量重聚类（**O(N²)**，会话越长每句越贵）；而会话结束时还会**全量重聚一遍**，流式期间每句的聚类结果会被最终结果覆盖——对最终输出而言属于重复计算。长会话 + 高并发下尤其明显。
- **建议**：多客户端实时转写优先保持默认关闭；确需 diarization 时再加 `--enable-spk`，并以 STOP 后的最终 `spk` 标签为准。
- **长会话诊断**：如果 session 仍然逐渐变慢或断开，请用 `--log-session-stats-interval 30` 复测，并观察 `audio_buffer_samples`、`locked_sentences`、`speaker_history_chunks`、`speaker_history_embeddings` 和 `speaker_centers` 是否保持有界。如果这些计数都接近上限但 RTF 仍持续升高，剩余瓶颈更可能在模型推理、返回 payload 大小或环境调度，而不是 session 状态继续泄漏。

### 6.7 生产并发与多进程部署

`serve_realtime_ws.py` 把网络 I/O 保留在一个 asyncio 事件循环中，但每个连接的阻塞 session 工作会在线程中执行。同时到达且解码参数兼容的 ASR 请求，会在 `--decode-batch-wait-ms` 窗口内汇聚（默认 10 ms），合并为一次 `AutoModelVLLM.generate()` 调用，并由 `--decode-max-batch-size` 限制单批音频段数（默认 16）。共享 vLLM 引擎仍只有一个受控调用入口，但音频接收和 session 状态不再被全进程锁串行化。

- **流式 VAD 默认使用 CPU。** 每个连接拥有独立 FSMN-VAD 实例，默认参数为 `--vad-device cpu --vad-ncpu 1`，从而隔离可变 cache，并避免每帧触发 CUDA allocator 同步。只有在完整 WebSocket 压测证明有收益后才建议改到 GPU；同时也要避免“每 session 线程数 × 活跃 session 数”造成 CPU 过量并行。
- **按真实话务调整批处理。** 默认 10 ms 相对 480 ms 的 partial 解码周期很小，又能让同时说话的连接共享 encoder 与 vLLM batch。追求最低单路排队延迟时可降到 `0`；吞吐优先时可谨慎调高。扩大最大 batch 前，应先测显存与尾延迟。
- **先把单进程作为第一扩展单元。** 先用内置批处理压测单进程；只有当单进程达到实测 GPU、CPU 或尾延迟上限后，再增加进程或按 GPU 横向扩容。每个额外进程都会复制模型显存，也可能减少单进程内形成 batch 的机会。
- **vLLM 收益取决于请求是否同时到达。** 真实轮流对话可能连接很多，但同时解码很少；把同一段连续独白同步回放给所有客户端，则会刻意形成大批次。发布结果时必须同时报告话务形态和批处理参数。
- **可持续并发没有通用的“支持 N 路”数字。** 上限主要取决于同时说话数、静音比例、句长、partial 刷新间隔、说话人分离、batch 等待时间以及 GPU/CPU 能力。长时间不停顿的语音仍会重复编码临时窗口（见 §6.5），成本更高。请按自己的真实话务压测，不要把其他部署的连接数直接当成规格。
- **L20 历史实测起点，不是容量承诺。** [#3528](https://github.com/modelscope/FunASR/issues/3528) 的一组历史测量在单张 L20 上同步回放 47 秒连续语音、16 路客户端，并关闭 SPK 与客户端 ping；`--partial-window-sec 8 --decode-interval 2.0` 是该组测量中表现最好的配置：408 次解码请求、累计编码 3,072.1 秒音频、完成 p50 51.18 秒、输出滞后 4.5 秒、聚合实时率 14.2x、首词 1.31 秒。该组测量当时采用的默认 15 秒窗口在 16 路负载上未完成；这不是当前源码的默认值（当前为 8 秒），也不能代替后续版本和不同 keepalive 配置的结果。请只把 `8 / 2.0` 当作该话务的调优起点，记录版本、窗口、keepalive、首词、输出滞后、最终完成时间、请求数和累计编码量。Git 安装成功不等于并发问题解决；该问题仍需独立验收。

```bash
CUDA_VISIBLE_DEVICES=0 python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py \
  --partial-window-sec 8 --decode-interval 2.0 --log-decode-profile
```

---

## 7. 动态 VAD

动态静音属于 **VAD 阶段**，不是 ASR 解码参数。[FSMN-VAD](../funasr/models/fsmn_vad_streaming/model.py) 读取 `dynamic_silence` 和 `silence_schedule`；显式设置 `max_end_silence_time` 会关闭默认动态策略，除非另行覆盖。`AutoModelVLLM.generate(inputs, **kwargs)` 只转交给 ASR 引擎，不运行 VAD。

下表在边界时长采样 SDK 的 `DEFAULT_SILENCE_SCHEDULE` 与 `STREAMING_SILENCE_SCHEDULE` 常量。每个 schedule 选择首个大于或等于累积语音时长的上限对应项；这些是静音阈值，不是语音段长上限。

| 累积语音时长采样点 | DEFAULT_SILENCE_SCHEDULE | STREAMING_SILENCE_SCHEDULE |
| --- | --- | --- |
| 5000 ms | 2000 ms | 2000 ms |
| 10000 ms | 2000 ms | 1500 ms |
| 15000 ms | 1000 ms | 1000 ms |
| 20000 ms | 1000 ms | 800 ms |
| 30000 ms | 800 ms | 800 ms |
| 40000 ms | 600 ms | 400 ms |
| 45000 ms | 400 ms | 400 ms |
| 50000 ms | 400 ms | 100 ms |
| 60000 ms | 200 ms | 100 ms |
| 60001 ms | 100 ms | 100 ms |

分块输入不会自动选择名称带 streaming 的常量。服务包装层可能使用独立策略，例如 [DynamicStreamingVAD](../funasr/models/fsmn_vad_streaming/dynamic_vad.py) 自己维护 schedule，并以 `dynamic_silence=False` 调用底层 VAD。不要把此 SDK 表格当作所有服务的配置。

### 自定义

```python
from funasr import AutoModel

vad = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
segments = vad.generate(
    input="audio.wav",
    cache={},
    dynamic_silence=True,
    silence_schedule=[(5000, 1500), (20000, 800), (float("inf"), 300)],
)
print(segments[0]["value"])
```

将 `audio.wav` 替换为实际录音。此独立 VAD 调用在 `value` 中返回毫秒单位的 `[start_ms, end_ms]` 区间，不返回转写文本。接入 ASR 时，按引擎要求加载并重采样音频，依据毫秒区间切片，再将波形列表传给 `model.generate(inputs=audio_segments)`；合并结果时保留原始偏移。本示例只演示分段，不会自动把 VAD 连接到 vLLM。

> GLM-ASR 应先分段，并验证所选 checkpoint 的段长限制。需要固定静音阈值时，将 `dynamic_silence=False` 传给 **VAD** 调用，而不是 ASR 引擎；固定静音本身不限制最大段长。

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
长录音应先分段，并针对所选 GLM checkpoint 验证段长。`dynamic_silence=False` 配置独立的 FSMN-VAD 阶段，不是 `AutoModelVLLM`；关闭动态静音本身不保证段长符合 ASR 限制。

**Q: SPK 性能影响？**
表中离线服务关闭/开启 SPK 的 RTFx 分别为 102/46，CER 分别为 8.14%/8.19%。SPK 默认关闭；这组测量不代表其他部署的开销或精度。

**Q: 二次开发入口？**
离线：`serve_vllm.process_audio()` / `FunASRNanoVLLM.generate()`
流式：`serve_realtime_ws.RealtimeASRSession`

**Q: 首次慢？**
模型加载、KV cache 分配及可选 CUDA Graph 预热都会影响首次启动。请分别测量冷启动与预热后推理，二者均没有固定耗时或即时返回保证。

**Q: Fun-ASR-Nano vLLM 使用 `dtype="fp16"` 时实际会怎样？**
音频 frontend 与 adaptor 仍使用 FP16，但 FunASR 会让 Qwen3 decoder 使用 BF16，
因为 vLLM 的 FP16 decoder 可能产生退化的重复输出。该行为自动生效，decoder 权重
仍是两字节精度；不支持 BF16 的硬件请使用 `dtype="fp32"`。vLLM 路径不宣称支持
端到端 FP16 decoder。

**Q: vLLM 输出连续标点（例如 `!!!!!!!!`），但 PyTorch/HF generate 正常，应该先查什么？**
这通常说明音频 frontend 和 checkpoint 本身能工作，但 vLLM prompt-embedding
路径或解码参数和 upstream runner 不一致。改模型前先检查这些项：

- 传给 vLLM 的 prompt embeddings 要显式转成 float32：
  `EmbedsPrompt` 的 `prompt_embeds` 参数应接收 `input_embeds.float()`。
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
