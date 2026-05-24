# Fun-ASR-Nano 流式语音识别服务（Streaming ASR + vLLM）

基于 WebSocket 的实时流式语音识别服务，使用 **vLLM 推理引擎**加速 LLM 解码。支持 VAD 分句、说话人分离（Beta）、热词定制化、语种指定。

## 功能特性

| 功能 | 说明 |
|------|------|
| 流式 ASR (vLLM) | 基于 Fun-ASR-Nano + **vLLM 推理引擎**（PagedAttention, 连续批处理），RTF < 0.1 |
| 流式 VAD | fsmn-vad 增量检测语音端点，动态调整静音阈值 |
| 说话人分离 (Beta) | eres2netv2 + ClusterBackend，流式分配+最终重聚类 |
| 热词定制化 | 加载人名、地名等实体词列表，提升专有名词识别准确率 |
| 幻觉检测 | 自动检测重复模式并截断，防止模型生成循环 |
| 多语言 | 31种语言、7大方言、26种地方口音 |

## 架构

```
客户端（浏览器/Python）               服务端（GPU）
┌─────────────────┐   WebSocket    ┌──────────────────────────┐
│                 │   PCM16 16kHz  │  serve_realtime_ws.py    │
│ client_mic.html │ ────────────→  │                          │
│ client_python.py│                │  ┌──────────┐            │
│                 │ ←────────────  │  │ VAD      │ 流式分句    │
│  实时显示结果    │   JSON         │  └────┬─────┘            │
└─────────────────┘                │       ↓                  │
                                   │  ┌──────────┐            │
                                   │  │ ASR      │ 逐段解码    │
                                   │  └────┬─────┘            │
                                   │       ↓                  │
                                   │  ┌──────────┐            │
                                   │  │ SPK      │ 说话人分离  │
                                   │  └──────────┘            │
                                   └──────────────────────────┘
```

## 推理逻辑

### 核心设计

服务**不是**将全部音频累积后一次性解码，而是基于 VAD 端点检测**逐段解码**：

```
音频流 ──→ StreamingVAD (60ms chunk) ──→ 检测到端点 ──→ _decode_segment (vLLM)
                                     │                        │
                                     │                        ▼
                                     │               locked_sentences (文字锁定)
                                     │                        │
                                     │                        ▼
                                     │               SPK assign (说话人分配)
                                     │
                                     └──→ 未结束 ──→ decode partial (vLLM, 每0.48s)
                                                           │
                                                           ▼
                                                    partial text (实时预览，会被覆盖)
```

### 两条推理路径

| 路径 | 触发条件 | 推理范围 | 输出 |
|------|---------|---------|------|
| **确认段解码** | VAD 检测到静音端点 | 该段完整音频 | 锁定到 sentences，永不改变 |
| **Partial 预览** | 每 0.48s + 新音频 ≥ 960ms | 当前说话中的全部音频 | 临时文字，随时覆盖 |

### 动态 VAD 阈值

为避免切碎短句或堆积过长段，静音阈值根据当前段已累积时长动态调整：

| 累积时长 | 需要静音 | 效果 |
|---------|---------|------|
| ≤ 5s | 2.0s | 短句不切碎 |
| 5-10s | 1.5s | 正常分句 |
| 10-15s | 1.0s | 开始收紧 |
| 15-30s | 0.8s | 较快切分 |
| 30-45s | 0.4s | 防止过长 |
| > 45s | 0.1s | 强制切分 |

### STOP 时的最终处理

1. 将剩余音频喂给 VAD（`is_final=True`），收集最后的确认段
2. 如果仍在说话中，强制结束并解码当前段
3. 对所有 `sentences` 运行 SPK `finalize()`（全局重聚类，修正说话人 ID）
4. 返回 `is_final: true` 的完整结果

### 与 FunASRNanoStreamingVLLM 的区别

| | WebSocket 服务 (`serve_realtime_ws.py`) | SDK 流式 (`FunASRNanoStreamingVLLM`) |
|---|---|---|
| 分句方式 | VAD 自然端点 | 固定 720ms chunk |
| 推理粒度 | 每个 VAD 段整体解码 | 累积重编码全部音频 |
| 准确率 | 更高（完整语义段） | 较低（前 3s 噪声多） |
| 适用场景 | 生产环境实时服务 | SDK 级流式展示 |


## 快速部署

### 环境要求

- Python >= 3.8
- CUDA >= 11.8
- GPU 显存 >= 8GB（vLLM 引擎需要分配 KV Cache）
- vLLM >= 0.6.0

### 安装依赖

```bash
cd examples/industrial_data_pretraining/fun_asr_nano
pip install -r requirements.txt

# 安装 vLLM（如未安装）
pip install vllm

# 安装 FunASR（开发模式，需要 auto_model_vllm）
cd /path/to/FunASR && pip install -e .
```

### 启动服务

```bash
# 基本启动
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095

# 完整参数
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py \
    --port 10095 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --hub ms \
    --device cuda:0 \
    --decode-interval 0.48 \
    --hotword-file 热词列表 \
    --language 中文

# 多卡 tensor parallel（大模型场景）
CUDA_VISIBLE_DEVICES=0,1 python serve_realtime_ws.py \
    --port 10095 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8
```

首次启动会自动下载模型（约 30 秒），之后即可接受连接。

### 服务端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 10095 | WebSocket 监听端口 |
| `--model` | `FunAudioLLM/Fun-ASR-Nano-2512` | ASR 模型名称 |
| `--hub` | `ms` | 模型来源（ms=ModelScope, hf=HuggingFace） |
| `--device` | `cuda:0` | GPU 设备 |
| `--decode-interval` | 0.48 | 实时解码间隔（秒） |
| `--hotword-file` | `热词列表` | 热词文件路径（一行一个词） |
| `--language` | `None` (auto) | 指定语种（如 中文、English、日本語） |
| `--dtype` | `bf16` | 推理精度（bf16/fp16/fp32） |
| `--tensor-parallel-size` | 1 | vLLM tensor parallel GPU 数量 |
| `--gpu-memory-utilization` | 0.8 | vLLM KV Cache GPU 显存占比 |
| `--max-model-len` | 2048 | vLLM 最大序列长度 |
| `--use-context` | True | 使用上下文辅助解码 |
| `--no-context` | - | 禁用上下文 |

## 客户端使用

### 1. 浏览器客户端（client_mic.html）

直接在浏览器中打开 `client_mic.html`：

```bash
# 本机访问
open client_mic.html  # macOS
xdg-open client_mic.html  # Linux

# 远程服务器（需 SSH 端口转发）
ssh -L 10095:localhost:10095 <server>
# 然后在本地浏览器打开 client_mic.html
```

功能：
- 麦克风实时录音
- 音频文件上传识别
- 热词列表文件加载（.txt，一行一个词）
- 说话人分离显示（可开关）

### 2. Python 客户端（client_python.py）

```bash
# 麦克风模式
python client_python.py --server ws://localhost:10095 --mic

# 文件模式
python client_python.py --server ws://localhost:10095 --file audio.wav

# 带热词
python client_python.py --server ws://localhost:10095 --file audio.wav \
    --hotwords "张三,李四,北京"

# 不显示说话人
python client_python.py --server ws://localhost:10095 --mic --no-spk
```

### 3. 测试脚本（client_test.py）

```bash
# 运行完整测试
python client_test.py --server ws://localhost:10095 --file test_audio.wav

# 带热词测试
python client_test.py --server ws://localhost:10095 --file test_audio.wav \
    --hotwords "人名,地名"
```

测试内容：
- 基本流式 ASR 流程验证
- 返回格式校验（sentences/partial/is_final）
- 空音频处理
- 多轮 session 复用
- 性能统计（RTF）

## API 协议文档

### 连接

```
WebSocket URL: ws://<host>:<port>
音频格式: PCM16, 16kHz, 单声道 (Little-Endian)
```

### 客户端 → 服务端

| 消息类型 | 格式 | 说明 |
|---------|------|------|
| 开始会话 | `"START"` (text) | 初始化/重置 session |
| 设置热词 | `"HOTWORDS:词1,词2,词3"` (text) | 可选，在 START 后发送 |
| 设置语种 | `"LANGUAGE:中文"` (text) | 可选，指定识别语种 |
| 音频数据 | `bytes` (binary) | PCM16 音频帧，任意长度 |
| 结束会话 | `"STOP"` (text) | 触发最终解码，返回完整结果 |

### 服务端 → 客户端

#### 1. 事件消息

```json
{"event": "started"}
{"event": "hotwords_set", "hotwords": ["词1", "词2"]}
{"event": "language_set", "language": "中文"}
{"event": "stopped"}
```

#### 2. 实时结果（流式，在 START 和 STOP 之间持续推送）

```json
{
    "sentences": [
        {"text": "已确认句子", "start": 1700, "end": 5500, "spk": 0}
    ],
    "partial": "正在说的文字...",
    "partial_start_ms": 5800,
    "duration_ms": 7200,
    "is_final": false
}
```

#### 3. 最终结果（STOP 后返回）

```json
{
    "sentences": [
        {"text": "第一句话。", "start": 1700, "end": 5500, "spk": 0},
        {"text": "第二句话。", "start": 6000, "end": 9200, "spk": 1}
    ],
    "partial": "",
    "partial_start_ms": 0,
    "duration_ms": 10000,
    "is_final": true
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `sentences` | array | 已确认的句子列表 |
| `sentences[].text` | string | 识别文本 |
| `sentences[].start` | int | 起始时间（毫秒） |
| `sentences[].end` | int | 结束时间（毫秒） |
| `sentences[].spk` | int | 说话人 ID（0-based），仅最终结果保证准确 |
| `partial` | string | 当前正在说的文本（未确认，会被更新） |
| `partial_start_ms` | int | partial 的起始时间（毫秒） |
| `duration_ms` | int | 已接收音频总时长（毫秒） |
| `is_final` | bool | 是否为最终结果 |

### 典型交互流程

```
Client                          Server
  │                               │
  │──── "START" ─────────────────→│
  │←──── {"event":"started"} ─────│
  │                               │
  │──── "HOTWORDS:张三,北京" ────→│  (可选)
  │←── {"event":"hotwords_set"} ──│
  │                               │
  │──── "LANGUAGE:中文" ─────────→│  (可选)
  │←── {"event":"language_set"} ──│
  │                               │
  │──── [audio bytes] ───────────→│
  │──── [audio bytes] ───────────→│
  │←── {sentences,partial} ───────│  (实时推送)
  │──── [audio bytes] ───────────→│
  │←── {sentences,partial} ───────│
  │        ...                    │
  │──── "STOP" ──────────────────→│
  │←── {sentences,is_final:true} ─│  (最终结果)
  │←── {"event":"stopped"} ───────│
  │                               │
```

## 热词文件格式

文件名默认为 `热词列表`（可通过 `--hotword-file` 指定），格式为纯文本，一行一个词：

```
张三
李四
北京大学
Fun-ASR-Nano
```

也可在运行时通过 WebSocket 动态设置：`HOTWORDS:词1,词2,词3`

## 注意事项

1. **vLLM 引擎**：首次启动时 vLLM 需要初始化 KV Cache，约需 60-90 秒，之后即可服务
2. **HTTPS 限制**：浏览器麦克风需要 HTTPS 或 localhost（Chrome 安全策略）
2. **远程访问**：使用 SSH 端口转发 `ssh -L 10095:localhost:10095 <server>`
3. **并发支持**：单个服务实例支持多个并发 WebSocket 连接（每个连接独立 session）
4. **显存占用**：vLLM 引擎 + VAD + SPK 共约 8-10GB 显存（gpu_memory_utilization=0.8 时）
5. **说话人分离**：Beta 功能，短音频或单人场景可能不准确
6. **模型来源**：默认从 ModelScope 下载，可用 `--hub hf` 切换 HuggingFace

## Related

- **[vLLM Offline Inference](vllm_inference.md)** — Batch inference API (FunASRNanoVLLM)
- **[vLLM Streaming Inference](vllm_streaming.md)** — Chunk-by-chunk streaming SDK (FunASRNanoStreamingVLLM)
