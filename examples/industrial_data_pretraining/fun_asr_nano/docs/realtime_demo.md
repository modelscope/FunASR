# Fun-ASR-Nano 实时语音识别 Demo

基于 WebSocket 的实时 ASR 演示，支持麦克风录音、流式 VAD 分句、逐句显示。

## 架构

```
浏览器（麦克风）                         服务器（GPU）
┌──────────────┐    WebSocket PCM16    ┌────────────────────────┐
│ client_mic   │ ──────────────────→   │  serve_realtime_ws.py  │
│   .html      │                       │                        │
│              │ ←────────────────── │  ┌─────────────────┐  │
│ 逐句显示结果  │    JSON sentences     │  │ Streaming VAD   │  │
└──────────────┘                       │  │ (fsmn-vad)      │  │
                                       │  └────────┬────────┘  │
                                       │           ↓           │
                                       │  ┌─────────────────┐  │
                                       │  │ ASR per-segment  │  │
                                       │  │ (Fun-ASR-Nano)   │  │
                                       │  └─────────────────┘  │
                                       └────────────────────────┘
```

## 核心设计

1. **流式 VAD 分句**：fsmn-vad 增量检测语音端点
   - `[start, -1]` = 检测到语音开始
   - `[-1, end]` = 检测到语音结束 → 确认分句
2. **逐段解码**：只对当前 VAD 段做 ASR（不累积全部音频）
   - 已确认段锁定文字，永不重新解码
   - 支持无限时长录音，不会因音频过长而崩溃
3. **prev_text 连续性**：当前段内使用 prev_text 保证文字稳定

## 使用方法

### 1. 启动服务

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# 单卡启动
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095

# 指定参数
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py \
    --port 10095 \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --device cuda:0 \
    --decode-interval 0.7
```

### 2. 打开客户端

直接用浏览器打开 `client_mic.html`，修改 Server 地址为：
- 本机：`ws://localhost:10095`
- 远程（需 SSH 端口转发）：先 `ssh -L 10095:localhost:10095 <server>`

### 3. 体验

1. 点击 **Start Recording** → 允许麦克风
2. 对着麦克风说话
3. 说话过程中，VAD 检测到停顿自动分句显示
4. 点击 **Stop** 结束

## 显示效果

```
0.3s - 2.1s   今天天气不错。
2.5s - 5.0s   我想去公园散步。
5.3s - ...    现在正在说的话...（斜体，未确认）
```

- 白色 = 已确认句子（VAD 检测到结束，文字锁定）
- 灰色斜体 = 正在说的句子（实时更新）
- 绿色时间戳 = VAD 检测的起止时间

## 服务端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 10095 | WebSocket 端口 |
| `--model` | `FunAudioLLM/Fun-ASR-Nano-2512` | ASR 模型 |
| `--device` | `cuda:0` | GPU 设备 |
| `--decode-interval` | 0.7 | 解码间隔（秒），越小延迟越低但 GPU 占用越高 |

## 协议

```
Client → Server:
    "START"         开始录音
    "STOP"          结束录音
    bytes           PCM16 音频数据（16kHz 单声道）

Server → Client:
    {"event": "started"}
    {"event": "stopped"}
    {
        "sentences": [
            {"text": "已确认的句子", "start_ms": 300, "end_ms": 2100},
            ...
        ],
        "partial": "正在说的...",
        "partial_start_ms": 5300,
        "duration_ms": 7200,
        "is_final": false
    }
```

## 注意事项

- 浏览器需要 HTTPS 或 localhost 才能使用麦克风（Chrome 安全限制）
- 首次使用需允许麦克风权限
- 模型加载约需 30 秒，之后即可接受连接
- 单个服务实例支持多个并发连接（每个连接独立 session）
