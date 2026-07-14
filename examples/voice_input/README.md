# FunASR Voice Input — 语音输入法

桌面语音输入工具。按快捷键录音，自动识别并粘贴到当前光标位置。

## 安装

```bash
pip install funasr sounddevice numpy pyperclip openai pynput
```

## 使用

**第一步：启动 FunASR 服务**
```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --device cuda  # 或 --device cpu
```

**第二步：启动语音输入法**
```bash
cd examples/voice_input
python funasr_input.py
```

**第三步：使用**
- 按 `Ctrl+Shift+Space` 开始录音
- 再按一次停止，自动识别并粘贴文字

## 工作流程

```
按快捷键 → 录音 → 再按快捷键 → 发送到 funasr-server → 识别 → 自动粘贴到光标位置
```

## 配置选项

```bash
python funasr_input.py --server http://localhost:8000/v1  # 服务器地址
python funasr_input.py --model paraformer                 # 模型选择
python funasr_input.py --hotkey "cmd+shift+space"         # macOS 快捷键
python funasr_input.py --lang zh                          # 语言
```

## 支持平台

| 平台 | 录音 | 自动粘贴 |
|------|------|---------|
| macOS | ✅ | ✅ (AppleScript) |
| Linux | ✅ | ✅ (xdotool) |
| Windows | ✅ | 手动 Ctrl+V |

## 支持模型

| 模型 | 速度 | 语言 |
|------|------|------|
| sensevoice (默认) | 170x GPU / 17x CPU | 中/英/日/韩/粤 |
| paraformer | 120x GPU / 15x CPU | 中/英 |
| fun-asr-nano | 17x GPU | 中/英/日 + 中文方言/口音 |

## 架构

```
[麦克风] → [本程序] → [funasr-server :8000] → [识别结果] → [粘贴到光标]
                         ↑
                    本地运行，音频不出机器
```

## 常见问题

**Q: 需要网络吗？**
不需要。funasr-server 和语音输入法都在本地运行。

**Q: 支持哪些音频格式？**
内部使用 WAV 16kHz，自动处理。

**Q: 延迟多少？**
GPU: 说完 1-2 秒出结果。CPU: 3-5 秒。
