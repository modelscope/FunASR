# Fun-ASR-Nano 实时 WebSocket 服务 — 快速上手

> 完整文档请参见：[FunASR vLLM 推理引擎指南](../../../../docs/vllm_guide.md)

## 30 秒启动

```bash
cd examples/industrial_data_pretraining/fun_asr_nano

# 安装依赖
pip install -r requirements.txt
# 按主文档选择并安装与当前 NVIDIA 驱动、CUDA runtime 和 PyTorch wheel 匹配的 vLLM 版本。

# 启动服务
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095 --language 中文
```

## 客户端

```bash
# 浏览器
open client_mic.html

# Python 麦克风
python client_python.py --server ws://localhost:10095 --mic

# Python 文件
python client_python.py --server ws://localhost:10095 --file audio.wav

# 自动化测试
python client_test.py --server ws://localhost:10095 --file audio.wav
```

## 远程访问

```bash
ssh -L 10095:localhost:10095 <server>
# 然后本地打开 client_mic.html
```

## 功能

- **vLLM 推理引擎**：RTF < 0.08，支持 tensor parallel 多卡加速
- **流式 VAD 分句**：动态静音阈值，自然断句
- **说话人分离 (Beta)**：流式 ID 分配 + 最终重聚类
- **热词定制化**：加载人名、地名等实体词文件
- **语种指定**：31 种语言 + 中文方言
- **幻觉检测**：自动检测重复模式并截断

## 文件列表

| 文件 | 说明 |
|------|------|
| `serve_realtime_ws.py` | WebSocket 服务端 |
| `client_mic.html` | 浏览器客户端 |
| `client_python.py` | Python CLI 客户端 |
| `client_test.py` | 自动化测试脚本 |
| `热词列表` | 热词文件示例 |
| `demo_vllm.py` | 离线 vLLM 推理 demo |

## 详细文档

- [FunASR vLLM 推理引擎指南](../../../../docs/vllm_guide.md) — 完整文档（离线/流式/WebSocket/API/FAQ）
