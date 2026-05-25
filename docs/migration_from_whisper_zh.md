# 从 Whisper 或云端 ASR 迁移到 FunASR

当你已经有 Whisper、OpenAI/云端 ASR 或自研语音流水线，并想判断是否值得切到 FunASR 时，可以按这个指南评估。目标不是用一个样例音频证明结论，而是在真实业务音频上比较质量、速度、成本和部署可行性。

## 什么时候值得评估 FunASR

如果你有以下需求，FunASR 通常值得优先测试：

- 音频需要留在私有环境内，不能上传到外部云服务。
- 会议、归档、媒体、客服录音等长音频需要高吞吐转写。
- 希望一条流水线内完成 VAD、标点、时间戳和说话人分离。
- 需要 OpenAI 兼容音频接口，接入 Agent、Dify、LangChain、AutoGen 或内部应用。
- 需要 WebSocket/runtime 服务支撑流式识别或实时字幕。
- 希望先用 CPU 做可复现 smoke test，再迁移到 GPU 服务。

如果你更需要完全托管服务、厂商 SLA，或你的自有评测显示目标语言/领域还不够好，可以暂时保留现有方案。

## 快速评估计划

1. 选择 20-50 条有代表性的音频，覆盖短音频、长录音、噪声、多说话人、目标语言和方言。
2. 按生产方式运行当前 Whisper 或云端 ASR，保存转写结果、延迟、成本和失败样例。
3. 用 README 快速开始跑 FunASR，也可以用 [迁移评测示例](../examples/migration/) 测量一组代表性音频。然后根据 [部署选型表](./deployment_matrix_zh.md) 选择服务路径。
4. 用人工审阅或现有 WER/CER 流程比较结果，不要只看一个干净 demo 音频。
5. 如果应用已经使用 OpenAI 风格客户端，运行 OpenAI 兼容 API smoke test。
6. 分开记录 warmup、模型下载、设备、GPU/CPU 型号、batch size、音频时长和稳定吞吐。

## 功能映射

| 现有流程 | FunASR 路径 | 需要验证什么 |
|---|---|---|
| Whisper 文件转写 | 使用 SenseVoice、Paraformer 或 Fun-ASR-Nano 的 [README 快速开始](../README_zh.md#快速开始) | 转写质量、时间戳、速度、模型下载、CPU/GPU 行为。 |
| Whisper + pyannote | VAD、标点和 `spk_model="cam++"` | 说话人标签、换人位置、重叠说话、长静音。 |
| OpenAI 音频 API 或云端批量 ASR | [OpenAI 兼容 API 示例](../examples/openai_api/) | `/v1/audio/transcriptions`、响应格式、客户端兼容性、上传限制。 |
| Dify/LangChain/AutoGen Agent 音频 | [客户端配方](../examples/openai_api/CLIENTS.md) 或 [MCP 服务](../examples/mcp_server/) | 工具延迟、文件处理、鉴权边界、错误返回。 |
| 实时字幕或客服流式识别 | [Runtime 服务文档](../runtime/readme_cn.md) | 分块、断句、重连、背压、中间/最终结果行为。 |
| 字幕生成 | [字幕示例](../examples/subtitle/) | 分段可读性、行长、说话人标签、SRT/VTT 兼容性。 |
| 离线归档处理 | [批处理示例](../examples/batch_asr_improved.py) | manifest、重试、进度日志、吞吐、失败文件恢复。 |

## 最小本地对比

安装 FunASR，并用你基线评测里的同一条音频运行：

```bash
pip install funasr
```

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",  # 便携 smoke test 可改成 "cpu"
)
result = model.generate(input="sample.wav")
print(result)
```

如果要按 API 服务方式对比：

```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

如果要对自己的音频目录做可复现评测，可以运行 [`examples/migration/benchmark_funasr.py`](../examples/migration/benchmark_funasr.py) 生成 `results.jsonl` 和 `summary.md`。如果需要容器 smoke test，可以从 `examples/openai_api/docker-compose.yml` 开始，并用 `BASE_URL=http://localhost:8000 bash examples/openai_api/smoke_test.sh` 验证。

## 质量与速度检查清单

对旧流水线和 FunASR 都记录这些字段：

- 音频时长、语言、领域、采样率、声道数和说话人数。
- 模型名、模型版本、FunASR 版本、Python/PyTorch/CUDA 版本，以及 Docker 镜像 tag。
- 硬件、设备模式、batch size、流式 chunk size，以及是否排除 warmup/模型下载时间。
- WER/CER 或人工审阅记录：姓名、数字、标点、说话人分离、时间戳、领域词。
- 延迟、吞吐、GPU/CPU 内存、每小时音频成本、失败文件比例。
- 运维要求：鉴权、上传限制、TLS、日志、监控、重试和数据留存规则。

## 上线检查清单

- 在代表性评测通过前，保留旧流水线作为回退。
- 先做内部 endpoint 或离线批处理，再对外暴露 API。
- 为每个请求记录 request id、音频时长、模型、设备、延迟和错误类型。
- 在 runbook 中固定模型别名和部署命令。
- 测试噪声、静音、多人重叠、长文件、非 UTF-8 文件名和网络中断。
- 遇到阻塞时，通过 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md) 提交命令、日志、模型、设备和样本特征。

## 分享迁移结果

如果 FunASR 替代或补充了你的现有 ASR 栈，欢迎开一个 [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md)。包含硬件、速度、质量记录和部署细节的迁移报告，能帮助新用户更快选型。
