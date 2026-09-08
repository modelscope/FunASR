# Qwen3-ASR 离线长音频 vLLM 示例

`transcribe_vllm_offline.py` 默认使用 Qwen3-ASR 自带的 `Qwen3ASRModel.LLM` 后端，不经过
`AutoModelVLLM`。后者服务于 FunASR 自有模型，当前不支持 Qwen3-ASR。

这个示例先用 ffmpeg 将输入统一为 16 kHz 单声道，再复用 qwen-asr 自带的静音边界切块，
逐块调用原生 vLLM 推理，并将结果还原到原始时间轴。默认块上限 180 秒，与 qwen-asr 的
长音频处理边界一致。该路径不需要额外 VAD 模型。

## 安装

建议使用独立环境，因为 `qwen-asr[vllm]==0.0.6` 固定依赖 `vllm==0.14.0`：

```bash
python -m venv .venv-qwen3-vllm
source .venv-qwen3-vllm/bin/activate
pip install -U "qwen-asr[vllm]==0.0.6" "transformers==4.57.6"
```

系统还需要可执行的 `ffmpeg`。

## 运行

```bash
python examples/industrial_data_pretraining/qwen3_asr/transcribe_vllm_offline.py \
  recording.mp3 \
  --model Qwen/Qwen3-ASR-1.7B \
  --language Chinese \
  --max-inference-batch-size 4
```

默认输出 `recording.qwen3-vllm.json`，包含全文以及每个块的 `start_ms`、`end_ms`、
`text` 和模型返回的语言。省略 `--language` 时启用自动语言识别。

可根据显存调整 `--max-inference-batch-size` 和 `--gpu-memory-utilization`。较短的
`--chunk-seconds` 能减小单次请求和重复退化的影响，但更多边界也可能增加漏字或断词；
修改前请在代表性音频上同时验证 CER、漏字和重复。

该示例受到 [qwen3-asr-service](https://github.com/LanceLRQ/qwen3-asr-service) 的离线
vLLM 实践以及 [FunASR #3419](https://github.com/modelscope/FunASR/issues/3419) 用户复测
启发。不同测试集、参考稿对齐方式和模型尺寸的 CER 不能直接横向等同。

## MOSS 离线转写与匿名说话人

需要录音内说话人归属时，显式选择 `--engine moss`。这是第三方
MOSS-Transcribe-Diarize 的独立 HTTP 路径，不是给 Qwen3-ASR 的文本追加标签。
客户端复用 FunASR `AutoModel` 的 MOSS 适配器，整条录音只上传一次，不经过上述
180 秒切块、外部 VAD 或第二个 speaker 模型，也不在客户端下载 MOSS 权重。

先在独立 GPU 环境按 [MOSS vLLM 部署指南](../../../docs/moss_transcribe_diarize.md#vllm)
启动服务，并确认其返回 `diarized_json`。该指南固定的服务端 vLLM 为 **0.27.1**；
不要把它安装进上面的 Qwen3 `vllm==0.14.0` 环境。本机只作为 HTTP 客户端时不需要 GPU。

在包含本示例更新的 FunASR 仓库根目录，为客户端建立另一环境。以下以 Linux/Python 3.12
CPU 客户端为例；FunASR 的安装依赖不自动包含 Torch，因此需要显式安装：

```bash
python3.12 -m venv .venv-moss-client
source .venv-moss-client/bin/activate
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

python examples/industrial_data_pretraining/qwen3_asr/transcribe_vllm_offline.py \
  recording.mp3 \
  --engine moss \
  --vllm-base-url http://127.0.0.1:8898/v1 \
  --served-model moss-transcribe-diarize \
  --request-timeout 600 \
  --max-completion-tokens 8192
```

将服务地址替换为你实际可访问的地址，`--served-model` 必须匹配服务端注册名。
服务启用认证时，事先通过密钥管理方式设置环境变量 `MOSS_VLLM_API_KEY`，不要把真实密钥
写进脚本或命令行。此模式不要传 Qwen3 专用的 `--model`、`--language`、切块或 GPU 参数。
示例来自当前源码，旧版 PyPI 安装中的脚本未必包含此模式。

默认输出 `recording.moss-vllm.json`，也可通过 `--output` 指定其他结果文件：

- `text`：MOSS 服务返回的全文，不与 Qwen3 结果混拼。
- `sentence_info`：每段的 `start`、`end`、`text`、`spk`；起止时间为毫秒。
- `timestamp`：对应的 `[start_ms, end_ms]` 列表。
- `raw_text`：在这个结构化模式下等于服务返回的清理后文本，并非原始带控制标签的生成串。

终端也会打印各段时间和匿名标签，例如 `10..90 ms [S02] ...`。HTTP 失败或畸形段数据会
报错，不生成新的成功结果，也不覆盖已有结果文件。`S01/S02` 仅在单条录音中表示匿名
说话人，不是已知人物身份、声纹验证或跨文件身份；这不是实时 WebSocket 路径。

`8192` 只是可调输出上限，不保证 480 秒或更长录音完整转写。需同时核对服务端上下文/
输出限制、最后一段时间、末尾语音和实际说话人归属。若自行切片，标签不能直接跨片合并；
本样例不会伪造身份连续性。本机 HTTP 契约测试不代表实际模型的 CER、说话人准确率或容量验证。
