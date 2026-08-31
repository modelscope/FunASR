# Qwen3-ASR 离线长音频 vLLM 示例

`transcribe_vllm_offline.py` 使用 Qwen3-ASR 自带的 `Qwen3ASRModel.LLM` 后端，不经过
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
