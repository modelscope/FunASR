# FunASR Migration Benchmark Example

Use this example when you are comparing FunASR with Whisper, OpenAI audio APIs, or a cloud ASR provider. It runs FunASR over your representative audio set and writes machine-readable results plus a Markdown summary.

The script does not claim accuracy by itself. Run your baseline on the same files, then compare transcripts with human review or your normal WER/CER workflow.

## Quick start

```bash
python examples/migration/benchmark_funasr.py \
  --input /path/to/audio_samples \
  --recursive \
  --model iic/SenseVoiceSmall \
  --device cuda \
  --spk-model cam++ \
  --output-dir outputs/funasr_migration_eval \
  --metadata baseline=whisper-large-v3
```

Outputs:

- `results.jsonl`: one JSON object per audio file with text, elapsed seconds, audio duration, realtime factor, model, device, and errors.
- `summary.md`: run configuration, aggregate speed, per-file previews, and next comparison steps.

## CPU smoke test

For a portable first check, use CPU and a small audio folder:

```bash
python examples/migration/benchmark_funasr.py \
  --input ./samples \
  --model iic/SenseVoiceSmall \
  --device cpu \
  --output-dir outputs/funasr_cpu_smoke
```


## 中文快速说明

这个示例用于从 Whisper、OpenAI 音频 API 或云端 ASR 迁移前的本地评测。请用同一批代表性音频分别跑旧方案和 FunASR，再用人工审阅或 WER/CER 流程比较质量。

```bash
python examples/migration/benchmark_funasr.py \
  --input /path/to/audio_samples \
  --recursive \
  --model iic/SenseVoiceSmall \
  --device cuda \
  --spk-model cam++ \
  --output-dir outputs/funasr_migration_eval \
  --metadata baseline=whisper-large-v3
```

输出文件：

- `results.jsonl`：每条音频的文本、耗时、音频时长、实时倍速和错误信息。
- `summary.md`：运行配置、总体速度、逐文件预览和下一步对比建议。

如果结果可以公开，欢迎提交 [Migration Benchmark Report](https://github.com/modelscope/FunASR/issues/new?template=migration_benchmark.md)，帮助其他用户参考你的硬件、音频领域和质量记录。

## What to compare

Track the same fields for your old ASR stack and FunASR:

| Field | Why it matters |
|---|---|
| Audio duration, language, domain, sample rate, speaker count | Keeps the comparison representative. |
| Model name, version, device, CUDA/PyTorch versions | Makes results reproducible. |
| Model load time vs inference time | Separates cold start from steady-state throughput. |
| WER/CER or human review notes | Captures quality beyond speed. |
| Failed-file rate and error messages | Shows operational risk before rollout. |

See the [migration guide](../../docs/migration_from_whisper.md) for the full evaluation and rollout checklist. If you can share results publicly, open a [Migration Benchmark Report](https://github.com/modelscope/FunASR/issues/new?template=migration_benchmark.md) so others can learn from your hardware, audio domain, and quality notes.
